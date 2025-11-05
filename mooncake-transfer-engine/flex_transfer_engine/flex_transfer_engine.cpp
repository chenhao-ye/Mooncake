#include "flex_transfer_engine.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "transfer_engine_c.h"
#include "util.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

FlexTransferEngine::FlexTransferEngine(const std::string &metadata_conn_string,
                                       const std::string &local_server_name,
                                       bool enable_copy,
                                       const std::string &ctrl_block_location)
    : local_server_name_(local_server_name),
      enable_copy_(enable_copy),
      ctrl_block_location_(ctrl_block_location),
      copy_server_(*this) {
    if (ctrl_block_location.find("cuda:") == 0) {
        throw std::invalid_argument(
            "ctrl_block_location must not be on CUDA: " + ctrl_block_location);
    }

    // Create the underlying TransferEngine
    engine_ = ::createTransferEngine(metadata_conn_string.c_str(),
                                     local_server_name.c_str(),
                                     /*unused*/ local_server_name.c_str(),
                                     /*unused*/ 12345, /*auto_discover*/ true);
    if (!engine_) throw std::runtime_error("Failed to create TransferEngine");

    // Start TCP listener only if enable_copy_ is true
    if (enable_copy_) copy_server_.startListener();
}

FlexTransferEngine::~FlexTransferEngine() {
    if (enable_copy_) copy_server_.cleanup();

    for (const auto &[_, conn] : copy_engine_connections_) close(conn.fd);

    {
        std::lock_guard<std::mutex> segment_lock(segment_cache_mutex_);
        for (auto &[segment_name, segment_id] : segment_cache_)
            ::closeSegment(engine_, segment_id);
        // segment_cache_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(ctrl_block_mutex_);
        for (CopyCtrlBlock *ctrl_block : copy_ctrl_block_cache_) {
            ::unregisterLocalMemory(engine_, ctrl_block);
            delete ctrl_block;
        }
        // copy_ctrl_block_cache_.clear();
    }

    ::destroyTransferEngine(engine_);
}

int FlexTransferEngine::registerLocalMemory(void *addr, size_t length,
                                            const std::string &location,
                                            int remote_accessible,
                                            bool force_direct) {
    // do actual RDMA registration
    if (force_direct || !enable_copy_) {
        return ::registerLocalMemory(engine_, addr, length, location.c_str(),
                                     remote_accessible);
    } else {  // register for copy-based transfer
        return copy_server_.registerLocalMemory(addr, length, location);
    }
}

int FlexTransferEngine::unregisterLocalMemory(void *addr, bool force_direct) {
    if (force_direct || !enable_copy_) {
        return ::unregisterLocalMemory(engine_, addr);
    } else {
        return copy_server_.unregisterLocalMemory(addr);
    }
}

int FlexTransferEngine::registerLocalMemoryBatch(
    std::vector<buffer_entry_t> &buffer_list, const std::string &location,
    bool force_direct) {
    if (force_direct || !enable_copy_) {
        return ::registerLocalMemoryBatch(engine_, buffer_list.data(),
                                          buffer_list.size(), location.c_str());
    } else {
        return copy_server_.registerLocalMemoryBatch(buffer_list, location);
    }
}

int FlexTransferEngine::unregisterLocalMemoryBatch(
    std::vector<void *> &addr_list, bool force_direct) {
    if (force_direct || !enable_copy_) {
        return ::unregisterLocalMemoryBatch(engine_, addr_list.data(),
                                            addr_list.size());
    } else {
        return copy_server_.unregisterLocalMemoryBatch(addr_list);
    }
}

segment_id_t FlexTransferEngine::getSegmentId(const std::string &segment_name) {
    std::lock_guard<std::mutex> lock(segment_cache_mutex_);
    auto it = segment_cache_.find(segment_name);
    if (it != segment_cache_.end()) return it->second;

    segment_id_t segment_id = ::openSegment(engine_, segment_name.c_str());
    if (segment_id < 0) return segment_id;  // error
    segment_cache_[segment_name] = segment_id;
    return segment_id;
}

CopyCtrlBlock *FlexTransferEngine::acquireCopyCtrlBlock() {
    std::lock_guard<std::mutex> lock(ctrl_block_mutex_);

    // Try to get from cache first
    if (!copy_ctrl_block_cache_.empty()) {
        CopyCtrlBlock *ctrl_block = copy_ctrl_block_cache_.back();
        copy_ctrl_block_cache_.pop_back();
        ctrl_block->progress_counter.store(0, std::memory_order_release);
        return ctrl_block;
    }

    // Cache is empty, allocate a new one
    CopyCtrlBlock *ctrl_block = new CopyCtrlBlock();

    // Register it with RDMA using the specified location
    int rc = ::registerLocalMemory(engine_, ctrl_block, sizeof(CopyCtrlBlock),
                                   ctrl_block_location_.c_str(),
                                   /*remote_accessible*/ true);
    if (rc) {
        delete ctrl_block;
        return nullptr;
    }
    return ctrl_block;
}

void FlexTransferEngine::releaseCopyCtrlBlock(CopyCtrlBlock *ctrl_block) {
    if (!ctrl_block) return;
    std::lock_guard<std::mutex> lock(ctrl_block_mutex_);
    copy_ctrl_block_cache_.emplace_back(ctrl_block);
}

int FlexTransferEngine::submitTransferToCopyEngine(
    std::vector<transfer_request_t> &entries, const std::string &server_url,
    CopyCtrlBlock *ctrl_block, ClientConnection **out_conn) {
    if (entries.empty()) return 0;
    assert(ctrl_block);

    // Connect to the remote FlexTransferEngine (or reuse existing connection)
    ClientConnection *conn = nullptr;
    int fd = -1;
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = copy_engine_connections_.find(server_url);
        if (it != copy_engine_connections_.end()) {
            conn = &it->second;
            fd = conn->fd;
            // Reset progress tracking for new batch
            conn->last_progress = 0;
            conn->finalized_received = false;
            conn->finalized_value = 0;
        } else {
            fd = connectToCopyEngine(server_url);
            if (fd < 0) {
                std::cerr << "Failed to connect to FlexTransferEngine at "
                          << server_url << std::endl;
                return -1;
            }
            auto result = copy_engine_connections_.emplace(
                std::piecewise_construct, std::forward_as_tuple(server_url),
                std::forward_as_tuple(fd));
            conn = &result.first->second;
        }
    }

    // Return connection pointer if requested
    if (out_conn) *out_conn = conn;

    // Send protocol to remote FlexTransferEngine:
    // 1. Segment name length (4 bytes)
    // 2. Segment name (variable length)
    // 3. Progress address (8 bytes)
    // 4. Number of requests (8 bytes)
    // 5. For each request: source_addr (8 bytes), target_addr (8 bytes), length
    // (8 bytes)

    // Send segment name length
    uint32_t segment_name_len = local_server_name_.size();
    if (writeFully(fd, &segment_name_len, sizeof(segment_name_len)) !=
        sizeof(segment_name_len)) {
        throw std::runtime_error(
            "Failed to send segment name length to FlexTransferEngine");
    }

    // Send segment name
    if (writeFully(fd, local_server_name_.c_str(), segment_name_len) !=
        segment_name_len) {
        throw std::runtime_error(
            "Failed to send segment name to FlexTransferEngine");
    }

    // Send batch info
    struct BatchInfo {
        uint64_t progress_addr;
        uint64_t num_requests;
    };

    BatchInfo batch_info;
    batch_info.progress_addr =
        reinterpret_cast<uint64_t>(&ctrl_block->progress_counter);
    batch_info.num_requests = entries.size();

    if (writeFully(fd, &batch_info, sizeof(batch_info)) != sizeof(batch_info)) {
        throw std::runtime_error(
            "Failed to send batch info to FlexTransferEngine");
    }

    // Send request details (source on remote FlexTransferEngine, target on
    // local FlexTransferEngine)
    for (const auto &entry : entries) {
        assert(entry.opcode == OPCODE_READ);
        struct RequestInfo {
            uint64_t source_addr;
            uint64_t target_addr;
            uint64_t length;
        };

        RequestInfo req_info;
        req_info.source_addr = reinterpret_cast<uint64_t>(entry.source);
        req_info.target_addr = entry.target_offset;
        req_info.length = entry.length;

        if (writeFully(fd, &req_info, sizeof(req_info)) != sizeof(req_info)) {
            throw std::runtime_error(
                "Failed to send request info to FlexTransferEngine");
        }
    }

    // The remote FlexTransferEngine will now process the requests
    // asynchronously and update the progress counter via RDMA writes. The
    // caller should poll the progress to check completion.

    std::cerr << "Submitted " << entries.size()
              << " requests to FlexTransferEngine at " << server_url
              << std::endl;

    return 0;
}

/* Private functions */

int FlexTransferEngine::connectToCopyEngine(const std::string &server_url) {
    // Parse server_url in format "ip_addr:port"
    // For IPv6: "[ipv6_addr]:port" or "ipv6_addr:port"
    // For IPv4: "ipv4_addr:port"

    std::string hostname;
    std::string port_str;

    // Check for IPv6 format with brackets: [addr]:port
    if (server_url[0] == '[') {
        size_t bracket_end = server_url.find(']');
        if (bracket_end == std::string::npos) {
            std::cerr << "Invalid IPv6 format in server_url: " << server_url
                      << std::endl;
            return -1;
        }
        hostname = server_url.substr(1, bracket_end - 1);

        // Check if there's a port after the bracket
        if (bracket_end + 1 < server_url.length() &&
            server_url[bracket_end + 1] == ':') {
            port_str = server_url.substr(bracket_end + 2);
        } else {
            std::cerr << "Missing port in server_url: " << server_url
                      << std::endl;
            return -1;
        }
    } else {
        // IPv4 format or IPv6 without brackets
        size_t last_colon = server_url.rfind(':');
        if (last_colon == std::string::npos) {
            std::cerr << "Missing port in server_url: " << server_url
                      << std::endl;
            return -1;
        }
        hostname = server_url.substr(0, last_colon);
        port_str = server_url.substr(last_colon + 1);
    }

    // Resolve hostname
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;  // Support both IPv4 and IPv6
    hints.ai_socktype = SOCK_STREAM;

    int rc = getaddrinfo(hostname.c_str(), port_str.c_str(), &hints, &result);
    if (rc) {
        std::cerr << "getaddrinfo failed: " << gai_strerror(rc) << std::endl;
        return -1;
    }

    // Try to connect
    int fd = -1;
    for (struct addrinfo *rp = result; rp != nullptr; rp = rp->ai_next) {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) break;  // Success
        close(fd);
        fd = -1;
    }

    freeaddrinfo(result);

    if (fd < 0) {
        std::cerr << "Failed to connect to " << server_url << std::endl;
        return -1;
    }

    // Set socket to non-blocking for progress checking
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    std::cerr << "Connected to FlexTransferEngine at " << server_url
              << std::endl;
    return fd;
}
