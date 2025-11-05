#include "flex_transfer_engine.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
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
      worker_running_(false),
      listener_fd_(-1) {
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
    if (enable_copy_) startListener();
}

FlexTransferEngine::~FlexTransferEngine() {
    if (enable_copy_) {
        stopListener();

        std::lock_guard<std::mutex> lock(regions_mutex_);
        for (auto pair : buffer_pool_) freeBufferPair(pair);
        // buffer_pool_.clear();
    }

    for (const auto &[_, fd] : copy_engine_connections_) close(fd);

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
    // Do actual RDMA registration
    if (force_direct || !enable_copy_) {
        return ::registerLocalMemory(
            engine_, addr, length,
            location.empty() ? nullptr : location.c_str(), remote_accessible);
    }
    // else: register for copy-based transfer

    /**
     * Note here we don't track the duplicated regions: if the same region is
     * registered multiple times, only the last one is kept.
     * If there are duplicated unregistrations, only the first one will succeed.
     * Other unregistrations will return -1 (not found).
     */
    std::lock_guard<std::mutex> regions_lock(regions_mutex_);

    LocIdx loc_idx = getLocIdx(location);
    copiable_regions_[addr] = {addr, length, loc_idx};

    std::cerr << "Registered memory at " << addr << " size " << length
              << " location " << location << std::endl;

    // Check if we have a buffer pair for this location
    BufferPair *pair = buffer_pool_[loc_idx];
    if (!pair || pair->size < length) {
        freeBufferPair(pair);
        pair = allocBufferPair(loc_idx, location, length);
        buffer_pool_[loc_idx] = pair;
        if (!pair) goto err;
        std::cerr << "Allocated buffer pair of size " << length
                  << " for location " << location << std::endl;
    }

    return 0;

err:

    copiable_regions_.erase(addr);
    return -1;
}

int FlexTransferEngine::unregisterLocalMemory(void *addr, bool force_direct) {
    // Do actual RDMA unregistration if force_direct or !enable_copy_
    if (force_direct || !enable_copy_)
        return ::unregisterLocalMemory(engine_, addr);

    std::lock_guard<std::mutex> regions_lock(regions_mutex_);
    size_t num_erased = copiable_regions_.erase(addr);
    return num_erased > 0 ? 0 : -1;
}

int FlexTransferEngine::registerLocalMemoryBatch(
    std::vector<buffer_entry_t> &buffer_list, const std::string &location,
    bool force_direct) {
    // Do actual RDMA registration if force_direct or !enable_copy_
    if (force_direct || !enable_copy_) {
        return ::registerLocalMemoryBatch(engine_, buffer_list.data(),
                                          buffer_list.size(), location.c_str());
    }

    std::lock_guard<std::mutex> regions_lock(regions_mutex_);
    LocIdx loc_idx = getLocIdx(location);
    size_t max_size = 0;

    for (const auto &entry : buffer_list) {
        if (entry.length > max_size) max_size = entry.length;
        copiable_regions_[entry.addr] = {entry.addr, entry.length, loc_idx};
    }
    std::cerr << "Registered " << buffer_list.size() << " buffers for location "
              << location << ", max size " << max_size << std::endl;

    // Check if we have a buffer pair for this location
    BufferPair *pair = buffer_pool_[loc_idx];
    if (!pair || pair->size < max_size) {
        freeBufferPair(pair);
        pair = allocBufferPair(loc_idx, location, max_size);
        buffer_pool_[loc_idx] = pair;
        if (!pair) goto err;
        std::cerr << "Allocated buffer pair of size " << max_size
                  << " for location " << location << std::endl;
    }

    return 0;

err:

    for (const auto &entry : buffer_list) copiable_regions_.erase(entry.addr);
    return -1;
}

int FlexTransferEngine::unregisterLocalMemoryBatch(
    std::vector<void *> &addr_list, bool force_direct) {
    // Do actual RDMA unregistration if force_direct or !enable_copy_
    if (force_direct || !enable_copy_) {
        return ::unregisterLocalMemoryBatch(engine_, addr_list.data(),
                                            addr_list.size());
    }

    /**
     * If the same memory region is registered multiple times, only the last one
     * will be preserved. In that case, if calling unregister on the same number
     * of times, only the first one will succeed, the rest will be considered as
     * an error. However, in a batch mode, we prefer to tolerate this error:
     * other memory regions are unaffected (will continue the unregisteration),
     * and only return -1 to indicate that at least one region failed.
     */
    int rc = 0;

    std::lock_guard<std::mutex> regions_lock(regions_mutex_);
    for (void *addr : addr_list) {
        auto it = copiable_regions_.find(addr);
        if (it != copiable_regions_.end()) {
            copiable_regions_.erase(it);
        } else {
            rc = -1;  // Not found, but will continue
        }
    }

    return rc;
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
    CopyCtrlBlock *ctrl_block) {
    if (entries.empty()) return 0;
    assert(ctrl_block);

    // Connect to the remote FlexTransferEngine (or reuse existing connection)
    int fd = -1;
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = copy_engine_connections_.find(server_url);
        if (it != copy_engine_connections_.end()) {
            fd = it->second;
        } else {
            fd = connectToCopyEngine(server_url);
            if (fd < 0) {
                std::cerr << "Failed to connect to FlexTransferEngine at "
                          << server_url << std::endl;
                return -1;
            }
            copy_engine_connections_[server_url] = fd;
        }
    }

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

// Require regions_mutex_ to be held before calling
FlexTransferEngine::LocIdx FlexTransferEngine::getLocIdx(
    const std::string &location) {
    // Linear search to find existing location
    for (size_t i = 0; i < location_strings_.size(); ++i) {
        if (location_strings_[i] == location) return static_cast<LocIdx>(i);
    }
    // Not found, add new location
    LocIdx new_idx = static_cast<LocIdx>(location_strings_.size());
    location_strings_.emplace_back(location);
    buffer_pool_.emplace_back(nullptr);
    return new_idx;
}

// Require regions_mutex_ to be held before calling
FlexTransferEngine::MemoryRegion *FlexTransferEngine::getRegion(void *addr,
                                                                size_t length) {
    // fast path: the addr is the base of a registered region; we expect it to
    // be a common case
    auto it = copiable_regions_.find(addr);
    if (it != copiable_regions_.end() && length <= it->second.length)
        return &it->second;
    // slow path: scan to find addr within a registered region; scan is
    // acceptable because we don't expect to have too many regions
    for (auto &[base_addr, region] : copiable_regions_) {
        if (addr < static_cast<char *>(base_addr)) continue;
        if (static_cast<char *>(addr) + length >
            static_cast<char *>(base_addr) + region.length)
            continue;
        return &region;
    }
    return nullptr;
}

// Weakly wait for a task to complete
// Returns 0 on success, -1 on failure
int FlexTransferEngine::waitTask(Task &task) {
    if (task.batch_id == INVALID_BATCH) return 0;

    [[maybe_unused]] int rc;
    transfer_status_t status;

    while (true) {
        rc = ::getTransferStatus(engine_, task.batch_id, 0, &status);
        assert(rc == 0);
        if (status.status != STATUS_WAITING) {  // completed or error
            task.buffer_pair->users[task.buffer_idx] = -1;  // mark buffer free
            ::freeBatchID(engine_, task.batch_id);
            task.batch_id = INVALID_BATCH;
            task.buffer_pair = nullptr;
            task.buffer_idx = -1;
            return status.status == STATUS_COMPLETED ? 0 : -1;
        }
    }
}

// Require regions_mutex_ to be held before calling
FlexTransferEngine::BufferPair *FlexTransferEngine::allocBufferPair(
    LocIdx loc_idx, const std::string &location, size_t size) {
    bool is_cuda = location.find("cuda:") == 0;
    size_t total_size = 2 * size;  // allocate one contiguous buffer w/ 2*size
    char *buffer_base = nullptr;
    if (is_cuda) {
#ifdef USE_CUDA
        cudaError_t err = cudaMalloc(&buffer_base, total_size);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("Failed to allocate GPU memory: ") +
                cudaGetErrorString(err));
            return nullptr;
        }
#else
        throw std::runtime_error(
            "GPU memory requested but CUDA support not compiled");
#endif
    } else {
        buffer_base = new char[total_size];
    }

    // Register the entire contiguous buffer with RDMA
    int rc =
        ::registerLocalMemory(engine_, buffer_base, total_size,
                              location.c_str(), /*remote_accessible*/ true);
    if (rc) {
        if (is_cuda) {
#ifdef USE_CUDA
            cudaFree(buffer_base);
#endif
        } else {
            delete[] buffer_base;
        }
        return nullptr;
        // throw std::runtime_error("Failed to register buffer with RDMA");
    }

    FlexTransferEngine::BufferPair *pair =
        new FlexTransferEngine::BufferPair(buffer_base, size, is_cuda);

    std::cerr << "Allocated buffer pair of size " << size << " for location "
              << location << " (total=" << total_size << ")" << std::endl;
    return pair;
}

// Require regions_mutex_ to be held before calling
void FlexTransferEngine::freeBufferPair(BufferPair *pair) {
    if (!pair) return;

    ::unregisterLocalMemory(engine_, pair->buffers[0]);

    if (pair->is_cuda) {
#ifdef USE_CUDA
        cudaFree(pair->buffers[0]);
#endif
    } else {
        delete[] pair->buffers[0];
    }

    delete pair;
}

int FlexTransferEngine::copyMemory(void *dst, const void *src, size_t size,
                                   bool is_cuda) {
    if (is_cuda) {
#ifdef USE_CUDA
        cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err)
                      << std::endl;
            return -1;
        }
        return 0;
#else
        throw std::runtime_error(
            "GPU memory copy requested but CUDA support not compiled");
#endif
    } else {
        memcpy(dst, src, size);
        return 0;
    }
}

/* TCP listener and worker thread functions */

void FlexTransferEngine::startListener() {
    // Use findAvailableTcpPort to find an available port
    uint16_t tcp_port = findAvailableTcpPort(listener_fd_);
    if (tcp_port == 0)
        throw std::runtime_error("Failed to find available TCP port");

    // The socket is already bound by findAvailableTcpPort, just listen
    if (listen(listener_fd_, 128) < 0) {
        close(listener_fd_);
        listener_fd_ = -1;
        throw std::runtime_error("Failed to listen on port " +
                                 std::to_string(tcp_port) + ": " +
                                 strerror(errno));
    }

    // Determine the IP address to use for the server URL
    auto ip_list = findLocalIpAddresses();
    if (ip_list.empty() || ip_list[0].empty())
        throw std::runtime_error("Failed to find local IP addresses");
    const std::string &server_ip = ip_list[0];

    // Set local_copy_server_url_
    std::ostringstream oss;
    // Check if the IP is IPv6 (contains ':')
    bool is_ipv6 = (server_ip.find(':') != std::string::npos);
    if (is_ipv6)
        oss << "[" << server_ip << "]:" << tcp_port;
    else
        oss << server_ip << ":" << tcp_port;

    local_copy_server_url_ = oss.str();

    std::cerr << "TCP listener started on " << local_copy_server_url_
              << std::endl;

    // finally, start worker thread
    worker_running_ = true;
    worker_thread_ = std::thread(&FlexTransferEngine::workerThread, this);
}

void FlexTransferEngine::stopListener() {
    if (worker_running_.load(std::memory_order_acquire)) {
        worker_running_.store(false, std::memory_order_release);
        // Wait for worker thread to finish
        if (worker_thread_.joinable()) worker_thread_.join();

        // Close listener socket to unblock accept()
        if (listener_fd_ >= 0) {
            close(listener_fd_);
            listener_fd_ = -1;
        }

        std::cerr << "TCP listener stopped" << std::endl;
    }
}

void FlexTransferEngine::workerThread() {
    std::cerr << "Worker thread started" << std::endl;

    while (worker_running_.load(std::memory_order_acquire)) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        int client_fd =
            accept(listener_fd_, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            std::cerr << "Failed to accept connection: " << strerror(errno)
                      << std::endl;
            continue;
        }

        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        std::cerr << "Accepted connection from " << client_ip << ":"
                  << ntohs(client_addr.sin_port) << std::endl;

        // Handle and process the request directly
        handleAndProcessRequest(client_fd);
        close(client_fd);
    }

    std::cerr << "Worker thread stopped" << std::endl;
}

void FlexTransferEngine::handleAndProcessRequest(int client_fd) {
    // Read segment name length
    uint32_t segment_name_len;
    if (readFully(client_fd, &segment_name_len, sizeof(segment_name_len)) !=
        sizeof(segment_name_len)) {
        std::cerr << "Failed to read segment name length" << std::endl;
        return;
    }

    const static size_t kMaxLength = 1ull << 20;
    if (segment_name_len == 0 || segment_name_len > kMaxLength) {
        std::cerr << "Invalid segment name length: " << segment_name_len
                  << std::endl;
        return;
    }

    // Read segment name
    std::string segment_name(segment_name_len + 1, '\0');
    if (readFully(client_fd, segment_name.data(), segment_name_len) !=
        segment_name_len) {
        std::cerr << "Failed to read segment name" << std::endl;
        return;
    }

    std::cerr << "Received request for segment: " << segment_name << std::endl;

    // Get or open segment
    segment_id_t target_segment_id = getSegmentId(segment_name);
    if (target_segment_id < 0) {
        std::cerr << "Failed to open segment: " << segment_name << std::endl;
        return;
    }

    // Read batch info
    struct BatchInfo {
        uint64_t progress_addr;
        uint64_t num_requests;
    };

    BatchInfo batch_info;
    if (readFully(client_fd, &batch_info, sizeof(batch_info)) !=
        sizeof(batch_info)) {
        std::cerr << "Failed to read batch info" << std::endl;
        return;
    }

    std::cerr << "Received transfer request: progress_addr=0x" << std::hex
              << batch_info.progress_addr << std::dec
              << ", num_requests=" << batch_info.num_requests << std::endl;

    // Read request details and convert them into tasks
    std::vector<Task> tasks;

    for (uint64_t i = 0; i < batch_info.num_requests; ++i) {
        struct RequestInfo {
            uint64_t source_addr;
            uint64_t target_addr;
            uint64_t length;
        };

        RequestInfo req_info;
        if (readFully(client_fd, &req_info, sizeof(req_info)) !=
            sizeof(req_info)) {
            std::cerr << "Failed to read request info" << std::endl;
            return;
        }
        tasks.emplace_back(reinterpret_cast<void *>(req_info.source_addr),
                           req_info.target_addr, req_info.length);
    }

    CopyCtrlBlock *copy_ctrl_block = acquireCopyCtrlBlock();
    if (!copy_ctrl_block) {
        std::cerr << "Failed to acquire CopyCtrlBlock" << std::endl;
        return;
    }

    int num_tasks_done = 0;
    std::lock_guard<std::mutex> regions_lock(regions_mutex_);

    for (size_t i = 0; i < tasks.size(); ++i) {
        Task &task = tasks[i];

        // delayed source address validation:
        // if source_addr is invalid, will be detected here
        MemoryRegion *region = getRegion(task.source_addr, task.length);
        if (!region) {
            std::cerr << "Source address " << task.source_addr
                      << " not in registered copiable regions" << std::endl;
            goto err;
        }

        // Get a buffer pair for this location
        BufferPair &buffer_pair = getBufferPair(region->loc_idx);
        assert(buffer_pair.size >= task.length);

        int buffer_idx = buffer_pair.selectNextBuffer();
        int buffer_used_by_task_idx = buffer_pair.users[buffer_idx];
        if (buffer_used_by_task_idx >= 0) {
            int rc = waitTask(tasks[buffer_used_by_task_idx]);
            if (rc) {
                std::cerr << "Failed to wait for previous task on buffer "
                          << buffer_idx << std::endl;
                goto err;
            }
        }

        void *buffer = buffer_pair.buffers[buffer_idx];

        // Copy data from source to buffer
        int rc = copyMemory(buffer, task.source_addr, task.length,
                            buffer_pair.is_cuda);
        if (rc) {
            std::cerr << "Failed to copy memory from " << task.source_addr
                      << " to buffer " << buffer << std::endl;
            goto err;
        }

        // Submit RDMA write from buffer to remote target
        batch_id_t batch_id = ::allocateBatchID(engine_, 1);
        transfer_request_t write_req = {
            .opcode = OPCODE_WRITE,
            .source = buffer,
            .target_id = target_segment_id,
            .target_offset = task.target_addr,
            .length = task.length,
        };

        int submit_rc = ::submitTransfer(engine_, batch_id, &write_req, 1);
        if (submit_rc < 0) {
            std::cerr << "Failed to submit RDMA write" << std::endl;
            ::freeBatchID(engine_, batch_id);
            goto err;
        }

        task.batch_id = batch_id;
        task.buffer_pair = &buffer_pair;
        task.buffer_idx = buffer_idx;
    }

    for (auto &task : tasks) {
        int rc = waitTask(task);
        if (rc) {
            std::cerr << "Failed to wait for task completion" << std::endl;
            goto err;
        }
    }

    // Update progress to indicate completion
    {
        copy_ctrl_block->progress_counter.store(tasks.size(),
                                                std::memory_order_release);
        batch_id_t batch_id = ::allocateBatchID(engine_, 1);
        transfer_request_t progress_req = {
            .opcode = OPCODE_WRITE,
            .source = (void *)&(copy_ctrl_block->progress_counter),
            .target_id = target_segment_id,
            .target_offset =
                reinterpret_cast<uint64_t>(batch_info.progress_addr),
            .length = sizeof(int64_t),
        };
        ::submitTransfer(engine_, batch_id, &progress_req, 1);

        // Wait for completion
        transfer_status_t status;
        while (::getTransferStatus(engine_, batch_id, 0, &status) == 0 &&
               status.status != STATUS_COMPLETED) {
            usleep(1);
        }
        ::freeBatchID(engine_, batch_id);

        std::cerr << "Completed transfer request" << std::endl;
    }

err:

    for (auto &task : tasks) {
        int rc = waitTask(task);
        if (rc) std::cerr << "Failed to wait for task completion" << std::endl;
    }
}

int FlexTransferEngine::connectToCopyEngine(const std::string &server_url) {
    // Parse server_url in format "ip_addr:port"
    // For IPv6: "[ipv6_addr]:port" or "ipv6_addr:port"
    // For IPv4: "ipv4_addr:port"

    if (server_url.empty()) {
        std::cerr << "Empty server_url provided" << std::endl;
        return -1;
    }

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

    std::cerr << "Connected to FlexTransferEngine at " << server_url
              << std::endl;
    return fd;
}
