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
#include <unordered_set>

#include "transfer_engine_c.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "util.h"

FlexTransferEngine::~FlexTransferEngine() {
    if (enable_copy_) {
        stopListener();

        std::lock_guard<std::mutex> lock(regions_mutex_);
        for (auto pair : buffer_pool_) freeBufferPair(pair);
        buffer_pool_.clear();
    }

    for (const auto &[_, fd] : copy_engine_connections_) close(fd);

    {
        std::lock_guard<std::mutex> segment_lock(segment_cache_mutex_);
        for (auto &[segment_name, segment_id] : segment_cache_)
            ::closeSegment(engine_, segment_id);
        segment_cache_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(ctrl_block_mutex_);
        for (CopyCtrlBlock *ctrl_block : copy_ctrl_block_cache_) {
            if (engine_) ::unregisterLocalMemory(engine_, ctrl_block);
            delete ctrl_block;
        }
        copy_ctrl_block_cache_.clear();
    }

    if (engine_) ::destroyTransferEngine(engine_);
}

int FlexTransferEngine::init(const std::string &metadata_conn_string,
                             const std::string &local_server_name,
                             bool auto_discover) {
    local_server_name_ = local_server_name;

    // Create the underlying TransferEngine
    engine_ = ::createTransferEngine(metadata_conn_string.c_str(),
                                     local_server_name.c_str(),
                                     /*unused*/ local_server_name.c_str(),
                                     /*unused*/ 12345, auto_discover);
    if (!engine_) {
        std::cerr << "Failed to create TransferEngine" << std::endl;
        return -1;
    }

    // Start TCP listener only if enable_copy_ is true
    if (enable_copy_) {
        int ret = startListener();
        if (ret < 0) {
            std::cerr << "Failed to start TCP listener" << std::endl;
            return ret;
        }
        std::cerr << "FlexTransferEngine initialized successfully with copy "
                     "support enabled"
                  << std::endl;
    } else {
        std::cerr << "FlexTransferEngine initialized successfully in direct "
                     "mode"
                  << std::endl;
    }

    return 0;
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
        if (pair) freeBufferPair(pair);
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
        if (pair) freeBufferPair(pair);
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
    int ret = 0;

    std::lock_guard<std::mutex> regions_lock(regions_mutex_);
    for (void *addr : addr_list) {
        auto it = copiable_regions_.find(addr);
        if (it != copiable_regions_.end()) {
            copiable_regions_.erase(it);
        } else {
            ret = -1;  // Not found, but will continue
        }
    }

    return ret;
}

int FlexTransferEngine::syncSegmentCache() {
    return ::syncSegmentCache(engine_);
}

CopyCtrlBlock *FlexTransferEngine::acquireCopyCtrlBlock() {
    std::lock_guard<std::mutex> lock(ctrl_block_mutex_);

    // Try to get from cache first
    if (!copy_ctrl_block_cache_.empty()) {
        CopyCtrlBlock *ctrl_block = copy_ctrl_block_cache_.back();
        copy_ctrl_block_cache_.pop_back();
        ctrl_block->progress_counter = 0;
        return ctrl_block;
    }

    // Cache is empty, allocate a new one
    CopyCtrlBlock *ctrl_block = new CopyCtrlBlock();
    ctrl_block->progress_counter = 0;

    // Register it with RDMA using the specified location
    int ret = ::registerLocalMemory(engine_, ctrl_block, sizeof(CopyCtrlBlock),
                                    ctrl_block_location_.c_str(), 1);
    if (ret < 0) {
        std::cerr << "Failed to register CopyCtrlBlock with RDMA" << std::endl;
        delete ctrl_block;
        return nullptr;
    }

    std::cerr << "Allocated and registered new CopyCtrlBlock at " << ctrl_block
              << std::endl;
    return ctrl_block;
}

void FlexTransferEngine::releaseCopyCtrlBlock(CopyCtrlBlock *ctrl_block) {
    if (!ctrl_block) return;
    std::lock_guard<std::mutex> lock(ctrl_block_mutex_);
    copy_ctrl_block_cache_.push_back(ctrl_block);
}

int FlexTransferEngine::submitTransferToCopyEngine(
    std::vector<transfer_request_t> &entries, const std::string &server_url,
    CopyCtrlBlock *ctrl_block) {
    // Verify all entries are read requests
    for (const auto &entry : entries) {
        if (entry.opcode != OPCODE_READ) {
            std::cerr << "Only read requests are supported when target is "
                         "FlexTransferEngine with copy mode"
                      << std::endl;
            return -1;
        }
    }

    if (entries.empty()) return 0;

    if (!ctrl_block) {
        std::cerr << "Error: ctrl_block is nullptr for copy-based transfer"
                  << std::endl;
        return -1;
    }

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
        std::cerr << "Failed to send segment name length to FlexTransferEngine"
                  << std::endl;
        return -1;
    }

    // Send segment name
    if (writeFully(fd, local_server_name_.c_str(), segment_name_len) !=
        segment_name_len) {
        std::cerr << "Failed to send segment name to FlexTransferEngine"
                  << std::endl;
        return -1;
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
        std::cerr << "Failed to send batch info to FlexTransferEngine"
                  << std::endl;
        return -1;
    }

    // Send request details (source on remote FlexTransferEngine, target on
    // local FlexTransferEngine)
    for (const auto &entry : entries) {
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
            std::cerr << "Failed to send request info to FlexTransferEngine"
                      << std::endl;
            return -1;
        }
    }

    // The remote FlexTransferEngine will now process the requests
    // asynchronously and update the progress counter via RDMA writes. The
    // caller should poll the progress or use getTransferStatus to check
    // completion.

    std::cerr << "Submitted " << entries.size()
              << " requests to FlexTransferEngine at " << server_url
              << std::endl;

    return 0;
}

int FlexTransferEngine::startListener() {
    // Use findAvailableTcpPort to find an available port
    uint16_t tcp_port = findAvailableTcpPort(listener_fd_);
    if (tcp_port == 0) {
        std::cerr << "Failed to find available TCP port" << std::endl;
        return -1;
    }

    // The socket is already bound by findAvailableTcpPort, just listen
    if (listen(listener_fd_, 128) < 0) {
        std::cerr << "Failed to listen on port " << tcp_port << ": "
                  << strerror(errno) << std::endl;
        close(listener_fd_);
        listener_fd_ = -1;
        return -1;
    }

    // Determine the IP address to use for the server URL
    auto ip_list = findLocalIpAddresses();
    if (ip_list.empty() || ip_list[0].empty()) {
        std::cerr << "Failed to find local IP addresses" << std::endl;
        return -1;
    }
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
    return 0;
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

    void *progress_addr = reinterpret_cast<void *>(batch_info.progress_addr);

    std::cerr << "Received transfer request: progress_addr=0x" << std::hex
              << batch_info.progress_addr << std::dec
              << ", num_requests=" << batch_info.num_requests << std::endl;

    // Read request details
    struct Request {
        void *source_addr;
        uint64_t target_addr;
        size_t length;
    };
    std::vector<Request> requests;

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

        Request req;
        req.source_addr = reinterpret_cast<void *>(req_info.source_addr);
        req.target_addr = req_info.target_addr;
        req.length = req_info.length;
        requests.push_back(req);
    }

    std::lock_guard<std::mutex> regions_lock(regions_mutex_);

    // these buffer pairs are used when processing this request; when processing
    // is done, ensure all these buffer parirs are not used by any batches
    std::unordered_set<struct BufferPair *> active_buffer_pairs;

    for (size_t i = 0; i < requests.size(); ++i) {
        void *source_addr = requests[i].source_addr;
        size_t length = requests[i].length;
        MemoryRegion *region = getRegion(source_addr, length);
        if (!region) {
            std::cerr << "Source address " << source_addr
                      << " not in registered copiable regions" << std::endl;
            goto err;
        }

        // Get a buffer pair for this location
        BufferPair *buffer_pair = getBufferPair(region->loc_idx, length);
        assert(buffer_pair);
        active_buffer_pairs.insert(buffer_pair);

        // Use one of the buffers (alternate between them for double buffering)
        int buffer_idx = waitOneBufferAvailable(buffer_pair);
        assert(buffer_pair->buffers_user[buffer_idx] == INVALID_BATCH);
        void *buffer = buffer_pair->buffers[buffer_idx];

        // Copy data from source to buffer
        int ret = copyMemory(buffer, source_addr, length, buffer_pair->is_cuda);
        if (ret < 0) {
            std::cerr << "Failed to copy memory from " << source_addr
                      << " to buffer " << buffer << std::endl;
            return;
        }

        // Submit RDMA write from buffer to remote target
        batch_id_t batch_id = ::allocateBatchID(engine_, 1);
        transfer_request_t write_req;
        write_req.opcode = OPCODE_WRITE;
        write_req.source = buffer;
        write_req.target_id = target_segment_id;
        write_req.target_offset = requests[i].target_addr;
        write_req.length = length;

        int submit_ret = ::submitTransfer(engine_, batch_id, &write_req, 1);
        if (submit_ret < 0) {
            std::cerr << "Failed to submit RDMA write" << std::endl;
            ::freeBatchID(engine_, batch_id);
            return;
        }

        buffer_pair->buffers_user[buffer_idx] = batch_id;
    }

    for (auto pair : active_buffer_pairs) waitAllBuffersAvailable(pair);

    // Update progress to indicate completion
    {
        int64_t completion_val = requests.size();
        batch_id_t batch_id = ::allocateBatchID(engine_, 1);
        transfer_request_t progress_req;
        progress_req.opcode = OPCODE_WRITE;
        progress_req.source = &completion_val;
        progress_req.target_id = target_segment_id;
        progress_req.target_offset = reinterpret_cast<uint64_t>(progress_addr);
        progress_req.length = sizeof(int64_t);
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

    for (auto pair : active_buffer_pairs) waitAllBuffersAvailable(pair);
}

int FlexTransferEngine::waitOneBufferAvailable(BufferPair *pair) {
    // Check if any buffer is immediately available
    for (int i = 0; i < 2; ++i) {
        if (pair->buffers_user[i] == INVALID_BATCH) return i;
    }

    [[maybe_unused]] int rc;
    transfer_status_t status;
    while (true) {
        for (int i = 0; i < 2; ++i) {
            batch_id_t batch_id = pair->buffers_user[i];
            rc = ::getTransferStatus(engine_, batch_id, 0, &status);
            assert(rc == 0);
            // TODO: error handling
            if (status.status != STATUS_PENDING) {
                // Buffer is now available (batch completed or errored)
                ::freeBatchID(engine_, batch_id);
                pair->buffers_user[i] = INVALID_BATCH;
                return i;
            }
        }
    }
}

void FlexTransferEngine::waitAllBuffersAvailable(BufferPair *pair) {
    [[maybe_unused]] int rc;
    transfer_status_t status;
    for (int i = 0; i < 2; ++i) {
        batch_id_t batch_id = pair->buffers_user[i];
        if (batch_id == INVALID_BATCH) continue;
        rc = ::getTransferStatus(engine_, batch_id, 0, &status);
        assert(rc == 0);
        // TODO: error handling
        if (status.status != STATUS_PENDING) {
            // Buffer is now available (batch completed or errored)
            ::freeBatchID(engine_, batch_id);
            pair->buffers_user[i] = INVALID_BATCH;
        }
    }
}

segment_id_t FlexTransferEngine::getSegmentId(const std::string &segment_name) {
    std::lock_guard<std::mutex> lock(segment_cache_mutex_);
    auto it = segment_cache_.find(segment_name);
    if (it != segment_cache_.end()) return it->second;

    segment_id_t segment_id = ::openSegment(engine_, segment_name.c_str());
    if (segment_id < 0) {
        std::cerr << "Failed to open segment: " << segment_name << std::endl;
        return -1;
    }
    segment_cache_[segment_name] = segment_id;
    return segment_id;
}

// Require regions_mutex_ to be held before calling
FlexTransferEngine::BufferPair *FlexTransferEngine::getBufferPair(
    LocIdx loc_idx, size_t size) {
    // Check if we have a suitable buffer pair
    BufferPair *pair = buffer_pool_[loc_idx];
    if (pair && pair->size >= size) return pair;

    // No suitable buffer pair found or need larger size, allocate a new one
    if (pair) freeBufferPair(pair);

    const std::string &location = location_strings_[loc_idx];
    pair = allocBufferPair(loc_idx, location, size);
    buffer_pool_[loc_idx] = pair;

    return pair;
}

// Require regions_mutex_ to be held before calling
FlexTransferEngine::BufferPair *FlexTransferEngine::allocBufferPair(
    LocIdx loc_idx, const std::string &location, size_t size) {
    FlexTransferEngine::BufferPair *pair = new FlexTransferEngine::BufferPair();
    pair->size = size;
    pair->is_cuda = (location.find("cuda:") == 0);
    pair->buffers_user[0] = INVALID_BATCH;
    pair->buffers_user[1] = INVALID_BATCH;

    // Allocate one contiguous buffer that's 2*size
    size_t total_size = 2 * size;
    if (pair->is_cuda) {
#ifdef USE_CUDA
        cudaError_t err = cudaMalloc(&pair->buffers[0], total_size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate GPU memory: "
                      << cudaGetErrorString(err) << std::endl;
            delete pair;
            return nullptr;
        }
        // Split into two halves
        pair->buffers[1] = static_cast<char *>(pair->buffers[0]) + size;
#else
        std::cerr << "GPU memory requested but CUDA support not compiled"
                  << std::endl;
        delete pair;
        return nullptr;
#endif
    } else {
        pair->buffers[0] = new char[total_size];
        pair->buffers[1] = static_cast<char *>(pair->buffers[0]) + size;
    }

    // Register the entire contiguous buffer with RDMA
    int ret = ::registerLocalMemory(engine_, pair->buffers[0], total_size,
                                    location.c_str(), 1);
    if (ret < 0) {
        std::cerr << "Failed to register buffer with RDMA" << std::endl;
        if (pair->is_cuda) {
#ifdef USE_CUDA
            cudaFree(pair->buffers[0]);
#endif
        } else {
            delete[] static_cast<char *>(pair->buffers[0]);
        }
        delete pair;
        return nullptr;
    }

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
        delete[] static_cast<char *>(pair->buffers[0]);
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
        std::cerr << "GPU memory copy requested but CUDA support not compiled"
                  << std::endl;
        return -1;
#endif
    } else {
        memcpy(dst, src, size);
        return 0;
    }
}

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

    int ret = getaddrinfo(hostname.c_str(), port_str.c_str(), &hints, &result);
    if (ret != 0) {
        std::cerr << "getaddrinfo failed: " << gai_strerror(ret) << std::endl;
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
