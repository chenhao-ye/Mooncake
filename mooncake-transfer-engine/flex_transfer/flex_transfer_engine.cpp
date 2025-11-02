// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "flex_transfer_engine.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <iostream>
#include <sstream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "util.h"

namespace mooncake {

// FlexBatch implementation

FlexBatch::~FlexBatch() {
    if (engine_) {
        // Return the CopyCtrlBlock to the cache if it exists
        if (ctrl_block_) {
            engine_->releaseCopyCtrlBlock(ctrl_block_);
            ctrl_block_ = nullptr;
        }

        // Free the batch ID if it was allocated
        if (batch_id_ != INVALID_BATCH) {
            ::freeBatchID(engine_->getEngine(), batch_id_);
            batch_id_ = INVALID_BATCH;
        }
    }
}

void FlexBatch::addReadRequest(uintptr_t local_addr, uintptr_t remote_addr,
                               uint64_t size) {
    entries_.emplace_back(
        transfer_request_t{.opcode = OPCODE_READ,
                           .source = reinterpret_cast<void *>(local_addr),
                           .target_id = -1,  // will be set upon submit()
                           .target_offset = remote_addr,
                           .length = size});
}
void FlexBatch::addWriteRequest(uintptr_t local_addr, uintptr_t remote_addr,
                                uint64_t size) {
    entries_.emplace_back(
        transfer_request_t{.opcode = OPCODE_WRITE,
                           .source = reinterpret_cast<void *>(local_addr),
                           .target_id = -1,  // will be set upon submit()
                           .target_offset = remote_addr,
                           .length = size});
}

int FlexBatch::submit(const std::string &target, bool is_target_copy) {
    if (entries_.empty()) {
        std::cerr << "Error: No transfer requests in batch" << std::endl;
        return -1;
    }

    if (!is_target_copy) {  // Then target is a segment name for direct RDMA
        batch_id_ = ::allocateBatchID(engine_->getEngine(), entries_.size());
        if (batch_id_ == INVALID_BATCH) {
            std::cerr << "Failed to allocate batch ID" << std::endl;
            return -1;
        }
        auto target_segment_id = engine_->getSegmentId(target);
        for (auto &entry : entries_) entry.target_id = target_segment_id;
        return ::submitTransfer(engine_->getEngine(), batch_id_,
                                entries_.data(), entries_.size());
    }

    // Copy-based transfer - acquire a CopyCtrlBlock
    ctrl_block_ = engine_->acquireCopyCtrlBlock();
    if (!ctrl_block_) {
        std::cerr << "Failed to acquire CopyCtrlBlock" << std::endl;
        return -1;
    }

    // Submit to remote FlexTransferEngine
    return engine_->submitTransferToCopyEngine(entries_, target, ctrl_block_);
}

int FlexBatch::getTransferStatus(size_t task_id, transfer_status_t &status) {
    if (!ctrl_block_) {  // Direct RDMA transfer
        return ::getTransferStatus(engine_->getEngine(), batch_id_, task_id,
                                   &status);
    }

    // For copy-based transfers, check ctrl_block progress
    int64_t progress = ctrl_block_->progress_counter;
    if (static_cast<int64_t>(task_id) < progress) {  // Task completed
        status.status = STATUS_COMPLETED;
        status.transferred_bytes = entries_[task_id].length;
        return 0;
    }
    // Task still in progress or waiting
    status.status = STATUS_PENDING;
    status.transferred_bytes = 0;
    return 0;
}

FlexTransferEngine::~FlexTransferEngine() {
    // Stop listener if it was running
    stopListener();

    // Clean up buffer pool (only if enable_copy_)
    if (enable_copy_) {
        std::lock_guard<std::mutex> pool_lock(pool_mutex_);
        for (auto &[location, pair] : buffer_pool_) {
            if (pair) {
                ::unregisterLocalMemory(engine_, pair->buffer0);
                if (pair->is_cuda) {
#ifdef USE_CUDA
                    cudaFree(pair->buffer0);
#endif
                } else {
                    delete[] static_cast<char *>(pair->buffer0);
                }
                delete pair;
            }
        }
        buffer_pool_.clear();
    }

    // Close all cached segments
    {
        std::lock_guard<std::mutex> segment_lock(segment_cache_mutex_);
        for (auto &[segment_name, segment_id] : segment_cache_) {
            ::closeSegment(engine_, segment_id);
        }
        segment_cache_.clear();
    }

    // Close all TCP connections
    for (const auto &[_, fd] : copy_engine_connections_) {
        close(fd);
    }

    // Clean up CopyCtrlBlock cache
    {
        std::lock_guard<std::mutex> lock(ctrl_block_mutex_);
        for (CopyCtrlBlock *ctrl_block : copy_ctrl_block_cache_) {
            if (engine_) ::unregisterLocalMemory(engine_, ctrl_block);
            delete ctrl_block;
        }
        copy_ctrl_block_cache_.clear();
    }

    // Destroy the transfer engine
    if (engine_) {
        ::destroyTransferEngine(engine_);
        engine_ = nullptr;
    }
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
            ::destroyTransferEngine(engine_);
            engine_ = nullptr;
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
    // Do actual RDMA registration if force_direct or !enable_copy_
    if (force_direct || !enable_copy_) {
        return ::registerLocalMemory(
            engine_, addr, length,
            location.empty() ? nullptr : location.c_str(), remote_accessible);
    }

    // Track copiable regions
    {
        std::lock_guard<std::mutex> regions_lock(regions_mutex_);

        // Store the memory region info
        copiable_regions_[addr] = {addr, length, location};

        std::cerr << "Registered memory at " << addr << " size " << length
                  << " location " << location << std::endl;
    }

    // Allocate buffer pair only if enable_copy_ is true
    std::lock_guard<std::mutex> pool_lock(pool_mutex_);

    // Find the largest memory region for this location
    size_t max_size = length;
    for (const auto &[_, reg] : copiable_regions_) {
        if (reg.location == location && reg.length > max_size)
            max_size = reg.length;
    }

    // Check if we have a buffer pair for this location
    auto it = buffer_pool_.find(location);
    if (it == buffer_pool_.end() || it->second->size < max_size) {
        // Need to allocate or resize buffer pair
        if (it != buffer_pool_.end()) {
            // Free old buffer pair
            BufferPair *old_pair = it->second;
            ::unregisterLocalMemory(engine_, old_pair->buffer0);
            if (old_pair->is_cuda) {
#ifdef USE_CUDA
                cudaFree(old_pair->buffer0);
#endif
            } else {
                delete[] static_cast<char *>(old_pair->buffer0);
            }
            delete old_pair;
        }

        // Allocate a new buffer pair
        BufferPair *new_pair = allocateBufferPair(location, max_size);
        if (!new_pair) {
            std::cerr << "Failed to allocate buffer pair for location "
                      << location << std::endl;
            copiable_regions_.erase(addr);
            return -1;
        }
        buffer_pool_[location] = new_pair;
        std::cerr << "Allocated buffer pair of size " << max_size
                  << " for location " << location << std::endl;
    }

    return 0;
}

int FlexTransferEngine::unregisterLocalMemory(void *addr, bool force_direct) {
    // Do actual RDMA unregistration if force_direct or !enable_copy_
    if (force_direct || !enable_copy_)
        return ::unregisterLocalMemory(engine_, addr);

    std::lock_guard<std::mutex> lock(regions_mutex_);

    auto it = copiable_regions_.find(addr);
    if (it == copiable_regions_.end()) return -1;  // Not found
    copiable_regions_.erase(it);
    return 0;
}

int FlexTransferEngine::registerLocalMemoryBatch(
    std::vector<buffer_entry_t> &buffer_list, const std::string &location,
    bool force_direct) {
    // Do actual RDMA registration if force_direct or !enable_copy_
    if (force_direct || !enable_copy_) {
        return ::registerLocalMemoryBatch(engine_, buffer_list.data(),
                                          buffer_list.size(), location.c_str());
    }

    // Find the largest buffer in the batch
    size_t max_size = 0;
    // Track copiable regions
    {
        std::lock_guard<std::mutex> regions_lock(regions_mutex_);

        for (const auto &entry : buffer_list) {
            if (entry.length > max_size) max_size = entry.length;
            // Store the memory region info
            copiable_regions_[entry.addr] = {entry.addr, entry.length,
                                             location};
        }

        std::cerr << "Registered " << buffer_list.size()
                  << " buffers for location " << location << ", max size "
                  << max_size << std::endl;
    }

    // Allocate buffer pair only if enable_copy_ is true
    std::lock_guard<std::mutex> pool_lock(pool_mutex_);

    // Check if we have a buffer pair for this location
    auto it = buffer_pool_.find(location);
    if (it == buffer_pool_.end() || it->second->size < max_size) {
        // Need to allocate or resize buffer pair
        if (it != buffer_pool_.end()) {
            // Free old buffer pair
            BufferPair *old_pair = it->second;
            ::unregisterLocalMemory(engine_, old_pair->buffer0);
            if (old_pair->is_cuda) {
#ifdef USE_CUDA
                cudaFree(old_pair->buffer0);
#endif
            } else {
                delete[] static_cast<char *>(old_pair->buffer0);
            }
            delete old_pair;
        }

        // Allocate a new buffer pair
        BufferPair *new_pair = allocateBufferPair(location, max_size);
        if (!new_pair) {
            std::cerr << "Failed to allocate buffer pair for location "
                      << location << std::endl;
            return -1;
        }
        buffer_pool_[location] = new_pair;
        std::cerr << "Allocated buffer pair of size " << max_size
                  << " for location " << location << std::endl;
    }

    return 0;
}

int FlexTransferEngine::unregisterLocalMemoryBatch(
    std::vector<void *> &addr_list, bool force_direct) {
    // Do actual RDMA unregistration if force_direct or !enable_copy_
    if (force_direct || !enable_copy_) {
        return ::unregisterLocalMemoryBatch(engine_, addr_list.data(),
                                            addr_list.size());
    }

    // Remove from tracked copiable regions
    int ret = 0;
    {
        std::lock_guard<std::mutex> lock(regions_mutex_);

        for (void *addr : addr_list) {
            auto it = copiable_regions_.find(addr);
            if (it != copiable_regions_.end()) {
                copiable_regions_.erase(it);
            } else {
                ret = -1;  // Not found, but will continue
            }
        }
    }

    return ret;
}

int FlexTransferEngine::syncSegmentCache() {
    return ::syncSegmentCache(engine_);
}

// Private methods from CopyTransferEngine

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

    // Start worker thread
    worker_running_ = true;
    worker_thread_ = std::thread(&FlexTransferEngine::workerThread, this);

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
    return 0;
}

void FlexTransferEngine::stopListener() {
    if (worker_running_.load(std::memory_order_acquire)) {
        worker_running_.store(false, std::memory_order_release);
        // Wait for worker thread to finish
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }

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
            if (worker_running_.load(std::memory_order_acquire)) {
                std::cerr << "Failed to accept connection: " << strerror(errno)
                          << std::endl;
            }
            continue;
        }

        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        std::cerr << "Accepted connection from " << client_ip << ":"
                  << ntohs(client_addr.sin_port) << std::endl;

        // Handle and process the request directly
        handleAndProcessRequest(client_fd);
    }

    std::cerr << "Worker thread stopped" << std::endl;
}

void FlexTransferEngine::handleAndProcessRequest(int client_fd) {
    // Read segment name length
    uint32_t segment_name_len;
    if (readFully(client_fd, &segment_name_len, sizeof(segment_name_len)) !=
        sizeof(segment_name_len)) {
        std::cerr << "Failed to read segment name length" << std::endl;
        close(client_fd);
        return;
    }

    const static size_t kMaxLength = 1ull << 20;
    if (segment_name_len == 0 || segment_name_len > kMaxLength) {
        std::cerr << "Invalid segment name length: " << segment_name_len
                  << std::endl;
        close(client_fd);
        return;
    }

    // Read segment name
    std::string segment_name(segment_name_len, '\0');
    if (readFully(client_fd, &segment_name[0], segment_name_len) !=
        segment_name_len) {
        std::cerr << "Failed to read segment name" << std::endl;
        close(client_fd);
        return;
    }

    std::cerr << "Received request for segment: " << segment_name << std::endl;

    // Get or open segment
    segment_id_t target_segment_id = getSegmentId(segment_name);
    if (target_segment_id < 0) {
        std::cerr << "Failed to open segment: " << segment_name << std::endl;
        close(client_fd);
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
        close(client_fd);
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
            close(client_fd);
            return;
        }

        Request req;
        req.source_addr = reinterpret_cast<void *>(req_info.source_addr);
        req.target_addr = req_info.target_addr;
        req.length = req_info.length;
        requests.push_back(req);
    }

    // Verify all addresses are registered
    for (size_t i = 0; i < requests.size(); ++i) {
        void *addr = requests[i].source_addr;
        if (!isAddressRegistered(addr)) {
            std::cerr << "Error: Address " << addr << " is not registered"
                      << std::endl;
            // Write -1 to progress address to indicate error
            int64_t error_val = -1;
            batch_id_t batch_id = ::allocateBatchID(engine_, 1);
            transfer_request_t error_req;
            error_req.opcode = OPCODE_READ;
            error_req.source = &error_val;
            error_req.target_id = LOCAL_SEGMENT;
            error_req.target_offset = reinterpret_cast<uint64_t>(progress_addr);
            error_req.length = sizeof(int64_t);
            ::submitTransfer(engine_, batch_id, &error_req, 1);
            // Wait for completion
            transfer_status_t status;
            while (::getTransferStatus(engine_, batch_id, 0, &status) == 0 &&
                   status.status != STATUS_COMPLETED) {
                usleep(1000);
            }
            ::freeBatchID(engine_, batch_id);
            close(client_fd);
            return;
        }
    }

    // Process each request with double buffering
    for (size_t i = 0; i < requests.size(); ++i) {
        void *source_addr = requests[i].source_addr;
        size_t length = requests[i].length;
        std::string location = getLocation(source_addr);

        // Get a buffer pair for this location
        BufferPair *buffer_pair = getOrAllocateBufferPair(location, length);
        if (!buffer_pair) {
            std::cerr << "Failed to get buffer pair for location " << location
                      << std::endl;
            // Write -1 to progress address to indicate error
            int64_t error_val = -1;
            batch_id_t batch_id = ::allocateBatchID(engine_, 1);
            transfer_request_t error_req;
            error_req.opcode = OPCODE_WRITE;
            error_req.source = &error_val;
            error_req.target_id = LOCAL_SEGMENT;
            error_req.target_offset = reinterpret_cast<uint64_t>(progress_addr);
            error_req.length = sizeof(int64_t);
            ::submitTransfer(engine_, batch_id, &error_req, 1);
            transfer_status_t status;
            while (::getTransferStatus(engine_, batch_id, 0, &status) == 0 &&
                   status.status != STATUS_COMPLETED) {
                usleep(1000);
            }
            ::freeBatchID(engine_, batch_id);
            close(client_fd);
            return;
        }

        // Use one of the buffers (alternate between them for double buffering)
        void *buffer =
            (i % 2 == 0) ? buffer_pair->buffer0 : buffer_pair->buffer1;

        // Copy data from source to buffer
        int ret = copyMemory(buffer, source_addr, length, buffer_pair->is_cuda);
        if (ret < 0) {
            std::cerr << "Failed to copy memory from " << source_addr
                      << " to buffer " << buffer << std::endl;
            // Write -1 to progress address to indicate error
            int64_t error_val = -1;
            batch_id_t batch_id = ::allocateBatchID(engine_, 1);
            transfer_request_t error_req;
            error_req.opcode = OPCODE_WRITE;
            error_req.source = &error_val;
            error_req.target_id = LOCAL_SEGMENT;
            error_req.target_offset = reinterpret_cast<uint64_t>(progress_addr);
            error_req.length = sizeof(int64_t);
            ::submitTransfer(engine_, batch_id, &error_req, 1);
            transfer_status_t status;
            while (::getTransferStatus(engine_, batch_id, 0, &status) == 0 &&
                   status.status != STATUS_COMPLETED) {
                usleep(1000);
            }
            ::freeBatchID(engine_, batch_id);
            close(client_fd);
            return;
        }

        // Submit RDMA write from buffer to remote target
        batch_id_t write_batch_id = ::allocateBatchID(engine_, 1);
        transfer_request_t write_req;
        write_req.opcode = OPCODE_WRITE;
        write_req.source = buffer;
        write_req.target_id = target_segment_id;
        write_req.target_offset = requests[i].target_addr;
        write_req.length = length;

        int submit_ret =
            ::submitTransfer(engine_, write_batch_id, &write_req, 1);
        if (submit_ret < 0) {
            std::cerr << "Failed to submit RDMA write" << std::endl;
            ::freeBatchID(engine_, write_batch_id);
            // Write -1 to progress address to indicate error
            int64_t error_val = -1;
            batch_id_t error_batch = ::allocateBatchID(engine_, 1);
            transfer_request_t error_req;
            error_req.opcode = OPCODE_WRITE;
            error_req.source = &error_val;
            error_req.target_id = LOCAL_SEGMENT;
            error_req.target_offset = reinterpret_cast<uint64_t>(progress_addr);
            error_req.length = sizeof(int64_t);
            ::submitTransfer(engine_, error_batch, &error_req, 1);
            transfer_status_t status;
            while (::getTransferStatus(engine_, error_batch, 0, &status) == 0 &&
                   status.status != STATUS_COMPLETED) {
                usleep(1000);
            }
            ::freeBatchID(engine_, error_batch);
            close(client_fd);
            return;
        }

        // Wait for RDMA write to complete
        transfer_status_t write_status;
        while (::getTransferStatus(engine_, write_batch_id, 0, &write_status) ==
                   0 &&
               write_status.status != STATUS_COMPLETED) {
            usleep(1000);
        }
        ::freeBatchID(engine_, write_batch_id);

        if (write_status.status != STATUS_COMPLETED) {
            std::cerr << "RDMA write failed with status " << write_status.status
                      << std::endl;
            // Write -1 to progress address to indicate error
            int64_t error_val = -1;
            batch_id_t error_batch = ::allocateBatchID(engine_, 1);
            transfer_request_t error_req;
            error_req.opcode = OPCODE_WRITE;
            error_req.source = &error_val;
            error_req.target_id = LOCAL_SEGMENT;
            error_req.target_offset = reinterpret_cast<uint64_t>(progress_addr);
            error_req.length = sizeof(int64_t);
            ::submitTransfer(engine_, error_batch, &error_req, 1);
            transfer_status_t status;
            while (::getTransferStatus(engine_, error_batch, 0, &status) == 0 &&
                   status.status != STATUS_COMPLETED) {
                usleep(1000);
            }
            ::freeBatchID(engine_, error_batch);
            close(client_fd);
            return;
        }

        std::cerr << "Processed request " << i << ": copied " << length
                  << " bytes from " << source_addr << " to buffer " << buffer
                  << ", wrote to target segment " << target_segment_id
                  << " offset " << requests[i].target_addr << std::endl;
    }

    // Update progress to indicate completion
    int64_t completion_val = requests.size();
    batch_id_t batch_id = ::allocateBatchID(engine_, 1);
    transfer_request_t progress_req;
    progress_req.opcode = OPCODE_WRITE;
    progress_req.source = &completion_val;
    progress_req.target_id = LOCAL_SEGMENT;
    progress_req.target_offset = reinterpret_cast<uint64_t>(progress_addr);
    progress_req.length = sizeof(int64_t);
    ::submitTransfer(engine_, batch_id, &progress_req, 1);

    // Wait for completion
    transfer_status_t status;
    while (::getTransferStatus(engine_, batch_id, 0, &status) == 0 &&
           status.status != STATUS_COMPLETED) {
        usleep(1000);
    }
    ::freeBatchID(engine_, batch_id);

    close(client_fd);
    std::cerr << "Completed transfer request" << std::endl;
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

FlexTransferEngine::BufferPair *FlexTransferEngine::getOrAllocateBufferPair(
    const std::string &location, size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    auto it = buffer_pool_.find(location);
    if (it != buffer_pool_.end() && it->second->size >= size) {
        return it->second;
    }

    // No suitable buffer pair found or need larger size, allocate a new one
    if (it != buffer_pool_.end()) {
        // Free old buffer pair
        BufferPair *old_pair = it->second;
        ::unregisterLocalMemory(engine_, old_pair->buffer0);
        if (old_pair->is_cuda) {
#ifdef USE_CUDA
            cudaFree(old_pair->buffer0);
#endif
        } else {
            delete[] static_cast<char *>(old_pair->buffer0);
        }
        delete old_pair;
    }

    BufferPair *new_pair = allocateBufferPair(location, size);
    if (new_pair) buffer_pool_[location] = new_pair;

    return new_pair;
}

FlexTransferEngine::BufferPair *FlexTransferEngine::allocateBufferPair(
    const std::string &location, size_t size) {
    FlexTransferEngine::BufferPair *pair = new FlexTransferEngine::BufferPair();
    pair->size = size;
    pair->is_cuda = (location.find("cuda:") == 0);
    pair->buffer0_in_use = false;
    pair->buffer1_in_use = false;

    // Allocate one contiguous buffer that's 2*size
    size_t total_size = 2 * size;
    if (pair->is_cuda) {
#ifdef USE_CUDA
        cudaError_t err = cudaMalloc(&pair->buffer0, total_size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate GPU memory: "
                      << cudaGetErrorString(err) << std::endl;
            delete pair;
            return nullptr;
        }
        // Split into two halves
        pair->buffer1 = static_cast<char *>(pair->buffer0) + size;
#else
        std::cerr << "GPU memory requested but CUDA support not compiled"
                  << std::endl;
        delete pair;
        return nullptr;
#endif
    } else {
        pair->buffer0 = new char[total_size];
        if (!pair->buffer0) {
            std::cerr << "Failed to allocate CPU memory" << std::endl;
            delete pair;
            return nullptr;
        }
        // Split into two halves
        pair->buffer1 = static_cast<char *>(pair->buffer0) + size;
    }

    // Register the entire contiguous buffer with RDMA
    int ret = ::registerLocalMemory(engine_, pair->buffer0, total_size,
                                    location.c_str(), 1);
    if (ret < 0) {
        std::cerr << "Failed to register buffer with RDMA" << std::endl;
        if (pair->is_cuda) {
#ifdef USE_CUDA
            cudaFree(pair->buffer0);
#endif
        } else {
            delete[] static_cast<char *>(pair->buffer0);
        }
        delete pair;
        return nullptr;
    }

    std::cerr << "Allocated buffer pair of size " << size << " for location "
              << location << " (total=" << total_size << ")" << std::endl;
    return pair;
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

bool FlexTransferEngine::isAddressRegistered(void *addr) {
    std::lock_guard<std::mutex> lock(regions_mutex_);

    for (const auto &[base_addr, region] : copiable_regions_) {
        char *start = static_cast<char *>(base_addr);
        char *end = start + region.length;
        if (addr >= start && addr < end) {
            return true;
        }
    }
    return false;
}

std::string FlexTransferEngine::getLocation(void *addr) {
    std::lock_guard<std::mutex> lock(regions_mutex_);

    for (const auto &[base_addr, region] : copiable_regions_) {
        char *start = static_cast<char *>(base_addr);
        char *end = start + region.length;
        if (addr >= start && addr < end) {
            return region.location;
        }
    }
    return "";
}

// Private methods from DirectTransferEngine

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
    int ret = ::registerLocalMemory(
        engine_, ctrl_block, sizeof(CopyCtrlBlock),
        ctrl_block_location_.empty() ? nullptr : ctrl_block_location_.c_str(),
        1);
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

std::string FlexTransferEngine::getCopyServerUrl() const {
    return local_copy_server_url_;
}

}  // namespace mooncake
