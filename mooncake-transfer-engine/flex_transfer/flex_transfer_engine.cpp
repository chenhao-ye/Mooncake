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

#include <cstring>
#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "common.h"

namespace mooncake {

FlexTransferEngine::~FlexTransferEngine() {
    // Stop listener if it was running
    stopListener();

    // Clean up buffer pool (only if enable_copy_)
    if (enable_copy_) {
        std::lock_guard<std::mutex> pool_lock(pool_mutex_);
        for (auto &[location, pair] : buffer_pool_) {
            if (pair) {
                ::unregisterLocalMemory(engine_, pair->base_buffer);
                if (pair->is_cuda) {
#ifdef USE_CUDA
                    cudaFree(pair->base_buffer);
#endif
                } else {
                    free(pair->base_buffer);
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

    // Unregister progress address if it was registered
    if (progress_registered_ && engine_ != nullptr) {
        unregisterLocalMemory(&progress_);
    }

    // Destroy the transfer engine
    if (engine_ != nullptr) {
        ::destroyTransferEngine(engine_);
        engine_ = nullptr;
    }
}

int FlexTransferEngine::init(const std::string &metadata_conn_string,
                              const std::string &local_server_name,
                              const std::string &ip_or_host_name,
                              uint64_t rpc_port, uint16_t tcp_port,
                              int auto_discover) {
    local_server_name_ = local_server_name;

    // Create the underlying TransferEngine
    engine_ = ::createTransferEngine(
        metadata_conn_string.c_str(), local_server_name.c_str(),
        ip_or_host_name.empty() ? nullptr : ip_or_host_name.c_str(), rpc_port,
        auto_discover);
    if (engine_ == nullptr) {
        std::cerr << "Failed to create TransferEngine" << std::endl;
        return -1;
    }

    // Register progress address for copy-based transfers
    int ret = ::registerLocalMemory(engine_, &progress_, sizeof(progress_), "",
                                     1);
    if (ret < 0) {
        std::cerr << "Failed to register progress memory" << std::endl;
        ::destroyTransferEngine(engine_);
        engine_ = nullptr;
        return ret;
    }
    progress_registered_ = true;

    // Start TCP listener only if enable_copy_ is true
    if (enable_copy_) {
        ret = startListener(ip_or_host_name, tcp_port);
        if (ret < 0) {
            std::cerr << "Failed to start TCP listener" << std::endl;
            if (progress_registered_) {
                ::unregisterLocalMemory(engine_, &progress_);
                progress_registered_ = false;
            }
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

segment_id_t FlexTransferEngine::openSegment(const std::string &segment_name) {
    return ::openSegment(engine_, segment_name.c_str());
}

int FlexTransferEngine::closeSegment(segment_id_t segment_id) {
    return ::closeSegment(engine_, segment_id);
}

int FlexTransferEngine::registerLocalMemory(void *addr, size_t length,
                                             const std::string &location,
                                             int remote_accessible) {
    // Always register with the underlying engine
    int ret = ::registerLocalMemory(
        engine_, addr, length,
        location.empty() ? nullptr : location.c_str(), remote_accessible);
    if (ret < 0) {
        return ret;
    }

    // Track registered regions only if enable_copy_ is true
    if (enable_copy_) {
        std::lock_guard<std::mutex> regions_lock(regions_mutex_);

        // Check if already registered
        if (registered_regions_.find(addr) != registered_regions_.end()) {
            std::cerr << "Warning: Memory at " << addr << " already registered"
                      << std::endl;
            return 0;
        }

        // Store the memory region info
        MemoryRegion region;
        region.addr = addr;
        region.length = length;
        region.location = location;
        registered_regions_[addr] = region;

        // Check if we need to allocate/resize buffer pair for this location
        std::lock_guard<std::mutex> pool_lock(pool_mutex_);

        // Find the largest memory region for this location
        size_t max_size = length;
        for (const auto &[_, reg] : registered_regions_) {
            if (reg.location == location && reg.length > max_size) {
                max_size = reg.length;
            }
        }

        // Check if we have a buffer pair for this location
        auto it = buffer_pool_.find(location);
        if (it == buffer_pool_.end() || it->second->size < max_size) {
            // Need to allocate or resize buffer pair
            if (it != buffer_pool_.end()) {
                // Free old buffer pair
                BufferPair *old_pair = it->second;
                ::unregisterLocalMemory(engine_, old_pair->base_buffer);
                if (old_pair->is_cuda) {
#ifdef USE_CUDA
                    cudaFree(old_pair->base_buffer);
#endif
                } else {
                    free(old_pair->base_buffer);
                }
                delete old_pair;
            }

            // Allocate a new buffer pair
            BufferPair *new_pair = allocateBufferPair(location, max_size);
            if (new_pair == nullptr) {
                std::cerr << "Failed to allocate buffer pair for location "
                          << location << std::endl;
                registered_regions_.erase(addr);
                return -1;
            }
            buffer_pool_[location] = new_pair;
            std::cerr << "Allocated buffer pair of size " << max_size
                      << " for location " << location << std::endl;
        }

        std::cerr << "Registered memory at " << addr << " size " << length
                  << " location " << location << std::endl;
    }

    return 0;
}

int FlexTransferEngine::unregisterLocalMemory(void *addr) {
    // Always unregister from the underlying engine
    int ret = ::unregisterLocalMemory(engine_, addr);

    // Remove from tracked regions only if enable_copy_ is true
    if (enable_copy_) {
        std::lock_guard<std::mutex> lock(regions_mutex_);

        auto it = registered_regions_.find(addr);
        if (it != registered_regions_.end()) {
            registered_regions_.erase(it);
            std::cerr << "Unregistered memory at " << addr << std::endl;
        } else {
            std::cerr << "Warning: Memory at " << addr << " not registered"
                      << std::endl;
        }
    }

    return ret;
}

int FlexTransferEngine::registerLocalMemoryBatch(
    const std::vector<buffer_entry_t> &buffer_list,
    const std::string &location) {
    // Always register with the underlying engine
    std::vector<buffer_entry_t> buf_list = buffer_list;
    int ret = ::registerLocalMemoryBatch(engine_, buf_list.data(),
                                         buf_list.size(), location.c_str());
    if (ret < 0) {
        return ret;
    }

    // Track registered regions only if enable_copy_ is true
    if (enable_copy_) {
        std::lock_guard<std::mutex> regions_lock(regions_mutex_);

        // Find the largest buffer in the batch
        size_t max_size = 0;
        for (const auto &entry : buffer_list) {
            if (entry.length > max_size) {
                max_size = entry.length;
            }

            // Store the memory region info
            MemoryRegion region;
            region.addr = entry.addr;
            region.length = entry.length;
            region.location = location;
            registered_regions_[entry.addr] = region;
        }

        // Check if we need to allocate/resize buffer pair for this location
        std::lock_guard<std::mutex> pool_lock(pool_mutex_);

        // Consider all registered regions for this location
        for (const auto &[_, reg] : registered_regions_) {
            if (reg.location == location && reg.length > max_size) {
                max_size = reg.length;
            }
        }

        // Check if we have a buffer pair for this location
        auto it = buffer_pool_.find(location);
        if (it == buffer_pool_.end() || it->second->size < max_size) {
            // Need to allocate or resize buffer pair
            if (it != buffer_pool_.end()) {
                // Free old buffer pair
                BufferPair *old_pair = it->second;
                ::unregisterLocalMemory(engine_, old_pair->base_buffer);
                if (old_pair->is_cuda) {
#ifdef USE_CUDA
                    cudaFree(old_pair->base_buffer);
#endif
                } else {
                    free(old_pair->base_buffer);
                }
                delete old_pair;
            }

            // Allocate a new buffer pair
            BufferPair *new_pair = allocateBufferPair(location, max_size);
            if (new_pair == nullptr) {
                std::cerr << "Failed to allocate buffer pair for location "
                          << location << std::endl;
                return -1;
            }
            buffer_pool_[location] = new_pair;
            std::cerr << "Allocated buffer pair of size " << max_size
                      << " for location " << location << std::endl;
        }

        std::cerr << "Registered " << buffer_list.size()
                  << " buffers for location " << location << ", max size "
                  << max_size << std::endl;
    }

    return 0;
}

int FlexTransferEngine::unregisterLocalMemoryBatch(
    const std::vector<void *> &addr_list) {
    // Always unregister from the underlying engine
    std::vector<void *> addrs = addr_list;
    int ret = ::unregisterLocalMemoryBatch(engine_, addrs.data(), addrs.size());

    // Remove from tracked regions only if enable_copy_ is true
    if (enable_copy_) {
        std::lock_guard<std::mutex> lock(regions_mutex_);

        for (void *addr : addr_list) {
            auto it = registered_regions_.find(addr);
            if (it != registered_regions_.end()) {
                registered_regions_.erase(it);
            }
        }

        std::cerr << "Unregistered " << addr_list.size() << " buffers"
                  << std::endl;
    }

    return ret;
}

int FlexTransferEngine::syncSegmentCache() {
    return ::syncSegmentCache(engine_);
}

batch_id_t FlexTransferEngine::allocateBatchID(size_t batch_size) {
    return ::allocateBatchID(engine_, batch_size);
}

int FlexTransferEngine::freeBatchID(batch_id_t batch_id) {
    return ::freeBatchID(engine_, batch_id);
}

int FlexTransferEngine::submitTransfer(
    batch_id_t batch_id, const std::vector<transfer_request_t> &entries,
    const std::string &copy_server_name, uint16_t copy_server_port) {
    if (copy_server_name.empty()) {
        // Direct RDMA transfer through underlying TransferEngine
        std::vector<transfer_request_t> reqs = entries;
        return ::submitTransfer(engine_, batch_id, reqs.data(), reqs.size());
    }

    // Copy-based transfer via TCP to remote FlexTransferEngine
    return submitTransferToCopyEngine(batch_id, entries, copy_server_name,
                                      copy_server_port);
}

int FlexTransferEngine::getTransferStatus(batch_id_t batch_id, size_t task_id,
                                           transfer_status_t &status) {
    return ::getTransferStatus(engine_, batch_id, task_id, &status);
}

// Private methods from CopyTransferEngine

int FlexTransferEngine::startListener(const std::string &ip_or_host_name,
                                       uint16_t tcp_port) {
    // Create socket
    listener_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listener_fd_ < 0) {
        std::cerr << "Failed to create socket: " << strerror(errno)
                  << std::endl;
        return ERR_SOCKET;
    }

    // Set SO_REUSEADDR
    int optval = 1;
    if (setsockopt(listener_fd_, SOL_SOCKET, SO_REUSEADDR, &optval,
                   sizeof(optval)) < 0) {
        std::cerr << "Failed to set SO_REUSEADDR: " << strerror(errno)
                  << std::endl;
        close(listener_fd_);
        return ERR_SOCKET;
    }

    // Bind to address
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(tcp_port);

    if (ip_or_host_name.empty()) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_pton(AF_INET, ip_or_host_name.c_str(), &addr.sin_addr) <= 0) {
            std::cerr << "Invalid IP address: " << ip_or_host_name << std::endl;
            close(listener_fd_);
            return ERR_INVALID_ARGUMENT;
        }
    }

    if (bind(listener_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        std::cerr << "Failed to bind to port " << tcp_port << ": "
                  << strerror(errno) << std::endl;
        close(listener_fd_);
        return ERR_SOCKET;
    }

    // Get the actual port if tcp_port was 0
    if (tcp_port == 0) {
        socklen_t addr_len = sizeof(addr);
        if (getsockname(listener_fd_, (struct sockaddr *)&addr, &addr_len) <
            0) {
            std::cerr << "Failed to get socket name: " << strerror(errno)
                      << std::endl;
            close(listener_fd_);
            return ERR_SOCKET;
        }
        tcp_port_ = ntohs(addr.sin_port);
    } else {
        tcp_port_ = tcp_port;
    }

    // Listen
    if (listen(listener_fd_, 128) < 0) {
        std::cerr << "Failed to listen on port " << tcp_port_ << ": "
                  << strerror(errno) << std::endl;
        close(listener_fd_);
        return ERR_SOCKET;
    }

    // Start worker thread
    worker_running_ = true;
    worker_thread_ = std::thread(&FlexTransferEngine::workerThread, this);

    std::cerr << "TCP listener started on port " << tcp_port_ << std::endl;
    return 0;
}

void FlexTransferEngine::stopListener() {
    if (worker_running_) {
        worker_running_ = false;

        // Close listener socket to unblock accept()
        if (listener_fd_ >= 0) {
            close(listener_fd_);
            listener_fd_ = -1;
        }

        // Wait for worker thread to finish
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }

        std::cerr << "TCP listener stopped" << std::endl;
    }
}

void FlexTransferEngine::workerThread() {
    std::cerr << "Worker thread started" << std::endl;

    while (worker_running_) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        int client_fd =
            accept(listener_fd_, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            if (worker_running_) {
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
    segment_id_t target_segment_id;
    {
        std::lock_guard<std::mutex> lock(segment_cache_mutex_);
        auto it = segment_cache_.find(segment_name);
        if (it != segment_cache_.end()) {
            target_segment_id = it->second;
        } else {
            target_segment_id = ::openSegment(engine_, segment_name.c_str());
            if (target_segment_id < 0) {
                std::cerr << "Failed to open segment: " << segment_name
                          << std::endl;
                close(client_fd);
                return;
            }
            segment_cache_[segment_name] = target_segment_id;
            std::cerr << "Opened and cached segment: " << segment_name
                      << " (id=" << target_segment_id << ")" << std::endl;
        }
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

    // Process requests directly (no task queue)
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
        if (buffer_pair == nullptr) {
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
            (i % 2 == 0) ? buffer_pair->buffer1 : buffer_pair->buffer2;

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

        int submit_ret = ::submitTransfer(engine_, write_batch_id, &write_req, 1);
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
        while (::getTransferStatus(engine_, write_batch_id, 0, &write_status) == 0 &&
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
        ::unregisterLocalMemory(engine_, old_pair->base_buffer);
        if (old_pair->is_cuda) {
#ifdef USE_CUDA
            cudaFree(old_pair->base_buffer);
#endif
        } else {
            free(old_pair->base_buffer);
        }
        delete old_pair;
    }

    BufferPair *new_pair = allocateBufferPair(location, size);
    if (new_pair != nullptr) {
        buffer_pool_[location] = new_pair;
    }
    return new_pair;
}

FlexTransferEngine::BufferPair *FlexTransferEngine::allocateBufferPair(
    const std::string &location, size_t size) {
    FlexTransferEngine::BufferPair *pair = new FlexTransferEngine::BufferPair();
    pair->size = size;
    pair->is_cuda = (location.find("cuda:") == 0);
    pair->buffer1_in_use = false;
    pair->buffer2_in_use = false;

    // Allocate one contiguous buffer that's 2*size
    size_t total_size = 2 * size;
    if (pair->is_cuda) {
#ifdef USE_CUDA
        cudaError_t err = cudaMalloc(&pair->base_buffer, total_size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate GPU memory: "
                      << cudaGetErrorString(err) << std::endl;
            delete pair;
            return nullptr;
        }
        // Split into two halves
        pair->buffer1 = pair->base_buffer;
        pair->buffer2 = static_cast<char *>(pair->base_buffer) + size;
#else
        std::cerr << "GPU memory requested but CUDA support not compiled"
                  << std::endl;
        delete pair;
        return nullptr;
#endif
    } else {
        pair->base_buffer = malloc(total_size);
        if (pair->base_buffer == nullptr) {
            std::cerr << "Failed to allocate CPU memory" << std::endl;
            delete pair;
            return nullptr;
        }
        // Split into two halves
        pair->buffer1 = pair->base_buffer;
        pair->buffer2 = static_cast<char *>(pair->base_buffer) + size;
    }

    // Register the entire contiguous buffer with RDMA
    int ret = ::registerLocalMemory(engine_, pair->base_buffer, total_size,
                                     location.c_str(), 1);
    if (ret < 0) {
        std::cerr << "Failed to register buffer with RDMA" << std::endl;
        if (pair->is_cuda) {
#ifdef USE_CUDA
            cudaFree(pair->base_buffer);
#endif
        } else {
            free(pair->base_buffer);
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

    for (const auto &[base_addr, region] : registered_regions_) {
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

    for (const auto &[base_addr, region] : registered_regions_) {
        char *start = static_cast<char *>(base_addr);
        char *end = start + region.length;
        if (addr >= start && addr < end) {
            return region.location;
        }
    }
    return "";
}

// Private methods from DirectTransferEngine

int FlexTransferEngine::connectToCopyEngine(const std::string &server_name,
                                             uint16_t port) {
    // Parse server name and port
    auto [hostname, parsed_port] = parseHostNameWithPort(server_name);
    // Use parsed port if it's different from the default
    if (parsed_port != 0) {
        port = parsed_port;
    }

    // Resolve hostname
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(port);
    int ret = getaddrinfo(hostname.c_str(), port_str.c_str(), &hints, &result);
    if (ret != 0) {
        std::cerr << "getaddrinfo failed: " << gai_strerror(ret) << std::endl;
        return -1;
    }

    // Try to connect
    int fd = -1;
    for (struct addrinfo *rp = result; rp != nullptr; rp = rp->ai_next) {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0) {
            continue;
        }

        if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
            break;  // Success
        }

        close(fd);
        fd = -1;
    }

    freeaddrinfo(result);

    if (fd < 0) {
        std::cerr << "Failed to connect to " << hostname << ":" << port
                  << std::endl;
        return -1;
    }

    std::cerr << "Connected to FlexTransferEngine at " << hostname << ":"
              << port << std::endl;
    return fd;
}

int FlexTransferEngine::submitTransferToCopyEngine(
    batch_id_t batch_id, const std::vector<transfer_request_t> &entries,
    const std::string &server_name, uint16_t port) {
    // Verify all entries are read requests
    for (const auto &entry : entries) {
        if (entry.opcode != OPCODE_READ) {
            std::cerr << "Only read requests are supported when target is "
                         "FlexTransferEngine with copy mode"
                      << std::endl;
            return -1;
        }
    }

    if (entries.empty()) {
        return 0;
    }

    // Connect to the remote FlexTransferEngine (or reuse existing connection)
    int fd = -1;
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = copy_engine_connections_.find(server_name);
        if (it != copy_engine_connections_.end()) {
            fd = it->second;
        } else {
            fd = connectToCopyEngine(server_name, port);
            if (fd < 0) {
                std::cerr << "Failed to connect to FlexTransferEngine at "
                          << server_name << std::endl;
                return -1;
            }
            copy_engine_connections_[server_name] = fd;
        }
    }

    // Reset the pre-registered progress counter
    progress_.store(0);

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
    batch_info.progress_addr = reinterpret_cast<uint64_t>(&progress_);
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

    // The remote FlexTransferEngine will now process the requests asynchronously
    // and update the progress counter via RDMA writes.
    // The caller should poll the progress or use getTransferStatus to check
    // completion.

    std::cerr << "Submitted " << entries.size()
              << " requests to FlexTransferEngine at " << server_name
              << std::endl;

    return 0;
}

}  // namespace mooncake
