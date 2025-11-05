#include "copy_server.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "flex_transfer_engine.h"
#include "util.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

void CopyServer::cleanup() {
    stopListener();

    std::lock_guard<std::mutex> lock(regions_mutex_);
    for (auto pair : buffer_pool_) freeBufferPair(pair);
    buffer_pool_.clear();
}

int CopyServer::registerLocalMemory(void *addr, size_t length,
                                    const std::string &location) {
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

int CopyServer::unregisterLocalMemory(void *addr) {
    std::lock_guard<std::mutex> regions_lock(regions_mutex_);
    size_t num_erased = copiable_regions_.erase(addr);
    return num_erased > 0 ? 0 : -1;
}

int CopyServer::registerLocalMemoryBatch(
    std::vector<buffer_entry_t> &buffer_list, const std::string &location) {
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

int CopyServer::unregisterLocalMemoryBatch(std::vector<void *> &addr_list) {
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

/* TCP listener and worker thread functions */

void CopyServer::startListener() {
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
    worker_thread_ = std::thread(&CopyServer::workerThread, this);
}

void CopyServer::stopListener() {
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

void CopyServer::workerThread() {
    std::cerr << "Worker thread started" << std::endl;

    // Set listener to non-blocking
    int flags = fcntl(listener_fd_, F_GETFL, 0);
    fcntl(listener_fd_, F_SETFL, flags | O_NONBLOCK);

    while (worker_running_.load(std::memory_order_acquire)) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(listener_fd_, &read_fds);

        int max_fd = listener_fd_;

        // Add all active client connections to the set
        for (int client_fd : active_client_fds_) {
            FD_SET(client_fd, &read_fds);
            if (client_fd > max_fd) max_fd = client_fd;
        }

        // Use select with timeout to allow checking worker_running_
        struct timeval timeout;
        timeout.tv_sec = 1;  // 1s
        timeout.tv_usec = 0;

        int activity =
            select(max_fd + 1, &read_fds, nullptr, nullptr, &timeout);
        if (activity < 0) {
            std::cerr << "select error: " << strerror(errno) << std::endl;
            continue;
        }

        if (activity == 0) continue;  // timeout

        // Check for new connections on listener
        if (FD_ISSET(listener_fd_, &read_fds)) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);

            int client_fd = accept(
                listener_fd_, (struct sockaddr *)&client_addr, &client_len);
            if (client_fd >= 0) {
                char client_ip[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, &client_addr.sin_addr, client_ip,
                          sizeof(client_ip));
                std::cerr << "Accepted connection from " << client_ip << ":"
                          << ntohs(client_addr.sin_port) << std::endl;

                active_client_fds_.push_back(client_fd);
            }
        }

        // Check for data on existing connections (process one at a time)
        for (int client_fd : active_client_fds_) {
            if (FD_ISSET(client_fd, &read_fds)) {
                handleAndProcessRequest(client_fd);
                break;  // Process only one request per iteration
            }
        }
    }

    // Clean up all active connections
    for (int client_fd : active_client_fds_) close(client_fd);
    active_client_fds_.clear();

    std::cerr << "Worker thread stopped" << std::endl;
}

void CopyServer::handleAndProcessRequest(int client_fd) {
    int32_t num_completed = 0;
    std::vector<Task> tasks;
    CopyCtrlBlock *copy_ctrl_block = nullptr;

    // local variables as network buffers
    uint32_t segment_name_len;
    std::string segment_name;

    struct BatchInfo {
        uint64_t progress_addr;
        uint64_t num_requests;
    };

    BatchInfo batch_info;
    segment_id_t target_segment_id;

    // read segment name length
    if (readFully(client_fd, &segment_name_len, sizeof(segment_name_len)) !=
        sizeof(segment_name_len)) {
        std::cerr << "Failed to read segment name length" << std::endl;
        goto cleanup;
    }

    const static size_t kMaxLength = 1ull << 20;
    if (segment_name_len == 0 || segment_name_len > kMaxLength) {
        std::cerr << "Invalid segment name length: " << segment_name_len
                  << std::endl;
        goto cleanup;
    }

    // read segment name
    segment_name.resize(segment_name_len + 1);
    if (readFully(client_fd, segment_name.data(), segment_name_len) !=
        segment_name_len) {
        std::cerr << "Failed to read segment name" << std::endl;
        goto cleanup;
    }

    std::cerr << "Received request for segment: " << segment_name << std::endl;

    // read batch info
    if (readFully(client_fd, &batch_info, sizeof(batch_info)) !=
        sizeof(batch_info)) {
        std::cerr << "Failed to read batch info" << std::endl;
        goto cleanup;
    }

    std::cerr << "Received transfer request: progress_addr=0x" << std::hex
              << batch_info.progress_addr << std::dec
              << ", num_requests=" << batch_info.num_requests << std::endl;

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
            goto cleanup;
        }
        tasks.emplace_back(reinterpret_cast<void *>(req_info.source_addr),
                           req_info.target_addr, req_info.length);
    }

    target_segment_id = engine_.getSegmentId(segment_name);
    if (target_segment_id < 0) {
        std::cerr << "Failed to open segment: " << segment_name << std::endl;
        goto cleanup;
    }

    copy_ctrl_block = engine_.acquireCopyCtrlBlock();
    if (!copy_ctrl_block) {
        std::cerr << "Failed to acquire CopyCtrlBlock" << std::endl;
        goto cleanup;
    }

    {
        std::lock_guard<std::mutex> regions_lock(regions_mutex_);

        for (size_t i = 0; i < tasks.size(); ++i) {
            Task &task = tasks[i];

            // delayed source address validation:
            // if source_addr is invalid, will be detected here
            MemoryRegion *region = getRegion(task.source_addr, task.length);
            if (!region) {
                std::cerr << "Source address " << task.source_addr
                          << " not in registered copiable regions" << std::endl;
                goto cleanup;
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
                    goto cleanup;
                }
            }

            void *buffer = buffer_pair.buffers[buffer_idx];

            // Copy data from source to buffer
            int rc = copyMemory(buffer, task.source_addr, task.length,
                                buffer_pair.is_cuda);
            if (rc) {
                std::cerr << "Failed to copy memory from " << task.source_addr
                          << " to buffer " << buffer << std::endl;
                goto cleanup;
            }

            // Submit RDMA write from buffer to remote target
            batch_id_t batch_id = ::allocateBatchID(engine_.getEngine(), 1);
            transfer_request_t write_req = {
                .opcode = OPCODE_WRITE,
                .source = buffer,
                .target_id = target_segment_id,
                .target_offset = task.target_addr,
                .length = task.length,
            };

            int submit_rc =
                ::submitTransfer(engine_.getEngine(), batch_id, &write_req, 1);
            if (submit_rc < 0) {
                std::cerr << "Failed to submit RDMA write" << std::endl;
                ::freeBatchID(engine_.getEngine(), batch_id);
                goto cleanup;
            }

            task.batch_id = batch_id;
            task.buffer_pair = &buffer_pair;
            task.buffer_idx = buffer_idx;
        }

        for (auto &task : tasks) {
            int rc = waitTask(task);
            if (rc) {
                std::cerr << "Failed to wait for task completion" << std::endl;
                goto cleanup;
            }
            num_completed++;
        }
    }
    /**
     Ordering guarantee: progress counter update must be finished before
     return any value from the socket. In other words, once received a
     int32_t from the socket, the client can safely assume there will be
     no more update to the progress counter.
     */

    {  // Update progress to indicate completion via RDMA
        copy_ctrl_block->progress_counter.store(num_completed,
                                                std::memory_order_release);
        batch_id_t batch_id = ::allocateBatchID(engine_.getEngine(), 1);
        transfer_request_t progress_req = {
            .opcode = OPCODE_WRITE,
            .source = (void *)&(copy_ctrl_block->progress_counter),
            .target_id = target_segment_id,
            .target_offset =
                reinterpret_cast<uint64_t>(batch_info.progress_addr),
            .length = sizeof(int64_t),
        };
        ::submitTransfer(engine_.getEngine(), batch_id, &progress_req, 1);

        // Wait for completion
        transfer_status_t status;
        while (::getTransferStatus(engine_.getEngine(), batch_id, 0, &status) ==
                   0 &&
               status.status != STATUS_COMPLETED) {
            usleep(1);
        }
        ::freeBatchID(engine_.getEngine(), batch_id);
    }

    std::cerr << "Completed transfer request: " << num_completed << " tasks"
              << std::endl;

cleanup:
    // Clean up any pending tasks
    for (auto &task : tasks) {
        int rc = waitTask(task);
        if (rc) std::cerr << "Failed to wait for task completion" << std::endl;
    }

    if (copy_ctrl_block) engine_.releaseCopyCtrlBlock(copy_ctrl_block);
    // Send completion count via socket (0 or negative on error)
    if (writeFully(client_fd, &num_completed, sizeof(num_completed)) !=
        sizeof(num_completed)) {
        std::cerr << "Failed to send completion count" << std::endl;
    }
}

/* Helper methods */

CopyServer::LocIdx CopyServer::getLocIdx(const std::string &location) {
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

CopyServer::MemoryRegion *CopyServer::getRegion(void *addr, size_t length) {
    // fast path: the addr is the base of a registered region
    auto it = copiable_regions_.find(addr);
    if (it != copiable_regions_.end() && length <= it->second.length)
        return &it->second;
    // slow path: scan to find addr within a registered region
    for (auto &[base_addr, region] : copiable_regions_) {
        if (addr < static_cast<char *>(base_addr)) continue;
        if (static_cast<char *>(addr) + length >
            static_cast<char *>(base_addr) + region.length)
            continue;
        return &region;
    }
    return nullptr;
}

CopyServer::BufferPair *CopyServer::allocBufferPair(LocIdx loc_idx,
                                                    const std::string &location,
                                                    size_t size) {
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
        }
#else
        throw std::runtime_error(
            "GPU memory requested but CUDA support not compiled");
#endif
    } else {
        buffer_base = new char[total_size];
    }

    // Register the entire contiguous buffer with RDMA
    int rc = ::registerLocalMemory(engine_.getEngine(), buffer_base, total_size,
                                   location.c_str(),
                                   /*remote_accessible*/ true);
    if (rc) {
        if (is_cuda) {
#ifdef USE_CUDA
            cudaFree(buffer_base);
#endif
        } else {
            delete[] buffer_base;
        }
        return nullptr;
    }

    BufferPair *pair = new BufferPair(buffer_base, size, is_cuda);

    std::cerr << "Allocated buffer pair of size " << size << " for location "
              << location << " (total=" << total_size << ")" << std::endl;
    return pair;
}

void CopyServer::freeBufferPair(BufferPair *pair) {
    if (!pair) return;

    ::unregisterLocalMemory(engine_.getEngine(), pair->buffers[0]);

    if (pair->is_cuda) {
#ifdef USE_CUDA
        cudaFree(pair->buffers[0]);
#endif
    } else {
        delete[] pair->buffers[0];
    }

    delete pair;
}

int CopyServer::waitTask(Task &task) {
    if (task.batch_id == INVALID_BATCH) return 0;

    [[maybe_unused]] int rc;
    transfer_status_t status;

    while (true) {
        rc =
            ::getTransferStatus(engine_.getEngine(), task.batch_id, 0, &status);
        assert(rc == 0);
        if (status.status != STATUS_WAITING) {  // completed or error
            task.buffer_pair->users[task.buffer_idx] = -1;  // mark buffer free
            ::freeBatchID(engine_.getEngine(), task.batch_id);
            task.batch_id = INVALID_BATCH;
            task.buffer_pair = nullptr;
            task.buffer_idx = -1;
            return status.status == STATUS_COMPLETED ? 0 : -1;
        }
    }
}

int CopyServer::copyMemory(void *dst, const void *src, size_t size,
                           bool is_cuda) {
    if (is_cuda) {
#ifdef USE_CUDA
        cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err)
                      << std::endl;
            return -1;
        }
#else
        throw std::runtime_error(
            "GPU memory copy requested but CUDA support not compiled");
#endif
    } else {
        memcpy(dst, src, size);
    }
    return 0;
}
