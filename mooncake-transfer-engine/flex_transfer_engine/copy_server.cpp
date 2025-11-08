#include "copy_server.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "copy_transfer.h"
#include "flex_transfer_engine.h"
#include "transfer_engine_c.h"
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

int CopyServer::unregisterLocalMemoryBatch(std::vector<uintptr_t> &addr_list) {
    int rc = 0;
    std::lock_guard<std::mutex> regions_lock(regions_mutex_);
    for (uintptr_t addr : addr_list) {
        auto it = copiable_regions_.find(reinterpret_cast<void *>(addr));
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

    // Create eventfd for stopping the worker thread
    stop_event_fd_ = eventfd(0, EFD_NONBLOCK);
    if (stop_event_fd_ < 0) {
        throw std::runtime_error("Failed to create eventfd: " +
                                 std::string(strerror(errno)));
    }

    // Create epoll instance
    epoll_fd_ = epoll_create1(0);
    if (epoll_fd_ < 0) {
        throw std::runtime_error("Failed to create epoll instance: " +
                                 std::string(strerror(errno)));
    }

    // Add listener_fd to epoll
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = listener_fd_;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listener_fd_, &ev) < 0) {
        throw std::runtime_error("Failed to add listener to epoll: " +
                                 std::string(strerror(errno)));
    }

    // Add stop_event_fd to epoll
    ev.events = EPOLLIN;
    ev.data.fd = stop_event_fd_;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, stop_event_fd_, &ev) < 0) {
        throw std::runtime_error("Failed to add stop_event_fd to epoll: " +
                                 std::string(strerror(errno)));
    }

    // finally, start worker thread
    worker_running_ = true;
    worker_thread_ = std::thread(&CopyServer::workerThread, this);
}

void CopyServer::stopListener() {
    if (worker_running_.load(std::memory_order_acquire)) {
        worker_running_.store(false, std::memory_order_release);

        // Signal the eventfd to wake up the worker thread
        if (stop_event_fd_ >= 0) {
            [[maybe_unused]] ssize_t nbytes;
            uint64_t event_value = 1;
            nbytes = write(stop_event_fd_, &event_value, sizeof(event_value));
            assert(nbytes == sizeof(event_value));
        }

        if (worker_thread_.joinable()) worker_thread_.join();

        if (epoll_fd_ >= 0) {
            close(epoll_fd_);
            epoll_fd_ = -1;
        }

        if (stop_event_fd_ >= 0) {
            close(stop_event_fd_);
            stop_event_fd_ = -1;
        }

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

    const int MAX_EVENTS = 64;
    struct epoll_event events[MAX_EVENTS];

    while (worker_running_.load(std::memory_order_acquire)) {
        // Wait for events
        int nfds = epoll_wait(epoll_fd_, events, MAX_EVENTS, -1);
        if (nfds < 0) {
            if (errno == EINTR) continue;
            std::cerr << "epoll_wait error: " << strerror(errno) << std::endl;
            continue;
        }

        // Process all ready events
        for (int i = 0; i < nfds; ++i) {
            int ready_fd = events[i].data.fd;

            // Check if we were signaled to stop
            if (ready_fd == stop_event_fd_) {
                [[maybe_unused]] ssize_t nbytes;
                uint64_t event_value;
                nbytes =
                    read(stop_event_fd_, &event_value, sizeof(event_value));
                assert(nbytes == sizeof(event_value));
                goto cleanup;  // Exit the worker thread loop
            }

            // Check for new connections on listener
            if (ready_fd == listener_fd_) {
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

                    // Add new client to epoll
                    struct epoll_event ev;
                    ev.events = EPOLLIN;
                    ev.data.fd = client_fd;
                    int rc =
                        epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, client_fd, &ev);
                    if (rc < 0) {
                        std::cerr << "Failed to add client to epoll: "
                                  << strerror(errno) << std::endl;
                        close(client_fd);
                    } else {
                        active_client_fds_.insert(client_fd);
                    }
                }
                continue;
            }

            // Handle client data
            int rc = processRDMARequest(ready_fd);
            if (rc != 0) {
                // Error occurred or client disconnected, close and remove
                std::cerr << "Closing client connection fd=" << ready_fd
                          << std::endl;

                epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, ready_fd, nullptr);
                close(ready_fd);
                active_client_fds_.erase(ready_fd);
            }
        }
    }

cleanup:

    // Clean up all active connections
    for (int client_fd : active_client_fds_) close(client_fd);
    active_client_fds_.clear();

    std::cerr << "Worker thread stopped" << std::endl;
}

int CopyServer::processRDMARequest(int client_fd) {
    bool success = false;

    // num_done is a lower bound watermark: if task_idx < num_done, it is done
    // and its batch_id is INVALID_BATCH; otherwise, it may or may not be done
    // (it is possible done because another task waits on it for the buffer)
    int32_t num_done = 0;
    int32_t last_updated_num_done = 0;
    // submit a progress update if
    //      (num_done - last_updated_num_done) >= update_freq
    constexpr int32_t update_freq = 5;  // for now: every 5 requests

    std::vector<Task> tasks;
    RDMACopyCtrlBlock *ctrl_block = nullptr;

    std::string segment_name;
    uint64_t target_progress_addr = 0;
    segment_id_t target_segment_id;

    batch_id_t prorgess_batch_id = INVALID_BATCH;

    int rc, status;

    rc = readSegmentName(client_fd, segment_name);
    if (rc) goto cleanup;

    rc = readTasks(client_fd, target_progress_addr, tasks);
    if (rc) goto cleanup;

    target_segment_id = engine_.getSegmentId(segment_name);
    if (target_segment_id < 0) {
        std::cerr << "Failed to open segment: " << segment_name << std::endl;
        goto cleanup;
    }

    ctrl_block = engine_.allocRDMACopyCtrlBlock();
    if (!ctrl_block) {
        std::cerr << "Failed to acquire RDMACopyCtrlBlock" << std::endl;
        goto cleanup;
    }

    {
        std::lock_guard<std::mutex> regions_lock(regions_mutex_);

        for (size_t task_idx = 0; task_idx < tasks.size(); ++task_idx) {
            rc = executeTask(tasks, task_idx, target_segment_id);
            if (rc) goto cleanup;
            // check if any waiting tasks are done
            for (size_t i = num_done; i <= task_idx; ++i) {
                if (tasks[i].batch_id != INVALID_BATCH) {
                    status = pollBatch(tasks[i].batch_id);
                    if (status == STATUS_WAITING) break;
                    freeBatch(tasks[i].batch_id);
                    if (status != STATUS_COMPLETED) {
                        std::cerr << "Transfer error for task_idx=" << task_idx
                                  << std::endl;
                        goto cleanup;
                    }
                }
                num_done++;
                if ((num_done - last_updated_num_done) >= update_freq) {
                    rc = tryUpdateRemoteProgress(
                        prorgess_batch_id, last_updated_num_done, num_done,
                        ctrl_block, target_segment_id, target_progress_addr);
                    if (rc) {
                        std::cerr << "Fail to update the progress" << std::endl;
                        goto cleanup;
                    }
                }
            }
        }

        // wait for all tasks to complete (should be very few)
        // if all requests are on the same device, should be only 1~2 tasks
        for (size_t task_idx = num_done; task_idx < tasks.size(); ++task_idx) {
            status = waitTask(tasks[task_idx]);
            if (status != STATUS_COMPLETED) {
                std::cerr << "Transfer error for task_idx=" << task_idx
                          << std::endl;
                goto cleanup;
            }
            num_done++;
        }
    }

    /**
     Ordering guarantee: progress counter update must be finished before return
     any value from the socket. In other words, once received a int32_t from the
     socket, the client can safely assume there will be no more update to the
     progress counter.
     */
    rc = tryUpdateRemoteProgress(prorgess_batch_id, last_updated_num_done,
                                 num_done, ctrl_block, target_segment_id,
                                 target_progress_addr);
    if (rc) {
        std::cerr << "Fail to update the progress" << std::endl;
        goto cleanup;
    }
    if (prorgess_batch_id != INVALID_BATCH) {
        status = waitBatch(prorgess_batch_id);
        if (status != STATUS_COMPLETED) {
            std::cerr << "Error wait for progress update completion"
                      << std::endl;
            goto cleanup;
        }
    }

    success = true;

    std::cerr << "Completed transfer request: " << num_done << " tasks"
              << std::endl;

cleanup:
    // Clean up waiting tasks if any (only occur if not success)
    for (size_t task_idx = num_done; task_idx < tasks.size(); ++task_idx) {
        status = waitTask(tasks[task_idx]);
        if (status != STATUS_COMPLETED)
            std::cerr << "Failed to wait for task completion" << std::endl;
    }

    if (ctrl_block) engine_.freeRDMACopyCtrlBlock(ctrl_block);

    // Send completion count via socket (i.e., num_done)
    // If this fails, the connection should be closed
    if (writeFully(client_fd, &num_done, sizeof(num_done)) !=
        sizeof(num_done)) {
        std::cerr << "Failed to send completion count, closing connection"
                  << std::endl;
        return -1;
    }

    // Return 0 on success, or -1 if any error occurred during processing
    // (Errors during processing would have set num_done appropriately)
    return success ? 0 : -1;
}

// read segment name from fd and write into segment_name
int CopyServer::readSegmentName(int client_fd, std::string &segment_name) {
    uint32_t segment_name_len;
    // read segment name length
    if (readFully(client_fd, &segment_name_len, sizeof(segment_name_len)) !=
        sizeof(segment_name_len)) {
        std::cerr << "Failed to read segment name length" << std::endl;
        return -1;
    }

    const static size_t kMaxLength = 1ull << 20;
    if (segment_name_len == 0 || segment_name_len > kMaxLength) {
        std::cerr << "Invalid segment name length: " << segment_name_len
                  << std::endl;
        return -1;
    }

    // read segment name
    segment_name.resize(segment_name_len + 1);
    if (readFully(client_fd, segment_name.data(), segment_name_len) !=
        segment_name_len) {
        std::cerr << "Failed to read segment name" << std::endl;
        return -1;
    }

    std::cerr << "Received request for segment: " << segment_name << std::endl;
    return 0;
}

// read requests from fd and write into tasks
int CopyServer::readTasks(int client_fd, uint64_t &target_progress_addr,
                          std::vector<Task> &tasks) {
    struct BatchInfo {
        uint64_t progress_addr;
        uint64_t num_requests;
    };

    BatchInfo batch_info;

    // read batch info
    if (readFully(client_fd, &batch_info, sizeof(batch_info)) !=
        sizeof(batch_info)) {
        std::cerr << "Failed to read batch info" << std::endl;
        return -1;
    }

    target_progress_addr = batch_info.progress_addr;

    std::cerr << "Received transfer request: progress_addr=0x" << std::hex
              << target_progress_addr << std::dec
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
            return -1;
        }
        tasks.emplace_back(reinterpret_cast<void *>(req_info.source_addr),
                           req_info.target_addr, req_info.length);
    }
    return 0;
}

int CopyServer::executeTask(std::vector<Task> tasks, size_t task_idx,
                            int target_segment_id) {
    int rc, status;
    Task &task = tasks[task_idx];
    // delayed source address validation:
    // if source_addr is invalid, will be detected here
    MemoryRegion *region = getRegion(task.source_addr, task.length);
    if (!region) {
        std::cerr << "Source address " << task.source_addr
                  << " not in registered copiable regions" << std::endl;
        return -1;
    }

    // Get a buffer pair for this location
    BufferPair &buffer_pair = getBufferPair(region->loc_idx);
    assert(buffer_pair.size >= task.length);

    int buffer_idx = buffer_pair.selectNextBuffer();
    int buffer_used_by_task_idx = buffer_pair.users[buffer_idx];
    if (buffer_used_by_task_idx >= 0) {  // wait for a previous task to complete
        status = waitTask(tasks[buffer_used_by_task_idx]);
        if (status != STATUS_COMPLETED) {
            std::cerr << "Failed to wait for previous task on buffer "
                      << buffer_idx << std::endl;
            return -1;
        }
    }

    void *buffer = buffer_pair.buffers[buffer_idx];

    // Copy data from source to buffer
    copyMemory(buffer, task.source_addr, task.length, buffer_pair.is_cuda);

    // Submit RDMA write from buffer to remote target
    transfer_request_t req = {
        .opcode = OPCODE_WRITE,
        .source = buffer,
        .target_id = target_segment_id,
        .target_offset = task.target_addr,
        .length = task.length,
    };

    batch_id_t batch_id;
    rc = submitBatch(batch_id, req);
    if (rc) {
        std::cerr << "Failed to submit RDMA write" << std::endl;
        return rc;
    }

    task.batch_id = batch_id;
    task.buffer_pair = &buffer_pair;
    task.buffer_idx = buffer_idx;
    return 0;
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
    int rc = ::registerLocalMemory(
        engine_.getEngine(), buffer_base, total_size, location.c_str(),
        /*remote_accessible*/ true, /*remote_atomic*/ false);
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

    int status = waitBatch(task.batch_id);
    task.buffer_pair->users[task.buffer_idx] = -1;  // mark buffer free
    task.buffer_pair = nullptr;
    task.buffer_idx = -1;
    return status;
}

// Poll if the given prorgess_batch_id has finished; if so, submit another
// progress update via atomic fetch-add, which will update prorgess_batch_id
// and last_updated_num_done
int CopyServer::tryUpdateRemoteProgress(batch_id_t &prorgess_batch_id,
                                        int32_t &last_updated_num_done,
                                        int32_t num_done,
                                        RDMACopyCtrlBlock *ctrl_block,
                                        segment_id_t target_segment_id,
                                        uint64_t target_progress_addr) {
    int status, rc;
    if (prorgess_batch_id != INVALID_BATCH) {  // check the last progress update
        status = pollBatch(prorgess_batch_id);
        if (status == STATUS_WAITING) return 0;  // not done
        // done: completed or error
        freeBatch(prorgess_batch_id);
        if (status != STATUS_COMPLETED) return -1;
    }

    // no new update
    if (num_done == last_updated_num_done) return 0;

    // submit another batch for progress update
    transfer_request_t progress_req = {
        .opcode = OPCODE_ATOMIC_FETCH_ADD,
        .source = (void *)&(ctrl_block->progress_counter),
        .target_id = target_segment_id,
        .target_offset = target_progress_addr,
        // for atomic fetch-add, .length is overloaded as the operand value
        .length = static_cast<uint64_t>(num_done - last_updated_num_done),
    };

    rc = submitBatch(prorgess_batch_id, progress_req);
    if (rc) return rc;

    last_updated_num_done = num_done;
    return 0;
}

// copyMemory is expected to succeed because the given src and dst must have
// been validated; if an error occurs, it is our own fault, not due to invalid
// input; throw the error instead of gracefully handling
void CopyServer::copyMemory(void *dst, const void *src, size_t size,
                            bool is_cuda) {
    if (is_cuda) {
#ifdef USE_CUDA
        cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDefault);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy failed: ") +
                                     cudaGetErrorString(err))
        }
#else
        throw std::runtime_error(
            "GPU memory copy requested but CUDA support not compiled");
#endif
    } else {
        memcpy(dst, src, size);
    }
}

int CopyServer::submitBatch(batch_id_t &batch_id, transfer_request_t &req) {
    batch_id = ::allocateBatchID(engine_.getEngine(), 1);
    int rc = ::submitTransfer(engine_.getEngine(), batch_id, &req, 1);
    if (rc) freeBatch(batch_id);  // failed; reset
    return rc;
}

void CopyServer::freeBatch(batch_id_t &batch_id) {
    assert(batch_id != INVALID_BATCH);
    ::freeBatchID(engine_.getEngine(), batch_id);
    batch_id = INVALID_BATCH;
}

int CopyServer::pollBatch(batch_id_t batch_id) {
    assert(batch_id != INVALID_BATCH);
    [[maybe_unused]] int rc;
    transfer_status_t status;
    rc = ::getTransferStatus(engine_.getEngine(), batch_id, 0, &status);
    assert(rc == 0);
    return status.status;
}

// Wait until the given batch (size=1) is done and then free the batch; will
// update batch_id to INVALID_BATCH; return the status
int CopyServer::waitBatch(batch_id_t &batch_id) {
    assert(batch_id != INVALID_BATCH);

    int status;
    do {
        status = pollBatch(batch_id);
    } while (status == STATUS_WAITING);

    freeBatch(batch_id);
    return status;
}