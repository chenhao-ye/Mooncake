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

#include "copy_common.h"
#include "flex_transfer_engine.h"
#include "region.h"
#include "transfer_engine_c.h"
#include "util.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

void CopyServer::cleanup() {
    stopListener();

    std::lock_guard<std::mutex> lock(regions_mutex_);
    rdma_copy_backend_.cleanup();
}

int CopyServer::registerLocalMemory(void *addr, size_t length,
                                    const std::string &location) {
    std::lock_guard<std::mutex> regions_lock(regions_mutex_);

    LocIdx loc_idx = region_mgr_.getLocIdx(location);
    region_mgr_.addRegion(addr, length, loc_idx);
    int rc = rdma_copy_backend_.prepareBufferPair(loc_idx, location, length);
    if (rc) goto err;

    std::cerr << "Registered memory at " << addr << " size " << length
              << " location " << location << std::endl;

    return 0;

err:
    region_mgr_.removeRegion(addr);
    return -1;
}

int CopyServer::unregisterLocalMemory(void *addr) {
    std::lock_guard<std::mutex> regions_lock(regions_mutex_);
    return region_mgr_.removeRegion(addr);
}

int CopyServer::registerLocalMemoryBatch(
    std::vector<buffer_entry_t> &buffer_list, const std::string &location) {
    std::lock_guard<std::mutex> regions_lock(regions_mutex_);
    LocIdx loc_idx = region_mgr_.getLocIdx(location);
    size_t max_size = 0;

    for (const auto &entry : buffer_list) {
        if (entry.length > max_size) max_size = entry.length;
        region_mgr_.addRegion(entry.addr, entry.length, loc_idx);
    }
    std::cerr << "Registered " << buffer_list.size() << " buffers for location "
              << location << ", max size " << max_size << std::endl;

    int rc = rdma_copy_backend_.prepareBufferPair(loc_idx, location, max_size);
    if (rc) goto err;

    return 0;

err:
    for (const auto &entry : buffer_list) region_mgr_.removeRegion(entry.addr);
    return -1;
}

int CopyServer::unregisterLocalMemoryBatch(std::vector<uintptr_t> &addr_list) {
    bool all_success = true;
    std::lock_guard<std::mutex> regions_lock(regions_mutex_);
    for (uintptr_t addr : addr_list) {
        int rc = region_mgr_.removeRegion(reinterpret_cast<void *>(addr));
        if (rc) all_success = false;  // not found, but will continue
    }
    return all_success ? 0 : -1;
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
    int rc;
    std::string segment_name;
    uint64_t target_progress_addr = 0;
    std::vector<RDMACopyBackend::Task> tasks;
    segment_id_t target_segment_id;
    RDMACopyCtrlBlock *ctrl_block = nullptr;
    int32_t num_done = 0;
    bool success = false;

    rc = readSegmentName(client_fd, segment_name);
    if (rc) goto cleanup;

    rc = readRDMARequests(client_fd, target_progress_addr, tasks);
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
        rc = rdma_copy_backend_.processRequest(target_segment_id,
                                               target_progress_addr, tasks,
                                               ctrl_block, num_done);
        if (rc == 0) success = true;
    }

cleanup:
    if (ctrl_block) engine_.freeRDMACopyCtrlBlock(ctrl_block);

    // Send completion count via socket (i.e., num_done)
    // If this fails, the connection should be closed
    if (writeFully(client_fd, &num_done, sizeof(num_done)) !=
        sizeof(num_done)) {
        std::cerr << "Failed to send completion count, closing connection"
                  << std::endl;
        success = false;
    }

    // Return 0 if all tasks completed successfully, -1 otherwise
    return success ? 0 : -1;
}

// read segment name from fd into segment_name
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

// read RDMA requests from fd into tasks
int CopyServer::readRDMARequests(int client_fd, uint64_t &target_progress_addr,
                                 std::vector<RDMACopyBackend::Task> &tasks) {
    struct Header {
        uint64_t progress_addr;
        uint64_t num_reqs;
    };

    ssize_t nbytes;
    Header header;

    nbytes = readFully(client_fd, &header, sizeof(header));
    if (nbytes != sizeof(header)) {
        std::cerr << "Failed to read RDMA request header" << std::endl;
        return -1;
    }

    target_progress_addr = header.progress_addr;
    tasks.reserve(header.num_reqs);

    std::cerr << "Received transfer request: progress_addr=0x" << std::hex
              << target_progress_addr << std::dec
              << ", num_reqs=" << header.num_reqs << std::endl;

    struct Req {
        uint64_t source_addr;
        uint64_t target_addr;
        uint64_t length;
    };
    std::vector<Req> reqs(header.num_reqs);
    size_t reqs_nbytes = sizeof(Req) * header.num_reqs;

    nbytes = readFully(client_fd, reqs.data(), reqs_nbytes);
    if (nbytes != static_cast<ssize_t>(reqs_nbytes)) {
        std::cerr << "Failed to read requests" << std::endl;
        return -1;
    }

    // Here we have a request-task conversion:
    // For request (from client), source_addr is a client address; target_addr
    // is a server address. For task (on server), source_addr is a server
    // address (where to copy data from), and target_addr is a client address
    // (where to write data to). There source_addr and target_addr should be
    // swapped when converting from request to task.
    for (const auto &req : reqs) {
        tasks.emplace_back(reinterpret_cast<void *>(req.target_addr),
                           req.source_addr, req.length);
    }
    return 0;
}

// read TCP requests from fd into tasks
int readTCPRequests(int client_fd, std::vector<TCPCopyBackend::Task> &tasks) {
    ssize_t nbytes;
    uint64_t num_reqs;
    nbytes = readFully(client_fd, &num_reqs, sizeof(num_reqs));
    if (nbytes != sizeof(num_reqs)) {
        std::cerr << "Failed to read number of TCP requests" << std::endl;
        return -1;
    }
    tasks.reserve(num_reqs);

    struct Req {
        uint64_t target_addr;
        uint64_t length;
    };
    std::vector<Req> reqs(num_reqs);
    size_t reqs_nbytes = sizeof(Req) * num_reqs;

    nbytes = readFully(client_fd, reqs.data(), reqs_nbytes);
    if (nbytes != static_cast<ssize_t>(reqs_nbytes)) {
        std::cerr << "Failed to read TCP requests" << std::endl;
        return -1;
    }

    for (const auto &req : reqs)
        tasks.emplace_back(reinterpret_cast<void *>(req.target_addr),
                           req.length);

    return 0;
};
