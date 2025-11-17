#include "copy_server.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <glog/logging.h>
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

/* TCP listener and worker thread functions */

void CopyServer::startListener() {
    // Determine the IP address to use for the server URL
    bool use_ipv6 = true;
    auto ip_list = findLocalIpv6Addresses();
    if (ip_list.empty() || ip_list[0].empty()) {  // fallback to IPv4
        use_ipv6 = false;
        ip_list = findLocalIpv4Addresses();
        if (ip_list.empty() || ip_list[0].empty())
            throw std::runtime_error("Failed to find local IP addresses");
    }
    const std::string &server_ip = ip_list[0];

    // Use findAvailableTcpPort to find an available port
    uint16_t tcp_port = findAvailableTcpPort(listener_fd_, use_ipv6);
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

    // Set local_copy_server_url_
    local_copy_server_url_ =
        use_ipv6 ? ("[" + server_ip + "]:" + std::to_string(tcp_port))
                 : (server_ip + ":" + std::to_string(tcp_port));

    LOG(INFO) << "TCP listener started on " << local_copy_server_url_;

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
    worker_running_.store(true, std::memory_order_release);
    worker_thread_ = std::thread(&CopyServer::workerThread, this);
}

void CopyServer::stopListener() {
    if (worker_running_.load(std::memory_order_acquire)) {
        worker_running_.store(false, std::memory_order_release);

        // Signal the eventfd to wake up the worker thread
        if (stop_event_fd_ >= 0) {
            [[maybe_unused]] ssize_t nbytes;
            uint64_t event_value = 1;
            nbytes =
                writeFully(stop_event_fd_, &event_value, sizeof(event_value));
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

        LOG(INFO) << "TCP listener stopped";
    }
}

void CopyServer::workerThread() {
    LOG(INFO) << "Worker thread started";

    // Set listener to non-blocking
    int flags = fcntl(listener_fd_, F_GETFL, 0);
    fcntl(listener_fd_, F_SETFL, flags | O_NONBLOCK);

    constexpr int kMaxEvents = 64;
    struct epoll_event events[kMaxEvents];

    while (worker_running_.load(std::memory_order_acquire)) {
        // Wait for events
        int nfds = epoll_wait(epoll_fd_, events, kMaxEvents, -1);
        if (nfds < 0) {
            if (errno == EINTR) continue;
            LOG(ERROR) << "epoll_wait error: " << strerror(errno);
            continue;
        }

        // Process all ready events
        for (int i = 0; i < nfds; ++i) {
            int ready_fd = events[i].data.fd;

            // Check if we were signaled to stop
            if (ready_fd == stop_event_fd_) {
                [[maybe_unused]] ssize_t nbytes;
                uint64_t event_value;
                nbytes = readFully(stop_event_fd_, &event_value,
                                   sizeof(event_value));
                assert(nbytes == sizeof(event_value));
                goto cleanup;  // Exit the worker thread loop
            }

            // Check for new connections on listener
            if (ready_fd == listener_fd_) {
                struct sockaddr_storage client_addr;
                socklen_t client_len = sizeof(client_addr);

                int client_fd = accept(
                    listener_fd_, (struct sockaddr *)&client_addr, &client_len);
                if (client_fd >= 0) {
                    char client_ip[INET6_ADDRSTRLEN];
                    char client_port[NI_MAXSERV];
                    int rc = getnameinfo(
                        (struct sockaddr *)&client_addr, client_len, client_ip,
                        sizeof(client_ip), client_port, sizeof(client_port),
                        NI_NUMERICHOST | NI_NUMERICSERV);
                    if (rc == 0) {
                        LOG(INFO) << "Accepted connection from " << client_ip
                                  << ":" << client_port;
                    } else {
                        LOG(INFO) << "Accepted connection (failed to resolve "
                                     "address)";
                    }

                    // Add new client to epoll
                    struct epoll_event ev;
                    ev.events = EPOLLIN | EPOLLRDHUP;
                    ev.data.fd = client_fd;
                    rc = epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, client_fd, &ev);
                    if (rc < 0) {
                        LOG(ERROR) << "Failed to add client to epoll: "
                                   << strerror(errno);
                        close(client_fd);
                    } else {
                        active_client_fds_.insert(client_fd);
                    }
                }
                continue;
            }

            // Check for connection closure or errors
            if (events[i].events & (EPOLLHUP | EPOLLERR | EPOLLRDHUP)) {
                LOG(INFO) << "Client connection closed or error detected fd="
                          << ready_fd << " events=0x" << std::hex
                          << events[i].events << std::dec;
                epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, ready_fd, nullptr);
                close(ready_fd);
                active_client_fds_.erase(ready_fd);
                continue;
            }

            // Handle client requests
            int rc = 0;
            CopyMode copy_mode;
            ssize_t nbytes = readFully(ready_fd, &copy_mode, sizeof(copy_mode));
            if (nbytes == sizeof(copy_mode)) {
                if (copy_mode == CopyMode::RDMA) {
                    rc = processRdmaRequest(ready_fd);
                } else if (copy_mode == CopyMode::TCP) {
                    rc = processTcpRequest(ready_fd);
                } else {
                    LOG(ERROR) << "Invalid CopyMode value: "
                               << static_cast<uint32_t>(copy_mode);
                    rc = -1;
                }
            } else {
                LOG(ERROR) << "Failed to read CopyMode from client fd="
                           << ready_fd;
                rc = -1;
            }

            if (rc != 0) {
                // Error occurred or client disconnected, close and remove
                LOG(INFO) << "Closing client connection fd=" << ready_fd;

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

    LOG(INFO) << "Worker thread stopped";
}

int CopyServer::processRdmaRequest(int client_fd) {
    int rc;
    std::string target_segment_name;
    uint64_t target_progress_addr = 0;
    std::vector<RdmaCopyBackend::Task> tasks;
    RdmaCopyCtrlBlock *ctrl_block = nullptr;
    int32_t num_done = 0;
    bool success = false;

    rc = readSegmentName(client_fd, target_segment_name);
    if (rc) goto cleanup;

    rc = readRdmaRequests(client_fd, target_progress_addr, tasks);
    if (rc) goto cleanup;

    ctrl_block = rdma_copy_backend_.allocCtrlBlock();
    if (!ctrl_block) {
        LOG(ERROR) << "Failed to acquire RdmaCopyCtrlBlock";
        goto cleanup;
    }

    rc = rdma_copy_backend_.processRequest(
        target_segment_name, target_progress_addr, tasks, ctrl_block, num_done);
    if (rc == 0) success = true;

cleanup:
    if (ctrl_block) rdma_copy_backend_.freeCtrlBlock(ctrl_block);

    // Send completion count via socket (i.e., num_done)
    // If this fails, the connection should be closed
    if (writeFully(client_fd, &num_done, sizeof(num_done)) !=
        sizeof(num_done)) {
        LOG(ERROR) << "Failed to send completion count, closing connection";
        success = false;
    }

    // Return 0 if all tasks completed successfully, -1 otherwise
    return success ? 0 : -1;
}

int CopyServer::processTcpRequest(int client_fd) {
    int rc;
    std::vector<TcpCopyBackend::Task> tasks;

    rc = readTcpRequests(client_fd, tasks);
    if (rc) return -1;

    return tcp_copy_backend_.processRequest(client_fd, tasks);
}

// read segment name from fd into segment_name
int CopyServer::readSegmentName(int client_fd, std::string &segment_name) {
    uint32_t segment_name_len;
    // read segment name length
    if (readFully(client_fd, &segment_name_len, sizeof(segment_name_len)) !=
        sizeof(segment_name_len)) {
        LOG(ERROR) << "Failed to read segment name length";
        return -1;
    }

    constexpr size_t kMaxLength = 1ull << 20;
    if (segment_name_len == 0 || segment_name_len > kMaxLength) {
        LOG(ERROR) << "Invalid segment name length: " << segment_name_len;
        return -1;
    }

    // read segment name
    segment_name.resize(segment_name_len + 1);
    if (readFully(client_fd, segment_name.data(), segment_name_len) !=
        segment_name_len) {
        LOG(ERROR) << "Failed to read segment name";
        return -1;
    }

    LOG(INFO) << "Received request for segment: " << segment_name;
    return 0;
}

// read RDMA requests from fd into tasks
int CopyServer::readRdmaRequests(int client_fd, uint64_t &target_progress_addr,
                                 std::vector<RdmaCopyBackend::Task> &tasks) {
    ssize_t nbytes;
    RdmaHeader header;

    nbytes = readFully(client_fd, &header, sizeof(header));
    if (nbytes != sizeof(header)) {
        LOG(ERROR) << "Failed to read RDMA request header";
        return -1;
    }

    target_progress_addr = header.progress_addr;
    tasks.reserve(header.num_reqs);

    LOG(INFO) << "Received transfer request: progress_addr=0x" << std::hex
              << target_progress_addr << std::dec
              << ", num_reqs=" << header.num_reqs;

    std::vector<RdmaReq> reqs(header.num_reqs);
    size_t reqs_nbytes = sizeof(RdmaReq) * header.num_reqs;

    nbytes = readFully(client_fd, reqs.data(), reqs_nbytes);
    if (nbytes != static_cast<ssize_t>(reqs_nbytes)) {
        LOG(ERROR) << "Failed to read requests";
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
int CopyServer::readTcpRequests(int client_fd,
                                std::vector<TcpCopyBackend::Task> &tasks) {
    ssize_t nbytes;
    TcpHeader header;
    nbytes = readFully(client_fd, &header, sizeof(header));
    if (nbytes != sizeof(header)) {
        LOG(ERROR) << "Failed to read number of TCP requests";
        return -1;
    }
    tasks.reserve(header.num_reqs);

    std::vector<TcpReq> reqs(header.num_reqs);
    size_t reqs_nbytes = sizeof(TcpReq) * header.num_reqs;

    nbytes = readFully(client_fd, reqs.data(), reqs_nbytes);
    if (nbytes != static_cast<ssize_t>(reqs_nbytes)) {
        LOG(ERROR) << "Failed to read TCP requests";
        return -1;
    }

    for (const auto &req : reqs)
        tasks.emplace_back(reinterpret_cast<void *>(req.target_addr),
                           req.length);

    return 0;
};
