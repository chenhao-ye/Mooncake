#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "copy_backend_rdma.h"
#include "copy_backend_tcp.h"
#include "copy_common.h"
#include "region.h"
#include "transfer_engine_c.h"

class FlexTransferEngine;

class CopyServer {
   public:
    CopyServer(RdmaCopyBackend &rdma_copy_backend,
               TcpCopyBackend &tcp_copy_backend)
        : rdma_copy_backend_(rdma_copy_backend),
          tcp_copy_backend_(tcp_copy_backend),
          worker_running_(false),
          listener_fd_(-1),
          stop_event_fd_(-1),
          epoll_fd_(-1) {}

    // Start/stop the TCP listener
    void startListener();
    void stopListener();

    const std::string &getServerUrl() const { return local_copy_server_url_; }

   private:
    RdmaCopyBackend &rdma_copy_backend_;
    TcpCopyBackend &tcp_copy_backend_;

    // Active client connections
    // Server is single-threaded, no mutex needed
    std::unordered_set<int> active_client_fds_;

    std::thread worker_thread_;
    std::atomic<bool> worker_running_;
    int listener_fd_;
    int stop_event_fd_;  // eventfd to signal worker thread to stop
    int epoll_fd_;       // epoll instance for I/O multiplexing
    std::string local_copy_server_url_;

    void workerThread();

    int processRdmaRequest(int client_fd);
    int processTcpRequest(int client_fd);

    // read segment name from fd into segment_name
    int readSegmentName(int client_fd, std::string &segment_name);

    // read RDMA requests from fd into tasks
    int readRdmaRequests(int client_fd, uint64_t &target_progress_addr,
                         std::vector<RdmaCopyBackend::Task> &tasks);

    // read TCP requests from fd into tasks
    int readTcpRequests(int client_fd,
                        std::vector<TcpCopyBackend::Task> &tasks);
};
