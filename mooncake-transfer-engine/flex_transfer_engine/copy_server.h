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
    CopyServer(FlexTransferEngine &engine)
        : engine_(engine),
          region_mgr_(),
          rdma_copy_backend_(engine, region_mgr_),
          tcp_copy_backend_(engine, region_mgr_),
          worker_running_(false),
          listener_fd_(-1),
          stop_event_fd_(-1),
          epoll_fd_(-1) {}

    // must called in FlexTransferEngine's dtor, because it relies on a valid
    // TransferEngine
    void cleanup();

    // Register/unregister memory regions for copy-based transfer
    int registerLocalMemory(void *addr, size_t length,
                            const std::string &location);
    int unregisterLocalMemory(void *addr);
    int registerLocalMemoryBatch(std::vector<buffer_entry_t> &buffer_list,
                                 const std::string &location);
    int unregisterLocalMemoryBatch(std::vector<uintptr_t> &addr_list);

    // Start/stop the TCP listener
    void startListener();
    void stopListener();

    const std::string &getServerUrl() const { return local_copy_server_url_; }

   private:
    // Back pointer to FlexTransferEngine
    FlexTransferEngine &engine_;

    // Protects region_mgr_, rdma_copy_backend_, and tcp_copy_backend_
    std::mutex regions_mutex_;

    // Copiable memory regions
    RegionMgr region_mgr_;

    RdmaCopyBackend rdma_copy_backend_;
    TcpCopyBackend tcp_copy_backend_;

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
