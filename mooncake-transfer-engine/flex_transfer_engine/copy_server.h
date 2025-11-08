#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "transfer_engine_c.h"

// Forward declarations
struct CopyCtrlBlock;
class FlexTransferEngine;

class CopyServer {
   public:
    CopyServer(FlexTransferEngine &engine)
        : worker_running_(false),
          listener_fd_(-1),
          stop_event_fd_(-1),
          epoll_fd_(-1),
          engine_(engine) {}

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
    using LocIdx = int32_t;  // <0 for invalid location

    struct MemoryRegion {
        void *addr;
        size_t length;
        LocIdx loc_idx;
    };

    struct BufferPair {
        // buffers[0] is first half, buffers[1] is second half
        // buffers[0] is also the base address of allocation
        char *buffers[2];
        size_t size;   // Size of each half
        bool is_cuda;  // true if CUDA memory, false if CPU memory

        // Used by tasks with given index (-1 for unused)
        int users[2] = {-1, -1};

        BufferPair(char *buffer_base, size_t size, bool is_cuda)
            : buffers{buffer_base, buffer_base + size},
              size(size),
              is_cuda(is_cuda),
              users{-1, -1} {}

        // select the next buffer to use
        // return the one with a lower-index task (likely to finish earlier OR
        // is free for users[i]<0)
        int selectNextBuffer() { return users[0] < users[1] ? 0 : 1; }
    };

    struct Task {
        // request info
        void *source_addr;
        uint64_t target_addr;
        size_t length;
        // execution info
        batch_id_t batch_id = INVALID_BATCH;
        struct BufferPair *buffer_pair = nullptr;
        int buffer_idx = -1;

        Task(void *source_addr, uint64_t target_addr, size_t length)
            : source_addr(source_addr),
              target_addr(target_addr),
              length(length),
              batch_id(INVALID_BATCH),
              buffer_pair(nullptr),
              buffer_idx(-1) {}
    };

    // Protects copiable_regions_, location_strings_, and buffer_pool_
    std::mutex regions_mutex_;

    // Copiable memory regions (addr -> region info)
    // Tracks regions that can be read via copy transfer.
    // When enable_copy_ is true, these are NOT actually RDMA-registered,
    // only tracked for copy-based transfers.
    // When enable_copy_ is false, these ARE RDMA-registered.
    std::unordered_map<void *, MemoryRegion> copiable_regions_;

    // Buffer pool per location (LocIdx -> buffer pair)
    // Only used when enable_copy is true
    std::vector<BufferPair *> buffer_pool_;

    // Location string storage (LocIdx -> location string)
    // Append-only; will never remove entries
    std::vector<std::string> location_strings_;

    // Active client connections
    // Server is single-threaded, no mutex needed
    std::unordered_set<int> active_client_fds_;

    std::thread worker_thread_;
    std::atomic<bool> worker_running_;
    int listener_fd_;
    int stop_event_fd_;  // eventfd to signal worker thread to stop
    int epoll_fd_;       // epoll instance for I/O multiplexing
    std::string local_copy_server_url_;

    // Back pointer to FlexTransferEngine
    FlexTransferEngine &engine_;

    // Require regions_mutex_ to be held before calling
    LocIdx getLocIdx(const std::string &location);

    // Require regions_mutex_ to be held before calling
    MemoryRegion *getRegion(void *addr, size_t length);

    // Require regions_mutex_ to be held before calling
    // When this function is called, there MUST be a buffer pair ready with the
    // proper length (which should have been set up upon registration)
    BufferPair &getBufferPair(LocIdx loc_idx) { return *buffer_pool_[loc_idx]; }

    // Require regions_mutex_ to be held before calling
    BufferPair *allocBufferPair(LocIdx loc_idx, const std::string &location,
                                size_t size);
    // Require regions_mutex_ to be held before calling
    void freeBufferPair(BufferPair *pair);

    void copyMemory(void *dst, const void *src, size_t size, bool is_cuda);

    void workerThread();

    // Returns 0 on success, -1 on error (connection should be closed)
    int processRequest(int client_fd);

    // read segment name from fd and write into segment_name
    int readSegmentName(int client_fd, std::string &segment_name);

    // read requests from fd and write into tasks
    int readTasks(int client_fd, uint64_t &target_progress_addr,
                  std::vector<Task> &tasks);

    // execute the task specified by task_idx
    int executeTask(std::vector<Task> tasks, size_t task_idx,
                    int target_segment_id);

    // Wait until the given task is done
    int waitTask(Task &task);

    // Poll if the given prorgess_batch_id has finished; if so, submit another
    // progress update via atomic fetch-add, which will update prorgess_batch_id
    // and last_updated_progress
    int tryUpdateRemoteProgress(batch_id_t &prorgess_batch_id,
                                int32_t &last_updated_num_done,
                                int32_t num_done,
                                CopyCtrlBlock *copy_ctrl_block,
                                segment_id_t target_segment_id,
                                uint64_t target_progress_addr);

    // Some handy helper functions for a size=1 batch

    // Submit a size=1 batch with the given request; will updates batch_id; if
    // fail, will free the batch and reset batch_id to INVALID_BATCH
    // Return 0 for success; non-zero for error
    int submitBatch(batch_id_t &batch_id, transfer_request_t &req);

    // Free the batch and reset batch_id to INVALID_BATCH
    void freeBatch(batch_id_t &batch_id);

    // Poll size=1 batch and return status
    int pollBatch(batch_id_t batch_id);

    // Wait until the given batch (size=1) is done and then free the batch; will
    // update batch_id to INVALID_BATCH; return the status
    int waitBatch(batch_id_t &batch_id);
};
