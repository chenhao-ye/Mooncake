#pragma once

#include <sys/types.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "copy_common.h"
#include "region.h"
#include "transfer_engine_c.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

class FlexTransferEngine;
class TcpCopyBackend;
struct TcpCopyCtrlBlock;

class TcpCopyBackend {
    struct BufferPair;  // forward declaration

   public:
    struct Task {
        void *addr;
        size_t length;
        Region *region;  // set during processing

        Task(void *addr, size_t length)
            : addr(addr), length(length), region(nullptr) {}
    };

   private:
    FlexTransferEngine &engine_;
    RegionMgr &region_mgr_;

   public:
    TcpCopyBackend(FlexTransferEngine &engine, RegionMgr &region_mgr);
    ~TcpCopyBackend() { cleanup(); }

    TcpCopyBackend(const TcpCopyBackend &) = delete;
    TcpCopyBackend(TcpCopyBackend &&) = delete;
    TcpCopyBackend &operator=(const TcpCopyBackend &) = delete;
    TcpCopyBackend &operator=(TcpCopyBackend &&) = delete;

    // Note we assume there is no concurrent CopyServer calling processRequest
    // and CopyClient calling processResponse, so both functions will acquire
    // the regions_mutex_. It could be optimized as rwlock if necessary.

    // Will acquire regions_mutex_
    int processRequest(int client_fd, std::vector<Task> &tasks);

    // Will acquire regions_mutex_
    // for CopyClient to process TCP responses from processRequest
    // will lively update progress_counter for every task completion
    int processResponse(int server_fd, std::vector<Task> &tasks,
                        std::atomic_int64_t &progress_counter);

    void cleanup();

   public: /* Resource pools: ctrl blocks and buffer pairs */
    TcpCopyCtrlBlock *allocCtrlBlock();
    void freeCtrlBlock(TcpCopyCtrlBlock *ctrl_block);

   private:
    std::vector<TcpCopyCtrlBlock *> ctrl_block_cache_;
    std::mutex ctrl_block_mutex_;

   private:
    // Helper structure to represent a single chunk within the pipeline
    struct ChunkIter {
        void *addr{nullptr};
        size_t length{0};
        LocId loc_id{0, -1};
        int buffer_idx{0};
    };

    struct TaskIter {
        size_t task_idx{0};
        size_t chunk_offset{0};
    };

    // Helper function to get the next chunk from the task sequence
    // Advances TaskIter as chunks are consumed
    bool getNextChunk(TaskIter &task_iter, std::vector<Task> &tasks,
                      int buffer_idx, ChunkIter &iter_out);

#ifdef USE_CUDA
    // Some handy helpers
    // set CUDA device if needed (avoids redundant calls)
    void ensureCudaDevice(int target_device, int &curr_device);
    // start async memory copy (GPU <-> buffer) and handle errors
    void startAsyncCudaCopy(int buffer_idx, const ChunkIter *chunk,
                            cudaMemcpyKind direction);
    // wait for CUDA stream synchronization
    void waitForCudaCopy(cudaStream_t stream);
#endif

   private: /* BufferPair definition */
#ifdef USE_CUDA
    struct BufferPair {
        // Fixed buffer size for chunked transfers (only needed for CUDA)
        static constexpr size_t kBufferSize = 2 * 1024 * 1024;  // 2 MB

        // buffers[0] and buffers[1] are separate pinned host memory buffers
        char *buffers[2];
        // Each buffer has its own CUDA stream
        cudaStream_t streams[2];

        BufferPair();
        ~BufferPair();
        BufferPair(const BufferPair &) = delete;
        BufferPair(BufferPair &&) = delete;
        BufferPair &operator=(const BufferPair &) = delete;
        BufferPair &operator=(BufferPair &&) = delete;
    };

    BufferPair *buffer_pair_;
#endif
};

/**
 * Every TcpCopyCtrlBlock has a dedicated worker thread that processes
 * responses from socket asynchronously. Cache TcpCopyCtrlBlock to reuse the
 * worker thread.
 */
struct TcpCopyCtrlBlock {
    // update these fields when using the ctrl block
    std::atomic_int64_t progress_counter = 0;
    int server_fd = -1;  // only >= 0 means has work
    std::vector<TcpCopyBackend::Task> tasks;

    // Thread and synchronization primitives for async processing
    std::thread worker_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool worker_running_ = true;  // does not need atomic, protected by mutex

    TcpCopyCtrlBlock(TcpCopyBackend &backend);

   private:
    TcpCopyBackend &backend_;

    static void workerThreadFunc(TcpCopyCtrlBlock *);  // start upon ctor
};
