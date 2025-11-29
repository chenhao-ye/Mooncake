#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "copy_common.h"
#include "region.h"
#include "transfer_engine_c.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

class FlexTransferEngine;
class RdmaCopyBackend;
struct RdmaCopyCtrlBlock;

class RdmaCopyBackend {
    struct BufferPair;  // forward declaration

   public:
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

   private:
    FlexTransferEngine &engine_;  // Back pointer to FlexTransferEngine
    RegionMgr &region_mgr_;
    const std::string ctrl_block_location_;

   public:
    RdmaCopyBackend(FlexTransferEngine &engine, RegionMgr &region_mgr,
                    const std::string &ctrl_block_location)
        : engine_(engine),
          region_mgr_(region_mgr),
          ctrl_block_location_(ctrl_block_location) {}
    ~RdmaCopyBackend() { cleanup(); }

    RdmaCopyBackend(const RdmaCopyBackend &) = delete;
    RdmaCopyBackend(RdmaCopyBackend &&) = delete;
    RdmaCopyBackend &operator=(const RdmaCopyBackend &) = delete;
    RdmaCopyBackend &operator=(RdmaCopyBackend &&) = delete;

    // Require regions_mutex_
    // Ensure that a buffer pair is ready for the given location with at least
    // the given length
    int prepareBufferPair(LocId loc_id, const std::string &location,
                          size_t length);

    // Will acquire regions_mutex_
    // Process RDMA transfer requests
    // Return 0 for success; non-zero for error; update num_done
    int processRequest(const std::string &target_segment_name,
                       uint64_t target_progress_addr, std::vector<Task> &tasks,
                       RdmaCopyCtrlBlock *ctrl_block, int32_t &num_done);

    // Require regions_mutex_
    void cleanup();

   public: /* Resource pools: ctrl blocks and buffer pairs */
    RdmaCopyCtrlBlock *allocCtrlBlock();
    void freeCtrlBlock(RdmaCopyCtrlBlock *ctrl_block);

    // Require regions_mutex_ to be held before calling
    // When this function is called, there MUST be a buffer pair ready with the
    // proper length (which should have been set up upon registration)
    BufferPair &getBufferPair(LocId loc_id) {
        assert(static_cast<size_t>(loc_id.idx) < buffer_pool_.size());
        return *buffer_pool_[loc_id.idx];
    }
    // Require regions_mutex_ to be held before calling
    BufferPair *allocBufferPair(LocId loc_id, const std::string &location,
                                size_t size);
    // Require regions_mutex_ to be held before calling
    void freeBufferPair(BufferPair *pair);

   private:
    std::vector<RdmaCopyCtrlBlock *> ctrl_block_cache_;
    std::mutex ctrl_block_mutex_;

    // BufferPair per location
    // Require regions_mutex_ to be held before accessing
    std::vector<BufferPair *> buffer_pool_;

   private: /* Helper functions for task execution */
    // Acquire a buffer from the buffer pair for the given task
    // Waits for the previous task using the selected buffer if needed
    // Sets task.buffer_pair, task.buffer_idx, and marks buffer as owned
    // Returns 0 on success, -1 on error
    int acquireBuffer(BufferPair &buffer_pair, std::vector<Task> &tasks,
                      size_t task_idx);

    // Release a buffer back to the buffer pair (mark as free)
    void releaseBuffer(Task &task);

    // Copy memory from src to buffer; handle both CPU and CUDA memory
    // For CUDA memory, uses async copy and records event for the given
    // buffer_idx Destination is inferred from buffer_pair.buffers[buffer_idx]
    void memcpyAsync(const void *src, size_t size, BufferPair &buffer_pair,
                     int buffer_idx);

    // Phase 1: Start async CUDA copy (non-blocking)
    // Acquires buffer, validates region, starts copy, records CUDA event
    int startTaskCopy(std::vector<Task> &tasks, size_t task_idx,
                      int target_segment_id);

    // Phase 2: Wait for copy completion and submit RDMA transfer
    // Synchronizes on CUDA event if needed, then submits RDMA batch
    int submitTaskRdma(Task &task, int target_segment_id);

    // Wait until the given task is done
    int waitTask(Task &task);

    // Poll if the given progress_batch_id has finished; if so, submit another
    // progress update via atomic fetch-add, which will update progress_batch_id
    // and last_updated_progress
    int tryUpdateRemoteProgress(batch_id_t &progress_batch_id,
                                int32_t &last_updated_num_done,
                                int32_t num_done, RdmaCopyCtrlBlock *ctrl_block,
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

    /* BufferPair definition */
    struct BufferPair {
        struct Buffer {
            char *addr;
            int user;
#ifdef USE_CUDA
            cudaStream_t cuda_stream;
            cudaEvent_t copy_done_event;
#endif
        };
        // buffers[0] is first half, buffers[1] is second half
        // buffers[0].addr is also the base address of allocation
        Buffer buffers[2];
        size_t size;   // Size of each half
        bool is_cuda;  // true if CUDA memory, false if CPU memory

        BufferPair(char *buffer_base, size_t size, bool is_cuda);
        ~BufferPair();

        BufferPair(const BufferPair &) = delete;
        BufferPair(BufferPair &&) = delete;
        BufferPair &operator=(const BufferPair &) = delete;
        BufferPair &operator=(BufferPair &&) = delete;

        // select the next buffer to use
        // return the one with a lower-index task (likely to finish earlier OR
        // is free for buffers[i].user<0)
        int selectNextBuffer() {
            return buffers[0].user <= buffers[1].user ? 0 : 1;
        }
    };
};

/**
 * Used by both CopyClient and CopyServer. Must register with remote_atomic.
 */
struct RdmaCopyCtrlBlock {
    volatile std::atomic_int64_t progress_counter = 0;
    uint64_t padding[7];
};
static_assert(sizeof(RdmaCopyCtrlBlock) == 64,
              "RdmaCopyCtrlBlock must be cacheline-aligned");
