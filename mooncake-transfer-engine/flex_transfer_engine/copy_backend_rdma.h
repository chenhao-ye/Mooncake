#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "copy_common.h"
#include "region.h"
#include "transfer_engine_c.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

class FlexTransferEngine;

class RdmaCopyBackend {
   private:
    struct BufferPair {
        // buffers[0] is first half, buffers[1] is second half
        // buffers[0] is also the base address of allocation
        char *buffers[2];
        size_t size;   // Size of each half
        bool is_cuda;  // true if CUDA memory, false if CPU memory

        // Used by tasks with given index (-1 for unused)
        int users[2] = {-1, -1};

#ifdef USE_CUDA
        cudaStream_t cuda_stream;
#endif

        BufferPair(char *buffer_base, size_t size, bool is_cuda)
            : buffers{buffer_base, buffer_base + size},
              size(size),
              is_cuda(is_cuda),
              users{-1, -1}
#ifdef USE_CUDA
              ,
              cuda_stream(nullptr)
#endif
        {
#ifdef USE_CUDA
            if (is_cuda) {
                cudaError_t err = cudaStreamCreate(&cuda_stream);
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("Failed to create CUDA stream: ") +
                        cudaGetErrorString(err));
                }
            }
#endif
        }

        ~BufferPair() {
#ifdef USE_CUDA
            if (is_cuda && cuda_stream) {
                cudaStreamDestroy(cuda_stream);
            }
#endif
        }

        // select the next buffer to use
        // return the one with a lower-index task (likely to finish earlier OR
        // is free for users[i]<0)
        int selectNextBuffer() { return users[0] <= users[1] ? 0 : 1; }
    };

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

   public:
    RdmaCopyBackend(FlexTransferEngine &engine, RegionMgr &region_mgr)
        : engine_(engine), region_mgr_(region_mgr) {}
    ~RdmaCopyBackend() { cleanup(); }

    // Ensure that a buffer pair is ready for the given location with at least
    // the given length
    int prepareBufferPair(LocIdx loc_idx, const std::string &location,
                          size_t length);

    // Process RDMA transfer requests
    // Return 0 for success; non-zero for error; update num_done
    int processRequest(segment_id_t target_segment_id,
                       uint64_t target_progress_addr, std::vector<Task> &tasks,
                       RdmaCopyCtrlBlock *ctrl_block, int32_t &num_done);

    void cleanup();

   private:
    // Require regions_mutex_ to be held before calling
    // When this function is called, there MUST be a buffer pair ready with the
    // proper length (which should have been set up upon registration)
    BufferPair &getBufferPair(LocIdx loc_idx) { return *buffer_pool_[loc_idx]; }

    // Require regions_mutex_ to be held before calling
    BufferPair *allocBufferPair(LocIdx loc_idx, const std::string &location,
                                size_t size);
    // Require regions_mutex_ to be held before calling
    void freeBufferPair(BufferPair *pair);

    // Copy memory from src to dst; handle both CPU and CUDA memory
    // For CUDA memory, uses async copy with the stream from buffer_pair
    void copyMemory(void *dst, const void *src, size_t size,
                    BufferPair &buffer_pair);

    // Execute the task specified by task_idx
    int executeTask(std::vector<Task> &tasks, size_t task_idx,
                    int target_segment_id);

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

    // Back pointer to FlexTransferEngine
    FlexTransferEngine &engine_;
    // Reference to RegionMgr for region lookups
    RegionMgr &region_mgr_;
    // Buffer pool per location (LocIdx -> buffer pair)
    std::vector<BufferPair *> buffer_pool_;
};
