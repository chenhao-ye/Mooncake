#pragma once

#include <sys/types.h>

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

class TcpCopyBackend {
   public:
    struct Task {
        void *source_addr;
        size_t length;

        Task(void *source_addr, size_t length)
            : source_addr(source_addr), length(length) {}
    };

   public:
    TcpCopyBackend(FlexTransferEngine &engine, RegionMgr &region_mgr);
    ~TcpCopyBackend() { cleanup(); }

    int processRequest(int client_fd, std::vector<Task> &tasks);

    void cleanup();

   private:
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
#endif

    // Helper structure to represent a single chunk within the pipeline
    struct ChunkIter {
        void *source_addr{nullptr};
        size_t length{0};
        LocId loc_id{0, -1};
        int buffer_idx{0};
    };

    // Helper function to get the next chunk from the task sequence
    // Advances task_idx and chunk_offset as chunks are consumed
    bool getNextChunk(size_t &task_idx, size_t &chunk_offset,
                      const std::vector<Task> &tasks, int buffer_idx,
                      ChunkIter &iter_out);

    FlexTransferEngine &engine_;
    RegionMgr &region_mgr_;

#ifdef USE_CUDA
    BufferPair *buffer_pair_;
#endif
};
