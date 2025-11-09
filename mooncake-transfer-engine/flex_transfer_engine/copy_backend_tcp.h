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
   private:
#ifdef USE_CUDA
    // Fixed buffer size for chunked transfers (only needed for CUDA)
    constexpr size_t kBufferSize = 2 * 1024 * 1024;  // 2 MB

    struct BufferPair {
        // buffers[0] and buffers[1] are separate pinned host memory buffers
        char *buffers[2];
        size_t size;  // Size of each buffer (kBufferSize)

        // Each buffer has its own CUDA stream
        cudaStream_t streams[2];

        BufferPair(size_t size);
        ~BufferPair();

        BufferPair(const BufferPair &) = delete;
        BufferPair(BufferPair &&) = delete;
        BufferPair &operator=(const BufferPair &) = delete;
        BufferPair &operator=(BufferPair &&) = delete;
    };
#endif

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
    FlexTransferEngine &engine_;
    RegionMgr &region_mgr_;

#ifdef USE_CUDA
    BufferPair *buffer_pair_;
#endif

    // Process a single task with chunked transfer
    // Returns 0 on success, -1 on error
    int processTask(int client_fd, Task &task);
};
