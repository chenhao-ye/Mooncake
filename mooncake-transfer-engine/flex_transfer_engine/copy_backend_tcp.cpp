#include "copy_backend_tcp.h"

#include <cassert>
#include <cstring>
#include <iostream>

#include "flex_transfer_engine.h"
#include "util.h"

#ifdef USE_CUDA
TcpCopyBackend::BufferPair::BufferPair()
    : buffers{nullptr, nullptr}, streams{nullptr, nullptr} {
    cudaError_t err;

    err = cudaMallocHost(&buffers[0], kBufferSize);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMallocHost(&buffers[1], kBufferSize);
    if (err != cudaSuccess) goto cleanup;

    err = cudaStreamCreate(&streams[0]);
    if (err != cudaSuccess) goto cleanup;
    err = cudaStreamCreate(&streams[1]);
    if (err != cudaSuccess) goto cleanup;

    return;

cleanup:
    if (streams[0]) cudaStreamDestroy(streams[0]);
    if (streams[1]) cudaStreamDestroy(streams[1]);
    if (buffers[0]) cudaFreeHost(buffers[0]);
    if (buffers[1]) cudaFreeHost(buffers[1]);
    throw std::runtime_error(std::string("Failed to initialize BufferPair") +
                             cudaGetErrorString(err));
}

TcpCopyBackend::BufferPair::~BufferPair() {
    if (buffers[0]) cudaFreeHost(buffers[0]);
    if (buffers[1]) cudaFreeHost(buffers[1]);
    if (streams[0]) cudaStreamDestroy(streams[0]);
    if (streams[1]) cudaStreamDestroy(streams[1]);
}

#endif  // USE_CUDA

// TcpCopyBackend constructor
TcpCopyBackend::TcpCopyBackend(FlexTransferEngine &engine,
                               RegionMgr &region_mgr)
    : engine_(engine),
      region_mgr_(region_mgr)
#ifdef USE_CUDA
      ,
      buffer_pair_(nullptr)
#endif
{
}

void TcpCopyBackend::cleanup() {
#ifdef USE_CUDA
    if (buffer_pair_) {
        delete buffer_pair_;
        buffer_pair_ = nullptr;
    }
#endif
}

int TcpCopyBackend::processRequest(int client_fd, std::vector<Task> &tasks) {
#ifdef USE_CUDA
    // Initialize buffer pair on first use (only needed for CUDA)
    if (!buffer_pair_) buffer_pair_ = new BufferPair();

#endif

    // Process each task
    for (size_t task_idx = 0; task_idx < tasks.size(); ++task_idx) {
        Task &task = tasks[task_idx];
        int rc = processTask(client_fd, task);
        if (rc != 0) {
            std::cerr << "Failed to process task " << task_idx << std::endl;
            return -1;
        }
    }

    std::cerr << "Completed TCP transfer request: sent " << tasks.size()
              << " tasks" << std::endl;
    return 0;
}

int TcpCopyBackend::processTask(int client_fd, Task &task) {
#ifdef USE_CUDA
    // Get region info to determine if CUDA or CPU
    RegionMgr::Region *region =
        region_mgr_.getRegion(task.source_addr, task.length);
    if (!region) {
        std::cerr << "Source address " << task.source_addr
                  << " not in registered copiable regions" << std::endl;
        return -1;
    }

    bool is_cuda = region->loc_id.cuda_device >= 0;

    if (!is_cuda) {
        // CPU memory: send directly
        ssize_t nbytes = writeFully(client_fd, task.source_addr, task.length);
        if (nbytes != static_cast<ssize_t>(task.length)) {
            std::cerr << "Failed to send CPU data" << std::endl;
            return -1;
        }
        return 0;
    }

    // CUDA memory: chunked transfer with double buffering
    // Use CUDA device ID from loc_id
    int device_id = region->loc_id.cuda_device;
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device " << device_id << ": "
                  << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    char *src = static_cast<char *>(task.source_addr);
    size_t remaining = task.length;
    int current_buffer = 0;  // Start with buffer 0

    // Process chunks
    while (remaining > 0) {
        size_t chunk_size = std::min(remaining, BufferPair::kBufferSize);
        cudaStream_t stream = buffer_pair_->streams[current_buffer];
        char *dst = buffer_pair_->buffers[current_buffer];

        // Start async copy for current chunk
        err = cudaMemcpyAsync(dst, src, chunk_size, cudaMemcpyDeviceToHost,
                              stream);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpyAsync failed: " << cudaGetErrorString(err)
                      << std::endl;
            return -1;
        }

        // Wait for the copy to complete
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "cudaStreamSynchronize failed: "
                      << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        // Send the data
        ssize_t nbytes = writeFully(client_fd, dst, chunk_size);
        if (nbytes != static_cast<ssize_t>(chunk_size)) {
            std::cerr << "Failed to send CUDA data chunk" << std::endl;
            return -1;
        }

        // Move to next chunk and alternate buffer
        src += chunk_size;
        remaining -= chunk_size;
        current_buffer = 1 - current_buffer;  // Toggle between 0 and 1
    }

    return 0;
#else
    // CPU-only build: just send directly from source
    ssize_t nbytes = writeFully(client_fd, task.source_addr, task.length);
    if (nbytes != static_cast<ssize_t>(task.length)) {
        std::cerr << "Failed to send data" << std::endl;
        return -1;
    }
    return 0;
#endif
}
