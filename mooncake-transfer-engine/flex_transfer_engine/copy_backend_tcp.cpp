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

bool TcpCopyBackend::getNextChunk(size_t &task_idx, size_t &chunk_offset,
                                  const std::vector<Task> &tasks,
                                  int buffer_idx, ChunkIter &iter_out) {
    // iterate through tasks to find the next chunk
    while (task_idx < tasks.size()) {
        const Task &task = tasks[task_idx];
        if (chunk_offset >= task.length) {  // move to next task
            task_idx++;
            chunk_offset = 0;
            continue;
        }

        RegionMgr::Region *region =
            region_mgr_.getRegion(task.source_addr, task.length);
        if (!region) {
            std::cerr << "Source address 0x" << std::hex << task.source_addr
                      << " not in registered copiable regions" << std::endl;
            return false;
        }

        iter_out.loc_id = region->loc_id;

        // Calculate chunk parameters
        iter_out.source_addr =
            static_cast<char *>(task.source_addr) + chunk_offset;
        size_t remaining = task.length - chunk_offset;
#ifdef USE_CUDA
        iter_out.length = std::min(remaining, BufferPair::kBufferSize);
#else
        iter_out.length = remaining;  // No chunking for CPU-only builds
#endif
        iter_out.buffer_idx = buffer_idx;

        chunk_offset += iter_out.length;

        return true;
    }

    return false;  // no more chunks available
}

int TcpCopyBackend::processRequest(int client_fd, std::vector<Task> &tasks) {
#ifdef USE_CUDA
    // Initialize buffer pair on first use (only needed for CUDA)
    if (!buffer_pair_) buffer_pair_ = new BufferPair();
#endif

    if (tasks.empty()) return 0;

    // Iteration state for chunk traversal
    size_t task_idx = 0;
    size_t chunk_offset = 0;

    // Use two ChunkIter objects and swap pointers to avoid copying
    ChunkIter chunk_iters[2];
    ChunkIter *curr = &chunk_iters[0];
    ChunkIter *next = &chunk_iters[1];
    int buffer_idx = 0;

#ifdef USE_CUDA
    // Track curr CUDA device to avoid redundant cudaSetDevice calls
    int curr_device = -1;
#endif

    // ===== Phase 1: Prime the pipeline =====
    // Get first chunk
    if (!getNextChunk(task_idx, chunk_offset, tasks, buffer_idx, *curr))
        throw std::runtime_error("Failed to get first chunk");

    // Start async copy for first chunk if CUDA
    if (curr->loc_id.isCuda()) {
#ifdef USE_CUDA
        if (curr_device != curr->loc_id.cuda_device) {
            cudaError_t err = cudaSetDevice(curr->loc_id.cuda_device);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                                         cudaGetErrorString(err));
            }
            curr_device = curr->loc_id.cuda_device;
        }

        cudaError_t err = cudaMemcpyAsync(
            buffer_pair_->buffers[buffer_idx], curr->source_addr, curr->length,
            cudaMemcpyDeviceToHost, buffer_pair_->streams[buffer_idx]);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") +
                                     cudaGetErrorString(err));
        }
#else
        throw std::runtime_error(
            "GPU memory copy requested but CUDA support not compiled");
#endif
    }

    // Get second chunk and start its copy (prepare ahead)
    buffer_idx = 1 - buffer_idx;  // Toggle buffer
    bool has_next =
        getNextChunk(task_idx, chunk_offset, tasks, buffer_idx, *next);
    if (has_next) {
        if (next->loc_id.isCuda()) {
#ifdef USE_CUDA
            if (curr_device != next->loc_id.cuda_device) {
                cudaError_t err = cudaSetDevice(next->loc_id.cuda_device);
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("cudaSetDevice failed: ") +
                        cudaGetErrorString(err));
                }
                curr_device = next->loc_id.cuda_device;
            }

            cudaError_t err = cudaMemcpyAsync(
                buffer_pair_->buffers[buffer_idx], next->source_addr,
                next->length, cudaMemcpyDeviceToHost,
                buffer_pair_->streams[buffer_idx]);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaMemcpyAsync failed: ") +
                    cudaGetErrorString(err));
            }
#else
            throw std::runtime_error(
                "GPU memory copy requested but CUDA support not compiled");
#endif
        }
    }

    // ===== Phase 2: Main pipeline loop =====
    while (has_next) {
        // Wait for curr chunk to complete copy
#ifdef USE_CUDA
        if (curr->loc_id.isCuda()) {
            cudaError_t err =
                cudaStreamSynchronize(buffer_pair_->streams[curr->buffer_idx]);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaStreamSynchronize failed: ") +
                    cudaGetErrorString(err));
            }
        }
#endif

        // Send curr chunk
        void *send_addr = curr->source_addr;
#ifdef USE_CUDA
        if (curr->loc_id.isCuda()) {
            send_addr = buffer_pair_->buffers[curr->buffer_idx];
        }
#endif
        ssize_t nbytes = writeFully(client_fd, send_addr, curr->length);
        if (nbytes != static_cast<ssize_t>(curr->length)) {
            return -1;  // Network error - external failure, return gracefully
        }

        // Advance pipeline: swap pointers instead of copying
        ChunkIter *temp = curr;
        curr = next;
        next = temp;

        // Get next chunk and start its copy
        buffer_idx = 1 - buffer_idx;  // Toggle buffer
        has_next =
            getNextChunk(task_idx, chunk_offset, tasks, buffer_idx, *next);
        if (has_next) {
#ifdef USE_CUDA
            if (next->loc_id.isCuda()) {
                if (curr_device != next->loc_id.cuda_device) {
                    cudaError_t err = cudaSetDevice(next->loc_id.cuda_device);
                    if (err != cudaSuccess) {
                        throw std::runtime_error(
                            std::string("cudaSetDevice failed: ") +
                            cudaGetErrorString(err));
                    }
                    curr_device = next->loc_id.cuda_device;
                }

                cudaError_t err = cudaMemcpyAsync(
                    buffer_pair_->buffers[buffer_idx], next->source_addr,
                    next->length, cudaMemcpyDeviceToHost,
                    buffer_pair_->streams[buffer_idx]);
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("cudaMemcpyAsync failed: ") +
                        cudaGetErrorString(err));
                }
            }
#endif
        }
    }

    // ===== Phase 3: Drain the last chunk =====
#ifdef USE_CUDA
    if (curr->loc_id.isCuda()) {
        cudaError_t err =
            cudaStreamSynchronize(buffer_pair_->streams[curr->buffer_idx]);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaStreamSynchronize failed: ") +
                cudaGetErrorString(err));
        }
    }
#endif

    // Send last chunk
    void *send_addr = curr->source_addr;
#ifdef USE_CUDA
    if (curr->loc_id.isCuda()) {
        send_addr = buffer_pair_->buffers[curr->buffer_idx];
    }
#endif
    ssize_t nbytes = writeFully(client_fd, send_addr, curr->length);
    if (nbytes != static_cast<ssize_t>(curr->length)) {
        std::cerr << "Failed to send last chunk" << std::endl;
        return -1;
    }

    std::cerr << "Completed TCP transfer request: sent " << tasks.size()
              << " tasks" << std::endl;
    return 0;
}
