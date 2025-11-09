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

bool TcpCopyBackend::getNextChunk(TaskIter &task_iter, std::vector<Task> &tasks,
                                  int buffer_idx, ChunkIter &iter_out) {
    // iterate through tasks to find the next chunk
    while (task_iter.task_idx < tasks.size()) {
        Task &task = tasks[task_iter.task_idx];
        if (task_iter.chunk_offset >= task.length) {  // move to next task
            task_iter.task_idx++;
            task_iter.chunk_offset = 0;
            continue;
        }

        if (!task.region) {
            task.region = region_mgr_.getRegion(task.source_addr, task.length);
            if (!task.region) {
                std::cerr << "Source address 0x" << std::hex << task.source_addr
                          << " not in registered copiable regions" << std::endl;
                throw std::runtime_error(
                    "Source address not in registered copiable regions");
            }
        }

        iter_out.source_addr =
            static_cast<char *>(task.source_addr) + task_iter.chunk_offset;
        size_t remaining = task.length - task_iter.chunk_offset;
        iter_out.loc_id = task.region->loc_id;

        // Chunk only for CUDA memory; CPU memory can be sent directly
        if (task.region->loc_id.isCuda()) {
#ifdef USE_CUDA
            iter_out.length = std::min(remaining, BufferPair::kBufferSize);
#else
            // should not happen because register has checked CUDA support
            throw std::runtime_error(
                "CUDA memory encountered but CUDA support not compiled");
#endif
        } else {
            iter_out.length = remaining;  // Send entire remaining data for CPU
        }

        iter_out.buffer_idx = buffer_idx;
        task_iter.chunk_offset += iter_out.length;
        return true;
    }

    return false;  // no more chunks available
}

int TcpCopyBackend::processRequest(int client_fd, std::vector<Task> &tasks) {
    if (tasks.empty()) return 0;

#ifdef USE_CUDA
    // Initialize buffer pair on first use (only needed for CUDA)
    if (!buffer_pair_) buffer_pair_ = new BufferPair();
#endif

    // Iteration state for chunk traversal
    TaskIter task_iter;

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
    // Start async CUDA copy for the first two chunks (skip if not CUDA)

    if (!getNextChunk(task_iter, tasks, buffer_idx, *curr))
        throw std::runtime_error("Failed to get first chunk");

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
#endif  // skip else check because getNextChunk has already checked
    }

    buffer_idx = 1 - buffer_idx;  // toggle
    bool has_next = getNextChunk(task_iter, tasks, buffer_idx, *next);
#ifdef USE_CUDA
    if (has_next && next->loc_id.isCuda()) {
        if (curr_device != next->loc_id.cuda_device) {
            cudaError_t err = cudaSetDevice(next->loc_id.cuda_device);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                                         cudaGetErrorString(err));
            }
            curr_device = next->loc_id.cuda_device;
        }

        cudaError_t err = cudaMemcpyAsync(
            buffer_pair_->buffers[buffer_idx], next->source_addr, next->length,
            cudaMemcpyDeviceToHost, buffer_pair_->streams[buffer_idx]);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") +
                                     cudaGetErrorString(err));
        }
    }
#endif  // skip else check because getNextChunk has already checked

    // ===== Phase 2: Main pipeline loop =====
    // Repeat:
    // - wait for curr chunk to finish copy
    // - send curr chunk
    // - start async CUDA copy for next chunk
    while (has_next) {
#ifdef USE_CUDA
        // Wait for curr chunk to complete copy; no need waiting for CPU memory
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

        // Send current chunk
        void *send_addr = curr->source_addr;
#ifdef USE_CUDA
        if (curr->loc_id.isCuda()) {
            send_addr = buffer_pair_->buffers[curr->buffer_idx];
        }
#endif
        ssize_t nbytes = writeFully(client_fd, send_addr, curr->length);
        if (nbytes != static_cast<ssize_t>(curr->length)) return -1;

        // advance pipeline: swap curr and next, and start next async copy
        ChunkIter *temp = curr;
        curr = next;
        next = temp;

        buffer_idx = 1 - buffer_idx;  // toggle
        has_next = getNextChunk(task_iter, tasks, buffer_idx, *next);
#ifdef USE_CUDA
        if (has_next && next->loc_id.isCuda()) {
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
