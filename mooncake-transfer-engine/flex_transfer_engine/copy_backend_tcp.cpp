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
    {  // shutdown all worker threads and clean up ctrl blocks
        std::lock_guard<std::mutex> lock(ctrl_block_mutex_);
        for (auto *ctrl_block : ctrl_block_cache_) {
            {  // Signal thread to shutdown
                std::lock_guard<std::mutex> ctrl_block_lock(ctrl_block->mutex_);
                ctrl_block->worker_running_ = false;
            }
            ctrl_block->cv_.notify_one();
        }
        for (auto *ctrl_block : ctrl_block_cache_) {
            if (ctrl_block->worker_thread_.joinable())
                ctrl_block->worker_thread_.join();
            delete ctrl_block;
        }
        ctrl_block_cache_.clear();
    }

#ifdef USE_CUDA
    if (buffer_pair_) {
        delete buffer_pair_;
        buffer_pair_ = nullptr;
    }
#endif
}

TcpCopyCtrlBlock *TcpCopyBackend::allocCtrlBlock() {
    std::lock_guard<std::mutex> lock(ctrl_block_mutex_);

    TcpCopyCtrlBlock *ctrl_block = nullptr;
    if (!ctrl_block_cache_.empty()) {
        ctrl_block = ctrl_block_cache_.back();
        ctrl_block_cache_.pop_back();
    } else {  // empty cache; allocate new ctrl block
        ctrl_block = new TcpCopyCtrlBlock(*this);
    }
    // the caller should acquire the mutex and initialize fields
    return ctrl_block;
}

void TcpCopyBackend::freeCtrlBlock(TcpCopyCtrlBlock *ctrl_block) {
    if (!ctrl_block) return;
    std::lock_guard<std::mutex> lock(ctrl_block_mutex_);
    ctrl_block_cache_.emplace_back(ctrl_block);
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
            task.region = region_mgr_.getRegion(task.addr, task.length);
            if (!task.region) {
                std::cerr << "Source address 0x" << std::hex << task.addr
                          << " not in registered copiable regions" << std::endl;
                throw std::runtime_error(
                    "Source address not in registered copiable regions");
            }
        }

        iter_out.addr = static_cast<char *>(task.addr) + task_iter.chunk_offset;
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

#ifdef USE_CUDA
void TcpCopyBackend::ensureCudaDevice(int target_device, int &curr_device) {
    if (curr_device != target_device) {
        cudaError_t err = cudaSetDevice(target_device);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                                     cudaGetErrorString(err));
        }
        curr_device = target_device;
    }
}

void TcpCopyBackend::startAsyncCopy(int buffer_idx, const ChunkIter *chunk_iter,
                                    cudaMemcpyKind direction) {
    void *dst, *src;

    if (direction == cudaMemcpyDeviceToHost) {
        // GPU -> buffer (for send/processRequest)
        dst = buffer_pair_->buffers[buffer_idx];
        src = chunk_iter->addr;
    } else {  // cudaMemcpyHostToDevice
        // buffer -> GPU (for recv/processResponse)
        dst = chunk_iter->addr;
        src = buffer_pair_->buffers[buffer_idx];
    }

    cudaError_t err = cudaMemcpyAsync(dst, src, chunk_iter->length, direction,
                                      buffer_pair_->streams[buffer_idx]);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") +
                                 cudaGetErrorString(err));
    }
}

void TcpCopyBackend::waitForCudaCopy(cudaStream_t stream) {
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") +
                                 cudaGetErrorString(err));
    }
}
#endif

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
        ensureCudaDevice(curr->loc_id.cuda_device, curr_device);
        startAsyncCopy(buffer_idx, curr, cudaMemcpyDeviceToHost);
#endif  // skip else check because getNextChunk has already checked
    }

    buffer_idx = 1 - buffer_idx;  // toggle
    bool has_next = getNextChunk(task_iter, tasks, buffer_idx, *next);
#ifdef USE_CUDA
    if (has_next && next->loc_id.isCuda()) {
        ensureCudaDevice(next->loc_id.cuda_device, curr_device);
        startAsyncCopy(buffer_idx, next, cudaMemcpyDeviceToHost);
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
            waitForCudaCopy(buffer_pair_->streams[curr->buffer_idx]);
        }
#endif

        // Send current chunk
        void *send_addr = curr->addr;
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
            ensureCudaDevice(next->loc_id.cuda_device, curr_device);
            startAsyncCopy(buffer_idx, next, cudaMemcpyDeviceToHost);
        }
#endif
    }

    // ===== Phase 3: Drain the last chunk =====
    // Handle the last chunk
    void *send_addr = curr->addr;
#ifdef USE_CUDA
    if (curr->loc_id.isCuda()) {
        waitForCudaCopy(buffer_pair_->streams[curr->buffer_idx]);
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

int TcpCopyBackend::processResponse(int server_fd, std::vector<Task> &tasks) {
    if (tasks.empty()) return 0;

#ifdef USE_CUDA
    // Initialize buffer pair on first use (only needed for CUDA)
    if (!buffer_pair_) buffer_pair_ = new BufferPair();
#endif

    // Iteration state for chunk traversal
    TaskIter task_iter;

    // Use two ChunkIter objects and swap pointers to avoid copying
    ChunkIter chunk_iters[2];
    ChunkIter *prev = &chunk_iters[0];
    ChunkIter *curr = &chunk_iters[1];
    int buffer_idx = 0;

#ifdef USE_CUDA
    // Track prev CUDA device to avoid redundant cudaSetDevice calls
    int curr_device = -1;
#endif

    // ===== Phase 1: Prime the pipeline =====
    // Receive first chunks from TCP and start async GPU copy

    if (!getNextChunk(task_iter, tasks, buffer_idx, *prev))
        throw std::runtime_error("Failed to get first chunk");

    // Receive first chunk into buffer or direct to CPU memory
    void *recv_addr = prev->addr;
#ifdef USE_CUDA
    if (prev->loc_id.isCuda()) {
        recv_addr = buffer_pair_->buffers[buffer_idx];
    }
#endif
    ssize_t nbytes = readFully(server_fd, recv_addr, prev->length);
    if (nbytes != static_cast<ssize_t>(prev->length)) return -1;

#ifdef USE_CUDA
    if (prev->loc_id.isCuda()) {
        ensureCudaDevice(prev->loc_id.cuda_device, curr_device);
        startAsyncCopy(buffer_idx, prev, cudaMemcpyHostToDevice);
    }
#endif

    // ===== Phase 2: Main pipeline loop =====
    // Repeat uniform pattern:
    // - receive curr chunk from TCP
    // - start async GPU copy for curr
    // - wait for prev chunk to finish GPU copy
    // - swap prev and curr
    buffer_idx = 1 - buffer_idx;  // toggle
    while (getNextChunk(task_iter, tasks, buffer_idx, *curr)) {
        recv_addr = curr->addr;
#ifdef USE_CUDA
        if (curr->loc_id.isCuda()) {
            recv_addr = buffer_pair_->buffers[buffer_idx];
        }
#endif
        nbytes = readFully(server_fd, recv_addr, curr->length);
        if (nbytes != static_cast<ssize_t>(curr->length)) return -1;

#ifdef USE_CUDA
        if (curr->loc_id.isCuda()) {
            ensureCudaDevice(curr->loc_id.cuda_device, curr_device);
            startAsyncCopy(buffer_idx, curr, cudaMemcpyHostToDevice);
        }
#endif

        // Wait for prev chunk to complete GPU copy
#ifdef USE_CUDA
        if (prev->loc_id.isCuda()) {
            waitForCudaCopy(buffer_pair_->streams[prev->buffer_idx]);
        }
#endif

        // Advance pipeline: swap prev and curr
        ChunkIter *temp = prev;
        prev = curr;
        curr = temp;

        buffer_idx = 1 - buffer_idx;  // toggle
        // invariant: curr is done while prev may have pending copy
    }

    // ===== Phase 3: Drain the last chunk =====
#ifdef USE_CUDA
    if (prev->loc_id.isCuda()) {
        waitForCudaCopy(buffer_pair_->streams[prev->buffer_idx]);
    }
#endif

    std::cerr << "Completed TCP transfer response: received " << tasks.size()
              << " tasks" << std::endl;
    return 0;
}

TcpCopyCtrlBlock::TcpCopyCtrlBlock(TcpCopyBackend &backend)
    : progress_counter(0),
      server_fd(-1),
      worker_running_(true),
      backend_(backend) {
    worker_thread_ = std::thread(workerThreadFunc, this);
}

void TcpCopyCtrlBlock::workerThreadFunc(TcpCopyCtrlBlock *ctrl_block) {
    std::unique_lock<std::mutex> lock(ctrl_block->mutex_);
    while (true) {
        ctrl_block->cv_.wait(lock, [ctrl_block]() {
            return !ctrl_block->worker_running_ || ctrl_block->server_fd >= 0;
        });
        if (!ctrl_block->worker_running_) break;
        ctrl_block->backend_.processResponse(ctrl_block->server_fd,
                                             ctrl_block->tasks);
        // reset all fields to indicate done
        ctrl_block->server_fd = -1;
        int64_t num_done = ctrl_block->tasks.size();
        ctrl_block->tasks.clear();
        ctrl_block->progress_counter.store(num_done, std::memory_order_release);
    }
}
