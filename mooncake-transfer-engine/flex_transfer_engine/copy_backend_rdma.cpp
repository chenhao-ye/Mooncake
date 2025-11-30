#include "copy_backend_rdma.h"

#include <glog/logging.h>

#include <cassert>
#include <cstring>
#include <iostream>

#include "flex_transfer_engine.h"
#include "transfer_engine_c.h"

void RdmaCopyBackend::cleanup() {
    {
        std::lock_guard<std::mutex> lock(ctrl_block_mutex_);
        for (RdmaCopyCtrlBlock *ctrl_block : ctrl_block_cache_) {
            ::unregisterLocalMemory(engine_.getEngine(), ctrl_block);
            delete ctrl_block;
        }
        ctrl_block_cache_.clear();
    }
    for (auto pair : buffer_pool_) freeBufferPair(pair);
    buffer_pool_.clear();
}

RdmaCopyCtrlBlock *RdmaCopyBackend::allocCtrlBlock() {
    std::lock_guard<std::mutex> lock(ctrl_block_mutex_);

    // Try to get from cache first
    if (!ctrl_block_cache_.empty()) {
        RdmaCopyCtrlBlock *ctrl_block = ctrl_block_cache_.back();
        ctrl_block_cache_.pop_back();
        ctrl_block->progress_counter.store(0, std::memory_order_release);
        return ctrl_block;
    }

    // Cache is empty, allocate a new one
    RdmaCopyCtrlBlock *ctrl_block = new RdmaCopyCtrlBlock();

    // Register it with RDMA using the specified location
    int rc = ::registerLocalMemory(engine_.getEngine(), ctrl_block,
                                   sizeof(RdmaCopyCtrlBlock),
                                   ctrl_block_location_.c_str(),
                                   /*remote_accessible*/ true,
                                   /*remote_atomic*/ true);
    if (rc) {
        delete ctrl_block;
        return nullptr;
    }
    return ctrl_block;
}

void RdmaCopyBackend::freeCtrlBlock(RdmaCopyCtrlBlock *ctrl_block) {
    if (!ctrl_block) return;
    std::lock_guard<std::mutex> lock(ctrl_block_mutex_);
    ctrl_block_cache_.emplace_back(ctrl_block);
}

int RdmaCopyBackend::prepareBufferPair(LocId loc_id,
                                       const std::string &location,
                                       size_t length) {
    if (static_cast<size_t>(loc_id.idx) >= buffer_pool_.size()) {
        buffer_pool_.resize(loc_id.idx + 1, nullptr);
    }
    // Check if we have a buffer pair for this location
    BufferPair *pair = buffer_pool_[loc_id.idx];
    if (!pair || pair->size < length) {
        freeBufferPair(pair);
        pair = allocBufferPair(loc_id, location, length);
        buffer_pool_[loc_id.idx] = pair;
        if (!pair) return -1;
        // LOG(INFO) << "Allocated buffer pair of size " << length
        //           << " for location " << location;
    }
    return 0;
}

int RdmaCopyBackend::processRequest(const std::string &target_segment_name,
                                    uint64_t target_progress_addr,
                                    std::vector<Task> &tasks,
                                    RdmaCopyCtrlBlock *ctrl_block,
                                    int32_t &num_done) {
    segment_id_t target_segment_id = engine_.getSegmentId(target_segment_name);
    if (target_segment_id < 0) return -1;

    std::lock_guard<std::mutex> regions_lock(region_mgr_.regions_mutex_);

    // num_done is a lower bound watermark: if task_idx < num_done, it is done
    // and its batch_id is INVALID_BATCH; otherwise, it may or may not be done
    // (it may be done because another task has waited on it for its buffer)
    num_done = 0;
    int32_t last_updated_num_done = 0;
    // submit a progress update if
    //      (num_done - last_updated_num_done) >= update_freq
    constexpr int32_t update_freq = 5;  // for now: every 5 requests

    batch_id_t progress_batch_id = INVALID_BATCH;

    int rc, status;

    // pipeline: start copy for task N+1 before submitting RDMA for task N
    // this allows task N's copy to overlap with task N-1's RDMA transfer
    if (!tasks.empty()) {
        rc = startTaskCopy(tasks, 0, target_segment_id);
        if (rc) goto cleanup;
    }

    // loop where task_idx is the index for RDMA submission
    for (size_t task_idx = 0; task_idx < tasks.size(); ++task_idx) {
        // submit RDMA for current task (whose copy was started earlier)
        rc = submitTaskRdma(tasks[task_idx], target_segment_id);
        if (rc) goto cleanup;

        // start copy for next task (if exists) to overlap with current RDMA
        if (task_idx + 1 < tasks.size()) {
            rc = startTaskCopy(tasks, task_idx + 1, target_segment_id);
            if (rc) goto cleanup;
        }

        // check if any waiting tasks are done
        for (size_t i = num_done; i <= task_idx; ++i) {
            if (tasks[i].batch_id != INVALID_BATCH) {
                status = pollBatch(tasks[i].batch_id);
                if (status == STATUS_WAITING) break;
                freeBatch(tasks[i].batch_id);
                releaseBuffer(tasks[i]);  // release buffer regardless of status
                if (status != STATUS_COMPLETED) {
                    LOG(ERROR) << "Transfer error for task_idx=" << i;
                    goto cleanup;
                }
            }
            ++num_done;
        }

        // update remote progress if we have made enough progress
        if ((num_done - last_updated_num_done) >= update_freq) {
            rc = tryUpdateRemoteProgress(
                progress_batch_id, last_updated_num_done, num_done, ctrl_block,
                target_segment_id, target_progress_addr);
            if (rc) {
                LOG(ERROR) << "Fail to update the progress";
                goto cleanup;
            }
        }
    }

    // wait for all tasks to complete (should be very few)
    // if all requests are on the same device, should be only 1~2 tasks
    for (size_t task_idx = num_done; task_idx < tasks.size(); ++task_idx) {
        status = waitTask(tasks[task_idx]);
        if (status != STATUS_COMPLETED) {
            LOG(ERROR) << "Transfer error for task_idx=" << task_idx;
            goto cleanup;
        }
        ++num_done;
    }

    /**
     Ordering guarantee: progress counter update must be finished before return
     any value from the socket. In other words, once received a int32_t from the
     socket, the client can safely assume there will be no more update to the
     progress counter.
     */

    // wait for the previous progress update to complete
    if (progress_batch_id != INVALID_BATCH) {
        status = waitBatch(progress_batch_id);
        if (status != STATUS_COMPLETED) {
            LOG(ERROR) << "Error wait for progress update completion";
            goto cleanup;
        }
    }
    rc = tryUpdateRemoteProgress(progress_batch_id, last_updated_num_done,
                                 num_done, ctrl_block, target_segment_id,
                                 target_progress_addr);
    if (rc) {
        LOG(ERROR) << "Fail to update the progress";
        goto cleanup;
    }
    if (progress_batch_id != INVALID_BATCH) {
        status = waitBatch(progress_batch_id);
        if (status != STATUS_COMPLETED) {
            LOG(ERROR) << "Error wait for progress update completion";
            goto cleanup;
        }
    }

    // LOG(INFO) << "Completed transfer request: " << num_done << " tasks";
    return 0;

cleanup:
    // Clean up waiting tasks if any (only occur if not success)
    for (size_t task_idx = num_done; task_idx < tasks.size(); ++task_idx) {
        status = waitTask(tasks[task_idx]);
        if (status != STATUS_COMPLETED)
            LOG(ERROR) << "Failed to wait for task completion";
    }
    return -1;
}

RdmaCopyBackend::BufferPair *RdmaCopyBackend::allocBufferPair(
    LocId loc_id, const std::string &location, size_t size) {
    bool is_cuda = loc_id.cuda_device >= 0;
    size_t total_size = 2 * size;  // allocate one contiguous buffer w/ 2*size
    char *buffer_base = nullptr;
    if (is_cuda) {
#ifdef USE_CUDA
        // Use CUDA device ID from loc_id
        int device_id = loc_id.cuda_device;
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Failed to set CUDA device ") +
                                     std::to_string(device_id) + ": " +
                                     cudaGetErrorString(err));
        }

        err = cudaMalloc(&buffer_base, total_size);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("Failed to allocate GPU memory on device ") +
                std::to_string(device_id) + ": " + cudaGetErrorString(err));
        }
#else
        throw std::runtime_error(
            "GPU memory requested but CUDA support not compiled");
#endif
    } else {
        buffer_base = new char[total_size];
    }

    // Register the entire contiguous buffer with RDMA
    int rc = ::registerLocalMemory(
        engine_.getEngine(), buffer_base, total_size, location.c_str(),
        /*remote_accessible*/ true, /*remote_atomic*/ false);
    if (rc) {
        if (is_cuda) {
#ifdef USE_CUDA
            cudaFree(buffer_base);
#endif
        } else {
            delete[] buffer_base;
        }
        return nullptr;
    }

    BufferPair *pair = new BufferPair(buffer_base, size, is_cuda);

    // LOG(INFO) << "Allocated buffer pair of size " << size << " for location "
    //           << location << " (total=" << total_size << ")";
    return pair;
}

void RdmaCopyBackend::freeBufferPair(BufferPair *pair) {
    if (!pair) return;

    char *buffer_base = pair->buffers[0].addr;
    ::unregisterLocalMemory(engine_.getEngine(), buffer_base);

    if (pair->is_cuda) {
#ifdef USE_CUDA
        cudaFree(buffer_base);
#endif
    } else {
        delete[] buffer_base;
    }

    delete pair;
}

int RdmaCopyBackend::waitTask(Task &task) {
    if (task.batch_id == INVALID_BATCH) return 0;

    int status = waitBatch(task.batch_id);
    releaseBuffer(task);
    return status;
}

// Poll if the given progress_batch_id has finished; if so, submit another
// progress update via atomic fetch-add, which will update progress_batch_id
// and last_updated_num_done
int RdmaCopyBackend::tryUpdateRemoteProgress(batch_id_t &progress_batch_id,
                                             int32_t &last_updated_num_done,
                                             int32_t num_done,
                                             RdmaCopyCtrlBlock *ctrl_block,
                                             segment_id_t target_segment_id,
                                             uint64_t target_progress_addr) {
    int status, rc;
    if (progress_batch_id != INVALID_BATCH) {  // check the last progress update
        status = pollBatch(progress_batch_id);
        if (status == STATUS_WAITING) return 0;  // not done
        // done: completed or error
        freeBatch(progress_batch_id);
        if (status != STATUS_COMPLETED) return -1;
    }

    // no new update
    if (num_done == last_updated_num_done) return 0;

    // submit another batch for progress update
    transfer_request_t progress_req = {
        .opcode = OPCODE_ATOMIC_FETCH_ADD,
        .operand = static_cast<int32_t>(num_done - last_updated_num_done),
        .source = (void *)&(ctrl_block->progress_counter),
        .target_id = target_segment_id,
        .target_offset = target_progress_addr,
        .length = 8,
    };

    rc = submitBatch(progress_batch_id, progress_req);
    if (rc) return rc;

    last_updated_num_done = num_done;
    return 0;
}

int RdmaCopyBackend::acquireBuffer(BufferPair &buffer_pair,
                                   std::vector<Task> &tasks, size_t task_idx) {
    int buffer_idx = buffer_pair.selectNextBuffer();
    BufferPair::Buffer &buffer = buffer_pair.buffers[buffer_idx];
    int buffer_used_by_task_idx = buffer.user;

    // if buffer is in use by a previous task, wait for it to complete
    if (buffer_used_by_task_idx >= 0) {
        assert(buffer_used_by_task_idx < static_cast<int>(task_idx));
        int status = waitTask(tasks[buffer_used_by_task_idx]);
        if (status != STATUS_COMPLETED) {
            LOG(ERROR) << "Failed to wait for previous task "
                       << buffer_used_by_task_idx << " on buffer "
                       << buffer_idx;
            return -1;
        }
        // waitTask has released the buffer
    }

    // mark buffer as owned by current task and update task metadata
    buffer.user = task_idx;
    tasks[task_idx].buffer_pair = &buffer_pair;
    tasks[task_idx].buffer_idx = buffer_idx;
    return 0;
}

void RdmaCopyBackend::releaseBuffer(Task &task) {
    task.buffer_pair->buffers[task.buffer_idx].user = -1;
    task.buffer_pair = nullptr;
    task.buffer_idx = -1;
}

// memcpyAsync is expected to succeed because the given src and dst must have
// been validated; if an error occurs, it is our own fault, not due to invalid
// input; throw the error instead of gracefully handling
void RdmaCopyBackend::memcpyAsync(const void *src, size_t size,
                                  BufferPair &buffer_pair, int buffer_idx) {
    assert(buffer_idx >= 0 && buffer_idx < 2);
    BufferPair::Buffer &buffer = buffer_pair.buffers[buffer_idx];
    void *dst = buffer.addr;

    if (buffer_pair.is_cuda) {
#ifdef USE_CUDA
        cudaError_t err = cudaMemcpyAsync(
            dst, src, size, cudaMemcpyDeviceToDevice, buffer.cuda_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") +
                                     cudaGetErrorString(err));
        }

        // record event after async copy (for later synchronization)
        err = cudaEventRecord(buffer.copy_done_event, buffer.cuda_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaEventRecord failed: ") +
                                     cudaGetErrorString(err));
        }
#else
        throw std::runtime_error(
            "GPU memory copy requested but CUDA support not compiled");
#endif
    } else {
        memcpy(dst, src, size);
    }
}

int RdmaCopyBackend::startTaskCopy(std::vector<Task> &tasks, size_t task_idx,
                                   int target_segment_id) {
    Task &task = tasks[task_idx];

    // validate source address
    Region *region = region_mgr_.getRegion(task.source_addr, task.length);
    if (!region) {
        LOG(ERROR) << "Source address 0x" << std::hex << task.source_addr
                   << " not in registered copiable regions";
        return -1;
    }

    BufferPair &buffer_pair = getBufferPair(region->loc_id);
    assert(buffer_pair.size >= task.length);

    int rc = acquireBuffer(buffer_pair, tasks, task_idx);
    if (rc) return rc;

    // start async copy and record event (if CUDA)
    memcpyAsync(task.source_addr, task.length, buffer_pair, task.buffer_idx);

    return 0;
}

int RdmaCopyBackend::submitTaskRdma(Task &task, int target_segment_id) {
    assert(task.buffer_pair && task.buffer_idx >= 0);
    BufferPair::Buffer &buffer = task.buffer_pair->buffers[task.buffer_idx];

#ifdef USE_CUDA
    // wait for CUDA copy to complete before RDMA reads the buffer
    if (task.buffer_pair->is_cuda) {
        cudaError_t err = cudaEventSynchronize(buffer.copy_done_event);
        if (err != cudaSuccess) {
            LOG(ERROR) << "Failed to sync CUDA event: "
                       << cudaGetErrorString(err);
            releaseBuffer(task);
            return -1;
        }
    }
#endif

    // submit RDMA write from buffer to remote target
    transfer_request_t req = {
        .opcode = OPCODE_WRITE,
        .operand = 0,
        .source = buffer.addr,
        .target_id = target_segment_id,
        .target_offset = task.target_addr,
        .length = task.length,
    };

    int rc = submitBatch(task.batch_id, req);
    if (rc) {
        LOG(ERROR) << "Failed to submit RDMA write";
        releaseBuffer(task);
        return rc;
    }
    return 0;
}

int RdmaCopyBackend::submitBatch(batch_id_t &batch_id,
                                 transfer_request_t &req) {
    batch_id = ::allocateBatchID(engine_.getEngine(), 1);
    int rc = ::submitTransfer(engine_.getEngine(), batch_id, &req, 1);
    if (rc) freeBatch(batch_id);  // failed; reset
    return rc;
}

void RdmaCopyBackend::freeBatch(batch_id_t &batch_id) {
    assert(batch_id != INVALID_BATCH);
    ::freeBatchID(engine_.getEngine(), batch_id);
    batch_id = INVALID_BATCH;
}

int RdmaCopyBackend::pollBatch(batch_id_t batch_id) {
    assert(batch_id != INVALID_BATCH);
    [[maybe_unused]] int rc;
    transfer_status_t status;
    rc = ::getTransferStatus(engine_.getEngine(), batch_id, 0, &status);
    assert(rc == 0);
    return status.status;
}

// Wait until the given batch (size=1) is done and then free the batch; will
// update batch_id to INVALID_BATCH; return the status
int RdmaCopyBackend::waitBatch(batch_id_t &batch_id) {
    assert(batch_id != INVALID_BATCH);

    int status;
    do {
        status = pollBatch(batch_id);
    } while (status == STATUS_WAITING);

    freeBatch(batch_id);
    return status;
}

RdmaCopyBackend::BufferPair::BufferPair(char *buffer_base, size_t size,
                                        bool is_cuda)
    : size(size), is_cuda(is_cuda) {
    // Initialize buffer addresses and users
    buffers[0].addr = buffer_base;
    buffers[1].addr = buffer_base + size;
    buffers[0].user = -1;
    buffers[1].user = -1;

#ifdef USE_CUDA
    buffers[0].cuda_stream = nullptr;
    buffers[1].cuda_stream = nullptr;
    buffers[0].copy_done_event = nullptr;
    buffers[1].copy_done_event = nullptr;

    if (is_cuda) {  // Assume: cuda device has been set before initialization

        // Get the priority range and set stream to highest priority
        int leastPriority, greatestPriority;
        cudaError_t err =
            cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("Failed to get CUDA stream priority range: ") +
                cudaGetErrorString(err));
        }

        // Create streams and events for both buffers with highest priority
        for (int i = 0; i < 2; ++i) {
            err = cudaStreamCreateWithPriority(&buffers[i].cuda_stream,
                                               cudaStreamNonBlocking,
                                               greatestPriority);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Failed to create CUDA stream: ") +
                    cudaGetErrorString(err));
            }

            err = cudaEventCreate(&buffers[i].copy_done_event);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Failed to create CUDA event: ") +
                    cudaGetErrorString(err));
            }
        }
    }
#endif
}

RdmaCopyBackend::BufferPair::~BufferPair() {
#ifdef USE_CUDA
    if (is_cuda) {
        for (int i = 0; i < 2; ++i) {
            if (buffers[i].cuda_stream)
                cudaStreamDestroy(buffers[i].cuda_stream);
            if (buffers[i].copy_done_event)
                cudaEventDestroy(buffers[i].copy_done_event);
        }
    }
#endif
}
