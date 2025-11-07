#include "flex_transfer_engine.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "transfer_engine_c.h"
#include "util.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

FlexTransferEngine::FlexTransferEngine(const std::string &metadata_conn_string,
                                       const std::string &local_server_name,
                                       bool enable_copy,
                                       const std::string &ctrl_block_location)
    : local_server_name_(local_server_name),
      enable_copy_(enable_copy),
      ctrl_block_location_(ctrl_block_location),
      copy_server_(*this),
      copy_client_(*this) {
    if (ctrl_block_location.find("cuda:") == 0) {
        throw std::invalid_argument(
            "ctrl_block_location must not be on CUDA: " + ctrl_block_location);
    }

    // Create the underlying TransferEngine
    engine_ = ::createTransferEngine(metadata_conn_string.c_str(),
                                     local_server_name.c_str(),
                                     /*unused*/ local_server_name.c_str(),
                                     /*unused*/ 12345, /*auto_discover*/ true);
    if (!engine_) throw std::runtime_error("Failed to create TransferEngine");

    // Start TCP listener only if enable_copy_ is true
    if (enable_copy_) copy_server_.startListener();
}

FlexTransferEngine::~FlexTransferEngine() {
    if (enable_copy_)
        copy_server_.cleanup();  // release RDMA-registered buffers

    {
        std::lock_guard<std::mutex> segment_lock(segment_cache_mutex_);
        for (auto &[segment_name, segment_id] : segment_cache_)
            ::closeSegment(engine_, segment_id);
        // segment_cache_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(ctrl_block_mutex_);
        for (CopyCtrlBlock *ctrl_block : copy_ctrl_block_cache_) {
            ::unregisterLocalMemory(engine_, ctrl_block);
            delete ctrl_block;
        }
        // copy_ctrl_block_cache_.clear();
    }

    ::destroyTransferEngine(engine_);
}

int FlexTransferEngine::registerLocalMemory(uintptr_t addr, size_t length,
                                            const std::string &location,
                                            bool remote_accessible,
                                            bool remote_atomic,
                                            bool force_direct) {
    // do actual RDMA registration
    if (force_direct || !enable_copy_) {
        return ::registerLocalMemory(engine_, reinterpret_cast<void *>(addr),
                                     length, location.c_str(),
                                     remote_accessible, remote_atomic);
    } else {  // register for copy-based transfer
        return copy_server_.registerLocalMemory(reinterpret_cast<void *>(addr),
                                                length, location);
    }
}

int FlexTransferEngine::unregisterLocalMemory(uintptr_t addr,
                                              bool force_direct) {
    if (force_direct || !enable_copy_) {
        return ::unregisterLocalMemory(engine_, reinterpret_cast<void *>(addr));
    } else {
        return copy_server_.unregisterLocalMemory(
            reinterpret_cast<void *>(addr));
    }
}

int FlexTransferEngine::registerLocalMemoryBatch(
    std::vector<buffer_entry_t> &buffer_list, const std::string &location,
    bool force_direct) {
    if (force_direct || !enable_copy_) {
        return ::registerLocalMemoryBatch(engine_, buffer_list.data(),
                                          buffer_list.size(), location.c_str());
    } else {
        return copy_server_.registerLocalMemoryBatch(buffer_list, location);
    }
}

int FlexTransferEngine::unregisterLocalMemoryBatch(
    std::vector<uintptr_t> &addr_list, bool force_direct) {
    if (force_direct || !enable_copy_) {
        return ::unregisterLocalMemoryBatch(
            engine_, reinterpret_cast<void **>(addr_list.data()),
            addr_list.size());
    } else {
        return copy_server_.unregisterLocalMemoryBatch(addr_list);
    }
}

segment_id_t FlexTransferEngine::getSegmentId(const std::string &segment_name) {
    std::lock_guard<std::mutex> lock(segment_cache_mutex_);
    auto it = segment_cache_.find(segment_name);
    if (it != segment_cache_.end()) return it->second;

    segment_id_t segment_id = ::openSegment(engine_, segment_name.c_str());
    if (segment_id < 0) return segment_id;  // error
    segment_cache_[segment_name] = segment_id;
    return segment_id;
}

CopyCtrlBlock *FlexTransferEngine::acquireCopyCtrlBlock() {
    std::lock_guard<std::mutex> lock(ctrl_block_mutex_);

    // Try to get from cache first
    if (!copy_ctrl_block_cache_.empty()) {
        CopyCtrlBlock *ctrl_block = copy_ctrl_block_cache_.back();
        copy_ctrl_block_cache_.pop_back();
        ctrl_block->progress_counter.store(0, std::memory_order_release);
        return ctrl_block;
    }

    // Cache is empty, allocate a new one
    CopyCtrlBlock *ctrl_block = new CopyCtrlBlock();

    // Register it with RDMA using the specified location
    int rc = ::registerLocalMemory(engine_, ctrl_block, sizeof(CopyCtrlBlock),
                                   ctrl_block_location_.c_str(),
                                   /*remote_accessible*/ true,
                                   /*remote_atomic*/ true);
    if (rc) {
        delete ctrl_block;
        return nullptr;
    }
    return ctrl_block;
}

void FlexTransferEngine::releaseCopyCtrlBlock(CopyCtrlBlock *ctrl_block) {
    if (!ctrl_block) return;
    std::lock_guard<std::mutex> lock(ctrl_block_mutex_);
    copy_ctrl_block_cache_.emplace_back(ctrl_block);
}
