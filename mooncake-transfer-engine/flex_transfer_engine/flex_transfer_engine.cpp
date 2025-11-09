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

#include "copy_common.h"
#include "transfer_engine_c.h"
#include "util.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

FlexTransferEngine::FlexTransferEngine(const std::string &metadata_conn_string,
                                       const std::string &local_server_name,
                                       bool enable_copy,
                                       const std::string &ctrl_block_location)
    : enable_copy_(enable_copy),
      ctrl_block_location_(ctrl_block_location),
      copy_server_(*this),
      copy_client_(local_server_name) {
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
        for (RDMACopyCtrlBlock *ctrl_block : ctrl_block_cache_) {
            ::unregisterLocalMemory(engine_, ctrl_block);
            delete ctrl_block;
        }
        // ctrl_block_cache_.clear();
    }

    ::destroyTransferEngine(engine_);
}

int FlexTransferEngine::registerLocalMemory(uintptr_t addr, size_t length,
                                            const std::string &location,
                                            bool remote_accessible,
                                            bool remote_atomic,
                                            TransferMode mode) {
    if (mode == TransferMode::Auto)
        mode = enable_copy_ ? TransferMode::Copy : TransferMode::Direct;
    // do actual RDMA registration
    int rc = 0;
    if (mode & TransferMode::Direct) {
        rc = ::registerLocalMemory(engine_, reinterpret_cast<void *>(addr),
                                   length, location.c_str(), remote_accessible,
                                   remote_atomic);
        if (rc) return rc;
    }
    if (mode & TransferMode::Copy) {
        if (!enable_copy_)
            throw std::invalid_argument("Copy mode was not enabled");
        rc = copy_server_.registerLocalMemory(reinterpret_cast<void *>(addr),
                                              length, location);
        if (rc) return rc;
    }
    return 0;
}

int FlexTransferEngine::unregisterLocalMemory(uintptr_t addr,
                                              TransferMode mode) {
    if (mode == TransferMode::Auto)
        mode = enable_copy_ ? TransferMode::Copy : TransferMode::Direct;

    // Note if an address is registered multiple times, direct mode will expect
    // the exact number of unregister but the copy mode only expect one.
    // If multiple unregister are called, they will still be safely done, except
    // the copy mode will return -1 for unregister other than the first one.
    // That's also why we keep copy mode unregister after the diect mode, so
    // that the return code from copy mode won't affect direct mode when both
    // modes are enabled.

    int rc = 0;
    if (mode & TransferMode::Direct) {
        rc = ::unregisterLocalMemory(engine_, reinterpret_cast<void *>(addr));
        if (rc) return rc;
    }
    // Note if there are duplicated address, copy_server_.unregisterLocalMemory
    // can return an error
    if (mode & TransferMode::Copy) {
        if (!enable_copy_)
            throw std::invalid_argument("Copy mode was not enabled");
        rc = copy_server_.unregisterLocalMemory(reinterpret_cast<void *>(addr));
        if (rc) return rc;
    }
    return 0;
}

int FlexTransferEngine::registerLocalMemoryBatch(
    std::vector<buffer_entry_t> &buffer_list, const std::string &location,
    TransferMode mode) {
    if (mode == TransferMode::Auto)
        mode = enable_copy_ ? TransferMode::Copy : TransferMode::Direct;

    int rc = 0;
    if (mode & TransferMode::Direct) {
        rc = ::registerLocalMemoryBatch(engine_, buffer_list.data(),
                                        buffer_list.size(), location.c_str());
        if (rc) return rc;
    }
    if (mode & TransferMode::Copy) {
        if (!enable_copy_)
            throw std::invalid_argument("Copy mode was not enabled");
        rc = copy_server_.registerLocalMemoryBatch(buffer_list, location);
        if (rc) return rc;
    }
    return 0;
}

int FlexTransferEngine::unregisterLocalMemoryBatch(
    std::vector<uintptr_t> &addr_list, TransferMode mode) {
    if (mode == TransferMode::Auto)
        mode = enable_copy_ ? TransferMode::Copy : TransferMode::Direct;

    // Note if an address is registered multiple times, direct mode will expect
    // the exact number of unregister but the copy mode only expect one.
    // If multiple unregister are called, they will still be safely done, except
    // the copy mode will return -1 for unregister other than the first one.
    // That's also why we keep copy mode unregister after the diect mode, so
    // that the return code from copy mode won't affect direct mode when both
    // modes are enabled.

    int rc = 0;
    if (mode & TransferMode::Direct) {
        rc = ::unregisterLocalMemoryBatch(
            engine_, reinterpret_cast<void **>(addr_list.data()),
            addr_list.size());
        if (rc) return rc;
    }
    if (mode & TransferMode::Copy) {
        if (!enable_copy_)
            throw std::invalid_argument("Copy mode was not enabled");
        rc = copy_server_.unregisterLocalMemoryBatch(addr_list);
        if (rc) return rc;
    }
    return rc;
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

RDMACopyCtrlBlock *FlexTransferEngine::allocRDMACopyCtrlBlock() {
    std::lock_guard<std::mutex> lock(ctrl_block_mutex_);

    // Try to get from cache first
    if (!ctrl_block_cache_.empty()) {
        RDMACopyCtrlBlock *ctrl_block = ctrl_block_cache_.back();
        ctrl_block_cache_.pop_back();
        ctrl_block->progress_counter.store(0, std::memory_order_release);
        return ctrl_block;
    }

    // Cache is empty, allocate a new one
    RDMACopyCtrlBlock *ctrl_block = new RDMACopyCtrlBlock();

    // Register it with RDMA using the specified location
    int rc =
        ::registerLocalMemory(engine_, ctrl_block, sizeof(RDMACopyCtrlBlock),
                              ctrl_block_location_.c_str(),
                              /*remote_accessible*/ true,
                              /*remote_atomic*/ true);
    if (rc) {
        delete ctrl_block;
        return nullptr;
    }
    return ctrl_block;
}

void FlexTransferEngine::freeRDMACopyCtrlBlock(RDMACopyCtrlBlock *ctrl_block) {
    if (!ctrl_block) return;
    std::lock_guard<std::mutex> lock(ctrl_block_mutex_);
    ctrl_block_cache_.emplace_back(ctrl_block);
}
