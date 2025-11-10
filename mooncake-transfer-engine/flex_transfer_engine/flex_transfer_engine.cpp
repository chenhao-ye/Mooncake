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
      region_mgr_(),
      rdma_copy_backend_(*this, region_mgr_, ctrl_block_location),
      tcp_copy_backend_(*this, region_mgr_),
      copy_server_(*this, region_mgr_, rdma_copy_backend_, tcp_copy_backend_),
      copy_client_(local_server_name, rdma_copy_backend_, tcp_copy_backend_) {
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
