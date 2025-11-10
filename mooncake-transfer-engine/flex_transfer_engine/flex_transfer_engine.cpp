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
    : copy_server_enabled_(enable_copy),
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

    // Start TCP listener only if copy_server_enabled_ is true
    if (copy_server_enabled_) copy_server_.startListener();
}

FlexTransferEngine::~FlexTransferEngine() {
    if (copy_server_enabled_) copy_server_.stopListener();

    {
        std::lock_guard<std::mutex> lock(region_mgr_.regions_mutex_);
        // release RDMA-registered buffers before engine_ destruction
        rdma_copy_backend_.cleanup();
        tcp_copy_backend_.cleanup();
    }

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
                                            bool remote_atomic, RegMode mode) {
    if (mode == RegMode::Auto)
        mode = copy_server_enabled_ ? RegMode::Copy : RegMode::Direct;
    int rc = 0;
    if (mode & RegMode::Direct) {  // do actual RDMA registration
        rc = ::registerLocalMemory(engine_, reinterpret_cast<void *>(addr),
                                   length, location.c_str(), remote_accessible,
                                   remote_atomic);
        if (rc) return rc;
    }
    if (mode & RegMode::Copy) {
        std::lock_guard<std::mutex> regions_lock(region_mgr_.regions_mutex_);

        LocId loc_id = region_mgr_.getLocId(location);
        region_mgr_.addRegion(reinterpret_cast<void *>(addr), length, loc_id);

        if (copy_server_enabled_) {
            rc = rdma_copy_backend_.prepareBufferPair(loc_id, location, length);
            if (rc) {
                region_mgr_.removeRegion(reinterpret_cast<void *>(addr));
                return -1;
            }
        }
        std::cerr << "Registered memory at " << addr << " size " << length
                  << " for location " << location << std::endl;
    }
    return 0;
}

int FlexTransferEngine::unregisterLocalMemory(uintptr_t addr, RegMode mode) {
    if (mode == RegMode::Auto)
        mode = copy_server_enabled_ ? RegMode::Copy : RegMode::Direct;

    // Note if an address is registered multiple times, direct mode will expect
    // the exact number of unregister but the copy mode only expect one.
    // If multiple unregister are called, they will still be safely done, except
    // the copy mode will return -1 for unregister other than the first one.
    // That's also why we keep copy mode unregister after the diect mode, so
    // that the return code from copy mode won't affect direct mode when both
    // modes are enabled.

    int rc = 0;
    if (mode & RegMode::Direct) {
        rc = ::unregisterLocalMemory(engine_, reinterpret_cast<void *>(addr));
        if (rc) return rc;
    }
    // Note if there are duplicated address, copy_server_.unregisterLocalMemory
    // can return an error
    if (mode & RegMode::Copy) {
        std::lock_guard<std::mutex> regions_lock(region_mgr_.regions_mutex_);
        rc = region_mgr_.removeRegion(reinterpret_cast<void *>(addr));
        if (rc) return rc;
    }
    return 0;
}

int FlexTransferEngine::registerLocalMemoryBatch(
    std::vector<buffer_entry_t> &buffer_list, const std::string &location,
    RegMode mode) {
    if (mode == RegMode::Auto)
        mode = copy_server_enabled_ ? RegMode::Copy : RegMode::Direct;

    int rc = 0;
    if (mode & RegMode::Direct) {
        rc = ::registerLocalMemoryBatch(engine_, buffer_list.data(),
                                        buffer_list.size(), location.c_str());
        if (rc) return rc;
    }
    if (mode & RegMode::Copy) {
        std::lock_guard<std::mutex> regions_lock(region_mgr_.regions_mutex_);
        LocId loc_id = region_mgr_.getLocId(location);
        size_t max_size = 0;

        for (const auto &entry : buffer_list) {
            if (entry.length > max_size) max_size = entry.length;
            region_mgr_.addRegion(entry.addr, entry.length, loc_id);
        }

        if (copy_server_enabled_) {
            rc = rdma_copy_backend_.prepareBufferPair(loc_id, location,
                                                      max_size);
            if (rc) {
                for (const auto &entry : buffer_list)
                    region_mgr_.removeRegion(entry.addr);
                return -1;
            }
        }

        std::cerr << "Registered " << buffer_list.size()
                  << " buffers for location " << location << ", max size "
                  << max_size << std::endl;
    }
    return 0;
}

int FlexTransferEngine::unregisterLocalMemoryBatch(
    std::vector<uintptr_t> &addr_list, RegMode mode) {
    if (mode == RegMode::Auto)
        mode = copy_server_enabled_ ? RegMode::Copy : RegMode::Direct;

    // Note if an address is registered multiple times, direct mode will expect
    // the exact number of unregister but the copy mode only expect one.
    // If multiple unregister are called, they will still be safely done, except
    // the copy mode will return -1 for unregister other than the first one.
    // That's also why we keep copy mode unregister after the diect mode, so
    // that the return code from copy mode won't affect direct mode when both
    // modes are enabled.

    int rc = 0;
    if (mode & RegMode::Direct) {
        rc = ::unregisterLocalMemoryBatch(
            engine_, reinterpret_cast<void **>(addr_list.data()),
            addr_list.size());
        if (rc) return rc;
    }
    if (mode & RegMode::Copy) {
        bool all_success = true;
        std::lock_guard<std::mutex> regions_lock(region_mgr_.regions_mutex_);
        for (uintptr_t addr : addr_list) {
            rc = region_mgr_.removeRegion(reinterpret_cast<void *>(addr));
            if (rc) all_success = false;  // not found, but will continue
        }
        if (!all_success) return -1;
    }
    return 0;
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
