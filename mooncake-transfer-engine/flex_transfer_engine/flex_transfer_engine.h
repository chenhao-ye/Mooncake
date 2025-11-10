#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "copy_client.h"
#include "copy_common.h"
#include "copy_server.h"
#include "flex_batch.h"
#include "transfer_engine.h"
#include "transfer_engine_c.h"

class FlexTransferEngine;

/**
 * Mark the target transfer mode for registration.
 * Note these are flag enum, where `Direct | Copy` is acceptable.
 */
enum class RegMode : uint8_t {
    // let the transfer engine decide
    Auto = 0,
    // directly register with RDMA NICs, which enable zero-copy direct transfer
    Direct = 1 << 0,
    // record in copiable_regions for copy-based transfer
    Copy = 1 << 1,  // via either RDMA or TCP
};

inline RegMode operator|(RegMode a, RegMode b) {
    return static_cast<RegMode>(
        static_cast<std::underlying_type_t<RegMode>>(a) |
        static_cast<std::underlying_type_t<RegMode>>(b));
}

inline bool operator&(RegMode a, RegMode b) {
    return static_cast<std::underlying_type_t<RegMode>>(a) &
           static_cast<std::underlying_type_t<RegMode>>(b);
}

/**
 * FlexTransferEngine is a flexible wrapper on top of TransferEngine that
 * supports both direct RDMA transfers and copy-based transfers.
 *
 * Direct Mode: Register the user-specified memory directly with RDMA NICs;
 * remote FlexTransferEngine can directly read/write these memory without any
 * CPU involvement.
 *
 * Copy Mode: A background TCP listener thread will accept the read requests
 * (write is not supported yet) and send data back via RDMA or TCP. Note the
 * Copy Mode only implies the copying happens on the server side; the
 * client-side has no copy if using RDMA to receive the data.
 *
 * To use Direct Mode: the server registers memory with RegMode::Direct; the
 * client registers memory with RegMode::Direct and submits with is_direct=true.
 *
 * To use Copy Mode (RDMA): the server registers memory with RegMode::Copy; the
 * client registers memory with RegMode::Direct and submit with is_direct=false
 * and use_rdma=true.
 *
 * To use Copy Mode (TCP): the server registers memory with RegMode::Copy; the
 * client registers memory with RegMode::Copy and submits with is_direct=false
 * and use_rdma=false.
 *
 * Note registration supports dual-mode (RegMode::Direct | RegMode::Copy).
 */
class FlexTransferEngine {
   public:
    /**
     * Constructor.
     * @param metadata_conn_string Connection string for metadata server
     * @param local_server_name Local server name
     * @param rdma_ctrl_block_location Location for RdmaCopyCtrlBlock
     * registration (e.g., "cpu:0")
     * @param copy_server_enabled If true, starts TCP listener for copy-based
     * transfer; otherwise, only direct RDMA transfer is supported.
     * @param default_reg_mode Default registration mode when RegMode::Auto is
     * provided for (un)registerLocalMemory(Batch); if RegMode::Auto is provided
     * here, will infer: copy_server_enabled ? RegMode::Copy : RegMode::Direct,
     * which is desired if no TCP copy transfer is expected.
     */
    explicit FlexTransferEngine(const std::string &metadata_conn_string,
                                const std::string &local_server_name,
                                const std::string &rdma_ctrl_block_location,
                                bool copy_server_enabled,
                                RegMode default_reg_mode = RegMode::Auto);

    ~FlexTransferEngine();

    int registerLocalMemory(uintptr_t addr, size_t length,
                            const std::string &location, bool remote_accessible,
                            bool remote_atomic, RegMode mode = RegMode::Auto);

    int unregisterLocalMemory(uintptr_t addr, RegMode mode = RegMode::Auto);

    int registerLocalMemoryBatch(std::vector<buffer_entry_t> &buffer_list,
                                 const std::string &location,
                                 RegMode mode = RegMode::Auto);

    int registerLocalMemoryBatch(MemoryBatch &memory_batch,
                                 RegMode mode = RegMode::Auto) {
        for (auto &[location, buffers] : memory_batch.location_buffers_map) {
            int rc = registerLocalMemoryBatch(buffers, location, mode);
            if (rc) return rc;
        }
        return 0;
    }

    int unregisterLocalMemoryBatch(std::vector<uintptr_t> &addr_list,
                                   RegMode mode = RegMode::Auto);

    int syncSegmentCache() { return ::syncSegmentCache(engine_); }

    const std::string &getCopyServerUrl() const {
        return copy_server_.getServerUrl();
    }

    // for CopyServer and FlexBatch
    transfer_engine_t getEngine() { return engine_; }

    CopyClient &getCopyClient() { return copy_client_; }

    segment_id_t getSegmentId(const std::string &segment_name);

   private:
    // if enabled, the memory register with RegMode::Copy is accessible to
    // the remote FlexTransferEngine via copy-based transfer.
    // if disabled, the memory register with RegMode::Copy can only be used
    // for CopyClient.
    const bool copy_server_enabled_;
    const RegMode default_reg_mode_;

    transfer_engine_t engine_;

    // Segment cache (segment_name -> segment_id)
    // Used when acting as copy engine to cache opened segments
    // Append-only; will never remove entries
    std::unordered_map<std::string, segment_id_t> segment_cache_;
    std::mutex segment_cache_mutex_;

    RegionMgr region_mgr_;

    RdmaCopyBackend rdma_copy_backend_;
    TcpCopyBackend tcp_copy_backend_;

    // TCP listener and worker (only used when copy_server_enabled_ is true)
    CopyServer copy_server_;

    // Client when talking to a copy_server
    CopyClient copy_client_;
};
