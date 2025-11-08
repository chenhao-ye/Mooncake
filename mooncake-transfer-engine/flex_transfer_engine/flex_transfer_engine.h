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
#include "copy_server.h"
#include "copy_transfer.h"
#include "flex_batch.h"
#include "transfer_engine.h"
#include "transfer_engine_c.h"

// Forward declaration
class FlexTransferEngine;

/**
 * Mark the target transfer mode for registration.
 * Note these are flag enum where `Direct | Copy` is acceptable.
 */
enum class TransferMode : uint8_t {
    // let the transfer engine decide: enable_copy_ ? Copy : Direct
    Auto = 0,
    // directly register with RDMA NICs, which enable zero-copy direct transfer
    Direct = 1 << 0,
    // record in copiable_regions that can serve via copy-based transfer
    Copy = 1 << 1,  // via either RDMA write or TCP
};

inline TransferMode operator|(TransferMode a, TransferMode b) {
    return static_cast<TransferMode>(
        static_cast<std::underlying_type_t<TransferMode>>(a) |
        static_cast<std::underlying_type_t<TransferMode>>(b));
}

inline bool operator&(TransferMode a, TransferMode b) {
    return static_cast<std::underlying_type_t<TransferMode>>(a) &
           static_cast<std::underlying_type_t<TransferMode>>(b);
}

/**
 * FlexTransferEngine is a flexible wrapper on top of TransferEngine that
 * supports both direct RDMA transfers and copy-based transfers.
 *
 * Direct Mode: Register the user-specified memory directly with RDMA NICs;
 * other FlexTransferEngine can directly read/write these memory without
 * CPU involvement.
 *
 * Copy Mode: Only register some buffers with RDMA NICs. A background TCP
 * listener thread will accept the requests and copy dataf from user-specified
 * memory into the buffers and submit RDMA requests.
 *
 * Copy Mode is enabled via flag `enable_copy_` at construction time.
 */
class FlexTransferEngine {
   public:
    /**
     * Constructor.
     * @param metadata_conn_string Connection string for metadata server
     * @param local_server_name Local server name
     * @param enable_copy If true, starts TCP listener for copy-based transfers
     * @param ctrl_block_location Location for CopyCtrlBlock registration (e.g.,
     * "cpu:0")
     */
    explicit FlexTransferEngine(const std::string &metadata_conn_string,
                                const std::string &local_server_name,
                                bool enable_copy,
                                const std::string &ctrl_block_location);

    ~FlexTransferEngine();

    int registerLocalMemory(uintptr_t addr, size_t length,
                            const std::string &location, bool remote_accessible,
                            bool remote_atomic,
                            TransferMode mode = TransferMode::Auto);

    int unregisterLocalMemory(uintptr_t addr,
                              TransferMode mode = TransferMode::Auto);

    int registerLocalMemoryBatch(std::vector<buffer_entry_t> &buffer_list,
                                 const std::string &location,
                                 TransferMode mode = TransferMode::Auto);

    int registerLocalMemoryBatch(MemoryBatch &memory_batch,
                                 TransferMode mode = TransferMode::Auto) {
        for (auto &[location, buffers] : memory_batch.location_buffers_map) {
            int rc = registerLocalMemoryBatch(buffers, location, mode);
            if (rc) return rc;
        }
        return 0;
    }

    int unregisterLocalMemoryBatch(std::vector<uintptr_t> &addr_list,
                                   TransferMode mode = TransferMode::Auto);

    int syncSegmentCache() { return ::syncSegmentCache(engine_); }

    const std::string &getCopyServerUrl() const {
        return copy_server_.getServerUrl();
    }

    // for CopyServer and FlexBatch
    transfer_engine_t getEngine() { return engine_; }

    CopyClient &getCopyClient() { return copy_client_; }

    segment_id_t getSegmentId(const std::string &segment_name);

    CopyCtrlBlock *acquireCopyCtrlBlock();

    void releaseCopyCtrlBlock(CopyCtrlBlock *ctrl_block);

   private:
    const bool enable_copy_;  // Whether to enable copy-based transfer
    const std::string ctrl_block_location_;

    transfer_engine_t engine_;

    // Segment cache (segment_name -> segment_id)
    // Used when acting as copy engine to cache opened segments
    // Append-only; will never remove entries
    std::unordered_map<std::string, segment_id_t> segment_cache_;
    std::mutex segment_cache_mutex_;

    // Cache of CopyCtrlBlock objects for copy-based transfers
    std::vector<CopyCtrlBlock *> copy_ctrl_block_cache_;
    std::mutex ctrl_block_mutex_;

    // TCP listener and worker (only used when enable_copy is true)
    CopyServer copy_server_;

    // Client when talking to a copy_server
    CopyClient copy_client_;
};
