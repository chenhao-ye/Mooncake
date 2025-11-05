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
#include "transfer_engine.h"
#include "transfer_engine_c.h"

// Forward declaration
class FlexTransferEngine;

struct CopyCtrlBlock {
    volatile std::atomic_int64_t progress_counter = 0;
};

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

    /**
     * Register local memory with the transfer engine.
     * @param force_direct If true, forces RDMA registration even when
     * enable_copy_ is true
     */
    int registerLocalMemory(void *addr, size_t length,
                            const std::string &location, int remote_accessible,
                            bool force_direct = false);

    /**
     * Unregister local memory.
     * @param force_direct If true, forces RDMA unregistration even when
     * enable_copy_ is true
     */
    int unregisterLocalMemory(void *addr, bool force_direct = false);

    /**
     * Register a batch of local memory buffers.
     * @param force_direct If true, forces RDMA registration even when
     * enable_copy_ is true
     */
    int registerLocalMemoryBatch(std::vector<buffer_entry_t> &buffer_list,
                                 const std::string &location,
                                 bool force_direct = false);

    /**
     * Unregister a batch of local memory buffers.
     * @param force_direct If true, forces RDMA unregistration even when
     * enable_copy_ is true
     */
    int unregisterLocalMemoryBatch(std::vector<void *> &addr_list,
                                   bool force_direct = false);

    /**
     * Sync segment cache with metadata server.
     */
    int syncSegmentCache() { return ::syncSegmentCache(engine_); }

    /**
     * Get the copy server URL for this FlexTransferEngine instance.
     * Returns the URL in format "ip_addr:port" that can be used by other
     * instances to submit copy-based transfer requests.
     * Returns empty string if enable_copy_ is false.
     */
    const std::string &getCopyServerUrl() const {
        return copy_server_.getServerUrl();
    }

    // for CopyServer and FlexBatch
    transfer_engine_t getEngine() { return engine_; }

    CopyClient &getCopyClient() { return copy_client_; }

    // for CopyClient
    const std::string &getLocalServerName() const { return local_server_name_; }

    segment_id_t getSegmentId(const std::string &segment_name);

    CopyCtrlBlock *acquireCopyCtrlBlock();

    void releaseCopyCtrlBlock(CopyCtrlBlock *ctrl_block);

   private:
    std::string local_server_name_;
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
