#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

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
        return local_copy_server_url_;
    }

    // for FlexBatch
    transfer_engine_t getEngine() { return engine_; }

    segment_id_t getSegmentId(const std::string &segment_name);

    CopyCtrlBlock *acquireCopyCtrlBlock();

    void releaseCopyCtrlBlock(CopyCtrlBlock *ctrl_block);

    int submitTransferToCopyEngine(std::vector<transfer_request_t> &entries,
                                   const std::string &server_url,
                                   CopyCtrlBlock *ctrl_block);

   private:
    using LocIdx = int32_t;  // <0 for invalid location

    struct MemoryRegion {
        void *addr;
        size_t length;
        LocIdx loc_idx;
    };

    struct BufferPair {
        // buffers[0] is first half, buffers[1] is second half
        // buffers[0] is also the base address of allocation
        char *buffers[2];
        size_t size;   // Size of each half
        bool is_cuda;  // true if CUDA memory, false if CPU memory

        // Used by tasks with given index (-1 for unused)
        int users[2] = {-1, -1};

        BufferPair(char *buffer_base, size_t size, bool is_cuda)
            : buffers{buffer_base, buffer_base + size},
              size(size),
              is_cuda(is_cuda),
              users{-1, -1} {}

        // select the next buffer to use
        // return the one with a lower-index task (likely to finish earlier OR
        // is free for users[i]<0)
        int selectNextBuffer() { return users[0] < users[1] ? 0 : 1; }
    };

    struct Task {
        // request info
        void *source_addr;
        uint64_t target_addr;
        size_t length;
        // execution info
        batch_id_t batch_id = INVALID_BATCH;
        struct BufferPair *buffer_pair = nullptr;
        int buffer_idx = -1;

        Task(void *source_addr, uint64_t target_addr, size_t length)
            : source_addr(source_addr),
              target_addr(target_addr),
              length(length),
              batch_id(INVALID_BATCH),
              buffer_pair(nullptr),
              buffer_idx(-1) {}
    };

    // Require regions_mutex_ to be held before calling
    LocIdx getLocIdx(const std::string &location);

    // Require regions_mutex_ to be held before calling
    MemoryRegion *getRegion(void *addr, size_t length);

    // Require regions_mutex_ to be held before calling
    // When this function is called, there MUST be a buffer pair ready with the
    // proper length (which should have been set up upon registration)
    BufferPair &getBufferPair(LocIdx loc_idx) { return *buffer_pool_[loc_idx]; }

    // Require regions_mutex_ to be held before calling
    BufferPair *allocBufferPair(LocIdx loc_idx, const std::string &location,
                                size_t size);
    // Require regions_mutex_ to be held before calling
    void freeBufferPair(BufferPair *pair);

    int waitTask(Task &task);

    int copyMemory(void *dst, const void *src, size_t size, bool is_cuda);

    // TCP listener and worker thread functions
    void startListener();

    void stopListener();

    void workerThread();

    void handleAndProcessRequest(int client_fd);

    int connectToCopyEngine(const std::string &server_url);

    std::string local_server_name_;
    const bool enable_copy_;  // Whether to enable copy-based transfer
    const std::string ctrl_block_location_;

    transfer_engine_t engine_;

    // Protects copiable_regions_, location_strings_, and buffer_pool_
    std::mutex regions_mutex_;

    // Copiable memory regions (addr -> region info)
    // Tracks regions that can be read via copy transfer.
    // When enable_copy_ is true, these are NOT actually RDMA-registered,
    // only tracked for copy-based transfers.
    // When enable_copy_ is false, these ARE RDMA-registered.
    std::unordered_map<void *, MemoryRegion> copiable_regions_;

    // Buffer pool per location (LocIdx -> buffer pair)
    // Only used when enable_copy is true
    std::vector<BufferPair *> buffer_pool_;

    // Location string storage (LocIdx -> location string)
    // Append-only; will never remove entries
    std::vector<std::string> location_strings_;

    // Segment cache (segment_name -> segment_id)
    // Used when acting as copy engine to cache opened segments
    // Append-only; will never remove entries
    std::unordered_map<std::string, segment_id_t> segment_cache_;
    std::mutex segment_cache_mutex_;

    // TCP listener and worker (only used when enable_copy is true)
    std::thread worker_thread_;
    std::atomic<bool> worker_running_;
    int listener_fd_;
    std::string local_copy_server_url_;  // Copy server URL for this instance

    // TCP connections to remote FlexTransferEngine (server_url -> fd)
    std::unordered_map<std::string, int> copy_engine_connections_;
    std::mutex connections_mutex_;

    // Cache of CopyCtrlBlock objects for copy-based transfers
    std::vector<CopyCtrlBlock *> copy_ctrl_block_cache_;
    std::mutex ctrl_block_mutex_;
};
