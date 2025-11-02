// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FLEX_TRANSFER_ENGINE_H_
#define FLEX_TRANSFER_ENGINE_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "transfer_engine_c.h"

namespace mooncake {

// Forward declaration
class FlexTransferEngine;

/**
 * Control block for copy-based transfers.
 * Contains progress counter that is RDMA-accessible.
 */
struct CopyCtrlBlock {
    volatile int64_t progress_counter;
};

/**
 * FlexBatch represents a batch transfer operation.
 * It manages the batch lifecycle including freeing the batch ID and
 * returning the CopyCtrlBlock to the engine cache on destruction.
 */
class FlexBatch {
   public:
    /**
     * Constructor.
     * @param engine Backpointer to the FlexTransferEngine
     */
    explicit FlexBatch(std::shared_ptr<FlexTransferEngine> engine)
        : engine_(std::move(engine)),
          batch_id_(INVALID_BATCH),
          copy_ctrl_block_(nullptr) {}

    ~FlexBatch();

    void addReadRequest(uintptr_t local_addr, uintptr_t remote_addr,
                        uint64_t size);
    void addWriteRequest(uintptr_t local_addr, uintptr_t remote_addr,
                         uint64_t size);

    /**
     * Submit the transfer batch.
     *
     * @param target Remote target, either Mooncake segment name or copy server
     * URL (formatted as "ip:port").
     * @param is_target_copy If true, target is a copy server URL; if false,
     * target is a segment name.
     */
    int submit(const std::string &target, bool is_target_copy = false);

    /**
     * Get the status of a transfer task.
     * @param task_id The task ID within this batch
     * @param status Output parameter for transfer status
     */
    int getTransferStatus(size_t task_id, transfer_status_t &status);

   private:
    std::shared_ptr<FlexTransferEngine> engine_;
    batch_id_t batch_id_;
    CopyCtrlBlock *copy_ctrl_block_;
    std::vector<transfer_request_t> entries_;
};

/**
 * FlexTransferEngine is a flexible wrapper on top of TransferEngine that
 * supports both direct RDMA transfers and copy-based transfers.
 *
 * When enable_copy is true, it starts a TCP listener to accept transfer
 * requests from other FlexTransferEngine instances, acting as a
 * CopyTransferEngine. It maintains pre-registered buffers to avoid frequent
 * RDMA memory registration overhead.
 *
 * When submitting transfers, users can specify a copy_server_url to use
 * copy-based transfer via TCP, or leave it empty for direct RDMA transfer.
 */
class FlexTransferEngine {
   public:
    // FlexBatch needs access to private methods like releaseCopyCtrlBlock
    friend class FlexBatch;

    /**
     * Constructor.
     * @param enable_copy If true, starts TCP listener for copy-based transfers
     * @param ctrl_block_location Location for CopyCtrlBlock registration (e.g.,
     * "cuda:0" for GPU)
     */
    explicit FlexTransferEngine(bool enable_copy = false,
                                const std::string &ctrl_block_location = "")
        : engine_(nullptr),
          enable_copy_(enable_copy),
          ctrl_block_location_(ctrl_block_location),
          worker_running_(false),
          listener_fd_(-1) {}

    ~FlexTransferEngine();

    /**
     * Initialize the transfer engine.
     * TCP port for copy-based transfer listener is automatically selected when
     * enable_copy is true.
     */
    int init(const std::string &metadata_conn_string,
             const std::string &local_server_name, bool auto_discover = true);

    /**
     * Get a segment ID by name (will open the segment if not cached).
     */
    segment_id_t getSegmentId(const std::string &segment_name);

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
    int syncSegmentCache();

    /**
     * Get the copy server URL for this FlexTransferEngine instance.
     * Returns the URL in format "ip_addr:port" that can be used by other
     * instances to submit copy-based transfer requests.
     * Returns empty string if enable_copy_ is false.
     */
    std::string getCopyServerUrl() const;

    /**
     * Get the underlying TransferEngine handle.
     */
    transfer_engine_t getEngine() { return engine_; }

   private:
    struct MemoryRegion {
        void *addr;
        size_t length;
        std::string location;
    };

    struct BufferPair {
        void *buffer0;  // First half (also the base address of allocation)
        void *buffer1;  // Second half
        size_t size;    // Size of each half
        bool is_cuda;   // true if CUDA memory, false if CPU memory
        bool buffer0_in_use;
        bool buffer1_in_use;
    };

    void workerThread();

    int startListener();
    void stopListener();

    void handleAndProcessRequest(int client_fd);

    BufferPair *getOrAllocBufferPair(const std::string &location,
                                        size_t size);

    BufferPair *allocBufferPair(const std::string &location, size_t size);
    void freeBufferPair(BufferPair *pair);

    int copyMemory(void *dst, const void *src, size_t size, bool is_cuda);

    bool isAddressRegistered(void *addr);

    std::string getLocation(void *addr);

    int connectToCopyEngine(const std::string &server_url);

    int submitTransferToCopyEngine(std::vector<transfer_request_t> &entries,
                                   const std::string &server_url,
                                   CopyCtrlBlock *ctrl_block);

    CopyCtrlBlock *acquireCopyCtrlBlock();

    void releaseCopyCtrlBlock(CopyCtrlBlock *ctrl_block);

    transfer_engine_t engine_;
    bool enable_copy_;  // Whether to enable copy-based transfer listener
    std::string ctrl_block_location_;  // CopyCtrlBlock registration location

    // Local server name (also serves as the local RAM segment name)
    std::string local_server_name_;

    // Copiable memory regions (addr -> region info)
    // Tracks regions that can be read via copy transfer.
    // When enable_copy_ is true, these are NOT actually RDMA-registered,
    // only tracked for copy-based transfers.
    // When enable_copy_ is false, these ARE RDMA-registered.
    std::unordered_map<void *, MemoryRegion> copiable_regions_;
    std::mutex regions_mutex_;

    // Buffer pool per location (location -> buffer pair)
    // Only used when enable_copy is true
    std::unordered_map<std::string, BufferPair *> buffer_pool_;
    std::mutex pool_mutex_;

    // Segment cache (segment_name -> segment_id)
    // Used when acting as copy engine to cache opened segments
    std::unordered_map<std::string, segment_id_t> segment_cache_;
    std::mutex segment_cache_mutex_;

    // TCP listener and worker (only used when enable_copy is true)
    std::thread worker_thread_;
    std::atomic<bool> worker_running_;
    int listener_fd_;
    std::string local_copy_server_url_;  // Copy server URL for this instance

    // TCP connections to remote FlexTransferEngine instances (server_url -> fd)
    std::unordered_map<std::string, int> copy_engine_connections_;
    std::mutex connections_mutex_;

    // Cache of CopyCtrlBlock objects for copy-based transfers
    std::vector<CopyCtrlBlock *> copy_ctrl_block_cache_;
    std::mutex ctrl_block_mutex_;
};

}  // namespace mooncake

#endif  // FLEX_TRANSFER_ENGINE_H_
