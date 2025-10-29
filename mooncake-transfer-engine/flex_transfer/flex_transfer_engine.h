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
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "transfer_engine_c.h"

namespace mooncake {

/**
 * FlexTransferEngine is a flexible wrapper on top of TransferEngine that
 * supports both direct RDMA transfers and copy-based transfers.
 *
 * When enable_copy is true, it starts a TCP listener to accept transfer
 * requests from other FlexTransferEngine instances, acting as a
 * CopyTransferEngine. It maintains pre-registered buffers to avoid frequent
 * RDMA memory registration overhead.
 *
 * When submitting transfers, users can specify a copy_server_name to use
 * copy-based transfer via TCP, or leave it empty for direct RDMA transfer.
 */
class FlexTransferEngine {
   public:
    /**
     * Constructor.
     * @param enable_copy If true, starts TCP listener for copy-based transfers
     */
    explicit FlexTransferEngine(bool enable_copy = false)
        : engine_(nullptr),
          enable_copy_(enable_copy),
          worker_running_(false),
          listener_fd_(-1),
          tcp_port_(0),
          progress_(0),
          progress_registered_(false) {}

    ~FlexTransferEngine();

    /**
     * Initialize the transfer engine.
     * @param tcp_port TCP port for copy-based transfer listener (only used if
     *                 enable_copy is true)
     */
    int init(const std::string &metadata_conn_string,
             const std::string &local_server_name,
             const std::string &ip_or_host_name = "", uint64_t rpc_port = 12345,
             uint16_t tcp_port = 12346, int auto_discover = 1);

    /**
     * Open a segment by name.
     */
    segment_id_t openSegment(const std::string &segment_name);

    /**
     * Close a segment.
     */
    int closeSegment(segment_id_t segment_id);

    /**
     * Register local memory with the transfer engine.
     */
    int registerLocalMemory(void *addr, size_t length,
                            const std::string &location, int remote_accessible);

    /**
     * Unregister local memory.
     */
    int unregisterLocalMemory(void *addr);

    /**
     * Register a batch of local memory buffers.
     */
    int registerLocalMemoryBatch(const std::vector<buffer_entry_t> &buffer_list,
                                 const std::string &location);

    /**
     * Unregister a batch of local memory buffers.
     */
    int unregisterLocalMemoryBatch(const std::vector<void *> &addr_list);

    /**
     * Sync segment cache with metadata server.
     */
    int syncSegmentCache();

    /**
     * Allocate a batch ID for transfer operations.
     */
    batch_id_t allocateBatchID(size_t batch_size);

    /**
     * Free a batch ID.
     */
    int freeBatchID(batch_id_t batch_id);

    /**
     * Submit a transfer batch.
     *
     * @param batch_id The batch ID allocated by allocateBatchID.
     * @param entries The transfer requests.
     * @param copy_server_name Optional server name for copy-based transfer. If
     *        provided, transfer will use copy-based approach via TCP.
     * @param copy_server_port TCP port for copy-based transfer (default 12346).
     */
    int submitTransfer(batch_id_t batch_id,
                       const std::vector<transfer_request_t> &entries,
                       const std::string &copy_server_name = "",
                       uint16_t copy_server_port = 12346);

    /**
     * Get the status of a transfer.
     */
    int getTransferStatus(batch_id_t batch_id, size_t task_id,
                          transfer_status_t &status);

    /**
     * Get the TCP port that the listener is bound to.
     */
    uint16_t getTcpPort() const { return tcp_port_; }

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
        void *base_buffer;  // Single contiguous allocation for both buffers
        void *buffer1;      // First half of base_buffer
        void *buffer2;      // Second half of base_buffer
        size_t size;        // Size of each half
        bool is_gpu;        // true if CUDA memory, false if CPU memory
        bool buffer1_in_use;
        bool buffer2_in_use;
    };

    /**
     * Start the TCP listener thread (only if enable_copy is true).
     */
    int startListener(const std::string &ip_or_host_name, uint16_t tcp_port);

    /**
     * Stop the TCP listener thread.
     */
    void stopListener();

    /**
     * Worker thread that listens and processes requests.
     */
    void workerThread();

    /**
     * Handle and process a transfer request from a client.
     */
    void handleAndProcessRequest(int client_fd);

    /**
     * Get or allocate a buffer pair for the given location and size.
     */
    BufferPair *getOrAllocateBufferPair(const std::string &location,
                                        size_t size);

    /**
     * Allocate and register a new buffer pair.
     */
    BufferPair *allocateBufferPair(const std::string &location, size_t size);

    /**
     * Copy data from source to destination (handles both CPU and GPU memory).
     */
    int copyMemory(void *dst, const void *src, size_t size, bool is_gpu);

    /**
     * Check if an address is registered.
     */
    bool isAddressRegistered(void *addr);

    /**
     * Get location for an address (returns empty string if not found).
     */
    std::string getLocation(void *addr);

    /**
     * Connect to a remote FlexTransferEngine via TCP.
     */
    int connectToCopyEngine(const std::string &server_name, uint16_t port);

    /**
     * Send batch info to remote FlexTransferEngine and initiate transfer.
     */
    int submitTransferToCopyEngine(
        batch_id_t batch_id, const std::vector<transfer_request_t> &entries,
        const std::string &server_name, uint16_t port);

    transfer_engine_t engine_;
    bool enable_copy_;  // Whether to enable copy-based transfer listener

    // Local server name (also serves as the local RAM segment name)
    std::string local_server_name_;

    // Registered memory regions (addr -> region info)
    // Only used when enable_copy is true
    std::unordered_map<void *, MemoryRegion> registered_regions_;
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
    uint16_t tcp_port_;

    // TCP connections to remote FlexTransferEngine instances (server_name -> fd)
    std::unordered_map<std::string, int> copy_engine_connections_;
    std::mutex connections_mutex_;

    // Pre-registered progress counter for copy-based transfers
    std::atomic<int64_t> progress_;
    bool progress_registered_;
};

}  // namespace mooncake

#endif  // FLEX_TRANSFER_ENGINE_H_
