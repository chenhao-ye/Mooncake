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

#ifndef COPY_TRANSFER_ENGINE_H_
#define COPY_TRANSFER_ENGINE_H_

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "transfer_engine_c.h"

namespace mooncake {

/**
 * CopyTransferEngine wraps TransferEngine to avoid frequent RDMA memory
 * registration/unregistration overhead.
 *
 * Instead of directly registering user memory with RDMA NICs, it maintains
 * a pool of pre-registered buffers and copies data between user memory and
 * these buffers.
 *
 * CopyTransferEngine does not provide APIs to submit read/write requests.
 * Instead, it listens on TCP for transfer requests from DirectTransferEngine.
 */
class CopyTransferEngine {
   public:
    CopyTransferEngine()
        : engine_(nullptr),
          worker_running_(false),
          listener_fd_(-1),
          tcp_port_(0) {}

    ~CopyTransferEngine();

    /**
     * Initialize the transfer engine and start TCP listener.
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
     * Register local memory - keeps a record of the memory region and
     * allocates/registers RDMA buffers if needed.
     */
    int registerLocalMemory(void *addr, size_t length,
                            const std::string &location, int remote_accessible);

    /**
     * Unregister local memory - only removes from internal data structures.
     */
    int unregisterLocalMemory(void *addr);

    /**
     * Register a batch of local memory buffers - keeps records and
     * allocates/registers RDMA buffers if needed.
     */
    int registerLocalMemoryBatch(const std::vector<buffer_entry_t> &buffer_list,
                                 const std::string &location);

    /**
     * Unregister a batch of local memory buffers - only removes from
     * internal data structures.
     */
    int unregisterLocalMemoryBatch(const std::vector<void *> &addr_list);

    /**
     * Sync segment cache with metadata server.
     */
    int syncSegmentCache();

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
     * Start the TCP listener thread.
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

    transfer_engine_t engine_;

    // Registered memory regions (addr -> region info)
    std::unordered_map<void *, MemoryRegion> registered_regions_;
    std::mutex regions_mutex_;

    // Buffer pool per location (location -> buffer pair)
    std::unordered_map<std::string, BufferPair *> buffer_pool_;
    std::mutex pool_mutex_;

    // Segment cache (segment_name -> segment_id)
    std::unordered_map<std::string, segment_id_t> segment_cache_;
    std::mutex segment_cache_mutex_;

    // TCP listener and worker
    std::thread worker_thread_;
    std::atomic<bool> worker_running_;
    int listener_fd_;
    uint16_t tcp_port_;
    std::string local_server_name_;
};

}  // namespace mooncake

#endif  // COPY_TRANSFER_ENGINE_H_
