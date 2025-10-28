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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "transfer_engine.h"

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
    CopyTransferEngine(bool auto_discover = false)
        : engine_(std::make_unique<TransferEngine>(auto_discover)),
          worker_running_(false),
          listener_fd_(-1),
          tcp_port_(0) {}

    CopyTransferEngine(bool auto_discover,
                       const std::vector<std::string> &filter)
        : engine_(std::make_unique<TransferEngine>(auto_discover, filter)),
          worker_running_(false),
          listener_fd_(-1),
          tcp_port_(0) {}

    ~CopyTransferEngine() { stopListener(); }

    /**
     * Initialize the transfer engine and start TCP listener.
     */
    int init(const std::string &metadata_conn_string,
             const std::string &local_server_name,
             const std::string &ip_or_host_name = "", uint64_t rpc_port = 12345,
             uint16_t tcp_port = 12346);

    /**
     * Open a segment by name.
     */
    SegmentHandle openSegment(const std::string &segment_name) {
        return engine_->openSegment(segment_name);
    }

    /**
     * Close a segment.
     */
    int closeSegment(SegmentHandle handle) {
        return engine_->closeSegment(handle);
    }

    /**
     * Register local memory - keeps a record of the memory region and
     * allocates/registers RDMA buffers if needed.
     */
    int registerLocalMemory(void *addr, size_t length,
                            const std::string &location = kWildcardLocation,
                            bool remote_accessible = true,
                            bool update_metadata = true);

    /**
     * Unregister local memory - only removes from internal data structures.
     */
    int unregisterLocalMemory(void *addr, bool update_metadata = true);

    /**
     * Register a batch of local memory buffers - keeps records and
     * allocates/registers RDMA buffers if needed.
     */
    int registerLocalMemoryBatch(const std::vector<BufferEntry> &buffer_list,
                                 const std::string &location);

    /**
     * Unregister a batch of local memory buffers - only removes from
     * internal data structures.
     */
    int unregisterLocalMemoryBatch(const std::vector<void *> &addr_list);

    /**
     * Sync segment cache with metadata server.
     */
    int syncSegmentCache(const std::string &segment_name = "") {
        return engine_->syncSegmentCache(segment_name);
    }

    /**
     * Get the TCP port that the listener is bound to.
     */
    uint16_t getTcpPort() const { return tcp_port_; }

    /**
     * Get the underlying TransferEngine.
     */
    TransferEngine *getEngine() { return engine_.get(); }

   private:
    struct MemoryRegion {
        void *addr;
        size_t length;
        std::string location;
    };

    struct BufferPair {
        void *buffer1;
        void *buffer2;
        size_t size;
        bool is_gpu;  // true if CUDA memory, false if CPU memory
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

    std::unique_ptr<TransferEngine> engine_;

    // Registered memory regions (addr -> region info)
    std::unordered_map<void *, MemoryRegion> registered_regions_;
    std::mutex regions_mutex_;

    // Buffer pool per location (location -> buffer pair)
    std::unordered_map<std::string, BufferPair *> buffer_pool_;
    std::mutex pool_mutex_;

    // TCP listener and worker
    std::thread worker_thread_;
    std::atomic<bool> worker_running_;
    int listener_fd_;
    uint16_t tcp_port_;
    std::string local_server_name_;
};

}  // namespace mooncake

#endif  // COPY_TRANSFER_ENGINE_H_
