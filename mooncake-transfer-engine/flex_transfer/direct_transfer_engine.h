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

#ifndef DIRECT_TRANSFER_ENGINE_H_
#define DIRECT_TRANSFER_ENGINE_H_

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "transfer_engine_c.h"

namespace mooncake {

/**
 * DirectTransferEngine is a thin wrapper on top of TransferEngine that
 * supports reading from CopyTransferEngine.
 *
 * It exposes the same API as TransferEngine, and all data transfer between
 * two DirectTransferEngine instances work the same as raw TransferEngine.
 * The only difference is how it interacts with CopyTransferEngine.
 */
class DirectTransferEngine {
   public:
    DirectTransferEngine()
        : engine_(nullptr), progress_(0), progress_registered_(false) {}

    ~DirectTransferEngine();

    /**
     * Initialize the transfer engine.
     */
    int init(const std::string &metadata_conn_string,
             const std::string &local_server_name,
             const std::string &ip_or_host_name = "",
             uint64_t rpc_port = 12345, int auto_discover = 1);

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
     * @param copy_server_name Optional server name for CopyTransferEngine. If
     *        provided, transfer will use copy-based approach via TCP.
     * @param copy_server_port TCP port for CopyTransferEngine (default 12346).
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
     * Get the underlying TransferEngine handle.
     */
    transfer_engine_t getEngine() { return engine_; }

   private:
    /**
     * Connect to a CopyTransferEngine via TCP.
     */
    int connectToCopyEngine(const std::string &server_name, uint16_t port);

    /**
     * Send batch info to CopyTransferEngine and initiate transfer.
     */
    int submitTransferToCopyEngine(
        batch_id_t batch_id, const std::vector<transfer_request_t> &entries,
        const std::string &server_name, uint16_t port);

    transfer_engine_t engine_;

    // Local server name (also serves as the local RAM segment name)
    std::string local_server_name_;

    // TCP connections to CopyTransferEngine instances (server_name -> fd)
    std::unordered_map<std::string, int> copy_engine_connections_;
    std::mutex connections_mutex_;

    // Pre-registered progress counter for CopyTransferEngine transfers
    std::atomic<int64_t> progress_;
    bool progress_registered_;
};

}  // namespace mooncake

#endif  // DIRECT_TRANSFER_ENGINE_H_
