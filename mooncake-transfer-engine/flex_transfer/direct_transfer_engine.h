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
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "transfer_engine.h"

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
    DirectTransferEngine(bool auto_discover = false)
        : engine_(std::make_unique<TransferEngine>(auto_discover)),
          progress_(0),
          progress_registered_(false) {}

    DirectTransferEngine(bool auto_discover,
                         const std::vector<std::string> &filter)
        : engine_(std::make_unique<TransferEngine>(auto_discover, filter)),
          progress_(0),
          progress_registered_(false) {}

    ~DirectTransferEngine();

    /**
     * Initialize the transfer engine.
     */
    int init(const std::string &metadata_conn_string,
             const std::string &local_server_name,
             const std::string &ip_or_host_name = "",
             uint64_t rpc_port = 12345);

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
     * Register local memory with the transfer engine.
     */
    int registerLocalMemory(void *addr, size_t length,
                            const std::string &location = kWildcardLocation,
                            bool remote_accessible = true,
                            bool update_metadata = true) {
        return engine_->registerLocalMemory(addr, length, location,
                                            remote_accessible, update_metadata);
    }

    /**
     * Unregister local memory.
     */
    int unregisterLocalMemory(void *addr, bool update_metadata = true) {
        return engine_->unregisterLocalMemory(addr, update_metadata);
    }

    /**
     * Register a batch of local memory buffers.
     */
    int registerLocalMemoryBatch(const std::vector<BufferEntry> &buffer_list,
                                 const std::string &location) {
        return engine_->registerLocalMemoryBatch(buffer_list, location);
    }

    /**
     * Unregister a batch of local memory buffers.
     */
    int unregisterLocalMemoryBatch(const std::vector<void *> &addr_list) {
        return engine_->unregisterLocalMemoryBatch(addr_list);
    }

    /**
     * Sync segment cache with metadata server.
     */
    int syncSegmentCache(const std::string &segment_name = "") {
        return engine_->syncSegmentCache(segment_name);
    }

    /**
     * Allocate a batch ID for transfer operations.
     */
    BatchID allocateBatchID(size_t batch_size) {
        return engine_->allocateBatchID(batch_size);
    }

    /**
     * Free a batch ID.
     */
    Status freeBatchID(BatchID batch_id) {
        return engine_->freeBatchID(batch_id);
    }

    /**
     * Submit a transfer batch.
     *
     * @param batch_id The batch ID allocated by allocateBatchID.
     * @param entries The transfer requests.
     * @param target_is_copy_engine If true, the target is a CopyTransferEngine
     *        and this batch should only contain read requests.
     */
    Status submitTransfer(BatchID batch_id,
                          const std::vector<TransferRequest> &entries,
                          bool target_is_copy_engine = false);

    /**
     * Get the status of a transfer.
     */
    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus &status) {
        return engine_->getTransferStatus(batch_id, task_id, status);
    }

    /**
     * Get the underlying TransferEngine.
     */
    TransferEngine *getEngine() { return engine_.get(); }

   private:
    /**
     * Connect to a CopyTransferEngine via TCP.
     */
    int connectToCopyEngine(const std::string &server_name, uint16_t port);

    /**
     * Send batch info to CopyTransferEngine and initiate transfer.
     */
    Status submitTransferToCopyEngine(
        BatchID batch_id, const std::vector<TransferRequest> &entries);

    std::unique_ptr<TransferEngine> engine_;

    // TCP connections to CopyTransferEngine instances (server_name -> fd)
    std::unordered_map<std::string, int> copy_engine_connections_;
    std::mutex connections_mutex_;

    // Pre-registered progress counter for CopyTransferEngine transfers
    std::atomic<int64_t> progress_;
    bool progress_registered_;
};

}  // namespace mooncake

#endif  // DIRECT_TRANSFER_ENGINE_H_
