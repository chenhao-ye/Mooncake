#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "transfer_engine_c.h"

struct CopyCtrlBlock;
struct ClientConnection;
class FlexTransferEngine;

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
          copy_ctrl_block_(nullptr),
          client_conn_(nullptr) {}

    ~FlexBatch() { free(); }

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
     * @param use_rdma If true, use RDMA; if false, use TCP. Only valid if
     * is_target_copy is true (direct mode must use RDMA).
     */
    int submit(const std::string &target, bool is_target_copy = false,
               bool use_rdma = true);

    /**
     * Get the status of a transfer task.
     * For copy-based transfers, this automatically checks the socket for
     * finalized progress if the progress counter hasn't changed.
     * @param task_id The task ID within this batch
     */
    int getTransferStatus(size_t task_id);

    void free();

   private:
    std::shared_ptr<FlexTransferEngine> engine_;
    std::vector<transfer_request_t> entries_;

    // for direct transfer
    batch_id_t batch_id_;

    // for copy transfer
    CopyCtrlBlock *copy_ctrl_block_;
    // once see a progress that implies fully finished OR received a int32_t
    // from the socket, it means this batch is done with the connection; then
    // set client_conn_ to nullptr
    ClientConnection *client_conn_;
    int64_t last_progress_;
};

struct MemoryBatch {
    std::unordered_map<std::string, std::vector<buffer_entry_t>>
        location_buffers_map;

    void add(void *addr, size_t length, const std::string &location) {
        location_buffers_map[location].emplace_back(
            buffer_entry_t{addr, length});
    };
};
