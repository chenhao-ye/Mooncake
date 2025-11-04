#pragma once

#include <memory>
#include <string>
#include <vector>

#include "transfer_engine_c.h"

struct CopyCtrlBlock;
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
