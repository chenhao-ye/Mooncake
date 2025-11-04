#include "batch.h"

#include "flex_transfer_engine.h"

FlexBatch::~FlexBatch() {
    if (copy_ctrl_block_) engine_->releaseCopyCtrlBlock(copy_ctrl_block_);
    if (batch_id_ != INVALID_BATCH)
        ::freeBatchID(engine_->getEngine(), batch_id_);
}

void FlexBatch::addReadRequest(uintptr_t local_addr, uintptr_t remote_addr,
                               uint64_t size) {
    entries_.emplace_back(
        transfer_request_t{.opcode = OPCODE_READ,
                           .source = reinterpret_cast<void *>(local_addr),
                           .target_id = -1,  // will be set upon submit()
                           .target_offset = remote_addr,
                           .length = size});
}
void FlexBatch::addWriteRequest(uintptr_t local_addr, uintptr_t remote_addr,
                                uint64_t size) {
    entries_.emplace_back(
        transfer_request_t{.opcode = OPCODE_WRITE,
                           .source = reinterpret_cast<void *>(local_addr),
                           .target_id = -1,  // will be set upon submit()
                           .target_offset = remote_addr,
                           .length = size});
}

int FlexBatch::submit(const std::string &target, bool is_target_copy) {
    if (entries_.empty()) return 0;

    if (!is_target_copy) {  // target is a segment name for direct RDMA
        batch_id_ = ::allocateBatchID(engine_->getEngine(), entries_.size());
        if (batch_id_ == INVALID_BATCH) return -1;

        auto target_segment_id = engine_->getSegmentId(target);
        for (auto &entry : entries_) entry.target_id = target_segment_id;
        return ::submitTransfer(engine_->getEngine(), batch_id_,
                                entries_.data(), entries_.size());
    }
    // else: copy-based transfer
    copy_ctrl_block_ = engine_->acquireCopyCtrlBlock();
    if (!copy_ctrl_block_) return -1;
    return engine_->submitTransferToCopyEngine(entries_, target,
                                               copy_ctrl_block_);
}

int FlexBatch::getTransferStatus(size_t task_id, transfer_status_t &status) {
    if (!copy_ctrl_block_) {  // Direct RDMA transfer
        return ::getTransferStatus(engine_->getEngine(), batch_id_, task_id,
                                   &status);
    }

    // for copy-based transfers, check ctrl_block progress
    int64_t progress = copy_ctrl_block_->progress_counter;
    if (static_cast<int64_t>(task_id) < progress) {  // Task completed
        status.status = STATUS_COMPLETED;
        status.transferred_bytes = entries_[task_id].length;
    } else {  // still in progress or waiting
        status.status = STATUS_PENDING;
        status.transferred_bytes = 0;
    }
    return 0;
}
