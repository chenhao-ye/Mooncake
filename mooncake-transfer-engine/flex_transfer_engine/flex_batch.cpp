#include "flex_batch.h"

#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "flex_transfer_engine.h"
#include "transfer_engine_c.h"

void FlexBatch::free() {
    if (batch_id_ != INVALID_BATCH) {
        ::freeBatchID(engine_->getEngine(), batch_id_);
        batch_id_ = INVALID_BATCH;
    }
    if (client_conn_) {
        engine_->getCopyClient().freeConnection(client_conn_);
        client_conn_ = nullptr;
    }
    // TODO: free ctrl blocks
    // if (ctrl_block_) {
    //     engine_->freeRdmaCopyCtrlBlock(ctrl_block_);
    //     ctrl_block_ = nullptr;
    // }
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

int FlexBatch::submit(const std::string &target, bool is_target_copy,
                      bool use_rdma) {
    if (entries_.empty()) return 0;

    if (!is_target_copy) {  // target is a segment name for direct RDMA
        batch_id_ = ::allocateBatchID(engine_->getEngine(), entries_.size());
        if (batch_id_ == INVALID_BATCH) return -1;

        auto target_segment_id = engine_->getSegmentId(target);
        if (target_segment_id < 0)
            throw std::runtime_error("Failed to get segment ID for segment: " +
                                     target);
        for (auto &entry : entries_) entry.target_id = target_segment_id;
        return ::submitTransfer(engine_->getEngine(), batch_id_,
                                entries_.data(), entries_.size());
    }
    // else: copy-based transfer
    auto &copy_client = engine_->getCopyClient();
    client_conn_ = copy_client.allocConnection(target);
    if (use_rdma) {
        rdma_ctrl_block_ =
            copy_client.submitRdmaRequests(client_conn_, entries_);
    } else {
        tcp_ctrl_block_ = copy_client.submitTcpRequests(client_conn_, entries_);
    }
    return 0;
}

int FlexBatch::getTransferStatus(size_t task_id) {
    if (!rdma_ctrl_block_ && !tcp_ctrl_block_) {  // Direct RDMA transfer
        transfer_status_t status;
        int rc = ::getTransferStatus(engine_->getEngine(), batch_id_, task_id,
                                     &status);
        if (rc) throw std::runtime_error("Fail to get transfer status");
        return status.status;
    }

    if (static_cast<int64_t>(task_id) < last_progress_) return STATUS_COMPLETED;

    // it has been finalized, so no more progress will be made
    if (!client_conn_) return STATUS_FAILED;

    if (rdma_ctrl_block_) checkRdmaProgress();
    // TODO: else: checkTcpProgress();

    if (static_cast<int64_t>(task_id) < last_progress_) return STATUS_COMPLETED;

    // if finalized, all non-completed task are failed
    return client_conn_ ? STATUS_WAITING : STATUS_FAILED;
}

// read from rdma_ctrl_block_ and update last_progress_
// if it is finalized, free client_conn_
void FlexBatch::checkRdmaProgress() {
    int64_t prev_progress = last_progress_;
    last_progress_ =
        rdma_ctrl_block_->progress_counter.load(std::memory_order_acquire);
    assert(last_progress_ >= prev_progress);
    if (last_progress_ > prev_progress) {  // new progress made
        if (last_progress_ == static_cast<int64_t>(entries_.size())) {
            // all done; finalize it (but keep pending=True because we didn't
            // pop the finalized value yet from the socket)
            assert(client_conn_->has_pending);
            engine_->getCopyClient().freeConnection(client_conn_);
            client_conn_ = nullptr;
        }
        return;
    }
    // no new progress; check socket to see if finalized

    // check socket to see the server finalizes this connection
    uint32_t finalized_value = 0;
    ssize_t nbytes = recv(client_conn_->fd, &finalized_value,
                          sizeof(finalized_value), MSG_DONTWAIT);

    if (nbytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        return;  // the server did not finalize it

    // finalized: either the server has finalized it OR something went wrong

    // the server has finalized it
    if (nbytes == sizeof(finalized_value)) {
        last_progress_ = finalized_value;
        if (last_progress_ == static_cast<int64_t>(entries_.size())) {
            // all done; finalize it and mark it with no pending
            client_conn_->has_pending = false;
            engine_->getCopyClient().freeConnection(client_conn_);
            client_conn_ = nullptr;
            return;
        }
    }

    // something went wrong so that we need to close this connection, e.g.,
    // - the server has closed the connection (nbytes=0) OR
    // - the server crashed (nbytes<0 with unexpected errno) OR
    // - the finalized value is not an expected value
    client_conn_->free();
    delete client_conn_;
    client_conn_ = nullptr;
}
