#include "flex_batch.h"

#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>

#include "flex_transfer_engine.h"
#include "transfer_engine_c.h"

void FlexBatch::free() {
    if (batch_id_ != INVALID_BATCH) {
        ::freeBatchID(engine_->getEngine(), batch_id_);
        batch_id_ = INVALID_BATCH;
    }
    // it is generally unexpected that free() see non-null client_conn_ or
    // xxx_ctrl_blocks, because they should have been freed when finalized upon
    // getTransferStatus. if that didn't happen, likely something went wrong; we
    // therefore don't reuse the connection.
    if (client_conn_) {
        client_conn_->cleanup();
        delete client_conn_;
        client_conn_ = nullptr;
    }
    if (rdma_ctrl_block_) {
        // FIXME: unclear whether the remote server may update the progress
        // counter via RDMA; should revisit with the transfer failure protocol.
        // for now, assume it won't.
        engine_->getCopyClient().freeCtrlBlock(rdma_ctrl_block_);
        rdma_ctrl_block_ = nullptr;
    }
    if (tcp_ctrl_block_) {
        // wait until the working thread to finalize it (otherwise it is unsafe
        // to have the worker still modifying the copiable region).
        // the worker will release the mutex once it is done.
        std::lock_guard lock(tcp_ctrl_block_->mutex_);
        engine_->getCopyClient().freeCtrlBlock(tcp_ctrl_block_);
        tcp_ctrl_block_ = nullptr;
    }
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

void FlexBatch::addFetchAddRequest(uintptr_t local_addr, uintptr_t remote_addr,
                                   uint64_t value) {
    entries_.emplace_back(
        transfer_request_t{.opcode = OPCODE_ATOMIC_FETCH_ADD,
                           .source = reinterpret_cast<void *>(local_addr),
                           .target_id = -1,  // will be set upon submit()
                           .target_offset = remote_addr,
                           // for fetch-add, .length is overloaded for operand
                           .length = value});
}

int FlexBatch::submit(const std::string &target, bool is_direct,
                      bool use_rdma) {
    is_direct_ = is_direct;
    if (entries_.empty()) return 0;

    if (is_direct) {  // target is a segment name for direct RDMA
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
    if (is_direct_) {
        transfer_status_t status;
        int rc = ::getTransferStatus(engine_->getEngine(), batch_id_, task_id,
                                     &status);
        if (rc) throw std::runtime_error("Fail to get transfer status");
        return status.status;
    }

    if (static_cast<int64_t>(task_id) < known_progress_)
        return STATUS_COMPLETED;

    // it has been finalized, so no more progress will be made
    if (isCopyFinalized()) return STATUS_FAILED;

    assert(rdma_copy_backend_ || tcp_ctrl_block_);
    if (rdma_ctrl_block_) {
        checkRdmaProgress();
    } else if (tcp_ctrl_block_) {
        checkTcpProgress();
    } else {
        assert(false);
    }

    // re-check progress
    if (static_cast<int64_t>(task_id) < known_progress_)
        return STATUS_COMPLETED;
    // if finalized, all incompleted task are considered failed
    return isCopyFinalized() ? STATUS_FAILED : STATUS_WAITING;
}

// read from rdma_ctrl_block_ and update known_progress_
// if it is finalized, free client_conn_
void FlexBatch::checkRdmaProgress() {
    ssize_t nbytes;
    uint32_t finalized_value;

    int64_t prev_progress = known_progress_;
    known_progress_ =
        rdma_ctrl_block_->progress_counter.load(std::memory_order_acquire);
    assert(known_progress_ >= prev_progress);
    if (known_progress_ > prev_progress) {  // new progress made
        if (known_progress_ == static_cast<int64_t>(entries_.size())) {
            // all done; finalize it (but keep pending=True because we
            // didn't pop the finalized value yet from the socket)
            client_conn_->has_pending = true;
            goto completed;
        }
        return;
    }
    /* no new progress; check socket to see if finalized */

    // check socket to see the server finalizes this connection
    nbytes = recv(client_conn_->fd, &finalized_value, sizeof(finalized_value),
                  MSG_DONTWAIT);

    if (nbytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        return;  // the server did not finalize it

    /* finalized: either the server has finalized it OR something went wrong */

    // the server has finalized it
    if (nbytes == sizeof(finalized_value)) {
        known_progress_ = finalized_value;
        if (known_progress_ == static_cast<int64_t>(entries_.size())) {
            // all done; finalize it and mark it with no pending
            client_conn_->has_pending = false;
            goto completed;
        }
    }

    // something went wrong so that we need to close this connection, e.g.,
    // - the server has closed the connection (nbytes=0) OR
    // - the server crashed (nbytes<0 with unexpected errno) OR
    // - the finalized value is not an expected value
    client_conn_->cleanup();
    delete client_conn_;
    client_conn_ = nullptr;
    engine_->getCopyClient().freeCtrlBlock(rdma_ctrl_block_);
    rdma_ctrl_block_ = nullptr;
    return;

completed:  // it is finalized and all tasks are completed
    auto &copy_client = engine_->getCopyClient();
    copy_client.freeConnection(client_conn_);
    client_conn_ = nullptr;
    copy_client.freeCtrlBlock(rdma_ctrl_block_);
    rdma_ctrl_block_ = nullptr;
}

void FlexBatch::checkTcpProgress() {
    int64_t prev_progress = known_progress_;
    known_progress_ =
        tcp_ctrl_block_->progress_counter.load(std::memory_order_acquire);
    assert(known_progress_ >= prev_progress);
    if (known_progress_ > prev_progress) {  // new progress made
        if (known_progress_ == static_cast<int64_t>(entries_.size()))
            goto completed;  // must be finalized
        return;
    }
    {  // no new progress; check if finalized
        std::unique_lock<std::mutex> lock(tcp_ctrl_block_->mutex_,
                                          std::try_to_lock);
        if (!lock.owns_lock()) return;  // Worker is still working

        // else: mutex is successfully acquired, meaning the worker has
        // finalized it read progress again to prevent race
        known_progress_ =
            tcp_ctrl_block_->progress_counter.load(std::memory_order_acquire);
    }
    if (known_progress_ == static_cast<int64_t>(entries_.size()))
        goto completed;

    // else: something went wrong (some tasks failed); free the connection
    client_conn_->cleanup();
    delete client_conn_;
    client_conn_ = nullptr;
    engine_->getCopyClient().freeCtrlBlock(tcp_ctrl_block_);
    tcp_ctrl_block_ = nullptr;
    return;

completed:
    // Tcp requests do not expect finalized value from socket
    client_conn_->has_pending = false;
    auto &copy_client = engine_->getCopyClient();
    copy_client.freeConnection(client_conn_);
    client_conn_ = nullptr;
    copy_client.freeCtrlBlock(tcp_ctrl_block_);
    tcp_ctrl_block_ = nullptr;
}
