#include "batch.h"

#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <iostream>

#include "flex_transfer_engine.h"
#include "transfer_engine_c.h"

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

// Helper function to check finalized progress from socket
static int checkFinalizedProgress(ClientConnection *conn) {
    if (!conn) return -1;

    // Socket is already non-blocking (set during connection creation)
    // Try to read the finalized int32_t from socket
    int32_t finalized_value = 0;
    ssize_t bytes_read =
        recv(conn->fd, &finalized_value, sizeof(finalized_value), 0);

    if (bytes_read == sizeof(finalized_value)) {
        // Successfully read the finalized value
        conn->finalized_received = true;
        conn->finalized_value = finalized_value;
        std::cerr << "Received finalized progress: " << finalized_value
                  << std::endl;
        return 0;
    } else if (bytes_read == 0) {
        // Connection closed
        std::cerr << "Connection closed by server" << std::endl;
        return -1;
    } else if (bytes_read < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        // No data available yet
        return 0;
    } else {
        // Error
        std::cerr << "Error reading finalized progress: " << strerror(errno)
                  << std::endl;
        return -1;
    }
}

int FlexBatch::submit(const std::string &target, bool is_target_copy) {
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
    copy_ctrl_block_ = engine_->acquireCopyCtrlBlock();
    if (!copy_ctrl_block_) return -1;
    return engine_->submitTransferToCopyEngine(entries_, target,
                                               copy_ctrl_block_, &client_conn_);
}

int FlexBatch::getTransferStatus(size_t task_id, transfer_status_t &status) {
    if (!copy_ctrl_block_) {  // Direct RDMA transfer
        return ::getTransferStatus(engine_->getEngine(), batch_id_, task_id,
                                   &status);
    }

    // For copy-based transfers, check ctrl_block progress
    if (!client_conn_) return -1;

    int64_t progress;

    // If we've already received the finalized value, use it
    if (client_conn_->finalized_received) {
        progress = client_conn_->finalized_value;
    } else {
        // Check the current progress counter
        progress =
            copy_ctrl_block_->progress_counter.load(std::memory_order_acquire);

        // If progress hasn't changed, check the socket for finalized value
        if (progress == client_conn_->last_progress) {
            int rc = checkFinalizedProgress(client_conn_);
            if (rc < 0) return rc;  // Error occurred

            // If finalized value was received, use it
            if (client_conn_->finalized_received)
                progress = client_conn_->finalized_value;
        } else {  // progress has changed, update last_progress
            client_conn_->last_progress = progress;
        }
    }

    // Determine task status based on progress
    if (static_cast<int64_t>(task_id) < progress) {  // Task completed
        status.status = STATUS_COMPLETED;
        status.transferred_bytes = entries_[task_id].length;
    } else {
        // if already finalized, the given task will never complete
        status.status =
            client_conn_->finalized_received ? STATUS_FAILED : STATUS_WAITING;
        status.transferred_bytes = 0;
    }
    return 0;
}
