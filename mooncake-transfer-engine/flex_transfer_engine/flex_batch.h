#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "copy_backend_rdma.h"
#include "copy_backend_tcp.h"
#include "copy_common.h"
#include "transfer_engine_c.h"

struct ClientConnection;
class FlexTransferEngine;

/**
 * FlexBatch represents a batch transfer operation.
 * It manages the batch lifecycle including freeing the batch ID and
 * returning the RdmaCopyCtrlBlock to the engine cache on destruction.
 */
class FlexBatch {
   public:
    /**
     * Constructor.
     * @param engine Backpointer to the FlexTransferEngine
     */
    explicit FlexBatch(std::shared_ptr<FlexTransferEngine> engine)
        : engine_(std::move(engine)),
          is_direct_(true),
          batch_id_(INVALID_BATCH),
          client_conn_(nullptr),
          known_progress_(0),
          rdma_ctrl_block_(nullptr),
          tcp_ctrl_block_(nullptr) {}

    ~FlexBatch() { free(); }

    void addReadRequest(uintptr_t local_addr, uintptr_t remote_addr,
                        uint64_t size);
    void addWriteRequest(uintptr_t local_addr, uintptr_t remote_addr,
                         uint64_t size);
    void addFetchAddRequest(uintptr_t local_addr, uintptr_t remote_addr,
                            uint64_t value);

    /**
     * Submit the transfer batch.
     *
     * @param target Remote target, either segment name or copy server URL
     * (formatted as "ip:port").
     * @param is_direct If true, issue direct EDMA-read, in which case target
     * should be a segment name; otherwise, target is a copy server URL.
     * @param use_rdma If true, use RDMA; if false, use TCP. Ignored if
     * is_direct is true (direct mode must use RDMA).
     */
    int submit(const std::string &target, bool is_direct = true,
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
    // read ctrl block and save progress into known_progress_
    void checkRdmaProgress();
    void checkTcpProgress();

   private:
    std::shared_ptr<FlexTransferEngine> engine_;
    std::vector<transfer_request_t> entries_;

    bool is_direct_;

    /* for direct transfer */
    batch_id_t batch_id_;

    /* for copy transfer */
    // once this batch is considered done with the connection (no more progress
    // will be made), client_conn_ and xxx_ctrl_block_ will be set to nullptr.
    // - for RDMA-based copy, it requires seeing a progress that implies all
    //   requests completed OR received a int32_t from the socket.
    // - for TCP-based copy, it requires seeing a progress that implies all
    //   requests completed OR notice the mutex on the ctrl block is available,
    //   meaning the worker thread has finalized it.
    ClientConnection *client_conn_;
    int64_t known_progress_;

    RdmaCopyCtrlBlock *rdma_ctrl_block_;
    TcpCopyCtrlBlock *tcp_ctrl_block_;

    // only valid for copy-based transfer
    bool isCopyFinalized() { return !client_conn_; }
};

struct MemoryBatch {
    std::unordered_map<std::string, std::vector<buffer_entry_t>>
        location_buffers_map;

    void add(void *addr, size_t length, const std::string &location) {
        location_buffers_map[location].emplace_back(
            buffer_entry_t{addr, length});
    };
};
