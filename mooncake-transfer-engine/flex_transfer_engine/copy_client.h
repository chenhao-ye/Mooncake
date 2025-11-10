#pragma once

#include <sys/socket.h>
#include <unistd.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "copy_backend_rdma.h"
#include "copy_backend_tcp.h"
#include "copy_common.h"
#include "transfer_engine_c.h"

class FlexTransferEngine;

struct ClientConnection {
    int fd;
    // it is expected that a connection is returned to cache while it still has
    // a 4-byte finalized value (has_pending=True); this ensures a client can
    // return as quick as possible once see a valid progress via RDMA, without
    // waiting for socket
    bool has_pending;  // if true, this fd has a 4-byte value to read
    std::string server_url;
    ClientConnection(int fd, const std::string &server_url)
        : fd(fd), has_pending(false), server_url(server_url) {}

    void clear_pending() {
        if (!has_pending) return;
        int32_t dummy;
        ssize_t nbytes = recv(fd, &dummy, sizeof(dummy), 0);
        if (nbytes != sizeof(dummy))
            throw std::runtime_error("Fail to clear ClientConnection");
    }

    void free() {
        if (fd >= 0) close(fd);
        fd = -1;
    }
};

/**
 * CopyClient handles client-side copy-based transfer operations.
 * It manages TCP connections to remote CopyServers and submits
 * transfer requests via the copy protocol.
 */
class CopyClient {
   public:
    CopyClient(RdmaCopyBackend &rdma_copy_backend,
               TcpCopyBackend &tcp_copy_backend,
               const std::string local_segment_name)
        : rdma_copy_backend_(rdma_copy_backend),
          tcp_copy_backend_(tcp_copy_backend),
          local_segment_name_(local_segment_name) {}
    ~CopyClient();

    ClientConnection *allocConnection(const std::string &server_url);

    void freeConnection(ClientConnection *conn);

    RdmaCopyCtrlBlock *submitRdmaRequests(
        ClientConnection *conn, std::vector<transfer_request_t> &entries);

    TcpCopyCtrlBlock *submitTcpRequests(
        ClientConnection *conn, std::vector<transfer_request_t> &entries);

    void freeCtrlBlock(RdmaCopyCtrlBlock *ctrl_block) {
        rdma_copy_backend_.freeCtrlBlock(ctrl_block);
    }
    void freeCtrlBlock(TcpCopyCtrlBlock *ctrl_block) {
        tcp_copy_backend_.freeCtrlBlock(ctrl_block);
    }

   private:
    void writeSegmentName(int fd);
    void writeRdmaRequests(int fd, std::vector<transfer_request_t> &entries,
                           RdmaCopyCtrlBlock *ctrl_block);
    void writeTcpRequests(int fd, std::vector<transfer_request_t> &entries);

    /**
     * Connect to a remote CopyServer.
     * @param server_url URL in format "ip:port"
     * @return server fd
     */
    int connectToCopyServer(const std::string &server_url);

    RdmaCopyBackend &rdma_copy_backend_;
    TcpCopyBackend &tcp_copy_backend_;

    // Let the remote CopyServer know where to submit RDMA write
    const std::string local_segment_name_;

    // Cached TCP connections to remote CopyServers (server_url -> connection)
    std::unordered_map<std::string, ClientConnection *> connection_cache_;
    std::mutex connection_cache_mutex_;
};
