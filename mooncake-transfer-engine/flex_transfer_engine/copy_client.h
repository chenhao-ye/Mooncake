#pragma once

#include <sys/socket.h>
#include <unistd.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
    CopyClient(const std::string local_segment_name)
        : local_segment_name_(local_segment_name) {}
    ~CopyClient();

    ClientConnection *allocConnection(const std::string &server_url);

    void freeConnection(ClientConnection *conn);

    /**
     * Submit a batch of transfer requests to a remote CopyServer.
     * Assumption: the server machine uses the same endianness as the client.
     * @param entries Vector of transfer requests
     * @param conn Connect to submit requests to
     * @param ctrl_block Control block for progress tracking
     * @return connection pointer
     */
    void submitRdmaRequests(std::vector<transfer_request_t> &entries,
                            ClientConnection *conn,
                            RdmaCopyCtrlBlock *ctrl_block);

   private:
    void writeSegmentName(int fd);
    void writeRdmaRequests(int fd, std::vector<transfer_request_t> &entries,
                           RdmaCopyCtrlBlock *ctrl_block);

    /**
     * Connect to a remote CopyServer.
     * @param server_url URL in format "ip:port"
     * @return File descriptor
     */
    int connectToCopyServer(const std::string &server_url);

    // Let the remote CopyServer know where to submit RDMA write
    const std::string local_segment_name_;

    // Cached TCP connections to remote CopyServers (server_url -> connection)
    std::unordered_map<std::string, ClientConnection *> connection_cache_;
    std::mutex connection_cache_mutex_;
};
