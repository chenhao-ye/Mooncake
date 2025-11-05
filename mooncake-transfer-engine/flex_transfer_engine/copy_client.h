#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "transfer_engine_c.h"

// Forward declarations
struct CopyCtrlBlock;
class FlexTransferEngine;

// Client-side connection state (combines fd + progress tracking)
struct ClientConnection {
    int fd;
    int64_t last_progress;
    bool finalized_received;
    int32_t finalized_value;

    ClientConnection(int fd)
        : fd(fd),
          last_progress(0),
          finalized_received(false),
          finalized_value(0) {}
};

/**
 * CopyClient handles client-side copy-based transfer operations.
 * It manages TCP connections to remote CopyServers and submits
 * transfer requests via the copy protocol.
 */
class CopyClient {
   public:
    CopyClient(FlexTransferEngine &engine) : engine_(engine) {}
    ~CopyClient();

    /**
     * Submit a batch of transfer requests to a remote CopyServer.
     * @param entries Vector of transfer requests
     * @param server_url URL of remote CopyServer in format "ip:port"
     * @param ctrl_block Control block for progress tracking
     * @return connection pointer
     */
    ClientConnection *submitTransferToCopyServer(
        std::vector<transfer_request_t> &entries, const std::string &server_url,
        CopyCtrlBlock *ctrl_block);

   private:
    /**
     * Connect to a remote CopyServer.
     * @param server_url URL in format "ip:port"
     * @return File descriptor
     */
    int connectToCopyServer(const std::string &server_url);

    FlexTransferEngine &engine_;

    // Cached TCP connections to remote CopyServers (server_url -> connection)
    std::unordered_map<std::string, ClientConnection> copy_server_connections_;
    std::mutex connections_mutex_;
};
