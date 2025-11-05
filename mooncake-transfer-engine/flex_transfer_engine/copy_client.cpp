#include "copy_client.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <cstring>
#include <iostream>

#include "flex_transfer_engine.h"
#include "util.h"

CopyClient::~CopyClient() {
    for (const auto &[_, conn] : copy_server_connections_) close(conn.fd);
}

void CopyClient::submitTransferToCopyServer(
    std::vector<transfer_request_t> &entries, const std::string &server_url,
    CopyCtrlBlock *ctrl_block, ClientConnection **out_conn) {
    if (entries.empty()) return;
    assert(ctrl_block);

    // Connect to the remote CopyServer (or reuse existing connection)
    ClientConnection *conn = nullptr;
    int fd = -1;
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = copy_server_connections_.find(server_url);
        if (it != copy_server_connections_.end()) {
            conn = &it->second;
            fd = conn->fd;
            // Reset progress tracking for new batch
            // TODO: clear if not finalized; check fd validity
            conn->last_progress = 0;
            conn->finalized_received = false;
            conn->finalized_value = 0;
        } else {
            fd = connectToCopyServer(server_url);
            auto result = copy_server_connections_.emplace(
                std::piecewise_construct, std::forward_as_tuple(server_url),
                std::forward_as_tuple(fd));
            conn = &result.first->second;
        }
    }

    // Return connection pointer if requested
    if (out_conn) *out_conn = conn;

    // Send protocol to remote CopyServer:
    // 1. Segment name length (4 bytes)
    // 2. Segment name (variable length)
    // 3. Progress address (8 bytes)
    // 4. Number of requests (8 bytes)
    // 5. For each request: source_addr (8 bytes), target_addr (8 bytes), length
    // (8 bytes)

    // Get local server name from FlexTransferEngine
    const std::string &local_server_name = engine_.getLocalServerName();

    ssize_t nbytes;

    // Send segment name length
    uint32_t segment_name_len = local_server_name.size();
    nbytes = writeFully(fd, &segment_name_len, sizeof(segment_name_len));
    if (nbytes != sizeof(segment_name_len)) {
        throw std::runtime_error(
            "Failed to send segment name length to CopyServer");
    }

    // Send segment name
    nbytes = writeFully(fd, local_server_name.c_str(), segment_name_len);
    if (nbytes != segment_name_len)
        throw std::runtime_error("Failed to send segment name to CopyServer");

    // Send batch info
    struct BatchInfo {
        uint64_t progress_addr;
        uint64_t num_requests;
    };

    BatchInfo batch_info;
    batch_info.progress_addr =
        reinterpret_cast<uint64_t>(&ctrl_block->progress_counter);
    batch_info.num_requests = entries.size();

    nbytes = writeFully(fd, &batch_info, sizeof(batch_info));
    if (nbytes != sizeof(batch_info))
        throw std::runtime_error("Failed to send batch info to CopyServer");

    // Send request details (source on remote CopyServer, target on local)
    for (const auto &entry : entries) {
        assert(entry.opcode == OPCODE_READ);
        struct RequestInfo {
            uint64_t source_addr;
            uint64_t target_addr;
            uint64_t length;
        };

        RequestInfo req_info;
        req_info.source_addr = reinterpret_cast<uint64_t>(entry.source);
        req_info.target_addr = entry.target_offset;
        req_info.length = entry.length;

        nbytes = writeFully(fd, &req_info, sizeof(req_info));
        if (nbytes != sizeof(req_info)) {
            throw std::runtime_error(
                "Failed to send request info to CopyServer");
        }
    }

    // The remote CopyServer will now process the requests asynchronously
    // and update the progress counter via RDMA writes. The caller should
    // poll the progress to check completion.

    std::cerr << "Submitted " << entries.size() << " requests to CopyServer at "
              << server_url << std::endl;
}

int CopyClient::connectToCopyServer(const std::string &server_url) {
    // Parse server_url in format "ip_addr:port"
    // For IPv6: "[ipv6_addr]:port" or "ipv6_addr:port"
    // For IPv4: "ipv4_addr:port"

    std::string hostname;
    std::string port_str;

    // Check for IPv6 format with brackets: [addr]:port
    if (server_url[0] == '[') {
        size_t bracket_end = server_url.find(']');
        if (bracket_end == std::string::npos) {
            throw std::invalid_argument("Invalid IPv6 format in server_url: " +
                                        server_url);
        }
        hostname = server_url.substr(1, bracket_end - 1);

        // Check if there's a port after the bracket
        if (bracket_end + 1 < server_url.length() &&
            server_url[bracket_end + 1] == ':') {
            port_str = server_url.substr(bracket_end + 2);
        } else {
            throw std::invalid_argument("Missing port in server_url: " +
                                        server_url);
        }
    } else {
        // IPv4 format or IPv6 without brackets
        size_t last_colon = server_url.rfind(':');
        if (last_colon == std::string::npos) {
            throw std::invalid_argument("Missing port in server_url: " +
                                        server_url);
        }
        hostname = server_url.substr(0, last_colon);
        port_str = server_url.substr(last_colon + 1);
    }

    // Resolve hostname
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;  // Support both IPv4 and IPv6
    hints.ai_socktype = SOCK_STREAM;

    int rc = getaddrinfo(hostname.c_str(), port_str.c_str(), &hints, &result);
    if (rc) {
        throw std::runtime_error("getaddrinfo failed for " + server_url + ": " +
                                 gai_strerror(rc));
    }

    // Try to connect
    int fd = -1;
    for (struct addrinfo *rp = result; rp != nullptr; rp = rp->ai_next) {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) break;  // Success
        close(fd);
        fd = -1;
    }

    freeaddrinfo(result);

    if (fd < 0) throw std::runtime_error("Failed to connect to " + server_url);

    // Set socket to non-blocking for progress checking
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    std::cerr << "Connected to CopyServer at " << server_url << std::endl;
    return fd;
}
