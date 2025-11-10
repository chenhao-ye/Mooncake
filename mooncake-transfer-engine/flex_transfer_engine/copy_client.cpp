#include "copy_client.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "copy_backend_tcp.h"
#include "copy_common.h"
#include "flex_transfer_engine.h"
#include "util.h"

CopyClient::~CopyClient() {
    std::lock_guard<std::mutex> lock(connection_cache_mutex_);
    for (const auto &[_, conn] : connection_cache_) {
        if (conn) conn->cleanup();
    }
}

ClientConnection *CopyClient::allocConnection(const std::string &server_url) {
    std::lock_guard<std::mutex> lock(connection_cache_mutex_);
    ClientConnection *conn = connection_cache_[server_url];

    if (conn) {
        connection_cache_[server_url] = nullptr;
        conn->clear_pending();
        return conn;
    }

    int fd = connectToCopyServer(server_url);
    conn = new ClientConnection(fd, server_url);
    return conn;
}

void CopyClient::freeConnection(ClientConnection *conn) {
    {  // if possible, return to the cache
        std::lock_guard<std::mutex> lock(connection_cache_mutex_);
        auto &prev_conn = connection_cache_[conn->server_url];
        if (!prev_conn) {
            prev_conn = conn;
            return;
        }
    }
    // otherwise, close this connection and free it
    conn->cleanup();
    delete conn;
}

RdmaCopyCtrlBlock *CopyClient::submitRdmaRequests(
    ClientConnection *conn, std::vector<transfer_request_t> &entries) {
    RdmaCopyCtrlBlock *ctrl_block = rdma_copy_backend_.allocCtrlBlock();
    CopyMode mode = CopyMode::RDMA;
    writeFully(conn->fd, &mode, sizeof(mode));
    writeSegmentName(conn->fd);
    writeRdmaRequests(conn->fd, entries, ctrl_block);

    // The remote CopyServer will now process the requests asynchronously
    // and update the progress counter via RDMA writes. The caller should
    // poll the progress to check completion.

    std::cerr << "Submitted " << entries.size() << " requests to CopyServer at "
              << conn->server_url << std::endl;
    conn->has_pending = true;
    return ctrl_block;
}

TcpCopyCtrlBlock *CopyClient::submitTcpRequests(
    ClientConnection *conn, std::vector<transfer_request_t> &entries) {
    TcpCopyCtrlBlock *ctrl_block = tcp_copy_backend_.allocCtrlBlock();
    CopyMode mode = CopyMode::TCP;
    writeFully(conn->fd, &mode, sizeof(mode));
    writeTcpRequests(conn->fd, entries);
    {
        std::lock_guard<std::mutex> lock(ctrl_block->mutex_);
        ctrl_block->server_fd = conn->fd;
        ctrl_block->tasks.clear();
        ctrl_block->tasks.reserve(entries.size());
        for (const auto &entry : entries)
            ctrl_block->tasks.emplace_back(entry.source, entry.length);
    }
    ctrl_block->cv_.notify_one();
    return ctrl_block;
}

void CopyClient::writeSegmentName(int fd) {
    ssize_t nbytes;

    // Send segment name length
    uint32_t segment_name_len = local_segment_name_.size();
    nbytes = writeFully(fd, &segment_name_len, sizeof(segment_name_len));
    if (nbytes != sizeof(segment_name_len)) {
        throw std::runtime_error(
            "Failed to send segment name length to CopyServer");
    }

    // Send segment name
    nbytes = writeFully(fd, local_segment_name_.c_str(), segment_name_len);
    if (nbytes != segment_name_len)
        throw std::runtime_error("Failed to send segment name to CopyServer");
}

void CopyClient::writeRdmaRequests(int fd,
                                   std::vector<transfer_request_t> &entries,
                                   RdmaCopyCtrlBlock *ctrl_block) {
    ssize_t nbytes;
    RdmaHeader header{.progress_addr = reinterpret_cast<uint64_t>(
                          &ctrl_block->progress_counter),
                      .num_reqs = entries.size()};

    nbytes = writeFully(fd, &header, sizeof(header));
    if (nbytes != sizeof(header))
        throw std::runtime_error("Failed to send batch info to CopyServer");

    std::vector<RdmaReq> reqs;
    reqs.reserve(entries.size());
    for (const auto &entry : entries) {
        if (entry.opcode != OPCODE_READ)
            throw std::invalid_argument(
                "Only read operations are supported in copy mode");
        reqs.emplace_back(reinterpret_cast<uint64_t>(entry.source),
                          entry.target_offset, entry.length);
    }
    size_t reqs_nbytes = sizeof(RdmaReq) * reqs.size();

    nbytes = writeFully(fd, reqs.data(), reqs_nbytes);
    if (nbytes != static_cast<ssize_t>(reqs_nbytes))
        throw std::runtime_error("Failed to send request info to CopyServer");
}

void CopyClient::writeTcpRequests(int fd,
                                  std::vector<transfer_request_t> &entries) {
    ssize_t nbytes;
    TcpHeader header{.num_reqs = entries.size()};

    nbytes = writeFully(fd, &header, sizeof(header));
    if (nbytes != sizeof(header))
        throw std::runtime_error("Failed to send TCP batch info to CopyServer");

    std::vector<TcpReq> reqs;
    reqs.reserve(entries.size());
    for (const auto &entry : entries) {
        if (entry.opcode != OPCODE_READ)
            throw std::invalid_argument(
                "Only read operations are supported in copy mode");
        reqs.emplace_back(entry.target_offset, entry.length);
    }
    size_t reqs_nbytes = sizeof(TcpReq) * reqs.size();

    nbytes = writeFully(fd, reqs.data(), reqs_nbytes);
    if (nbytes != static_cast<ssize_t>(reqs_nbytes))
        throw std::runtime_error(
            "Failed to send TCP request info to CopyServer");
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

    std::cerr << "Connected to CopyServer at " << server_url << std::endl;
    return fd;
}
