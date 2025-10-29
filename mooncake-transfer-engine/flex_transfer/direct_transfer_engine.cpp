// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "direct_transfer_engine.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

#include "common.h"

namespace mooncake {

int DirectTransferEngine::init(const std::string &metadata_conn_string,
                               const std::string &local_server_name,
                               const std::string &ip_or_host_name,
                               uint64_t rpc_port, int auto_discover) {
    local_server_name_ = local_server_name;

    engine_ = createTransferEngine(
        metadata_conn_string.c_str(), local_server_name.c_str(),
        ip_or_host_name.empty() ? nullptr : ip_or_host_name.c_str(), rpc_port,
        auto_discover);
    if (engine_ == nullptr) {
        std::cerr << "Failed to create TransferEngine" << std::endl;
        return -1;
    }

    // Register progress address for CopyTransferEngine transfers
    int ret = registerLocalMemory(&progress_, sizeof(progress_), "", 1);
    if (ret < 0) {
        std::cerr << "Failed to register progress memory" << std::endl;
        destroyTransferEngine(engine_);
        engine_ = nullptr;
        return ret;
    }
    progress_registered_ = true;

    return 0;
}

DirectTransferEngine::~DirectTransferEngine() {
    // Unregister progress address if it was registered
    if (progress_registered_ && engine_ != nullptr) {
        unregisterLocalMemory(&progress_);
    }

    // Destroy the engine
    if (engine_ != nullptr) {
        destroyTransferEngine(engine_);
    }

    // Close all TCP connections
    for (const auto &[_, fd] : copy_engine_connections_) {
        close(fd);
    }
}

segment_id_t DirectTransferEngine::openSegment(
    const std::string &segment_name) {
    return ::openSegment(engine_, segment_name.c_str());
}

int DirectTransferEngine::closeSegment(segment_id_t segment_id) {
    return ::closeSegment(engine_, segment_id);
}

int DirectTransferEngine::registerLocalMemory(void *addr, size_t length,
                                               const std::string &location,
                                               int remote_accessible) {
    return ::registerLocalMemory(
        engine_, addr, length,
        location.empty() ? nullptr : location.c_str(), remote_accessible);
}

int DirectTransferEngine::unregisterLocalMemory(void *addr) {
    return ::unregisterLocalMemory(engine_, addr);
}

int DirectTransferEngine::registerLocalMemoryBatch(
    const std::vector<buffer_entry_t> &buffer_list,
    const std::string &location) {
    std::vector<buffer_entry_t> buf_list = buffer_list;
    return ::registerLocalMemoryBatch(engine_, buf_list.data(),
                                      buf_list.size(), location.c_str());
}

int DirectTransferEngine::unregisterLocalMemoryBatch(
    const std::vector<void *> &addr_list) {
    std::vector<void *> addrs = addr_list;
    return ::unregisterLocalMemoryBatch(engine_, addrs.data(), addrs.size());
}

int DirectTransferEngine::syncSegmentCache() {
    return ::syncSegmentCache(engine_);
}

batch_id_t DirectTransferEngine::allocateBatchID(size_t batch_size) {
    return ::allocateBatchID(engine_, batch_size);
}

int DirectTransferEngine::freeBatchID(batch_id_t batch_id) {
    return ::freeBatchID(engine_, batch_id);
}

int DirectTransferEngine::submitTransfer(
    batch_id_t batch_id, const std::vector<transfer_request_t> &entries,
    const std::string &copy_server_name, uint16_t copy_server_port) {
    if (copy_server_name.empty()) {
        // Normal transfer through underlying TransferEngine
        std::vector<transfer_request_t> reqs = entries;
        return ::submitTransfer(engine_, batch_id, reqs.data(), reqs.size());
    }

    // Transfer to CopyTransferEngine
    return submitTransferToCopyEngine(batch_id, entries, copy_server_name,
                                      copy_server_port);
}

int DirectTransferEngine::getTransferStatus(batch_id_t batch_id,
                                             size_t task_id,
                                             transfer_status_t &status) {
    return ::getTransferStatus(engine_, batch_id, task_id, &status);
}

int DirectTransferEngine::submitTransferToCopyEngine(
    batch_id_t batch_id, const std::vector<transfer_request_t> &entries,
    const std::string &server_name, uint16_t port) {
    // Verify all entries are read requests
    for (const auto &entry : entries) {
        if (entry.opcode != OPCODE_READ) {
            std::cerr << "Only read requests are supported when target is "
                         "CopyTransferEngine"
                      << std::endl;
            return -1;
        }
    }

    if (entries.empty()) {
        return 0;
    }

    // Connect to the CopyTransferEngine (or reuse existing connection)
    int fd = -1;
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = copy_engine_connections_.find(server_name);
        if (it != copy_engine_connections_.end()) {
            fd = it->second;
        } else {
            fd = connectToCopyEngine(server_name, port);
            if (fd < 0) {
                std::cerr << "Failed to connect to CopyTransferEngine at "
                          << server_name << std::endl;
                return -1;
            }
            copy_engine_connections_[server_name] = fd;
        }
    }

    // Reset the pre-registered progress counter
    progress_.store(0);

    // Send protocol to CopyTransferEngine:
    // 1. Segment name length (4 bytes)
    // 2. Segment name (variable length)
    // 3. Progress address (8 bytes)
    // 4. Number of requests (8 bytes)
    // 5. For each request: source_addr (8 bytes), target_addr (8 bytes), length
    // (8 bytes)

    // Send segment name length
    uint32_t segment_name_len = local_server_name_.size();
    if (writeFully(fd, &segment_name_len, sizeof(segment_name_len)) !=
        sizeof(segment_name_len)) {
        std::cerr << "Failed to send segment name length to CopyTransferEngine"
                  << std::endl;
        return -1;
    }

    // Send segment name
    if (writeFully(fd, local_server_name_.c_str(), segment_name_len) !=
        segment_name_len) {
        std::cerr << "Failed to send segment name to CopyTransferEngine"
                  << std::endl;
        return -1;
    }

    // Send batch info
    struct BatchInfo {
        uint64_t progress_addr;
        uint64_t num_requests;
    };

    BatchInfo batch_info;
    batch_info.progress_addr = reinterpret_cast<uint64_t>(&progress_);
    batch_info.num_requests = entries.size();

    if (writeFully(fd, &batch_info, sizeof(batch_info)) != sizeof(batch_info)) {
        std::cerr << "Failed to send batch info to CopyTransferEngine"
                  << std::endl;
        return -1;
    }

    // Send request details (source on CopyTransferEngine, target on
    // DirectTransferEngine)
    for (const auto &entry : entries) {
        struct RequestInfo {
            uint64_t source_addr;
            uint64_t target_addr;
            uint64_t length;
        };

        RequestInfo req_info;
        req_info.source_addr = reinterpret_cast<uint64_t>(entry.source);
        req_info.target_addr = entry.target_offset;
        req_info.length = entry.length;

        if (writeFully(fd, &req_info, sizeof(req_info)) != sizeof(req_info)) {
            std::cerr << "Failed to send request info to CopyTransferEngine"
                      << std::endl;
            return -1;
        }
    }

    // The CopyTransferEngine will now process the requests asynchronously
    // and update the progress counter via RDMA writes.
    // The caller should poll the progress or use getTransferStatus to check
    // completion.

    std::cerr << "Submitted " << entries.size()
              << " requests to CopyTransferEngine at " << server_name
              << std::endl;

    return 0;
}

int DirectTransferEngine::connectToCopyEngine(const std::string &server_name,
                                               uint16_t port) {
    // Parse server name and port
    auto [hostname, parsed_port] = parseHostNameWithPort(server_name);
    // Use parsed port if it's different from the default
    if (parsed_port != 0) {
        port = parsed_port;
    }

    // Resolve hostname
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(port);
    int ret = getaddrinfo(hostname.c_str(), port_str.c_str(), &hints, &result);
    if (ret != 0) {
        std::cerr << "getaddrinfo failed: " << gai_strerror(ret) << std::endl;
        return -1;
    }

    // Try to connect
    int fd = -1;
    for (struct addrinfo *rp = result; rp != nullptr; rp = rp->ai_next) {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0) {
            continue;
        }

        if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
            break;  // Success
        }

        close(fd);
        fd = -1;
    }

    freeaddrinfo(result);

    if (fd < 0) {
        std::cerr << "Failed to connect to " << hostname << ":" << port
                  << std::endl;
        return -1;
    }

    std::cerr << "Connected to CopyTransferEngine at " << hostname << ":"
              << port << std::endl;
    return fd;
}

}  // namespace mooncake
