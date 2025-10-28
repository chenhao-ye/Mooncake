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
#include "error.h"

namespace mooncake {

int DirectTransferEngine::init(const std::string &metadata_conn_string,
                               const std::string &local_server_name,
                               const std::string &ip_or_host_name,
                               uint64_t rpc_port) {
    int ret = engine_->init(metadata_conn_string, local_server_name,
                            ip_or_host_name, rpc_port);
    if (ret < 0) {
        return ret;
    }

    // Register progress address for CopyTransferEngine transfers
    ret = engine_->registerLocalMemory(&progress_, sizeof(progress_),
                                       kWildcardLocation, true, true);
    if (ret < 0) {
        std::cerr << "Failed to register progress memory" << std::endl;
        return ret;
    }
    progress_registered_ = true;

    return 0;
}

DirectTransferEngine::~DirectTransferEngine() {
    // Unregister progress address if it was registered
    if (progress_registered_) {
        engine_->unregisterLocalMemory(&progress_, true);
    }
}

Status DirectTransferEngine::submitTransfer(
    BatchID batch_id, const std::vector<TransferRequest> &entries,
    bool target_is_copy_engine) {
    if (!target_is_copy_engine) {
        // Normal transfer through underlying TransferEngine
        return engine_->submitTransfer(batch_id, entries);
    }

    // Transfer to CopyTransferEngine
    return submitTransferToCopyEngine(batch_id, entries);
}

Status DirectTransferEngine::submitTransferToCopyEngine(
    BatchID batch_id, const std::vector<TransferRequest> &entries) {
    // Verify all entries are read requests
    for (const auto &entry : entries) {
        if (entry.opcode != TransferRequest::READ) {
            std::cerr << "Only read requests are supported when target is "
                         "CopyTransferEngine"
                      << std::endl;
            return Status::InvalidArgument(
                "Only read requests supported for CopyTransferEngine");
        }
    }

    if (entries.empty()) {
        return Status::OK();
    }

    // Get the target segment from metadata
    // For simplicity, we'll use the local server name as target
    // In a real implementation, this should be obtained from the segment
    // descriptor
    std::string server_name =
        "target_server";  // TODO: Get from segment metadata

    // Connect to the CopyTransferEngine (or reuse existing connection)
    int fd = -1;
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = copy_engine_connections_.find(server_name);
        if (it != copy_engine_connections_.end()) {
            fd = it->second;
        } else {
            // Default port for CopyTransferEngine TCP listener
            fd = connectToCopyEngine(server_name, 12346);
            if (fd < 0) {
                std::cerr << "Failed to connect to CopyTransferEngine at "
                          << server_name << std::endl;
                return Status::Socket(
                    "Failed to connect to CopyTransferEngine");
            }
            copy_engine_connections_[server_name] = fd;
        }
    }

    // Reset the pre-registered progress counter
    progress_.store(0);

    // Send batch info to CopyTransferEngine:
    // 1. Progress address (8 bytes)
    // 2. Number of requests (8 bytes)
    // 3. For each request: source_addr (8 bytes), target_addr (8 bytes), length
    // (8 bytes)

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
        return Status::Socket("Failed to send batch info");
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
            return Status::Socket("Failed to send request info");
        }
    }

    // The CopyTransferEngine will now process the requests asynchronously
    // and update the progress counter via RDMA writes.
    // The caller should poll the progress or use getTransferStatus to check
    // completion.

    std::cerr << "Submitted " << entries.size()
              << " requests to CopyTransferEngine at " << server_name
              << std::endl;

    return Status::OK();
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
