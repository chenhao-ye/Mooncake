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

#include <unistd.h>

#include <cstring>
#include <iostream>
#include <vector>

#include "copy_transfer_engine.h"
#include "direct_transfer_engine.h"

using namespace mooncake;

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <mode> <metadata_server> <local_server_name>"
                  << std::endl;
        std::cerr << "  mode: 'copy' or 'direct'" << std::endl;
        std::cerr << "  metadata_server: e.g., 'http://127.0.0.1:8080/metadata'"
                  << std::endl;
        std::cerr << "  local_server_name: e.g., '127.0.0.1'" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    std::string metadata_server = argv[2];
    std::string local_server_name = argv[3];

    if (mode == "copy") {
        // Run as CopyTransferEngine (target/server)
        std::cerr << "Starting CopyTransferEngine..." << std::endl;

        CopyTransferEngine copy_engine;
        int ret = copy_engine.init(metadata_server, local_server_name, "",
                                   12345, 12346, 1);
        if (ret < 0) {
            std::cerr << "Failed to initialize CopyTransferEngine" << std::endl;
            return 1;
        }

        std::cerr << "CopyTransferEngine initialized on port "
                  << copy_engine.getTcpPort() << std::endl;

        // Allocate and register some test memory
        const size_t buffer_size = 1024 * 1024;  // 1 MB
        void *test_buffer = malloc(buffer_size);
        if (!test_buffer) {
            std::cerr << "Failed to allocate test buffer" << std::endl;
            return 1;
        }

        // Fill buffer with test data
        memset(test_buffer, 0xAB, buffer_size);

        // Register the buffer
        ret = copy_engine.registerLocalMemory(test_buffer, buffer_size, "cpu",
                                              1);
        if (ret < 0) {
            std::cerr << "Failed to register test buffer" << std::endl;
            free(test_buffer);
            return 1;
        }

        std::cerr << "Registered test buffer at " << test_buffer << " size "
                  << buffer_size << std::endl;
        std::cerr << "CopyTransferEngine ready. Press Ctrl+C to exit."
                  << std::endl;

        // Keep running
        while (true) {
            sleep(1);
        }

        free(test_buffer);

    } else if (mode == "direct") {
        // Run as DirectTransferEngine (initiator/client)
        std::cerr << "Starting DirectTransferEngine..." << std::endl;

        DirectTransferEngine direct_engine;
        int ret =
            direct_engine.init(metadata_server, local_server_name, "", 12345, 1);
        if (ret < 0) {
            std::cerr << "Failed to initialize DirectTransferEngine"
                      << std::endl;
            return 1;
        }

        std::cerr << "DirectTransferEngine initialized" << std::endl;

        // Wait a bit for CopyTransferEngine to be ready
        sleep(2);

        // Allocate local buffer to receive data
        const size_t buffer_size = 1024 * 1024;  // 1 MB
        void *local_buffer = malloc(buffer_size);
        if (!local_buffer) {
            std::cerr << "Failed to allocate local buffer" << std::endl;
            return 1;
        }

        memset(local_buffer, 0, buffer_size);

        // Register local buffer
        ret = direct_engine.registerLocalMemory(local_buffer, buffer_size,
                                                "cpu", 1);
        if (ret < 0) {
            std::cerr << "Failed to register local buffer" << std::endl;
            free(local_buffer);
            return 1;
        }

        std::cerr << "Registered local buffer at " << local_buffer << " size "
                  << buffer_size << std::endl;

        // TODO: In a real implementation, we would:
        // 1. Open the remote segment
        // 2. Create transfer requests to read from CopyTransferEngine
        // 3. Submit the transfer with copy_server_name and copy_server_port
        // 4. Wait for completion
        // 5. Verify the data

        std::cerr << "DirectTransferEngine setup complete" << std::endl;
        std::cerr
            << "Note: Full transfer test requires additional implementation"
            << std::endl;

        free(local_buffer);

    } else {
        std::cerr << "Invalid mode: " << mode << std::endl;
        std::cerr << "Must be 'copy' or 'direct'" << std::endl;
        return 1;
    }

    return 0;
}
