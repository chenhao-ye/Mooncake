# Flex Transfer Engine

When using RDMA transfer, the user must first register the memory, which is costly. It is acceptable as one-time cost, but in some cases, register/unregister can happen very frequently, causing significant overhead. To address this problem, we provide `FlexTransferEngine`, which unifies direct RDMA transfer and copy-based transfer in a single flexible API.

## Implementation Status

This implementation includes:
- `FlexTransferEngine`: Unified transfer engine supporting both direct RDMA and copy-based transfers
- `FlexBatch`: High-level batch transfer abstraction with RAII lifecycle management
- TCP-based protocol for transfer coordination between engines
- Pre-registered RDMA buffer pools (one buffer pair per memory location) for copy-based transfers
- Double buffering support for overlapping copy and RDMA operations
- Support for both CPU and GPU memory (when compiled with CUDA support)
- Single contiguous allocation for buffer pairs to reduce registration overhead
- Connection pooling for efficient TCP connection reuse

## Architecture Overview

### FlexTransferEngine

`FlexTransferEngine` manages the underlying transfer infrastructure and memory registration. It handles:
- RDMA transfer engine lifecycle
- Memory registration/unregistration (with `force_direct` option)
- Segment cache for opened remote segments
- CopyServer (when `enable_copy = true`) for accepting copy requests
- CopyClient for submitting requests to remote CopyServers
- CopyCtrlBlock cache for progress tracking

**Constructor:**
```cpp
FlexTransferEngine(const std::string &metadata_conn_string,
                   const std::string &local_server_name,
                   bool enable_copy,
                   const std::string &ctrl_block_location);
```
- `metadata_conn_string`: Metadata server URL (e.g., "http://127.0.0.1:8080/metadata")
- `local_server_name`: Local segment name for this engine
- `enable_copy`: If true, starts TCP listener for copy-based transfers
- `ctrl_block_location`: Memory location for CopyCtrlBlock (e.g., "cpu:0", must be CPU)

### FlexBatch

`FlexBatch` represents a batch of transfer operations with automatic resource management (RAII). It provides:
- Builder pattern for adding read/write requests
- Automatic cleanup of batch IDs, control blocks, and connections
- Support for both direct RDMA and copy-based transfers
- Progress tracking via status queries

**Key Methods:**
```cpp
void addReadRequest(uintptr_t local_addr, uintptr_t remote_addr, uint64_t size);
void addWriteRequest(uintptr_t local_addr, uintptr_t remote_addr, uint64_t size);
int submit(const std::string &target, bool is_target_copy = false);
int getTransferStatus(size_t task_id);
void free();  // Called automatically by destructor
```

## Copy-Based Transfer Mode

When `enable_copy = true`, the engine:

1. Starts a background TCP listener thread on an automatically selected port (15000-17000 range)
2. Tracks registered memory regions internally (separate from RDMA registration)
3. Manages a buffer pool with one buffer pair per memory location
4. Each buffer pair is allocated as a single contiguous buffer (size 2×largest_region) and split into two halves
5. Buffer pairs are registered with RDMA only once and reused for all transfers

**Memory Registration Behavior:**
- `registerLocalMemory(addr, len, location, remote_accessible, remote_atomic, force_direct=false)`
  - When `force_direct=false` and `enable_copy=true`: Tracks region in CopyServer, allocates/resizes buffer pair
  - When `force_direct=true` or `enable_copy=false`: Performs actual RDMA registration
- `unregisterLocalMemory(addr, force_direct=false)`: Removes from tracking; buffer pairs remain cached
- `registerLocalMemoryBatch(buffer_list, location, force_direct=false)`: Batch version for efficiency
- Note: Locations starting with "cuda:" indicate GPU memory; otherwise, CPU memory

## Transfer Modes

FlexBatch supports two transfer modes via the `submit()` method:

### Direct RDMA Mode
```cpp
FlexBatch batch(engine);
batch.addReadRequest(local_addr, remote_addr, size);
batch.submit("remote_segment_name", false);  // is_target_copy = false
```
- Uses standard RDMA transfer to a remote segment
- `target` parameter is the remote segment name
- No TCP coordination needed

### Copy-Based Mode
```cpp
FlexBatch batch(engine);
batch.addReadRequest(local_addr, remote_addr, size);
batch.submit("192.168.1.100:15234", true);  // is_target_copy = true
```
- Uses TCP-coordinated copy transfer to a remote CopyServer
- `target` parameter is the copy server URL (obtained via `getCopyServerUrl()`)
- Supports reading from frequently-changing memory without RDMA re-registration

**Copy-Based Transfer Protocol:**

Currently only supports read requests (reading from a remote engine with `enable_copy = true`).

1. **Initiator (CopyClient) sends via TCP:**
   - Segment name length (4 bytes)
   - Segment name (variable, the initiator's local segment name)
   - Progress address (8 bytes, for atomic RDMA updates)
   - Number of requests (8 bytes)
   - For each request: source_addr (8 bytes), target_addr (8 bytes), length (8 bytes)
   - **Note:** Protocol uses native endianness (same-endian machines required)

2. **Remote engine (CopyServer) processes:**
   - Opens the initiator's segment (cached for future requests)
   - For each request:
     - Validates source address is in registered `copiable_regions_`
     - Selects buffer from buffer pair (double buffering)
     - Copies data from source to pre-registered buffer (CPU or GPU memory)
     - Submits RDMA write from buffer to initiator's segment
   - Updates progress counter via RDMA atomic fetch-add
   - Sends final count via TCP socket

3. **Initiator polls for completion:**
   - Monitors progress counter via RDMA reads
   - Receives final count via TCP socket
   - Returns connection to pool for reuse

**Key Features:**
- Double buffering enables overlapping of memory copy and RDMA operations
- Connection pooling reduces TCP handshake overhead
- Progress tracking uses RDMA atomics for minimal latency
- Buffer pairs auto-resize when larger regions are registered

## Building

The flex transfer engine is built as part of the main Mooncake build process:

```bash
mkdir build
cd build
cmake ..
make -j
```

The following components are built:
- `libflex_transfer_engine.a` - The flex transfer engine library
- `flex_transfer_example` - Example program (if `BUILD_EXAMPLES=ON`)

## Usage

### Complete Example: Copy-Based Transfer

```cpp
#include "flex_batch.h"
#include "flex_transfer_engine.h"
#include <memory>

// ============================================================
// Target Node: Server with frequently changing memory
// ============================================================
auto copy_engine = std::make_shared<FlexTransferEngine>(
    "http://127.0.0.1:8080/metadata",  // metadata_conn_string
    "192.168.1.100",                   // local_server_name
    true,                              // enable_copy = true
    "cpu:0"                            // ctrl_block_location
);

std::cout << "Copy server listening on: " << copy_engine->getCopyServerUrl()
          << std::endl;

// Allocate and register memory (without RDMA registration)
const size_t data_size = 1024 * 1024;  // 1 MB
void *data = malloc(data_size);
memset(data, 0xAB, data_size);

// Register for copy-based transfer (force_direct = false)
copy_engine->registerLocalMemory(
    reinterpret_cast<uintptr_t>(data),
    data_size,
    "cpu",           // location
    false,           // remote_accessible (not used for copy mode)
    false,           // remote_atomic (not used for copy mode)
    false            // force_direct = false
);

// Keep server running...
// ============================================================

// ============================================================
// Initiator Node: Client that reads data
// ============================================================
auto direct_engine = std::make_shared<FlexTransferEngine>(
    "http://127.0.0.1:8080/metadata",  // metadata_conn_string
    "192.168.1.101",                   // local_server_name
    false,                             // enable_copy = false (client only)
    "cpu:0"                            // ctrl_block_location
);

// Allocate local buffer to receive data
void *local_buffer = malloc(data_size);
memset(local_buffer, 0, data_size);

// Register local buffer with RDMA (force_direct = true or just use default)
direct_engine->registerLocalMemory(
    reinterpret_cast<uintptr_t>(local_buffer),
    data_size,
    "cpu",
    true,            // remote_accessible (for RDMA write from server)
    false,           // remote_atomic
    true             // force_direct = true (actual RDMA registration)
);

// Create a FlexBatch for the transfer
FlexBatch batch(direct_engine);

// Add read request (read from remote server into local buffer)
batch.addReadRequest(
    reinterpret_cast<uintptr_t>(local_buffer),  // local_addr
    reinterpret_cast<uintptr_t>(data),          // remote_addr
    data_size                                    // size
);

// Submit to copy server (get URL from copy_engine->getCopyServerUrl())
std::string copy_server_url = "192.168.1.100:15234";  // From getCopyServerUrl()
int rc = batch.submit(copy_server_url, true);  // is_target_copy = true
if (rc != 0) {
    std::cerr << "Failed to submit batch" << std::endl;
    return -1;
}

// Poll for completion
while (true) {
    int status = batch.getTransferStatus(0);  // Check first task
    if (status == STATUS_COMPLETED) {
        std::cout << "Transfer completed successfully!" << std::endl;
        break;
    } else if (status == STATUS_FAILED) {
        std::cerr << "Transfer failed!" << std::endl;
        return -1;
    }
    usleep(1000);  // Poll every 1ms
}

// Verify data
if (memcmp(local_buffer, data, data_size) == 0) {
    std::cout << "Data verification passed!" << std::endl;
}

// Cleanup is automatic via FlexBatch destructor
free(local_buffer);
free(data);
```

### Example: Direct RDMA Transfer

```cpp
#include "flex_batch.h"
#include "flex_transfer_engine.h"

auto engine = std::make_shared<FlexTransferEngine>(
    "http://127.0.0.1:8080/metadata",
    "192.168.1.100",
    false,  // enable_copy = false
    "cpu:0"
);

// Register local memory with RDMA
void *local_buffer = malloc(size);
engine->registerLocalMemory(
    reinterpret_cast<uintptr_t>(local_buffer),
    size,
    "cpu",
    true,   // remote_accessible
    false   // remote_atomic
);

// Create batch and add requests
FlexBatch batch(engine);
batch.addReadRequest(
    reinterpret_cast<uintptr_t>(local_buffer),
    remote_offset,
    size
);

// Submit to remote segment via direct RDMA
batch.submit("remote_segment_name", false);  // is_target_copy = false

// Poll for completion
while (batch.getTransferStatus(0) == STATUS_WAITING) {
    usleep(100);
}
```

### Example: Batch Memory Registration

```cpp
#include "flex_batch.h"
#include "flex_transfer_engine.h"

// Prepare multiple buffers
std::vector<buffer_entry_t> buffers;
for (int i = 0; i < 100; i++) {
    void *buf = malloc(buffer_size);
    buffers.push_back({.addr = buf, .length = buffer_size});
}

// Batch register (more efficient than individual calls)
engine->registerLocalMemoryBatch(buffers, "cpu:0", false);

// Or use MemoryBatch for mixed locations
MemoryBatch memory_batch;
memory_batch.add(cpu_buffer1, size, "cpu:0");
memory_batch.add(cpu_buffer2, size, "cpu:1");
#ifdef USE_CUDA
memory_batch.add(gpu_buffer, size, "cuda:0");
#endif
engine->registerLocalMemoryBatch(memory_batch, false);
```

### Running the Example

Start the metadata server first (e.g., HTTP metadata server):

```bash
# In terminal 1
mooncake_http_metadata_server
```

Start the CopyTransferEngine (target):

```bash
# In terminal 2
./flex_transfer_example copy http://127.0.0.1:8080/metadata 127.0.0.1
```

Start the DirectTransferEngine (initiator):

```bash
# In terminal 3
./flex_transfer_example direct http://127.0.0.1:8080/metadata 127.0.0.2
```
