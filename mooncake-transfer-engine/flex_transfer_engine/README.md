# Flex Transfer Engine

When using RDMA transfer, the user must first register the memory, which is costly. It is acceptable as one-time cost, but in some cases, register/unregister can happen very frequently, causing significant overhead. To address this problem, we provide `FlexTransferEngine`, which unifies direct RDMA transfer and copy-based transfer in a single flexible API.

## Implementation Status

This implementation includes:
- `FlexTransferEngine`: Unified transfer engine supporting both direct RDMA and copy-based transfers
- TCP-based protocol for transfer coordination between engines
- Pre-registered RDMA buffer pools (one buffer pair per memory location) for copy-based transfers
- Double buffering support for overlapping copy and RDMA operations
- Support for both CPU and GPU memory (when compiled with CUDA support)
- Single contiguous allocation for buffer pairs to reduce registration overhead

## FlexTransferEngine

`FlexTransferEngine` is a flexible wrapper on top of the C API `transfer_engine_c.h`. It provides the same API as TransferEngine with additional support for copy-based transfers.

**Constructor:**
```cpp
FlexTransferEngine(bool enable_copy = false)
```
- `enable_copy = false`: Engine operates in direct RDMA mode only
- `enable_copy = true`: Engine also starts a TCP listener for copy-based transfers

**API:**
The engine exposes standard TransferEngine APIs: `openSegment`, `closeSegment`, `registerLocalMemory`, `unregisterLocalMemory`, `registerLocalMemoryBatch`, `unregisterLocalMemoryBatch`, `syncSegmentCache`, `allocateBatchID`, `freeBatchID`, `submitTransfer`, and `getTransferStatus`.

## Copy-Based Transfer Mode

When `enable_copy = true`, the engine:

1. Starts a background TCP listener thread on the specified port
2. Tracks registered memory regions internally
3. Manages a buffer pool with one buffer pair per memory location
4. Each buffer pair is allocated as a single contiguous buffer (size 2×largest_region) and split into two halves
5. Buffer pairs are registered with RDMA only once and reused for all transfers

**Memory Registration Behavior:**
- `registerLocalMemory` keeps a record of the memory region and ensures a buffer pair exists for the location
- If no buffer pair exists or the existing one is too small, allocates/reallocates the buffer pair
- `unregisterLocalMemory` only removes the region from internal tracking; buffer pairs remain for reuse
- Note: Locations starting with "cuda" indicate GPU memory; otherwise, CPU memory

## Transfer Modes

When calling `submitTransfer`, users can specify the transfer mode:

```cpp
int submitTransfer(batch_id_t batch_id,
                   const std::vector<transfer_request_t> &entries,
                   const std::string &copy_server_name = "",
                   uint16_t copy_server_port = 12346);
```

- **Direct RDMA mode** (`copy_server_name` is empty): Uses standard RDMA transfer
- **Copy-based mode** (`copy_server_name` provided): Uses TCP-coordinated copy transfer

**Copy-Based Transfer Protocol:**

Currently only supports read requests (reading from a remote engine with `enable_copy = true`).

1. Initiator sends via TCP:
   - Target segment name (the initiator's local segment name where data will be written)
   - Progress address (for tracking completion)
   - Number of requests
   - For each request: source address, target offset, length

2. Remote engine (with `enable_copy = true`):
   - Opens the target segment (caches segment_id)
   - For each request:
     - Verifies source address is registered
     - Copies data from source to pre-registered buffer
     - Performs RDMA write from buffer to target segment
   - Updates progress counter via RDMA write

3. Double buffering ensures overlap between memory copy and RDMA transfer operations

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

### API Example

```cpp
#include "flex_transfer_engine.h"

// On the target node (server with frequently changing memory regions)
// Enable copy-based transfer mode
FlexTransferEngine copy_engine(true);  // enable_copy = true
copy_engine.init(metadata_server, local_server_name, "", 12345, 12346, 1);

// Register memory that changes frequently
void *data = malloc(size);
copy_engine.registerLocalMemory(data, size, "cpu", 1);

// On the initiator node (client that reads data)
// Direct RDMA mode (no copy listener needed)
FlexTransferEngine direct_engine(false);  // enable_copy = false
direct_engine.init(metadata_server, local_server_name, "", 12345, 12346, 1);

// Allocate local buffer
void *local_buffer = malloc(size);
direct_engine.registerLocalMemory(local_buffer, size, "cpu", 1);

// Submit read requests from remote engine
batch_id_t batch_id = direct_engine.allocateBatchID(1);
std::vector<transfer_request_t> requests;
transfer_request_t req;
req.opcode = OPCODE_READ;
req.source = remote_addr;  // Address on remote engine
req.target_id = LOCAL_SEGMENT;
req.target_offset = (uint64_t)local_buffer;
req.length = size;
requests.push_back(req);

// Option 1: Use copy-based transfer (copy_server_name provided)
direct_engine.submitTransfer(batch_id, requests, "target_server", 12346);

// Option 2: Use direct RDMA transfer (copy_server_name empty)
// direct_engine.submitTransfer(batch_id, requests);
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

## Implementation Notes

- The CopyTransferEngine listens on TCP port 12346 by default
- Progress tracking is done via RDMA writes to a registered progress counter
- Buffer pool management is automatic - each memory location has exactly one buffer pair that is resized as needed
- Double buffering allows overlapping of memory copy and RDMA transfer operations
- Error handling includes writing -1 to the progress counter on failures
- Single worker thread handles both TCP listening and request processing

## Limitations and TODOs

- Full RDMA write implementation from CopyTransferEngine to DirectTransferEngine target addresses needs metadata service integration
- Segment discovery and connection management can be enhanced
- Performance tuning for buffer sizes
- Additional testing with GPU memory transfers
- Support for write operations (currently only read is implemented)
