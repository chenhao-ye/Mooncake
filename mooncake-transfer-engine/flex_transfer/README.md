# Flex Transfer Engine

When using RDMA transfer, the user must first register the memory, which is costly. It is acceptable as one-time cost, but in some cases, register/unregister can happen very frequently, causing significant overhead. To address these problems, we augment the transfer engine with different backends: Direct Transfer Engine and Copy Transfer Engine.

## Implementation Status

This implementation includes:
- `DirectTransferEngine`: Thin wrapper around TransferEngine that supports reading from CopyTransferEngine
- `CopyTransferEngine`: Manages pre-registered RDMA buffers (one buffer pair per memory location) and handles copy-based transfers
- TCP-based protocol for transfer coordination between Direct and Copy engines
- Double buffering support for overlapping copy and RDMA operations
- Support for both CPU and GPU memory (when compiled with CUDA support)

## Direct Transfer Engine

`DirectTransferEngine` is a thin wrapper on top of existing `class TransferEngine` (as in `mooncake-transfer-engine/include/transfer_engine.h`). It exposes `openSegment`, `closeSegment`, `registerLocalMemory`, `unregisterLocalMemory`, `registerLocalMemoryBatch`, `unregisterLocalMemoryBatch`, `syncSegmentCache`, `allocateBatchID`, `freeBatchID`, `submitTransfer`, and `getTransferStatus`. All data transfer between two DirectTransferEngine are the same as the raw TranserEngine. The only difference is how it interacts with CopyTransferEngine (see below).

## Copy Transfer Engine

`CopyTransferEngine` is another a wrapped on top existing transfer engine and expose these APIs similar to `class TransferEngine`: `openSegment`, `closeSegment`, `registerLocalMemory`, `unregisterLocalMemory`, `registerLocalMemoryBatch`, `unregisterLocalMemoryBatch`, `syncSegmentCache` (note it does not provide APIs to submit read/write requests).

Most of the CopyTransferEngine APIs are just a simple pass-through to the underlying TransferEngine except the register-related APIs.

- `registerLocalMemoryBatch` will not directly register the memory to RDMA NICs. It keeps a record of the provided memory regions. Then checks for each device ("location") if it has a pair of memory buffers that is larger than the largest memory region among the batch. If not (or if the existing buffer pair is too small), it allocates/reallocates a buffer pair and registers it with RDMA NICs. Each location has exactly one buffer pair. Note the location starting with "cuda" is a GPU memory; otherwise, it is a CPU memory.
- `unregisterLocalMemoryBatch` only remove these memory regions from its internal data structures. Do not actually unregister the buffers (left for future reuse).
- `registerLocalMemory` and `unregisterLocalMemory` are similar.

CopyTransferEngine will have a background thread listening on TCP to initiate data transfer (see below).

## DirectTransferEngine Reads from CopyTransferEnginie

Currently, DirectTransferEngine only supports to read data from CopyTransferEngine; other supports may be addded later. When the user call `submitTransfer` to DirectTransferEngine, it can set a flag to specify the target is a CopyTransferEngine (such transfer batch should only contain read requests; an error will be thrown if any non-read request detected during executioln).

DirectTransferEngine first sends the batch info to the CopyTransferEngine via TCP. The background listener thread will receive a "progress" address, a counter of how many requests, and a sequence of addr-size pairs as requests; it then starts to transfer the data:
1. It first confirms the given addresses are registered; return an error if not.
2. It then copies the data from the given address to a RDMA-registered buffer, and then issues RDMA write (through the underlying `TransferEngine::submitTransfer`). Note the address could be in CUDA memory.
3. For performance, the copy and RDMA transfer should be overlapped by utilizing the buffer pairs: one buffer is do copying while the other is doing RDMA transfer. Once a transfer is done, that buffer can be reused for the next request.

During the transfer, once the request `i` is done, add a RDMA-write request (to the progress address with value `i`) to the next batch. If there is an error (e.g., a given address is not registered), write -1 to the progress address.

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
#include "copy_transfer_engine.h"
#include "direct_transfer_engine.h"

// On the target node (server with frequently changing memory regions)
CopyTransferEngine copy_engine(true);
copy_engine.init(metadata_server, local_server_name, "", 12345, 12346);

// Register memory that changes frequently
void *data = malloc(size);
copy_engine.registerLocalMemory(data, size, "cpu", true, true);

// On the initiator node (client that reads data)
DirectTransferEngine direct_engine(true);
direct_engine.init(metadata_server, local_server_name, "", 12345);

// Allocate local buffer
void *local_buffer = malloc(size);
direct_engine.registerLocalMemory(local_buffer, size, "cpu", true, true);

// Submit read requests from CopyTransferEngine
BatchID batch_id = direct_engine.allocateBatchID(1);
std::vector<TransferRequest> requests;
TransferRequest req;
req.opcode = TransferRequest::READ;
req.source = remote_addr;  // Address on CopyTransferEngine
req.target_id = 0;          // Local segment
req.target_offset = (uint64_t)local_buffer;
req.length = size;
requests.push_back(req);

// Submit with flag indicating target is CopyTransferEngine
direct_engine.submitTransfer(batch_id, requests, /*target_is_copy_engine=*/true);
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
