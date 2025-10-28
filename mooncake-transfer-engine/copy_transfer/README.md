# Copy-Based Transfer

When using RDMA transfer, the user must first register the memory, which is costly. It is acceptable as one-time cost, but in some cases, register/unregister can happen very frequently, causing signifiant overhead. To address these problems, we augment the transfer engine with different backends: Direct Transfer Engine and Copy Transfer Engine.

## Direct Transfer Engine

`DirectTransferEngine` is a thin wrapper on top of existing `class TransferEngine` (as in `mooncake-transfer-engine/include/transfer_engine.h`). It exposes `openSegment`, `closeSegment`, `registerLocalMemory`, `unregisterLocalMemory`, `registerLocalMemoryBatch`, `unregisterLocalMemoryBatch`, `syncSegmentCache`, `allocateBatchID`, `freeBatchID`, `submitTransfer`, and `getTransferStatus`. All data transfer between two DirectTransferEngine are the same as the raw TranserEngine. The only difference is how it interacts with CopyTransferEngine (see below).

## Copy Transfer Engine

`CopyTransferEngine` is another a wrapped on top existing transfer engine and expose these APIs similar to `class TransferEngine`: `openSegment`, `closeSegment`, `registerLocalMemory`, `unregisterLocalMemory`, `registerLocalMemoryBatch`, `unregisterLocalMemoryBatch`, `syncSegmentCache` (note it does not provide APIs to submit read/write requests).

Most of the CopyTransferEngine APIs are just a simple pass-through to the underlying TransferEngine except the register-related APIs.

- `registerLocalMemoryBatch` will not directly register the memory to RDMA NICs. It keeps a record of the provided memory regions. Then checkes for each device ("location") if it has a pair of memory buffers that is larger than the largest memory region among the batch. If not, allocates two such buffers and registers them with RDMA NICs (if there is a buffer pair but too small, unregister them first); add the new buffer pairs to the cache. Note the location starting with "cuda" is a GPU memory; otherwise, it is a CPU memory.
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
