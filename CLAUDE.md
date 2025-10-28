# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mooncake is a KVCache-centric disaggregated architecture for LLM serving, used in production by Kimi (Moonshot AI). It separates prefill and decoding clusters and leverages underutilized CPU, DRAM, and SSD resources for disaggregated KVCache storage.

**Key Components:**
- **Transfer Engine**: Core data transfer framework supporting TCP, RDMA (InfiniBand/RoCEv2/eRDMA/GPUDirect), and NVMe-of protocols
- **Mooncake Store**: Distributed KVCache storage engine for LLM inference (supports vLLM and LMCache integration)
- **P2P Store**: Peer-to-peer object sharing for scenarios like checkpoint transfer
- **Integration Layer**: Official integrations with vLLM and SGLang for prefill-decode disaggregation

## Build Commands

### Prerequisites
```bash
# Install all dependencies (requires stable internet)
bash dependencies.sh
```

### Standard Build
```bash
mkdir build
cd build
cmake ..
make -j
sudo make install  # Installs python package and mooncake_master executable
```

### Build with Options
```bash
cmake .. \
  -DUSE_CUDA=ON \           # Enable GPU Direct RDMA and NVMe-of
  -DUSE_CXL=ON \            # Enable CXL support
  -DWITH_STORE=ON \         # Build Mooncake Store (default: ON)
  -DWITH_P2P_STORE=OFF \    # Build P2P Store (requires Go 1.23+, default: OFF)
  -DUSE_REDIS=OFF \         # Enable Redis metadata service
  -DUSE_HTTP=OFF \          # Enable HTTP metadata service
  -DBUILD_SHARED_LIBS=OFF \ # Build as shared library
  -DBUILD_UNIT_TESTS=ON \   # Build unit tests (default: ON)
  -DBUILD_EXAMPLES=ON       # Build examples (default: ON)

make -j
```

### Python Package Build
```bash
# Build wheel package
bash scripts/build_wheel.sh

# Or install directly via pip
pip install mooncake-transfer-engine
```

### Docker Build
```bash
docker pull alogfans/mooncake
# Run with RDMA device access
docker run --net=host --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/rdma_cm --ulimit memlock=-1 -t -i mooncake:v0.9.0 /bin/bash
```

## Testing

### Run All Tests
```bash
./scripts/run_tests.sh
```

This script:
1. Runs Transfer Engine tests (target + initiator with HTTP metadata server)
2. Runs Mooncake Store master tests
3. Validates CLI entry points

### Run Individual Test Components
```bash
# Transfer Engine tests
cd mooncake-wheel/tests
MC_METADATA_SERVER=http://127.0.0.1:8080/metadata python transfer_engine_target.py &
MC_METADATA_SERVER=http://127.0.0.1:8080/metadata python transfer_engine_initiator_test.py

# Mooncake Store tests
mooncake_master &
MC_METADATA_SERVER=http://127.0.0.1:8080/metadata python test_distributed_object_store.py
```

### Unit Tests (C++)
After building with `-DBUILD_UNIT_TESTS=ON`:
```bash
cd build
ctest  # Run all C++ unit tests
```

## Architecture

### Component Structure
```
mooncake-common/           # Shared utilities and etcd integration
mooncake-transfer-engine/  # Core transfer engine (C++)
  ├── include/             # Public API headers (transfer_engine.h, common.h)
  ├── src/                 # Implementation (transport layers)
  └── example/             # transfer_engine_bench tool

mooncake-store/            # Distributed KVCache storage
  ├── include/             # Store API (client.h, master_service.h, types.h)
  ├── src/                 # Master service, RPC, allocators
  └── tests/               # Store integration tests

mooncake-p2p-store/        # P2P object sharing (Go-based)
  └── src/                 # P2P store implementation

mooncake-integration/      # Integration with LLM frameworks
  ├── vllm/                # vLLM adaptor
  ├── store/               # Store integration helpers
  └── transfer_engine/     # Transfer engine Python bindings

mooncake-wheel/            # Python package
  ├── mooncake/            # Python library
  └── setup.py             # Package build configuration
```

### Transfer Engine Architecture
Transfer Engine provides zero-copy data transfer via two core abstractions:

**Segment**: Represents remotely accessible address space
- RAM Segment: Non-persistent DRAM/VRAM storage
- NVMeof Segment: Persistent NVMe storage via NVMe-of protocol
- Each process has exactly one RAM segment named by `local_hostname`
- Segments are divided into Buffers with separate permissions and NIC affinity

**BatchTransfer**: Asynchronous scatter/gather operations
- Supports Read/Write between non-contiguous data spaces across segments
- Handles data transfer between DRAM ↔ DRAM, DRAM ↔ VRAM, VRAM ↔ VRAM, and NVMe-of
- Uses topology-aware path selection for optimal RDMA NIC utilization
- Implements automatic failover and retry on network errors

**Transport Backends**:
- `TcpTransport`: TCP-based transfer (DRAM ↔ remote DRAM)
- `RdmaTransport`: RDMA-based transfer with multi-NIC pooling and GPUDirect support
- `NVMeoFTransport`: NVMe-of with cuFile (GPUDirect Storage)

### Mooncake Store Architecture
Distributed KVCache storage with master-worker topology:

**Master Node** (`master_service.h`):
- Centralized metadata management for object-to-buffer mappings
- Drives managed pool buffer nodes via Transfer Engine APIs
- Coordinates replication strategies and eviction policies

**Managed Pool Buffer Nodes**:
- Provide DRAM/VRAM space for storing objects
- Execute data transfer requests from master using Transfer Engine
- Support multi-replica storage for hotspot mitigation

**Features**:
- Object-level operations: Get/Put/List/Del/Replicate
- Multi-replica support for access pressure distribution
- Striping and parallel I/O for large objects (multi-NIC bandwidth aggregation)
- Atomic write operations (Get always reads consistent version)
- Dynamic cache resource addition/removal

### Metadata Services
Mooncake supports multiple metadata backends (enabled via CMake options):
- **etcd** (default with `-DUSE_ETCD`): Distributed key-value store
- **Redis** (with `-DUSE_REDIS`): Alternative key-value backend
- **HTTP** (with `-DUSE_HTTP`): Simple HTTP-based metadata service

The metadata service manages:
- Transfer Engine connection state
- Segment and buffer registry
- Store object-to-buffer mappings

## Code Style

### C++
Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Header files use `.h` extension
- Use `#pragma once` for header guards (seen in codebase)
- Well-documented code is essential for maintainability

### Python
Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Python 3.8+ required (3.10+ recommended for type hints)

### General
- Include sufficient unit tests and integration tests
- Update documentation in `doc/` for user-facing changes
- Use appropriate PR title prefix: `[Bugfix]`, `[CI/Build]`, `[Doc]`, `[Integration]`, `[P2PStore]`, `[TransferEngine]`, `[Misc]`

## Integration Guides

### vLLM Integration
Mooncake provides disaggregated prefilling support for vLLM:
- See `doc/en/vllm-integration-v0.2.md` for setup guide
- Requires `mooncake_vllm_adaptor` Python package installed
- Uses Transfer Engine for inter-node KVCache transfer (replaces nccl/gloo)
- Supports both prefill-decode disaggregation and xPyD architecture

### SGLang Integration
- See `doc/en/sglang-integration-v1.md` for integration details
- Official support for disaggregated prefilling and KV cache transfer

### LMCache Integration
- See `doc/en/lmcache-integration.md`
- Mooncake Store serves as remote connector for LMCache

## Important Files

**Core APIs**:
- `mooncake-transfer-engine/include/transfer_engine.h` - Main Transfer Engine interface
- `mooncake-transfer-engine/include/common.h` - Common data structures and constants
- `mooncake-store/include/client.h` - Store client API
- `mooncake-store/include/master_service.h` - Store master service interface
- `mooncake-store/include/types.h` - Store type definitions

**Configuration**:
- `CMakeLists.txt` - Top-level build configuration
- `mooncake-common/common.cmake` - Shared CMake settings
- `mooncake-wheel/pyproject.toml` - Python package metadata

**Documentation**:
- `doc/en/architecture.md` - System architecture overview
- `doc/en/transfer-engine.md` - Transfer Engine detailed guide
- `doc/en/build.md` - Build instructions and requirements

## Common Development Tasks

### Verifying Installation
```bash
# Test mooncake_master is installed correctly
which mooncake_master

# Verify Python package imports
python -c "import mooncake_vllm_adaptor"

# Run full test suite
./scripts/test_installation.sh
```

### Running Transfer Engine Benchmark
```bash
cd build/mooncake-transfer-engine/example

# Start metadata service (etcd or HTTP)
# For HTTP: mooncake_http_metadata_server

# On target node
./transfer_engine_bench --device_name=<nic_name> --metadata_server=<ip:port> --mode=target --local_server_name=<local_ip>

# On initiator node
./transfer_engine_bench --device_name=<nic_name> --metadata_server=<ip:port> --mode=initiator --local_server_name=<local_ip> --target_name=<target_ip>
```

### Working with Mooncake Store
After building and installing, `mooncake_master` executable provides the store service:
```bash
# Start master service
mooncake_master

# Start REST API server
mc_store_rest_server
```

## RDMA Requirements

Mooncake is designed for high-speed RDMA networks. While TCP-only transfer is supported, RDMA is **strongly recommended** for production use.

**Prerequisites**:
- RDMA Driver & SDK (e.g., Mellanox OFED)
- For GPU support: CUDA 12.1+ with NVIDIA GPUDirect Storage
- For GPU-RDMA: nvidia-peermem kernel module (see build.md Section 2.2)

**Environment Setup**:
```bash
# For CUDA builds
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

## Performance Notes

- With 4×200 Gbps RoCE: Transfer Engine achieves ~87 GB/s (2.4x faster than TCP)
- With 8×400 Gbps RoCE: Transfer Engine achieves ~190 GB/s (4.6x faster than TCP)
- vLLM integration shows up to 25% lower Mean TTFT compared to TCP-based transports
- Topology-aware path selection and multi-card aggregation are key to performance
