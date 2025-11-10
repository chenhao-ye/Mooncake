/**
 * Some common definitions for both copy server and copy client.
 */

#pragma once

#include <atomic>
#include <cstdint>

enum class CopyMode : uint32_t { RDMA, TCP };

/* Assumption: the server machine uses the same endianness as the client. */

struct RdmaHeader {
    uint64_t progress_addr;
    uint64_t num_reqs;
};

struct RdmaReq {
    uint64_t source_addr;
    uint64_t target_addr;
    uint64_t length;
};

struct TcpHeader {
    uint64_t num_reqs;
};

struct TcpReq {
    uint64_t target_addr;
    uint64_t length;
};
