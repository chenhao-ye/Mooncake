/**
 * Some common definitions for both copy server and copy client.
 */

#pragma once

#include <atomic>
#include <cstdint>

enum class CopyMode : uint32_t { RDMA, TCP };

/**
 * Used by both CopyClient and CopyServer. Must be register with remote_atomic.
 */
struct RDMACopyCtrlBlock {
    volatile std::atomic_int64_t progress_counter = 0;
    uint64_t padding[7];
};

static_assert(sizeof(RDMACopyCtrlBlock) == 64,
              "RDMACopyCtrlBlock must be 64-byte");
