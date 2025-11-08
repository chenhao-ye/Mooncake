#pragma once

#include <atomic>
#include <cstdint>

/**
 * For remote CopyServer to update the progress counter to indicate how many
 * requests have been transferred.
 */
struct RDMACopyCtrlBlock {
    volatile std::atomic_int64_t progress_counter = 0;
    uint64_t padding[7];
};

static_assert(sizeof(RDMACopyCtrlBlock) == 64, "RDMACopyCtrlBlock must be 64-byte");

enum class CopyMode : uint32_t { RDMA, TCP };
