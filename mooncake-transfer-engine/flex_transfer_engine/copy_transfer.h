#pragma once

#include <atomic>
#include <cstdint>

/**
 * For remote CopyServer to update the progress counter to indicate how many
 * requests have been transferred.
 */
struct CopyCtrlBlock {
    volatile std::atomic_int64_t progress_counter = 0;
    uint64_t padding[7];
};

static_assert(sizeof(CopyCtrlBlock) == 64, "CopyCtrlBlock must be 64-byte");

enum class CopyMode : uint32_t { RDMA, TCP };
