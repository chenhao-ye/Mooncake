#pragma once

/**
 * Control block for copy-based transfers.
 * Contains progress counter that is RDMA-accessible.
 */
#include <cstdint>

struct CopyCtrlBlock {
    volatile int64_t progress_counter;
};
