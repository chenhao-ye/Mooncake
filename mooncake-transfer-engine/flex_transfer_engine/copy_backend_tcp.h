#pragma once

#include <sys/types.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "copy_common.h"
#include "region.h"
#include "transfer_engine_c.h"

class FlexTransferEngine;

class TcpCopyBackend {
   public:
    struct Task {
        void *source_addr;
        size_t length;
        // TODO: maybe other fields for tracking execution status
    };

   public:
    TcpCopyBackend(FlexTransferEngine &engine, RegionMgr &region_mgr)
        : engine_(engine), region_mgr_(region_mgr) {}

    int processRequest(int client_fd, std::vector<Task> &tasks);

   private:
    FlexTransferEngine &engine_;
    RegionMgr &region_mgr_;
};
