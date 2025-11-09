#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using LocIdx = int32_t;  // <0 for invalid location

class RegionMgr {
    // Not thread-safe; caller must hold regions_mutex_

   public:
    struct Region {
        void *addr;
        size_t length;
        LocIdx loc_idx;
    };

    // Require regions_mutex_ to be held before calling
    LocIdx getLocIdx(const std::string &location) {
        // Linear search to find existing location
        for (size_t i = 0; i < location_strings_.size(); ++i) {
            if (location_strings_[i] == location) return static_cast<LocIdx>(i);
        }
        // Not found, add new location
        LocIdx new_idx = static_cast<LocIdx>(location_strings_.size());
        location_strings_.emplace_back(location);
        return new_idx;
    }

    void addRegion(void *addr, size_t length, LocIdx loc_idx) {
        regions_[addr] = {addr, length, loc_idx};
    }

    int removeRegion(void *addr) {
        size_t num_erased = regions_.erase(addr);
        return num_erased > 0 ? 0 : -1;
    }

    // Require regions_mutex_ to be held before calling
    Region *getRegion(void *addr, size_t length) {
        // fast path: the addr is the base of a registered region
        auto it = regions_.find(addr);
        if (it != regions_.end() && length <= it->second.length)
            return &it->second;
        // slow path: scan to find addr within a registered region
        for (auto &[base_addr, region] : regions_) {
            if (addr < static_cast<char *>(base_addr)) continue;
            if (static_cast<char *>(addr) + length >
                static_cast<char *>(base_addr) + region.length)
                continue;
            return &region;
        }
        return nullptr;
    }

   private:
    // Copiable memory regions (addr -> region info)
    // Tracks regions that can be read via copy transfer.
    // When enable_copy_ is true, these are NOT actually RDMA-registered,
    // only tracked for copy-based transfers.
    // When enable_copy_ is false, these ARE RDMA-registered.
    std::unordered_map<void *, Region> regions_;

    // Location string storage (LocIdx -> location string)
    // Append-only; will never remove entries
    std::vector<std::string> location_strings_;
};
