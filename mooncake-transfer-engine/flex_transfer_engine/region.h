#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct LocId {
    uint32_t idx;         // index into location_strings_
    int32_t cuda_device;  // -1 for CPU
};

static_assert(sizeof(LocId) == sizeof(uint64_t), "LocId size must be 8 bytes");

class RegionMgr {
    // Not thread-safe; caller must hold regions_mutex_

   public:
    struct Region {
        void *addr;
        size_t length;
        LocId loc_id;
    };

    // Require regions_mutex_ to be held before calling
    LocId getLocId(const std::string &location) {
        // extract CUDA device from location string (e.g., "cuda:0" -> 0)
        int32_t cuda_device = -1;  // -1 for CPU
        if (location.find("cuda:") == 0)
            cuda_device = std::stoi(location.substr(5));

        // search to find existing location
        for (uint32_t i = 0; i < location_strings_.size(); ++i) {
            if (location_strings_[i] == location)
                return LocId{.idx = i, .cuda_device = cuda_device};
        }
        // not found, add new location
        uint32_t idx = static_cast<int32_t>(location_strings_.size());
        location_strings_.emplace_back(location);
        return LocId{.idx = idx, .cuda_device = cuda_device};
    }

    void addRegion(void *addr, size_t length, LocId loc_id) {
        regions_[addr] = {addr, length, loc_id};
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

    // Location string storage (LocId -> location string)
    // Append-only; will never remove entries
    std::vector<std::string> location_strings_;
};
