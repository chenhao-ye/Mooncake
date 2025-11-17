#pragma once

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct LocId {
    uint32_t idx;         // index into location_strings_
    int32_t cuda_device;  // -1 for CPU

    bool isCuda() const { return cuda_device >= 0; }
};

static_assert(sizeof(LocId) == sizeof(uint64_t), "LocId size must be 8 bytes");

struct Region {
    void *addr;
    size_t length;
    LocId loc_id;
};

// Not thread-safe; caller must hold regions_mutex_
class RegionMgr {
   public:
    std::mutex regions_mutex_;

   public:
    RegionMgr() : physical_to_logical_cuda_device_(parseCudaVisibleDevices()) {}

    LocId getLocId(const std::string &location) {
        // extract CUDA device from location string (e.g., "cuda:0" -> 0)
        int cuda_device = -1;  // -1 for CPU

        if (location.find("cuda:") == 0) {
            int physical_device = std::stoi(location.substr(5));

            // Convert physical device ID to logical device ID if mapping exists
            if (!physical_to_logical_cuda_device_.empty()) {
                if (physical_device < 0 ||
                    physical_device >=
                        static_cast<int>(
                            physical_to_logical_cuda_device_.size()) ||
                    physical_to_logical_cuda_device_[physical_device] == -1) {
                    throw std::runtime_error(
                        "CUDA device " + std::to_string(physical_device) +
                        " not found in CUDA_VISIBLE_DEVICES");
                }
                cuda_device = physical_to_logical_cuda_device_[physical_device];
            } else {
                cuda_device = physical_device;
            }
        }

        // search to find existing location
        for (uint32_t i = 0; i < location_strings_.size(); ++i) {
            if (location_strings_[i] == location)
                return LocId{.idx = i, .cuda_device = cuda_device};
        }
        // not found, add new location
        uint32_t idx = static_cast<uint32_t>(location_strings_.size());
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
    std::vector<int> parseCudaVisibleDevices() {
        const char *cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
        if (cuda_visible_devices == nullptr ||
            cuda_visible_devices[0] == '\0') {
            return {};  // No mapping needed, use physical device IDs directly
        }

        std::vector<int> mapping;
        std::istringstream iss(cuda_visible_devices);
        std::string token;
        int logical_idx = 0;

        while (std::getline(iss, token, ',')) {
            int physical_id = std::stoi(token);

            if (physical_id >= static_cast<int>(mapping.size()))
                mapping.resize(physical_id + 1, -1);

            mapping[physical_id] = logical_idx;
            ++logical_idx;
        }

        return mapping;
    }

    // Copiable memory regions (addr -> region info)
    // Tracks regions that can be read via copy transfer.
    std::unordered_map<void *, Region> regions_;

    // Location string storage (LocId -> location string)
    // Append-only; will never remove entries
    std::vector<std::string> location_strings_;

    // CUDA device mapping (physical device ID -> logical device ID for
    // cudaSetDevice) -1 indicates unavailable/invalid device. Empty if
    // CUDA_VISIBLE_DEVICES not set.
    const std::vector<int> physical_to_logical_cuda_device_;
};
