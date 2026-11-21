#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace sm::detail {
inline constexpr uint64_t cached_metal_sample_min_capacity = uint64_t{256} << 20;
inline constexpr size_t cached_metal_sample_count = 8;

// These are Mach page-query dispositions, checked against the SDK where the
// query is made. REF, DIRTY and COPIED may accompany exactly one state bit.
// Neither missing PRESENT nor an unfamiliar disposition proves paging.
inline bool cached_metal_pages_need_reset(const std::array<int32_t, cached_metal_sample_count>& statuses,
                                         const std::array<uint32_t, cached_metal_sample_count>& dispositions) {
    uint32_t paged_out = 0;
    bool recognized = true;
    for (size_t index = 0; index < cached_metal_sample_count; ++index) {
        const auto flags = dispositions[index];
        const auto state = flags & 0x11;
        if (statuses[index] != 0 || (flags & ~uint32_t{0x3d}) || state == 0 || state == 0x11)
            recognized = false;
        else if (state == 0x10) ++paged_out;
    }
    return recognized && paged_out >= 2;
}

inline std::optional<uint64_t> cached_metal_sample_offset(uintptr_t host, uint64_t capacity,
                                                         uint64_t page, size_t index) {
    if (capacity < cached_metal_sample_min_capacity || page < 64 || (page & (page - 1)) ||
        index >= cached_metal_sample_count || host % 64 || capacity > UINTPTR_MAX - host)
        return std::nullopt;
    // floor(capacity*(index+1)/9), without overflowing the multiplication.
    const uint64_t nominal = (capacity / 9) * (index + 1) + ((capacity % 9) * (index + 1)) / 9;
    const uint64_t sampled = (host + nominal) & ~(page - 1);
    const uint64_t first = host & ~(page - 1), last = (host + capacity - 1) & ~(page - 1);
    if (sampled <= first || sampled >= last || sampled < host || page > host + capacity - sampled)
        return std::nullopt;
    return sampled - host;
}
} // namespace sm::detail
