#pragma once
#include "sm/detail/backend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace sm::detail::parallel {

constexpr unsigned max_lanes = 32;

// SciMetal: lanes are sized to the performance cores. Efficiency cores slow
// a statically partitioned loop down to their own pace.
inline unsigned hardware_lanes() noexcept {
    static const unsigned cached = []() noexcept {
        unsigned hw = 0;
#ifdef __APPLE__
        int performance = 0;
        size_t size = sizeof(performance);
        if (sysctlbyname("hw.perflevel0.physicalcpu", &performance, &size, nullptr, 0) == 0 && performance > 0)
            hw = static_cast<unsigned>(performance);
#endif
        if (!hw) hw = std::thread::hardware_concurrency();
        return std::clamp<unsigned>(hw ? hw : 4u, 1u, max_lanes);
    }();
    return cached;
}

inline unsigned choose_lanes(uint64_t items, uint64_t min_items_per_lane = 16384) noexcept {
    if (items == 0 || min_items_per_lane == 0 || items <= min_items_per_lane) return 1u;
    const uint64_t needed = (items + min_items_per_lane - 1) / min_items_per_lane;
    return static_cast<unsigned>(std::min<uint64_t>({needed, uint64_t(hardware_lanes()), uint64_t(max_lanes)}));
}

// Deterministic earliest-key error tracker for parallel loops. Workers never
// let C++ exceptions cross libdispatch C frames; the lowest failing key wins.
class EarliestError {
public:
    bool has_error() const noexcept { return key_.load(std::memory_order_relaxed) != UINT64_MAX; }
    uint64_t min_key() const noexcept { return key_.load(std::memory_order_relaxed); }

    void record(uint64_t key, sm_status status, const char* message) noexcept {
        if (key == UINT64_MAX) key = UINT64_MAX - 1;
        if (key >= key_.load(std::memory_order_relaxed)) return;
        std::lock_guard lock(mutex_);
        if (key < key_.load(std::memory_order_relaxed)) {
            status_ = status;
            const char* src = message ? message : "";
            size_t len = 0;
            while (len + 1 < sizeof(message_) && src[len]) ++len;
            std::memcpy(message_, src, len);
            message_[len] = '\0';
            key_.store(key, std::memory_order_relaxed);
        }
    }

    void rethrow() const {
        if (key_.load(std::memory_order_relaxed) != UINT64_MAX) throw Error(status_, message_);
    }

private:
    std::atomic<uint64_t> key_{UINT64_MAX};
    mutable std::mutex mutex_;
    sm_status status_ = SM_EBACKEND;
    char message_[256]{};
};

// Executes fn(lane_index) across [0, lanes) using macOS GCD (or std::jthread
// on non-Apple platforms). Catches any exception thrown inside fn so unwinding
// never crosses C libdispatch frames, and rethrows the lowest-lane failure.
template<class F>
inline void run_lanes(unsigned lanes, F&& fn) {
    if (lanes == 0) return;
    if (lanes == 1) {
        fn(0u);
        return;
    }
    lanes = std::min(lanes, max_lanes);
    EarliestError unexpected;
    auto safe_step = [&](unsigned lane) noexcept {
        try {
            fn(lane);
        } catch (const Error& err) {
            unexpected.record(lane, err.status, err.what());
        } catch (const std::bad_alloc&) {
            unexpected.record(lane, SM_EOOM, "native allocation failed");
        } catch (const std::exception& err) {
            unexpected.record(lane, SM_EBACKEND, err.what());
        } catch (...) {
            unexpected.record(lane, SM_EBACKEND, "unknown native exception");
        }
    };
#ifdef __APPLE__
    using Step = decltype(safe_step);
    dispatch_apply_f(lanes, static_cast<dispatch_queue_t>(nullptr), &safe_step,
                     [](void* ctx, size_t i) noexcept { (*static_cast<Step*>(ctx))(static_cast<unsigned>(i)); });
#else
    std::array<std::jthread, max_lanes - 1> threads;
    for (unsigned i = 1; i < lanes; ++i) threads[i - 1] = std::jthread([&safe_step, i]() noexcept { safe_step(i); });
    safe_step(0u);
    for (unsigned i = 1; i < lanes; ++i) {
        if (threads[i - 1].joinable()) threads[i - 1].join();
    }
#endif
    unexpected.rethrow();
}

// Partitions [0, count) into up to max_lanes contiguous ranges [begin, end)
// whose interior boundaries are multiples of `alignment` (use alignment = 64
// when writing Arrow LSB-first bitmaps so lanes touch disjoint 8-byte words).
template<class F>
inline unsigned parallel_ranges(uint64_t count, uint64_t min_items_per_lane, uint64_t alignment, F&& fn) {
    if (count == 0) return 0u;
    const uint64_t align = std::max<uint64_t>(1, alignment);
    const uint64_t aligned_units = (count + align - 1) / align;
    const uint64_t min_units = std::max<uint64_t>(1, (min_items_per_lane + align - 1) / align);
    const unsigned lanes = aligned_units <= min_units
        ? 1u
        : static_cast<unsigned>(std::min<uint64_t>({(aligned_units + min_units - 1) / min_units,
                                                    uint64_t(hardware_lanes()), uint64_t(max_lanes)}));
    if (lanes == 1u) {
        fn(0u, uint64_t{0}, count);
        return 1u;
    }
    const uint64_t units_per_lane = (aligned_units + lanes - 1) / lanes;
    run_lanes(lanes, [&](unsigned lane) {
        const uint64_t u_begin = uint64_t(lane) * units_per_lane;
        if (u_begin >= aligned_units) return;
        const uint64_t begin = u_begin * align;
        const uint64_t end = std::min<uint64_t>(count, (u_begin + units_per_lane) * align);
        if (begin < end) fn(lane, begin, end);
    });
    return lanes;
}

template<class F>
inline unsigned parallel_for(uint64_t count, uint64_t min_items_per_lane, F&& fn) {
    return parallel_ranges(count, min_items_per_lane, 1,
                           [&](unsigned lane, uint64_t begin, uint64_t end) { fn(lane, begin, end); });
}

} // namespace sm::detail::parallel

namespace sm::detail::simd {

inline uint64_t count_set_bytes(const uint8_t* bytes, uint64_t num_bytes) noexcept {
    uint64_t total = 0;
    uint64_t i = 0;
#if defined(__ARM_NEON)
    while (num_bytes - i >= 64) {
        // Each iteration adds at most 32 per uint16_t lane; 512 * 32 = 16,384 <= 65,535.
        const uint64_t batch_iters = std::min<uint64_t>((num_bytes - i) / 64, 512);
        uint16x8_t acc0 = vdupq_n_u16(0);
        uint16x8_t acc1 = vdupq_n_u16(0);
        for (uint64_t k = 0; k < batch_iters; ++k, i += 64) {
            const uint8x16_t v0 = vld1q_u8(bytes + i);
            const uint8x16_t v1 = vld1q_u8(bytes + i + 16);
            const uint8x16_t v2 = vld1q_u8(bytes + i + 32);
            const uint8x16_t v3 = vld1q_u8(bytes + i + 48);
            acc0 = vpadalq_u8(acc0, vaddq_u8(vcntq_u8(v0), vcntq_u8(v1)));
            acc1 = vpadalq_u8(acc1, vaddq_u8(vcntq_u8(v2), vcntq_u8(v3)));
        }
        total += vaddlvq_u16(acc0) + vaddlvq_u16(acc1);
    }
    for (; num_bytes - i >= 16; i += 16) {
        total += vaddlvq_u8(vcntq_u8(vld1q_u8(bytes + i)));
    }
#endif
    for (; num_bytes - i >= 8; i += 8) {
        uint64_t word;
        std::memcpy(&word, bytes + i, sizeof(word));
        total += static_cast<uint64_t>(std::popcount(word));
    }
    for (; i < num_bytes; ++i) {
        total += static_cast<uint64_t>(std::popcount(bytes[i]));
    }
    return total;
}

#if defined(__ARM_NEON)
// Packs 16 0xFF/0x00 mask lanes into a 16-bit Arrow LSB-first bitmask.
inline uint16_t pack_mask16_u8(uint8x16_t mask) noexcept {
    static constexpr uint8_t kWeights[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t weighted = vandq_u8(mask, vld1q_u8(kWeights));
    const uint16_t lo = vaddv_u8(vget_low_u8(weighted));
    const uint16_t hi = vaddv_u8(vget_high_u8(weighted));
    return static_cast<uint16_t>(lo | (hi << 8));
}

// Packs 64 0xFF/0x00 mask lanes (m0..m3) into a 64-bit Arrow LSB-first word.
inline uint64_t pack_mask64_u8(uint8x16_t m0, uint8x16_t m1, uint8x16_t m2, uint8x16_t m3) noexcept {
    static constexpr uint8_t kWeights[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t w = vld1q_u8(kWeights);
    const uint8x16_t p01 = vpaddq_u8(vandq_u8(m0, w), vandq_u8(m1, w));
    const uint8x16_t p23 = vpaddq_u8(vandq_u8(m2, w), vandq_u8(m3, w));
    const uint8x16_t p0123 = vpaddq_u8(p01, p23);
    const uint8x8_t bytes = vpadd_u8(vget_low_u8(p0123), vget_high_u8(p0123));
    return vget_lane_u64(vreinterpret_u64_u8(bytes), 0);
}

inline uint8x16_t narrow_mask_u16(uint16x8_t a, uint16x8_t b) noexcept {
    return vuzp1q_u8(vreinterpretq_u8_u16(a), vreinterpretq_u8_u16(b));
}

inline uint8x16_t narrow_mask_u32(uint32x4_t a, uint32x4_t b, uint32x4_t c, uint32x4_t d) noexcept {
    const uint16x8_t ab = vuzp1q_u16(vreinterpretq_u16_u32(a), vreinterpretq_u16_u32(b));
    const uint16x8_t cd = vuzp1q_u16(vreinterpretq_u16_u32(c), vreinterpretq_u16_u32(d));
    return narrow_mask_u16(ab, cd);
}

inline uint8x16_t narrow_mask_u64(uint64x2_t m0, uint64x2_t m1, uint64x2_t m2, uint64x2_t m3,
                                  uint64x2_t m4, uint64x2_t m5, uint64x2_t m6, uint64x2_t m7) noexcept {
    const uint32x4_t q0 = vuzp1q_u32(vreinterpretq_u32_u64(m0), vreinterpretq_u32_u64(m1));
    const uint32x4_t q1 = vuzp1q_u32(vreinterpretq_u32_u64(m2), vreinterpretq_u32_u64(m3));
    const uint32x4_t q2 = vuzp1q_u32(vreinterpretq_u32_u64(m4), vreinterpretq_u32_u64(m5));
    const uint32x4_t q3 = vuzp1q_u32(vreinterpretq_u32_u64(m6), vreinterpretq_u32_u64(m7));
    return narrow_mask_u32(q0, q1, q2, q3);
}
#endif

} // namespace sm::detail::simd
