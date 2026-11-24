#pragma once
// Separable N-D filters: the validated plan both engines execute.
//
// A pass sees the array as outer x length x inner: `length` positions along
// the filtered axis, `inner` elements between consecutive positions (the
// product of the later extents) and `outer` such blocks. Tap j of output
// position p reads extension position p - before + j.
#include "sm_ndimage.h"
#include "core/array.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sm::detail::ndimage {
// How SciPy's NI_Correlate1D classifies odd-length weights (to DBL_EPSILON):
// it then sums centre tap first, pairs from the outermost inwards.
enum class Symmetry : int32_t { none = 0, even = 1, odd = -1 };

struct Pass {
    uint32_t kind = 0;   // sm_ndimage_pass_kind
    uint32_t mode = 0;   // sm_ndimage_mode
    double cval = 0.0;
    uint64_t size = 0;   // taps
    const double* weights = nullptr;  // caller's weights (correlate passes)
    Symmetry symmetry = Symmetry::none;
    uint64_t length = 0, inner = 0, outer = 0;
    uint64_t before = 0;  // size/2 + origin: extension positions read before the output position
    uint64_t after = 0;   // size - size/2 - 1 - origin
};

struct Plan {
    array::Shape shape;
    uint32_t dtype = 0;      // SM_FLOAT32 or SM_FLOAT64 (input)
    uint32_t out_dtype = 0;  // 0 (same as dtype), SM_FLOAT32 or SM_FLOAT64
    std::vector<Pass> passes;
};

struct CompositePlan {
    array::Shape shape;
    uint32_t in_dtype = 0;
    uint32_t out_dtype = 0;
    uint32_t combine_op = 0; // SM_NDIMAGE_COMBINE_SUM or SM_NDIMAGE_COMBINE_SQRT_SUM_SQ
    std::vector<std::vector<Pass>> chains;
};

// The line element an extension position outside [0, length) repeats, or -1
// for the constant. Matches SciPy's NI_ExtendLine for every window length,
// including windows longer than the line.
inline int64_t extend(int64_t position, int64_t length, uint32_t mode) noexcept {
    switch (mode) {
    case SM_NDIMAGE_NEAREST:
        return position < 0 ? 0 : length - 1;
    case SM_NDIMAGE_WRAP: {
        const int64_t m = position % length;
        return m < 0 ? m + length : m;
    }
    case SM_NDIMAGE_REFLECT: {
        const int64_t period = 2 * length;
        int64_t m = position % period;
        if (m < 0) m += period;
        return m < length ? m : period - 1 - m;
    }
    case SM_NDIMAGE_MIRROR: {
        if (length == 1) return 0;
        const int64_t period = 2 * length - 2;
        int64_t m = position % period;
        if (m < 0) m += period;
        return m < length ? m : period - m;
    }
    default:
        return -1;
    }
}

// These passes only convert the input through the SciPy line-buffer dtype.
inline bool minmax_identity(const Pass& pass) noexcept {
    return (pass.kind == SM_NDIMAGE_MINIMUM || pass.kind == SM_NDIMAGE_MAXIMUM) &&
           (pass.size == 1 || (pass.length == 1 && pass.mode != SM_NDIMAGE_CONSTANT));
}

// True when a 1-D min/max pass is eligible for the 2-pass van Herk GPU path
// (window > 64, enough blocks for GPU occupancy, and extended buffer fits 32-bit indices).
inline bool gpu_minmax_use_vanherk(const Pass& pass) noexcept {
    if (pass.kind != SM_NDIMAGE_MINIMUM && pass.kind != SM_NDIMAGE_MAXIMUM) return false;
    if (pass.size <= 64) return false;
    const uint64_t ext_len = pass.length + pass.size - 1;
    if (ext_len > (uint64_t{1} << 29)) return false;
    uint64_t ext_elems = 0;
    if (sm_checked_mul(pass.outer, ext_len, &ext_elems) != SM_OK ||
        sm_checked_mul(ext_elems, pass.inner, &ext_elems) != SM_OK ||
        ext_elems > INT32_MAX) {
        return false;
    }
    const uint64_t blocks_per_line = (ext_len + pass.size - 1) / pass.size;
    uint64_t total_blocks = 0;
    if (sm_checked_mul(pass.outer * pass.inner, blocks_per_line, &total_blocks) != SM_OK) return false;
    return total_blocks >= 512;
}

// CPU engine: every pass in double precision on the performance cores.
BufferPtr run_cpu(sm_context*, const Plan&, const void* input);
BufferPtr run_cpu_composite(sm_context*, const CompositePlan&, const void* input);

// GPU engine: single precision, one command buffer. gpu_supports says whether
// the plan fits the kernels' 32-bit indexing and the single-buffer time budget.
// If the input range or filter gain rules out float32 execution: when
// `allow_cpu_fallback` is true (SM_EXECUTION_AUTO), returns null and writes the
// reason to `overflow_reason` before any GPU operation begins; when false
// (SM_EXECUTION_GPU), throws SM_ENOTSUP.
bool gpu_supports(const Plan&) noexcept;
bool gpu_supports_composite(const CompositePlan&) noexcept;
BufferPtr run_gpu(sm_context*, const Plan&, const sm_array& input, bool allow_cpu_fallback,
                  std::string& overflow_reason, bool dense_safe = false);
BufferPtr run_gpu_composite(sm_context*, const CompositePlan&, const sm_array& input, bool allow_cpu_fallback,
                            std::string& overflow_reason);
} // namespace sm::detail::ndimage
