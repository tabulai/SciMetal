#pragma once
// Separable N-D filters: the validated plan both engines execute.
//
// A pass sees the array as outer x length x inner: `length` positions along
// the filtered axis, `inner` elements between consecutive positions (the
// product of the later extents) and `outer` such blocks. Tap j of output
// position p reads extension position p - before + j.
#include "ms_ndimage.h"
#include "core/array.hpp"

#include <cstdint>
#include <vector>

namespace ms::detail::ndimage {
// How SciPy's NI_Correlate1D classifies odd-length weights (to DBL_EPSILON):
// it then sums centre tap first, pairs from the outermost inwards.
enum class Symmetry : int32_t { none = 0, even = 1, odd = -1 };

struct Pass {
    uint32_t kind = 0;   // ms_ndimage_pass_kind
    uint32_t mode = 0;   // ms_ndimage_mode
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
    uint32_t dtype = 0;  // MS_FLOAT32 or MS_FLOAT64: input, intermediates and result
    std::vector<Pass> passes;
};

// The line element an extension position outside [0, length) repeats, or -1
// for the constant. Matches SciPy's NI_ExtendLine for every window length,
// including windows longer than the line.
inline int64_t extend(int64_t position, int64_t length, uint32_t mode) noexcept {
    switch (mode) {
    case MS_NDIMAGE_NEAREST:
        return position < 0 ? 0 : length - 1;
    case MS_NDIMAGE_WRAP: {
        const int64_t m = position % length;
        return m < 0 ? m + length : m;
    }
    case MS_NDIMAGE_REFLECT: {
        const int64_t period = 2 * length;
        int64_t m = position % period;
        if (m < 0) m += period;
        return m < length ? m : period - 1 - m;
    }
    case MS_NDIMAGE_MIRROR: {
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

// CPU engine: every pass in double precision on the performance cores.
BufferPtr run_cpu(ms_context*, const Plan&, const void* input);

// GPU engine: single precision, one command buffer. gpu_supports says whether
// the plan fits the kernels' 32-bit indexing and the single-buffer time budget.
bool gpu_supports(const Plan&) noexcept;
BufferPtr run_gpu(ms_context*, const Plan&, const ms_array& input);
} // namespace ms::detail::ndimage
