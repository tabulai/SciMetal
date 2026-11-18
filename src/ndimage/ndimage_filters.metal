// scipy.ndimage separable filters: one 1-D correlation pass per dispatch, in
// float32 arithmetic. See ndimage_kernels.h for the grid and the parameters.
#include <metal_stdlib>
#include "ndimage_kernels.h"
using namespace metal;

// float32 bits -> float64 bits as a C cast does it: exact for every finite
// value including subnormals, NaNs quieted with their payload kept (the code
// probes/e3_narrow validated bit for bit against numpy.astype).
static inline ulong ms_ndimage_widen(uint b) {
    const uint sign = b & 0x80000000u;
    const uint e = (b >> 23) & 0xFFu;
    uint m = b & 0x007FFFFFu;
    uint hi;
    if (e == 0xFFu) {
        hi = sign | 0x7FF00000u | (m != 0u ? 0x00080000u : 0u) | (m >> 3);
    } else if (e == 0u) {
        if (m == 0u) return ulong(sign) << 32;
        const uint s = clz(m) - 8u;
        m = (m << s) & 0x007FFFFFu;
        hi = sign | ((897u - s) << 20) | (m >> 3);
    } else {
        hi = sign | ((e + 896u) << 20) | (m >> 3);
    }
    return (ulong(hi) << 32) | ulong(m << 29);
}

static inline void ms_ndimage_store(device float* dst, uint index, float value) { dst[index] = value; }
static inline void ms_ndimage_store(device ulong* dst, uint index, float value) {
    dst[index] = ms_ndimage_widen(as_type<uint>(value));
}

// The value at extension position `position` of the line whose first element
// is src[line] (ms_ndimage_mode; the host's `extend` in filters.hpp).
static inline float ms_ndimage_tap(device const float* src, int line, int stride, int position, int length,
                                   uint mode, float cval) {
    if (position < 0 || position >= length) {
        if (mode == 0u) {                       // nearest
            position = position < 0 ? 0 : length - 1;
        } else if (mode == 1u) {                // wrap
            position = position % length;
            position = position < 0 ? position + length : position;
        } else if (mode == 2u) {                // reflect: period 2n, pairs-sample symmetric
            const int period = 2 * length;
            position = position % period;
            position = position < 0 ? position + period : position;
            position = position < length ? position : period - 1 - position;
        } else if (mode == 3u) {                // mirror: period 2n - 2, whole-sample symmetric
            if (length == 1) {
                position = 0;
            } else {
                const int period = 2 * length - 2;
                position = position % period;
                position = position < 0 ? position + period : position;
                position = position < length ? position : period - position;
            }
        } else {                                // constant
            return cval;
        }
    }
    return src[line + position * stride];
}

// SymmetryKind is a compile-time constant: one kernel per summation order
// keeps each inner loop free of the others' code, which measured 10% faster
// on 4096 x 4096 than one kernel branching on a parameter. The boundary mode
// stays a parameter: only threads near the ends of a line look at it.
template <typename Destination, uint SymmetryKind>
static inline void ms_ndimage_correlate1d(constant MSNdimageCorrelateParams& p, device const float* src,
                                          Destination dst, constant float* all_weights, uint3 pos) {
    if (pos.x >= p.extent[0] || pos.y >= p.extent[1] || pos.z >= p.extent[2]) return;
    const uint index = (pos.z * p.extent[1] + pos.y) * p.extent[0] + pos.x;
    const int at = int(p.axis_in_y != 0u ? pos.y : pos.x);
    const int length = int(p.length);
    const int stride = int(p.stride);
    const int taps = int(p.taps);
    const int first = at - int(p.before);  // extension position of tap 0
    constant float* w = all_weights + p.weight_offset;
    const int pairs = taps / 2;
    float acc;
    if (first >= 0 && first + taps <= length) {
        // Every tap is inside the line: plain strided loads.
        device const float* window = src + (int(index) - int(p.before) * stride);
        if (SymmetryKind == MS_NDIMAGE_SYMMETRY_NONE) {
            acc = window[(taps - 1) * stride] * w[taps - 1];
            for (int j = 0; j < taps - 1; ++j) acc += window[j * stride] * w[j];
        } else {
            device const float* centre = window + pairs * stride;
            acc = centre[0] * w[pairs];
            for (int d = pairs; d >= 1; --d) {
                const float below = centre[-d * stride], above = centre[d * stride];
                acc += (SymmetryKind == MS_NDIMAGE_SYMMETRY_ODD ? below - above : below + above) * w[pairs - d];
            }
        }
    } else {
        const int line = int(index) - at * stride;
        const uint mode = p.mode;
        const float cval = p.cval;
        if (SymmetryKind == MS_NDIMAGE_SYMMETRY_NONE) {
            acc = ms_ndimage_tap(src, line, stride, first + taps - 1, length, mode, cval) * w[taps - 1];
            for (int j = 0; j < taps - 1; ++j)
                acc += ms_ndimage_tap(src, line, stride, first + j, length, mode, cval) * w[j];
        } else {
            const int centre = first + pairs;
            acc = ms_ndimage_tap(src, line, stride, centre, length, mode, cval) * w[pairs];
            for (int d = pairs; d >= 1; --d) {
                const float below = ms_ndimage_tap(src, line, stride, centre - d, length, mode, cval);
                const float above = ms_ndimage_tap(src, line, stride, centre + d, length, mode, cval);
                acc += (SymmetryKind == MS_NDIMAGE_SYMMETRY_ODD ? below - above : below + above) * w[pairs - d];
            }
        }
    }
    ms_ndimage_store(dst, index, acc);
}

#define MS_NDIMAGE_CORRELATE1D_KERNEL(name, element, symmetry)                                      \
    kernel void name(constant MSNdimageCorrelateParams& p [[buffer(0)]],                            \
                     device const float* src [[buffer(1)]],                                         \
                     device element* dst [[buffer(2)]],                                             \
                     constant float* weights [[buffer(3)]],                                         \
                     uint3 pos [[thread_position_in_grid]]) {                                       \
        ms_ndimage_correlate1d<device element*, symmetry>(p, src, dst, weights, pos);              \
    }

MS_NDIMAGE_CORRELATE1D_KERNEL(ms_ndimage_correlate1d_any_f32, float, MS_NDIMAGE_SYMMETRY_NONE)
MS_NDIMAGE_CORRELATE1D_KERNEL(ms_ndimage_correlate1d_even_f32, float, MS_NDIMAGE_SYMMETRY_EVEN)
MS_NDIMAGE_CORRELATE1D_KERNEL(ms_ndimage_correlate1d_odd_f32, float, MS_NDIMAGE_SYMMETRY_ODD)
MS_NDIMAGE_CORRELATE1D_KERNEL(ms_ndimage_correlate1d_any_f64, ulong, MS_NDIMAGE_SYMMETRY_NONE)
MS_NDIMAGE_CORRELATE1D_KERNEL(ms_ndimage_correlate1d_even_f64, ulong, MS_NDIMAGE_SYMMETRY_EVEN)
MS_NDIMAGE_CORRELATE1D_KERNEL(ms_ndimage_correlate1d_odd_f64, ulong, MS_NDIMAGE_SYMMETRY_ODD)
