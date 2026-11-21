// Shared dense N-D correlation kernels (float32/float64, complex64/complex128).
// Compiled with -std=metal3.1 -O2 -fno-fast-math -ffp-contract=off.
#include <metal_stdlib>
#include "dense_correlate_kernels.h"
using namespace metal;

static inline ulong sm_dense_widen(uint b) {
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

static inline void sm_dense_store(device float* dst, uint index, float value) { dst[index] = value; }
static inline void sm_dense_store(device ulong* dst, uint index, float value) {
    dst[index] = sm_dense_widen(as_type<uint>(value));
}
static inline void sm_dense_store(device float2* dst, uint index, float2 value) { dst[index] = value; }
static inline void sm_dense_store(device ulong2* dst, uint index, float2 value) {
    dst[index] = ulong2(sm_dense_widen(as_type<uint>(value.x)),
                        sm_dense_widen(as_type<uint>(value.y)));
}

// Boundary extension avoiding integer % when pos is in [-L, 2L) (or [-(L-1), 2L-2] for mirror).
// Returns -1 for CONSTANT mode.
static inline int sm_dense_extend(int pos, int length, uint mode) {
    if (pos >= 0 && pos < length) return pos;
    if (mode == 0u) { // nearest
        return pos < 0 ? 0 : length - 1;
    } else if (mode == 1u) { // wrap
        if (pos >= -length && pos < 0) return pos + length;
        if (pos >= length && pos < 2 * length) return pos - length;
        int m = pos % length;
        return m < 0 ? m + length : m;
    } else if (mode == 2u) { // reflect
        if (pos >= -length && pos < 0) return -1 - pos;
        if (pos >= length && pos < 2 * length) return 2 * length - 1 - pos;
        const int period = 2 * length;
        int m = pos % period;
        if (m < 0) m += period;
        return m < length ? m : period - 1 - m;
    } else if (mode == 3u) { // mirror
        if (length == 1) return 0;
        if (pos > -length && pos < 0) return -pos;
        if (pos >= length && pos <= 2 * length - 2) return 2 * length - 2 - pos;
        const int period = 2 * length - 2;
        int m = pos % period;
        if (m < 0) m += period;
        return m < length ? m : period - m;
    } else { // constant
        return -1;
    }
}

template <typename DstT>
static inline void sm_dense_correlate_real(constant SMDenseCorrelateParams& p,
                                           device const float* src,
                                           device DstT* dst,
                                           device const SMDenseTap* taps,
                                           uint gid) {
    if (gid >= p.out_elements) return;
    if (p.num_taps == 0u) {
        sm_dense_store(dst, gid, 0.0f);
        return;
    }
    int p_coord[8];
    uint rem = gid;
    for (uint d = 0u; d < p.ndim; ++d) {
        const uint s = p.out_stride[d];
        const uint c = (d + 1u == p.ndim) ? rem : rem / s;
        p_coord[d] = int(c);
        rem -= c * s;
    }
    bool is_interior = (p.all_interior != 0u);
    if (!is_interior) {
        is_interior = true;
        for (uint d = 0u; d < p.ndim; ++d) {
            if (p_coord[d] < p.interior_lo[d] || p_coord[d] >= p.interior_hi[d]) {
                is_interior = false;
                break;
            }
        }
    }
    float acc = 0.0f;
    if (is_interior) {
        long base_in = 0;
        for (uint d = 0u; d < p.ndim; ++d) {
            base_in += long(p_coord[d]) * long(p.in_stride[d]);
        }
        acc = src[base_in + long(taps[0].linear_offset)] * taps[0].weight_re;
        for (uint t = 1u; t < p.num_taps; ++t) {
            acc = fma(src[base_in + long(taps[t].linear_offset)], taps[t].weight_re, acc);
        }
    } else {
        for (uint t = 0u; t < p.num_taps; ++t) {
            int in_idx = 0;
            bool in_bounds = true;
            for (uint d = 0u; d < p.ndim; ++d) {
                const int q = sm_dense_extend(p_coord[d] + taps[t].coord[d], int(p.in_shape[d]), p.mode[d]);
                if (q < 0) {
                    in_bounds = false;
                    break;
                }
                in_idx += q * int(p.in_stride[d]);
            }
            const float v = in_bounds ? src[in_idx] : p.cval_re;
            const float w = taps[t].weight_re;
            acc = (t == 0u) ? (v * w) : fma(v, w, acc);
        }
    }
    sm_dense_store(dst, gid, acc);
}

template <typename DstT>
static inline void sm_dense_correlate_complex(constant SMDenseCorrelateParams& p,
                                              device const float2* src,
                                              device DstT* dst,
                                              device const SMDenseTap* taps,
                                              uint gid) {
    if (gid >= p.out_elements) return;
    if (p.num_taps == 0u) {
        sm_dense_store(dst, gid, float2(0.0f, 0.0f));
        return;
    }
    int p_coord[8];
    uint rem = gid;
    for (uint d = 0u; d < p.ndim; ++d) {
        const uint s = p.out_stride[d];
        const uint c = (d + 1u == p.ndim) ? rem : rem / s;
        p_coord[d] = int(c);
        rem -= c * s;
    }
    bool is_interior = (p.all_interior != 0u);
    if (!is_interior) {
        is_interior = true;
        for (uint d = 0u; d < p.ndim; ++d) {
            if (p_coord[d] < p.interior_lo[d] || p_coord[d] >= p.interior_hi[d]) {
                is_interior = false;
                break;
            }
        }
    }
    float2 acc = float2(0.0f, 0.0f);
    if (is_interior) {
        long base_in = 0;
        for (uint d = 0u; d < p.ndim; ++d) {
            base_in += long(p_coord[d]) * long(p.in_stride[d]);
        }
        for (uint t = 0u; t < p.num_taps; ++t) {
            const float2 v = src[base_in + long(taps[t].linear_offset)];
            const float wr = taps[t].weight_re;
            const float wi = taps[t].weight_im;
            const float2 prod = float2(fma(-v.y, wi, v.x * wr), fma(v.x, wi, v.y * wr));
            acc = (t == 0u) ? prod : (acc + prod);
        }
    } else {
        const float2 cval = float2(p.cval_re, p.cval_im);
        for (uint t = 0u; t < p.num_taps; ++t) {
            int in_idx = 0;
            bool in_bounds = true;
            for (uint d = 0u; d < p.ndim; ++d) {
                const int q = sm_dense_extend(p_coord[d] + taps[t].coord[d], int(p.in_shape[d]), p.mode[d]);
                if (q < 0) {
                    in_bounds = false;
                    break;
                }
                in_idx += q * int(p.in_stride[d]);
            }
            const float2 v = in_bounds ? src[in_idx] : cval;
            const float wr = taps[t].weight_re;
            const float wi = taps[t].weight_im;
            const float2 prod = float2(fma(-v.y, wi, v.x * wr), fma(v.x, wi, v.y * wr));
            acc = (t == 0u) ? prod : (acc + prod);
        }
    }
    sm_dense_store(dst, gid, acc);
}

kernel void sm_dense_correlate_f32_f32(constant SMDenseCorrelateParams& p [[buffer(0)]],
                                       device const float* src [[buffer(1)]],
                                       device float* dst [[buffer(2)]],
                                       device const SMDenseTap* taps [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]]) {
    sm_dense_correlate_real<float>(p, src, dst, taps, gid);
}

kernel void sm_dense_correlate_f32_f64(constant SMDenseCorrelateParams& p [[buffer(0)]],
                                       device const float* src [[buffer(1)]],
                                       device ulong* dst [[buffer(2)]],
                                       device const SMDenseTap* taps [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]]) {
    sm_dense_correlate_real<ulong>(p, src, dst, taps, gid);
}

template <typename DstT>
static inline void sm_dense_correlate2d_real(constant SMDenseCorrelateParams& p,
                                             device const float* src,
                                             device DstT* dst,
                                             device const SMDenseTap* taps,
                                             uint2 pos) {
    if (pos.x >= p.out_shape[1] || pos.y >= p.out_shape[0]) return;
    const uint gid = pos.y * p.out_shape[1] + pos.x;
    if (p.num_taps == 0u) {
        sm_dense_store(dst, gid, 0.0f);
        return;
    }
    const int py = int(pos.y);
    const int px = int(pos.x);
    const bool is_interior = (p.all_interior != 0u) ||
                             (py >= p.interior_lo[0] && py < p.interior_hi[0] &&
                              px >= p.interior_lo[1] && px < p.interior_hi[1]);
    float acc = 0.0f;
    if (is_interior) {
        const long base_in = long(py) * long(p.in_stride[0]) + long(px) * long(p.in_stride[1]);
        acc = src[base_in + long(taps[0].linear_offset)] * taps[0].weight_re;
        for (uint t = 1u; t < p.num_taps; ++t) {
            acc = fma(src[base_in + long(taps[t].linear_offset)], taps[t].weight_re, acc);
        }
    } else {
        const int h = int(p.in_shape[0]);
        const int w = int(p.in_shape[1]);
        const int sy = int(p.in_stride[0]);
        const int sx = int(p.in_stride[1]);
        for (uint t = 0u; t < p.num_taps; ++t) {
            const int qy = sm_dense_extend(py + taps[t].coord[0], h, p.mode[0]);
            const int qx = sm_dense_extend(px + taps[t].coord[1], w, p.mode[1]);
            const float v = (qy >= 0 && qx >= 0) ? src[qy * sy + qx * sx] : p.cval_re;
            const float wt = taps[t].weight_re;
            acc = (t == 0u) ? (v * wt) : fma(v, wt, acc);
        }
    }
    sm_dense_store(dst, gid, acc);
}

kernel void sm_dense_correlate2d_f32_f32(constant SMDenseCorrelateParams& p [[buffer(0)]],
                                         device const float* src [[buffer(1)]],
                                         device float* dst [[buffer(2)]],
                                         device const SMDenseTap* taps [[buffer(3)]],
                                         uint2 pos [[thread_position_in_grid]]) {
    sm_dense_correlate2d_real<float>(p, src, dst, taps, pos);
}

kernel void sm_dense_correlate2d_f32_f64(constant SMDenseCorrelateParams& p [[buffer(0)]],
                                         device const float* src [[buffer(1)]],
                                         device ulong* dst [[buffer(2)]],
                                         device const SMDenseTap* taps [[buffer(3)]],
                                         uint2 pos [[thread_position_in_grid]]) {
    sm_dense_correlate2d_real<ulong>(p, src, dst, taps, pos);
}

kernel void sm_dense_correlate_c64_c64(constant SMDenseCorrelateParams& p [[buffer(0)]],
                                       device const float2* src [[buffer(1)]],
                                       device float2* dst [[buffer(2)]],
                                       device const SMDenseTap* taps [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]]) {
    sm_dense_correlate_complex<float2>(p, src, dst, taps, gid);
}

kernel void sm_dense_correlate_c64_c128(constant SMDenseCorrelateParams& p [[buffer(0)]],
                                        device const float2* src [[buffer(1)]],
                                        device ulong2* dst [[buffer(2)]],
                                        device const SMDenseTap* taps [[buffer(3)]],
                                        uint gid [[thread_position_in_grid]]) {
    sm_dense_correlate_complex<ulong2>(p, src, dst, taps, gid);
}
