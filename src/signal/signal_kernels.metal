// Metal kernels for signal FFT/OA convolution and direct reverse+conj.
// Compiled with -std=metal3.1 -O2 -fno-fast-math -ffp-contract=off.
#include <metal_stdlib>
#include "signal_kernels.h"
using namespace metal;

static inline ulong ms_signal_widen(uint b) {
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

static inline void ms_signal_store(device float* dst, uint index, float value) { dst[index] = value; }
static inline void ms_signal_store(device ulong* dst, uint index, float value) {
    dst[index] = ms_signal_widen(as_type<uint>(value));
}
static inline void ms_signal_store(device float2* dst, uint index, float2 value) { dst[index] = value; }
static inline void ms_signal_store(device ulong2* dst, uint index, float2 value) {
    dst[index] = ulong2(ms_signal_widen(as_type<uint>(value.x)),
                        ms_signal_widen(as_type<uint>(value.y)));
}

static inline float2 ms_signal_cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

static inline bool ms_signal_pad_lookup(constant MSSignalPadParams& p, uint gid, thread uint& out_in_idx) {
    if (p.ndim == 1u) {
        const uint fl = p.fft_len[0];
        uint pos;
        if (p.nsteps[0] == 1u && p.step[0] == fl) {
            if (gid >= p.in_shape[0]) return false;
            pos = gid;
        } else {
            const uint blk = gid / fl;
            const uint loc = gid - blk * fl;
            if (loc >= p.step[0]) return false;
            pos = blk * p.step[0] + loc;
            if (pos >= p.in_shape[0]) return false;
        }
        if (p.reverse_mask & 1u) pos = (p.in_shape[0] - 1u) - pos;
        out_in_idx = pos;
        return true;
    }
    uint rem = gid;
    uint in_idx = 0u;
    for (uint d = 0u; d < p.ndim; ++d) {
        const uint s = p.dst_stride[d];
        const uint c = (d + 1u == p.ndim) ? rem : rem / s;
        rem -= c * s;
        const uint fl = p.fft_len[d];
        uint pos;
        if (p.nsteps[d] == 1u && p.step[d] == fl) {
            if (c >= p.in_shape[d]) return false;
            pos = c;
        } else {
            const uint blk = c / fl;
            const uint loc = c - blk * fl;
            if (loc >= p.step[d]) return false;
            pos = blk * p.step[d] + loc;
            if (pos >= p.in_shape[d]) return false;
        }
        if (p.reverse_mask & (1u << d)) pos = (p.in_shape[d] - 1u) - pos;
        in_idx += pos * p.in_stride[d];
    }
    out_in_idx = in_idx;
    return true;
}

kernel void ms_signal_fft_pad_f32_f32(constant MSSignalPadParams& p [[buffer(0)]],
                                      device const float* src [[buffer(1)]],
                                      device float* dst [[buffer(2)]],
                                      device atomic_uint* nf_flag [[buffer(3)]],
                                      uint gid [[thread_position_in_grid]]) {
    if (gid >= p.dst_elements) return;
    uint in_idx = 0u;
    float v = 0.0f;
    if (ms_signal_pad_lookup(p, gid, in_idx)) {
        v = src[in_idx];
        if (!isfinite(v)) atomic_fetch_or_explicit(nf_flag, 1u, memory_order_relaxed);
    }
    dst[gid] = v;
}

kernel void ms_signal_fft_pad_f32_c64(constant MSSignalPadParams& p [[buffer(0)]],
                                      device const float* src [[buffer(1)]],
                                      device float2* dst [[buffer(2)]],
                                      device atomic_uint* nf_flag [[buffer(3)]],
                                      uint gid [[thread_position_in_grid]]) {
    if (gid >= p.dst_elements) return;
    uint in_idx = 0u;
    float v = 0.0f;
    if (ms_signal_pad_lookup(p, gid, in_idx)) {
        v = src[in_idx];
        if (!isfinite(v)) atomic_fetch_or_explicit(nf_flag, 1u, memory_order_relaxed);
    }
    dst[gid] = float2(v, 0.0f);
}

kernel void ms_signal_fft_pad_c64_c64(constant MSSignalPadParams& p [[buffer(0)]],
                                      device const float2* src [[buffer(1)]],
                                      device float2* dst [[buffer(2)]],
                                      device atomic_uint* nf_flag [[buffer(3)]],
                                      uint gid [[thread_position_in_grid]]) {
    if (gid >= p.dst_elements) return;
    uint in_idx = 0u;
    float2 v = float2(0.0f, 0.0f);
    if (ms_signal_pad_lookup(p, gid, in_idx)) {
        v = src[in_idx];
        if (!isfinite(v.x) || !isfinite(v.y)) atomic_fetch_or_explicit(nf_flag, 1u, memory_order_relaxed);
        if (p.flags & MS_SIGNAL_PAD_CONJ) v.y = -v.y;
    }
    dst[gid] = v;
}

kernel void ms_signal_spectral_mul_inplace(constant MSSignalMulInplaceParams& p [[buffer(0)]],
                                           device float2* a [[buffer(1)]],
                                           device const float2* b [[buffer(2)]],
                                           device atomic_uint* nf_flag [[buffer(3)]],
                                           uint gid [[thread_position_in_grid]]) {
    if (gid >= p.elements) return;
    const float2 prod = ms_signal_cmul(a[gid], b[gid]);
    if (!isfinite(prod.x) || !isfinite(prod.y)) atomic_fetch_or_explicit(nf_flag, 2u, memory_order_relaxed);
    a[gid] = prod;
}

kernel void ms_signal_spectral_mul_broadcast_inplace(constant MSSignalMulBroadcastParams& p [[buffer(0)]],
                                                     device float2* a [[buffer(1)]],
                                                     device const float2* b [[buffer(2)]],
                                                     device atomic_uint* nf_flag [[buffer(3)]],
                                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= p.elements) return;
    uint rem = gid;
    uint ib = 0u;
    for (uint d = 0u; d < p.ndim; ++d) {
        const uint s = p.out_stride[d];
        const uint c = (d + 1u == p.ndim) ? rem : rem / s;
        rem -= c * s;
        ib += c * p.b_stride[d];
    }
    const float2 prod = ms_signal_cmul(a[gid], b[ib]);
    if (!isfinite(prod.x) || !isfinite(prod.y)) atomic_fetch_or_explicit(nf_flag, 2u, memory_order_relaxed);
    a[gid] = prod;
}

kernel void ms_signal_spectral_mul_broadcast(constant MSSignalMulBroadcastParams& p [[buffer(0)]],
                                             device const float2* a [[buffer(1)]],
                                             device const float2* b [[buffer(2)]],
                                             device float2* out [[buffer(3)]],
                                             device atomic_uint* nf_flag [[buffer(4)]],
                                             uint gid [[thread_position_in_grid]]) {
    if (gid >= p.elements) return;
    uint rem = gid;
    uint ia = 0u;
    uint ib = 0u;
    for (uint d = 0u; d < p.ndim; ++d) {
        const uint s = p.out_stride[d];
        const uint c = (d + 1u == p.ndim) ? rem : rem / s;
        rem -= c * s;
        ia += c * p.a_stride[d];
        ib += c * p.b_stride[d];
    }
    const float2 prod = ms_signal_cmul(a[ia], b[ib]);
    if (!isfinite(prod.x) || !isfinite(prod.y)) atomic_fetch_or_explicit(nf_flag, 2u, memory_order_relaxed);
    out[gid] = prod;
}

template <typename SrcT>
static inline SrcT ms_signal_combine_eval(constant MSSignalCombineParams& p,
                                          device const SrcT* src,
                                          uint gid) {
    const uint rev_mask = p.flags >> 8u;
    uint base_idx = 0u;
    int split_deltas[8];
    uint num_active = 0u;

    if (p.ndim == 1u) {
        const uint c = (rev_mask & 1u) ? ((p.out_shape[0] - 1u) - gid) : gid;
        const uint q = p.crop_start[0] + c;
        if (p.nsteps[0] == 1u) {
            base_idx = q * p.conv_stride[0];
        } else {
            const uint b_hi = min(q / p.step[0], p.nsteps[0] - 1u);
            const uint l_hi = q - b_hi * p.step[0];
            base_idx = b_hi * p.block_stride[0] + l_hi * p.conv_stride[0];
            if (b_hi > 0u && l_hi + p.step[0] < p.block_size[0]) {
                const uint alt = (b_hi - 1u) * p.block_stride[0] + (l_hi + p.step[0]) * p.conv_stride[0];
                split_deltas[0] = int(alt) - int(base_idx);
                num_active = 1u;
            }
        }
    } else {
        uint rem = gid;
        for (uint d = 0u; d < p.ndim; ++d) {
            const uint s = p.out_stride[d];
            uint c = (d + 1u == p.ndim) ? rem : rem / s;
            rem -= c * s;
            if (rev_mask & (1u << d)) c = (p.out_shape[d] - 1u) - c;
            const uint q = p.crop_start[d] + c;
            if (p.nsteps[d] == 1u) {
                base_idx += q * p.conv_stride[d];
            } else {
                const uint b_hi = min(q / p.step[d], p.nsteps[d] - 1u);
                const uint l_hi = q - b_hi * p.step[d];
                const uint off0 = b_hi * p.block_stride[d] + l_hi * p.conv_stride[d];
                base_idx += off0;
                if (b_hi > 0u && l_hi + p.step[d] < p.block_size[d]) {
                    const uint off1 = (b_hi - 1u) * p.block_stride[d] + (l_hi + p.step[d]) * p.conv_stride[d];
                    split_deltas[num_active++] = int(off1) - int(off0);
                }
            }
        }
    }

    SrcT acc = src[base_idx];
    if (num_active == 1u) {
        acc += src[uint(int(base_idx) + split_deltas[0])];
    } else if (num_active > 1u) {
        const uint total_combos = 1u << num_active;
        for (uint mask = 1u; mask < total_combos; ++mask) {
            int idx = int(base_idx);
            for (uint j = 0u; j < num_active; ++j) {
                if (mask & (1u << j)) idx += split_deltas[j];
            }
            acc += src[uint(idx)];
        }
    }
    return acc * p.scale;
}

kernel void ms_signal_oa_combine_r_f32(constant MSSignalCombineParams& p [[buffer(0)]],
                                       device const float* src [[buffer(1)]],
                                       device float* dst [[buffer(2)]],
                                       device atomic_uint* nf_flag [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]]) {
    if (gid >= p.out_elements) return;
    const float v = ms_signal_combine_eval<float>(p, src, gid);
    if (!isfinite(v)) atomic_fetch_or_explicit(nf_flag, 2u, memory_order_relaxed);
    ms_signal_store(dst, gid, v);
}

kernel void ms_signal_oa_combine_r_f64(constant MSSignalCombineParams& p [[buffer(0)]],
                                       device const float* src [[buffer(1)]],
                                       device ulong* dst [[buffer(2)]],
                                       device atomic_uint* nf_flag [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]]) {
    if (gid >= p.out_elements) return;
    const float v = ms_signal_combine_eval<float>(p, src, gid);
    if (!isfinite(v)) atomic_fetch_or_explicit(nf_flag, 2u, memory_order_relaxed);
    ms_signal_store(dst, gid, v);
}

kernel void ms_signal_oa_combine_c_f32(constant MSSignalCombineParams& p [[buffer(0)]],
                                       device const float2* src [[buffer(1)]],
                                       device float* dst [[buffer(2)]],
                                       device atomic_uint* nf_flag [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]]) {
    if (gid >= p.out_elements) return;
    const float v = ms_signal_combine_eval<float2>(p, src, gid).x;
    if (!isfinite(v)) atomic_fetch_or_explicit(nf_flag, 2u, memory_order_relaxed);
    ms_signal_store(dst, gid, v);
}

kernel void ms_signal_oa_combine_c_f64(constant MSSignalCombineParams& p [[buffer(0)]],
                                       device const float2* src [[buffer(1)]],
                                       device ulong* dst [[buffer(2)]],
                                       device atomic_uint* nf_flag [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]]) {
    if (gid >= p.out_elements) return;
    const float v = ms_signal_combine_eval<float2>(p, src, gid).x;
    if (!isfinite(v)) atomic_fetch_or_explicit(nf_flag, 2u, memory_order_relaxed);
    ms_signal_store(dst, gid, v);
}

kernel void ms_signal_oa_combine_c_c64(constant MSSignalCombineParams& p [[buffer(0)]],
                                       device const float2* src [[buffer(1)]],
                                       device float2* dst [[buffer(2)]],
                                       device atomic_uint* nf_flag [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]]) {
    if (gid >= p.out_elements) return;
    float2 v = ms_signal_combine_eval<float2>(p, src, gid);
    if (!isfinite(v.x) || !isfinite(v.y)) atomic_fetch_or_explicit(nf_flag, 2u, memory_order_relaxed);
    if (p.flags & MS_SIGNAL_COMBINE_REV_CONJ) v.y = -v.y;
    ms_signal_store(dst, gid, v);
}

kernel void ms_signal_oa_combine_c_c128(constant MSSignalCombineParams& p [[buffer(0)]],
                                        device const float2* src [[buffer(1)]],
                                        device ulong2* dst [[buffer(2)]],
                                        device atomic_uint* nf_flag [[buffer(3)]],
                                        uint gid [[thread_position_in_grid]]) {
    if (gid >= p.out_elements) return;
    float2 v = ms_signal_combine_eval<float2>(p, src, gid);
    if (!isfinite(v.x) || !isfinite(v.y)) atomic_fetch_or_explicit(nf_flag, 2u, memory_order_relaxed);
    if (p.flags & MS_SIGNAL_COMBINE_REV_CONJ) v.y = -v.y;
    ms_signal_store(dst, gid, v);
}

kernel void ms_signal_rev_conj_f32(constant MSSignalRevConjParams& p [[buffer(0)]],
                                   device const float* src [[buffer(1)]],
                                   device float* dst [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= p.elements) return;
    dst[gid] = src[(p.elements - 1u) - gid];
}

kernel void ms_signal_rev_conj_f64(constant MSSignalRevConjParams& p [[buffer(0)]],
                                   device const ulong* src [[buffer(1)]],
                                   device ulong* dst [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= p.elements) return;
    dst[gid] = src[(p.elements - 1u) - gid];
}

kernel void ms_signal_rev_conj_c64(constant MSSignalRevConjParams& p [[buffer(0)]],
                                   device const float2* src [[buffer(1)]],
                                   device float2* dst [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= p.elements) return;
    float2 v = src[(p.elements - 1u) - gid];
    if (p.conj) v.y = -v.y;
    dst[gid] = v;
}

kernel void ms_signal_rev_conj_c128(constant MSSignalRevConjParams& p [[buffer(0)]],
                                    device const ulong2* src [[buffer(1)]],
                                    device ulong2* dst [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.elements) return;
    ulong2 v = src[(p.elements - 1u) - gid];
    if (p.conj) v.y ^= 0x8000000000000000ul;
    dst[gid] = v;
}

kernel void ms_signal_check_finite_f32(constant MSSignalFiniteParams& p [[buffer(0)]],
                                       device const float* src [[buffer(1)]],
                                       device atomic_uint* nf_flag [[buffer(2)]],
                                       uint gid [[thread_position_in_grid]],
                                       uint grid_size [[threads_per_grid]]) {
    for (ulong i = ulong(gid); i < p.count; i += ulong(grid_size)) {
        if (!isfinite(src[i])) {
            atomic_fetch_or_explicit(nf_flag, 2u, memory_order_relaxed);
            return;
        }
    }
}
