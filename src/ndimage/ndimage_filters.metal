// scipy.ndimage separable filters: one 1-D correlation pass per dispatch, in
// float32 arithmetic. See ndimage_kernels.h for the grid and the parameters.
#include <metal_stdlib>
#include "ndimage_kernels.h"
using namespace metal;

// float32 bits -> float64 bits as a C cast does it: exact for every finite
// value including subnormals, NaNs quieted with their payload kept (the code
// probes/e3_narrow validated bit for bit against numpy.astype).
static inline ulong sm_ndimage_widen(uint b) {
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

static inline void sm_ndimage_store(device float* dst, uint index, float value) { dst[index] = value; }
static inline void sm_ndimage_store(device ulong* dst, uint index, float value) {
    dst[index] = sm_ndimage_widen(as_type<uint>(value));
}

// SymmetryKind is a compile-time constant: one kernel per summation order
// keeps each inner loop free of the others' code, which measured 10% faster
// on 4096 x 4096 than one kernel branching on a parameter. The boundary mode
// stays a parameter: only threads near the ends of a line (or overhanging
// windows) take the boundary path, where mode is dispatched outside the tap
// loop with 3-segment direct loops (NEAREST/CONSTANT) and modulo-free cursor
// stepping (WRAP/REFLECT/MIRROR).
template <typename Destination, uint SymmetryKind>
static inline void sm_ndimage_correlate1d(constant SMNdimageCorrelateParams& p, device const float* src,
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
        if (SymmetryKind == SM_NDIMAGE_SYMMETRY_NONE) {
            acc = window[(taps - 1) * stride] * w[taps - 1];
            for (int j = 0; j < taps - 1; ++j) acc += window[j * stride] * w[j];
        } else {
            device const float* centre = window + pairs * stride;
            acc = centre[0] * w[pairs];
            for (int d = pairs; d >= 1; --d) {
                const float below = centre[-d * stride], above = centre[d * stride];
                acc += (SymmetryKind == SM_NDIMAGE_SYMMETRY_ODD ? below - above : below + above) * w[pairs - d];
            }
        }
    } else {
        device const float* line_src = src + (int(index) - at * stride);
        const uint mode = p.mode;
        const float cval = p.cval;
        if (SymmetryKind == SM_NDIMAGE_SYMMETRY_NONE) {
            const int last_pos = first + taps - 1;
            if (mode == 0u) {  // nearest
                const float last_v = line_src[clamp(last_pos, 0, length - 1) * stride];
                acc = last_v * w[taps - 1];
                const int j_lo = clamp(-first, 0, taps - 1);
                const int j_hi = clamp(length - first, j_lo, taps - 1);
                if (j_lo > 0) {
                    const float left_v = line_src[0];
                    for (int j = 0; j < j_lo; ++j) acc += left_v * w[j];
                }
                for (int j = j_lo; j < j_hi; ++j) acc += line_src[(first + j) * stride] * w[j];
                if (j_hi < taps - 1) {
                    const float right_v = line_src[(length - 1) * stride];
                    for (int j = j_hi; j < taps - 1; ++j) acc += right_v * w[j];
                }
            } else if (mode == 1u) {  // wrap
                int ql = last_pos % length;
                if (ql < 0) ql += length;
                acc = line_src[ql * stride] * w[taps - 1];
                if (taps > 1) {
                    int q = first % length;
                    if (q < 0) q += length;
                    for (int j = 0; j < taps - 1; ++j) {
                        acc += line_src[q * stride] * w[j];
                        if (++q == length) q = 0;
                    }
                }
            } else if (mode == 2u) {  // reflect
                const int period = 2 * length;
                int ml = last_pos % period;
                if (ml < 0) ml += period;
                const int ql = ml < length ? ml : period - 1 - ml;
                acc = line_src[ql * stride] * w[taps - 1];
                if (taps > 1) {
                    int m = first % period;
                    if (m < 0) m += period;
                    for (int j = 0; j < taps - 1; ++j) {
                        const int q = m < length ? m : period - 1 - m;
                        acc += line_src[q * stride] * w[j];
                        if (++m == period) m = 0;
                    }
                }
            } else if (mode == 3u) {  // mirror
                if (length == 1) {
                    const float v0 = line_src[0];
                    acc = v0 * w[taps - 1];
                    for (int j = 0; j < taps - 1; ++j) acc += v0 * w[j];
                } else {
                    const int period = 2 * length - 2;
                    int ml = last_pos % period;
                    if (ml < 0) ml += period;
                    const int ql = ml < length ? ml : period - ml;
                    acc = line_src[ql * stride] * w[taps - 1];
                    if (taps > 1) {
                        int m = first % period;
                        if (m < 0) m += period;
                        for (int j = 0; j < taps - 1; ++j) {
                            const int q = m < length ? m : period - m;
                            acc += line_src[q * stride] * w[j];
                            if (++m == period) m = 0;
                        }
                    }
                }
            } else {  // constant
                const float last_v = (last_pos >= 0 && last_pos < length) ? line_src[last_pos * stride] : cval;
                acc = last_v * w[taps - 1];
                const int j_lo = clamp(-first, 0, taps - 1);
                const int j_hi = clamp(length - first, j_lo, taps - 1);
                for (int j = 0; j < j_lo; ++j) acc += cval * w[j];
                for (int j = j_lo; j < j_hi; ++j) acc += line_src[(first + j) * stride] * w[j];
                for (int j = j_hi; j < taps - 1; ++j) acc += cval * w[j];
            }
        } else {
            const int centre = first + pairs;
            if (mode == 0u) {  // nearest
                acc = line_src[clamp(centre, 0, length - 1) * stride] * w[pairs];
                const int d_both_out = max(centre, length - 1 - centre);
                const int d_in_stop = max(0, min(pairs, min(centre, length - 1 - centre)));
                int d = pairs;
                if (d > d_both_out) {
                    const float left_v = line_src[0];
                    const float right_v = line_src[(length - 1) * stride];
                    const float diff = (SymmetryKind == SM_NDIMAGE_SYMMETRY_ODD ? left_v - right_v : left_v + right_v);
                    for (; d > d_both_out; --d) acc += diff * w[pairs - d];
                }
                for (; d > d_in_stop; --d) {
                    const float below = line_src[clamp(centre - d, 0, length - 1) * stride];
                    const float above = line_src[clamp(centre + d, 0, length - 1) * stride];
                    acc += (SymmetryKind == SM_NDIMAGE_SYMMETRY_ODD ? below - above : below + above) * w[pairs - d];
                }
                if (d >= 1) {
                    device const float* c_src = line_src + centre * stride;
                    for (; d >= 1; --d) {
                        const float below = c_src[-d * stride], above = c_src[d * stride];
                        acc += (SymmetryKind == SM_NDIMAGE_SYMMETRY_ODD ? below - above : below + above) * w[pairs - d];
                    }
                }
            } else if (mode == 1u) {  // wrap
                int qc = centre % length;
                if (qc < 0) qc += length;
                acc = line_src[qc * stride] * w[pairs];
                if (pairs >= 1) {
                    int qb = (centre - pairs) % length;
                    if (qb < 0) qb += length;
                    int qa = (centre + pairs) % length;
                    if (qa < 0) qa += length;
                    for (int d = pairs; d >= 1; --d) {
                        const float below = line_src[qb * stride];
                        const float above = line_src[qa * stride];
                        acc += (SymmetryKind == SM_NDIMAGE_SYMMETRY_ODD ? below - above : below + above) * w[pairs - d];
                        if (++qb == length) qb = 0;
                        if (--qa < 0) qa = length - 1;
                    }
                }
            } else if (mode == 2u) {  // reflect
                const int period = 2 * length;
                int mc = centre % period;
                if (mc < 0) mc += period;
                const int qc = mc < length ? mc : period - 1 - mc;
                acc = line_src[qc * stride] * w[pairs];
                if (pairs >= 1) {
                    int mb = (centre - pairs) % period;
                    if (mb < 0) mb += period;
                    int ma = (centre + pairs) % period;
                    if (ma < 0) ma += period;
                    for (int d = pairs; d >= 1; --d) {
                        const int qb = mb < length ? mb : period - 1 - mb;
                        const int qa = ma < length ? ma : period - 1 - ma;
                        const float below = line_src[qb * stride];
                        const float above = line_src[qa * stride];
                        acc += (SymmetryKind == SM_NDIMAGE_SYMMETRY_ODD ? below - above : below + above) * w[pairs - d];
                        if (++mb == period) mb = 0;
                        if (--ma < 0) ma = period - 1;
                    }
                }
            } else if (mode == 3u) {  // mirror
                if (length == 1) {
                    const float v0 = line_src[0];
                    acc = v0 * w[pairs];
                    const float diff = (SymmetryKind == SM_NDIMAGE_SYMMETRY_ODD ? v0 - v0 : v0 + v0);
                    for (int d = pairs; d >= 1; --d) acc += diff * w[pairs - d];
                } else {
                    const int period = 2 * length - 2;
                    int mc = centre % period;
                    if (mc < 0) mc += period;
                    const int qc = mc < length ? mc : period - mc;
                    acc = line_src[qc * stride] * w[pairs];
                    if (pairs >= 1) {
                        int mb = (centre - pairs) % period;
                        if (mb < 0) mb += period;
                        int ma = (centre + pairs) % period;
                        if (ma < 0) ma += period;
                        for (int d = pairs; d >= 1; --d) {
                            const int qb = mb < length ? mb : period - mb;
                            const int qa = ma < length ? ma : period - ma;
                            const float below = line_src[qb * stride];
                            const float above = line_src[qa * stride];
                            acc += (SymmetryKind == SM_NDIMAGE_SYMMETRY_ODD ? below - above : below + above) * w[pairs - d];
                            if (++mb == period) mb = 0;
                            if (--ma < 0) ma = period - 1;
                        }
                    }
                }
            } else {  // constant
                const float vc = (centre >= 0 && centre < length) ? line_src[centre * stride] : cval;
                acc = vc * w[pairs];
                const int d_both_out = max(centre, length - 1 - centre);
                const int d_in_stop = max(0, min(pairs, min(centre, length - 1 - centre)));
                int d = pairs;
                if (d > d_both_out) {
                    const float diff = (SymmetryKind == SM_NDIMAGE_SYMMETRY_ODD ? cval - cval : cval + cval);
                    for (; d > d_both_out; --d) acc += diff * w[pairs - d];
                }
                for (; d > d_in_stop; --d) {
                    const int pb = centre - d, pa = centre + d;
                    const float below = (pb >= 0 && pb < length) ? line_src[pb * stride] : cval;
                    const float above = (pa >= 0 && pa < length) ? line_src[pa * stride] : cval;
                    acc += (SymmetryKind == SM_NDIMAGE_SYMMETRY_ODD ? below - above : below + above) * w[pairs - d];
                }
                if (d >= 1) {
                    device const float* c_src = line_src + centre * stride;
                    for (; d >= 1; --d) {
                        const float below = c_src[-d * stride], above = c_src[d * stride];
                        acc += (SymmetryKind == SM_NDIMAGE_SYMMETRY_ODD ? below - above : below + above) * w[pairs - d];
                    }
                }
            }
        }
    }
    sm_ndimage_store(dst, index, acc);
}

#define SM_NDIMAGE_CORRELATE1D_KERNEL(name, element, symmetry)                                      \
    kernel void name(constant SMNdimageCorrelateParams& p [[buffer(0)]],                            \
                     device const float* src [[buffer(1)]],                                         \
                     device element* dst [[buffer(2)]],                                             \
                     constant float* weights [[buffer(3)]],                                         \
                     uint3 pos [[thread_position_in_grid]]) {                                       \
        sm_ndimage_correlate1d<device element*, symmetry>(p, src, dst, weights, pos);              \
    }

SM_NDIMAGE_CORRELATE1D_KERNEL(sm_ndimage_correlate1d_any_f32, float, SM_NDIMAGE_SYMMETRY_NONE)
SM_NDIMAGE_CORRELATE1D_KERNEL(sm_ndimage_correlate1d_even_f32, float, SM_NDIMAGE_SYMMETRY_EVEN)
SM_NDIMAGE_CORRELATE1D_KERNEL(sm_ndimage_correlate1d_odd_f32, float, SM_NDIMAGE_SYMMETRY_ODD)
SM_NDIMAGE_CORRELATE1D_KERNEL(sm_ndimage_correlate1d_any_f64, ulong, SM_NDIMAGE_SYMMETRY_NONE)
SM_NDIMAGE_CORRELATE1D_KERNEL(sm_ndimage_correlate1d_even_f64, ulong, SM_NDIMAGE_SYMMETRY_EVEN)
SM_NDIMAGE_CORRELATE1D_KERNEL(sm_ndimage_correlate1d_odd_f64, ulong, SM_NDIMAGE_SYMMETRY_ODD)

static inline int sm_ndimage_extend1d(int pos, int length, uint mode) {
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

template <bool IsMax>
static inline float sm_ndimage_minmax_step(float acc, float v) {
    return IsMax ? (v > acc ? v : acc) : (v < acc ? v : acc);
}

template <typename Destination, bool IsMax>
static inline void sm_ndimage_minmax1d_impl(constant SMNdimageMinMax1DParams& p, device const float* src,
                                            Destination dst, uint3 pos) {
    if (pos.x >= p.extent[0] || pos.y >= p.extent[1] || pos.z >= p.extent[2]) return;
    const uint index = (pos.z * p.extent[1] + pos.y) * p.extent[0] + pos.x;
    const int at = int(p.axis_in_y != 0u ? pos.y : pos.x);
    const int length = int(p.length);
    const int stride = int(p.stride);
    const int taps = int(p.taps);
    const int first = at - int(p.before);
    float acc;
    if (first >= 0 && first + taps <= length) {
        device const float* window = src + (int(index) - int(p.before) * stride);
        acc = window[0];
        for (int j = 1; j < taps; ++j) {
            acc = sm_ndimage_minmax_step<IsMax>(acc, window[j * stride]);
        }
    } else {
        device const float* line_src = src + (int(index) - at * stride);
        const uint mode = p.mode;
        const float cval = p.cval;
        const int q0 = sm_ndimage_extend1d(first, length, mode);
        acc = q0 < 0 ? cval : line_src[q0 * stride];
        for (int j = 1; j < taps; ++j) {
            const int q = sm_ndimage_extend1d(first + j, length, mode);
            const float v = q < 0 ? cval : line_src[q * stride];
            acc = sm_ndimage_minmax_step<IsMax>(acc, v);
        }
    }
    sm_ndimage_store(dst, index, acc);
}

#define SM_NDIMAGE_MINMAX1D_KERNEL(name, element, is_max)                                           \
    kernel void name(constant SMNdimageMinMax1DParams& p [[buffer(0)]],                             \
                     device const float* src [[buffer(1)]],                                         \
                     device element* dst [[buffer(2)]],                                             \
                     uint3 pos [[thread_position_in_grid]]) {                                       \
        sm_ndimage_minmax1d_impl<device element*, is_max>(p, src, dst, pos);                        \
    }

SM_NDIMAGE_MINMAX1D_KERNEL(sm_ndimage_minmax1d_min_f32, float, false)
SM_NDIMAGE_MINMAX1D_KERNEL(sm_ndimage_minmax1d_max_f32, float, true)
SM_NDIMAGE_MINMAX1D_KERNEL(sm_ndimage_minmax1d_min_f64, ulong, false)
SM_NDIMAGE_MINMAX1D_KERNEL(sm_ndimage_minmax1d_max_f64, ulong, true)

template <bool IsMax>
static inline void sm_ndimage_minmax1d_vh_pass1_impl(constant SMNdimageMinMax1DParams& p,
                                                     device const float* src,
                                                     device float* prefix,
                                                     device float* suffix,
                                                     uint3 pos) {
    if (pos.x >= p.extent[0] || pos.y >= p.extent[1] || pos.z >= p.extent[2]) return;
    const uint blk = p.axis_in_y != 0u ? pos.y : pos.x;
    const uint t0 = blk * p.taps;
    if (t0 >= p.ext_len) return;
    const uint t1 = min(t0 + p.taps, p.ext_len);
    const int length = int(p.length);
    const int before = int(p.before);
    const uint stride = p.stride;
    const uint mode = p.mode;
    const float cval = p.cval;
    uint line_src_base;
    uint line_ext_base;
    if (p.axis_in_y != 0u) {
        line_src_base = pos.z * p.length * stride + pos.x;
        line_ext_base = pos.z * p.ext_len * stride + pos.x;
    } else {
        line_src_base = pos.y * p.length;
        line_ext_base = pos.y * p.ext_len;
    }
    // Forward prefix scan
    float acc;
    {
        const int q = sm_ndimage_extend1d(int(t0) - before, length, mode);
        acc = q < 0 ? cval : src[line_src_base + uint(q) * stride];
        prefix[line_ext_base + t0 * stride] = acc;
    }
    for (uint t = t0 + 1u; t < t1; ++t) {
        const int q = sm_ndimage_extend1d(int(t) - before, length, mode);
        const float v = q < 0 ? cval : src[line_src_base + uint(q) * stride];
        acc = sm_ndimage_minmax_step<IsMax>(acc, v);
        prefix[line_ext_base + t * stride] = acc;
    }
    // Backward suffix scan
    {
        const uint tl = t1 - 1u;
        const int q = sm_ndimage_extend1d(int(tl) - before, length, mode);
        acc = q < 0 ? cval : src[line_src_base + uint(q) * stride];
        suffix[line_ext_base + tl * stride] = acc;
    }
    for (uint t = t1 - 1u; t > t0; --t) {
        const uint idx = t - 1u;
        const int q = sm_ndimage_extend1d(int(idx) - before, length, mode);
        const float v = q < 0 ? cval : src[line_src_base + uint(q) * stride];
        acc = sm_ndimage_minmax_step<IsMax>(acc, v);
        suffix[line_ext_base + idx * stride] = acc;
    }
}

kernel void sm_ndimage_minmax1d_vh_pass1_min(constant SMNdimageMinMax1DParams& p [[buffer(0)]],
                                             device const float* src [[buffer(1)]],
                                             device float* prefix [[buffer(2)]],
                                             device float* suffix [[buffer(3)]],
                                             uint3 pos [[thread_position_in_grid]]) {
    sm_ndimage_minmax1d_vh_pass1_impl<false>(p, src, prefix, suffix, pos);
}

kernel void sm_ndimage_minmax1d_vh_pass1_max(constant SMNdimageMinMax1DParams& p [[buffer(0)]],
                                             device const float* src [[buffer(1)]],
                                             device float* prefix [[buffer(2)]],
                                             device float* suffix [[buffer(3)]],
                                             uint3 pos [[thread_position_in_grid]]) {
    sm_ndimage_minmax1d_vh_pass1_impl<true>(p, src, prefix, suffix, pos);
}

template <typename Destination, bool IsMax>
static inline void sm_ndimage_minmax1d_vh_pass2_impl(constant SMNdimageMinMax1DParams& p,
                                                     device const float* prefix,
                                                     device const float* suffix,
                                                     Destination dst,
                                                     uint3 pos) {
    if (pos.x >= p.extent[0] || pos.y >= p.extent[1] || pos.z >= p.extent[2]) return;
    const uint index = (pos.z * p.extent[1] + pos.y) * p.extent[0] + pos.x;
    const uint at = p.axis_in_y != 0u ? pos.y : pos.x;
    const uint stride = p.stride;
    const uint line_ext_base = p.axis_in_y != 0u ? (pos.z * p.ext_len * stride + pos.x) : (pos.y * p.ext_len);
    const float a = suffix[line_ext_base + at * stride];
    const float b = prefix[line_ext_base + (at + p.taps - 1u) * stride];
    sm_ndimage_store(dst, index, sm_ndimage_minmax_step<IsMax>(a, b));
}

#define SM_NDIMAGE_MINMAX1D_VH_PASS2_KERNEL(name, element, is_max)                                  \
    kernel void name(constant SMNdimageMinMax1DParams& p [[buffer(0)]],                             \
                     device const float* prefix [[buffer(1)]],                                      \
                     device const float* suffix [[buffer(2)]],                                      \
                     device element* dst [[buffer(3)]],                                             \
                     uint3 pos [[thread_position_in_grid]]) {                                       \
        sm_ndimage_minmax1d_vh_pass2_impl<device element*, is_max>(p, prefix, suffix, dst, pos);    \
    }

SM_NDIMAGE_MINMAX1D_VH_PASS2_KERNEL(sm_ndimage_minmax1d_vh_pass2_min_f32, float, false)
SM_NDIMAGE_MINMAX1D_VH_PASS2_KERNEL(sm_ndimage_minmax1d_vh_pass2_max_f32, float, true)
SM_NDIMAGE_MINMAX1D_VH_PASS2_KERNEL(sm_ndimage_minmax1d_vh_pass2_min_f64, ulong, false)
SM_NDIMAGE_MINMAX1D_VH_PASS2_KERNEL(sm_ndimage_minmax1d_vh_pass2_max_f64, ulong, true)

template <typename Destination>
static inline void sm_ndimage_combine_impl(constant SMNdimageCombineParams& p,
                                           device const float* a,
                                           device const float* b,
                                           Destination dst,
                                           uint gid) {
    if (gid >= p.elements) return;
    const float va = a[gid];
    float out;
    if (p.op == SM_NDIMAGE_COMBINE_OP_COPY) {
        out = va;
    } else if (p.op == SM_NDIMAGE_COMBINE_OP_SUM2) {
        out = va + b[gid];
    } else if (p.op == SM_NDIMAGE_COMBINE_OP_SUB2) {
        out = va - b[gid];
    } else if (p.op == SM_NDIMAGE_COMBINE_OP_SQRT_SQ1) {
        out = sqrt(va * va);
    } else if (p.op == SM_NDIMAGE_COMBINE_OP_SUM_SQ2) {
        const float vb = b[gid];
        out = va * va + vb * vb;
    } else if (p.op == SM_NDIMAGE_COMBINE_OP_ADD_SQ) {
        const float vb = b[gid];
        out = va + vb * vb;
    } else if (p.op == SM_NDIMAGE_COMBINE_OP_SQRT_SUM_SQ2) {
        const float vb = b[gid];
        out = sqrt(va * va + vb * vb);
    } else {
        const float vb = b[gid];
        out = sqrt(va + vb * vb);
    }
    sm_ndimage_store(dst, gid, out);
}

kernel void sm_ndimage_combine_f32(constant SMNdimageCombineParams& p [[buffer(0)]],
                                   device const float* a [[buffer(1)]],
                                   device const float* b [[buffer(2)]],
                                   device float* dst [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
    sm_ndimage_combine_impl<device float*>(p, a, b, dst, gid);
}

kernel void sm_ndimage_combine_f64(constant SMNdimageCombineParams& p [[buffer(0)]],
                                   device const float* a [[buffer(1)]],
                                   device const float* b [[buffer(2)]],
                                   device ulong* dst [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
    sm_ndimage_combine_impl<device ulong*>(p, a, b, dst, gid);
}

kernel void sm_ndimage_combine_morph_laplace_f32(constant SMNdimageCombineParams& p [[buffer(0)]],
                                                 device const float* a [[buffer(1)]],
                                                 device const float* b [[buffer(2)]],
                                                 device const float* c [[buffer(3)]],
                                                 device float* dst [[buffer(4)]],
                                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= p.elements) return;
    const float vc = c[gid];
    dst[gid] = ((a[gid] + b[gid]) - vc) - vc;
}

kernel void sm_ndimage_combine_morph_laplace_f64(constant SMNdimageCombineParams& p [[buffer(0)]],
                                                 device const float* a [[buffer(1)]],
                                                 device const float* b [[buffer(2)]],
                                                 device const float* c [[buffer(3)]],
                                                 device ulong* dst [[buffer(4)]],
                                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= p.elements) return;
    const float vc = c[gid];
    sm_ndimage_store(dst, gid, ((a[gid] + b[gid]) - vc) - vc);
}
