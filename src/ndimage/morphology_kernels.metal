// scipy.ndimage dense min/max morphology and rank/median Metal kernels.
// Compiled with -std=metal3.1 -O2 -fno-fast-math -ffp-contract=off.
#include <metal_stdlib>
#include "morphology_kernels.h"
using namespace metal;

static inline ulong sm_morph_widen(uint b) {
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

static inline void sm_morph_store(device float* dst, uint index, float value) { dst[index] = value; }
static inline void sm_morph_store(device ulong* dst, uint index, float value) {
    dst[index] = sm_morph_widen(as_type<uint>(value));
}

static inline int sm_morph_extend(int pos, int length, uint mode) {
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

// ---- Dense N-D min/max (flat & non-flat grayscale morphology) --------------

template <typename DstT, bool IsMax, bool NonFlat>
static inline void sm_morph_dense_minmax_impl(constant SMNdimageDenseMinMaxParams& p,
                                              device const float* src,
                                              device DstT* dst,
                                              device const SMNdimageMorphTap* taps,
                                              uint3 pos) {
    uint gid;
    int p_coord[8];
    if (p.ndim == 2u) {
        if (pos.x >= p.in_shape[1] || pos.y >= p.in_shape[0]) return;
        p_coord[0] = int(pos.y);
        p_coord[1] = int(pos.x);
        gid = pos.y * p.in_shape[1] + pos.x;
    } else {
        gid = pos.x;
        if (gid >= p.elements) return;
        uint rem = gid;
        for (uint d = 0u; d < p.ndim; ++d) {
            const uint s = p.in_stride[d];
            const uint c = (d + 1u == p.ndim) ? rem : rem / s;
            p_coord[d] = int(c);
            rem -= c * s;
        }
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

    float acc;
    if (is_interior) {
        const long base_in = long(gid);
        acc = src[base_in + long(taps[0].linear_offset)];
        if (NonFlat) acc += taps[0].struct_offset;
        for (uint t = 1u; t < p.num_taps; ++t) {
            float v = src[base_in + long(taps[t].linear_offset)];
            if (NonFlat) v += taps[t].struct_offset;
            acc = IsMax ? (v > acc ? v : acc) : (v < acc ? v : acc);
        }
    } else {
        for (uint t = 0u; t < p.num_taps; ++t) {
            int in_idx = 0;
            bool in_bounds = true;
            for (uint d = 0u; d < p.ndim; ++d) {
                const int q = sm_morph_extend(p_coord[d] + taps[t].coord[d], int(p.in_shape[d]), p.mode);
                if (q < 0) {
                    in_bounds = false;
                    break;
                }
                in_idx += q * int(p.in_stride[d]);
            }
            float v = in_bounds ? src[in_idx] : p.cval;
            if (NonFlat) v += taps[t].struct_offset;
            if (t == 0u) {
                acc = v;
            } else {
                acc = IsMax ? (v > acc ? v : acc) : (v < acc ? v : acc);
            }
        }
    }
    sm_morph_store(dst, gid, acc);
}

#define SM_MORPH_DENSE_MINMAX_KERNEL(name, element, is_max, nonflat)                                \
    kernel void name(constant SMNdimageDenseMinMaxParams& p [[buffer(0)]],                          \
                     device const float* src [[buffer(1)]],                                         \
                     device element* dst [[buffer(2)]],                                             \
                     device const SMNdimageMorphTap* taps [[buffer(3)]],                            \
                     uint3 pos [[thread_position_in_grid]]) {                                       \
        sm_morph_dense_minmax_impl<element, is_max, nonflat>(p, src, dst, taps, pos);               \
    }

SM_MORPH_DENSE_MINMAX_KERNEL(sm_ndimage_dense_minmax_min_flat_f32, float, false, false)
SM_MORPH_DENSE_MINMAX_KERNEL(sm_ndimage_dense_minmax_max_flat_f32, float, true, false)
SM_MORPH_DENSE_MINMAX_KERNEL(sm_ndimage_dense_minmax_min_nonflat_f32, float, false, true)
SM_MORPH_DENSE_MINMAX_KERNEL(sm_ndimage_dense_minmax_max_nonflat_f32, float, true, true)
SM_MORPH_DENSE_MINMAX_KERNEL(sm_ndimage_dense_minmax_min_flat_f64, ulong, false, false)
SM_MORPH_DENSE_MINMAX_KERNEL(sm_ndimage_dense_minmax_max_flat_f64, ulong, true, false)
SM_MORPH_DENSE_MINMAX_KERNEL(sm_ndimage_dense_minmax_min_nonflat_f64, ulong, false, true)
SM_MORPH_DENSE_MINMAX_KERNEL(sm_ndimage_dense_minmax_max_nonflat_f64, ulong, true, true)

// ---- Register-only sorting & Forgetful Selection primitives ----------------

#define SM_CAS(a, b) do { const float _lo = min(a, b); const float _hi = max(a, b); a = _lo; b = _hi; } while (0)

static inline float sm_median3(float a, float b, float c) {
    return max(min(a, b), min(max(a, b), c));
}

// Moves min of r[0..N-1] to r[0] and max of r[0..N-1] to r[N-1] using only
// compile-time indices (zero register spilling, branchless).
template <int N>
static inline void sm_minmax_pull(thread float* r) {
    for (int i = 0; i + 1 < N; i += 2) {
        SM_CAS(r[i], r[i + 1]);
    }
    for (int i = 2; i < N; i += 2) {
        SM_CAS(r[0], r[i]);
    }
    for (int i = 1; i < N - 1; i += 2) {
        SM_CAS(r[i], r[N - 1]);
    }
}

// ---- Dedicated 2-D Tiled Median 3x3 (optimal 19-comparator network) --------

template <typename DstT>
static inline void sm_morph_median_3x3_impl(constant SMNdimageMedian2DParams& p,
                                            device const float* src,
                                            device DstT* dst,
                                            threadgroup float (&tile)[18][18],
                                            uint2 gpos,
                                            uint2 lpos,
                                            uint2 tg_id) {
    constexpr uint TILE_W = 18u;
    constexpr uint TILE_H = 18u;

    const int base_y = int(tg_id.y * 16u) + p.start_y;
    const int base_x = int(tg_id.x * 16u) + p.start_x;
    const int H = int(p.height);
    const int W = int(p.width);
    const uint tid = lpos.y * 16u + lpos.x;

    for (uint idx = tid; idx < TILE_H * TILE_W; idx += 256u) {
        const uint ry = idx / TILE_W;
        const uint rx = idx - ry * TILE_W;
        const int sy = sm_morph_extend(base_y + int(ry), H, p.mode);
        const int sx = sm_morph_extend(base_x + int(rx), W, p.mode);
        tile[ry][rx] = (sy >= 0 && sx >= 0) ? src[uint(sy) * p.width + uint(sx)] : p.cval;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gpos.x >= p.width || gpos.y >= p.height) return;

    const uint ty = lpos.y;
    const uint tx = lpos.x;
    float v0 = tile[ty + 0u][tx + 0u], v1 = tile[ty + 0u][tx + 1u], v2 = tile[ty + 0u][tx + 2u];
    float v3 = tile[ty + 1u][tx + 0u], v4 = tile[ty + 1u][tx + 1u], v5 = tile[ty + 1u][tx + 2u];
    float v6 = tile[ty + 2u][tx + 0u], v7 = tile[ty + 2u][tx + 1u], v8 = tile[ty + 2u][tx + 2u];

    SM_CAS(v0, v1); SM_CAS(v1, v2); SM_CAS(v0, v1);
    SM_CAS(v3, v4); SM_CAS(v4, v5); SM_CAS(v3, v4);
    SM_CAS(v6, v7); SM_CAS(v7, v8); SM_CAS(v6, v7);

    const float max_min = max(max(v0, v3), v6);
    const float min_max = min(min(v2, v5), v8);
    const float med_mid = sm_median3(v1, v4, v7);
    const float med = sm_median3(max_min, med_mid, min_max);

    sm_morph_store(dst, gpos.y * p.width + gpos.x, med);
}

kernel void sm_ndimage_median_3x3_f32(constant SMNdimageMedian2DParams& p [[buffer(0)]],
                                      device const float* src [[buffer(1)]],
                                      device float* dst [[buffer(2)]],
                                      uint2 gpos [[thread_position_in_grid]],
                                      uint2 lpos [[thread_position_in_threadgroup]],
                                      uint2 tg_id [[threadgroup_position_in_grid]]) {
    threadgroup float tile[18][18];
    sm_morph_median_3x3_impl<float>(p, src, dst, tile, gpos, lpos, tg_id);
}

kernel void sm_ndimage_median_3x3_f64(constant SMNdimageMedian2DParams& p [[buffer(0)]],
                                      device const float* src [[buffer(1)]],
                                      device ulong* dst [[buffer(2)]],
                                      uint2 gpos [[thread_position_in_grid]],
                                      uint2 lpos [[thread_position_in_threadgroup]],
                                      uint2 tg_id [[threadgroup_position_in_grid]]) {
    threadgroup float tile[18][18];
    sm_morph_median_3x3_impl<ulong>(p, src, dst, tile, gpos, lpos, tg_id);
}

// ---- Dedicated 2-D Tiled Median 5x5 (14-register Forgetful Selection) ------

template <typename DstT>
static inline void sm_morph_median_5x5_impl(constant SMNdimageMedian2DParams& p,
                                            device const float* src,
                                            device DstT* dst,
                                            threadgroup float (&tile)[20][20],
                                            uint2 gpos,
                                            uint2 lpos,
                                            uint2 tg_id) {
    constexpr uint TILE_W = 20u;
    constexpr uint TILE_H = 20u;

    const int base_y = int(tg_id.y * 16u) + p.start_y;
    const int base_x = int(tg_id.x * 16u) + p.start_x;
    const int H = int(p.height);
    const int W = int(p.width);
    const uint tid = lpos.y * 16u + lpos.x;

    for (uint idx = tid; idx < TILE_H * TILE_W; idx += 256u) {
        const uint ry = idx / TILE_W;
        const uint rx = idx - ry * TILE_W;
        const int sy = sm_morph_extend(base_y + int(ry), H, p.mode);
        const int sx = sm_morph_extend(base_x + int(rx), W, p.mode);
        tile[ry][rx] = (sy >= 0 && sx >= 0) ? src[uint(sy) * p.width + uint(sx)] : p.cval;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gpos.x >= p.width || gpos.y >= p.height) return;

    const uint ty = lpos.y;
    const uint tx = lpos.x;
    float r[14];
    r[0]  = tile[ty + 0u][tx + 0u]; r[1]  = tile[ty + 0u][tx + 1u]; r[2]  = tile[ty + 0u][tx + 2u];
    r[3]  = tile[ty + 0u][tx + 3u]; r[4]  = tile[ty + 0u][tx + 4u];
    r[5]  = tile[ty + 1u][tx + 0u]; r[6]  = tile[ty + 1u][tx + 1u]; r[7]  = tile[ty + 1u][tx + 2u];
    r[8]  = tile[ty + 1u][tx + 3u]; r[9]  = tile[ty + 1u][tx + 4u];
    r[10] = tile[ty + 2u][tx + 0u]; r[11] = tile[ty + 2u][tx + 1u]; r[12] = tile[ty + 2u][tx + 2u];
    r[13] = tile[ty + 2u][tx + 3u];

    sm_minmax_pull<14>(r); r[0] = tile[ty + 2u][tx + 4u];
    sm_minmax_pull<13>(r); r[0] = tile[ty + 3u][tx + 0u];
    sm_minmax_pull<12>(r); r[0] = tile[ty + 3u][tx + 1u];
    sm_minmax_pull<11>(r); r[0] = tile[ty + 3u][tx + 2u];
    sm_minmax_pull<10>(r); r[0] = tile[ty + 3u][tx + 3u];
    sm_minmax_pull<9>(r);  r[0] = tile[ty + 3u][tx + 4u];
    sm_minmax_pull<8>(r);  r[0] = tile[ty + 4u][tx + 0u];
    sm_minmax_pull<7>(r);  r[0] = tile[ty + 4u][tx + 1u];
    sm_minmax_pull<6>(r);  r[0] = tile[ty + 4u][tx + 2u];
    sm_minmax_pull<5>(r);  r[0] = tile[ty + 4u][tx + 3u];
    sm_minmax_pull<4>(r);  r[0] = tile[ty + 4u][tx + 4u];

    sm_morph_store(dst, gpos.y * p.width + gpos.x, sm_median3(r[0], r[1], r[2]));
}

kernel void sm_ndimage_median_5x5_f32(constant SMNdimageMedian2DParams& p [[buffer(0)]],
                                      device const float* src [[buffer(1)]],
                                      device float* dst [[buffer(2)]],
                                      uint2 gpos [[thread_position_in_grid]],
                                      uint2 lpos [[thread_position_in_threadgroup]],
                                      uint2 tg_id [[threadgroup_position_in_grid]]) {
    threadgroup float tile[20][20];
    sm_morph_median_5x5_impl<float>(p, src, dst, tile, gpos, lpos, tg_id);
}

kernel void sm_ndimage_median_5x5_f64(constant SMNdimageMedian2DParams& p [[buffer(0)]],
                                      device const float* src [[buffer(1)]],
                                      device ulong* dst [[buffer(2)]],
                                      uint2 gpos [[thread_position_in_grid]],
                                      uint2 lpos [[thread_position_in_threadgroup]],
                                      uint2 tg_id [[threadgroup_position_in_grid]]) {
    threadgroup float tile[20][20];
    sm_morph_median_5x5_impl<ulong>(p, src, dst, tile, gpos, lpos, tg_id);
}

// ---- Dedicated 2-D Tiled Median 7x7 (26-register Forgetful Selection) ------

template <typename DstT>
static inline void sm_morph_median_7x7_impl(constant SMNdimageMedian2DParams& p,
                                            device const float* src,
                                            device DstT* dst,
                                            threadgroup float (&tile)[22][22],
                                            uint2 gpos,
                                            uint2 lpos,
                                            uint2 tg_id) {
    constexpr uint TILE_W = 22u;
    constexpr uint TILE_H = 22u;

    const int base_y = int(tg_id.y * 16u) + p.start_y;
    const int base_x = int(tg_id.x * 16u) + p.start_x;
    const int H = int(p.height);
    const int W = int(p.width);
    const uint tid = lpos.y * 16u + lpos.x;

    for (uint idx = tid; idx < TILE_H * TILE_W; idx += 256u) {
        const uint ry = idx / TILE_W;
        const uint rx = idx - ry * TILE_W;
        const int sy = sm_morph_extend(base_y + int(ry), H, p.mode);
        const int sx = sm_morph_extend(base_x + int(rx), W, p.mode);
        tile[ry][rx] = (sy >= 0 && sx >= 0) ? src[uint(sy) * p.width + uint(sx)] : p.cval;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gpos.x >= p.width || gpos.y >= p.height) return;

    const uint ty = lpos.y;
    const uint tx = lpos.x;
    float r[26];
    for (uint dy = 0u; dy < 3u; ++dy) {
        for (uint dx = 0u; dx < 7u; ++dx) {
            r[dy * 7u + dx] = tile[ty + dy][tx + dx];
        }
    }
    r[21] = tile[ty + 3u][tx + 0u];
    r[22] = tile[ty + 3u][tx + 1u];
    r[23] = tile[ty + 3u][tx + 2u];
    r[24] = tile[ty + 3u][tx + 3u];
    r[25] = tile[ty + 3u][tx + 4u];

    sm_minmax_pull<26>(r); r[0] = tile[ty + 3u][tx + 5u];
    sm_minmax_pull<25>(r); r[0] = tile[ty + 3u][tx + 6u];
    sm_minmax_pull<24>(r); r[0] = tile[ty + 4u][tx + 0u];
    sm_minmax_pull<23>(r); r[0] = tile[ty + 4u][tx + 1u];
    sm_minmax_pull<22>(r); r[0] = tile[ty + 4u][tx + 2u];
    sm_minmax_pull<21>(r); r[0] = tile[ty + 4u][tx + 3u];
    sm_minmax_pull<20>(r); r[0] = tile[ty + 4u][tx + 4u];
    sm_minmax_pull<19>(r); r[0] = tile[ty + 4u][tx + 5u];
    sm_minmax_pull<18>(r); r[0] = tile[ty + 4u][tx + 6u];
    sm_minmax_pull<17>(r); r[0] = tile[ty + 5u][tx + 0u];
    sm_minmax_pull<16>(r); r[0] = tile[ty + 5u][tx + 1u];
    sm_minmax_pull<15>(r); r[0] = tile[ty + 5u][tx + 2u];
    sm_minmax_pull<14>(r); r[0] = tile[ty + 5u][tx + 3u];
    sm_minmax_pull<13>(r); r[0] = tile[ty + 5u][tx + 4u];
    sm_minmax_pull<12>(r); r[0] = tile[ty + 5u][tx + 5u];
    sm_minmax_pull<11>(r); r[0] = tile[ty + 5u][tx + 6u];
    sm_minmax_pull<10>(r); r[0] = tile[ty + 6u][tx + 0u];
    sm_minmax_pull<9>(r);  r[0] = tile[ty + 6u][tx + 1u];
    sm_minmax_pull<8>(r);  r[0] = tile[ty + 6u][tx + 2u];
    sm_minmax_pull<7>(r);  r[0] = tile[ty + 6u][tx + 3u];
    sm_minmax_pull<6>(r);  r[0] = tile[ty + 6u][tx + 4u];
    sm_minmax_pull<5>(r);  r[0] = tile[ty + 6u][tx + 5u];
    sm_minmax_pull<4>(r);  r[0] = tile[ty + 6u][tx + 6u];

    sm_morph_store(dst, gpos.y * p.width + gpos.x, sm_median3(r[0], r[1], r[2]));
}

kernel void sm_ndimage_median_7x7_f32(constant SMNdimageMedian2DParams& p [[buffer(0)]],
                                      device const float* src [[buffer(1)]],
                                      device float* dst [[buffer(2)]],
                                      uint2 gpos [[thread_position_in_grid]],
                                      uint2 lpos [[thread_position_in_threadgroup]],
                                      uint2 tg_id [[threadgroup_position_in_grid]]) {
    threadgroup float tile[22][22];
    sm_morph_median_7x7_impl<float>(p, src, dst, tile, gpos, lpos, tg_id);
}

kernel void sm_ndimage_median_7x7_f64(constant SMNdimageMedian2DParams& p [[buffer(0)]],
                                      device const float* src [[buffer(1)]],
                                      device ulong* dst [[buffer(2)]],
                                      uint2 gpos [[thread_position_in_grid]],
                                      uint2 lpos [[thread_position_in_threadgroup]],
                                      uint2 tg_id [[threadgroup_position_in_grid]]) {
    threadgroup float tile[22][22];
    sm_morph_median_7x7_impl<ulong>(p, src, dst, tile, gpos, lpos, tg_id);
}

// ---- General N-D Rank / Median / Percentile (Forgetful Selection) ----------

static inline float sm_morph_load_tap(constant SMNdimageRankParams& p,
                                      device const float* src,
                                      device const SMNdimageMorphTap* taps,
                                      bool is_interior,
                                      long base_in,
                                      thread const int* p_coord,
                                      uint t) {
    if (is_interior) {
        return src[base_in + long(taps[t].linear_offset)];
    }
    int in_idx = 0;
    for (uint dim = 0u; dim < p.ndim; ++dim) {
        const int q = sm_morph_extend(p_coord[dim] + taps[t].coord[dim], int(p.in_shape[dim]), p.mode);
        if (q < 0) return p.cval;
        in_idx += q * int(p.in_stride[dim]);
    }
    return src[in_idx];
}

template <typename DstT, uint MaxS>
static inline void sm_morph_rank_impl(constant SMNdimageRankParams& p,
                                      device const float* src,
                                      device DstT* dst,
                                      device const SMNdimageMorphTap* taps,
                                      uint3 pos) {
    uint gid;
    int p_coord[8];
    if (p.ndim == 2u) {
        if (pos.x >= p.in_shape[1] || pos.y >= p.in_shape[0]) return;
        p_coord[0] = int(pos.y);
        p_coord[1] = int(pos.x);
        gid = pos.y * p.in_shape[1] + pos.x;
    } else {
        gid = pos.x;
        if (gid >= p.elements) return;
        uint rem = gid;
        for (uint d = 0u; d < p.ndim; ++d) {
            const uint s = p.in_stride[d];
            const uint c = (d + 1u == p.ndim) ? rem : rem / s;
            p_coord[d] = int(c);
            rem -= c * s;
        }
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

    const uint K = p.num_taps;
    const uint a = p.rank;
    const uint b = K - 1u - a;
    const bool drop_max_first = (a <= b);
    const uint s = (drop_max_first ? a : b) + 2u;
    const uint d_extra = drop_max_first ? (b - a) : (a - b);
    const long base_in = long(gid);

    float buf[MaxS];
    for (uint t = 0u; t < s; ++t) {
        buf[t] = sm_morph_load_tap(p, src, taps, is_interior, base_in, p_coord, t);
    }

    // One-sided drops until remaining a == b.
    for (uint t = s; t < s + d_extra; ++t) {
        const float next = sm_morph_load_tap(p, src, taps, is_interior, base_in, p_coord, t);
        if (drop_max_first) {
            uint hi_i = 0u;
            float hi_v = buf[0];
            for (uint j = 1u; j < s; ++j) {
                if (buf[j] > hi_v) {
                    hi_v = buf[j];
                    hi_i = j;
                }
            }
            buf[hi_i] = next;
        } else {
            uint lo_i = 0u;
            float lo_v = buf[0];
            for (uint j = 1u; j < s; ++j) {
                if (buf[j] < lo_v) {
                    lo_v = buf[j];
                    lo_i = j;
                }
            }
            buf[lo_i] = next;
        }
    }

    // Symmetric shrink from cur_s down to 3.
    uint cur_s = s;
    for (uint t = s + d_extra; t < K; ++t) {
        uint lo_i = 0u, hi_i = 1u;
        float lo_v = buf[0], hi_v = buf[1];
        if (hi_v < lo_v) {
            lo_i = 1u; hi_i = 0u;
            float tmp = lo_v; lo_v = hi_v; hi_v = tmp;
        }
        for (uint j = 2u; j < cur_s; ++j) {
            const float v = buf[j];
            if (v < lo_v) {
                lo_v = v;
                lo_i = j;
            } else if (v > hi_v) {
                hi_v = v;
                hi_i = j;
            }
        }
        const float next = sm_morph_load_tap(p, src, taps, is_interior, base_in, p_coord, t);
        const uint last = cur_s - 1u;
        if (lo_i == last) {
            buf[hi_i] = next;
        } else if (hi_i == last) {
            buf[lo_i] = next;
        } else {
            buf[lo_i] = buf[last];
            buf[hi_i] = next;
        }
        cur_s = last;
    }

    sm_morph_store(dst, gid, sm_median3(buf[0], buf[1], buf[2]));
}

#define SM_MORPH_RANK_KERNEL(name, element, max_s)                                                  \
    kernel void name(constant SMNdimageRankParams& p [[buffer(0)]],                                 \
                     device const float* src [[buffer(1)]],                                         \
                     device element* dst [[buffer(2)]],                                             \
                     device const SMNdimageMorphTap* taps [[buffer(3)]],                            \
                     uint3 pos [[thread_position_in_grid]]) {                                       \
        sm_morph_rank_impl<element, max_s>(p, src, dst, taps, pos);                                 \
    }

SM_MORPH_RANK_KERNEL(sm_ndimage_rank_small_f32, float, SM_NDIMAGE_RANK_SMALL_MAX_S)
SM_MORPH_RANK_KERNEL(sm_ndimage_rank_small_f64, ulong, SM_NDIMAGE_RANK_SMALL_MAX_S)
SM_MORPH_RANK_KERNEL(sm_ndimage_rank_medium_f32, float, SM_NDIMAGE_RANK_MEDIUM_MAX_S)
SM_MORPH_RANK_KERNEL(sm_ndimage_rank_medium_f64, ulong, SM_NDIMAGE_RANK_MEDIUM_MAX_S)
