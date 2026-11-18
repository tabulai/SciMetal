// E4 probe kernels: 1-D correlation along one axis of a row-major H x W image with
// scipy.ndimage 'reflect' boundary handling (d c b a | a b c d | d c b a), float32 arithmetic.
//
// Source / destination element type is a template parameter:
//   float  : plain float32 buffer
//   ulong  : float64 NumPy memory accessed as raw IEEE-754 bit patterns; narrowed with
//            round-to-nearest-even on load, widened exactly on store ("fused" conversion).
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint W;
    uint H;
    uint radius;
    uint axis;    // 0: along rows index (vertical), 1: along columns index (horizontal)
    uint span;    // outputs per thread along the filter axis (sliding-window variants)
    uint pad0;
    uint pad1;
    uint pad2;
};

// ---- float64 bits <-> float32 (same code as probe E3, validated bit-for-bit there) -------------
static inline uint narrow_bits(ulong b) {
    const uint hi = uint(b >> 32);
    const uint lo = uint(b);
    const uint sign = hi & 0x80000000u;
    const uint e = (hi >> 20) & 0x7FFu;
    const uint mant = ((hi & 0x000FFFFFu) << 3) | (lo >> 29);
    const uint rem = lo & 0x1FFFFFFFu;
    if (e == 0x7FFu) {
        const bool is_nan = (mant | rem) != 0u;
        return sign | 0x7F800000u | (is_nan ? (0x00400000u | mant) : 0u);
    }
    const int E = int(e) - 896;
    if (E >= 255) return sign | 0x7F800000u;
    if (E >= 1) {
        uint r = (uint(E) << 23) | mant;
        r += (rem > 0x10000000u || (rem == 0x10000000u && (mant & 1u) != 0u)) ? 1u : 0u;
        return sign | r;
    }
    if (E < -23) return sign;
    const ulong M = (b & 0x000FFFFFFFFFFFFFul) | 0x0010000000000000ul;
    const uint shift = uint(30 - E);
    const ulong q = M >> shift;
    const ulong r2 = M & ((1ul << shift) - 1ul);
    const ulong tie = 1ul << (shift - 1u);
    uint r = uint(q);
    r += (r2 > tie || (r2 == tie && (r & 1u) != 0u)) ? 1u : 0u;
    return sign | r;
}

static inline ulong widen_bits(uint b) {
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

// Cheaper narrowing for the fused read path: exact for every float64 whose float32 result is a
// normal number (same rounding), but no subnormal results (flushed to +-0, which the GPU float
// datapath does anyway) and NaNs become a canonical quiet NaN.
static inline float narrow_fast(ulong b) {
    const uint hi = uint(b >> 32);
    const uint lo = uint(b);
    const uint sign = hi & 0x80000000u;
    const uint e = (hi >> 20) & 0x7FFu;
    const uint mant = ((hi & 0x000FFFFFu) << 3) | (lo >> 29);
    const uint rem = lo & 0x1FFFFFFFu;
    uint r = ((e - 896u) << 23) | mant;
    r += (rem > 0x10000000u || (rem == 0x10000000u && (mant & 1u) != 0u)) ? 1u : 0u;
    r = e < 897u ? 0u : r;                                     // underflow -> 0
    r = e > 1150u ? 0x7F800000u : r;                            // overflow / inf
    r = (e == 0x7FFu && (mant | rem) != 0u) ? 0x7FC00000u : r;  // NaN
    return as_type<float>(sign | r);
}

static inline float ld(const device float* s, uint i) { return s[i]; }
static inline float ld(const device ulong* s, uint i) { return as_type<float>(narrow_bits(s[i])); }
static inline void st(device float* d, uint i, float v) { d[i] = v; }
static inline void st(device ulong* d, uint i, float v) { d[i] = widen_bits(as_type<uint>(v)); }

// ndimage 'reflect': period 2n, half-sample symmetric.
static inline int reflect_idx(int i, int n) {
    const int period = 2 * n;
    i = i % period;
    i = i < 0 ? i + period : i;
    return i >= n ? period - 1 - i : i;
}

// One output per thread.  Summation order is SciPy's symmetric-kernel order:
// centre tap first, then pairs from the outermost inwards.
template <typename S, typename D>
static inline void corr1d(S src, D dst, constant float* w, constant Params& p, uint2 gid) {
    const int r = int(p.radius);
    const bool ax0 = p.axis == 0u;
    const int pos = int(ax0 ? gid.y : gid.x);
    const int len = int(ax0 ? p.H : p.W);
    const int stride = ax0 ? int(p.W) : 1;
    const int base = int(gid.y * p.W + gid.x);
    float acc = w[r] * ld(src, uint(base));
    if (pos >= r && pos + r < len) {
        for (int d = r; d >= 1; --d)
            acc += (ld(src, uint(base - d * stride)) + ld(src, uint(base + d * stride))) * w[r - d];
    } else {
        for (int d = r; d >= 1; --d) {
            const int a = reflect_idx(pos - d, len);
            const int b = reflect_idx(pos + d, len);
            acc += (ld(src, uint(base + (a - pos) * stride)) + ld(src, uint(base + (b - pos) * stride))) * w[r - d];
        }
    }
    st(dst, uint(base), acc);
}

kernel void corr_f32_f32(const device float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                         constant float* w [[buffer(2)]], constant Params& p [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    corr1d(src, dst, w, p, gid);
}
kernel void corr_f64_f32(const device ulong* src [[buffer(0)]], device float* dst [[buffer(1)]],
                         constant float* w [[buffer(2)]], constant Params& p [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    corr1d(src, dst, w, p, gid);
}
kernel void corr_f32_f64(const device float* src [[buffer(0)]], device ulong* dst [[buffer(1)]],
                         constant float* w [[buffer(2)]], constant Params& p [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    corr1d(src, dst, w, p, gid);
}

// Fused read with the cheaper narrowing (see narrow_fast).
struct FastSrc { const device ulong* p; };
static inline float ld(FastSrc s, uint i) { return narrow_fast(s.p[i]); }
kernel void corr_f64fast_f32(const device ulong* src [[buffer(0)]], device float* dst [[buffer(1)]],
                             constant float* w [[buffer(2)]], constant Params& p [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
    FastSrc s;
    s.p = src;
    corr1d(s, dst, w, p, gid);
}

// ---- sliding-window variant: one thread produces `span` consecutive outputs along the filter
// axis from a single window of (2r + span) loads.  gid indexes (orthogonal coordinate, block).
#define MAX_SPAN 8
#define MAX_WIN (2 * 64 + MAX_SPAN)
template <typename S, typename D>
static inline void corr1d_span(S src, D dst, constant float* w, constant Params& p, uint2 gid) {
    const int r = int(p.radius);
    const int span = int(p.span);
    const bool ax0 = p.axis == 0u;
    const int len = int(ax0 ? p.H : p.W);
    const int stride = ax0 ? int(p.W) : 1;
    // gid.x: column (axis 0) or column-block (axis 1); gid.y: row-block (axis 0) or row (axis 1)
    const int pos0 = (ax0 ? int(gid.y) : int(gid.x)) * span;
    if (pos0 >= len) return;
    const int ortho = ax0 ? int(gid.x) : int(gid.y) * int(p.W);
    float win[MAX_WIN];
    const int n = 2 * r + span;
    if (pos0 >= r && pos0 + span - 1 + r < len) {
        const int first = ortho + (pos0 - r) * stride;
        for (int k = 0; k < n; ++k) win[k] = ld(src, uint(first + k * stride));
    } else {
        for (int k = 0; k < n; ++k) win[k] = ld(src, uint(ortho + reflect_idx(pos0 - r + k, len) * stride));
    }
    for (int s = 0; s < span; ++s) {
        if (pos0 + s >= len) break;
        float acc = w[r] * win[s + r];
        for (int d = r; d >= 1; --d) acc += (win[s + r - d] + win[s + r + d]) * w[r - d];
        st(dst, uint(ortho + (pos0 + s) * stride), acc);
    }
}

kernel void corr_span_f32_f32(const device float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                              constant float* w [[buffer(2)]], constant Params& p [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
    corr1d_span(src, dst, w, p, gid);
}
kernel void corr_span_f64_f32(const device ulong* src [[buffer(0)]], device float* dst [[buffer(1)]],
                              constant float* w [[buffer(2)]], constant Params& p [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
    corr1d_span(src, dst, w, p, gid);
}
kernel void corr_span_f32_f64(const device float* src [[buffer(0)]], device ulong* dst [[buffer(1)]],
                              constant float* w [[buffer(2)]], constant Params& p [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
    corr1d_span(src, dst, w, p, gid);
}

// ---- stand-alone conversion passes (one element per thread; best shape found in E3).
// Pure bit moves: the float32 side is addressed as uint so no float register sees the value.
kernel void narrow_pass(const device ulong* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                        constant Params& p [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.y * p.W + gid.x;
    dst[i] = narrow_bits(src[i]);
}
kernel void widen_pass(const device uint* src [[buffer(0)]], device ulong* dst [[buffer(1)]],
                       constant Params& p [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.y * p.W + gid.x;
    dst[i] = widen_bits(src[i]);
}
