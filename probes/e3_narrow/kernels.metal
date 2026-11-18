// E3 probe kernels: float64 <-> float32 conversion done entirely with integer bit
// manipulation, so a GPU kernel can consume / produce float64 NumPy memory directly.
//
//   narrow: IEEE-754 binary64 bits (read as ulong) -> correctly rounded (nearest-even)
//           binary32 bits.  Matches AArch64 FCVT / numpy.astype(float32) bit for bit,
//           including NaN payload truncation + quieting, overflow to inf, subnormals.
//   widen:  binary32 bits -> binary64 bits (exact), NaNs quieted like FCVT.
#include <metal_stdlib>
using namespace metal;

struct Params {
    ulong n;       // number of elements
    uint  width;   // grid width in threads
    uint  chunk;   // consecutive elements handled by one thread
};

static inline uint narrow_bits(ulong b) {
    const uint hi = uint(b >> 32);
    const uint lo = uint(b);
    const uint sign = hi & 0x80000000u;
    const uint e = (hi >> 20) & 0x7FFu;
    // top 23 fraction bits and the 29 discarded ones
    const uint mant = ((hi & 0x000FFFFFu) << 3) | (lo >> 29);
    const uint rem = lo & 0x1FFFFFFFu;
    if (e == 0x7FFu) {  // inf / NaN: quiet the NaN, keep the top 22 payload bits
        const bool is_nan = (mant | rem) != 0u;
        return sign | 0x7F800000u | (is_nan ? (0x00400000u | mant) : 0u);
    }
    const int E = int(e) - 896;  // biased binary32 exponent
    if (E >= 255) return sign | 0x7F800000u;  // overflow -> inf
    if (E >= 1) {  // normal result; a carry out of the mantissa correctly bumps the exponent (up to inf)
        uint r = (uint(E) << 23) | mant;
        r += (rem > 0x10000000u || (rem == 0x10000000u && (mant & 1u) != 0u)) ? 1u : 0u;
        return sign | r;
    }
    if (E < -23) return sign;  // below half of the smallest subnormal (or exactly half: ties to even = 0)
    // subnormal result: shift the 53-bit significand right by (30 - E) in [30, 53]
    const ulong M = (b & 0x000FFFFFFFFFFFFFul) | 0x0010000000000000ul;
    const uint shift = uint(30 - E);
    const ulong q = M >> shift;
    const ulong r2 = M & ((1ul << shift) - 1ul);
    const ulong tie = 1ul << (shift - 1u);
    uint r = uint(q);
    r += (r2 > tie || (r2 == tie && (r & 1u) != 0u)) ? 1u : 0u;
    return sign | r;  // may round up to FLT_MIN (0x00800000), which is the right answer
}

static inline ulong widen_bits(uint b) {
    const uint sign = b & 0x80000000u;
    const uint e = (b >> 23) & 0xFFu;
    uint m = b & 0x007FFFFFu;
    uint hi;
    if (e == 0xFFu) {  // inf / NaN (quieted)
        hi = sign | 0x7FF00000u | (m != 0u ? 0x00080000u : 0u) | (m >> 3);
    } else if (e == 0u) {
        if (m == 0u) return ulong(sign) << 32;  // +-0
        const uint s = clz(m) - 8u;             // normalise the subnormal
        m = (m << s) & 0x007FFFFFu;
        hi = sign | ((897u - s) << 20) | (m >> 3);
    } else {
        hi = sign | ((e + 896u) << 20) | (m >> 3);
    }
    return (ulong(hi) << 32) | ulong(m << 29);
}

// float64 bits -> float32 bits.  dst is declared uint so no float register ever sees the value.
kernel void narrow_f64_to_f32(const device ulong* src [[buffer(0)]],
                              device uint* dst [[buffer(1)]],
                              constant Params& p [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]]) {
    const ulong t = ulong(gid.y) * ulong(p.width) + ulong(gid.x);
    const ulong i0 = t * ulong(p.chunk);
    if (i0 >= p.n) return;
    const ulong i1 = min(i0 + ulong(p.chunk), p.n);
    for (ulong i = i0; i < i1; ++i) dst[i] = narrow_bits(src[i]);
}

// float32 bits -> float64 bits (written as ulong).
kernel void widen_f32_to_f64(const device uint* src [[buffer(0)]],
                             device ulong* dst [[buffer(1)]],
                             constant Params& p [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    const ulong t = ulong(gid.y) * ulong(p.width) + ulong(gid.x);
    const ulong i0 = t * ulong(p.chunk);
    if (i0 >= p.n) return;
    const ulong i1 = min(i0 + ulong(p.chunk), p.n);
    for (ulong i = i0; i < i1; ++i) dst[i] = widen_bits(src[i]);
}

// Variant that goes through a real float: bits -> float -> arithmetic -> bits.  Used to see
// what the GPU's float datapath does to NaN payloads / subnormals once a kernel actually
// computes with the narrowed value (x * 1.0f and x + 0.0f are IEEE identities for non-NaN).
kernel void narrow_through_float(const device ulong* src [[buffer(0)]],
                                 device float* dst [[buffer(1)]],
                                 constant Params& p [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    const ulong t = ulong(gid.y) * ulong(p.width) + ulong(gid.x);
    const ulong i0 = t * ulong(p.chunk);
    if (i0 >= p.n) return;
    const ulong i1 = min(i0 + ulong(p.chunk), p.n);
    for (ulong i = i0; i < i1; ++i) {
        const float f = as_type<float>(narrow_bits(src[i]));
        dst[i] = f * float(p.width != 0u ? 1.0f : 2.0f);  // runtime 1.0f so the multiply is not folded away
    }
}

// float -> ulong bits via a real float load (src declared float).
kernel void widen_from_float(const device float* src [[buffer(0)]],
                             device ulong* dst [[buffer(1)]],
                             constant Params& p [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    const ulong t = ulong(gid.y) * ulong(p.width) + ulong(gid.x);
    const ulong i0 = t * ulong(p.chunk);
    if (i0 >= p.n) return;
    const ulong i1 = min(i0 + ulong(p.chunk), p.n);
    for (ulong i = i0; i < i1; ++i) {
        const float f = src[i] * float(p.width != 0u ? 1.0f : 2.0f);
        dst[i] = widen_bits(as_type<uint>(f));
    }
}
