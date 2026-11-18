// E2 probe kernels: trivial read/write kernels used to prove that a no-copy
// wrapped buffer is really the same memory on both sides.
#include <metal_stdlib>
using namespace metal;

struct Params {
    ulong n;      // number of elements
    uint  mul;
    uint  add;
    uint  width;  // 2-D grid width (elements per grid row)
    uint  pad;
};

static inline ulong lin(uint2 gid, uint width) { return ulong(gid.y) * ulong(width) + ulong(gid.x); }

// In place: buf[i] = buf[i] * mul + add   (reads AND writes the wrapped memory)
kernel void muladd_u32(device uint* buf [[buffer(0)]],
                       constant Params& p [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    ulong i = lin(gid, p.width);
    if (i >= p.n) return;
    buf[i] = buf[i] * p.mul + p.add;
}

// Write only: out[i] = uint(i) * mul + add
kernel void fill_u32(device uint* out [[buffer(0)]],
                     constant Params& p [[buffer(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
    ulong i = lin(gid, p.width);
    if (i >= p.n) return;
    out[i] = uint(i) * p.mul + p.add;
}

// Read src (read-only binding), write dst: dst[i] = src[i] * mul + add
kernel void copy_muladd_u32(const device uint* src [[buffer(0)]],
                            device uint* dst [[buffer(2)]],
                            constant Params& p [[buffer(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
    ulong i = lin(gid, p.width);
    if (i >= p.n) return;
    dst[i] = src[i] * p.mul + p.add;
}

// In place 64-bit increment; used as the "touch every page" first-use kernel.
kernel void inc_u64(device ulong* buf [[buffer(0)]],
                    constant Params& p [[buffer(1)]],
                    uint2 gid [[thread_position_in_grid]]) {
    ulong i = lin(gid, p.width);
    if (i >= p.n) return;
    buf[i] = buf[i] + 1ul;
}
