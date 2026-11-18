// E8 probe kernel: 1-D correlate along the last axis of a float32 (rows x n)
// array with scipy.ndimage 'reflect' (half-sample symmetric) boundary handling.
//
// This file is deliberately self-contained: no MLX headers, no MLX helper
// macros.  It is compiled OFFLINE into our own metallib and then encoded by
// two different hosts (our standalone Metal runner and an MLX Primitive).
//
//   out[r, i] = sum_{j=0..nw-1} w[j] * in[r, reflect(i + j - shift)]
//
// with shift = nw / 2 + origin (scipy.ndimage.correlate1d convention, origin=0
// in this probe) and reflect() mapping (d c b a | a b c d | d c b a).

#include <metal_stdlib>
using namespace metal;

struct E8CorrelateParams {
  uint rows;  // number of independent rows (product of all leading dims)
  uint n;     // length of the last axis
  uint nw;    // number of weights
  int shift;  // nw / 2 + origin
};

static inline uint e8_reflect(int idx, int n) {
  // Half-sample symmetric extension with period 2n; valid for any idx, so
  // nw > n is handled as well.
  int period = 2 * n;
  int m = idx % period;
  if (m < 0) {
    m += period;
  }
  if (m >= n) {
    m = period - 1 - m;
  }
  return uint(m);
}

kernel void e8_correlate1d_reflect_f32(
    const device float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    constant E8CorrelateParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.n || gid.y >= p.rows) {
    return;
  }
  const ulong base = ulong(gid.y) * ulong(p.n);
  const int n = int(p.n);
  const int start = int(gid.x) - p.shift;
  float acc = 0.0f;
  if (start >= 0 && start + int(p.nw) <= n) {
    // Interior: no boundary handling needed.
    const ulong off = base + ulong(start);
    for (uint j = 0; j < p.nw; ++j) {
      acc += weights[j] * in[off + j];
    }
  } else {
    for (uint j = 0; j < p.nw; ++j) {
      acc += weights[j] * in[base + e8_reflect(start + int(j), n)];
    }
  }
  out[base + gid.x] = acc;
}
