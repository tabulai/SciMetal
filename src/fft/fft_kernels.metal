// FFT kernels: the vendored MLX kernels (third_party/mlx-kernels, MIT, byte
// identical to MLX 0.31.1) plus metal-sci's own passes built on their radix
// codelets.
//
// MLX's four-step kernels stop at two passes and power-of-two sizes. The
// strided pass below is the general building block they lack: it transforms
// one factor of a transform length along any axis, so a plan can chain as many
// passes as the size needs (src/fft/plan.cpp chooses them).

// The vendored sources are not ours to tidy: keep their warnings out of the build.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wextra"
#pragma clang diagnostic ignored "-Wc++17-extensions"
#pragma clang diagnostic ignored "-Wc++20-extensions"
#include "mlx/backend/metal/kernels/fft.metal"
#pragma clang diagnostic pop

#include "fft_kernels.h"

// exp(-2 pi i k / p) for 0 <= k < p < 2^31, given inverse_p = 1 / float(p).
//
// The angle is reduced to [0, pi/4] with exact integer arithmetic before any
// rounding, and cos and sin are evaluated there by their Taylor polynomials
// (truncation below 3e-8). The result is within about 2e-7 of the exact root
// of unity for every k, where forming -2 pi k / p directly (MLX's get_twiddle)
// loses up to ~1e-6 near k = p, and it does not depend on the accuracy of the
// device's fast trigonometric functions.
METAL_FUNC float2 ms_fft_twiddle(uint k, uint p, float inverse_p) {
  // octant = floor(8k / p). The float estimate is within one of it; the
  // remainder, computed exactly, corrects it without an integer division.
  uint octant = uint(8.0f * float(k) * inverse_p);
  uint rem;
  if (p <= (1u << 28)) {
    // 8k < 2^31: the difference, in (-p, 2p), is exact in 32 bits.
    int difference = int(8u * k - octant * p);
    if (difference < 0) {
      octant -= 1u;
      difference += int(p);
    } else if (difference >= int(p)) {
      octant += 1u;
      difference -= int(p);
    }
    rem = uint(difference);
  } else {
    long difference = long(8ul * ulong(k)) - long(ulong(octant) * ulong(p));
    if (difference < 0) {
      octant -= 1u;
      difference += long(p);
    } else if (difference >= long(p)) {
      octant += 1u;
      difference -= long(p);
    }
    rem = uint(difference);
  }
  // Within its quarter turn the angle is (pi/4) ((octant & 1) + rem / p). In
  // the second half, cos and sin are the sin and cos of the angle measured
  // back from the quarter's end, which is again at most pi/4.
  const bool mirror = (octant & 1u) != 0u;
  const float x = M_PI_4_F * (float(mirror ? p - rem : rem) * inverse_p);
  const float x2 = x * x;
  float c = 1.0f + x2 * (-0.5f + x2 * (4.16666667e-2f + x2 * (-1.38888889e-3f + x2 * 2.48015873e-5f)));
  float s = x * (1.0f + x2 * (-1.66666667e-1f + x2 * (8.33333333e-3f + x2 * (-1.98412698e-4f + x2 * 2.75573192e-6f))));
  if (mirror) {
    const float t = c;
    c = s;
    s = t;
  }
  // Advance by the whole quarter turns, then conjugate for the -i sign.
  switch (octant >> 1) {
    case 0:
      return float2(c, -s);
    case 1:
      return float2(-s, -c);
    case 2:
      return float2(-c, s);
    default:
      return float2(s, c);
  }
}

template <int tg_mem_size>
[[kernel]] void ms_fft_strided(
    const device float2* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    constant MSFFTStridedParams& p [[buffer(2)]],
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int n = p.n;
  const int stride = p.stride;
  const int width = grid.y; // transforms per threadgroup
  const int m = grid.z; // threads per transform

  threadgroup float2 shared[tg_mem_size];

  // Device access is laid out like MLX's strided reads: thread t of the group
  // handles `elems_per_thread_` consecutive points of column t % width, so
  // neighbouring threads touch neighbouring addresses. Unlike MLX's, the
  // column count need not be a multiple of the width: columns past the end
  // of a block are loaded as zeros and never stored.
  const int groups_per_block = (stride + width - 1) / width;
  const uint gpb = uint(groups_per_block);
  int block;
  int group_in_block;
  if ((gpb & (gpb - 1u)) == 0u) {
    const uint gpb_shift = metal::ctz(gpb);
    block = int(uint(elem.x) >> gpb_shift);
    group_in_block = int(uint(elem.x) & (gpb - 1u));
  } else {
    block = int(uint(elem.x) / gpb);
    group_in_block = int(uint(elem.x) - uint(block) * gpb);
  }
  const int tg_idx = elem.y * m + elem.z;
  const uint uwidth = uint(width);
  int lane;
  int first;
  if ((uwidth & (uwidth - 1u)) == 0u) {
    const uint w_shift = metal::ctz(uwidth);
    lane = int(uint(tg_idx) & (uwidth - 1u));
    first = int(uint(tg_idx) >> w_shift) * elems_per_thread_;
  } else {
    lane = tg_idx % width;
    first = (tg_idx / width) * elems_per_thread_;
  }
  const int column = group_in_block * width + lane;
  const bool live = block < p.blocks && column < stride;
  const uint source = uint(block) * uint(n) * uint(stride) + uint(column);
  const int slot = lane * n + first;

  for (int e = 0; e < elems_per_thread_; e++) {
    float2 value = 0.0f;
    if (live) {
      value = in[source + uint(first + e) * uint(stride)];
      if (p.flags & MS_FFT_CONJ_LOAD) {
        value.y = -value.y;
      }
    }
    shared[slot + e] = value;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int stage = 1;
  perform_fft(int(elem.z), &stage, m, n, &shared[elem.y * n]);

  if (!live) {
    return;
  }

  const uint tail = uint(p.tail);
  uint remaining;
  uint r;
  if ((tail & (tail - 1u)) == 0u) {
    const uint tail_shift = metal::ctz(tail);
    remaining = uint(stride) >> tail_shift;
    r = uint(column) >> tail_shift;
  } else {
    remaining = uint(stride) / tail; // extent of the factors still to transform
    r = uint(column) / tail;
  }
  const uint period = uint(n) * remaining;
  const float inverse_period = 1.0f / float(period);
  const uint outer = uint(p.outer);
  uint lead;
  uint done;
  if ((outer & (outer - 1u)) == 0u) {
    const uint outer_shift = metal::ctz(outer);
    lead = uint(block) >> outer_shift;
    done = uint(block) & (outer - 1u);
  } else {
    lead = uint(block) / outer;
    done = uint(block) - lead * outer;
  }
  const bool has_scale = (p.scale != 1.0f);
  for (int e = 0; e < elems_per_thread_; e++) {
    const uint k = uint(first + e);
    float2 value = shared[slot + e];
    if (remaining > 1u) {
      value = complex_mul(value, ms_fft_twiddle(k * r, period, inverse_period));
    }
    if (p.flags & MS_FFT_CONJ_STORE) {
      value.y = -value.y;
    }
    const uint target = ((lead * uint(n) + k) * outer + done) * uint(stride) + uint(column);
    out[target] = has_scale ? value * p.scale : value;
  }
}

#define instantiate_ms_fft_strided(tg_mem_size) \
  instantiate_kernel("ms_fft_strided_mem_" #tg_mem_size, ms_fft_strided, tg_mem_size)

instantiate_ms_fft_strided(256)
instantiate_ms_fft_strided(512)
instantiate_ms_fft_strided(1024)
instantiate_ms_fft_strided(2048)
instantiate_ms_fft_strided(4096)

// Normalisation that no transform pass could absorb: data[i] *= scale.
kernel void ms_fft_scale(constant MSFFTScaleParams& p [[buffer(0)]],
                         device float* data [[buffer(1)]],
                         uint pos [[thread_position_in_grid]]) {
  const ulong begin = ulong(pos) * MS_FFT_SCALE_CHUNK;
  if (begin >= p.count) {
    return;
  }
  const ulong end = min(begin + MS_FFT_SCALE_CHUNK, p.count);
  for (ulong i = begin; i < end; i++) {
    data[i] *= p.scale;
  }
}
