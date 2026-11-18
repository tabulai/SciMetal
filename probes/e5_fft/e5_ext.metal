// metal-sci probe E5 -- additions layered on top of the vendored MLX FFT kernels.
//
// This file is NOT part of MLX. It includes the vendored (unmodified) MLX
// headers and adds one kernel that MLX does not have: the transposing middle
// pass of a three-pass ("six-step"-style) FFT, used for power-of-two n > 2^20
// where MLX 0.31.1's own two-pass four-step path stops working.
//
// Decomposition N = M1 * M2 * N2, array viewed as [a][b][c] (row-major):
//   pass A (vendored four_step_fft step 0, n1=M1, n2=M2*N2):
//       FFT over a (stride M2*N2), twiddle w_N^(ka*(b*N2+c)), in-place layout
//   pass B (this kernel, n1=M2, n2=N2, outer_n=M1):
//       FFT over b (stride N2) inside every [ka] block, twiddle
//       w_(M2*N2)^(kb*c), written TRANSPOSED to [kb][ka][c]
//   pass C (vendored four_step_fft step 1, n1=M1*M2, n2=N2):
//       FFT over c (contiguous rows), strided write -> X[kc*M1*M2 + kb*M1 + ka]
//
// All device accesses stay coalesced across grid.y exactly like MLX's passes.

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/fft.h"

template <int tg_mem_size>
[[kernel]] void ms_mid_fft(
    const device float2* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    constant const int& n1,
    constant const int& n2,
    constant const int& batch_size,
    constant const int& outer_n,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  int n = n1;
  int stride = n2;
  int block = n1 * n2;

  int m = grid.z;
  int fft_idx = elem.z;

  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2* buf = &shared_in[elem.y * n];

  // The middle pass never conjugates: the ifft conj/scale is applied by the
  // first pass's load and the last pass's write (same convention as MLX).
  using read_writer_t = ReadWriter<float2, float2, 0, false>;
  read_writer_t read_writer = read_writer_t(
      in,
      &shared_in[0],
      out,
      n,
      batch_size,
      elems_per_thread_,
      elem,
      grid,
      false);

  if (read_writer.out_of_bounds()) {
    return;
  };
  read_writer.load_strided(stride, block);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;
  perform_fft(fft_idx, &p, m, n, buf);

  // Transposed, twiddled write: [blk = row*M1 + ka][kb][c] -> [row][kb][ka][c]
  for (int e = 0; e < elems_per_thread_; e++) {
    float2 output = shared_in[read_writer.strided_shared_idx + e];
    int d = read_writer.strided_device_idx + e * stride;
    int blk = d / block;
    int rem = d - blk * block;
    int kb = rem / stride;
    int c = rem - kb * stride;
    int row = blk / outer_n;
    int ka = blk - row * outer_n;
    float2 twiddle = get_twiddle(kb * c, block);
    int o = (row * n1 + kb) * outer_n + ka;
    out[o * stride + c] = complex_mul(output, twiddle);
  }
}

#define instantiate_ms_mid(tg_mem_size) \
  instantiate_kernel("ms_mid_mem_" #tg_mem_size, ms_mid_fft, tg_mem_size)

instantiate_ms_mid(256)
instantiate_ms_mid(512)
instantiate_ms_mid(1024)
instantiate_ms_mid(2048)
instantiate_ms_mid(4096)
