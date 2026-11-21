// fft_kernels.h - the kernel <-> host contract for metal-sci's own FFT kernels
// (fft_kernels.metal). The vendored MLX kernels take their arguments one by
// one at explicit indices instead; src/fft/plan.cpp states those.
//
// Rules as in src/kernels/ms_kernels.h: fixed-width fields, every struct a
// multiple of 8 bytes, change kernels and host together.
#ifndef MS_FFT_KERNELS_H
#define MS_FFT_KERNELS_H

#include "../kernels/ms_kernels.h"

// The largest FFT a threadgroup holds: 4096 float2 fill the 32 KiB of
// threadgroup memory an Apple GPU provides. Kernels are instantiated for
// 256, 512, 1024, 2048 and 4096 elements.
#define MS_FFT_MIN_GROUP_ELEMENTS 256
#define MS_FFT_MAX_GROUP_ELEMENTS 4096

// ms_fft_strided_mem_<elements>: one pass of a multi-pass transform along an
// axis whose elements are `stride` apart.
//
// The source is viewed as [block][a : n][s : stride], with `blocks` blocks;
// block = lead * outer + K, where K (extent `outer`) indexes the output of the
// passes that already ran. Each trailing position s = r * tail + t carries the
// factors still to be transformed (r, extent stride / tail) and the untouched
// trailing axes (t, extent tail). The pass transforms a, multiplies by the
// inter-pass twiddle exp(-2 pi i k r / (n stride / tail)) and stores
// transposed, as [lead][k : n][K : outer][s : stride]. After the last pass the
// transformed index is in natural order.
//
// Bindings: 0 source, 1 destination (both float2), 2 these parameters.
// Dispatch: threadgroups (blocks * ceil(stride / width), 1, 1) of
// (1, width, n / elems_per_thread) threads; width * n elements of threadgroup
// memory. elems_per_thread (function constant 2) must divide n.
#define MS_FFT_CONJ_LOAD 1u   // conjugate every element read (first pass of an inverse)
#define MS_FFT_CONJ_STORE 2u  // conjugate every element stored (last pass of an inverse)
struct MSFFTStridedParams {
    ms_i32 n;
    ms_i32 stride;
    ms_i32 blocks;
    ms_i32 outer;
    ms_i32 tail;
    ms_u32 flags;
    float scale;   // multiplies every stored element
    ms_u32 reserved;
};

// ms_fft_scale: data[i] *= scale for `count` floats, in place.
// Bindings (1-D convention): 0 parameters, 1 data. Each thread scales
// MS_FFT_SCALE_CHUNK consecutive floats.
#define MS_FFT_SCALE_CHUNK 16u
struct MSFFTScaleParams {
    ms_u64 count;
    float scale;
    ms_u32 reserved;
};

// ms_fft_c2r_hermitian_fix: clears imag(X[0]) and (for even n) imag(X[n/2])
// on each row before a four-step or strided C2R stage in a multi-axis irfftn.
// Bindings (1-D convention): 0 parameters, 1 data (float2).
struct MSFFTC2RHermitianParams {
    ms_u32 rows;
    ms_u32 row_stride; /* float2 elements per row */
    ms_u32 nyquist;    /* n/2 when n is even, 0 when n is odd */
    ms_u32 reserved;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(MSFFTStridedParams) == 32, "kernel parameter layouts changed: update fft_kernels.h and kernels together");
static_assert(sizeof(MSFFTScaleParams) == 16, "kernel parameter layouts changed: update fft_kernels.h and kernels together");
static_assert(sizeof(MSFFTC2RHermitianParams) == 16, "kernel parameter layouts changed: update fft_kernels.h and kernels together");
#endif

#endif // MS_FFT_KERNELS_H
