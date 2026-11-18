#ifndef MS_FFT_H
#define MS_FFT_H
#include "ms_array.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Discrete Fourier transforms, ABI version 1.
 *
 * The transforms follow scipy.fft: the forward transform uses exp(-2 pi i jk/n)
 * and is unnormalised under MS_FFT_NORM_BACKWARD.
 *
 * Engines. The GPU engine computes in single precision; a double-precision
 * input is narrowed on the way in and the result widened to the precision
 * SciPy would return, so under MS_PRECISION_FAST and MS_PRECISION_REFINED
 * such results carry single-precision accuracy (about 1e-6 relative to the
 * largest output). There is no native CPU FFT: a call the router would send
 * to the CPU - forced CPU, MS_EXECUTION_AUTO below the crossover, or a
 * double-precision input under MS_PRECISION_STRICT - returns MS_ENOTSUP, as
 * does a transform length the planner has no correct plan for and a batch
 * the kernels cannot index (2^31 points or more) or the context cannot hold.
 * Callers hand those to their own FFT. */
#define MS_FFT_ABI_VERSION 1u

typedef enum ms_fft_kind {
    MS_FFT_FORWARD = 1, /* complex to complex; a real input is taken as complex */
    MS_FFT_INVERSE = 2, /* complex to complex, exp(+2 pi i jk/n) */
    MS_FFT_R2C = 3,     /* real input to the first n/2 + 1 points of its forward transform */
    MS_FFT_C2R = 4      /* n/2 + 1 points of a Hermitian spectrum to the n real points of its inverse */
} ms_fft_kind;

/* Scaling by direction, with N the product of the transform lengths:
 * backward: forward 1, inverse 1/N;  ortho: both 1/sqrt(N);  forward: forward 1/N, inverse 1. */
typedef enum ms_fft_norm {
    MS_FFT_NORM_BACKWARD = 0, MS_FFT_NORM_ORTHO = 1, MS_FFT_NORM_FORWARD = 2
} ms_fft_norm;

typedef struct ms_fft_options {
    uint32_t struct_size;
    uint32_t abi_version;      /* MS_FFT_ABI_VERSION */
    uint32_t kind;             /* ms_fft_kind */
    uint32_t norm;             /* ms_fft_norm */
    /* The axes to transform: axis_count distinct axes of the input, in any
     * order. MS_FFT_R2C and MS_FFT_C2R must list the input's last axis, which
     * is the one with the half spectrum (scipy.fft.rfftn with that axis last). */
    uint32_t axis_count;
    uint32_t axes[MS_MAX_NDIM];
    /* Threads the caller's own FFT would use for this call (0 counts as 1).
     * Only MS_EXECUTION_AUTO reads it: a threaded alternative moves the
     * crossover for batched inputs. */
    uint32_t host_threads;
    /* Length of the transform along the input's last axis: the input is
     * truncated or zero padded to it as scipy.fft does (for MS_FFT_C2R, to
     * n/2 + 1 points; n is the output length). 0 means the input's own length,
     * or 2 (m - 1) for MS_FFT_C2R with m input points. Must be 0 when the last
     * axis is not transformed. */
    uint64_t n;
    uint64_t reserved[2];
} ms_fft_options;

/* Transforms `input` along the listed axes. The result has the input's shape
 * except along the last axis (n, or n/2 + 1 for MS_FFT_R2C) and the dtype
 * SciPy returns: complex of the input's precision, or real for MS_FFT_C2R.
 * MS_FFT_R2C needs a real input (MS_EINVAL otherwise). */
ms_status ms_fft_transform(ms_context*, const ms_array* input, const ms_fft_options*, const ms_route_options*,
                           ms_array_result* out, ms_route_info* info);

/* How ms_fft_transform would handle a call. Initialize struct_size and abi_version. */
typedef struct ms_fft_plan_info {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t status;       /* the ms_status ms_fft_transform would return for this request (MS_OK or MS_ENOTSUP) */
    uint32_t engine;       /* MS_EXECUTION_GPU when it would run */
    uint32_t passes;       /* kernel dispatches, all in one command buffer */
    uint32_t reserved;
    uint64_t device_bytes; /* buffers the run holds at once, the result included */
    char description[256]; /* the plan per axis, or why the call is declined; NUL terminated */
} ms_fft_plan_info;

/* Plans a transform of an array of the given dtype and shape without running
 * it. With prepare != 0 and a plan that would run on the GPU, also builds the
 * plan's pipelines, so the first transform does not pay for them. Returns
 * MS_OK whenever *out was filled, including for requests that would decline. */
ms_status ms_fft_plan(ms_context*, uint32_t dtype, uint32_t ndim, const uint64_t* shape, const ms_fft_options*,
                      const ms_route_options*, uint32_t prepare, ms_fft_plan_info* out);

/* The smallest number of input points MS_EXECUTION_AUTO sends to the GPU for
 * any shape: callers can skip the call for anything smaller. */
uint64_t ms_fft_auto_min_points(void);

#ifdef __cplusplus
}
#endif
#endif
