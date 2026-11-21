#ifndef SM_FFT_H
#define SM_FFT_H
#include "sm_array.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Discrete Fourier transforms, ABI version 1.
 *
 * The transforms follow scipy.fft: the forward transform uses exp(-2 pi i jk/n)
 * and is unnormalised under SM_FFT_NORM_BACKWARD.
 *
 * Engines. The GPU engine computes in single precision; a double-precision
 * input is narrowed on the way in and the result widened to the precision
 * SciPy would return, so under SM_PRECISION_FAST and SM_PRECISION_REFINED
 * such results carry single-precision accuracy (about 1e-6 relative to the
 * largest output). There is no native CPU FFT: a call the router would send
 * to the CPU - forced CPU, SM_EXECUTION_AUTO below the crossover, or a
 * double-precision input under SM_PRECISION_STRICT - returns SM_ENOTSUP, as
 * does a transform length the planner has no correct plan for and a batch
 * the kernels cannot index (2^31 points or more) or the context cannot hold.
 * Callers hand those to their own FFT. */
#define SM_FFT_ABI_VERSION 1u

typedef enum sm_fft_kind {
    SM_FFT_FORWARD = 1, /* complex to complex; a real input is taken as complex */
    SM_FFT_INVERSE = 2, /* complex to complex, exp(+2 pi i jk/n) */
    SM_FFT_R2C = 3,     /* real input to the first n/2 + 1 points of its forward transform */
    SM_FFT_C2R = 4      /* n/2 + 1 points of a Hermitian spectrum to the n real points of its inverse */
} sm_fft_kind;

/* Scaling by direction, with N the product of the transform lengths:
 * backward: forward 1, inverse 1/N;  ortho: both 1/sqrt(N);  forward: forward 1/N, inverse 1. */
typedef enum sm_fft_norm {
    SM_FFT_NORM_BACKWARD = 0, SM_FFT_NORM_ORTHO = 1, SM_FFT_NORM_FORWARD = 2
} sm_fft_norm;

typedef struct sm_fft_options {
    uint32_t struct_size;
    uint32_t abi_version;      /* SM_FFT_ABI_VERSION */
    uint32_t kind;             /* sm_fft_kind */
    uint32_t norm;             /* sm_fft_norm */
    /* The axes to transform: axis_count distinct axes of the input, in any
     * order. SM_FFT_R2C and SM_FFT_C2R must list the input's last axis, which
     * is the one with the half spectrum (scipy.fft.rfftn with that axis last). */
    uint32_t axis_count;
    uint32_t axes[SM_MAX_NDIM];
    /* Threads the caller's own FFT would use for this call (0 counts as 1).
     * Only SM_EXECUTION_AUTO reads it: a threaded alternative moves the
     * crossover for batched inputs. */
    uint32_t host_threads;
    /* Length of the transform along the input's last axis: the input is
     * truncated or zero padded to it as scipy.fft does (for SM_FFT_C2R, to
     * n/2 + 1 points; n is the output length). 0 means the input's own length,
     * or 2 (m - 1) for SM_FFT_C2R with m input points. Must be 0 when the last
     * axis is not transformed. */
    uint64_t n;
    uint64_t reserved[2];
} sm_fft_options;

/* Transforms `input` along the listed axes. The result has the input's shape
 * except along the last axis (n, or n/2 + 1 for SM_FFT_R2C) and the dtype
 * SciPy returns: complex of the input's precision, or real for SM_FFT_C2R.
 * SM_FFT_R2C needs a real input (SM_EINVAL otherwise). */
sm_status sm_fft_transform(sm_context*, const sm_array* input, const sm_fft_options*, const sm_route_options*,
                           sm_array_result* out, sm_route_info* info);

/* How sm_fft_transform would handle a call. Initialize struct_size and abi_version. */
typedef struct sm_fft_plan_info {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t status;       /* the sm_status sm_fft_transform would return for this request (SM_OK or SM_ENOTSUP) */
    uint32_t engine;       /* SM_EXECUTION_GPU when it would run */
    uint32_t passes;       /* kernel dispatches, all in one command buffer */
    uint32_t reserved;
    uint64_t device_bytes; /* buffers the run holds at once, the result included */
    char description[256]; /* the plan per axis, or why the call is declined; NUL terminated */
} sm_fft_plan_info;

/* Plans a transform of an array of the given dtype and shape without running
 * it. With prepare != 0 and a plan that would run on the GPU, also builds the
 * plan's pipelines, so the first transform does not pay for them. Returns
 * SM_OK whenever *out was filled, including for requests that would decline. */
sm_status sm_fft_plan(sm_context*, uint32_t dtype, uint32_t ndim, const uint64_t* shape, const sm_fft_options*,
                      const sm_route_options*, uint32_t prepare, sm_fft_plan_info* out);

/* The smallest number of input points SM_EXECUTION_AUTO sends to the GPU for
 * any shape: callers can skip the call for anything smaller. */
uint64_t sm_fft_auto_min_points(void);

#ifdef __cplusplus
}
#endif
#endif
