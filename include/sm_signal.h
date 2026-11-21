#ifndef SM_SIGNAL_H
#define SM_SIGNAL_H
#include "sm_array.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Signal convolution and correlation, ABI version 1.
 *
 * Supports N-D convolution and cross-correlation (convolve, correlate,
 * fftconvolve, oaconvolve, convolve2d, correlate2d, choose_conv_method) on
 * C-contiguous real and complex float arrays (SM_FLOAT32, SM_FLOAT64,
 * SM_COMPLEX64, SM_COMPLEX128) of rank 1..SM_MAX_NDIM.
 *
 * Engines and error bounds:
 * - Direct CPU: computes in input precision when both inputs are single
 *   precision (float32/complex64) and in double precision otherwise. Its
 *   dense tap order can round differently from SciPy's convolve2d. The Python
 *   strict policy delegates that function to the installed SciPy.
 * - Direct GPU: computes in single precision (float32/complex64). For finite
 *   inputs and weights, error is bounded by <= u * M * max|in1, fill| * max|in2|
 *   (u = 2^-24, M = kernel elements), with subnormals below ~1.2e-38 flushed
 *   to zero. Double-precision inputs/weights/fillvalues that overflow single
 *   precision or whose gain exceeds single-precision range are re-routed to the
 *   CPU engine under SM_EXECUTION_AUTO (recording the reason in sm_route_info)
 *   or return SM_ENOTSUP under SM_EXECUTION_GPU.
 * - FFT / Overlap-Add GPU: GPU-only (single precision). In one command buffer
 *   per call, pads/chunks inputs, runs forward FFTs, multiplies spectra, runs
 *   inverse FFT, and overlap-adds / crops / widens to the promoted output
 *   dtype. Error is O(u * max|output|) (typically ~1e-6 * max|output| or
 *   ~1e-6 * ||in1||_2 * ||in2||_2); entries much smaller than max|output| can
 *   be dominated by u * max|output|. Double-precision calls under
 *   SM_PRECISION_STRICT, SM_EXECUTION_CPU, builds without Metal, calls below
 *   the GPU crossover, or double-precision inputs whose magnitudes fall outside
 *   [2^-100, 2^96] or whose product peak1 * peak2 * overlap_elems > 2^124
 *   return SM_ENOTSUP so callers fall through to SciPy (threaded via
 *   scipy.fft.set_workers).
 */
#define SM_SIGNAL_ABI_VERSION 1u

/* Bit flags set in sm_route_info::reserved[0] by sm_signal_convolve. */
#define SM_SIGNAL_ROUTE_FFT       1ull
#define SM_SIGNAL_ROUTE_NONFINITE 2ull

typedef enum sm_signal_mode {
    SM_SIGNAL_FULL = 0,
    SM_SIGNAL_SAME = 1,
    SM_SIGNAL_VALID = 2
} sm_signal_mode;

typedef enum sm_signal_method {
    SM_SIGNAL_METHOD_AUTO = 0,
    SM_SIGNAL_METHOD_DIRECT = 1,
    SM_SIGNAL_METHOD_FFT = 2
} sm_signal_method;

typedef enum sm_signal_boundary {
    SM_SIGNAL_BOUNDARY_FILL = 0, /* zero or fillvalue padding (constant) */
    SM_SIGNAL_BOUNDARY_WRAP = 1, /* circular wrap */
    SM_SIGNAL_BOUNDARY_SYMM = 2  /* symmetric reflection (SciPy REFLECT: dcba|abcd|dcba) */
} sm_signal_boundary;

typedef enum sm_signal_same_center {
    SM_SIGNAL_CENTER_ND = 0,      /* N-D convolve/correlate & convolve2d: start = -(k / 2) */
    SM_SIGNAL_CENTER_2D_CORR = 1  /* correlate2d: start = -((k - 1) / 2) */
} sm_signal_same_center;

typedef struct sm_signal_conv_options {
    uint32_t struct_size;
    uint32_t abi_version;      /* SM_SIGNAL_ABI_VERSION */
    uint32_t mode;             /* sm_signal_mode */
    uint32_t method;           /* sm_signal_method */
    uint32_t correlate;        /* 0: convolution, nonzero: cross-correlation */
    uint32_t boundary;         /* sm_signal_boundary (FFT requires FILL with fillvalue = 0) */
    uint32_t center;           /* sm_signal_same_center */
    uint32_t host_threads;     /* SciPy worker count if caller fell through (0 counts as 1) */
    double fillvalue_real;     /* constant boundary fill value (real part) */
    double fillvalue_imag;     /* constant boundary fill value (imag part) */
    uint64_t reserved[2];
} sm_signal_conv_options;

typedef struct sm_signal_fftconv_options {
    uint32_t struct_size;
    uint32_t abi_version;      /* SM_SIGNAL_ABI_VERSION */
    uint32_t mode;             /* sm_signal_mode */
    uint32_t correlate;        /* 0: convolution, nonzero: cross-correlation */
    uint32_t overlap_add;      /* 0: single FFT (fftconvolve), nonzero: overlap-add (oaconvolve) */
    uint32_t axis_count;       /* 0 means all axes (0..ndim-1); >0 lists distinct axes */
    uint32_t axes[SM_MAX_NDIM];
    uint32_t host_threads;     /* 0 counts as 1 */
    uint32_t reserved0;
    uint64_t reserved[2];
} sm_signal_fftconv_options;

sm_status sm_signal_convolve(sm_context* context, const sm_array* in1, const sm_array* in2,
                             const sm_signal_conv_options* options, const sm_route_options* route_options,
                             sm_array_result* out, sm_route_info* info);

sm_status sm_signal_fftconvolve(sm_context* context, const sm_array* in1, const sm_array* in2,
                                const sm_signal_fftconv_options* options, const sm_route_options* route_options,
                                sm_array_result* out, sm_route_info* info);

sm_status sm_signal_choose_conv_method(sm_context* context, uint32_t dtype1, uint32_t ndim1, const uint64_t* shape1,
                                       uint32_t dtype2, uint32_t ndim2, const uint64_t* shape2, uint32_t mode,
                                       uint32_t host_threads, const sm_route_options* route_options,
                                       uint32_t* out_method);

#ifdef __cplusplus
}
#endif
#endif
