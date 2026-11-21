#ifndef MS_NDIMAGE_H
#define MS_NDIMAGE_H
#include "ms_array.h"
#ifdef __cplusplus
extern "C" {
#endif

/* scipy.ndimage separable filters, version 1.
 *
 * One primitive: a sequence of 1-D passes over an N-D array, each along one
 * axis. It carries correlate1d, convolve1d, gaussian_filter1d, gaussian_filter
 * (any order), uniform_filter1d, uniform_filter, prewitt and sobel; callers
 * build the weights. Arrays follow ms_array.h: C-contiguous real input
 * (float32, float64, bool, signed/unsigned 8..64-bit integer), an owned result
 * of the requested dtype and shape. */
#define MS_NDIMAGE_ABI_VERSION 1u
#define MS_NDIMAGE_MAX_PASSES 32u
/* Largest window (weights or uniform length) of one pass. */
#define MS_NDIMAGE_MAX_WINDOW ((uint64_t)1 << 30)

/* How a line a b c d continues past its ends; the codes are SciPy's
 * NI_ExtendMode. SciPy's 'grid-mirror' is REFLECT; for these filters its
 * 'grid-wrap' is WRAP and 'grid-constant' is CONSTANT. */
typedef enum ms_ndimage_mode {
    MS_NDIMAGE_NEAREST = 0,  /* a a a | a b c d | d d d */
    MS_NDIMAGE_WRAP = 1,     /* b c d | a b c d | a b c */
    MS_NDIMAGE_REFLECT = 2,  /* c b a | a b c d | d c b */
    MS_NDIMAGE_MIRROR = 3,   /* d c b | a b c d | c b a; a line of one element repeats it */
    MS_NDIMAGE_CONSTANT = 4  /* k k k | a b c d | k k k, k = cval */
} ms_ndimage_mode;

typedef enum ms_ndimage_pass_kind {
    /* out[i] = sum over j of weights[j] * e[i + j - size/2 - origin], where e
     * is the line extended by `mode` (scipy.ndimage.correlate1d). */
    MS_NDIMAGE_CORRELATE = 1,
    /* out[i] = mean of e[i - size/2 - origin .. + size - 1]
     * (scipy.ndimage.uniform_filter1d); weights must be NULL. */
    MS_NDIMAGE_UNIFORM = 2
} ms_ndimage_pass_kind;

/* One 1-D pass. Initialize struct_size and abi_version (MS_NDIMAGE_ABI_VERSION)
 * and zero the reserved fields. */
typedef struct ms_ndimage_pass {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t kind;          /* ms_ndimage_pass_kind */
    uint32_t axis;          /* 0 .. ndim-1 */
    uint32_t mode;          /* ms_ndimage_mode */
    uint32_t reserved0;
    int64_t origin;         /* -(size/2) <= origin <= (size-1)/2; positive shifts the window left */
    uint64_t size;          /* weights, or uniform window length: 1 .. MS_NDIMAGE_MAX_WINDOW */
    const double* weights;  /* `size` weights in correlation order; NULL for a uniform pass */
    double cval;            /* fill value of MS_NDIMAGE_CONSTANT */
    uint64_t reserved[2];
} ms_ndimage_pass;

/* out = pass[count-1](... pass[0](input)), 1 <= count <= MS_NDIMAGE_MAX_PASSES.
 * Intermediate results are stored in the input's dtype (or out_dtype for
 * separable_filter_to), as SciPy stores them in its output array.
 *
 * CPU engine: arithmetic in double precision whatever the dtype (results are
 * truncated/rounded to the pass output dtype once per pass), with SciPy's
 * operation order and NI_LineBufferToArray integer/bool saturation rules;
 * uniform passes use SciPy's running sum. MS_PRECISION_STRICT with
 * MS_EXECUTION_AUTO always runs here, float32 and integer/bool arrays included.
 * GPU engine: single-precision arithmetic in one command buffer; float64
 * results are widened exactly and integer/bool results are cast on output.
 * Integer inputs with |x| >= 2^24 are not exact in float32, so AUTO re-routes
 * them to the CPU and forced GPU returns MS_ENOTSUP. With u = 2^-24 and
 * S_i = sum|weights_i| (1 for a uniform pass), each pass turns a bound m on
 * the magnitude of its input and a bound e on its error (m = max|input|,
 * e = u * m to begin with) into
 *     e <- S_i * e + (size_i + 2) * u * S_i * m,    m <- S_i * m,
 * after raising m to |cval| and e to u * |cval| if its boundary is constant;
 * the last e bounds the absolute error of the float32 intermediate result. For
 * unit gains that is u * max|input, cval| * (1 + sum_i(size_i + 2)). Results
 * below the smallest normal float32, 1.2e-38, may be flushed to zero. A uniform
 * pass is a direct sum over its window, so a non-finite value affects only the
 * windows that contain it, where the CPU engine's (and SciPy's) running sum
 * carries it to the end of the line.
 *
 * MS_EINVAL for malformed arguments, MS_ENOTSUP for complex arrays and for a
 * forced GPU route that cannot run: see ms_route_options, and the GPU engine
 * also declines arrays of 2^31 or more elements, axes or windows longer than
 * 2^29, integer inputs with |x| >= 2^24, and calls of more than 2^38
 * element-taps, which MS_EXECUTION_AUTO sends to the CPU engine. `info` may be NULL. */
ms_status ms_ndimage_separable_filter(ms_context*, const ms_array* input, const ms_ndimage_pass* passes,
                                      uint32_t count, const ms_route_options* options,
                                      ms_array_result* out, ms_route_info* info);

/* Same as ms_ndimage_separable_filter, except `out_dtype` may be 0 (same as
 * input->dtype) or any real ms_dtype. Pass 0 converts input->dtype to out_dtype
 * (with CPU accumulation still in double), and passes 1..count-1 read and write
 * out_dtype, matching SciPy's multi-pass intermediate dtype. */
ms_status ms_ndimage_separable_filter_to(ms_context*, const ms_array* input, const ms_ndimage_pass* passes,
                                         uint32_t count, uint32_t out_dtype,
                                         const ms_route_options* options,
                                         ms_array_result* out, ms_route_info* info);

/* Dense N-D correlation options. Initialize struct_size and abi_version
 * (MS_NDIMAGE_ABI_VERSION) and zero the reserved fields.
 *
 * out_dtype may be 0 (same dtype as input) or any real ms_dtype.
 * origins[d] must satisfy -(wshape[d] / 2) <= origins[d] <= (wshape[d] - 1) / 2. */
typedef struct ms_ndimage_dense_options {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t mode;                /* ms_ndimage_mode */
    uint32_t out_dtype;           /* 0 (same as input) or any real ms_dtype */
    uint32_t wshape[MS_MAX_NDIM]; /* weights shape along axes 0..ndim-1 */
    int64_t origins[MS_MAX_NDIM]; /* correlation origin along each axis */
    const double* weights;        /* C-contiguous product(wshape) weights */
    double cval;                  /* fill value of MS_NDIMAGE_CONSTANT */
    uint64_t reserved[2];
} ms_ndimage_dense_options;

/* Multidimensional correlation matching scipy.ndimage.correlate.
 *
 * CPU engine: iterates non-zero taps (|w| > DBL_EPSILON) in C row-major order,
 * accumulating in double with FMA, bit-identical to SciPy 1.18.1 arm64.
 * GPU engine: single-precision arithmetic in one command buffer; in
 * MS_PRECISION_FAST / MS_PRECISION_REFINED, rank-1 separable kernels with
 * ndim >= 2 and >= 2 non-trivial axes may run as separable 1-D passes with
 * adjusted constant fill cval_k = cval * prod_{m < k} sum(v_m).
 * Error bound (dense GPU): with u = 2^-24, S = sum|weights|, K = nonzero taps,
 * m = max(|input|, |cval| if constant), e_0 = u * m, result error <=
 * S * e_0 + (K + 3) * u * S * m. */
ms_status ms_ndimage_dense_filter(ms_context*, const ms_array* input,
                                  const ms_ndimage_dense_options* dense_options,
                                  const ms_route_options* options,
                                  ms_array_result* out, ms_route_info* info);

typedef enum ms_ndimage_combine_op {
    MS_NDIMAGE_COMBINE_SUM = 1,         /* sum_{k=0}^{K-1} chain_k(input) */
    MS_NDIMAGE_COMBINE_SQRT_SUM_SQ = 2  /* sqrt(sum_{k=0}^{K-1} chain_k(input)^2) */
} ms_ndimage_combine_op;

/* One chain of 0..MS_NDIMAGE_MAX_PASSES 1-D passes inside a composite filter.
 * pass_count == 0 represents the identity chain (passes must be NULL). */
typedef struct ms_ndimage_chain {
    uint32_t struct_size;
    uint32_t abi_version;          /* MS_NDIMAGE_ABI_VERSION */
    uint32_t pass_count;           /* 0 .. MS_NDIMAGE_MAX_PASSES */
    uint32_t reserved0;
    const ms_ndimage_pass* passes; /* NULL iff pass_count == 0 */
    uint64_t reserved[2];
} ms_ndimage_chain;

/* Evaluates 1 <= chain_count <= MS_MAX_NDIM chains on `input` and combines
 * them via `combine_op` (laplace, gaussian_laplace, gaussian_gradient_magnitude,
 * generic_laplace, generic_gradient_magnitude). out_dtype may be 0 (same as
 * input) or any real ms_dtype.
 *
 * CPU engine: runs each chain in double precision, fusing combination and sqrt
 * into the final pass store loop (bit-identical to SciPy).
 * GPU engine: single precision, all chains + combine encoded into one command
 * buffer. */
ms_status ms_ndimage_composite_filter(ms_context*, const ms_array* input,
                                      const ms_ndimage_chain* chains, uint32_t chain_count,
                                      uint32_t combine_op, uint32_t out_dtype,
                                      const ms_route_options* options,
                                      ms_array_result* out, ms_route_info* info);

#ifdef __cplusplus
}
#endif
#endif
