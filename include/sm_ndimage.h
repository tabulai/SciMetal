#ifndef SM_NDIMAGE_H
#define SM_NDIMAGE_H
#include "sm_array.h"
#ifdef __cplusplus
extern "C" {
#endif

/* scipy.ndimage separable filters, version 1.
 *
 * One primitive: a sequence of 1-D passes over an N-D array, each along one
 * axis. It carries correlate1d, convolve1d, gaussian_filter1d, gaussian_filter
 * (any order), uniform_filter1d, uniform_filter, prewitt and sobel; callers
 * build the weights. Arrays follow sm_array.h: C-contiguous real input
 * (float32, float64, bool, signed/unsigned 8..64-bit integer), an owned result
 * of the requested dtype and shape. */
#define SM_NDIMAGE_ABI_VERSION 1u
#define SM_NDIMAGE_MAX_PASSES 32u
/* Largest window (weights or uniform length) of one pass. */
#define SM_NDIMAGE_MAX_WINDOW ((uint64_t)1 << 30)

/* How a line a b c d continues past its ends; the codes are SciPy's
 * NI_ExtendMode. SciPy's 'grid-mirror' is REFLECT; for these filters its
 * 'grid-wrap' is WRAP and 'grid-constant' is CONSTANT. */
typedef enum sm_ndimage_mode {
    SM_NDIMAGE_NEAREST = 0,  /* a a a | a b c d | d d d */
    SM_NDIMAGE_WRAP = 1,     /* b c d | a b c d | a b c */
    SM_NDIMAGE_REFLECT = 2,  /* c b a | a b c d | d c b */
    SM_NDIMAGE_MIRROR = 3,   /* d c b | a b c d | c b a; a line of one element repeats it */
    SM_NDIMAGE_CONSTANT = 4  /* k k k | a b c d | k k k, k = cval */
} sm_ndimage_mode;

typedef enum sm_ndimage_pass_kind {
    /* out[i] = sum over j of weights[j] * e[i + j - size/2 - origin], where e
     * is the line extended by `mode` (scipy.ndimage.correlate1d). */
    SM_NDIMAGE_CORRELATE = 1,
    /* out[i] = mean of e[i - size/2 - origin .. + size - 1]
     * (scipy.ndimage.uniform_filter1d); weights must be NULL. */
    SM_NDIMAGE_UNIFORM = 2,
    /* out[i] = min of e[i - size/2 - origin .. + size - 1]
     * (scipy.ndimage.minimum_filter1d); weights must be NULL. */
    SM_NDIMAGE_MINIMUM = 3,
    /* out[i] = max of e[i - size/2 - origin .. + size - 1]
     * (scipy.ndimage.maximum_filter1d); weights must be NULL. */
    SM_NDIMAGE_MAXIMUM = 4
} sm_ndimage_pass_kind;

/* One 1-D pass. Initialize struct_size and abi_version (SM_NDIMAGE_ABI_VERSION)
 * and zero the reserved fields. */
typedef struct sm_ndimage_pass {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t kind;          /* sm_ndimage_pass_kind */
    uint32_t axis;          /* 0 .. ndim-1 */
    uint32_t mode;          /* sm_ndimage_mode */
    uint32_t reserved0;
    int64_t origin;         /* -(size/2) <= origin <= (size-1)/2; positive shifts the window left */
    uint64_t size;          /* weights, or uniform window length: 1 .. SM_NDIMAGE_MAX_WINDOW */
    const double* weights;  /* `size` weights in correlation order; NULL for a uniform pass */
    double cval;            /* fill value of SM_NDIMAGE_CONSTANT */
    uint64_t reserved[2];
} sm_ndimage_pass;

/* out = pass[count-1](... pass[0](input)), 1 <= count <= SM_NDIMAGE_MAX_PASSES.
 * Intermediate results are stored in the input's dtype (or out_dtype for
 * separable_filter_to), as SciPy stores them in its output array.
 *
 * CPU engine: arithmetic in double precision whatever the dtype (results are
 * truncated/rounded to the pass output dtype once per pass), with SciPy's
 * operation order and NI_LineBufferToArray integer/bool saturation rules;
 * uniform passes use SciPy's running sum. SM_PRECISION_STRICT with
 * SM_EXECUTION_AUTO always runs here, float32 and integer/bool arrays included.
 * GPU engine: single-precision arithmetic in one command buffer; float64
 * results are widened exactly and integer/bool results are cast on output.
 * Integer inputs with |x| >= 2^24 are not exact in float32, so AUTO re-routes
 * them to the CPU and forced GPU returns SM_ENOTSUP. With u = 2^-24 and
 * S_i = sum|weights_i| (1 for a uniform pass), each pass turns a bound m on
 * the magnitude of its input and a bound e on its error (m = max|input|,
 * e = u * m to begin with) into
 *     e <- S_i * e + (size_i + 2) * u * S_i * m,    m <- S_i * m,
 * after raising m to |cval| and e to u * |cval| if its boundary is constant;
 * the last e bounds the absolute error of the float32 intermediate result. For
 * unit gains that is u * max|input, cval| * (1 + sum_i(size_i + 2)). Results
 * below the smallest normal float32, 2^-126, may be flushed to zero. The GPU
 * declines inputs whose peak finite magnitude is nonzero and below that
 * threshold. AUTO sends these inputs to the CPU. A uniform pass is a direct
 * sum over its window, so a non-finite value affects only the
 * windows that contain it, where the CPU engine's (and SciPy's) running sum
 * carries it to the end of the line.
 *
 * SM_EINVAL for malformed arguments, SM_ENOTSUP for complex arrays and for a
 * forced GPU route that cannot run: see sm_route_options, and the GPU engine
 * also declines arrays of 2^31 or more elements, axes or windows longer than
 * 2^29, integer inputs with |x| >= 2^24, and calls of more than 2^38
 * element-taps, which SM_EXECUTION_AUTO sends to the CPU engine. `info` may be NULL. */
sm_status sm_ndimage_separable_filter(sm_context*, const sm_array* input, const sm_ndimage_pass* passes,
                                      uint32_t count, const sm_route_options* options,
                                      sm_array_result* out, sm_route_info* info);

/* Same as sm_ndimage_separable_filter, except `out_dtype` may be 0 (same as
 * input->dtype) or any real sm_dtype. Pass 0 converts input->dtype to out_dtype
 * (with CPU accumulation still in double), and passes 1..count-1 read and write
 * out_dtype, matching SciPy's multi-pass intermediate dtype. */
sm_status sm_ndimage_separable_filter_to(sm_context*, const sm_array* input, const sm_ndimage_pass* passes,
                                         uint32_t count, uint32_t out_dtype,
                                         const sm_route_options* options,
                                         sm_array_result* out, sm_route_info* info);

/* Dense N-D correlation options. Initialize struct_size and abi_version
 * (SM_NDIMAGE_ABI_VERSION) and zero the reserved fields.
 *
 * out_dtype may be 0 (same dtype as input) or any real sm_dtype.
 * origins[d] must satisfy -(wshape[d] / 2) <= origins[d] <= (wshape[d] - 1) / 2. */
typedef struct sm_ndimage_dense_options {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t mode;                /* sm_ndimage_mode */
    uint32_t out_dtype;           /* 0 (same as input) or any real sm_dtype */
    uint32_t wshape[SM_MAX_NDIM]; /* weights shape along axes 0..ndim-1 */
    int64_t origins[SM_MAX_NDIM]; /* correlation origin along each axis */
    const double* weights;        /* C-contiguous product(wshape) weights */
    double cval;                  /* fill value of SM_NDIMAGE_CONSTANT */
    uint64_t reserved[2];
} sm_ndimage_dense_options;

/* Multidimensional correlation matching scipy.ndimage.correlate.
 *
 * CPU engine: iterates non-zero taps (|w| > DBL_EPSILON) in C row-major order,
 * accumulating in double with FMA, bit-identical to SciPy 1.18.1 arm64.
 * GPU engine: single-precision arithmetic in one command buffer; in
 * SM_PRECISION_FAST / SM_PRECISION_REFINED, rank-1 separable kernels with
 * ndim >= 2 and >= 2 non-trivial axes may run as separable 1-D passes with
 * adjusted constant fill cval_k = cval * prod_{m < k} sum(v_m).
 * Error bound (dense GPU): with u = 2^-24, S = sum|weights|, K = nonzero taps,
 * m = max(|input|, |cval| if constant), e_0 = u * m, result error <=
 * S * e_0 + (K + 3) * u * S * m. */
sm_status sm_ndimage_dense_filter(sm_context*, const sm_array* input,
                                  const sm_ndimage_dense_options* dense_options,
                                  const sm_route_options* options,
                                  sm_array_result* out, sm_route_info* info);

typedef enum sm_ndimage_combine_op {
    SM_NDIMAGE_COMBINE_SUM = 1,           /* sum_{k=0}^{K-1} chain_k(input) */
    SM_NDIMAGE_COMBINE_SQRT_SUM_SQ = 2,   /* sqrt(sum_{k=0}^{K-1} chain_k(input)^2) */
    SM_NDIMAGE_COMBINE_SUB = 3,           /* chain_0(input) - chain_1(input) (K == 2, non-bool) */
    SM_NDIMAGE_COMBINE_MORPH_LAPLACE = 4, /* ((chain_0(input) + chain_1(input)) - input) - input (K == 2, non-bool) */
    SM_NDIMAGE_COMBINE_XOR = 5            /* (chain_0(input) != 0) ^ (chain_1(input) != 0) (K == 2, bool) */
} sm_ndimage_combine_op;

/* One chain of 0..SM_NDIMAGE_MAX_PASSES 1-D passes inside a composite filter.
 * pass_count == 0 represents the identity chain (passes must be NULL). */
typedef struct sm_ndimage_chain {
    uint32_t struct_size;
    uint32_t abi_version;          /* SM_NDIMAGE_ABI_VERSION */
    uint32_t pass_count;           /* 0 .. SM_NDIMAGE_MAX_PASSES */
    uint32_t reserved0;
    const sm_ndimage_pass* passes; /* NULL iff pass_count == 0 */
    uint64_t reserved[2];
} sm_ndimage_chain;

/* Evaluates 1 <= chain_count <= SM_MAX_NDIM chains on `input` and combines
 * them via `combine_op`. SciMetal uses this for laplace, gaussian_laplace,
 * gaussian_gradient_magnitude, and separable morphological_gradient,
 * morphological_laplace, white_tophat, and black_tophat. out_dtype may be 0
 * (same as input) or any real sm_dtype.
 *
 * CPU engine: runs each chain in double precision, fusing combination into
 * the final pass store loop (bit-identical to SciPy).
 * GPU engine: single precision, all chains + combine encoded into one command
 * buffer. */
sm_status sm_ndimage_composite_filter(sm_context*, const sm_array* input,
                                      const sm_ndimage_chain* chains, uint32_t chain_count,
                                      uint32_t combine_op, uint32_t out_dtype,
                                      const sm_route_options* options,
                                      sm_array_result* out, sm_route_info* info);

/* Dense N-D minimum/maximum & grayscale morphology options. Initialize
 * struct_size and abi_version (SM_NDIMAGE_ABI_VERSION) and zero reserved.
 *
 * footprint: C-contiguous product(fshape) bytes (nonzero = active tap). Must
 *   have at least one active tap. NULL means all-active rectangular footprint.
 * structure: NULL for flat min/max/erosion/dilation; otherwise C-contiguous
 *   product(fshape) doubles giving the additive offset per footprint position
 *   (already negated for erosion, matching SciPy's NI_MinOrMaxFilter `ss`).
 * out_dtype: 0 (same as input) or any real sm_dtype.
 * origins[d]: -(fshape[d] / 2) <= origins[d] <= (fshape[d] - 1) / 2. */
typedef struct sm_ndimage_minmax_options {
    uint32_t struct_size;
    uint32_t abi_version;         /* SM_NDIMAGE_ABI_VERSION */
    uint32_t is_max;              /* 0: minimum/erosion, nonzero: maximum/dilation */
    uint32_t mode;                /* sm_ndimage_mode */
    uint32_t out_dtype;           /* 0 (same as input) or any real sm_dtype */
    uint32_t reserved0;
    uint32_t fshape[SM_MAX_NDIM]; /* footprint shape along axes 0..ndim-1 */
    int64_t origins[SM_MAX_NDIM]; /* filter origin along each axis */
    const uint8_t* footprint;     /* NULL (all-true) or C-contiguous product(fshape) bytes */
    const double* structure;      /* NULL (flat) or C-contiguous product(fshape) additive offsets */
    double cval;                  /* fill value of SM_NDIMAGE_CONSTANT */
    uint64_t reserved[2];
} sm_ndimage_minmax_options;

/* Multidimensional non-separable minimum/maximum and grayscale erosion/dilation
 * matching scipy.ndimage's NI_MinOrMaxFilter.
 *
 * CPU engine: computes in input dtype with SciPy's exact tap order, cval cast,
 * structure offset cast, and CASE_FILTER_OUT[_SAFE] store (bit-identical to
 * SciPy 1.18.1 arm64).
 * GPU engine: single precision, one command buffer. Exact for flat footprints
 * on float32, bool, int8/16, uint8/16, and int32/64/uint32/64 with |x| < 2^24;
 * for float64 inputs under SM_PRECISION_FAST/REFINED, comparisons and output
 * values carry float32 rounding (error <= 2^-24 * max|input, cval| for flat;
 * <= 2^-23 * (max|input, cval| + max|structure|) for non-flat). Non-flat
 * integer/bool calls where structure values are non-integral or could wrap the
 * input integer type re-route to CPU under AUTO (or return SM_ENOTSUP under
 * forced GPU). */
sm_status sm_ndimage_minmax_filter(sm_context*, const sm_array* input,
                                   const sm_ndimage_minmax_options* minmax_options,
                                   const sm_route_options* options,
                                   sm_array_result* out, sm_route_info* info);

/* Multidimensional rank / median / percentile filter options. Initialize
 * struct_size and abi_version (SM_NDIMAGE_ABI_VERSION) and zero reserved.
 *
 * rank: 0 <= rank < active_taps (the 0-indexed order statistic).
 * footprint: NULL (all-true rectangular footprint of fshape) or C-contiguous
 *   product(fshape) bytes (nonzero = active tap; at least one active tap).
 * out_dtype: 0 (same as input) or any real sm_dtype.
 * origins[d]: -(fshape[d] / 2) <= origins[d] <= (fshape[d] - 1) / 2. */
typedef struct sm_ndimage_rank_options {
    uint32_t struct_size;
    uint32_t abi_version;         /* SM_NDIMAGE_ABI_VERSION */
    uint32_t rank;                /* 0 <= rank < active_taps */
    uint32_t mode;                /* sm_ndimage_mode */
    uint32_t out_dtype;           /* 0 (same as input) or any real sm_dtype */
    uint32_t reserved0;
    uint32_t fshape[SM_MAX_NDIM]; /* footprint shape along axes 0..ndim-1 */
    int64_t origins[SM_MAX_NDIM]; /* filter origin along each axis */
    const uint8_t* footprint;     /* NULL (all-true) or C-contiguous product(fshape) bytes */
    double cval;                  /* fill value of SM_NDIMAGE_CONSTANT */
    uint64_t reserved[2];
} sm_ndimage_rank_options;

/* Multidimensional rank / median / percentile filter matching scipy.ndimage's
 * NI_RankFilter.
 *
 * CPU engine: threaded across P-cores, exact order statistic in double with
 * SciPy's input-type cval cast and CASE_FILTER_OUT[_SAFE] output conversion
 * (bit-identical to SciPy 1.18.1 arm64 on finite inputs; uses SciPy's exact
 * NI_Select when NaNs are present).
 * GPU engine: single precision, one command buffer. Uses dedicated tiled
 * register-only sorting/selection networks for 2-D 3x3, 5x5, and 7x7 medians,
 * and Forgetful Selection for arbitrary N-D footprints and ranks with
 * min(rank, K - 1 - rank) + 2 <= 64 (covering all footprints up to 127 taps,
 * including 11x11 and 3x3x3). Exact for float32, bool, int8/16, uint8/16, and
 * representable (|x| < 2^24) int32/64/uint32/64; for float64 under FAST/REFINED,
 * values are float32-rounded (absolute error <= 2^-24 * max|input, cval|). */
sm_status sm_ndimage_rank_filter(sm_context*, const sm_array* input,
                                 const sm_ndimage_rank_options* rank_options,
                                 const sm_route_options* options,
                                 sm_array_result* out, sm_route_info* info);

#ifdef __cplusplus
}
#endif
#endif
