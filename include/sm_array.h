#ifndef SM_ARRAY_H
#define SM_ARRAY_H
#include "sm_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Array operations ABI, version 3. Shared by every numerical family
 * (sm_ndimage.h, sm_fft.h, ...).
 *
 * Inputs are C-contiguous host arrays the caller keeps alive and unmodified
 * until the synchronous call returns. Results are owned buffers holding a
 * C-contiguous array; release them with sm_buffer_release. Released result
 * storage returns to the context's buffer cache, so repeated calls reuse it.
 * Functions leave *out and *info unchanged on failure. */
#define SM_ARRAY_ABI_VERSION 3u
#define SM_MAX_NDIM 8u

typedef enum sm_dtype {
    SM_FLOAT32 = 1, SM_FLOAT64 = 2, SM_COMPLEX64 = 3, SM_COMPLEX128 = 4,
    SM_BOOL = 5,
    SM_INT8 = 6, SM_INT16 = 7, SM_INT32 = 8, SM_INT64 = 9,
    SM_UINT8 = 10, SM_UINT16 = 11, SM_UINT32 = 12, SM_UINT64 = 13
} sm_dtype;

/* fast:    double-precision inputs may be computed in single precision (on the
 *          GPU) and widened on output: about 6-7 significant digits.
 * refined: single-precision work plus a double-precision correction where one
 *          exists; otherwise as fast.
 * strict:  arithmetic in the input's own precision; double stays on the CPU. */
typedef enum sm_precision {
    SM_PRECISION_FAST = 1, SM_PRECISION_REFINED = 2, SM_PRECISION_STRICT = 3
} sm_precision;

typedef struct sm_array {
    uint32_t struct_size;
    uint32_t abi_version;
    const void* data;      /* first element; may be NULL only for empty arrays */
    uint32_t dtype;        /* sm_dtype */
    uint32_t ndim;         /* 1..SM_MAX_NDIM */
    uint64_t shape[SM_MAX_NDIM];
} sm_array;

/* execution: SM_EXECUTION_AUTO lets the router choose; SM_EXECUTION_CPU and
 * SM_EXECUTION_GPU force an engine. Forced GPU never falls back: it returns
 * SM_ENOTSUP when the call cannot run there (CPU-only build, CPU-storage
 * context, or strict precision with a double-precision input). */
typedef struct sm_route_options {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t execution;    /* sm_execution */
    uint32_t precision;    /* sm_precision */
    uint64_t reserved[2];
} sm_route_options;

/* How a call ran. Initialize struct_size and abi_version. */
typedef struct sm_route_info {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t engine;         /* SM_EXECUTION_CPU or SM_EXECUTION_GPU */
    uint32_t compute_dtype;  /* sm_dtype the arithmetic ran in */
    uint64_t reserved[2];
    char reason[256];        /* NUL-terminated re-route reason, or "" when none */
} sm_route_info;

typedef struct sm_array_result {
    sm_buffer* buffer;     /* one retained reference on success */
    uint32_t dtype;        /* sm_dtype of the stored elements */
    uint32_t ndim;
    uint64_t shape[SM_MAX_NDIM];
} sm_array_result;

/* What this build of the core can do: a bit set of sm_capability. */
typedef enum sm_capability {
    SM_CAPABILITY_METAL = 1u /* shared Metal storage and the GPU engines are compiled in */
} sm_capability;
uint32_t sm_array_capabilities(void);

/* Copies `input` into a new array of `dtype`, converting among real dtypes
 * (float32, float64, bool, signed/unsigned 8..64-bit integers matching NumPy
 * astype semantics on arm64) or between single and double complex precision.
 * Real and complex types do not convert into each other (SM_EINVAL). Runs on
 * the CPU. */
sm_status sm_array_astype(sm_context*, const sm_array* input, uint32_t dtype, sm_array_result* out);

#ifdef __cplusplus
}
#endif
#endif
