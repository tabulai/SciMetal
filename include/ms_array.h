#ifndef MS_ARRAY_H
#define MS_ARRAY_H
#include "ms_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Array operations ABI, version 3. Shared by every numerical family
 * (ms_ndimage.h, ms_fft.h, ...).
 *
 * Inputs are C-contiguous host arrays the caller keeps alive and unmodified
 * until the synchronous call returns. Results are owned buffers holding a
 * C-contiguous array; release them with ms_buffer_release. Released result
 * storage returns to the context's buffer cache, so repeated calls reuse it.
 * Functions leave *out and *info unchanged on failure. */
#define MS_ARRAY_ABI_VERSION 3u
#define MS_MAX_NDIM 8u

typedef enum ms_dtype {
    MS_FLOAT32 = 1, MS_FLOAT64 = 2, MS_COMPLEX64 = 3, MS_COMPLEX128 = 4,
    MS_BOOL = 5,
    MS_INT8 = 6, MS_INT16 = 7, MS_INT32 = 8, MS_INT64 = 9,
    MS_UINT8 = 10, MS_UINT16 = 11, MS_UINT32 = 12, MS_UINT64 = 13
} ms_dtype;

/* fast:    double-precision inputs may be computed in single precision (on the
 *          GPU) and widened on output: about 6-7 significant digits.
 * refined: single-precision work plus a double-precision correction where one
 *          exists; otherwise as fast.
 * strict:  arithmetic in the input's own precision; double stays on the CPU. */
typedef enum ms_precision {
    MS_PRECISION_FAST = 1, MS_PRECISION_REFINED = 2, MS_PRECISION_STRICT = 3
} ms_precision;

typedef struct ms_array {
    uint32_t struct_size;
    uint32_t abi_version;
    const void* data;      /* first element; may be NULL only for empty arrays */
    uint32_t dtype;        /* ms_dtype */
    uint32_t ndim;         /* 1..MS_MAX_NDIM */
    uint64_t shape[MS_MAX_NDIM];
} ms_array;

/* execution: MS_EXECUTION_AUTO lets the router choose; MS_EXECUTION_CPU and
 * MS_EXECUTION_GPU force an engine. Forced GPU never falls back: it returns
 * MS_ENOTSUP when the call cannot run there (CPU-only build, CPU-storage
 * context, or strict precision with a double-precision input). */
typedef struct ms_route_options {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t execution;    /* ms_execution */
    uint32_t precision;    /* ms_precision */
    uint64_t reserved[2];
} ms_route_options;

/* How a call ran. Initialize struct_size and abi_version. */
typedef struct ms_route_info {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t engine;         /* MS_EXECUTION_CPU or MS_EXECUTION_GPU */
    uint32_t compute_dtype;  /* ms_dtype the arithmetic ran in */
    uint64_t reserved[2];
    char reason[256];        /* NUL-terminated re-route reason, or "" when none */
} ms_route_info;

typedef struct ms_array_result {
    ms_buffer* buffer;     /* one retained reference on success */
    uint32_t dtype;        /* ms_dtype of the stored elements */
    uint32_t ndim;
    uint64_t shape[MS_MAX_NDIM];
} ms_array_result;

/* What this build of the core can do: a bit set of ms_capability. */
typedef enum ms_capability {
    MS_CAPABILITY_METAL = 1u /* shared Metal storage and the GPU engines are compiled in */
} ms_capability;
uint32_t ms_array_capabilities(void);

/* Copies `input` into a new array of `dtype`, converting among real dtypes
 * (float32, float64, bool, signed/unsigned 8..64-bit integers matching NumPy
 * astype semantics on arm64) or between single and double complex precision.
 * Real and complex types do not convert into each other (MS_EINVAL). Runs on
 * the CPU. */
ms_status ms_array_astype(ms_context*, const ms_array* input, uint32_t dtype, ms_array_result* out);

#ifdef __cplusplus
}
#endif
#endif
