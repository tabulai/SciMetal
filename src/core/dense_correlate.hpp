#pragma once
// Shared dense N-D correlation primitive (CPU + Metal GPU).
// Used by both `ndimage` (correlate / convolve) and `signal` (direct N-D / 2-D).
#include "ms_array.h"
#include "core/array.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ms::detail::gpu {
class Batch;
}

namespace ms::detail::dense_correlate {

enum class AccumMode : uint32_t {
    Double = 0,     // SciPy ndimage (_NI_Correlate) & signal strict CPU
    InputDtype = 1, // float32 / complex64 accumulates in float32
};

// Boundary mode codes matching ms_ndimage_mode.
enum Mode : uint32_t {
    Nearest = 0,
    Wrap = 1,
    Reflect = 2,
    Mirror = 3,
    Constant = 4,
};

struct StencilSpec {
    array::Shape in_shape;
    array::Shape out_shape;          // == in_shape for ndimage; valid/same/full for signal
    uint32_t in_dtype = 0;           // MS_FLOAT32, MS_FLOAT64, MS_COMPLEX64, MS_COMPLEX128
    uint32_t out_dtype = 0;          // MS_FLOAT32, MS_FLOAT64, MS_COMPLEX64, MS_COMPLEX128
    uint32_t mode[MS_MAX_NDIM] = {}; // per-axis Mode
    double cval_real = 0.0;
    double cval_imag = 0.0;
    AccumMode cpu_accum = AccumMode::Double;
    bool prune_zeros = true;         // true for ndimage (|w| > DBL_EPSILON), false for signal
    bool omit_oob = false;           // true for 1-D numpy-style signal convolve/correlate (skip OOB instead of cval * w)
    uint32_t wshape[MS_MAX_NDIM] = {};
    int64_t start[MS_MAX_NDIM] = {}; // input coord for tap 0 at out coord 0
    uint32_t weights_dtype = MS_FLOAT64; // MS_FLOAT32, MS_FLOAT64, MS_COMPLEX64, MS_COMPLEX128
    const void* weights = nullptr;   // C row-major product(wshape) weights

    void set_mode(uint32_t single_mode) noexcept {
        for (uint32_t d = 0; d < MS_MAX_NDIM; ++d) mode[d] = single_mode;
    }
};

struct Tap {
    int64_t coord[MS_MAX_NDIM] = {}; // = w_coord[ax] + start[ax]
    int64_t linear_offset = 0;       // sum_ax coord[ax] * in_stride[ax]
    double weight_re_f64 = 0.0;
    double weight_im_f64 = 0.0;
    float weight_re_f32 = 0.0f;
    float weight_im_f32 = 0.0f;
};

struct TapTable {
    std::vector<Tap> taps;
    int64_t interior_lo[MS_MAX_NDIM] = {}; // min out coord where all taps are in [0, in_dim)
    int64_t interior_hi[MS_MAX_NDIM] = {}; // max+1 out coord where all taps are in [0, in_dim)
    bool has_interior = false;
    bool all_interior = false;
    bool is_complex_weights = false;
    bool weights_overflow_f32 = false;
};

TapTable build_taps(const StencilSpec& spec);

// CPU engine: threaded across P-cores, vectorized across innermost axis.
BufferPtr run_cpu(ms_context* ctx, const StencilSpec& spec, const TapTable& taps, const void* input);

// GPU engine support & execution.
bool gpu_supports(const StencilSpec& spec, const TapTable& taps) noexcept;

// Encodes dense correlation from `single_input` into `output_buf` inside caller's `batch`.
// Returns the device tap-table BufferPtr that the caller must keep alive until `batch.commit_and_wait()`.
BufferPtr encode_gpu(ms_context* ctx, gpu::Batch& batch, const StencilSpec& spec,
                     const TapTable& taps, const ms_buffer* single_input, const ms_buffer* output_buf);

// Full standalone GPU run with A5 overflow guard rails.
BufferPtr run_gpu(ms_context* ctx, const StencilSpec& spec, const TapTable& taps,
                  const ms_array& input, bool allow_cpu_fallback, std::string& overflow_reason);

} // namespace ms::detail::dense_correlate
