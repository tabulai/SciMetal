// GPU engine of the shared dense N-D correlation primitive.
#include "dense_correlate.hpp"
#include "dense_correlate_kernels.h"
#include "runtime/metal_compute.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace sm::detail::dense_correlate {
namespace {

constexpr double float32_overflow_threshold = 0x1.ffffffp127;
// Cap tap-elements so one command buffer does not hold the GPU too long.
constexpr uint64_t gpu_max_tap_elements = uint64_t{1} << 38;

bool overflows_float32(double value) noexcept {
    return std::isfinite(value) && std::fabs(value) >= float32_overflow_threshold;
}

std::string check_parameters_range(const StencilSpec& spec, const TapTable& taps) {
    bool has_constant_boundary = false;
    for (uint32_t d = 0; d < spec.out_shape.ndim; ++d) {
        if (spec.mode[d] == Constant) {
            has_constant_boundary = true;
            break;
        }
    }
    if (has_constant_boundary &&
        (overflows_float32(spec.cval_real) ||
         (array::is_complex(spec.in_dtype) && overflows_float32(spec.cval_imag)))) {
        return "constant cval exceeds single-precision range; re-routed to CPU";
    }
    if (taps.weights_overflow_f32) {
        return "filter weights exceed single-precision range; re-routed to CPU";
    }
    return {};
}

std::string check_gain_range(const StencilSpec& spec, const TapTable& taps, const array::ConvertReport& report) {
    if (report.overflowed) {
        return "float64 input overflowed float32; re-routed to CPU";
    }
    constexpr double u = 0x1p-24;
    double m = std::fabs(static_cast<double>(static_cast<float>(report.max_finite)));
    double e = u * m;
    bool has_constant_boundary = false;
    for (uint32_t d = 0; d < spec.out_shape.ndim; ++d) {
        if (spec.mode[d] == Constant) {
            has_constant_boundary = true;
            break;
        }
    }
    if (has_constant_boundary) {
        if (std::isfinite(spec.cval_real)) {
            const double cr = std::fabs(static_cast<double>(static_cast<float>(spec.cval_real)));
            if (cr > m) m = cr;
            e = std::max(e, u * cr);
        }
        if (array::is_complex(spec.in_dtype) && std::isfinite(spec.cval_imag)) {
            const double ci = std::fabs(static_cast<double>(static_cast<float>(spec.cval_imag)));
            if (ci > m) m = ci;
            e = std::max(e, u * ci);
        }
    }

    double gain = 0.0;
    for (const Tap& t : taps.taps) {
        const double awr = std::fabs(static_cast<double>(t.weight_re_f32));
        if (std::isfinite(awr)) gain += awr;
        if (taps.is_complex_weights) {
            const double awi = std::fabs(static_cast<double>(t.weight_im_f32));
            if (std::isfinite(awi)) gain += awi;
        }
    }
    e = gain * e + static_cast<double>(taps.taps.size() + 3) * u * gain * m;
    m *= gain;
    if (!std::isfinite(m + e) || m + e >= float32_overflow_threshold) {
        return array::is_double(spec.in_dtype)
                   ? "float64 dense filter gain overflowed float32; re-routed to CPU"
                   : "mixed-precision dense filter gain overflowed float32; re-routed to CPU";
    }
    return {};
}

const char* kernel_for(const StencilSpec& spec) {
    const bool in_c = array::is_complex(spec.in_dtype);
    const bool out_d = array::is_double(spec.out_dtype);
    if (!in_c) {
        return out_d ? "sm_dense_correlate_f32_f64" : "sm_dense_correlate_f32_f32";
    }
    return out_d ? "sm_dense_correlate_c64_c128" : "sm_dense_correlate_c64_c64";
}

} // namespace

bool gpu_supports(const StencilSpec& spec, const TapTable& taps) noexcept {
    if (spec.in_shape.ndim < 1 || spec.in_shape.ndim > SM_MAX_NDIM || spec.out_shape.ndim != spec.in_shape.ndim)
        return false;
    if (taps.is_complex_weights && !array::is_complex(spec.in_dtype)) return false;
    if (array::is_complex(spec.in_dtype) != array::is_complex(spec.out_dtype)) return false;
    if ((spec.in_shape.elements == 0 && spec.out_shape.elements > 0) || spec.in_shape.elements > INT32_MAX ||
        spec.out_shape.elements > INT32_MAX)
        return false;
    if (taps.taps.size() > SM_DENSE_GPU_MAX_TAPS) return false;
    for (uint32_t d = 0; d < spec.in_shape.ndim; ++d) {
        if (spec.in_shape.dims[d] > SM_DENSE_GPU_MAX_EXTENT || spec.out_shape.dims[d] > SM_DENSE_GPU_MAX_EXTENT)
            return false;
    }
    for (const Tap& tap : taps.taps) {
        if (tap.linear_offset < INT32_MIN || tap.linear_offset > INT32_MAX) return false;
        for (uint32_t d = 0; d < spec.in_shape.ndim; ++d) {
            if (tap.coord[d] < -static_cast<int64_t>(SM_DENSE_GPU_MAX_EXTENT) ||
                tap.coord[d] > static_cast<int64_t>(SM_DENSE_GPU_MAX_EXTENT))
                return false;
        }
    }
    uint64_t tap_elements = 0;
    if (sm_checked_mul(spec.out_shape.elements, static_cast<uint64_t>(taps.taps.size()), &tap_elements) != SM_OK)
        return false;
    return tap_elements <= gpu_max_tap_elements;
}

BufferPtr encode_gpu(sm_context* ctx, gpu::Batch& batch, const StencilSpec& spec,
                     const TapTable& taps, const sm_buffer* single_input, const sm_buffer* output_buf) {
    if (spec.out_shape.elements == 0) return BufferPtr{};
    if (taps.taps.empty()) {
        batch.fill(output_buf, 0, spec.out_shape.elements * array::dtype_size(spec.out_dtype), 0);
        return BufferPtr{};
    }

    BufferPtr tap_buf = array::allocate(ctx, taps.taps.size() * sizeof(SMDenseTap));
    {
        SMDenseTap* dst = reinterpret_cast<SMDenseTap*>(mutable_data(tap_buf.get()));
        for (size_t i = 0; i < taps.taps.size(); ++i) {
            const Tap& t = taps.taps[i];
            SMDenseTap& dt = dst[i];
            dt = SMDenseTap{};
            for (uint32_t d = 0; d < spec.in_shape.ndim; ++d) dt.coord[d] = static_cast<sm_i32>(t.coord[d]);
            dt.linear_offset = static_cast<sm_i32>(t.linear_offset);
            dt.weight_re = t.weight_re_f32;
            dt.weight_im = t.weight_im_f32;
        }
    }

    const uint32_t ndim = spec.out_shape.ndim;
    SMDenseCorrelateParams params{};
    params.ndim = ndim;
    params.num_taps = static_cast<sm_u32>(taps.taps.size());
    params.out_elements = static_cast<sm_u32>(spec.out_shape.elements);
    params.all_interior = taps.all_interior ? 1u : 0u;
    params.cval_re = static_cast<float>(spec.cval_real);
    params.cval_im = static_cast<float>(spec.cval_imag);

    uint32_t out_s = 1, in_s = 1;
    for (int32_t d = static_cast<int32_t>(ndim) - 1; d >= 0; --d) {
        params.out_shape[d] = static_cast<sm_u32>(spec.out_shape.dims[d]);
        params.in_shape[d] = static_cast<sm_u32>(spec.in_shape.dims[d]);
        params.out_stride[d] = out_s;
        params.in_stride[d] = in_s;
        params.mode[d] = spec.mode[d];
        params.interior_lo[d] = static_cast<sm_i32>(taps.interior_lo[d]);
        params.interior_hi[d] = static_cast<sm_i32>(taps.interior_hi[d]);
        out_s *= static_cast<uint32_t>(spec.out_shape.dims[d]);
        in_s *= static_cast<uint32_t>(spec.in_shape.dims[d]);
    }

    if (!array::is_complex(spec.in_dtype) && ndim == 2) {
        const char* kernel = array::is_double(spec.out_dtype) ? "sm_dense_correlate2d_f32_f64"
                                                              : "sm_dense_correlate2d_f32_f32";
        const uint32_t w = static_cast<uint32_t>(spec.out_shape.dims[1]);
        const uint32_t h = static_cast<uint32_t>(spec.out_shape.dims[0]);
        gpu::Grid grid;
        grid.threads[0] = std::min<uint32_t>(32, std::max<uint32_t>(1, w));
        grid.threads[1] = std::min<uint32_t>(8, std::max<uint32_t>(1, h));
        grid.threads[2] = 1;
        grid.groups[0] = (w + grid.threads[0] - 1) / grid.threads[0];
        grid.groups[1] = (h + grid.threads[1] - 1) / grid.threads[1];
        grid.groups[2] = 1;
        batch.dispatch(kernel, params,
                       {gpu::bind(single_input, ctx), gpu::bind(output_buf, ctx), gpu::bind(tap_buf.get(), ctx)},
                       grid);
        return tap_buf;
    }

    const char* kernel = kernel_for(spec);
    const uint64_t threadgroups = (spec.out_shape.elements + SM_GPU_THREADS - 1) / SM_GPU_THREADS;
    batch.dispatch(kernel, params,
                   {gpu::bind(single_input, ctx), gpu::bind(output_buf, ctx), gpu::bind(tap_buf.get(), ctx)},
                   threadgroups);
    return tap_buf;
}

BufferPtr run_gpu(sm_context* ctx, const StencilSpec& spec, const TapTable& taps,
                  const sm_array& input, bool allow_cpu_fallback, std::string& overflow_reason) {
    overflow_reason = check_parameters_range(spec, taps);
    if (!overflow_reason.empty()) {
        if (!allow_cpu_fallback) {
            throw Error(SM_ENOTSUP, "values exceed single-precision range; forced GPU cannot fall back to CPU");
        }
        return BufferPtr{};
    }

    const bool output_is_double = array::is_double(spec.out_dtype);
    const bool needs_post_cast = array::is_int_or_bool(spec.out_dtype);
    const bool need_report = (spec.in_dtype != SM_FLOAT32 && spec.in_dtype != SM_COMPLEX64);
    const bool need_gain_check =
        need_report || output_is_double || needs_post_cast || spec.cpu_accum == AccumMode::Double;
    array::ConvertReport report;
    BufferPtr single_in = array::load_single(ctx, input, spec.in_shape, need_gain_check ? &report : nullptr,
                                             /*raw_bool=*/true);
    if (need_report && report.inexact_int) {
        single_in.reset();
        if (!allow_cpu_fallback) {
            throw Error(SM_ENOTSUP, "integer input exceeds exact float32 range (|x| >= 2^24); forced GPU cannot fall back to CPU");
        }
        overflow_reason = "integer input exceeds exact float32 range (|x| >= 2^24); re-routed to CPU";
        return BufferPtr{};
    }
    if (need_gain_check) {
        overflow_reason = check_gain_range(spec, taps, report);
        if (!overflow_reason.empty()) {
            single_in.reset();
            if (!allow_cpu_fallback) {
                throw Error(SM_ENOTSUP, "values exceed single-precision range; forced GPU cannot fall back to CPU");
            }
            return BufferPtr{};
        }
    }

    StencilSpec gpu_spec = spec;
    if (needs_post_cast) gpu_spec.out_dtype = SM_FLOAT32;

    uint64_t gpu_out_bytes = 0;
    if (sm_checked_mul(gpu_spec.out_shape.elements, array::dtype_size(gpu_spec.out_dtype), &gpu_out_bytes) != SM_OK)
        throw Error(SM_EOVERFLOW, "dense_correlate: output byte size overflows");
    uint64_t final_out_bytes = gpu_out_bytes;
    if (needs_post_cast &&
        sm_checked_mul(spec.out_shape.elements, array::dtype_size(spec.out_dtype), &final_out_bytes) != SM_OK)
        throw Error(SM_EOVERFLOW, "dense_correlate: output byte size overflows");

    GpuOperationScope operation(ctx, true);
    BufferPtr output = array::allocate(ctx, gpu_out_bytes);
    gpu::Batch batch(ctx);
    BufferPtr tap_buf = encode_gpu(ctx, batch, gpu_spec, taps, single_in.get(), output.get());
    batch.commit_and_wait();
    operation.complete();
    if (needs_post_cast) {
        BufferPtr typed_out = array::allocate(ctx, final_out_bytes);
        array::ndimage_cast_from_float(
            reinterpret_cast<const float*>(read(output.get(), ctx).data),
            spec.out_dtype, mutable_data(typed_out.get()), spec.out_shape.elements, /*dense_safe=*/true);
        return typed_out;
    }
    return output;
}

} // namespace sm::detail::dense_correlate
