// Metal GPU engines for dense N-D min/max morphology and rank/median/percentile filters.
// Single precision on GPU, one command buffer per call, direct float64 widening on store.
#include "morphology.hpp"
#include "morphology_kernels.h"
#include "runtime/metal_compute.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace sm::detail::ndimage {
namespace {

constexpr double float32_overflow_threshold = 0x1.ffffffp127;

bool overflows_float32(double value) noexcept {
    return std::isfinite(value) && std::fabs(value) >= float32_overflow_threshold;
}

// Compute effective cval cast to input dtype (matching NI_MinOrMaxFilter & NI_RankFilter),
// returning false if the effective cval cannot be represented cleanly in float32.
bool effective_cval_f32(uint32_t in_dtype, double cval, float& out_f32, double& out_abs, std::string& reason) {
    double eff = cval;
    switch (in_dtype) {
    case SM_FLOAT64:
        if (overflows_float32(cval)) {
            reason = "constant cval exceeds single-precision range; re-routed to CPU";
            return false;
        }
        eff = cval;
        break;
    case SM_FLOAT32:
        eff = static_cast<double>(static_cast<float>(cval));
        break;
    case SM_BOOL:
        eff = array::cast_double_ndimage<uint8_t>(cval, /*dense_safe=*/false) != 0 ? 1.0 : 0.0;
        break;
    case SM_UINT8: eff = static_cast<double>(array::cast_double_ndimage<uint8_t>(cval, false)); break;
    case SM_UINT16: eff = static_cast<double>(array::cast_double_ndimage<uint16_t>(cval, false)); break;
    case SM_UINT32: eff = static_cast<double>(array::cast_double_ndimage<uint32_t>(cval, false)); break;
    case SM_UINT64: eff = static_cast<double>(array::cast_double_ndimage<uint64_t>(cval, false)); break;
    case SM_INT8: eff = static_cast<double>(array::cast_double_ndimage<int8_t>(cval, false)); break;
    case SM_INT16: eff = static_cast<double>(array::cast_double_ndimage<int16_t>(cval, false)); break;
    case SM_INT32: eff = static_cast<double>(array::cast_double_ndimage<int32_t>(cval, false)); break;
    case SM_INT64: eff = static_cast<double>(array::cast_double_ndimage<int64_t>(cval, false)); break;
    default: break;
    }
    if (array::is_integer(in_dtype) && std::fabs(eff) >= 16777216.0) {
        reason = "integer constant cval exceeds exact float32 range (|x| >= 2^24); re-routed to CPU";
        return false;
    }
    out_f32 = static_cast<float>(eff);
    out_abs = std::isfinite(eff) ? std::fabs(eff) : 0.0;
    return true;
}

// Checks whether non-flat integer addition `x + s` stays strictly within `in_dtype` range (no wrap).
bool integer_nonflat_safe(uint32_t in_dtype, double max_in_abs, double max_struct_abs, bool non_integral) noexcept {
    if (non_integral) return false;
    if (in_dtype == SM_BOOL) return false;
    double lo = 0.0, hi = 0.0;
    switch (in_dtype) {
    case SM_UINT8: lo = 0.0; hi = 255.0; break;
    case SM_UINT16: lo = 0.0; hi = 65535.0; break;
    case SM_UINT32: lo = 0.0; hi = 16777215.0; break;
    case SM_UINT64: lo = 0.0; hi = 16777215.0; break;
    case SM_INT8: lo = -128.0; hi = 127.0; break;
    case SM_INT16: lo = -32768.0; hi = 32767.0; break;
    case SM_INT32: lo = -16777215.0; hi = 16777215.0; break;
    case SM_INT64: lo = -16777215.0; hi = 16777215.0; break;
    default: return true;
    }
    // Conservative check: unsigned with negative offset (e.g. erosion) could underflow below 0.
    if (array::is_unsigned_int(in_dtype)) {
        return max_struct_abs == 0.0;
    }
    return (max_in_abs + max_struct_abs) <= std::min(-lo, hi);
}

BufferPtr upload_taps(sm_context* context, const MorphTapTable& table) {
    const uint64_t bytes = table.taps.size() * sizeof(SMNdimageMorphTap);
    BufferPtr buf = array::allocate(context, bytes);
    auto* dst = reinterpret_cast<SMNdimageMorphTap*>(mutable_data(buf.get()));
    for (size_t i = 0; i < table.taps.size(); ++i) {
        SMNdimageMorphTap t{};
        for (uint32_t d = 0; d < SM_MAX_NDIM; ++d) t.coord[d] = static_cast<sm_i32>(table.taps[i].coord[d]);
        t.linear_offset = static_cast<sm_i32>(table.taps[i].linear_offset);
        t.struct_offset = table.taps[i].struct_offset_f32;
        dst[i] = t;
    }
    return buf;
}

gpu::Grid grid_for_nd(const array::Shape& shape) {
    gpu::Grid grid;
    if (shape.ndim == 2) {
        grid.threads[0] = 16;
        grid.threads[1] = 16;
        grid.threads[2] = 1;
        grid.groups[0] = (shape.dims[1] + 15) / 16;
        grid.groups[1] = (shape.dims[0] + 15) / 16;
        grid.groups[2] = 1;
    } else {
        grid = gpu::Grid::linear((shape.elements + SM_GPU_THREADS - 1) / SM_GPU_THREADS);
    }
    return grid;
}

} // namespace

BufferPtr run_gpu_dense_minmax(sm_context* context, const DenseMinMaxPlan& plan, const sm_array& input,
                               bool allow_cpu_fallback, std::string& overflow_reason) {
    float cval_f32 = 0.0f;
    double cval_abs = 0.0;
    if (plan.mode == SM_NDIMAGE_CONSTANT) {
        if (!effective_cval_f32(plan.in_dtype, plan.cval, cval_f32, cval_abs, overflow_reason)) {
            if (!allow_cpu_fallback) throw Error(SM_ENOTSUP, overflow_reason);
            return BufferPtr{};
        }
    }
    if (plan.taps.has_structure && plan.taps.structure_overflow_f32) {
        overflow_reason = "morphology structure offsets exceed single-precision range; re-routed to CPU";
        if (!allow_cpu_fallback) throw Error(SM_ENOTSUP, overflow_reason);
        return BufferPtr{};
    }

    array::ConvertReport report;
    BufferPtr single_in = array::load_single(context, input, plan.shape, &report, /*raw_bool=*/true);
    if (const char* err = report.filter_input_range_error()) {
        if (!allow_cpu_fallback) throw Error(SM_ENOTSUP, err);
        overflow_reason = err;
        return BufferPtr{};
    }
    const double peak = std::max(report.max_finite, cval_abs);
    if (plan.taps.has_structure) {
        if (array::is_int_or_bool(plan.in_dtype)) {
            if (!integer_nonflat_safe(plan.in_dtype, peak, plan.taps.structure_max_abs,
                                      plan.taps.structure_non_integral)) {
                overflow_reason = "non-flat integer morphology structure may wrap or truncate; re-routed to CPU";
                if (!allow_cpu_fallback) throw Error(SM_ENOTSUP, overflow_reason);
                return BufferPtr{};
            }
        } else if (peak + plan.taps.structure_max_abs >= float32_overflow_threshold) {
            overflow_reason = "non-flat morphology sum exceeds single-precision range; re-routed to CPU";
            if (!allow_cpu_fallback) throw Error(SM_ENOTSUP, overflow_reason);
            return BufferPtr{};
        }
    }

    const uint64_t elements = plan.shape.elements;
    const bool direct_f64 = plan.out_dtype == SM_FLOAT64;
    const bool needs_post_cast = array::is_int_or_bool(plan.out_dtype);

    GpuOperationScope operation(context, true);
    BufferPtr tap_buf = upload_taps(context, plan.taps);
    const uint64_t gpu_out_bytes = direct_f64 ? elements * sizeof(double) : elements * sizeof(float);
    BufferPtr out_buf = array::allocate(context, gpu_out_bytes);

    SMNdimageDenseMinMaxParams params{};
    params.ndim = plan.shape.ndim;
    params.num_taps = static_cast<sm_u32>(plan.taps.taps.size());
    params.elements = static_cast<sm_u32>(elements);
    params.mode = plan.mode;
    params.all_interior = plan.taps.all_interior ? 1u : 0u;
    params.cval = cval_f32;

    uint64_t stride = 1;
    for (int32_t d = static_cast<int32_t>(plan.shape.ndim) - 1; d >= 0; --d) {
        params.in_shape[d] = static_cast<sm_u32>(plan.shape.dims[d]);
        params.in_stride[d] = static_cast<sm_u32>(stride);
        params.interior_lo[d] = static_cast<sm_i32>(plan.taps.interior_lo[d]);
        params.interior_hi[d] = static_cast<sm_i32>(plan.taps.interior_hi[d]);
        stride *= plan.shape.dims[d];
    }

    static constexpr const char* kKernels[2][2][2] = {
        {{"sm_ndimage_dense_minmax_min_flat_f32", "sm_ndimage_dense_minmax_min_nonflat_f32"},
         {"sm_ndimage_dense_minmax_max_flat_f32", "sm_ndimage_dense_minmax_max_nonflat_f32"}},
        {{"sm_ndimage_dense_minmax_min_flat_f64", "sm_ndimage_dense_minmax_min_nonflat_f64"},
         {"sm_ndimage_dense_minmax_max_flat_f64", "sm_ndimage_dense_minmax_max_nonflat_f64"}},
    };
    const char* kernel = kKernels[direct_f64 ? 1 : 0][plan.is_max ? 1 : 0][plan.taps.has_structure ? 1 : 0];

    gpu::Batch batch(context);
    batch.dispatch(kernel, params,
                   {gpu::bind(single_in.get(), context), gpu::bind(out_buf.get(), context),
                    gpu::bind(tap_buf.get(), context)},
                   grid_for_nd(plan.shape));
    batch.commit_and_wait();
    operation.complete();

    if (needs_post_cast) {
        BufferPtr typed_out = array::allocate(context, elements * array::dtype_size(plan.out_dtype));
        array::ndimage_cast_from_float(reinterpret_cast<const float*>(read(out_buf.get(), context).data),
                                       plan.out_dtype, mutable_data(typed_out.get()), elements, /*dense_safe=*/true);
        return typed_out;
    }
    return out_buf;
}

BufferPtr run_gpu_rank(sm_context* context, const RankPlan& plan, const sm_array& input,
                       bool allow_cpu_fallback, std::string& overflow_reason) {
    float cval_f32 = 0.0f;
    double cval_abs = 0.0;
    if (plan.mode == SM_NDIMAGE_CONSTANT) {
        if (!effective_cval_f32(plan.in_dtype, plan.cval, cval_f32, cval_abs, overflow_reason)) {
            if (!allow_cpu_fallback) throw Error(SM_ENOTSUP, overflow_reason);
            return BufferPtr{};
        }
    }

    array::ConvertReport report;
    BufferPtr single_in = array::load_single(context, input, plan.shape, &report, /*raw_bool=*/true);
    if (const char* err = report.filter_input_range_error()) {
        if (!allow_cpu_fallback) throw Error(SM_ENOTSUP, err);
        overflow_reason = err;
        return BufferPtr{};
    }

    const uint64_t elements = plan.shape.elements;
    const bool direct_f64 = plan.out_dtype == SM_FLOAT64;
    const bool needs_post_cast = array::is_int_or_bool(plan.out_dtype);

    GpuOperationScope operation(context, true);
    const uint64_t gpu_out_bytes = direct_f64 ? elements * sizeof(double) : elements * sizeof(float);
    BufferPtr out_buf = array::allocate(context, gpu_out_bytes);

    // Check dedicated tiled register-only 2-D median fast paths (3x3, 5x5, 7x7).
    const char* fast_median_kernel = nullptr;
    if (plan.shape.ndim == 2 && plan.all_true_footprint) {
        if (plan.fshape[0] == 3 && plan.fshape[1] == 3 && plan.rank == 4) {
            fast_median_kernel = direct_f64 ? "sm_ndimage_median_3x3_f64" : "sm_ndimage_median_3x3_f32";
        } else if (plan.fshape[0] == 5 && plan.fshape[1] == 5 && plan.rank == 12) {
            fast_median_kernel = direct_f64 ? "sm_ndimage_median_5x5_f64" : "sm_ndimage_median_5x5_f32";
        } else if (plan.fshape[0] == 7 && plan.fshape[1] == 7 && plan.rank == 24) {
            fast_median_kernel = direct_f64 ? "sm_ndimage_median_7x7_f64" : "sm_ndimage_median_7x7_f32";
        }
    }

    gpu::Batch batch(context);
    BufferPtr tap_buf;
    if (fast_median_kernel) {
        SMNdimageMedian2DParams params{};
        params.height = static_cast<sm_u32>(plan.shape.dims[0]);
        params.width = static_cast<sm_u32>(plan.shape.dims[1]);
        params.start_y = static_cast<sm_i32>(plan.start[0]);
        params.start_x = static_cast<sm_i32>(plan.start[1]);
        params.mode = plan.mode;
        params.cval = cval_f32;
        batch.dispatch(fast_median_kernel, params,
                       {gpu::bind(single_in.get(), context), gpu::bind(out_buf.get(), context)},
                       grid_for_nd(plan.shape));
    } else {
        tap_buf = upload_taps(context, plan.taps);
        SMNdimageRankParams params{};
        params.ndim = plan.shape.ndim;
        params.num_taps = static_cast<sm_u32>(plan.taps.taps.size());
        params.elements = static_cast<sm_u32>(elements);
        params.rank = plan.rank;
        params.mode = plan.mode;
        params.all_interior = plan.taps.all_interior ? 1u : 0u;
        params.cval = cval_f32;

        uint64_t stride = 1;
        for (int32_t d = static_cast<int32_t>(plan.shape.ndim) - 1; d >= 0; --d) {
            params.in_shape[d] = static_cast<sm_u32>(plan.shape.dims[d]);
            params.in_stride[d] = static_cast<sm_u32>(stride);
            params.interior_lo[d] = static_cast<sm_i32>(plan.taps.interior_lo[d]);
            params.interior_hi[d] = static_cast<sm_i32>(plan.taps.interior_hi[d]);
            stride *= plan.shape.dims[d];
        }

        const uint32_t s = std::min<uint32_t>(plan.rank, static_cast<uint32_t>(plan.taps.taps.size()) - 1u - plan.rank) + 2u;
        const char* kernel = (s <= SM_NDIMAGE_RANK_SMALL_MAX_S)
                                 ? (direct_f64 ? "sm_ndimage_rank_small_f64" : "sm_ndimage_rank_small_f32")
                                 : (direct_f64 ? "sm_ndimage_rank_medium_f64" : "sm_ndimage_rank_medium_f32");
        batch.dispatch(kernel, params,
                       {gpu::bind(single_in.get(), context), gpu::bind(out_buf.get(), context),
                        gpu::bind(tap_buf.get(), context)},
                       grid_for_nd(plan.shape));
    }
    batch.commit_and_wait();
    operation.complete();

    if (needs_post_cast) {
        BufferPtr typed_out = array::allocate(context, elements * array::dtype_size(plan.out_dtype));
        array::ndimage_cast_from_float(reinterpret_cast<const float*>(read(out_buf.get(), context).data),
                                       plan.out_dtype, mutable_data(typed_out.get()), elements, /*dense_safe=*/true);
        return typed_out;
    }
    return out_buf;
}

} // namespace sm::detail::ndimage
