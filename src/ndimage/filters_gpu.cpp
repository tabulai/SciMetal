// GPU engine of the separable filters: every pass in float32, one dispatch per
// pass, all in one command buffer.
//
// The input is narrowed (or copied) once into pooled storage on the
// performance cores; passes then alternate between two pooled float32
// buffers. Only the last pass differs for float64 arrays: its kernel stores
// the widened result directly, so no pass ever reads a float64 and no separate
// conversion pass runs (probes/README.md, rules 1 and 2).
#include "filters.hpp"
#include "ndimage_kernels.h"
#include "runtime/metal_compute.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <string>
#include <vector>

namespace sm::detail::ndimage {
namespace {
// One kernel per summation order (ndimage_kernels.h) and destination type.
constexpr const char* kernels[2][3] = {
    {"sm_ndimage_correlate1d_any_f32", "sm_ndimage_correlate1d_even_f32", "sm_ndimage_correlate1d_odd_f32"},
    {"sm_ndimage_correlate1d_any_f64", "sm_ndimage_correlate1d_even_f64", "sm_ndimage_correlate1d_odd_f64"},
};
constexpr const char* minmax_kernels[2][2] = {
    {"sm_ndimage_minmax1d_min_f32", "sm_ndimage_minmax1d_max_f32"},
    {"sm_ndimage_minmax1d_min_f64", "sm_ndimage_minmax1d_max_f64"},
};
constexpr const char* minmax_vh_pass1_kernels[2] = {
    "sm_ndimage_minmax1d_vh_pass1_min", "sm_ndimage_minmax1d_vh_pass1_max",
};
constexpr const char* minmax_vh_pass2_kernels[2][2] = {
    {"sm_ndimage_minmax1d_vh_pass2_min_f32", "sm_ndimage_minmax1d_vh_pass2_max_f32"},
    {"sm_ndimage_minmax1d_vh_pass2_min_f64", "sm_ndimage_minmax1d_vh_pass2_max_f64"},
};

// The largest power-of-two threadgroup a kernel's pipeline accepts, at most
// 1024 (shader validation lowers a pipeline's capacity to 512).
uint32_t threadgroup_capacity(const char* kernel, std::atomic<uint32_t>& cached) {
    if (const uint32_t known = cached.load(std::memory_order_relaxed)) return known;
    const uint32_t limit = gpu::max_threads_per_group(kernel);
    uint32_t capacity = 1024;
    while (capacity > 1 && capacity > limit) capacity >>= 1;
    cached.store(capacity, std::memory_order_relaxed);
    return capacity;
}

// The smallest power of two that covers `extent`, at most `limit` (a power of two).
uint32_t covering(uint64_t extent, uint32_t limit) {
    uint32_t threads = 1;
    while (threads < extent && threads < limit) threads <<= 1;
    return threads;
}

// The dispatch grid of one pass and the matching kernel extents.
//
// Probe E4 measured the threadgroup shapes on 4096 x 4096: along the
// contiguous axis flat groups of 256-1024 threads are best (0.85 ms against
// 1.37 ms for 32 x 32); along a strided axis 32 x 32 is best (0.86 ms against
// 1.45 ms flat, and 7.4 ms for one column of 1024 rows). What those shapes
// share is that a group's reads cover a compact run of whole cache lines.
// Extents smaller than the measured shapes keep that property by growing the
// group along the next slower direction instead, so no group is mostly idle.
gpu::Grid grid_for_extents(const uint64_t extent[3], bool contiguous, uint32_t capacity, sm_u32 out_extent[3],
                           sm_u32& axis_in_y) {
    gpu::Grid grid;
    if (contiguous) {
        grid.threads[0] = covering(extent[0], capacity);
        grid.threads[1] = covering(extent[1], capacity / grid.threads[0]);
    } else {
        grid.threads[0] = covering(extent[0], std::min<uint32_t>(32, capacity));
        grid.threads[1] = covering(extent[1], capacity / grid.threads[0]);
        // A short axis leaves room: widen the group along the rows instead.
        grid.threads[0] = covering(extent[0], capacity / grid.threads[1]);
        grid.threads[2] = covering(extent[2], capacity / (grid.threads[0] * grid.threads[1]));
    }
    for (int axis = 0; axis < 3; ++axis) {
        grid.groups[axis] = (extent[axis] + grid.threads[axis] - 1) / grid.threads[axis];
        out_extent[axis] = static_cast<uint32_t>(extent[axis]);
    }
    axis_in_y = contiguous ? 0u : 1u;
    return grid;
}

gpu::Grid grid_for(const Pass& pass, uint32_t capacity, SMNdimageCorrelateParams& params) {
    const uint64_t extent[3] = {pass.inner == 1 ? pass.length : pass.inner,
                                pass.inner == 1 ? pass.outer : pass.length,
                                pass.inner == 1 ? 1 : pass.outer};
    return grid_for_extents(extent, pass.inner == 1, capacity, params.extent, params.axis_in_y);
}

gpu::Grid grid_for_minmax(const Pass& pass, uint32_t capacity, SMNdimageMinMax1DParams& params) {
    const uint64_t extent[3] = {pass.inner == 1 ? pass.length : pass.inner,
                                pass.inner == 1 ? pass.outer : pass.length,
                                pass.inner == 1 ? 1 : pass.outer};
    return grid_for_extents(extent, pass.inner == 1, capacity, params.extent, params.axis_in_y);
}

gpu::Grid grid_for_minmax_vh_pass1(const Pass& pass, uint64_t blocks_per_line, uint32_t capacity,
                                   SMNdimageMinMax1DParams& params) {
    const uint64_t extent[3] = {pass.inner == 1 ? blocks_per_line : pass.inner,
                                pass.inner == 1 ? pass.outer : blocks_per_line,
                                pass.inner == 1 ? 1 : pass.outer};
    return grid_for_extents(extent, pass.inner == 1, capacity, params.extent, params.axis_in_y);
}

uint64_t pass_vh_ext_elems(const Pass& pass) noexcept {
    if (!gpu_minmax_use_vanherk(pass)) return 0;
    return pass.outer * (pass.length + pass.size - 1) * pass.inner;
}

void encode_minmax_pass(sm_context* context, gpu::Batch& batch, const Pass& pass, const sm_buffer* src,
                        const sm_buffer* dst, bool widen_store, const sm_buffer* vh_prefix,
                        const sm_buffer* vh_suffix) {
    static std::atomic<uint32_t> direct_caps[2][2] = {};
    static std::atomic<uint32_t> vh1_caps[2] = {};
    static std::atomic<uint32_t> vh2_caps[2][2] = {};
    const uint32_t is_max = pass.kind == SM_NDIMAGE_MAXIMUM ? 1u : 0u;
    const uint32_t wide_idx = widen_store ? 1u : 0u;
    if (gpu_minmax_use_vanherk(pass) && vh_prefix && vh_suffix) {
        const uint64_t ext_len = pass.length + pass.size - 1;
        const uint64_t blocks = (ext_len + pass.size - 1) / pass.size;
        SMNdimageMinMax1DParams p1{};
        const char* k1 = minmax_vh_pass1_kernels[is_max];
        const gpu::Grid g1 = grid_for_minmax_vh_pass1(pass, blocks, threadgroup_capacity(k1, vh1_caps[is_max]), p1);
        p1.length = static_cast<uint32_t>(pass.length);
        p1.stride = static_cast<uint32_t>(pass.inner);
        p1.taps = static_cast<uint32_t>(pass.size);
        p1.before = static_cast<uint32_t>(pass.before);
        p1.mode = pass.mode;
        p1.ext_len = static_cast<uint32_t>(ext_len);
        p1.cval = static_cast<float>(pass.cval);
        batch.dispatch(k1, p1,
                       {gpu::bind(src, context), gpu::bind(vh_prefix, context), gpu::bind(vh_suffix, context)}, g1);

        SMNdimageMinMax1DParams p2{};
        const char* k2 = minmax_vh_pass2_kernels[wide_idx][is_max];
        const gpu::Grid g2 = grid_for_minmax(pass, threadgroup_capacity(k2, vh2_caps[wide_idx][is_max]), p2);
        p2.length = static_cast<uint32_t>(pass.length);
        p2.stride = static_cast<uint32_t>(pass.inner);
        p2.taps = static_cast<uint32_t>(pass.size);
        p2.before = static_cast<uint32_t>(pass.before);
        p2.mode = pass.mode;
        p2.ext_len = static_cast<uint32_t>(ext_len);
        p2.cval = static_cast<float>(pass.cval);
        batch.dispatch(k2, p2,
                       {gpu::bind(vh_prefix, context), gpu::bind(vh_suffix, context), gpu::bind(dst, context)}, g2);
        return;
    }
    SMNdimageMinMax1DParams params{};
    const char* kernel = minmax_kernels[wide_idx][is_max];
    const gpu::Grid grid = grid_for_minmax(pass, threadgroup_capacity(kernel, direct_caps[wide_idx][is_max]), params);
    params.length = static_cast<uint32_t>(pass.length);
    params.stride = static_cast<uint32_t>(pass.inner);
    params.taps = static_cast<uint32_t>(pass.size);
    params.before = static_cast<uint32_t>(pass.before);
    params.mode = pass.mode;
    params.ext_len = 0u;
    params.cval = static_cast<float>(pass.cval);
    batch.dispatch(kernel, params, {gpu::bind(src, context), gpu::bind(dst, context)}, grid);
}

uint32_t symmetry_index(Symmetry symmetry) {
    switch (symmetry) {
    case Symmetry::even: return SM_NDIMAGE_SYMMETRY_EVEN;
    case Symmetry::odd: return SM_NDIMAGE_SYMMETRY_ODD;
    default: return SM_NDIMAGE_SYMMETRY_NONE;
    }
}
constexpr double float32_overflow_threshold = 0x1.ffffffp127;

bool overflows_float32(double value) noexcept {
    return std::isfinite(value) && std::fabs(value) >= float32_overflow_threshold;
}

std::string check_passes_parameters_range(const std::vector<Pass>& passes) {
    for (const Pass& pass : passes) {
        if (pass.mode == SM_NDIMAGE_CONSTANT && overflows_float32(pass.cval)) {
            return "constant cval exceeds single-precision range; re-routed to CPU";
        }
        if (pass.kind == SM_NDIMAGE_CORRELATE) {
            for (uint64_t j = 0; j < pass.size; ++j) {
                if (overflows_float32(pass.weights[j])) {
                    return "filter weights exceed single-precision range; re-routed to CPU";
                }
            }
        }
    }
    return {};
}

std::string check_parameters_range(const Plan& plan) {
    return check_passes_parameters_range(plan.passes);
}

std::string check_chain_gain_range(const std::vector<Pass>& passes, const array::ConvertReport& report,
                                   bool input_is_double, double& out_m, double& out_e) {
    constexpr double u = 0x1p-24;
    double m = std::fabs(static_cast<double>(static_cast<float>(report.max_finite)));
    double e = u * m;
    for (const Pass& pass : passes) {
        if (pass.mode == SM_NDIMAGE_CONSTANT && std::isfinite(pass.cval)) {
            const double c = std::fabs(static_cast<double>(static_cast<float>(pass.cval)));
            if (c > m) {
                m = c;
            }
            e = std::max(e, u * c);
        }
        if (pass.kind == SM_NDIMAGE_MINIMUM || pass.kind == SM_NDIMAGE_MAXIMUM) {
            continue;
        }
        // Symmetric and antisymmetric kernels (including odd-sized uniform passes)
        // evaluate (below +- above) in float32 before multiplying by the tap weight.
        if (pass.size > 1 && pass.symmetry != Symmetry::none &&
            (!std::isfinite(m + e) || 2.0 * (m + e) >= float32_overflow_threshold)) {
            return input_is_double
                       ? "float64 symmetric filter pair sum overflowed float32; re-routed to CPU"
                       : "mixed-precision symmetric filter pair sum overflowed float32; re-routed to CPU";
        }
        double gain = 0.0;
        if (pass.kind == SM_NDIMAGE_UNIFORM) {
            const double w = std::fabs(static_cast<double>(static_cast<float>(1.0 / static_cast<double>(pass.size))));
            gain = static_cast<double>(pass.size) * w;
        } else if (pass.symmetry != Symmetry::none) {
            // Symmetric/antisymmetric kernels read weights[0..half] and multiply
            // weights[j] (j < half) by (below +- above), even if weights[size-1-j] is NaN.
            const uint64_t half = pass.size / 2;
            const double center = std::fabs(static_cast<double>(static_cast<float>(pass.weights[half])));
            if (std::isfinite(center)) gain += center;
            for (uint64_t j = 0; j < half; ++j) {
                const double aw = std::fabs(static_cast<double>(static_cast<float>(pass.weights[j])));
                if (std::isfinite(aw)) gain += 2.0 * aw;
            }
        } else {
            for (uint64_t j = 0; j < pass.size; ++j) {
                const double aw = std::fabs(static_cast<double>(static_cast<float>(pass.weights[j])));
                if (std::isfinite(aw)) gain += aw;
            }
        }
        e = gain * e + static_cast<double>(pass.size + 3) * u * gain * m;
        m *= gain;
        if (!std::isfinite(m + e) || m + e >= float32_overflow_threshold) {
            return input_is_double
                       ? "float64 filter gain overflowed float32; re-routed to CPU"
                       : "mixed-precision filter gain overflowed float32; re-routed to CPU";
        }
    }
    out_m = m;
    out_e = e;
    return {};
}

std::string check_gain_range(const Plan& plan, const array::ConvertReport& report, bool input_is_double) {
    double m = 0.0, e = 0.0;
    return check_chain_gain_range(plan.passes, report, input_is_double, m, e);
}

std::string check_composite_gain_range(const CompositePlan& plan, const array::ConvertReport& report,
                                       bool input_is_double) {
    constexpr double u = 0x1p-24;
    const size_t k_count = plan.chains.size();
    double sum_m = 0.0, sum_e = 0.0, sq_sum = 0.0;
    for (size_t k = 0; k < k_count; ++k) {
        double mk = 0.0, ek = 0.0;
        std::string err = check_chain_gain_range(plan.chains[k], report, input_is_double, mk, ek);
        if (!err.empty()) return err;
        sum_m += mk;
        sum_e += ek;
        const double bound_k = mk + ek;
        sq_sum += bound_k * bound_k;
    }
    if (plan.combine_op == SM_NDIMAGE_COMBINE_SUM || plan.combine_op == SM_NDIMAGE_COMBINE_SUB) {
        sum_e += static_cast<double>(k_count > 0 ? k_count - 1 : 0) * u * sum_m;
        if (!std::isfinite(sum_m + sum_e) || sum_m + sum_e >= float32_overflow_threshold) {
            return input_is_double
                       ? "float64 composite filter sum overflowed float32; re-routed to CPU"
                       : "mixed-precision composite filter sum overflowed float32; re-routed to CPU";
        }
        if (plan.combine_op == SM_NDIMAGE_COMBINE_SUB && array::is_integer(plan.out_dtype) &&
            sum_m >= static_cast<double>(uint64_t{1} << 24)) {
            return "integer composite subtraction exceeds exact float32 range; re-routed to CPU";
        }
    } else if (plan.combine_op == SM_NDIMAGE_COMBINE_MORPH_LAPLACE) {
        const double in_m = std::fabs(static_cast<double>(static_cast<float>(report.max_finite)));
        const double total_m = sum_m + 2.0 * in_m;
        const double total_e = sum_e + 3.0 * u * total_m;
        if (!std::isfinite(total_m + total_e) || total_m + total_e >= float32_overflow_threshold) {
            return input_is_double
                       ? "float64 morphological laplace overflowed float32; re-routed to CPU"
                       : "mixed-precision morphological laplace overflowed float32; re-routed to CPU";
        }
        if (array::is_integer(plan.out_dtype) && total_m >= static_cast<double>(uint64_t{1} << 24)) {
            return "integer morphological laplace exceeds exact float32 range; re-routed to CPU";
        }
    } else {
        if (!std::isfinite(sq_sum) ||
            sq_sum * (1.0 + static_cast<double>(k_count + 1) * u) >= float32_overflow_threshold) {
            return input_is_double
                       ? "float64 composite gradient square sum overflowed float32; re-routed to CPU"
                       : "mixed-precision composite gradient square sum overflowed float32; re-routed to CPU";
        }
    }
    return {};
}
} // namespace

BufferPtr run_gpu(sm_context* context, const Plan& plan, const sm_array& input, bool allow_cpu_fallback,
                  std::string& overflow_reason, bool dense_safe) {
    static std::atomic<uint32_t> capacities[2][3] = {};
    overflow_reason = check_parameters_range(plan);
    if (!overflow_reason.empty()) {
        if (!allow_cpu_fallback) {
            throw Error(SM_ENOTSUP, "values exceed single-precision range; forced GPU cannot fall back to CPU");
        }
        return BufferPtr{};
    }

    const size_t count = plan.passes.size();
    const uint64_t elements = plan.shape.elements;
    const bool input_is_double = plan.dtype == SM_FLOAT64;
    const uint32_t out_dtype = plan.out_dtype ? plan.out_dtype : plan.dtype;
    const bool direct_f64_store = out_dtype == SM_FLOAT64;
    const bool needs_post_cast = array::is_int_or_bool(out_dtype);

    array::ConvertReport report;
    BufferPtr single[2];
    single[0] = array::load_single(context, input, plan.shape, &report, /*raw_bool=*/true);
    if (const char* error = report.filter_input_range_error()) {
        if (!allow_cpu_fallback) throw Error(SM_ENOTSUP, error);
        overflow_reason = error;
        return BufferPtr{};
    }
    overflow_reason = check_gain_range(plan, report, input_is_double);
    if (!overflow_reason.empty()) {
        single[0].reset();
        if (!allow_cpu_fallback) {
            throw Error(SM_ENOTSUP, "values exceed single-precision range; forced GPU cannot fall back to CPU");
        }
        return BufferPtr{};
    }

    GpuOperationScope operation(context, true);

    // All correlate/uniform passes' weights in one small buffer, narrowed like the data.
    uint64_t total_taps = 0;
    uint64_t max_vh_elems = 0;
    for (const Pass& pass : plan.passes) {
        if (pass.kind == SM_NDIMAGE_CORRELATE || pass.kind == SM_NDIMAGE_UNIFORM) total_taps += pass.size;
        max_vh_elems = std::max(max_vh_elems, pass_vh_ext_elems(pass));
    }
    BufferPtr weights = total_taps ? array::allocate(context, total_taps * sizeof(float)) : BufferPtr();
    if (total_taps) {
        float* w = reinterpret_cast<float*>(mutable_data(weights.get()));
        for (const Pass& pass : plan.passes) {
            if (pass.kind == SM_NDIMAGE_UNIFORM) {
                std::fill_n(w, pass.size, static_cast<float>(1.0 / static_cast<double>(pass.size)));
                w += pass.size;
            } else if (pass.kind == SM_NDIMAGE_CORRELATE) {
                for (uint64_t j = 0; j < pass.size; ++j) w[j] = static_cast<float>(pass.weights[j]);
                w += pass.size;
            }
        }
    }
    BufferPtr vh_prefix = max_vh_elems ? array::allocate(context, max_vh_elems * sizeof(float)) : BufferPtr();
    BufferPtr vh_suffix = max_vh_elems ? array::allocate(context, max_vh_elems * sizeof(float)) : BufferPtr();

    // Pass i reads buffer i % 2 and writes the other one; a float64 result has
    // its own buffer, written by the last pass. Integer/bool results write
    // float32 then post-cast on P-cores.
    if (count > 1 || !direct_f64_store) single[1] = array::allocate(context, elements * sizeof(float));
    BufferPtr wide = direct_f64_store ? array::allocate(context, elements * sizeof(double)) : BufferPtr();

    gpu::Batch batch(context);
    uint32_t weight_offset = 0;
    for (size_t i = 0; i < count; ++i) {
        const Pass& pass = plan.passes[i];
        const bool last = i + 1 == count;
        const sm_buffer* destination = last && direct_f64_store ? wide.get() : single[(i + 1) % 2].get();
        if (pass.kind == SM_NDIMAGE_MINIMUM || pass.kind == SM_NDIMAGE_MAXIMUM) {
            encode_minmax_pass(context, batch, pass, single[i % 2].get(), destination, last && direct_f64_store,
                               vh_prefix.get(), vh_suffix.get());
            continue;
        }
        const uint32_t wide_store = last && direct_f64_store, order = symmetry_index(pass.symmetry);
        const char* kernel = kernels[wide_store][order];
        const uint32_t capacity = threadgroup_capacity(kernel, capacities[wide_store][order]);
        SMNdimageCorrelateParams params{};
        const gpu::Grid grid = grid_for(pass, capacity, params);
        params.length = static_cast<uint32_t>(pass.length);
        params.stride = static_cast<uint32_t>(pass.inner);
        params.taps = static_cast<uint32_t>(pass.size);
        params.weight_offset = weight_offset;
        params.before = static_cast<uint32_t>(pass.before);
        params.mode = pass.mode;
        params.cval = static_cast<float>(pass.cval);
        batch.dispatch(kernel, params,
                       {gpu::bind(single[i % 2].get(), context), gpu::bind(destination, context),
                        gpu::bind(weights.get(), context)},
                       grid);
        weight_offset += params.taps;
    }
    batch.commit_and_wait();
    operation.complete();
    if (direct_f64_store) return wide;
    if (needs_post_cast) {
        BufferPtr out_buf = array::allocate(context, elements * array::dtype_size(out_dtype));
        array::ndimage_cast_from_float(
            reinterpret_cast<const float*>(read(single[count % 2].get(), context).data),
            out_dtype, mutable_data(out_buf.get()), elements, dense_safe);
        return out_buf;
    }
    return std::move(single[count % 2]);
}

BufferPtr run_gpu_composite(sm_context* context, const CompositePlan& plan, const sm_array& input,
                            bool allow_cpu_fallback, std::string& overflow_reason) {
    static std::atomic<uint32_t> capacities[2][3] = {};
    for (const auto& chain : plan.chains) {
        overflow_reason = check_passes_parameters_range(chain);
        if (!overflow_reason.empty()) {
            if (!allow_cpu_fallback) {
                throw Error(SM_ENOTSUP, "values exceed single-precision range; forced GPU cannot fall back to CPU");
            }
            return BufferPtr{};
        }
    }

    const uint64_t elements = plan.shape.elements;
    const bool input_is_double = plan.in_dtype == SM_FLOAT64;
    const bool direct_f64_store = plan.out_dtype == SM_FLOAT64;
    const bool needs_post_cast = array::is_int_or_bool(plan.out_dtype);

    array::ConvertReport report;
    BufferPtr single_in = array::load_single(context, input, plan.shape, &report, /*raw_bool=*/true);
    if (const char* error = report.filter_input_range_error()) {
        if (!allow_cpu_fallback) throw Error(SM_ENOTSUP, error);
        overflow_reason = error;
        return BufferPtr{};
    }
    overflow_reason = check_composite_gain_range(plan, report, input_is_double);
    if (!overflow_reason.empty()) {
        single_in.reset();
        if (!allow_cpu_fallback) {
            throw Error(SM_ENOTSUP, "values exceed single-precision range; forced GPU cannot fall back to CPU");
        }
        return BufferPtr{};
    }

    GpuOperationScope operation(context, true);

    uint64_t total_taps = 0;
    uint64_t max_vh_elems = 0;
    size_t max_passes = 0;
    for (const auto& chain : plan.chains) {
        max_passes = std::max(max_passes, chain.size());
        for (const Pass& pass : chain) {
            if (pass.kind == SM_NDIMAGE_CORRELATE || pass.kind == SM_NDIMAGE_UNIFORM) total_taps += pass.size;
            max_vh_elems = std::max(max_vh_elems, pass_vh_ext_elems(pass));
        }
    }
    BufferPtr weights = total_taps ? array::allocate(context, total_taps * sizeof(float)) : BufferPtr();
    if (total_taps) {
        float* w = reinterpret_cast<float*>(mutable_data(weights.get()));
        for (const auto& chain : plan.chains) {
            for (const Pass& pass : chain) {
                if (pass.kind == SM_NDIMAGE_UNIFORM) {
                    std::fill_n(w, pass.size, static_cast<float>(1.0 / static_cast<double>(pass.size)));
                    w += pass.size;
                } else if (pass.kind == SM_NDIMAGE_CORRELATE) {
                    for (uint64_t j = 0; j < pass.size; ++j) w[j] = static_cast<float>(pass.weights[j]);
                    w += pass.size;
                }
            }
        }
    }
    BufferPtr vh_prefix = max_vh_elems ? array::allocate(context, max_vh_elems * sizeof(float)) : BufferPtr();
    BufferPtr vh_suffix = max_vh_elems ? array::allocate(context, max_vh_elems * sizeof(float)) : BufferPtr();

    const size_t k_count = plan.chains.size();
    const bool is_grad = plan.combine_op == SM_NDIMAGE_COMBINE_SQRT_SUM_SQ;
    const uint64_t f32_bytes = elements * sizeof(float);

    BufferPtr ping = (max_passes >= 2) ? array::allocate(context, f32_bytes) : BufferPtr();
    BufferPtr pong = (max_passes >= 3) ? array::allocate(context, f32_bytes) : BufferPtr();
    BufferPtr res0 = ((k_count >= 2 || (k_count == 1 && is_grad)) && !plan.chains[0].empty())
                         ? array::allocate(context, f32_bytes)
                         : BufferPtr();
    BufferPtr res1;
    if (k_count >= 2) {
        bool need_res1 = false;
        for (size_t k = 1; k < k_count; ++k) {
            if (!plan.chains[k].empty()) {
                need_res1 = true;
                break;
            }
        }
        if (need_res1) res1 = array::allocate(context, f32_bytes);
    }
    BufferPtr acc = (k_count >= 3) ? array::allocate(context, f32_bytes) : BufferPtr();
    const uint64_t gpu_out_bytes = needs_post_cast ? f32_bytes : elements * array::dtype_size(plan.out_dtype);
    BufferPtr out_buf = array::allocate(context, gpu_out_bytes);

    gpu::Batch batch(context);
    uint32_t weight_offset = 0;

    auto encode_chain = [&](const std::vector<Pass>& chain, const sm_buffer* dst_buf, bool widen_last) -> const sm_buffer* {
        if (chain.empty()) return single_in.get();
        const size_t count = chain.size();
        const sm_buffer* src = single_in.get();
        for (size_t i = 0; i < count; ++i) {
            const Pass& pass = chain[i];
            const bool last = (i + 1 == count);
            const sm_buffer* dst = last ? dst_buf : (((count - 1 - i) % 2 == 1) ? ping.get() : pong.get());
            if (pass.kind == SM_NDIMAGE_MINIMUM || pass.kind == SM_NDIMAGE_MAXIMUM) {
                encode_minmax_pass(context, batch, pass, src, dst, last && widen_last, vh_prefix.get(), vh_suffix.get());
                src = dst;
                continue;
            }
            const uint32_t wide_store = (last && widen_last) ? 1u : 0u;
            const uint32_t order = symmetry_index(pass.symmetry);
            const char* kernel = kernels[wide_store][order];
            const uint32_t capacity = threadgroup_capacity(kernel, capacities[wide_store][order]);
            SMNdimageCorrelateParams params{};
            const gpu::Grid grid = grid_for(pass, capacity, params);
            params.length = static_cast<uint32_t>(pass.length);
            params.stride = static_cast<uint32_t>(pass.inner);
            params.taps = static_cast<uint32_t>(pass.size);
            params.weight_offset = weight_offset;
            params.before = static_cast<uint32_t>(pass.before);
            params.mode = pass.mode;
            params.cval = static_cast<float>(pass.cval);
            batch.dispatch(kernel, params,
                           {gpu::bind(src, context), gpu::bind(dst, context), gpu::bind(weights.get(), context)},
                           grid);
            weight_offset += params.taps;
            src = dst;
        }
        return dst_buf;
    };

    auto encode_combine = [&](uint32_t op, const sm_buffer* a, const sm_buffer* b, const sm_buffer* dst, bool widen_store) {
        const char* kernel = widen_store ? "sm_ndimage_combine_f64" : "sm_ndimage_combine_f32";
        SMNdimageCombineParams params{};
        params.elements = static_cast<sm_u32>(elements);
        params.op = op;
        const uint64_t groups = (elements + SM_GPU_THREADS - 1) / SM_GPU_THREADS;
        batch.dispatch(kernel, params,
                       {gpu::bind(a, context), gpu::bind(b, context), gpu::bind(dst, context)}, groups);
    };

    if (plan.combine_op == SM_NDIMAGE_COMBINE_SUB) {
        const sm_buffer* buf0 = encode_chain(plan.chains[0], res0.get(), false);
        const sm_buffer* buf1 = encode_chain(plan.chains[1], res1.get(), false);
        encode_combine(SM_NDIMAGE_COMBINE_OP_SUB2, buf0, buf1, out_buf.get(), direct_f64_store);
    } else if (plan.combine_op == SM_NDIMAGE_COMBINE_MORPH_LAPLACE) {
        const sm_buffer* buf0 = encode_chain(plan.chains[0], res0.get(), false);
        const sm_buffer* buf1 = encode_chain(plan.chains[1], res1.get(), false);
        const char* kernel = direct_f64_store ? "sm_ndimage_combine_morph_laplace_f64"
                                              : "sm_ndimage_combine_morph_laplace_f32";
        SMNdimageCombineParams params{};
        params.elements = static_cast<sm_u32>(elements);
        params.op = 0u;
        const uint64_t groups = (elements + SM_GPU_THREADS - 1) / SM_GPU_THREADS;
        batch.dispatch(kernel, params,
                       {gpu::bind(buf0, context), gpu::bind(buf1, context), gpu::bind(single_in.get(), context),
                        gpu::bind(out_buf.get(), context)},
                       groups);
    } else if (k_count == 1) {
        if (plan.chains[0].empty()) {
            const uint32_t op = is_grad ? SM_NDIMAGE_COMBINE_OP_SQRT_SQ1 : SM_NDIMAGE_COMBINE_OP_COPY;
            encode_combine(op, single_in.get(), single_in.get(), out_buf.get(), direct_f64_store);
        } else if (!is_grad) {
            encode_chain(plan.chains[0], out_buf.get(), direct_f64_store);
        } else {
            const sm_buffer* buf0 = encode_chain(plan.chains[0], res0.get(), false);
            encode_combine(SM_NDIMAGE_COMBINE_OP_SQRT_SQ1, buf0, buf0, out_buf.get(), direct_f64_store);
        }
    } else if (k_count == 2) {
        const sm_buffer* buf0 = encode_chain(plan.chains[0], res0.get(), false);
        const sm_buffer* buf1 = encode_chain(plan.chains[1], res1.get(), false);
        const uint32_t op = is_grad ? SM_NDIMAGE_COMBINE_OP_SQRT_SUM_SQ2 : SM_NDIMAGE_COMBINE_OP_SUM2;
        encode_combine(op, buf0, buf1, out_buf.get(), direct_f64_store);
    } else {
        const sm_buffer* buf0 = encode_chain(plan.chains[0], res0.get(), false);
        const sm_buffer* buf1 = encode_chain(plan.chains[1], res1.get(), false);
        encode_combine(is_grad ? SM_NDIMAGE_COMBINE_OP_SUM_SQ2 : SM_NDIMAGE_COMBINE_OP_SUM2,
                       buf0, buf1, acc.get(), false);
        for (size_t k = 2; k < k_count; ++k) {
            const sm_buffer* bufk = encode_chain(plan.chains[k], res1.get(), false);
            if (k + 1 < k_count) {
                encode_combine(is_grad ? SM_NDIMAGE_COMBINE_OP_ADD_SQ : SM_NDIMAGE_COMBINE_OP_SUM2,
                               acc.get(), bufk, acc.get(), false);
            } else {
                encode_combine(is_grad ? SM_NDIMAGE_COMBINE_OP_SQRT_ADD_SQ : SM_NDIMAGE_COMBINE_OP_SUM2,
                               acc.get(), bufk, out_buf.get(), direct_f64_store);
            }
        }
    }

    batch.commit_and_wait();
    operation.complete();
    if (needs_post_cast) {
        BufferPtr typed_out = array::allocate(context, elements * array::dtype_size(plan.out_dtype));
        array::ndimage_cast_from_float(
            reinterpret_cast<const float*>(read(out_buf.get(), context).data),
            plan.out_dtype, mutable_data(typed_out.get()), elements, /*dense_safe=*/false);
        return typed_out;
    }
    return out_buf;
}
} // namespace sm::detail::ndimage
