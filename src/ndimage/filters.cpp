// scipy.ndimage separable, dense, and composite filters: argument validation, routing and the C ABI.
#include "filters.hpp"
#include "core/dense_correlate.hpp"
#include "core/router.hpp"
#include "ndimage_kernels.h"

#include <cfloat>
#include <cmath>
#include <vector>

namespace ms::detail::ndimage {
namespace {
// AUTO crossover, calibrated on applegpu_g13d (M1 Ultra, 16 performance
// cores) with bench/ndimage_filters.py --calibrate.
//
// The work unit is the CPU engine's cost in tap operations (routing_work):
// per pass and element, its arithmetic (one operation per symmetric tap pair,
// one per tap otherwise, about 16 for a uniform pass of any length) plus,
// along any axis but the innermost, about 12 for the strided memory traffic
// (lines along the innermost axis are contiguous and cost nothing extra).
// The GPU engine costs a fixed 0.3 to 0.4 ms per call plus a per-element
// share that depends little on the taps, so it wins from about this much CPU
// work: 4 M elements for a 2-D Gaussian of sigma 1 to 3, 1 M for sigma 10,
// 2 M for a 3-D Gaussian or a 2-D uniform filter, 8 M for a 3-tap sobel.
// Just below the threshold the GPU would be up to 1.5 times faster and just
// above it sometimes 0.8 times as fast (few taps along the innermost axis):
// in between, the engines are within half a millisecond of each other.
constexpr router::Family family{"ndimage.separable_filter", uint64_t{80} << 20};
// Dense N-D correlation crossover (elements x non-zero taps), calibrated on
// applegpu_g13d with bench/ndimage_filters.py --calibrate:
//   5x5 2048^2 (100 Mi): CPU 1.04 ms vs GPU 1.42 ms (cpu wins)
//   3x3 4096^2 (144 Mi): CPU 2.82 ms vs GPU 2.26 ms (gpu wins)
//   5x5 2896^2 (200 Mi): CPU 2.41 ms vs GPU 2.12 ms (gpu wins)
//   5x5 4096^2 (400 Mi): CPU 4.66 ms vs GPU 3.53 ms (gpu wins)
constexpr router::Family dense_family{"ndimage.dense_filter", uint64_t{128} << 20};
constexpr uint64_t uniform_operations = 16;
constexpr uint64_t strided_pass_operations = 12;

// One GPU tap-element costs about a third of a CPU operation. A plan whose
// GPU taps exceed its CPU work by more than this factor stays on the CPU:
// long uniform windows, where the CPU's running sum does not grow with the
// window and the GPU's direct sum does (the engines meet near 80 taps).
constexpr uint64_t gpu_taps_per_cpu_operation = 3;

// One command buffer should not hold the GPU, which the window server
// shares, for long. The kernels do about 5e11 tap-elements per second, 2e11
// when the windows overhang the lines (every tap then goes through the
// boundary code), so this is 0.5 to 1.5 s. Larger plans run on the CPU.
constexpr uint64_t gpu_max_tap_elements = uint64_t{1} << 38;
// All weights of a call share one float32 buffer indexed with 32 bits.
constexpr uint64_t gpu_max_weights = uint64_t{1} << 24;

// NI_Correlate1D's test, to the letter: odd-length weights whose halves agree
// (or cancel) within DBL_EPSILON take the paired loops. NaN weights compare
// false and so count as symmetric there too.
Symmetry classify(const double* weights, uint64_t size) {
    if (!(size & 1)) return Symmetry::none;
    const uint64_t half = size / 2;
    bool even = true;
    for (uint64_t i = 1; i <= half; ++i) {
        if (std::fabs(weights[half + i] - weights[half - i]) > DBL_EPSILON) {
            even = false;
            break;
        }
    }
    if (even) return Symmetry::even;
    for (uint64_t i = 1; i <= half; ++i) {
        if (std::fabs(weights[half + i] + weights[half - i]) > DBL_EPSILON) return Symmetry::none;
    }
    return Symmetry::odd;
}

Pass parse_pass(const array::Shape& shape, const ms_ndimage_pass& given) {
    if (given.struct_size != sizeof(ms_ndimage_pass) || given.abi_version != MS_NDIMAGE_ABI_VERSION)
        throw Error(MS_EINVAL, "filter pass size or ABI version mismatch");
    if (given.reserved0 || given.reserved[0] || given.reserved[1])
        throw Error(MS_EINVAL, "filter pass reserved fields must be zero");
    if (given.kind != MS_NDIMAGE_CORRELATE && given.kind != MS_NDIMAGE_UNIFORM)
        throw Error(MS_EINVAL, "unknown filter pass kind");
    if (given.axis >= shape.ndim) throw Error(MS_EINVAL, "filter axis is out of range");
    if (given.mode > MS_NDIMAGE_CONSTANT) throw Error(MS_EINVAL, "unknown boundary mode");
    if (given.size < 1 || given.size > MS_NDIMAGE_MAX_WINDOW)
        throw Error(MS_EINVAL, "filter size must be 1 to 2^30");
    const int64_t size = static_cast<int64_t>(given.size);
    if (given.origin < -(size / 2) || given.origin > (size - 1) / 2)
        throw Error(MS_EINVAL, "filter origin must satisfy -(size / 2) <= origin <= (size - 1) / 2");
    if (given.kind == MS_NDIMAGE_CORRELATE && !given.weights) throw Error(MS_EINVAL, "null filter weights");
    if (given.kind == MS_NDIMAGE_UNIFORM && given.weights)
        throw Error(MS_EINVAL, "a uniform filter pass takes no weights");

    Pass pass;
    pass.kind = given.kind;
    pass.mode = given.mode;
    pass.cval = given.cval;
    pass.size = given.size;
    pass.weights = given.weights;
    // A uniform window is symmetric when it has a centre; the GPU engine sums it as weights 1/size.
    pass.symmetry = given.kind == MS_NDIMAGE_UNIFORM ? (given.size & 1 ? Symmetry::even : Symmetry::none)
                                                     : classify(given.weights, given.size);
    pass.length = shape.dims[given.axis];
    pass.outer = pass.inner = 1;
    for (uint32_t axis = 0; axis < given.axis; ++axis) pass.outer *= shape.dims[axis];
    for (uint32_t axis = given.axis + 1; axis < shape.ndim; ++axis) pass.inner *= shape.dims[axis];
    pass.before = static_cast<uint64_t>(size / 2 + given.origin);
    pass.after = static_cast<uint64_t>(size - size / 2 - 1 - given.origin);
    return pass;
}

uint32_t resolve_out_dtype(uint32_t in_dtype, uint32_t requested_out_dtype) {
    if (requested_out_dtype == 0) return in_dtype;
    if (requested_out_dtype != MS_FLOAT32 && requested_out_dtype != MS_FLOAT64)
        throw Error(MS_ENOTSUP, "ndimage output dtype must be float32 or float64");
    return requested_out_dtype;
}

Plan make_plan(const ms_array* input, const ms_ndimage_pass* passes, uint32_t count, uint32_t out_dtype = 0) {
    Plan plan;
    plan.shape = array::checked_shape(input);
    if (input->dtype != MS_FLOAT32 && input->dtype != MS_FLOAT64)
        throw Error(MS_ENOTSUP, "ndimage filters take float32 or float64 arrays");
    plan.dtype = input->dtype;
    plan.out_dtype = resolve_out_dtype(input->dtype, out_dtype);
    if (reinterpret_cast<uintptr_t>(input->data) % array::dtype_size(input->dtype))
        throw Error(MS_EINVAL, "array data is not aligned to its element size");
    if (!passes) throw Error(MS_EINVAL, "null filter passes");
    if (count < 1 || count > MS_NDIMAGE_MAX_PASSES) throw Error(MS_EINVAL, "filter pass count must be 1 to 32");
    plan.passes.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        plan.passes.push_back(parse_pass(plan.shape, passes[i]));
    }
    return plan;
}

CompositePlan make_composite_plan(const ms_array* input, const ms_ndimage_chain* chains, uint32_t chain_count,
                                    uint32_t combine_op, uint32_t out_dtype) {
    CompositePlan plan;
    plan.shape = array::checked_shape(input);
    if (input->dtype != MS_FLOAT32 && input->dtype != MS_FLOAT64)
        throw Error(MS_ENOTSUP, "ndimage filters take float32 or float64 arrays");
    if (reinterpret_cast<uintptr_t>(input->data) % array::dtype_size(input->dtype))
        throw Error(MS_EINVAL, "array data is not aligned to its element size");
    plan.in_dtype = input->dtype;
    plan.out_dtype = resolve_out_dtype(input->dtype, out_dtype);
    if (combine_op != MS_NDIMAGE_COMBINE_SUM && combine_op != MS_NDIMAGE_COMBINE_SQRT_SUM_SQ)
        throw Error(MS_EINVAL, "unknown composite combine op");
    plan.combine_op = combine_op;
    if (!chains) throw Error(MS_EINVAL, "null filter chains");
    if (chain_count < 1 || chain_count > MS_MAX_NDIM)
        throw Error(MS_EINVAL, "composite filter chain count must be 1 to MS_MAX_NDIM");
    plan.chains.resize(chain_count);
    for (uint32_t k = 0; k < chain_count; ++k) {
        const ms_ndimage_chain& c = chains[k];
        if (c.struct_size != sizeof(ms_ndimage_chain) || c.abi_version != MS_NDIMAGE_ABI_VERSION)
            throw Error(MS_EINVAL, "filter chain size or ABI version mismatch");
        if (c.reserved0 || c.reserved[0] || c.reserved[1])
            throw Error(MS_EINVAL, "filter chain reserved fields must be zero");
        if (c.pass_count > MS_NDIMAGE_MAX_PASSES)
            throw Error(MS_EINVAL, "filter chain pass count must be 0 to 32");
        if (c.pass_count == 0 && c.passes != nullptr)
            throw Error(MS_EINVAL, "empty filter chain must have null passes");
        if (c.pass_count > 0 && !c.passes)
            throw Error(MS_EINVAL, "non-empty filter chain has null passes");
        plan.chains[k].reserve(c.pass_count);
        for (uint32_t i = 0; i < c.pass_count; ++i) {
            plan.chains[k].push_back(parse_pass(plan.shape, c.passes[i]));
        }
    }
    return plan;
}

// The CPU engine's cost of one pass per element, in tap operations.
uint64_t cpu_operations(const Pass& pass) {
    const uint64_t arithmetic = pass.kind == MS_NDIMAGE_UNIFORM ? uniform_operations
                              : pass.symmetry == Symmetry::none ? pass.size : pass.size / 2 + 1;
    return arithmetic + (pass.inner == 1 ? 0 : strided_pass_operations);
}

uint64_t saturating_mul(uint64_t a, uint64_t b) {
    uint64_t product = 0;
    return ms_checked_mul(a, b, &product) == MS_OK ? product : UINT64_MAX;
}
uint64_t saturating_add(uint64_t a, uint64_t b) {
    uint64_t sum = 0;
    return ms_checked_add(a, b, &sum) == MS_OK ? sum : UINT64_MAX;
}

uint64_t gpu_tap_elements(const Plan& plan) {
    uint64_t taps = 0;
    for (const Pass& pass : plan.passes) taps = saturating_add(taps, pass.size);
    return saturating_mul(plan.shape.elements, taps);
}

// The router's work figure: elements x CPU operations over all passes, or 0
// (never worth the GPU) when the GPU would have to do disproportionately more.
uint64_t routing_work(const Plan& plan) {
    uint64_t operations = 0;
    for (const Pass& pass : plan.passes) operations = saturating_add(operations, cpu_operations(pass));
    const uint64_t work = saturating_mul(plan.shape.elements, operations);
    if (gpu_tap_elements(plan) / gpu_taps_per_cpu_operation > work) return 0;
    return work;
}
uint64_t composite_tap_elements(const CompositePlan& plan) {
    uint64_t taps = 0;
    for (const auto& chain : plan.chains) {
        for (const Pass& pass : chain) taps = saturating_add(taps, pass.size);
    }
    return saturating_mul(plan.shape.elements, taps);
}

uint64_t composite_routing_work(const CompositePlan& plan) {
    uint64_t operations = 0;
    for (const auto& chain : plan.chains) {
        for (const Pass& pass : chain) operations = saturating_add(operations, cpu_operations(pass));
    }
    if (plan.chains.size() > 1 || plan.combine_op == MS_NDIMAGE_COMBINE_SQRT_SUM_SQ) {
        operations = saturating_add(operations, 1);
    }
    const uint64_t work = saturating_mul(plan.shape.elements, operations);
    if (composite_tap_elements(plan) / gpu_taps_per_cpu_operation > work) return 0;
    return work;
}

struct SeparableDecomposition {
    bool ok = false;
    std::vector<uint32_t> axes;
    std::vector<std::vector<double>> vectors;
    std::vector<int64_t> origins;
    std::vector<double> cvals;
};

SeparableDecomposition try_decompose_rank1(const array::Shape& shape, const ms_ndimage_dense_options& opts) {
    SeparableDecomposition result;
    const uint32_t ndim = shape.ndim;
    if (ndim < 2) return result;
    if (opts.mode == MS_NDIMAGE_CONSTANT && !std::isfinite(opts.cval)) return result;

    uint32_t non_trivial = 0;
    uint64_t total_w = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
        if (opts.wshape[d] > 1) ++non_trivial;
        if (ms_checked_mul(total_w, opts.wshape[d], &total_w) != MS_OK || total_w > (uint64_t{1} << 20)) {
            return result;
        }
    }
    if (non_trivial < 2) return result;

    // Locate largest-magnitude weight as pivot. Reject non-finite or sub-DBL_EPSILON
    // entries so separable passes never diverge from dense_correlate's prune_zeros=true
    // behavior (e.g. 0.0 * NaN = NaN).
    uint64_t pivot_idx = 0;
    double abs_max = 0.0;
    for (uint64_t i = 0; i < total_w; ++i) {
        const double w = opts.weights[i];
        if (!std::isfinite(w)) return result;
        const double aw = std::fabs(w);
        if (aw <= DBL_EPSILON) return result;
        if (aw > abs_max) {
            abs_max = aw;
            pivot_idx = i;
        }
    }
    if (abs_max <= DBL_EPSILON) return result;

    uint64_t strides[MS_MAX_NDIM] = {};
    strides[ndim - 1] = 1;
    for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; --d) {
        strides[d] = strides[d + 1] * opts.wshape[d + 1];
    }

    uint32_t pivot_coord[MS_MAX_NDIM] = {};
    {
        uint64_t rem = pivot_idx;
        for (uint32_t d = 0; d < ndim; ++d) {
            pivot_coord[d] = static_cast<uint32_t>(rem / strides[d]);
            rem %= strides[d];
        }
    }

    const double pivot_val = opts.weights[pivot_idx];
    std::vector<std::vector<double>> vecs(ndim);
    for (uint32_t d = 0; d < ndim; ++d) {
        vecs[d].resize(opts.wshape[d]);
        const uint64_t base = pivot_idx - static_cast<uint64_t>(pivot_coord[d]) * strides[d];
        for (uint32_t j = 0; j < opts.wshape[d]; ++j) {
            const double val = opts.weights[base + static_cast<uint64_t>(j) * strides[d]];
            const double v = (d == 0) ? val : (val / pivot_val);
            if (!std::isfinite(v) || std::fabs(v) <= DBL_EPSILON) return result;
            vecs[d][j] = v;
        }
    }

    // Verify rank-1 outer product individually per entry.
    uint32_t coord[MS_MAX_NDIM] = {};
    for (uint64_t i = 0; i < total_w; ++i) {
        double prod = 1.0;
        for (uint32_t d = 0; d < ndim; ++d) prod *= vecs[d][coord[d]];
        const double w = opts.weights[i];
        const double aw = std::fabs(w);
        const double ap = std::fabs(prod);
        if (!std::isfinite(w) || !std::isfinite(prod)) return result;
        if ((aw > DBL_EPSILON && prod == 0.0) || (aw <= DBL_EPSILON && ap > DBL_EPSILON)) return result;
        if (std::fabs(w - prod) > 64.0 * DBL_EPSILON * std::max(aw, ap)) return result;
        for (int32_t d = static_cast<int32_t>(ndim) - 1; d >= 0; --d) {
            if (++coord[d] < opts.wshape[d]) break;
            coord[d] = 0;
        }
    }

    // Fold any wshape[d] == 1 axes' scalar factor into the first active axis.
    double unit_scale = 1.0;
    for (uint32_t d = 0; d < ndim; ++d) {
        if (opts.wshape[d] == 1) {
            unit_scale *= vecs[d][0];
        } else {
            result.axes.push_back(d);
            result.vectors.push_back(std::move(vecs[d]));
            result.origins.push_back(opts.origins[d]);
        }
    }
    for (double& w : result.vectors[0]) {
        w *= unit_scale;
        if (!std::isfinite(w) || std::fabs(w) <= DBL_EPSILON || static_cast<float>(w) == 0.0f) {
            return SeparableDecomposition{};
        }
    }

    // Adjust constant cval per pass: cval_k = cval * prod_{m < k} sum(v_m).
    result.cvals.resize(result.axes.size(), opts.cval);
    if (opts.mode == MS_NDIMAGE_CONSTANT) {
        double prefix_sum = 1.0;
        for (size_t k = 0; k < result.axes.size(); ++k) {
            result.cvals[k] = opts.cval * prefix_sum;
            if (!std::isfinite(result.cvals[k])) return SeparableDecomposition{};
            double s = 0.0;
            for (double v : result.vectors[k]) s += v;
            prefix_sum *= s;
        }
    }
    result.ok = true;
    return result;
}
} // namespace

bool gpu_supports(const Plan& plan) noexcept {
    if (plan.shape.elements > INT32_MAX) return false;
    uint64_t weights = 0;
    for (const Pass& pass : plan.passes) {
        if (pass.length > MS_NDIMAGE_GPU_MAX_EXTENT || pass.size > MS_NDIMAGE_GPU_MAX_EXTENT) return false;
        weights += pass.size;
    }
    return weights <= gpu_max_weights && gpu_tap_elements(plan) <= gpu_max_tap_elements;
}

bool gpu_supports_composite(const CompositePlan& plan) noexcept {
    if (plan.shape.elements > INT32_MAX) return false;
    uint64_t weights = 0;
    for (const auto& chain : plan.chains) {
        for (const Pass& pass : chain) {
            if (pass.length > MS_NDIMAGE_GPU_MAX_EXTENT || pass.size > MS_NDIMAGE_GPU_MAX_EXTENT) return false;
            weights += pass.size;
        }
    }
    return weights <= gpu_max_weights && composite_tap_elements(plan) <= gpu_max_tap_elements;
}
} // namespace ms::detail::ndimage

using ms::detail::Error;

ms_status ms_ndimage_separable_filter_to(ms_context* context, const ms_array* input, const ms_ndimage_pass* passes,
                                         uint32_t count, uint32_t out_dtype,
                                         const ms_route_options* options,
                                         ms_array_result* out, ms_route_info* info) {
    namespace array = ms::detail::array;
    namespace ndimage = ms::detail::ndimage;
    namespace router = ms::detail::router;
    return ms::detail::api(context, [&] {
        if (!out) throw Error(MS_EINVAL, "null output");
        array::check_route_info(info);
        array::check_options(options);
        const ndimage::Plan plan = ndimage::make_plan(input, passes, count, out_dtype);

        ms_route_options route = *options;
        if (!ndimage::gpu_supports(plan)) {
            if (route.execution == MS_EXECUTION_GPU)
                throw Error(MS_ENOTSUP, "the array or a filter window is too large for the GPU engine");
            route.execution = MS_EXECUTION_CPU;
        }
        // Strict means SciPy's result. SciPy accumulates float32 arrays in
        // double too, which only the CPU engine reproduces, so here strict
        // keeps every dtype on the CPU unless the caller forces the GPU (the
        // router alone would let float32 arrays cross over).
        if (route.precision == MS_PRECISION_STRICT && route.execution == MS_EXECUTION_AUTO)
            route.execution = MS_EXECUTION_CPU;
        const router::Decision decision =
            router::decide(context, ndimage::family, route, plan.dtype, ndimage::routing_work(plan));

        ms::detail::BufferPtr result;
        // The CPU engine accumulates in double whatever the array's dtype.
        uint32_t engine = plan.shape.elements ? decision.engine : uint32_t{MS_EXECUTION_CPU};
        uint32_t compute_dtype = MS_FLOAT64;
        std::string reason;
        if (!plan.shape.elements) {
            result = array::allocate(context, 0);
        } else if (decision.engine == MS_EXECUTION_GPU) {
#if defined(MS_ENABLE_METAL)
            const bool allow_fallback = options->execution == MS_EXECUTION_AUTO;
            result = ndimage::run_gpu(context, plan, *input, allow_fallback, reason);
            if (result) {
                compute_dtype = MS_FLOAT32;
            } else {
                engine = MS_EXECUTION_CPU;
                result = ndimage::run_cpu(context, plan, input->data);
            }
#else
            throw Error(MS_ENOTSUP, "GPU execution needs a Metal build");
#endif
        } else {
            result = ndimage::run_cpu(context, plan, input->data);
        }
        array::publish(out, std::move(result), plan.out_dtype, plan.shape);
        array::report(info, engine, compute_dtype, reason.c_str());
    });
}

ms_status ms_ndimage_separable_filter(ms_context* context, const ms_array* input, const ms_ndimage_pass* passes,
                                      uint32_t count, const ms_route_options* options,
                                      ms_array_result* out, ms_route_info* info) {
    return ms_ndimage_separable_filter_to(context, input, passes, count, 0, options, out, info);
}

ms_status ms_ndimage_dense_filter(ms_context* context, const ms_array* input,
                                  const ms_ndimage_dense_options* dense_options,
                                  const ms_route_options* options,
                                  ms_array_result* out, ms_route_info* info) {
    namespace array = ms::detail::array;
    namespace dense = ms::detail::dense_correlate;
    namespace ndimage = ms::detail::ndimage;
    namespace router = ms::detail::router;
    return ms::detail::api(context, [&] {
        if (!out) throw Error(MS_EINVAL, "null output");
        array::check_route_info(info);
        array::check_options(options);
        if (!dense_options) throw Error(MS_EINVAL, "null dense filter options");
        if (dense_options->struct_size != sizeof(ms_ndimage_dense_options) ||
            dense_options->abi_version != MS_NDIMAGE_ABI_VERSION)
            throw Error(MS_EINVAL, "dense filter options size or ABI version mismatch");
        if (dense_options->reserved[0] || dense_options->reserved[1])
            throw Error(MS_EINVAL, "dense filter options reserved fields must be zero");
        if (dense_options->mode > MS_NDIMAGE_CONSTANT) throw Error(MS_EINVAL, "unknown boundary mode");

        const array::Shape shape = array::checked_shape(input);
        if (input->dtype != MS_FLOAT32 && input->dtype != MS_FLOAT64)
            throw Error(MS_ENOTSUP, "ndimage filters take float32 or float64 arrays");
        if (reinterpret_cast<uintptr_t>(input->data) % array::dtype_size(input->dtype))
            throw Error(MS_EINVAL, "array data is not aligned to its element size");
        const uint32_t out_dtype = ndimage::resolve_out_dtype(input->dtype, dense_options->out_dtype);
        if (!dense_options->weights) throw Error(MS_EINVAL, "null filter weights");

        uint64_t total_w = 1;
        for (uint32_t d = 0; d < shape.ndim; ++d) {
            if (dense_options->wshape[d] < 1 || dense_options->wshape[d] > MS_NDIMAGE_MAX_WINDOW)
                throw Error(MS_EINVAL, "filter weight extent must be 1 to 2^30");
            const int64_t size = static_cast<int64_t>(dense_options->wshape[d]);
            if (dense_options->origins[d] < -(size / 2) || dense_options->origins[d] > (size - 1) / 2)
                throw Error(MS_EINVAL, "filter origin must satisfy -(size / 2) <= origin <= (size - 1) / 2");
            if (ms_checked_mul(total_w, dense_options->wshape[d], &total_w) != MS_OK)
                throw Error(MS_EOVERFLOW, "filter weight count overflows");
        }

        if (!shape.elements) {
            ms::detail::BufferPtr result = array::allocate(context, 0);
            array::publish(out, std::move(result), out_dtype, shape);
            array::report(info, MS_EXECUTION_CPU, MS_FLOAT64);
            return;
        }

        dense::StencilSpec spec;
        spec.in_shape = shape;
        spec.out_shape = shape;
        spec.in_dtype = input->dtype;
        spec.out_dtype = out_dtype;
        spec.set_mode(dense_options->mode);
        spec.cval_real = dense_options->cval;
        spec.cpu_accum = dense::AccumMode::Double;
        spec.prune_zeros = true;
        spec.weights_dtype = MS_FLOAT64;
        spec.weights = dense_options->weights;
        for (uint32_t d = 0; d < shape.ndim; ++d) {
            spec.wshape[d] = dense_options->wshape[d];
            spec.start[d] = -(static_cast<int64_t>(dense_options->wshape[d]) / 2) - dense_options->origins[d];
        }

        // In FAST / REFINED (never STRICT), try rank-1 separable decomposition when routing to GPU.
        const bool try_separable = options->precision != MS_PRECISION_STRICT &&
                                   options->execution != MS_EXECUTION_CPU &&
                                   shape.ndim >= 2;
        if (try_separable) {
            const ndimage::SeparableDecomposition decomp = ndimage::try_decompose_rank1(shape, *dense_options);
            if (decomp.ok) {
                ndimage::Plan sep_plan;
                sep_plan.shape = shape;
                sep_plan.dtype = input->dtype;
                sep_plan.out_dtype = out_dtype;
                std::vector<ms_ndimage_pass> raw_passes(decomp.axes.size());
                for (size_t k = 0; k < decomp.axes.size(); ++k) {
                    ms_ndimage_pass& p = raw_passes[k];
                    p.struct_size = sizeof(p);
                    p.abi_version = MS_NDIMAGE_ABI_VERSION;
                    p.kind = MS_NDIMAGE_CORRELATE;
                    p.axis = decomp.axes[k];
                    p.mode = dense_options->mode;
                    p.origin = decomp.origins[k];
                    p.size = decomp.vectors[k].size();
                    p.weights = decomp.vectors[k].data();
                    p.cval = decomp.cvals[k];
                    sep_plan.passes.push_back(ndimage::parse_pass(shape, p));
                }
                if (ndimage::gpu_supports(sep_plan)) {
                    uint64_t dense_taps = 0;
                    for (uint64_t i = 0; i < total_w; ++i) {
                        if (std::fabs(dense_options->weights[i]) > DBL_EPSILON) ++dense_taps;
                    }
                    uint64_t dense_work = 0;
                    if (ms_checked_mul(shape.elements, std::max<uint64_t>(1, dense_taps), &dense_work) != MS_OK)
                        dense_work = UINT64_MAX;
                    const router::Decision decision =
                        router::decide(context, ndimage::dense_family, *options, input->dtype, dense_work);
                    if (decision.engine == MS_EXECUTION_GPU) {
#if defined(MS_ENABLE_METAL)
                        const bool allow_fallback = options->execution == MS_EXECUTION_AUTO;
                        std::string reason;
                        ms::detail::BufferPtr result =
                            ndimage::run_gpu(context, sep_plan, *input, allow_fallback, reason);
                        if (result) {
                            array::publish(out, std::move(result), out_dtype, shape);
                            array::report(info, MS_EXECUTION_GPU, MS_FLOAT32, reason.c_str());
                            return;
                        }
                        // Overflow in AUTO: fall through to exact dense CPU engine so result matches SciPy.
                        const dense::TapTable taps = dense::build_taps(spec);
                        result = dense::run_cpu(context, spec, taps, input->data);
                        array::publish(out, std::move(result), out_dtype, shape);
                        array::report(info, MS_EXECUTION_CPU, MS_FLOAT64, reason.c_str());
                        return;
#else
                        throw Error(MS_ENOTSUP, "GPU execution needs a Metal build");
#endif
                    }
                }
            }
        }

        const dense::TapTable taps = dense::build_taps(spec);
        ms_route_options route = *options;
        if (!dense::gpu_supports(spec, taps)) {
            if (route.execution == MS_EXECUTION_GPU)
                throw Error(MS_ENOTSUP, "the array or filter kernel is too large for the GPU engine");
            route.execution = MS_EXECUTION_CPU;
        }
        if (route.precision == MS_PRECISION_STRICT && route.execution == MS_EXECUTION_AUTO)
            route.execution = MS_EXECUTION_CPU;

        uint64_t work = 0;
        if (ms_checked_mul(shape.elements, std::max<uint64_t>(1, taps.taps.size()), &work) != MS_OK)
            work = UINT64_MAX;
        const router::Decision decision =
            router::decide(context, ndimage::dense_family, route, input->dtype, work);

        ms::detail::BufferPtr result;
        uint32_t engine = decision.engine;
        uint32_t compute_dtype = MS_FLOAT64;
        std::string reason;
        if (decision.engine == MS_EXECUTION_GPU) {
#if defined(MS_ENABLE_METAL)
            const bool allow_fallback = options->execution == MS_EXECUTION_AUTO;
            result = dense::run_gpu(context, spec, taps, *input, allow_fallback, reason);
            if (result) {
                compute_dtype = MS_FLOAT32;
            } else {
                engine = MS_EXECUTION_CPU;
                result = dense::run_cpu(context, spec, taps, input->data);
            }
#else
            throw Error(MS_ENOTSUP, "GPU execution needs a Metal build");
#endif
        } else {
            result = dense::run_cpu(context, spec, taps, input->data);
        }
        array::publish(out, std::move(result), out_dtype, shape);
        array::report(info, engine, compute_dtype, reason.c_str());
    });
}

ms_status ms_ndimage_composite_filter(ms_context* context, const ms_array* input,
                                      const ms_ndimage_chain* chains, uint32_t chain_count,
                                      uint32_t combine_op, uint32_t out_dtype,
                                      const ms_route_options* options,
                                      ms_array_result* out, ms_route_info* info) {
    namespace array = ms::detail::array;
    namespace ndimage = ms::detail::ndimage;
    namespace router = ms::detail::router;
    return ms::detail::api(context, [&] {
        if (!out) throw Error(MS_EINVAL, "null output");
        array::check_route_info(info);
        array::check_options(options);
        const ndimage::CompositePlan plan =
            ndimage::make_composite_plan(input, chains, chain_count, combine_op, out_dtype);

        ms_route_options route = *options;
        if (!ndimage::gpu_supports_composite(plan)) {
            if (route.execution == MS_EXECUTION_GPU)
                throw Error(MS_ENOTSUP, "the array or a filter window is too large for the GPU engine");
            route.execution = MS_EXECUTION_CPU;
        }
        if (route.precision == MS_PRECISION_STRICT && route.execution == MS_EXECUTION_AUTO)
            route.execution = MS_EXECUTION_CPU;
        const router::Decision decision =
            router::decide(context, ndimage::family, route, plan.in_dtype, ndimage::composite_routing_work(plan));

        ms::detail::BufferPtr result;
        uint32_t engine = plan.shape.elements ? decision.engine : uint32_t{MS_EXECUTION_CPU};
        uint32_t compute_dtype = MS_FLOAT64;
        std::string reason;
        if (!plan.shape.elements) {
            result = array::allocate(context, 0);
        } else if (decision.engine == MS_EXECUTION_GPU) {
#if defined(MS_ENABLE_METAL)
            const bool allow_fallback = options->execution == MS_EXECUTION_AUTO;
            result = ndimage::run_gpu_composite(context, plan, *input, allow_fallback, reason);
            if (result) {
                compute_dtype = MS_FLOAT32;
            } else {
                engine = MS_EXECUTION_CPU;
                result = ndimage::run_cpu_composite(context, plan, input->data);
            }
#else
            throw Error(MS_ENOTSUP, "GPU execution needs a Metal build");
#endif
        } else {
            result = ndimage::run_cpu_composite(context, plan, input->data);
        }
        array::publish(out, std::move(result), plan.out_dtype, plan.shape);
        array::report(info, engine, compute_dtype, reason.c_str());
    });
}
