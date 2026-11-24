// scipy.ndimage separable, dense, and composite filters: argument validation, routing and the C ABI.
#include "filters.hpp"
#include "core/dense_correlate.hpp"
#include "core/router.hpp"
#include "ndimage_kernels.h"

#include <cfloat>
#include <cmath>
#include <vector>

namespace sm::detail::ndimage {
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

Pass parse_pass(const array::Shape& shape, const sm_ndimage_pass& given) {
    if (given.struct_size != sizeof(sm_ndimage_pass) || given.abi_version != SM_NDIMAGE_ABI_VERSION)
        throw Error(SM_EINVAL, "filter pass size or ABI version mismatch");
    if (given.reserved0 || given.reserved[0] || given.reserved[1])
        throw Error(SM_EINVAL, "filter pass reserved fields must be zero");
    if (given.kind != SM_NDIMAGE_CORRELATE && given.kind != SM_NDIMAGE_UNIFORM &&
        given.kind != SM_NDIMAGE_MINIMUM && given.kind != SM_NDIMAGE_MAXIMUM)
        throw Error(SM_EINVAL, "unknown filter pass kind");
    if (given.axis >= shape.ndim) throw Error(SM_EINVAL, "filter axis is out of range");
    if (given.mode > SM_NDIMAGE_CONSTANT) throw Error(SM_EINVAL, "unknown boundary mode");
    if (given.size < 1 || given.size > SM_NDIMAGE_MAX_WINDOW)
        throw Error(SM_EINVAL, "filter size must be 1 to 2^30");
    const int64_t size = static_cast<int64_t>(given.size);
    if (given.origin < -(size / 2) || given.origin > (size - 1) / 2)
        throw Error(SM_EINVAL, "filter origin must satisfy -(size / 2) <= origin <= (size - 1) / 2");
    if (given.kind == SM_NDIMAGE_CORRELATE) {
        if (!given.weights) throw Error(SM_EINVAL, "null filter weights");
        if (reinterpret_cast<uintptr_t>(given.weights) % alignof(double) != 0)
            throw Error(SM_EINVAL, "filter weights are not aligned for double");
    }
    if (given.kind != SM_NDIMAGE_CORRELATE && given.weights)
        throw Error(SM_EINVAL, "a uniform or min/max filter pass takes no weights");

    Pass pass;
    pass.kind = given.kind;
    pass.mode = given.mode;
    pass.cval = given.cval;
    pass.size = given.size;
    pass.weights = given.weights;
    // A uniform window is symmetric when it has a centre; the GPU engine sums it as weights 1/size.
    pass.symmetry = given.kind == SM_NDIMAGE_UNIFORM
                        ? (given.size & 1 ? Symmetry::even : Symmetry::none)
                        : (given.kind == SM_NDIMAGE_CORRELATE ? classify(given.weights, given.size) : Symmetry::none);
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
    (void)array::dtype_size(requested_out_dtype);
    if (!array::is_real(requested_out_dtype))
        throw Error(SM_ENOTSUP, "ndimage filters require a real output dtype");
    return requested_out_dtype;
}

void check_output_bytes(const array::Shape& shape, uint32_t out_dtype) {
    uint64_t out_bytes = 0;
    if (sm_checked_mul(shape.elements, array::dtype_size(out_dtype), &out_bytes) != SM_OK)
        throw Error(SM_EOVERFLOW, "output array byte size overflows");
}

Plan make_plan(const sm_array* input, const sm_ndimage_pass* passes, uint32_t count, uint32_t out_dtype = 0) {
    Plan plan;
    plan.shape = array::checked_shape(input);
    if (!array::is_real(input->dtype))
        throw Error(SM_ENOTSUP, "ndimage filters take real arrays (float, integer, or bool)");
    plan.dtype = input->dtype;
    plan.out_dtype = resolve_out_dtype(input->dtype, out_dtype);
    check_output_bytes(plan.shape, plan.out_dtype);
    if (plan.shape.elements && reinterpret_cast<uintptr_t>(input->data) % array::dtype_size(input->dtype))
        throw Error(SM_EINVAL, "array data is not aligned to its element size");
    if (!passes) throw Error(SM_EINVAL, "null filter passes");
    if (count < 1 || count > SM_NDIMAGE_MAX_PASSES) throw Error(SM_EINVAL, "filter pass count must be 1 to 32");
    plan.passes.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        plan.passes.push_back(parse_pass(plan.shape, passes[i]));
    }
    return plan;
}

CompositePlan make_composite_plan(const sm_array* input, const sm_ndimage_chain* chains, uint32_t chain_count,
                                    uint32_t combine_op, uint32_t out_dtype) {
    CompositePlan plan;
    plan.shape = array::checked_shape(input);
    if (!array::is_real(input->dtype))
        throw Error(SM_ENOTSUP, "ndimage filters take real arrays (float, integer, or bool)");
    if (plan.shape.elements && reinterpret_cast<uintptr_t>(input->data) % array::dtype_size(input->dtype))
        throw Error(SM_EINVAL, "array data is not aligned to its element size");
    plan.in_dtype = input->dtype;
    plan.out_dtype = resolve_out_dtype(input->dtype, out_dtype);
    check_output_bytes(plan.shape, plan.out_dtype);
    if (combine_op != SM_NDIMAGE_COMBINE_SUM && combine_op != SM_NDIMAGE_COMBINE_SQRT_SUM_SQ &&
        combine_op != SM_NDIMAGE_COMBINE_SUB && combine_op != SM_NDIMAGE_COMBINE_MORPH_LAPLACE &&
        combine_op != SM_NDIMAGE_COMBINE_XOR)
        throw Error(SM_EINVAL, "unknown composite combine op");
    if (!chains) throw Error(SM_EINVAL, "null filter chains");
    if (chain_count < 1 || chain_count > SM_MAX_NDIM)
        throw Error(SM_EINVAL, "composite filter chain count must be 1 to SM_MAX_NDIM");
    if ((combine_op == SM_NDIMAGE_COMBINE_SUB || combine_op == SM_NDIMAGE_COMBINE_MORPH_LAPLACE ||
         combine_op == SM_NDIMAGE_COMBINE_XOR) &&
        chain_count != 2)
        throw Error(SM_EINVAL, "binary composite combine op requires exactly 2 chains");
    if ((combine_op == SM_NDIMAGE_COMBINE_SUB || combine_op == SM_NDIMAGE_COMBINE_MORPH_LAPLACE) &&
        plan.out_dtype == SM_BOOL)
        throw Error(SM_ENOTSUP, "subtraction composite op is not supported for boolean output");
    if (combine_op == SM_NDIMAGE_COMBINE_XOR && plan.out_dtype != SM_BOOL)
        throw Error(SM_EINVAL, "xor composite op requires boolean output");
    plan.combine_op = combine_op;
    plan.chains.resize(chain_count);
    for (uint32_t k = 0; k < chain_count; ++k) {
        const sm_ndimage_chain& c = chains[k];
        if (c.struct_size != sizeof(sm_ndimage_chain) || c.abi_version != SM_NDIMAGE_ABI_VERSION)
            throw Error(SM_EINVAL, "filter chain size or ABI version mismatch");
        if (c.reserved0 || c.reserved[0] || c.reserved[1])
            throw Error(SM_EINVAL, "filter chain reserved fields must be zero");
        if (c.pass_count > SM_NDIMAGE_MAX_PASSES)
            throw Error(SM_EINVAL, "filter chain pass count must be 0 to 32");
        if (c.pass_count == 0 && c.passes != nullptr)
            throw Error(SM_EINVAL, "empty filter chain must have null passes");
        if (c.pass_count > 0 && !c.passes)
            throw Error(SM_EINVAL, "non-empty filter chain has null passes");
        plan.chains[k].reserve(c.pass_count);
        for (uint32_t i = 0; i < c.pass_count; ++i) {
            plan.chains[k].push_back(parse_pass(plan.shape, c.passes[i]));
        }
    }
    return plan;
}

// The CPU engine's cost of one pass per element, in tap operations.
uint64_t cpu_operations(const Pass& pass) {
    uint64_t arithmetic = 0;
    if (gpu_minmax_use_vanherk(pass)) {
        // Two-pass GPU van Herk has twice the memory traffic of a direct pass, so it
        // needs a larger image (~2048^2 vs ~1448^2) to beat threaded CPU van Herk.
        arithmetic = uniform_operations / 2;
    } else if (pass.kind == SM_NDIMAGE_UNIFORM || pass.kind == SM_NDIMAGE_MINIMUM ||
               pass.kind == SM_NDIMAGE_MAXIMUM) {
        arithmetic = uniform_operations;
    } else {
        arithmetic = (pass.symmetry == Symmetry::none ? pass.size : pass.size / 2 + 1);
    }
    return arithmetic + (pass.inner == 1 ? 0 : strided_pass_operations);
}

uint64_t saturating_mul(uint64_t a, uint64_t b) {
    uint64_t product = 0;
    return sm_checked_mul(a, b, &product) == SM_OK ? product : UINT64_MAX;
}
uint64_t saturating_add(uint64_t a, uint64_t b) {
    uint64_t sum = 0;
    return sm_checked_add(a, b, &sum) == SM_OK ? sum : UINT64_MAX;
}

uint64_t effective_gpu_taps(const Pass& pass) {
    if (gpu_minmax_use_vanherk(pass)) return 12;
    return pass.size;
}

uint64_t gpu_tap_elements(const Plan& plan) {
    uint64_t taps = 0;
    for (const Pass& pass : plan.passes) taps = saturating_add(taps, effective_gpu_taps(pass));
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
        for (const Pass& pass : chain) taps = saturating_add(taps, effective_gpu_taps(pass));
    }
    return saturating_mul(plan.shape.elements, taps);
}

uint64_t composite_routing_work(const CompositePlan& plan) {
    uint64_t operations = 0;
    for (const auto& chain : plan.chains) {
        for (const Pass& pass : chain) operations = saturating_add(operations, cpu_operations(pass));
    }
    if (plan.chains.size() > 1 || plan.combine_op == SM_NDIMAGE_COMBINE_SQRT_SUM_SQ) {
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

SeparableDecomposition try_decompose_rank1(const array::Shape& shape, const sm_ndimage_dense_options& opts) {
    SeparableDecomposition result;
    const uint32_t ndim = shape.ndim;
    if (ndim < 2) return result;
    if (opts.mode == SM_NDIMAGE_CONSTANT && !std::isfinite(opts.cval)) return result;

    uint32_t non_trivial = 0;
    uint64_t total_w = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
        if (opts.wshape[d] > 1) ++non_trivial;
        if (sm_checked_mul(total_w, opts.wshape[d], &total_w) != SM_OK || total_w > (uint64_t{1} << 20)) {
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

    uint64_t strides[SM_MAX_NDIM] = {};
    strides[ndim - 1] = 1;
    for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; --d) {
        strides[d] = strides[d + 1] * opts.wshape[d + 1];
    }

    uint32_t pivot_coord[SM_MAX_NDIM] = {};
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
    uint32_t coord[SM_MAX_NDIM] = {};
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
    if (opts.mode == SM_NDIMAGE_CONSTANT) {
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
        if (pass.length > SM_NDIMAGE_GPU_MAX_EXTENT || pass.size > SM_NDIMAGE_GPU_MAX_EXTENT) return false;
        if (pass.kind == SM_NDIMAGE_CORRELATE || pass.kind == SM_NDIMAGE_UNIFORM) weights += pass.size;
    }
    return weights <= gpu_max_weights && gpu_tap_elements(plan) <= gpu_max_tap_elements;
}

bool gpu_supports_composite(const CompositePlan& plan) noexcept {
    if (plan.shape.elements > INT32_MAX) return false;
    // Boolean composite combine uses logical OR / XOR and canonical {0,1} semantics
    // (and 0-pass identity uses != 0), which float32 GPU kernels do not preserve.
    if (plan.out_dtype == SM_BOOL) return false;
    // Unsigned 32/64-bit subtraction/laplace can underflow modularly, which float32
    // saturating cast does not match.
    if ((plan.combine_op == SM_NDIMAGE_COMBINE_SUB || plan.combine_op == SM_NDIMAGE_COMBINE_MORPH_LAPLACE) &&
        (plan.out_dtype == SM_UINT32 || plan.out_dtype == SM_UINT64)) {
        return false;
    }
    uint64_t weights = 0;
    for (const auto& chain : plan.chains) {
        for (const Pass& pass : chain) {
            if (pass.length > SM_NDIMAGE_GPU_MAX_EXTENT || pass.size > SM_NDIMAGE_GPU_MAX_EXTENT) return false;
            if (pass.kind == SM_NDIMAGE_CORRELATE || pass.kind == SM_NDIMAGE_UNIFORM) weights += pass.size;
        }
    }
    return weights <= gpu_max_weights && composite_tap_elements(plan) <= gpu_max_tap_elements;
}
} // namespace sm::detail::ndimage

using sm::detail::Error;

sm_status sm_ndimage_separable_filter_to(sm_context* context, const sm_array* input, const sm_ndimage_pass* passes,
                                         uint32_t count, uint32_t out_dtype,
                                         const sm_route_options* options,
                                         sm_array_result* out, sm_route_info* info) {
    namespace array = sm::detail::array;
    namespace ndimage = sm::detail::ndimage;
    namespace router = sm::detail::router;
    return sm::detail::api(context, [&] {
        if (!out) throw Error(SM_EINVAL, "null output");
        array::check_route_info(info);
        array::check_options(options);
        const ndimage::Plan plan = ndimage::make_plan(input, passes, count, out_dtype);

        sm_route_options route = *options;
        if (!ndimage::gpu_supports(plan)) {
            if (route.execution == SM_EXECUTION_GPU)
                throw Error(SM_ENOTSUP, "the array or a filter window is too large for the GPU engine");
            route.execution = SM_EXECUTION_CPU;
        }
        // Strict means SciPy's result. SciPy accumulates float32 arrays in
        // double too, which only the CPU engine reproduces, so here strict
        // keeps every dtype on the CPU unless the caller forces the GPU (the
        // router alone would let float32 arrays cross over).
        if (route.precision == SM_PRECISION_STRICT && route.execution == SM_EXECUTION_AUTO)
            route.execution = SM_EXECUTION_CPU;
        if (route.precision == SM_PRECISION_STRICT && route.execution == SM_EXECUTION_GPU &&
            plan.out_dtype != SM_FLOAT32)
            throw Error(SM_ENOTSUP, "strict precision keeps non-float32 outputs on the CPU");
        const router::Decision decision =
            router::decide(context, ndimage::family, route, plan.dtype, ndimage::routing_work(plan));

        sm::detail::BufferPtr result;
        // The CPU engine accumulates in double whatever the array's dtype.
        uint32_t engine = plan.shape.elements ? decision.engine : uint32_t{SM_EXECUTION_CPU};
        uint32_t compute_dtype = SM_FLOAT64;
        std::string reason;
        if (!plan.shape.elements) {
            result = array::allocate(context, 0);
        } else if (decision.engine == SM_EXECUTION_GPU) {
#if defined(SM_ENABLE_METAL)
            const bool allow_fallback = options->execution == SM_EXECUTION_AUTO;
            result = ndimage::run_gpu(context, plan, *input, allow_fallback, reason);
            if (result) {
                compute_dtype = SM_FLOAT32;
            } else {
                engine = SM_EXECUTION_CPU;
                result = ndimage::run_cpu(context, plan, input->data);
            }
#else
            throw Error(SM_ENOTSUP, "GPU execution needs a Metal build");
#endif
        } else {
            result = ndimage::run_cpu(context, plan, input->data);
        }
        array::publish(out, std::move(result), plan.out_dtype, plan.shape);
        array::report(info, engine, compute_dtype, reason.c_str());
    });
}

sm_status sm_ndimage_separable_filter(sm_context* context, const sm_array* input, const sm_ndimage_pass* passes,
                                      uint32_t count, const sm_route_options* options,
                                      sm_array_result* out, sm_route_info* info) {
    return sm_ndimage_separable_filter_to(context, input, passes, count, 0, options, out, info);
}

sm_status sm_ndimage_dense_filter(sm_context* context, const sm_array* input,
                                  const sm_ndimage_dense_options* dense_options,
                                  const sm_route_options* options,
                                  sm_array_result* out, sm_route_info* info) {
    namespace array = sm::detail::array;
    namespace dense = sm::detail::dense_correlate;
    namespace ndimage = sm::detail::ndimage;
    namespace router = sm::detail::router;
    return sm::detail::api(context, [&] {
        if (!out) throw Error(SM_EINVAL, "null output");
        array::check_route_info(info);
        array::check_options(options);
        if (!dense_options) throw Error(SM_EINVAL, "null dense filter options");
        if (dense_options->struct_size != sizeof(sm_ndimage_dense_options) ||
            dense_options->abi_version != SM_NDIMAGE_ABI_VERSION)
            throw Error(SM_EINVAL, "dense filter options size or ABI version mismatch");
        if (dense_options->reserved[0] || dense_options->reserved[1])
            throw Error(SM_EINVAL, "dense filter options reserved fields must be zero");
        if (dense_options->mode > SM_NDIMAGE_CONSTANT) throw Error(SM_EINVAL, "unknown boundary mode");

        const array::Shape shape = array::checked_shape(input);
        if (!array::is_real(input->dtype))
            throw Error(SM_ENOTSUP, "ndimage filters take real arrays (float, integer, or bool)");
        if (shape.elements && reinterpret_cast<uintptr_t>(input->data) % array::dtype_size(input->dtype))
            throw Error(SM_EINVAL, "array data is not aligned to its element size");
        const uint32_t out_dtype = ndimage::resolve_out_dtype(input->dtype, dense_options->out_dtype);
        ndimage::check_output_bytes(shape, out_dtype);
        if (!dense_options->weights) throw Error(SM_EINVAL, "null filter weights");
        if (reinterpret_cast<uintptr_t>(dense_options->weights) % alignof(double) != 0)
            throw Error(SM_EINVAL, "filter weights are not aligned for double");

        uint64_t total_w = 1;
        for (uint32_t d = 0; d < shape.ndim; ++d) {
            if (dense_options->wshape[d] < 1 || dense_options->wshape[d] > SM_NDIMAGE_MAX_WINDOW)
                throw Error(SM_EINVAL, "filter weight extent must be 1 to 2^30");
            const int64_t size = static_cast<int64_t>(dense_options->wshape[d]);
            if (dense_options->origins[d] < -(size / 2) || dense_options->origins[d] > (size - 1) / 2)
                throw Error(SM_EINVAL, "filter origin must satisfy -(size / 2) <= origin <= (size - 1) / 2");
            if (sm_checked_mul(total_w, dense_options->wshape[d], &total_w) != SM_OK)
                throw Error(SM_EOVERFLOW, "filter weight count overflows");
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
        spec.weights_dtype = SM_FLOAT64;
        spec.weights = dense_options->weights;
        for (uint32_t d = 0; d < shape.ndim; ++d) {
            spec.wshape[d] = dense_options->wshape[d];
            spec.start[d] = -(static_cast<int64_t>(dense_options->wshape[d]) / 2) - dense_options->origins[d];
        }

        // In FAST / REFINED (never STRICT), try rank-1 separable decomposition when routing to GPU.
        const bool try_separable = shape.elements > 0 &&
                                   options->precision != SM_PRECISION_STRICT &&
                                   options->execution != SM_EXECUTION_CPU &&
                                   shape.ndim >= 2;
        if (try_separable) {
            const ndimage::SeparableDecomposition decomp = ndimage::try_decompose_rank1(shape, *dense_options);
            if (decomp.ok) {
                ndimage::Plan sep_plan;
                sep_plan.shape = shape;
                sep_plan.dtype = input->dtype;
                sep_plan.out_dtype = out_dtype;
                std::vector<sm_ndimage_pass> raw_passes(decomp.axes.size());
                for (size_t k = 0; k < decomp.axes.size(); ++k) {
                    sm_ndimage_pass& p = raw_passes[k];
                    p.struct_size = sizeof(p);
                    p.abi_version = SM_NDIMAGE_ABI_VERSION;
                    p.kind = SM_NDIMAGE_CORRELATE;
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
                    if (sm_checked_mul(shape.elements, std::max<uint64_t>(1, dense_taps), &dense_work) != SM_OK)
                        dense_work = UINT64_MAX;
                    const router::Decision decision =
                        router::decide(context, ndimage::dense_family, *options, input->dtype, dense_work);
                    if (decision.engine == SM_EXECUTION_GPU) {
#if defined(SM_ENABLE_METAL)
                        std::string reason;
                        // Allow the optional rank-1 separable attempt to decline
                        // cleanly into the dense GPU/CPU path if its split factors
                        // or symmetric pair sums overflow float32.
                        sm::detail::BufferPtr result =
                            ndimage::run_gpu(context, sep_plan, *input, /*allow_cpu_fallback=*/true, reason,
                                             /*dense_safe=*/true);
                        if (result) {
                            array::publish(out, std::move(result), out_dtype, shape);
                            array::report(info, SM_EXECUTION_GPU, SM_FLOAT32, reason.c_str());
                            return;
                        }
#else
                        throw Error(SM_ENOTSUP, "GPU execution needs a Metal build");
#endif
                    }
                }
            }
        }

        const dense::TapTable taps = dense::build_taps(spec);
        sm_route_options route = *options;
        if (!dense::gpu_supports(spec, taps)) {
            if (route.execution == SM_EXECUTION_GPU)
                throw Error(SM_ENOTSUP, "the array or filter kernel is too large for the GPU engine");
            route.execution = SM_EXECUTION_CPU;
        }
        if (route.precision == SM_PRECISION_STRICT && route.execution == SM_EXECUTION_AUTO)
            route.execution = SM_EXECUTION_CPU;
        if (route.precision == SM_PRECISION_STRICT && route.execution == SM_EXECUTION_GPU &&
            out_dtype != SM_FLOAT32)
            throw Error(SM_ENOTSUP, "strict precision keeps non-float32 outputs on the CPU");

        uint64_t work = 0;
        if (sm_checked_mul(shape.elements, std::max<uint64_t>(1, taps.taps.size()), &work) != SM_OK)
            work = UINT64_MAX;
        const router::Decision decision =
            router::decide(context, ndimage::dense_family, route, input->dtype, work);

        sm::detail::BufferPtr result;
        uint32_t engine = shape.elements ? decision.engine : uint32_t{SM_EXECUTION_CPU};
        uint32_t compute_dtype = SM_FLOAT64;
        std::string reason;
        if (!shape.elements) {
            result = array::allocate(context, 0);
        } else if (decision.engine == SM_EXECUTION_GPU) {
#if defined(SM_ENABLE_METAL)
            const bool allow_fallback = options->execution == SM_EXECUTION_AUTO;
            result = dense::run_gpu(context, spec, taps, *input, allow_fallback, reason);
            if (result) {
                compute_dtype = SM_FLOAT32;
            } else {
                engine = SM_EXECUTION_CPU;
                result = dense::run_cpu(context, spec, taps, input->data);
            }
#else
            throw Error(SM_ENOTSUP, "GPU execution needs a Metal build");
#endif
        } else {
            result = dense::run_cpu(context, spec, taps, input->data);
        }
        array::publish(out, std::move(result), out_dtype, shape);
        array::report(info, engine, compute_dtype, reason.c_str());
    });
}

sm_status sm_ndimage_composite_filter(sm_context* context, const sm_array* input,
                                      const sm_ndimage_chain* chains, uint32_t chain_count,
                                      uint32_t combine_op, uint32_t out_dtype,
                                      const sm_route_options* options,
                                      sm_array_result* out, sm_route_info* info) {
    namespace array = sm::detail::array;
    namespace ndimage = sm::detail::ndimage;
    namespace router = sm::detail::router;
    return sm::detail::api(context, [&] {
        if (!out) throw Error(SM_EINVAL, "null output");
        array::check_route_info(info);
        array::check_options(options);
        const ndimage::CompositePlan plan =
            ndimage::make_composite_plan(input, chains, chain_count, combine_op, out_dtype);

        sm_route_options route = *options;
        if (!ndimage::gpu_supports_composite(plan)) {
            if (route.execution == SM_EXECUTION_GPU)
                throw Error(SM_ENOTSUP, "the array or a filter window is too large for the GPU engine");
            route.execution = SM_EXECUTION_CPU;
        }
        if (route.precision == SM_PRECISION_STRICT && route.execution == SM_EXECUTION_AUTO)
            route.execution = SM_EXECUTION_CPU;
        if (route.precision == SM_PRECISION_STRICT && route.execution == SM_EXECUTION_GPU &&
            plan.out_dtype != SM_FLOAT32)
            throw Error(SM_ENOTSUP, "strict precision keeps non-float32 outputs on the CPU");
        const router::Decision decision =
            router::decide(context, ndimage::family, route, plan.in_dtype, ndimage::composite_routing_work(plan));

        sm::detail::BufferPtr result;
        uint32_t engine = plan.shape.elements ? decision.engine : uint32_t{SM_EXECUTION_CPU};
        uint32_t compute_dtype = SM_FLOAT64;
        std::string reason;
        if (!plan.shape.elements) {
            result = array::allocate(context, 0);
        } else if (decision.engine == SM_EXECUTION_GPU) {
#if defined(SM_ENABLE_METAL)
            const bool allow_fallback = options->execution == SM_EXECUTION_AUTO;
            result = ndimage::run_gpu_composite(context, plan, *input, allow_fallback, reason);
            if (result) {
                compute_dtype = SM_FLOAT32;
            } else {
                engine = SM_EXECUTION_CPU;
                result = ndimage::run_cpu_composite(context, plan, input->data);
            }
#else
            throw Error(SM_ENOTSUP, "GPU execution needs a Metal build");
#endif
        } else {
            result = ndimage::run_cpu_composite(context, plan, input->data);
        }
        array::publish(out, std::move(result), plan.out_dtype, plan.shape);
        array::report(info, engine, compute_dtype, reason.c_str());
    });
}
