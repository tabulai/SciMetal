// scipy.ndimage separable filters: argument validation, routing and the C ABI.
#include "filters.hpp"
#include "core/router.hpp"
#include "ndimage_kernels.h"

#include <cfloat>
#include <cmath>

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

Plan make_plan(const ms_array* input, const ms_ndimage_pass* passes, uint32_t count) {
    Plan plan;
    plan.shape = array::checked_shape(input);
    if (input->dtype != MS_FLOAT32 && input->dtype != MS_FLOAT64)
        throw Error(MS_ENOTSUP, "ndimage filters take float32 or float64 arrays");
    plan.dtype = input->dtype;
    if (reinterpret_cast<uintptr_t>(input->data) % array::dtype_size(input->dtype))
        throw Error(MS_EINVAL, "array data is not aligned to its element size");
    if (!passes) throw Error(MS_EINVAL, "null filter passes");
    if (count < 1 || count > MS_NDIMAGE_MAX_PASSES) throw Error(MS_EINVAL, "filter pass count must be 1 to 32");
    plan.passes.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const ms_ndimage_pass& given = passes[i];
        if (given.struct_size != sizeof(ms_ndimage_pass) || given.abi_version != MS_NDIMAGE_ABI_VERSION)
            throw Error(MS_EINVAL, "filter pass size or ABI version mismatch");
        if (given.reserved0 || given.reserved[0] || given.reserved[1])
            throw Error(MS_EINVAL, "filter pass reserved fields must be zero");
        if (given.kind != MS_NDIMAGE_CORRELATE && given.kind != MS_NDIMAGE_UNIFORM)
            throw Error(MS_EINVAL, "unknown filter pass kind");
        if (given.axis >= plan.shape.ndim) throw Error(MS_EINVAL, "filter axis is out of range");
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
        pass.length = plan.shape.dims[given.axis];
        pass.outer = pass.inner = 1;
        for (uint32_t axis = 0; axis < given.axis; ++axis) pass.outer *= plan.shape.dims[axis];
        for (uint32_t axis = given.axis + 1; axis < plan.shape.ndim; ++axis) pass.inner *= plan.shape.dims[axis];
        pass.before = static_cast<uint64_t>(size / 2 + given.origin);
        pass.after = static_cast<uint64_t>(size - size / 2 - 1 - given.origin);
        plan.passes.push_back(pass);
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
} // namespace ms::detail::ndimage

using ms::detail::Error;

ms_status ms_ndimage_separable_filter(ms_context* context, const ms_array* input, const ms_ndimage_pass* passes,
                                      uint32_t count, const ms_route_options* options,
                                      ms_array_result* out, ms_route_info* info) {
    namespace array = ms::detail::array;
    namespace ndimage = ms::detail::ndimage;
    namespace router = ms::detail::router;
    return ms::detail::api(context, [&] {
        if (!out) throw Error(MS_EINVAL, "null output");
        if (info && (info->struct_size != sizeof(ms_route_info) || info->abi_version != MS_ARRAY_ABI_VERSION))
            throw Error(MS_EINVAL, "route info size or ABI version mismatch");
        array::check_options(options);
        const ndimage::Plan plan = ndimage::make_plan(input, passes, count);

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
        uint32_t compute_dtype = MS_FLOAT64;
        if (!plan.shape.elements) {
            result = array::allocate(context, 0);
        } else if (decision.engine == MS_EXECUTION_GPU) {
#if defined(MS_ENABLE_METAL)
            result = ndimage::run_gpu(context, plan, *input);
            compute_dtype = MS_FLOAT32;
#else
            throw Error(MS_ENOTSUP, "GPU execution needs a Metal build");
#endif
        } else {
            result = ndimage::run_cpu(context, plan, input->data);
        }
        array::publish(out, std::move(result), plan.dtype, plan.shape);
        array::report(info, plan.shape.elements ? decision.engine : uint32_t{MS_EXECUTION_CPU}, compute_dtype);
    });
}
