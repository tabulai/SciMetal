#include "router.hpp"
#include "array.hpp"

namespace ms::detail::router {
namespace {
bool gpu_available(ms_context* context) {
#if defined(MS_ENABLE_METAL)
    return storage_kind(context) == MS_BUFFER_METAL;
#else
    (void)context;
    return false;
#endif
}
} // namespace

Decision decide(ms_context* context, const Family& family, const ms_route_options& options,
                uint32_t input_dtype, uint64_t work) {
    const Decision cpu{MS_EXECUTION_CPU, input_dtype};
    const Decision gpu{MS_EXECUTION_GPU, array::single_of(input_dtype)};
    // A double-precision or integer/boolean input reaches the GPU only if the
    // policy allows single-precision arithmetic.
    const bool policy_allows_gpu =
        (!array::is_double(input_dtype) && !array::is_int_or_bool(input_dtype)) ||
        options.precision != MS_PRECISION_STRICT;
    switch (options.execution) {
    case MS_EXECUTION_CPU:
        return cpu;
    case MS_EXECUTION_GPU:
        if (!gpu_available(context))
            throw Error(MS_ENOTSUP, "GPU execution needs a Metal build and a Metal-storage context");
        if (!policy_allows_gpu)
            throw Error(MS_ENOTSUP, "strict precision keeps non-float32 inputs on the CPU");
        return gpu;
    default:
        return gpu_available(context) && policy_allows_gpu && work >= family.gpu_min_work ? gpu : cpu;
    }
}
} // namespace ms::detail::router
