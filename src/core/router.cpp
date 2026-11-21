#include "router.hpp"
#include "array.hpp"

namespace sm::detail::router {
namespace {
bool gpu_available(sm_context* context) {
#if defined(SM_ENABLE_METAL)
    return storage_kind(context) == SM_BUFFER_METAL;
#else
    (void)context;
    return false;
#endif
}
} // namespace

Decision decide(sm_context* context, const Family& family, const sm_route_options& options,
                uint32_t input_dtype, uint64_t work) {
    const Decision cpu{SM_EXECUTION_CPU, input_dtype};
    const Decision gpu{SM_EXECUTION_GPU, array::single_of(input_dtype)};
    // A double-precision or integer/boolean input reaches the GPU only if the
    // policy allows single-precision arithmetic.
    const bool policy_allows_gpu =
        (!array::is_double(input_dtype) && !array::is_int_or_bool(input_dtype)) ||
        options.precision != SM_PRECISION_STRICT;
    switch (options.execution) {
    case SM_EXECUTION_CPU:
        return cpu;
    case SM_EXECUTION_GPU:
        if (!gpu_available(context))
            throw Error(SM_ENOTSUP, "GPU execution needs a Metal build and a Metal-storage context");
        if (!policy_allows_gpu)
            throw Error(SM_ENOTSUP, "strict precision keeps non-float32 inputs on the CPU");
        return gpu;
    default:
        return gpu_available(context) && policy_allows_gpu && work >= family.gpu_min_work ? gpu : cpu;
    }
}
} // namespace sm::detail::router
