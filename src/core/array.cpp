#include "array.hpp"
#include "ms/detail/parallel.hpp"

#include <cstring>

namespace ms::detail::array {
namespace {
// Lanes get at least this many scalars each: below it one core is faster than
// waking the others.
constexpr uint64_t min_scalars_per_lane = uint64_t{1} << 18;

uint64_t scalars_per_element(uint32_t dtype) { return is_complex(dtype) ? 2 : 1; }
} // namespace

uint64_t dtype_size(uint32_t dtype) {
    switch (dtype) {
    case MS_FLOAT32: return 4;
    case MS_FLOAT64: return 8;
    case MS_COMPLEX64: return 8;
    case MS_COMPLEX128: return 16;
    default: throw Error(MS_EINVAL, "unknown array dtype");
    }
}
bool is_double(uint32_t dtype) { return dtype == MS_FLOAT64 || dtype == MS_COMPLEX128; }
bool is_complex(uint32_t dtype) { return dtype == MS_COMPLEX64 || dtype == MS_COMPLEX128; }
uint32_t single_of(uint32_t dtype) {
    if (dtype == MS_FLOAT64) return MS_FLOAT32;
    if (dtype == MS_COMPLEX128) return MS_COMPLEX64;
    return dtype;
}

Shape checked_shape(const ms_array* array) {
    if (!array) throw Error(MS_EINVAL, "null array");
    if (array->struct_size != sizeof(ms_array) || array->abi_version != MS_ARRAY_ABI_VERSION)
        throw Error(MS_EINVAL, "array descriptor size or ABI version mismatch");
    if (array->ndim < 1 || array->ndim > MS_MAX_NDIM) throw Error(MS_EINVAL, "array rank must be 1 to 8");
    const uint64_t width = dtype_size(array->dtype);
    Shape shape;
    shape.ndim = array->ndim;
    shape.elements = 1;
    for (uint32_t axis = 0; axis < array->ndim; ++axis) {
        shape.dims[axis] = array->shape[axis];
        if (ms_checked_mul(shape.elements, array->shape[axis], &shape.elements) != MS_OK)
            throw Error(MS_EOVERFLOW, "array element count overflows");
    }
    uint64_t bytes = 0;
    if (ms_checked_mul(shape.elements, width, &bytes) != MS_OK)
        throw Error(MS_EOVERFLOW, "array byte size overflows");
    if (shape.elements && !array->data) throw Error(MS_EINVAL, "null data for a non-empty array");
    return shape;
}

void check_options(const ms_route_options* options) {
    if (!options) throw Error(MS_EINVAL, "null route options");
    if (options->struct_size != sizeof(ms_route_options) || options->abi_version != MS_ARRAY_ABI_VERSION)
        throw Error(MS_EINVAL, "route options size or ABI version mismatch");
    if (options->execution != MS_EXECUTION_CPU && options->execution != MS_EXECUTION_GPU &&
        options->execution != MS_EXECUTION_AUTO)
        throw Error(MS_EINVAL, "unknown execution mode");
    if (options->precision != MS_PRECISION_FAST && options->precision != MS_PRECISION_REFINED &&
        options->precision != MS_PRECISION_STRICT)
        throw Error(MS_EINVAL, "unknown precision policy");
}

BufferPtr allocate(ms_context* context, uint64_t bytes) {
    auto reservation = reserve(context, padded_capacity(bytes, context));
    return allocate_reserved(context, reservation, bytes, Fill::unfilled);
}

void convert(const void* src, uint32_t src_dtype, void* dst, uint32_t dst_dtype, uint64_t elements) {
    if (is_complex(src_dtype) != is_complex(dst_dtype))
        throw Error(MS_EINVAL, "real and complex arrays do not convert into each other");
    (void)dtype_size(dst_dtype);
    const uint64_t scalars = elements * scalars_per_element(src_dtype);
    if (!scalars) return;
    const bool from_double = is_double(src_dtype), to_double = is_double(dst_dtype);
    parallel::parallel_for(scalars, min_scalars_per_lane, [&](unsigned, uint64_t begin, uint64_t end) {
        const uint64_t count = end - begin;
        if (from_double == to_double) {
            const uint64_t width = from_double ? 8 : 4;
            std::memcpy(static_cast<uint8_t*>(dst) + begin * width,
                        static_cast<const uint8_t*>(src) + begin * width, count * width);
        } else if (from_double) {
            const double* in = static_cast<const double*>(src) + begin;
            float* out = static_cast<float*>(dst) + begin;
            for (uint64_t i = 0; i < count; ++i) out[i] = static_cast<float>(in[i]);
        } else {
            const float* in = static_cast<const float*>(src) + begin;
            double* out = static_cast<double*>(dst) + begin;
            for (uint64_t i = 0; i < count; ++i) out[i] = static_cast<double>(in[i]);
        }
    });
}

BufferPtr load_single(ms_context* context, const ms_array& array, const Shape& shape) {
    const uint32_t single = single_of(array.dtype);
    BufferPtr buffer = allocate(context, shape.elements * dtype_size(single));
    convert(array.data, array.dtype, mutable_data(buffer.get()), single, shape.elements);
    return buffer;
}

void publish(ms_array_result* out, BufferPtr buffer, uint32_t dtype, const Shape& shape) noexcept {
    ms_array_result result{};
    result.dtype = dtype;
    result.ndim = shape.ndim;
    for (uint32_t axis = 0; axis < shape.ndim; ++axis) result.shape[axis] = shape.dims[axis];
    result.buffer = buffer.release();
    *out = result;
}

void report(ms_route_info* info, uint32_t engine, uint32_t compute_dtype) noexcept {
    if (!info) return;
    info->engine = engine;
    info->compute_dtype = compute_dtype;
}
} // namespace ms::detail::array

using ms::detail::Error;

ms_status ms_array_astype(ms_context* context, const ms_array* input, uint32_t dtype, ms_array_result* out) {
    namespace array = ms::detail::array;
    return ms::detail::api(context, [&] {
        if (!out) throw Error(MS_EINVAL, "null output");
        const array::Shape shape = array::checked_shape(input);
        const uint64_t width = array::dtype_size(dtype);
        if (array::is_complex(input->dtype) != array::is_complex(dtype))
            throw Error(MS_EINVAL, "real and complex arrays do not convert into each other");
        auto buffer = array::allocate(context, shape.elements * width);
        array::convert(input->data, input->dtype, ms::detail::mutable_data(buffer.get()), dtype, shape.elements);
        array::publish(out, std::move(buffer), dtype, shape);
    });
}

uint32_t ms_array_capabilities(void) {
#if defined(MS_ENABLE_METAL)
    return MS_CAPABILITY_METAL;
#else
    return 0;
#endif
}
