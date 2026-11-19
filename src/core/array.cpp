#include "array.hpp"
#include "ms/detail/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace ms::detail::array {
namespace {
uint64_t scalars_per_element(uint32_t dtype) { return is_complex(dtype) ? 2 : 1; }
// Lanes get at least this many scalars each: below it one core is faster than
// waking the others.
constexpr uint64_t min_scalars_per_lane = uint64_t{1} << 18;

struct alignas(64) LaneStats {
    double max_finite = 0.0;
    double max_magnitude = 0.0;
    bool has_nonfinite = false;
};

// Largest finite double that does NOT round to infinity in IEEE float32 is
// strictly less than the midpoint 0x1.ffffffp127 between FLT_MAX and 2^128.
constexpr double float32_overflow_threshold = 0x1.ffffffp127;
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

void convert(const void* src, uint32_t src_dtype, void* dst, uint32_t dst_dtype, uint64_t elements,
             ConvertReport* report) {
    if (is_complex(src_dtype) != is_complex(dst_dtype))
        throw Error(MS_EINVAL, "real and complex arrays do not convert into each other");
    (void)dtype_size(dst_dtype);
    if (report) *report = ConvertReport{};
    const uint64_t scalars = elements * scalars_per_element(src_dtype);
    if (!scalars) return;
    const bool from_double = is_double(src_dtype), to_double = is_double(dst_dtype);
    if (from_double && !to_double && report) {
        LaneStats lane_stats[parallel::max_lanes] = {};
        parallel::parallel_for(scalars, min_scalars_per_lane, [&](unsigned lane, uint64_t begin, uint64_t end) {
            const uint64_t count = end - begin;
            const double* __restrict__ in = static_cast<const double*>(src) + begin;
            float* __restrict__ out = static_cast<float*>(dst) + begin;
            uint64_t i = 0;
            double lane_max_fin = 0.0;
            double lane_max_mag = 0.0;
            bool lane_nonfinite = false;
#if defined(__ARM_NEON)
            const uint64x2_t inf_bits = vdupq_n_u64(0x7ff0000000000000ULL);
            uint64x2_t all_fin0 = vdupq_n_u64(~uint64_t{0});
            uint64x2_t all_fin1 = vdupq_n_u64(~uint64_t{0});
            float64x2_t max_fin0 = vdupq_n_f64(0.0);
            float64x2_t max_fin1 = vdupq_n_f64(0.0);
            float64x2_t max_mag0 = vdupq_n_f64(0.0);
            float64x2_t max_mag1 = vdupq_n_f64(0.0);
            for (; i + 8 <= count; i += 8) {
                const float64x2_t v0 = vld1q_f64(in + i);
                const float64x2_t v1 = vld1q_f64(in + i + 2);
                const float64x2_t v2 = vld1q_f64(in + i + 4);
                const float64x2_t v3 = vld1q_f64(in + i + 6);

                const float32x4_t f01 = vcvt_high_f32_f64(vcvt_f32_f64(v0), v1);
                const float32x4_t f23 = vcvt_high_f32_f64(vcvt_f32_f64(v2), v3);
                vst1q_f32(out + i, f01);
                vst1q_f32(out + i + 4, f23);

                const float64x2_t av0 = vabsq_f64(v0);
                const float64x2_t av1 = vabsq_f64(v1);
                const float64x2_t av2 = vabsq_f64(v2);
                const float64x2_t av3 = vabsq_f64(v3);

                const uint64x2_t u0 = vreinterpretq_u64_f64(av0);
                const uint64x2_t u1 = vreinterpretq_u64_f64(av1);
                const uint64x2_t u2 = vreinterpretq_u64_f64(av2);
                const uint64x2_t u3 = vreinterpretq_u64_f64(av3);

                const uint64x2_t fm0 = vcltq_u64(u0, inf_bits);
                const uint64x2_t fm1 = vcltq_u64(u1, inf_bits);
                const uint64x2_t fm2 = vcltq_u64(u2, inf_bits);
                const uint64x2_t fm3 = vcltq_u64(u3, inf_bits);

                all_fin0 = vandq_u64(all_fin0, vandq_u64(fm0, fm1));
                all_fin1 = vandq_u64(all_fin1, vandq_u64(fm2, fm3));

                max_fin0 = vmaxnmq_f64(max_fin0, vreinterpretq_f64_u64(vandq_u64(u0, fm0)));
                max_fin1 = vmaxnmq_f64(max_fin1, vreinterpretq_f64_u64(vandq_u64(u1, fm1)));
                max_fin0 = vmaxnmq_f64(max_fin0, vreinterpretq_f64_u64(vandq_u64(u2, fm2)));
                max_fin1 = vmaxnmq_f64(max_fin1, vreinterpretq_f64_u64(vandq_u64(u3, fm3)));

                const uint64x2_t mm0 = vcleq_u64(u0, inf_bits);
                const uint64x2_t mm1 = vcleq_u64(u1, inf_bits);
                const uint64x2_t mm2 = vcleq_u64(u2, inf_bits);
                const uint64x2_t mm3 = vcleq_u64(u3, inf_bits);

                max_mag0 = vmaxnmq_f64(max_mag0, vreinterpretq_f64_u64(vandq_u64(u0, mm0)));
                max_mag1 = vmaxnmq_f64(max_mag1, vreinterpretq_f64_u64(vandq_u64(u1, mm1)));
                max_mag0 = vmaxnmq_f64(max_mag0, vreinterpretq_f64_u64(vandq_u64(u2, mm2)));
                max_mag1 = vmaxnmq_f64(max_mag1, vreinterpretq_f64_u64(vandq_u64(u3, mm3)));
            }
            const float64x2_t max_fin_v = vmaxnmq_f64(max_fin0, max_fin1);
            const float64x2_t max_mag_v = vmaxnmq_f64(max_mag0, max_mag1);
            const uint64x2_t all_fin_v = vandq_u64(all_fin0, all_fin1);
            lane_max_fin = std::max(vgetq_lane_f64(max_fin_v, 0), vgetq_lane_f64(max_fin_v, 1));
            lane_max_mag = std::max(vgetq_lane_f64(max_mag_v, 0), vgetq_lane_f64(max_mag_v, 1));
            lane_nonfinite = (vgetq_lane_u64(all_fin_v, 0) & vgetq_lane_u64(all_fin_v, 1)) != ~uint64_t{0};
#endif
            for (; i < count; ++i) {
                const double x = in[i];
                out[i] = static_cast<float>(x);
                const double ax = std::fabs(x);
                if (std::isfinite(ax)) {
                    if (ax > lane_max_fin) lane_max_fin = ax;
                    if (ax > lane_max_mag) lane_max_mag = ax;
                } else {
                    lane_nonfinite = true;
                    if (std::isinf(ax)) lane_max_mag = INFINITY;
                }
            }
            lane_stats[lane] = {lane_max_fin, lane_max_mag, lane_nonfinite};
        });
        for (unsigned lane = 0; lane < parallel::max_lanes; ++lane) {
            if (lane_stats[lane].max_finite > report->max_finite) report->max_finite = lane_stats[lane].max_finite;
            if (lane_stats[lane].max_magnitude > report->max_magnitude)
                report->max_magnitude = lane_stats[lane].max_magnitude;
            report->has_nonfinite = report->has_nonfinite || lane_stats[lane].has_nonfinite;
        }
        report->overflowed = report->max_finite >= float32_overflow_threshold;
        return;
    }
    parallel::parallel_for(scalars, min_scalars_per_lane, [&](unsigned, uint64_t begin, uint64_t end) {
        const uint64_t count = end - begin;
        if (from_double == to_double) {
            const uint64_t width = from_double ? 8 : 4;
            std::memcpy(static_cast<uint8_t*>(dst) + begin * width,
                        static_cast<const uint8_t*>(src) + begin * width, count * width);
        } else if (from_double) {
            const double* __restrict__ in = static_cast<const double*>(src) + begin;
            float* __restrict__ out = static_cast<float*>(dst) + begin;
            for (uint64_t i = 0; i < count; ++i) out[i] = static_cast<float>(in[i]);
        } else {
            const float* __restrict__ in = static_cast<const float*>(src) + begin;
            double* __restrict__ out = static_cast<double*>(dst) + begin;
            for (uint64_t i = 0; i < count; ++i) out[i] = static_cast<double>(in[i]);
        }
    });
}

BufferPtr load_single(ms_context* context, const ms_array& array, const Shape& shape, ConvertReport* report) {
    const uint32_t single = single_of(array.dtype);
    BufferPtr buffer = allocate(context, shape.elements * dtype_size(single));
    convert(array.data, array.dtype, mutable_data(buffer.get()), single, shape.elements, report);
    return buffer;
}

void check_route_info(const ms_route_info* info) {
    if (!info) return;
    if (info->struct_size != sizeof(ms_route_info) || info->abi_version != MS_ARRAY_ABI_VERSION)
        throw Error(MS_EINVAL, "route info size or ABI version mismatch");
}

void publish(ms_array_result* out, BufferPtr buffer, uint32_t dtype, const Shape& shape) noexcept {
    ms_array_result result{};
    result.dtype = dtype;
    result.ndim = shape.ndim;
    for (uint32_t axis = 0; axis < shape.ndim; ++axis) result.shape[axis] = shape.dims[axis];
    result.buffer = buffer.release();
    *out = result;
}

void report(ms_route_info* info, uint32_t engine, uint32_t compute_dtype, const char* reason) noexcept {
    if (!info) return;
    info->engine = engine;
    info->compute_dtype = compute_dtype;
    info->reserved[0] = 0;
    if (!reason || reason[0] == '\0') {
        info->reason[0] = '\0';
    } else {
        std::snprintf(info->reason, sizeof(info->reason), "%s", reason);
    }
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
