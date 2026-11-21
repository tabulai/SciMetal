#include "array.hpp"
#include "sm/detail/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace sm::detail::array {
namespace {
uint64_t scalars_per_element(uint32_t dtype) { return is_complex(dtype) ? 2 : 1; }
// Lanes get at least this many scalars each: below it one core is faster than
// waking the others.
constexpr uint64_t min_scalars_per_lane = uint64_t{1} << 18;

struct alignas(64) LaneStats {
    double max_finite = 0.0;
    double max_magnitude = 0.0;
    bool has_nonfinite = false;
    bool inexact_int = false;
};

// Largest finite double that does NOT round to infinity in IEEE float32 is
// strictly less than the midpoint 0x1.ffffffp127 between FLT_MAX and 2^128.
constexpr double float32_overflow_threshold = 0x1.ffffffp127;
// Exact representable integer magnitude bound in IEEE float32: |x| < 2^24.
constexpr int64_t float32_exact_int_limit = int64_t{1} << 24;
constexpr uint64_t float32_exact_uint_limit = uint64_t{1} << 24;

template<class T>
void int_to_float32_lane(const T* __restrict__ in, float* __restrict__ out, uint64_t count,
                         ConvertReport* report, LaneStats& stats) {
    if (!report) {
        for (uint64_t i = 0; i < count; ++i) out[i] = static_cast<float>(in[i]);
        return;
    }
    if constexpr (sizeof(T) <= 2) {
        uint32_t max_u = 0;
        for (uint64_t i = 0; i < count; ++i) {
            const T x = in[i];
            out[i] = static_cast<float>(x);
            uint32_t mag;
            if constexpr (std::is_signed_v<T>) {
                mag = x < 0 ? (0u - static_cast<uint32_t>(static_cast<int32_t>(x)))
                            : static_cast<uint32_t>(static_cast<int32_t>(x));
            } else {
                mag = static_cast<uint32_t>(x);
            }
            if (mag > max_u) max_u = mag;
        }
        const double dm = static_cast<double>(max_u);
        stats.max_finite = dm;
        stats.max_magnitude = dm;
    } else if constexpr (std::is_signed_v<T>) {
        double max_fin = 0.0;
        bool inexact = false;
        for (uint64_t i = 0; i < count; ++i) {
            const T x = in[i];
            out[i] = static_cast<float>(x);
            if (x >= static_cast<T>(float32_exact_int_limit) || x <= -static_cast<T>(float32_exact_int_limit)) {
                inexact = true;
            }
            const double dx = static_cast<double>(x);
            const double ax = dx < 0.0 ? -dx : dx;
            if (ax > max_fin) max_fin = ax;
        }
        stats.max_finite = max_fin;
        stats.max_magnitude = max_fin;
        stats.inexact_int = inexact;
    } else {
        double max_fin = 0.0;
        bool inexact = false;
        for (uint64_t i = 0; i < count; ++i) {
            const T x = in[i];
            out[i] = static_cast<float>(x);
            if (x >= static_cast<T>(float32_exact_uint_limit)) {
                inexact = true;
            }
            const double ax = static_cast<double>(x);
            if (ax > max_fin) max_fin = ax;
        }
        stats.max_finite = max_fin;
        stats.max_magnitude = max_fin;
        stats.inexact_int = inexact;
    }
}

void convert_int_or_bool_to_float32(const void* src, uint32_t src_dtype, float* dst, uint64_t elements,
                                    ConvertReport* report) {
    LaneStats lane_stats[parallel::max_lanes] = {};
    parallel::parallel_for(elements, min_scalars_per_lane, [&](unsigned lane, uint64_t begin, uint64_t end) {
        const uint64_t count = end - begin;
        float* __restrict__ out = dst + begin;
        LaneStats& st = lane_stats[lane];
        switch (src_dtype) {
        case SM_BOOL: {
            const uint8_t* __restrict__ in = static_cast<const uint8_t*>(src) + begin;
            uint8_t any = 0;
            for (uint64_t i = 0; i < count; ++i) {
                const uint8_t v = in[i] ? uint8_t{1} : uint8_t{0};
                any |= v;
                out[i] = static_cast<float>(v);
            }
            st.max_finite = any ? 1.0 : 0.0;
            st.max_magnitude = st.max_finite;
            break;
        }
        case SM_INT8: int_to_float32_lane(static_cast<const int8_t*>(src) + begin, out, count, report, st); break;
        case SM_INT16: int_to_float32_lane(static_cast<const int16_t*>(src) + begin, out, count, report, st); break;
        case SM_INT32: int_to_float32_lane(static_cast<const int32_t*>(src) + begin, out, count, report, st); break;
        case SM_INT64: int_to_float32_lane(static_cast<const int64_t*>(src) + begin, out, count, report, st); break;
        case SM_UINT8: int_to_float32_lane(static_cast<const uint8_t*>(src) + begin, out, count, report, st); break;
        case SM_UINT16: int_to_float32_lane(static_cast<const uint16_t*>(src) + begin, out, count, report, st); break;
        case SM_UINT32: int_to_float32_lane(static_cast<const uint32_t*>(src) + begin, out, count, report, st); break;
        case SM_UINT64: int_to_float32_lane(static_cast<const uint64_t*>(src) + begin, out, count, report, st); break;
        default: break;
        }
    });
    if (report) {
        for (unsigned lane = 0; lane < parallel::max_lanes; ++lane) {
            if (lane_stats[lane].max_finite > report->max_finite) report->max_finite = lane_stats[lane].max_finite;
            if (lane_stats[lane].max_magnitude > report->max_magnitude)
                report->max_magnitude = lane_stats[lane].max_magnitude;
            report->inexact_int = report->inexact_int || lane_stats[lane].inexact_int;
        }
    }
}

void int_or_bool_to_float64_slice(const void* src, uint32_t src_dtype, double* dst, uint64_t begin, uint64_t count) {
    switch (src_dtype) {
    case SM_BOOL: {
        const uint8_t* in = static_cast<const uint8_t*>(src) + begin;
        for (uint64_t i = 0; i < count; ++i) dst[i] = in[i] ? 1.0 : 0.0;
        break;
    }
    case SM_INT8: {
        const int8_t* in = static_cast<const int8_t*>(src) + begin;
        for (uint64_t i = 0; i < count; ++i) dst[i] = static_cast<double>(in[i]);
        break;
    }
    case SM_INT16: {
        const int16_t* in = static_cast<const int16_t*>(src) + begin;
        for (uint64_t i = 0; i < count; ++i) dst[i] = static_cast<double>(in[i]);
        break;
    }
    case SM_INT32: {
        const int32_t* in = static_cast<const int32_t*>(src) + begin;
        for (uint64_t i = 0; i < count; ++i) dst[i] = static_cast<double>(in[i]);
        break;
    }
    case SM_INT64: {
        const int64_t* in = static_cast<const int64_t*>(src) + begin;
        for (uint64_t i = 0; i < count; ++i) dst[i] = static_cast<double>(in[i]);
        break;
    }
    case SM_UINT8: {
        const uint8_t* in = static_cast<const uint8_t*>(src) + begin;
        for (uint64_t i = 0; i < count; ++i) dst[i] = static_cast<double>(in[i]);
        break;
    }
    case SM_UINT16: {
        const uint16_t* in = static_cast<const uint16_t*>(src) + begin;
        for (uint64_t i = 0; i < count; ++i) dst[i] = static_cast<double>(in[i]);
        break;
    }
    case SM_UINT32: {
        const uint32_t* in = static_cast<const uint32_t*>(src) + begin;
        for (uint64_t i = 0; i < count; ++i) dst[i] = static_cast<double>(in[i]);
        break;
    }
    case SM_UINT64: {
        const uint64_t* in = static_cast<const uint64_t*>(src) + begin;
        for (uint64_t i = 0; i < count; ++i) dst[i] = static_cast<double>(in[i]);
        break;
    }
    default: break;
    }
}

template<class FloatT>
void float_to_int_or_bool_slice(const FloatT* src, void* dst, uint32_t dst_dtype, uint64_t begin, uint64_t count) {
    const FloatT* in = src + begin;
    switch (dst_dtype) {
    case SM_BOOL: {
        uint8_t* out = static_cast<uint8_t*>(dst) + begin;
        for (uint64_t i = 0; i < count; ++i) out[i] = (in[i] != FloatT{0}) ? uint8_t{1} : uint8_t{0};
        break;
    }
    case SM_INT8: {
        int8_t* out = static_cast<int8_t*>(dst) + begin;
        for (uint64_t i = 0; i < count; ++i) out[i] = cast_double_ndimage<int8_t>(static_cast<double>(in[i]));
        break;
    }
    case SM_INT16: {
        int16_t* out = static_cast<int16_t*>(dst) + begin;
        for (uint64_t i = 0; i < count; ++i) out[i] = cast_double_ndimage<int16_t>(static_cast<double>(in[i]));
        break;
    }
    case SM_INT32: {
        int32_t* out = static_cast<int32_t*>(dst) + begin;
        for (uint64_t i = 0; i < count; ++i) out[i] = cast_double_ndimage<int32_t>(static_cast<double>(in[i]));
        break;
    }
    case SM_INT64: {
        int64_t* out = static_cast<int64_t*>(dst) + begin;
        for (uint64_t i = 0; i < count; ++i) out[i] = cast_double_ndimage<int64_t>(static_cast<double>(in[i]));
        break;
    }
    case SM_UINT8: {
        uint8_t* out = static_cast<uint8_t*>(dst) + begin;
        for (uint64_t i = 0; i < count; ++i) out[i] = cast_double_ndimage<uint8_t>(static_cast<double>(in[i]));
        break;
    }
    case SM_UINT16: {
        uint16_t* out = static_cast<uint16_t*>(dst) + begin;
        for (uint64_t i = 0; i < count; ++i) out[i] = cast_double_ndimage<uint16_t>(static_cast<double>(in[i]));
        break;
    }
    case SM_UINT32: {
        uint32_t* out = static_cast<uint32_t*>(dst) + begin;
        for (uint64_t i = 0; i < count; ++i) out[i] = cast_double_ndimage<uint32_t>(static_cast<double>(in[i]));
        break;
    }
    case SM_UINT64: {
        uint64_t* out = static_cast<uint64_t*>(dst) + begin;
        for (uint64_t i = 0; i < count; ++i) out[i] = cast_double_ndimage<uint64_t>(static_cast<double>(in[i]));
        break;
    }
    default: break;
    }
}

void int_load_u64(const void* src, uint32_t src_dtype, uint64_t offset, uint64_t* dst, size_t n) {
    switch (src_dtype) {
    case SM_BOOL: {
        const uint8_t* p = static_cast<const uint8_t*>(src) + offset;
        for (size_t i = 0; i < n; ++i) dst[i] = p[i] ? uint64_t{1} : uint64_t{0};
        break;
    }
    case SM_INT8: {
        const int8_t* p = static_cast<const int8_t*>(src) + offset;
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<uint64_t>(static_cast<int64_t>(p[i]));
        break;
    }
    case SM_INT16: {
        const int16_t* p = static_cast<const int16_t*>(src) + offset;
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<uint64_t>(static_cast<int64_t>(p[i]));
        break;
    }
    case SM_INT32: {
        const int32_t* p = static_cast<const int32_t*>(src) + offset;
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<uint64_t>(static_cast<int64_t>(p[i]));
        break;
    }
    case SM_INT64: {
        const int64_t* p = static_cast<const int64_t*>(src) + offset;
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<uint64_t>(p[i]);
        break;
    }
    case SM_UINT8: {
        const uint8_t* p = static_cast<const uint8_t*>(src) + offset;
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<uint64_t>(p[i]);
        break;
    }
    case SM_UINT16: {
        const uint16_t* p = static_cast<const uint16_t*>(src) + offset;
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<uint64_t>(p[i]);
        break;
    }
    case SM_UINT32: {
        const uint32_t* p = static_cast<const uint32_t*>(src) + offset;
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<uint64_t>(p[i]);
        break;
    }
    case SM_UINT64: {
        const uint64_t* p = static_cast<const uint64_t*>(src) + offset;
        std::memcpy(dst, p, n * sizeof(uint64_t));
        break;
    }
    default: break;
    }
}

void u64_store_int(const uint64_t* src, void* dst, uint32_t dst_dtype, uint64_t offset, size_t n) {
    switch (dst_dtype) {
    case SM_BOOL: {
        uint8_t* p = static_cast<uint8_t*>(dst) + offset;
        for (size_t i = 0; i < n; ++i) p[i] = src[i] ? uint8_t{1} : uint8_t{0};
        break;
    }
    case SM_INT8: {
        int8_t* p = static_cast<int8_t*>(dst) + offset;
        for (size_t i = 0; i < n; ++i) p[i] = static_cast<int8_t>(static_cast<uint8_t>(src[i]));
        break;
    }
    case SM_INT16: {
        int16_t* p = static_cast<int16_t*>(dst) + offset;
        for (size_t i = 0; i < n; ++i) p[i] = static_cast<int16_t>(static_cast<uint16_t>(src[i]));
        break;
    }
    case SM_INT32: {
        int32_t* p = static_cast<int32_t*>(dst) + offset;
        for (size_t i = 0; i < n; ++i) p[i] = static_cast<int32_t>(static_cast<uint32_t>(src[i]));
        break;
    }
    case SM_INT64: {
        int64_t* p = static_cast<int64_t*>(dst) + offset;
        for (size_t i = 0; i < n; ++i) p[i] = static_cast<int64_t>(src[i]);
        break;
    }
    case SM_UINT8: {
        uint8_t* p = static_cast<uint8_t*>(dst) + offset;
        for (size_t i = 0; i < n; ++i) p[i] = static_cast<uint8_t>(src[i]);
        break;
    }
    case SM_UINT16: {
        uint16_t* p = static_cast<uint16_t*>(dst) + offset;
        for (size_t i = 0; i < n; ++i) p[i] = static_cast<uint16_t>(src[i]);
        break;
    }
    case SM_UINT32: {
        uint32_t* p = static_cast<uint32_t*>(dst) + offset;
        for (size_t i = 0; i < n; ++i) p[i] = static_cast<uint32_t>(src[i]);
        break;
    }
    case SM_UINT64: {
        uint64_t* p = static_cast<uint64_t*>(dst) + offset;
        std::memcpy(p, src, n * sizeof(uint64_t));
        break;
    }
    default: break;
    }
}
} // namespace

uint64_t dtype_size(uint32_t dtype) {
    switch (dtype) {
    case SM_BOOL:
    case SM_INT8:
    case SM_UINT8: return 1;
    case SM_INT16:
    case SM_UINT16: return 2;
    case SM_FLOAT32:
    case SM_INT32:
    case SM_UINT32: return 4;
    case SM_FLOAT64:
    case SM_COMPLEX64:
    case SM_INT64:
    case SM_UINT64: return 8;
    case SM_COMPLEX128: return 16;
    default: throw Error(SM_EINVAL, "unknown array dtype");
    }
}
uint64_t scalar_alignment(uint32_t dtype) {
    const uint64_t width = dtype_size(dtype);
    return is_complex(dtype) ? width / 2 : width;
}
bool is_double(uint32_t dtype) { return dtype == SM_FLOAT64 || dtype == SM_COMPLEX128; }
bool is_complex(uint32_t dtype) { return dtype == SM_COMPLEX64 || dtype == SM_COMPLEX128; }
bool is_float(uint32_t dtype) { return dtype == SM_FLOAT32 || dtype == SM_FLOAT64; }
bool is_bool(uint32_t dtype) { return dtype == SM_BOOL; }
bool is_integer(uint32_t dtype) { return dtype >= SM_INT8 && dtype <= SM_UINT64; }
bool is_signed_int(uint32_t dtype) { return dtype >= SM_INT8 && dtype <= SM_INT64; }
bool is_unsigned_int(uint32_t dtype) { return dtype >= SM_UINT8 && dtype <= SM_UINT64; }
bool is_int_or_bool(uint32_t dtype) { return dtype >= SM_BOOL && dtype <= SM_UINT64; }
bool is_real(uint32_t dtype) { return dtype == SM_FLOAT32 || dtype == SM_FLOAT64 || is_int_or_bool(dtype); }
uint32_t single_of(uint32_t dtype) {
    if (is_complex(dtype)) return SM_COMPLEX64;
    return SM_FLOAT32;
}

Shape checked_shape(const sm_array* array) {
    if (!array) throw Error(SM_EINVAL, "null array");
    if (array->struct_size != sizeof(sm_array) || array->abi_version != SM_ARRAY_ABI_VERSION)
        throw Error(SM_EINVAL, "array descriptor size or ABI version mismatch");
    if (array->ndim < 1 || array->ndim > SM_MAX_NDIM) throw Error(SM_EINVAL, "array rank must be 1 to 8");
    const uint64_t width = dtype_size(array->dtype);
    Shape shape;
    shape.ndim = array->ndim;
    shape.elements = 1;
    for (uint32_t axis = 0; axis < array->ndim; ++axis) {
        shape.dims[axis] = array->shape[axis];
        if (sm_checked_mul(shape.elements, array->shape[axis], &shape.elements) != SM_OK)
            throw Error(SM_EOVERFLOW, "array element count overflows");
    }
    uint64_t bytes = 0;
    if (sm_checked_mul(shape.elements, width, &bytes) != SM_OK)
        throw Error(SM_EOVERFLOW, "array byte size overflows");
    if (shape.elements) {
        if (!array->data) throw Error(SM_EINVAL, "null data for a non-empty array");
        const uint64_t align = scalar_alignment(array->dtype);
        if (reinterpret_cast<uintptr_t>(array->data) % align != 0)
            throw Error(SM_EINVAL, "array data pointer is not aligned for its dtype");
    }
    return shape;
}

void check_options(const sm_route_options* options) {
    if (!options) throw Error(SM_EINVAL, "null route options");
    if (options->struct_size != sizeof(sm_route_options) || options->abi_version != SM_ARRAY_ABI_VERSION)
        throw Error(SM_EINVAL, "route options size or ABI version mismatch");
    if (options->execution != SM_EXECUTION_CPU && options->execution != SM_EXECUTION_GPU &&
        options->execution != SM_EXECUTION_AUTO)
        throw Error(SM_EINVAL, "unknown execution mode");
    if (options->precision != SM_PRECISION_FAST && options->precision != SM_PRECISION_REFINED &&
        options->precision != SM_PRECISION_STRICT)
        throw Error(SM_EINVAL, "unknown precision policy");
    if (options->reserved[0] || options->reserved[1])
        throw Error(SM_EINVAL, "route options reserved fields must be zero");
}

BufferPtr allocate(sm_context* context, uint64_t bytes) {
    auto reservation = reserve(context, padded_capacity(bytes, context));
    return allocate_reserved(context, reservation, bytes, Fill::unfilled);
}

void convert(const void* src, uint32_t src_dtype, void* dst, uint32_t dst_dtype, uint64_t elements,
             ConvertReport* report) {
    if (is_complex(src_dtype) != is_complex(dst_dtype))
        throw Error(SM_EINVAL, "real and complex arrays do not convert into each other");
    const uint64_t src_width = dtype_size(src_dtype);
    (void)dtype_size(dst_dtype);
    if (report) *report = ConvertReport{};
    if (!elements) return;

    if (src_dtype == dst_dtype) {
        if (report && (src_dtype == SM_FLOAT32 || src_dtype == SM_COMPLEX64)) {
            uint64_t scalars = 0;
            if (sm_checked_mul(elements, scalars_per_element(src_dtype), &scalars) != SM_OK)
                throw Error(SM_EOVERFLOW, "array scalar count overflows");
            LaneStats lane_stats[parallel::max_lanes] = {};
            parallel::parallel_for(scalars, min_scalars_per_lane, [&](unsigned lane, uint64_t begin, uint64_t end) {
                const uint64_t count = end - begin;
                const float* __restrict__ in = static_cast<const float*>(src) + begin;
                float* __restrict__ out = static_cast<float*>(dst) + begin;
                uint64_t i = 0;
                double lane_max_fin = 0.0;
                double lane_max_mag = 0.0;
                bool lane_nonfinite = false;
#if defined(__ARM_NEON)
                const uint32x4_t inf_bits = vdupq_n_u32(0x7f800000u);
                uint32x4_t all_fin0 = vdupq_n_u32(~uint32_t{0});
                uint32x4_t all_fin1 = vdupq_n_u32(~uint32_t{0});
                float32x4_t max_fin0 = vdupq_n_f32(0.0f);
                float32x4_t max_fin1 = vdupq_n_f32(0.0f);
                float32x4_t max_mag0 = vdupq_n_f32(0.0f);
                float32x4_t max_mag1 = vdupq_n_f32(0.0f);
                for (; i + 16 <= count; i += 16) {
                    const float32x4_t v0 = vld1q_f32(in + i);
                    const float32x4_t v1 = vld1q_f32(in + i + 4);
                    const float32x4_t v2 = vld1q_f32(in + i + 8);
                    const float32x4_t v3 = vld1q_f32(in + i + 12);
                    vst1q_f32(out + i, v0);
                    vst1q_f32(out + i + 4, v1);
                    vst1q_f32(out + i + 8, v2);
                    vst1q_f32(out + i + 12, v3);

                    const uint32x4_t u0 = vreinterpretq_u32_f32(vabsq_f32(v0));
                    const uint32x4_t u1 = vreinterpretq_u32_f32(vabsq_f32(v1));
                    const uint32x4_t u2 = vreinterpretq_u32_f32(vabsq_f32(v2));
                    const uint32x4_t u3 = vreinterpretq_u32_f32(vabsq_f32(v3));

                    const uint32x4_t fm0 = vcltq_u32(u0, inf_bits);
                    const uint32x4_t fm1 = vcltq_u32(u1, inf_bits);
                    const uint32x4_t fm2 = vcltq_u32(u2, inf_bits);
                    const uint32x4_t fm3 = vcltq_u32(u3, inf_bits);

                    all_fin0 = vandq_u32(all_fin0, vandq_u32(fm0, fm1));
                    all_fin1 = vandq_u32(all_fin1, vandq_u32(fm2, fm3));

                    max_fin0 = vmaxnmq_f32(max_fin0, vreinterpretq_f32_u32(vandq_u32(u0, fm0)));
                    max_fin1 = vmaxnmq_f32(max_fin1, vreinterpretq_f32_u32(vandq_u32(u1, fm1)));
                    max_fin0 = vmaxnmq_f32(max_fin0, vreinterpretq_f32_u32(vandq_u32(u2, fm2)));
                    max_fin1 = vmaxnmq_f32(max_fin1, vreinterpretq_f32_u32(vandq_u32(u3, fm3)));

                    const uint32x4_t mm0 = vcleq_u32(u0, inf_bits);
                    const uint32x4_t mm1 = vcleq_u32(u1, inf_bits);
                    const uint32x4_t mm2 = vcleq_u32(u2, inf_bits);
                    const uint32x4_t mm3 = vcleq_u32(u3, inf_bits);

                    max_mag0 = vmaxnmq_f32(max_mag0, vreinterpretq_f32_u32(vandq_u32(u0, mm0)));
                    max_mag1 = vmaxnmq_f32(max_mag1, vreinterpretq_f32_u32(vandq_u32(u1, mm1)));
                    max_mag0 = vmaxnmq_f32(max_mag0, vreinterpretq_f32_u32(vandq_u32(u2, mm2)));
                    max_mag1 = vmaxnmq_f32(max_mag1, vreinterpretq_f32_u32(vandq_u32(u3, mm3)));
                }
                lane_max_fin = static_cast<double>(vmaxnmvq_f32(vmaxnmq_f32(max_fin0, max_fin1)));
                lane_max_mag = static_cast<double>(vmaxnmvq_f32(vmaxnmq_f32(max_mag0, max_mag1)));
                lane_nonfinite = vminvq_u32(vandq_u32(all_fin0, all_fin1)) != ~uint32_t{0};
#endif
                for (; i < count; ++i) {
                    const float x = in[i];
                    out[i] = x;
                    const double ax = std::fabs(static_cast<double>(x));
                    if (std::isfinite(ax)) {
                        if (ax > lane_max_fin) lane_max_fin = ax;
                        if (ax > lane_max_mag) lane_max_mag = ax;
                    } else {
                        lane_nonfinite = true;
                        if (std::isinf(ax)) lane_max_mag = INFINITY;
                    }
                }
                lane_stats[lane] = {lane_max_fin, lane_max_mag, lane_nonfinite, false};
            });
            for (unsigned lane = 0; lane < parallel::max_lanes; ++lane) {
                if (lane_stats[lane].max_finite > report->max_finite) report->max_finite = lane_stats[lane].max_finite;
                if (lane_stats[lane].max_magnitude > report->max_magnitude)
                    report->max_magnitude = lane_stats[lane].max_magnitude;
                report->has_nonfinite = report->has_nonfinite || lane_stats[lane].has_nonfinite;
            }
            return;
        }
        parallel::parallel_for(elements, min_scalars_per_lane, [&](unsigned, uint64_t begin, uint64_t end) {
            const uint64_t count = end - begin;
            std::memcpy(static_cast<uint8_t*>(dst) + begin * src_width,
                        static_cast<const uint8_t*>(src) + begin * src_width, count * src_width);
        });
        return;
    }

    if (is_int_or_bool(src_dtype) && dst_dtype == SM_FLOAT32) {
        convert_int_or_bool_to_float32(src, src_dtype, static_cast<float*>(dst), elements, report);
        return;
    }

    const bool from_double = is_double(src_dtype), to_double = is_double(dst_dtype);
    if ((src_dtype == SM_FLOAT64 && dst_dtype == SM_FLOAT32) ||
        (is_complex(src_dtype) && from_double && !to_double)) {
        const uint64_t scalars = elements * scalars_per_element(src_dtype);
        if (report) {
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
                lane_stats[lane] = {lane_max_fin, lane_max_mag, lane_nonfinite, false};
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
            const double* __restrict__ in = static_cast<const double*>(src) + begin;
            float* __restrict__ out = static_cast<float*>(dst) + begin;
            for (uint64_t i = 0; i < count; ++i) out[i] = static_cast<float>(in[i]);
        });
        return;
    }

    if ((src_dtype == SM_FLOAT32 && dst_dtype == SM_FLOAT64) ||
        (is_complex(src_dtype) && !from_double && to_double)) {
        const uint64_t scalars = elements * scalars_per_element(src_dtype);
        parallel::parallel_for(scalars, min_scalars_per_lane, [&](unsigned, uint64_t begin, uint64_t end) {
            const uint64_t count = end - begin;
            const float* __restrict__ in = static_cast<const float*>(src) + begin;
            double* __restrict__ out = static_cast<double*>(dst) + begin;
            for (uint64_t i = 0; i < count; ++i) out[i] = static_cast<double>(in[i]);
        });
        return;
    }

    if (is_int_or_bool(src_dtype) && dst_dtype == SM_FLOAT64) {
        parallel::parallel_for(elements, min_scalars_per_lane, [&](unsigned, uint64_t begin, uint64_t end) {
            int_or_bool_to_float64_slice(src, src_dtype, static_cast<double*>(dst) + begin, begin, end - begin);
        });
        return;
    }

    if (is_float(src_dtype) && is_int_or_bool(dst_dtype)) {
        parallel::parallel_for(elements, min_scalars_per_lane, [&](unsigned, uint64_t begin, uint64_t end) {
            if (src_dtype == SM_FLOAT32) {
                float_to_int_or_bool_slice(static_cast<const float*>(src), dst, dst_dtype, begin, end - begin);
            } else {
                float_to_int_or_bool_slice(static_cast<const double*>(src), dst, dst_dtype, begin, end - begin);
            }
        });
        return;
    }

    // Remaining pair: int/bool -> int/bool via 64-bit integer chunks (exact for int64/uint64).
    parallel::parallel_for(elements, min_scalars_per_lane, [&](unsigned, uint64_t begin, uint64_t end) {
        constexpr uint64_t chunk_elems = 1024;
        uint64_t tmp[chunk_elems];
        for (uint64_t pos = begin; pos < end; pos += chunk_elems) {
            const size_t n = static_cast<size_t>(std::min(chunk_elems, end - pos));
            int_load_u64(src, src_dtype, pos, tmp, n);
            u64_store_int(tmp, dst, dst_dtype, pos, n);
        }
    });
}

void ndimage_cast_from_float(const float* src, uint32_t dst_dtype, void* dst, uint64_t elements, bool dense_safe) {
    if (!elements) return;
    if (dst_dtype == SM_FLOAT32) {
        parallel::parallel_for(elements, min_scalars_per_lane, [&](unsigned, uint64_t begin, uint64_t end) {
            std::memcpy(static_cast<float*>(dst) + begin, src + begin, (end - begin) * sizeof(float));
        });
        return;
    }
    parallel::parallel_for(elements, min_scalars_per_lane, [&](unsigned, uint64_t begin, uint64_t end) {
        const uint64_t count = end - begin;
        const float* __restrict__ in = src + begin;
        auto run = [&](auto* out) {
            using DstT = std::remove_pointer_t<decltype(out)>;
            for (uint64_t i = 0; i < count; ++i) {
                out[i] = cast_double_ndimage<DstT>(static_cast<double>(in[i]), dense_safe);
            }
        };
        switch (dst_dtype) {
        case SM_FLOAT64: run(static_cast<double*>(dst) + begin); break;
        case SM_BOOL:
        case SM_UINT8: run(static_cast<uint8_t*>(dst) + begin); break;
        case SM_UINT16: run(static_cast<uint16_t*>(dst) + begin); break;
        case SM_UINT32: run(static_cast<uint32_t*>(dst) + begin); break;
        case SM_UINT64: run(static_cast<uint64_t*>(dst) + begin); break;
        case SM_INT8: run(static_cast<int8_t*>(dst) + begin); break;
        case SM_INT16: run(static_cast<int16_t*>(dst) + begin); break;
        case SM_INT32: run(static_cast<int32_t*>(dst) + begin); break;
        case SM_INT64: run(static_cast<int64_t*>(dst) + begin); break;
        default: throw Error(SM_EINVAL, "unknown ndimage output dtype");
        }
    });
}

BufferPtr load_single(sm_context* context, const sm_array& array, const Shape& shape, ConvertReport* report,
                      bool raw_bool) {
    const uint32_t single = single_of(array.dtype);
    uint64_t bytes = 0;
    if (sm_checked_mul(shape.elements, dtype_size(single), &bytes) != SM_OK)
        throw Error(SM_EOVERFLOW, "single-precision buffer byte size overflows");
    BufferPtr buffer = allocate(context, bytes);
    const uint32_t src_dtype = (raw_bool && array.dtype == SM_BOOL) ? SM_UINT8 : array.dtype;
    convert(array.data, src_dtype, mutable_data(buffer.get()), single, shape.elements, report);
    return buffer;
}

void check_route_info(const sm_route_info* info) {
    if (!info) return;
    if (info->struct_size != sizeof(sm_route_info) || info->abi_version != SM_ARRAY_ABI_VERSION)
        throw Error(SM_EINVAL, "route info size or ABI version mismatch");
}

void publish(sm_array_result* out, BufferPtr buffer, uint32_t dtype, const Shape& shape) noexcept {
    sm_array_result result{};
    result.dtype = dtype;
    result.ndim = shape.ndim;
    for (uint32_t axis = 0; axis < shape.ndim; ++axis) result.shape[axis] = shape.dims[axis];
    result.buffer = buffer.release();
    *out = result;
}

void report(sm_route_info* info, uint32_t engine, uint32_t compute_dtype, const char* reason) noexcept {
    if (!info) return;
    info->engine = engine;
    info->compute_dtype = compute_dtype;
    info->reserved[0] = 0;
    info->reserved[1] = 0;
    if (!reason || reason[0] == '\0') {
        info->reason[0] = '\0';
    } else {
        std::snprintf(info->reason, sizeof(info->reason), "%s", reason);
    }
}
} // namespace sm::detail::array

using sm::detail::Error;

sm_status sm_array_astype(sm_context* context, const sm_array* input, uint32_t dtype, sm_array_result* out) {
    namespace array = sm::detail::array;
    return sm::detail::api(context, [&] {
        if (!out) throw Error(SM_EINVAL, "null output");
        const array::Shape shape = array::checked_shape(input);
        const uint64_t width = array::dtype_size(dtype);
        if (array::is_complex(input->dtype) != array::is_complex(dtype))
            throw Error(SM_EINVAL, "real and complex arrays do not convert into each other");
        uint64_t out_bytes = 0;
        if (sm_checked_mul(shape.elements, width, &out_bytes) != SM_OK)
            throw Error(SM_EOVERFLOW, "output array byte size overflows");
        auto buffer = array::allocate(context, out_bytes);
        array::convert(input->data, input->dtype, sm::detail::mutable_data(buffer.get()), dtype, shape.elements);
        array::publish(out, std::move(buffer), dtype, shape);
    });
}

uint32_t sm_array_capabilities(void) {
#if defined(SM_ENABLE_METAL)
    return SM_CAPABILITY_METAL;
#else
    return 0;
#endif
}
