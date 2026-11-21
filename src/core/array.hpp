#pragma once
// Shared plumbing for array operations: descriptor checks, pooled output
// storage, precision conversion on the performance cores, and result/route
// publishing. Every family (ndimage, fft, ...) builds on these.
#include "sm_array.h"
#include "sm/detail/storage.hpp"

#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace sm::detail::array {
struct Shape {
    uint32_t ndim = 0;
    uint64_t dims[SM_MAX_NDIM] = {};
    uint64_t elements = 0;
};

// Bytes per element; SM_EINVAL for an unknown dtype.
uint64_t dtype_size(uint32_t dtype);
// Minimum natural scalar alignment (bytes) required for typed access.
uint64_t scalar_alignment(uint32_t dtype);
bool is_double(uint32_t dtype);       // SM_FLOAT64 or SM_COMPLEX128
bool is_complex(uint32_t dtype);      // SM_COMPLEX64 or SM_COMPLEX128
bool is_float(uint32_t dtype);        // SM_FLOAT32 or SM_FLOAT64
bool is_bool(uint32_t dtype);         // SM_BOOL
bool is_integer(uint32_t dtype);      // SM_INT8..SM_UINT64
bool is_signed_int(uint32_t dtype);   // SM_INT8..SM_INT64
bool is_unsigned_int(uint32_t dtype); // SM_UINT8..SM_UINT64
bool is_int_or_bool(uint32_t dtype);  // SM_BOOL or SM_INT8..SM_UINT64
bool is_real(uint32_t dtype);         // any valid non-complex dtype
// The single-precision counterpart (SM_COMPLEX64 for complex; SM_FLOAT32 for all real dtypes).
uint32_t single_of(uint32_t dtype);

// Validates a descriptor (sizes, version, dtype, rank, overflow, data pointer)
// and returns its shape. Zero-length arrays are valid.
Shape checked_shape(const sm_array* array);
void check_options(const sm_route_options* options);

// Owned storage whose bytes the caller writes completely before publishing.
// Comes from the context's buffer cache when a buffer of that capacity is idle.
BufferPtr allocate(sm_context*, uint64_t bytes);

struct ConvertReport {
    bool overflowed = false;     // finite double magnitude >= 0x1.ffffffp127 (rounded to +-inf in float32)
    bool inexact_int = false;    // 32/64-bit integer magnitude >= 2^24 (not exactly representable in float32)
    bool has_nonfinite = false;  // input contained at least one +-inf or NaN
    double max_finite = 0.0;     // largest |x| among finite elements
    double max_magnitude = 0.0;  // max_finite, or INFINITY if +-inf was present

    // Metal filter arithmetic flushes float32 subnormals to zero. Compare
    // the source peak before narrowing, including when the source is float32.
    const char* filter_input_range_error() const noexcept {
        if (inexact_int) return "integer input exceeds exact float32 range (|x| >= 2^24)";
        if (overflowed) return "float64 input overflowed float32";
        if (max_finite > 0.0 && max_finite < std::numeric_limits<float>::min())
            return "all finite nonzero input values underflow the normal float32 range";
        return nullptr;
    }
};

// UB-free arm64-exact saturating truncations from double to 32/64-bit integer.
inline int32_t double_to_i32_sat(double x) noexcept {
    if (std::isnan(x)) return 0;
    if (x >= 2147483647.0) return INT32_MAX;
    if (x <= -2147483648.0) return INT32_MIN;
    return static_cast<int32_t>(x);
}
inline uint32_t double_to_u32_sat(double x) noexcept {
    if (!(x >= 0.0)) return 0u;
    if (x >= 4294967295.0) return UINT32_MAX;
    return static_cast<uint32_t>(x);
}
inline int64_t double_to_i64_sat(double x) noexcept {
    if (std::isnan(x)) return 0;
    if (x >= 9223372036854775808.0) return INT64_MAX;
    if (x <= -9223372036854775808.0) return INT64_MIN;
    return static_cast<int64_t>(x);
}
inline uint64_t double_to_u64_sat(double x) noexcept {
    if (!(x >= 0.0)) return 0u;
    if (x >= 18446744073709551616.0) return UINT64_MAX;
    return static_cast<uint64_t>(x);
}

// Scalar SciPy-matching double -> real-dtype truncation casts:
//   dense_safe = false -> NI_LineBufferToArray semantics (separable & composite)
//   dense_safe = true  -> CASE_FILTER_OUT_SAFE semantics for bool/unsigned (dense)
template<class DstT>
inline DstT cast_double_ndimage(double x, bool dense_safe = false) noexcept {
    (void)dense_safe;
    if constexpr (std::is_same_v<DstT, float>) {
        return static_cast<float>(x);
    } else if constexpr (std::is_same_v<DstT, double>) {
        return x;
    } else if constexpr (std::is_same_v<DstT, int8_t>) {
        return static_cast<int8_t>(static_cast<uint8_t>(double_to_i32_sat(x)));
    } else if constexpr (std::is_same_v<DstT, int16_t>) {
        return static_cast<int16_t>(static_cast<uint16_t>(double_to_i32_sat(x)));
    } else if constexpr (std::is_same_v<DstT, int32_t>) {
        return double_to_i32_sat(x);
    } else if constexpr (std::is_same_v<DstT, int64_t>) {
        return double_to_i64_sat(x);
    } else if constexpr (std::is_same_v<DstT, uint8_t>) {
        if (dense_safe && x <= -1.0) return static_cast<uint8_t>(0u - static_cast<uint8_t>(double_to_i32_sat(-x)));
        return static_cast<uint8_t>(double_to_i32_sat(x));
    } else if constexpr (std::is_same_v<DstT, uint16_t>) {
        if (dense_safe && x <= -1.0) return static_cast<uint16_t>(0u - static_cast<uint16_t>(double_to_i32_sat(-x)));
        return static_cast<uint16_t>(double_to_i32_sat(x));
    } else if constexpr (std::is_same_v<DstT, uint32_t>) {
        if (dense_safe && x <= -1.0) return static_cast<uint32_t>(0u - double_to_u32_sat(-x));
        return double_to_u32_sat(x);
    } else if constexpr (std::is_same_v<DstT, uint64_t>) {
        if (dense_safe && x <= -1.0) return static_cast<uint64_t>(0ull - double_to_u64_sat(-x));
        return double_to_u64_sat(x);
    }
}

// dst[i] = (dst type) src[i] for `elements` elements, across the P-core lanes.
// Converts across real dtypes (matching NumPy astype on arm64) or between
// single and double complex, or copies when dtypes match. src and dst must not
// overlap. When `report` is non-null and converting to single precision,
// populates `report` in the same pass.
void convert(const void* src, uint32_t src_dtype, void* dst, uint32_t dst_dtype, uint64_t elements,
             ConvertReport* report = nullptr);

// Threaded P-core post-cast from single-precision GPU output to an ndimage
// real output dtype using SciPy's line-buffer (dense_safe=false) or dense
// (dense_safe=true) conversion rules.
void ndimage_cast_from_float(const float* src, uint32_t dst_dtype, void* dst, uint64_t elements, bool dense_safe);

// A single-precision copy of a host array in owned (device-visible) storage.
// When `raw_bool` is true, SM_BOOL inputs are read as raw uint8_t bytes (matching
// SciPy's ndimage filter input conversion rather than NumPy's normalized bool astype).
BufferPtr load_single(sm_context*, const sm_array& array, const Shape& shape, ConvertReport* report = nullptr,
                      bool raw_bool = false);

void check_route_info(const sm_route_info* info);
void publish(sm_array_result* out, BufferPtr buffer, uint32_t dtype, const Shape& shape) noexcept;
void report(sm_route_info* info, uint32_t engine, uint32_t compute_dtype, const char* reason = "") noexcept;
} // namespace sm::detail::array
