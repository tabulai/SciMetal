#pragma once
// Shared plumbing for array operations: descriptor checks, pooled output
// storage, precision conversion on the performance cores, and result/route
// publishing. Every family (ndimage, fft, ...) builds on these.
#include "ms_array.h"
#include "ms/detail/storage.hpp"

#include <cstdint>

namespace ms::detail::array {
struct Shape {
    uint32_t ndim = 0;
    uint64_t dims[MS_MAX_NDIM] = {};
    uint64_t elements = 0;
};

// Bytes per element; MS_EINVAL for an unknown dtype.
uint64_t dtype_size(uint32_t dtype);
bool is_double(uint32_t dtype);   // MS_FLOAT64 or MS_COMPLEX128
bool is_complex(uint32_t dtype);
// The single-precision counterpart (identity for single-precision dtypes).
uint32_t single_of(uint32_t dtype);

// Validates a descriptor (sizes, version, dtype, rank, overflow, data pointer)
// and returns its shape. Zero-length arrays are valid.
Shape checked_shape(const ms_array* array);
void check_options(const ms_route_options* options);

// Owned storage whose bytes the caller writes completely before publishing.
// Comes from the context's buffer cache when a buffer of that capacity is idle.
BufferPtr allocate(ms_context*, uint64_t bytes);

struct ConvertReport {
    bool overflowed = false;     // finite double magnitude >= 0x1.ffffffp127 (rounded to +-inf in float32)
    bool has_nonfinite = false;  // input contained at least one +-inf or NaN
    double max_finite = 0.0;     // largest |x| among finite elements
    double max_magnitude = 0.0;  // max_finite, or INFINITY if +-inf was present
};

// dst[i] = (dst type) src[i] for `elements` elements, across the P-core lanes.
// Converts between single and double precision of the same kind (real or
// complex), or copies when the dtypes match. src and dst must not overlap.
// When `report` is non-null and narrowing double -> single, populates `report`
// in the same pass.
void convert(const void* src, uint32_t src_dtype, void* dst, uint32_t dst_dtype, uint64_t elements,
             ConvertReport* report = nullptr);

// A single-precision copy of a host array in owned (device-visible) storage.
BufferPtr load_single(ms_context*, const ms_array& array, const Shape& shape, ConvertReport* report = nullptr);

void check_route_info(const ms_route_info* info);
void publish(ms_array_result* out, BufferPtr buffer, uint32_t dtype, const Shape& shape) noexcept;
void report(ms_route_info* info, uint32_t engine, uint32_t compute_dtype, const char* reason = "") noexcept;
} // namespace ms::detail::array
