// ms_kernels.h - the kernel <-> host contract for metal-sci GPU routes.
//
// Included by Metal Shading Language kernels and by C++ host code. It is the
// single source of truth for parameter structs (bound with setBytes at buffer
// 0), buffer binding tables and threadgroup geometry.
//
// Rules (inherited from metal-arrow): fixed-width fields only, 8-byte fields
// first, every struct a multiple of 8 bytes; host static_asserts check sizes.
// Buffers bind at their allocation start, and element offsets travel in the
// parameters, so binding offsets never need Metal-specific alignment. Change
// kernels and host together.
#ifndef MS_KERNELS_H
#define MS_KERNELS_H

#ifdef __METAL_VERSION__
typedef ulong ms_u64;
typedef long ms_i64;
typedef uint ms_u32;
typedef int ms_i32;
#else
#include <cstdint>
typedef uint64_t ms_u64;
typedef int64_t ms_i64;
typedef uint32_t ms_u32;
typedef int32_t ms_i32;
#endif

// The 1-D dispatch convention (Grid::linear): 256-thread threadgroups.
// Kernels with their own geometry state it next to their parameters.
#define MS_GPU_THREADS 256u

// Element storage a kernel reads or writes. F64 means the buffer holds IEEE
// double bit patterns that the kernel narrows on load or widens on store, so
// float64 NumPy memory is consumed and produced without a host conversion pass.
#define MS_ELEM_F32 0u
#define MS_ELEM_F64 1u

// Smoke test: y[i] = alpha * x[i] + y[i] over a width x height grid.
// Bindings: 1 x, 2 y. Threadgroups of 16 x 16 threads.
#define MS_SMOKE_GROUP 16u
struct MSSmokeParams {
    ms_u64 width;
    ms_u64 height;
    float alpha;
    ms_u32 reserved;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(MSSmokeParams) == 24, "kernel parameter layouts changed: update ms_kernels.h and kernels together");
#endif

#endif // MS_KERNELS_H
