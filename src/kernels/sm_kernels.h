// sm_kernels.h - the kernel <-> host contract for SciMetal GPU routes.
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
#ifndef SM_KERNELS_H
#define SM_KERNELS_H

#ifdef __METAL_VERSION__
typedef ulong sm_u64;
typedef long sm_i64;
typedef uint sm_u32;
typedef int sm_i32;
#else
#include <cstdint>
typedef uint64_t sm_u64;
typedef int64_t sm_i64;
typedef uint32_t sm_u32;
typedef int32_t sm_i32;
#endif

// The 1-D dispatch convention (Grid::linear): 256-thread threadgroups.
// Kernels with their own geometry state it next to their parameters.
#define SM_GPU_THREADS 256u

// Element storage a kernel reads or writes. F64 means the buffer holds IEEE
// double bit patterns that the kernel narrows on load or widens on store, so
// float64 NumPy memory is consumed and produced without a host conversion pass.
#define SM_ELEM_F32 0u
#define SM_ELEM_F64 1u

// Smoke test: y[i] = alpha * x[i] + y[i] over a width x height grid.
// Bindings: 1 x, 2 y. Threadgroups of 16 x 16 threads.
#define SM_SMOKE_GROUP 16u
struct SMSmokeParams {
    sm_u64 width;
    sm_u64 height;
    float alpha;
    sm_u32 reserved;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(SMSmokeParams) == 24, "kernel parameter layouts changed: update sm_kernels.h and kernels together");
#endif

#endif // SM_KERNELS_H
