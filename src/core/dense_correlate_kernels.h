// dense_correlate_kernels.h - kernel <-> host contract for the shared dense
// N-D correlation primitive.
// Follows the rules in ../kernels/sm_kernels.h; change kernels and host together.
#ifndef SM_DENSE_CORRELATE_KERNELS_H
#define SM_DENSE_CORRELATE_KERNELS_H

#include "../kernels/sm_kernels.h"

#define SM_DENSE_GPU_MAX_EXTENT (1u << 28)
#define SM_DENSE_GPU_MAX_TAPS (1u << 20)

struct SMDenseTap {
    sm_i32 coord[8];
    sm_i32 linear_offset;
    sm_i32 reserved;
    float weight_re;
    float weight_im;
};

struct SMDenseCorrelateParams {
    sm_u32 out_shape[8];
    sm_u32 out_stride[8];
    sm_u32 in_shape[8];
    sm_u32 in_stride[8];
    sm_u32 mode[8];
    sm_i32 interior_lo[8];
    sm_i32 interior_hi[8];
    sm_u32 ndim;
    sm_u32 num_taps;
    sm_u32 out_elements;
    sm_u32 all_interior;
    float cval_re;
    float cval_im;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(SMDenseTap) == 48,
              "SMDenseTap layout changed: update dense_correlate_kernels.h and dense_correlate.metal together");
static_assert(sizeof(SMDenseCorrelateParams) == 248,
              "SMDenseCorrelateParams layout changed: update dense_correlate_kernels.h and dense_correlate.metal together");
#endif

#endif // SM_DENSE_CORRELATE_KERNELS_H
