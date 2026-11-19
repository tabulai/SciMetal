// dense_correlate_kernels.h - kernel <-> host contract for the shared dense
// N-D correlation primitive.
// Follows the rules in ../kernels/ms_kernels.h; change kernels and host together.
#ifndef MS_DENSE_CORRELATE_KERNELS_H
#define MS_DENSE_CORRELATE_KERNELS_H

#include "../kernels/ms_kernels.h"

#define MS_DENSE_GPU_MAX_EXTENT (1u << 28)
#define MS_DENSE_GPU_MAX_TAPS (1u << 20)

struct MSDenseTap {
    ms_i32 coord[8];
    ms_i32 linear_offset;
    ms_i32 reserved;
    float weight_re;
    float weight_im;
};

struct MSDenseCorrelateParams {
    ms_u32 out_shape[8];
    ms_u32 out_stride[8];
    ms_u32 in_shape[8];
    ms_u32 in_stride[8];
    ms_u32 mode[8];
    ms_i32 interior_lo[8];
    ms_i32 interior_hi[8];
    ms_u32 ndim;
    ms_u32 num_taps;
    ms_u32 out_elements;
    ms_u32 all_interior;
    float cval_re;
    float cval_im;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(MSDenseTap) == 48,
              "MSDenseTap layout changed: update dense_correlate_kernels.h and dense_correlate.metal together");
static_assert(sizeof(MSDenseCorrelateParams) == 248,
              "MSDenseCorrelateParams layout changed: update dense_correlate_kernels.h and dense_correlate.metal together");
#endif

#endif // MS_DENSE_CORRELATE_KERNELS_H
