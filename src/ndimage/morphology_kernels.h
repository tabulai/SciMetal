// morphology_kernels.h - kernel <-> host contract for ndimage dense min/max and rank/median GPU routes.
// Follows the rules in ../kernels/sm_kernels.h; change kernels and host together.
#ifndef SM_NDIMAGE_MORPHOLOGY_KERNELS_H
#define SM_NDIMAGE_MORPHOLOGY_KERNELS_H

#include "../kernels/sm_kernels.h"

#define SM_NDIMAGE_MORPH_MAX_TAPS (1u << 20)
#define SM_NDIMAGE_RANK_SMALL_MAX_S 32u
#define SM_NDIMAGE_RANK_MEDIUM_MAX_S 64u

struct SMNdimageMorphTap {
    sm_i32 coord[8];
    sm_i32 linear_offset;
    float struct_offset;
};

struct SMNdimageDenseMinMaxParams {
    sm_u32 ndim;
    sm_u32 num_taps;
    sm_u32 elements;
    sm_u32 mode;
    sm_u32 all_interior;
    sm_u32 reserved0;
    sm_u32 in_shape[8];
    sm_u32 in_stride[8];
    sm_i32 interior_lo[8];
    sm_i32 interior_hi[8];
    float cval;
    sm_u32 reserved1;
};

struct SMNdimageRankParams {
    sm_u32 ndim;
    sm_u32 num_taps;
    sm_u32 elements;
    sm_u32 rank;
    sm_u32 mode;
    sm_u32 all_interior;
    sm_u32 in_shape[8];
    sm_u32 in_stride[8];
    sm_i32 interior_lo[8];
    sm_i32 interior_hi[8];
    float cval;
    sm_u32 reserved1;
};

// Dedicated 2-D tiled square median (3x3, 5x5, 7x7) parameters.
// Grid: 2-D threadgroups of (16, 16) covering (width, height).
struct SMNdimageMedian2DParams {
    sm_u32 height;
    sm_u32 width;
    sm_i32 start_y; // -(k/2) - origin_y
    sm_i32 start_x; // -(k/2) - origin_x
    sm_u32 mode;
    float cval;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(SMNdimageMorphTap) == 40,
              "SMNdimageMorphTap layout changed: update morphology_kernels.h and kernels together");
static_assert(sizeof(SMNdimageDenseMinMaxParams) == 160,
              "SMNdimageDenseMinMaxParams layout changed: update morphology_kernels.h and kernels together");
static_assert(sizeof(SMNdimageRankParams) == 160,
              "SMNdimageRankParams layout changed: update morphology_kernels.h and kernels together");
static_assert(sizeof(SMNdimageMedian2DParams) == 24,
              "SMNdimageMedian2DParams layout changed: update morphology_kernels.h and kernels together");
#endif

#endif // SM_NDIMAGE_MORPHOLOGY_KERNELS_H
