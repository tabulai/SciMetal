// signal_kernels.h - kernel <-> host contract for SciMetal's signal family.
// Rules as in ../kernels/sm_kernels.h: fixed-width fields, 8-byte fields first,
// every struct a multiple of 8 bytes; host static_asserts check sizes.
#ifndef SM_SIGNAL_KERNELS_H
#define SM_SIGNAL_KERNELS_H

#include "../kernels/sm_kernels.h"

#define SM_SIGNAL_PAD_SRC_COMPLEX 1u
#define SM_SIGNAL_PAD_DST_COMPLEX 2u
#define SM_SIGNAL_PAD_CONJ        4u

#define SM_SIGNAL_COMBINE_SRC_COMPLEX 1u
#define SM_SIGNAL_COMBINE_DST_COMPLEX 2u
#define SM_SIGNAL_COMBINE_DST_DOUBLE  4u
#define SM_SIGNAL_COMBINE_REV_CONJ    8u

#define SM_SIGNAL_MAX_SPEC_DIMS 16u

struct SMSignalPadParams {
    sm_u32 in_shape[8];
    sm_u32 in_stride[8];
    sm_u32 nsteps[8];
    sm_u32 step[8];
    sm_u32 fft_len[8];
    sm_u32 dst_stride[8];
    sm_u32 ndim;
    sm_u32 dst_elements;
    sm_u32 flags;
    sm_u32 reverse_mask;
};

struct SMSignalMulInplaceParams {
    sm_u32 elements;
    sm_u32 reserved;
};

struct SMSignalMulBroadcastParams {
    sm_u32 shape[SM_SIGNAL_MAX_SPEC_DIMS];
    sm_u32 out_stride[SM_SIGNAL_MAX_SPEC_DIMS];
    sm_u32 a_stride[SM_SIGNAL_MAX_SPEC_DIMS];
    sm_u32 b_stride[SM_SIGNAL_MAX_SPEC_DIMS];
    sm_u32 ndim;
    sm_u32 elements;
};

struct SMSignalCombineParams {
    sm_u32 out_shape[8];
    sm_u32 out_stride[8];
    sm_u32 conv_stride[8];
    sm_u32 block_stride[8];
    sm_u32 crop_start[8];
    sm_u32 block_size[8];
    sm_u32 step[8];
    sm_u32 nsteps[8];
    sm_u32 ndim;
    sm_u32 out_elements;
    sm_u32 flags;
    float scale;
};

struct SMSignalRevConjParams {
    sm_u32 elements;
    sm_u32 conj;
};

struct SMSignalFiniteParams {
    sm_u64 count;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(SMSignalPadParams) == 208,
              "SMSignalPadParams layout changed: update signal_kernels.h and signal_kernels.metal together");
static_assert(sizeof(SMSignalMulInplaceParams) == 8,
              "SMSignalMulInplaceParams layout changed: update signal_kernels.h and signal_kernels.metal together");
static_assert(sizeof(SMSignalMulBroadcastParams) == 264,
              "SMSignalMulBroadcastParams layout changed: update signal_kernels.h and signal_kernels.metal together");
static_assert(sizeof(SMSignalCombineParams) == 272,
              "SMSignalCombineParams layout changed: update signal_kernels.h and signal_kernels.metal together");
static_assert(sizeof(SMSignalRevConjParams) == 8,
              "SMSignalRevConjParams layout changed: update signal_kernels.h and signal_kernels.metal together");
static_assert(sizeof(SMSignalFiniteParams) == 8,
              "SMSignalFiniteParams layout changed: update signal_kernels.h and signal_kernels.metal together");
#endif

#endif // SM_SIGNAL_KERNELS_H
