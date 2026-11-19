// signal_kernels.h - kernel <-> host contract for metal-sci's signal family.
// Rules as in ../kernels/ms_kernels.h: fixed-width fields, 8-byte fields first,
// every struct a multiple of 8 bytes; host static_asserts check sizes.
#ifndef MS_SIGNAL_KERNELS_H
#define MS_SIGNAL_KERNELS_H

#include "../kernels/ms_kernels.h"

#define MS_SIGNAL_PAD_SRC_COMPLEX 1u
#define MS_SIGNAL_PAD_DST_COMPLEX 2u
#define MS_SIGNAL_PAD_CONJ        4u

#define MS_SIGNAL_COMBINE_SRC_COMPLEX 1u
#define MS_SIGNAL_COMBINE_DST_COMPLEX 2u
#define MS_SIGNAL_COMBINE_DST_DOUBLE  4u
#define MS_SIGNAL_COMBINE_REV_CONJ    8u

#define MS_SIGNAL_MAX_SPEC_DIMS 16u

struct MSSignalPadParams {
    ms_u32 in_shape[8];
    ms_u32 in_stride[8];
    ms_u32 nsteps[8];
    ms_u32 step[8];
    ms_u32 fft_len[8];
    ms_u32 dst_stride[8];
    ms_u32 ndim;
    ms_u32 dst_elements;
    ms_u32 flags;
    ms_u32 reverse_mask;
};

struct MSSignalMulInplaceParams {
    ms_u32 elements;
    ms_u32 reserved;
};

struct MSSignalMulBroadcastParams {
    ms_u32 shape[MS_SIGNAL_MAX_SPEC_DIMS];
    ms_u32 out_stride[MS_SIGNAL_MAX_SPEC_DIMS];
    ms_u32 a_stride[MS_SIGNAL_MAX_SPEC_DIMS];
    ms_u32 b_stride[MS_SIGNAL_MAX_SPEC_DIMS];
    ms_u32 ndim;
    ms_u32 elements;
};

struct MSSignalCombineParams {
    ms_u32 out_shape[8];
    ms_u32 out_stride[8];
    ms_u32 conv_stride[8];
    ms_u32 block_stride[8];
    ms_u32 crop_start[8];
    ms_u32 block_size[8];
    ms_u32 step[8];
    ms_u32 nsteps[8];
    ms_u32 ndim;
    ms_u32 out_elements;
    ms_u32 flags;
    float scale;
};

struct MSSignalRevConjParams {
    ms_u32 elements;
    ms_u32 conj;
};

struct MSSignalFiniteParams {
    ms_u64 count;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(MSSignalPadParams) == 208,
              "MSSignalPadParams layout changed: update signal_kernels.h and signal_kernels.metal together");
static_assert(sizeof(MSSignalMulInplaceParams) == 8,
              "MSSignalMulInplaceParams layout changed: update signal_kernels.h and signal_kernels.metal together");
static_assert(sizeof(MSSignalMulBroadcastParams) == 264,
              "MSSignalMulBroadcastParams layout changed: update signal_kernels.h and signal_kernels.metal together");
static_assert(sizeof(MSSignalCombineParams) == 272,
              "MSSignalCombineParams layout changed: update signal_kernels.h and signal_kernels.metal together");
static_assert(sizeof(MSSignalRevConjParams) == 8,
              "MSSignalRevConjParams layout changed: update signal_kernels.h and signal_kernels.metal together");
static_assert(sizeof(MSSignalFiniteParams) == 8,
              "MSSignalFiniteParams layout changed: update signal_kernels.h and signal_kernels.metal together");
#endif

#endif // MS_SIGNAL_KERNELS_H
