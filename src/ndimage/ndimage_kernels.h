// ndimage_kernels.h - kernel <-> host contract for the ndimage GPU routes.
// Follows the rules in ../kernels/sm_kernels.h; change kernels and host together.
#ifndef SM_NDIMAGE_KERNELS_H
#define SM_NDIMAGE_KERNELS_H

#include "../kernels/sm_kernels.h"

// sm_ndimage_correlate1d_{any,even,odd}_{f32,f64}: one 1-D correlation pass
// along one axis of an N-D float32 array, one output element per thread. The
// f64 kernels store IEEE double bit patterns (the exact widening of their
// float32 result).
// Bindings: 1 source (float32), 2 destination, 3 weights (float32).
//
// The grid is the array itself, fastest-varying memory direction first, and
// threads outside `extent` return at once:
//   axis_in_y == 0 (lines are contiguous):  x = position along the axis,
//                                           y = line, z unused (extent 1)
//   axis_in_y == 1:  x = element within one step of the axis (extent = stride),
//                    y = position along the axis, z = outer block
// so element index = (z * extent[1] + y) * extent[0] + x in both layouts.
// Indices are 32-bit: the host sends arrays of at most 2^31 - 1 elements with
// axis lengths and windows of at most SM_NDIMAGE_GPU_MAX_EXTENT.
//
// Tap j of position p reads extension position p - before + j; `mode` is an
// sm_ndimage_mode. The kernel names say how the taps are summed, in SciPy's
// order: even (symmetric windows of odd length) and odd (antisymmetric) sum
// the centre tap, then pairs from the outermost inwards with the weights of
// the lower half; any sums the last tap, then the rest in order.
#define SM_NDIMAGE_GPU_MAX_EXTENT (1u << 29)
#define SM_NDIMAGE_SYMMETRY_NONE 0u
#define SM_NDIMAGE_SYMMETRY_EVEN 1u
#define SM_NDIMAGE_SYMMETRY_ODD 2u
struct SMNdimageCorrelateParams {
    sm_u32 extent[3];
    sm_u32 axis_in_y;
    sm_u32 length;        // positions along the axis
    sm_u32 stride;        // elements between consecutive positions
    sm_u32 taps;
    sm_u32 weight_offset; // first weight of this pass in the weights buffer
    sm_u32 before;
    sm_u32 mode;
    float cval;
    sm_u32 reserved;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(SMNdimageCorrelateParams) == 48,
              "kernel parameter layouts changed: update ndimage_kernels.h and the kernels together");
#endif

// sm_ndimage_combine_{f32,f64}: element-wise chain combiner for composite filters.
// op codes:
//   0: copy (dst = a)
//   1: sum2 (dst = a + b)
//   2: sqrt_sq1 (dst = sqrt(a * a))
//   3: sum_sq2 (dst = a * a + b * b)
//   4: add_sq (dst = a + b * b)
//   5: sqrt_sum_sq2 (dst = sqrt(a * a + b * b))
//   6: sqrt_add_sq (dst = sqrt(a + b * b))
#define SM_NDIMAGE_COMBINE_OP_COPY 0u
#define SM_NDIMAGE_COMBINE_OP_SUM2 1u
#define SM_NDIMAGE_COMBINE_OP_SQRT_SQ1 2u
#define SM_NDIMAGE_COMBINE_OP_SUM_SQ2 3u
#define SM_NDIMAGE_COMBINE_OP_ADD_SQ 4u
#define SM_NDIMAGE_COMBINE_OP_SQRT_SUM_SQ2 5u
#define SM_NDIMAGE_COMBINE_OP_SQRT_ADD_SQ 6u

struct SMNdimageCombineParams {
    sm_u32 elements;
    sm_u32 op;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(SMNdimageCombineParams) == 8,
              "SMNdimageCombineParams layout changed: update ndimage_kernels.h and the kernels together");
#endif

#endif // SM_NDIMAGE_KERNELS_H
