// Runtime smoke kernel: exercises 2-D dispatch grids through Batch.
#include <metal_stdlib>
#include "sm_kernels.h"
using namespace metal;

kernel void sm_smoke_saxpy2d(constant SMSmokeParams& p [[buffer(0)]],
                             device const float* x [[buffer(1)]],
                             device float* y [[buffer(2)]],
                             uint3 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.width || pos.y >= p.height) return;
    const ulong i = ulong(pos.y) * p.width + pos.x;
    y[i] = p.alpha * x[i] + y[i];
}

// Indexed bindings and a function-constant specialization, as the vendored MLX
// kernels use: out[i] = (negate ? -scale : scale) * in[i], with in at buffer 3,
// out at buffer 5, the count at 2 and the scale at 7.
constant bool sm_smoke_negate [[function_constant(4)]];
kernel void sm_smoke_scale(device const float* in [[buffer(3)]],
                           device float* out [[buffer(5)]],
                           constant uint& count [[buffer(2)]],
                           constant float& scale [[buffer(7)]],
                           uint pos [[thread_position_in_grid]]) {
    if (pos >= count) return;
    out[pos] = (sm_smoke_negate ? -scale : scale) * in[pos];
}
