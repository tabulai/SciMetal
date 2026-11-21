// E1 probe: MLX "steel" fused GEMM instantiated in our own metallib.
//
// The kernel body is MLX's (MIT licence), taken verbatim from the headers that
// ship in the mlx wheel. We only choose which template instances to emit.
// Host names follow MLX's own naming scheme so the two can be compared 1:1.
//
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h"

#define sm_instantiate_gemm(tname, trans_a, trans_b, bm, bn, bk, wm, wn)  \
  instantiate_kernel(                                                     \
      "steel_gemm_fused_" #tname "_float32_float32"                       \
      "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn,                  \
      gemm, float, bm, bn, bk, wm, wn, trans_a, trans_b, float)

#define sm_instantiate_all(bm, bn, bk, wm, wn)               \
  sm_instantiate_gemm(nn, false, false, bm, bn, bk, wm, wn)  \
  sm_instantiate_gemm(nt, false, true , bm, bn, bk, wm, wn)  \
  sm_instantiate_gemm(tn, true , false, bm, bn, bk, wm, wn)  \
  sm_instantiate_gemm(tt, true , true , bm, bn, bk, wm, wn)

// The shapes MLX ships for float32 (steel_gemm_fused.metal).
// On a 'd' (Ultra) device, matmul.cpp picks:
//   M*N >= 2^20, float32            -> bm64 bn64 bk16 wm2 wn2   (the default)
//   M*N <  2^20, float32, nn/tn/tt  -> bm64 bn32 bk32 wm2 wn2
//   M*N <  2^20, float32, nt        -> bm32 bn64 bk16 wm1 wn2
sm_instantiate_all(64, 64, 16, 2, 2)
sm_instantiate_all(64, 32, 32, 2, 2)
sm_instantiate_all(32, 64, 16, 1, 2)
sm_instantiate_all(32, 32, 16, 2, 2)
// clang-format on
