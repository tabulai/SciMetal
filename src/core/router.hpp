#pragma once
// Chooses the engine and compute precision for one call.
//
// Engines: the CPU engine computes in the input's own precision (so float64
// stays exact there whatever the policy); the GPU engine computes in single
// precision. The policy therefore decides only whether a double-precision
// input may go to the GPU.
#include "sm_array.h"

#include <cstdint>

namespace sm::detail::router {
// Per-family crossover for SM_EXECUTION_AUTO: the smallest work size (in the
// family's own unit, e.g. elements x taps for a stencil) worth a GPU round
// trip. Calibrated on applegpu_g13d (M1 Ultra); see probes/README.md.
struct Family {
    const char* name;
    uint64_t gpu_min_work;
};

struct Decision {
    uint32_t engine;         // SM_EXECUTION_CPU or SM_EXECUTION_GPU
    uint32_t compute_dtype;  // sm_dtype the arithmetic runs in
};

// Throws SM_ENOTSUP when a forced GPU route cannot run (see sm_route_options).
Decision decide(sm_context*, const Family&, const sm_route_options&, uint32_t input_dtype, uint64_t work);
} // namespace sm::detail::router
