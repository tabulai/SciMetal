#pragma once
// Chooses the engine and compute precision for one call.
//
// Engines: the CPU engine computes in the input's own precision (so float64
// stays exact there whatever the policy); the GPU engine computes in single
// precision. The policy therefore decides only whether a double-precision
// input may go to the GPU.
#include "ms_array.h"

#include <cstdint>

namespace ms::detail::router {
// Per-family crossover for MS_EXECUTION_AUTO: the smallest work size (in the
// family's own unit, e.g. elements x taps for a stencil) worth a GPU round
// trip. Calibrated on applegpu_g13d (M1 Ultra); see probes/README.md.
struct Family {
    const char* name;
    uint64_t gpu_min_work;
};

struct Decision {
    uint32_t engine;         // MS_EXECUTION_CPU or MS_EXECUTION_GPU
    uint32_t compute_dtype;  // ms_dtype the arithmetic runs in
};

// Throws MS_ENOTSUP when a forced GPU route cannot run (see ms_route_options).
Decision decide(ms_context*, const Family&, const ms_route_options&, uint32_t input_dtype, uint64_t work);
} // namespace ms::detail::router
