#pragma once
// Internal header shared between src/signal/convolve.cpp and src/signal/fft_conv_gpu.cpp.
#include "sm_signal.h"
#include "core/array.hpp"
#include "fft/plan.hpp"

#include <cstdint>

namespace sm::detail::signal {

struct FftConvPlan {
    uint32_t ndim = 0;
    array::Shape s1;
    array::Shape s2;
    array::Shape out_shape;
    uint32_t out_dtype = 0;
    bool is_complex = false;
    bool real_last_axis = false;
    bool swapped = false;
    bool correlate = false;
    bool use_oa = false;
    uint32_t raw_axis_mask = 0;
    uint32_t axis_mask = 0;
    uint32_t axes[SM_MAX_NDIM] = {};
    uint32_t axis_count = 0;
    uint64_t fft_len[SM_MAX_NDIM] = {};
    uint64_t step1[SM_MAX_NDIM] = {};
    uint64_t step2[SM_MAX_NDIM] = {};
    uint64_t nsteps1[SM_MAX_NDIM] = {};
    uint64_t nsteps2[SM_MAX_NDIM] = {};
    uint64_t nsteps_out[SM_MAX_NDIM] = {};
    uint64_t overlap[SM_MAX_NDIM] = {};
    uint64_t crop_start[SM_MAX_NDIM] = {};
    uint64_t single_fft_elements = 0;
    uint64_t oa_fft_elements = 0;
    uint64_t effective_fft_elements = 0;
    uint64_t overlap_elements = 1;
};

uint32_t promote_dtype(uint32_t d1, uint32_t d2);

FftConvPlan plan_fft_conv(const array::Shape& s1_in, const array::Shape& s2_in,
                          uint32_t out_dtype, uint32_t mode, bool correlate,
                          bool allow_oa, bool force_oa,
                          const uint32_t* axes_in, uint32_t axis_count_in,
                          const fft::Limits& limits,
                          uint32_t center = SM_SIGNAL_CENTER_ND);

#if defined(SM_ENABLE_METAL)
BufferPtr run_fft_conv_gpu(sm_context* context, const sm_array& in1, const sm_array& in2,
                           const FftConvPlan& plan, const sm_route_options& route_options,
                           uint32_t& out_compute_dtype, uint32_t* out_nf_flags = nullptr);
#endif

} // namespace sm::detail::signal
