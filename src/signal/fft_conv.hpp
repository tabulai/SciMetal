#pragma once
// Internal header shared between src/signal/convolve.cpp and src/signal/fft_conv_gpu.cpp.
#include "ms_signal.h"
#include "core/array.hpp"
#include "fft/plan.hpp"

#include <cstdint>

namespace ms::detail::signal {

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
    uint32_t axes[MS_MAX_NDIM] = {};
    uint32_t axis_count = 0;
    uint64_t fft_len[MS_MAX_NDIM] = {};
    uint64_t step1[MS_MAX_NDIM] = {};
    uint64_t step2[MS_MAX_NDIM] = {};
    uint64_t nsteps1[MS_MAX_NDIM] = {};
    uint64_t nsteps2[MS_MAX_NDIM] = {};
    uint64_t nsteps_out[MS_MAX_NDIM] = {};
    uint64_t overlap[MS_MAX_NDIM] = {};
    uint64_t crop_start[MS_MAX_NDIM] = {};
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
                          uint32_t center = MS_SIGNAL_CENTER_ND);

#if defined(MS_ENABLE_METAL)
BufferPtr run_fft_conv_gpu(ms_context* context, const ms_array& in1, const ms_array& in2,
                           const FftConvPlan& plan, const ms_route_options& route_options,
                           uint32_t& out_compute_dtype, uint32_t* out_nf_flags = nullptr);
#endif

} // namespace ms::detail::signal
