#pragma once
// Dense N-D min/max (morphology) and rank/median/percentile filters.
#include "sm_ndimage.h"
#include "core/array.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sm::detail::ndimage {

struct MorphTap {
    int64_t coord[SM_MAX_NDIM] = {};
    int64_t linear_offset = 0;
    double struct_offset_f64 = 0.0;
    float struct_offset_f32 = 0.0f;
};

struct MorphTapTable {
    std::vector<MorphTap> taps;
    int64_t interior_lo[SM_MAX_NDIM] = {};
    int64_t interior_hi[SM_MAX_NDIM] = {};
    bool has_interior = false;
    bool all_interior = false;
    bool has_structure = false;
    bool structure_overflow_f32 = false;
    bool structure_non_integral = false;
    double structure_max_abs = 0.0;
};

struct DenseMinMaxPlan {
    array::Shape shape;
    uint32_t in_dtype = 0;
    uint32_t out_dtype = 0;
    bool is_max = false;
    uint32_t mode = 0;
    double cval = 0.0;
    uint32_t fshape[SM_MAX_NDIM] = {};
    int64_t start[SM_MAX_NDIM] = {};
    MorphTapTable taps;
};

struct RankPlan {
    array::Shape shape;
    uint32_t in_dtype = 0;
    uint32_t out_dtype = 0;
    uint32_t rank = 0;
    uint32_t mode = 0;
    double cval = 0.0;
    uint32_t fshape[SM_MAX_NDIM] = {};
    int64_t origins[SM_MAX_NDIM] = {};
    int64_t start[SM_MAX_NDIM] = {};
    bool all_true_footprint = false;
    MorphTapTable taps;
};

// CPU engines (threaded across P-cores, bit-identical to SciPy 1.18.1 arm64).
BufferPtr run_cpu_dense_minmax(sm_context*, const DenseMinMaxPlan&, const void* input);
BufferPtr run_cpu_rank(sm_context*, const RankPlan&, const void* input);

// GPU engines (Metal single precision, one command buffer per call).
bool gpu_supports_dense_minmax(const DenseMinMaxPlan&) noexcept;
bool gpu_supports_rank(const RankPlan&) noexcept;

BufferPtr run_gpu_dense_minmax(sm_context*, const DenseMinMaxPlan&, const sm_array& input,
                               bool allow_cpu_fallback, std::string& overflow_reason);
BufferPtr run_gpu_rank(sm_context*, const RankPlan&, const sm_array& input,
                       bool allow_cpu_fallback, std::string& overflow_reason);

} // namespace sm::detail::ndimage
