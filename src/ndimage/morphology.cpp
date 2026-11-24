// scipy.ndimage dense min/max morphology and rank/median/percentile filters:
// argument validation, routing, and C ABI entrypoints.
#include "morphology.hpp"
#include "morphology_kernels.h"
#include "core/router.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace sm::detail::ndimage {
namespace {

// Calibrated AUTO crossovers on applegpu_g13d (M1 Ultra, 16 P-cores;
// bench/ndimage_morphology.py --calibrate):
// - Dense min/max: 10M element-taps (disk 5x5 crosses between 724^2 [6.5M] and
//   1024^2 [13M]; disk 7x7 crosses between 512^2 [7.25M] and 724^2 [14.5M]).
// - Rank/median: 0.25M element-taps (median 3x3 128^2 [0.14M] stays on CPU;
//   median 5x5 128^2 [0.39M], 3x3 256^2 [0.56M], 3x3x3 32^3 [0.84M] win 1.5-3x on GPU).
constexpr router::Family dense_minmax_family{"ndimage.minmax_filter", uint64_t{10} << 20};
constexpr router::Family rank_family{"ndimage.rank_filter", uint64_t{1} << 18};

constexpr uint64_t gpu_max_morph_work = uint64_t{1} << 37;
constexpr double float32_overflow_threshold = 0x1.ffffffp127;

uint32_t resolve_morph_out_dtype(uint32_t in_dtype, uint32_t requested) {
    if (requested == 0) return in_dtype;
    (void)array::dtype_size(requested);
    if (!array::is_real(requested))
        throw Error(SM_ENOTSUP, "ndimage morphology/rank filters require a real output dtype");
    return requested;
}

MorphTapTable build_morph_taps(const array::Shape& shape, const uint32_t fshape[SM_MAX_NDIM],
                               const int64_t start[SM_MAX_NDIM], const uint8_t* footprint,
                               const double* structure, uint32_t mode, bool& all_true_out) {
    const uint32_t ndim = shape.ndim;
    uint64_t in_strides[SM_MAX_NDIM] = {};
    in_strides[ndim - 1] = 1;
    for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; --d) {
        in_strides[d] = in_strides[d + 1] * shape.dims[d + 1];
    }

    uint64_t total_f = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
        if (sm_checked_mul(total_f, fshape[d], &total_f) != SM_OK)
            throw Error(SM_EOVERFLOW, "footprint element count overflows");
    }

    MorphTapTable table;
    table.has_structure = (structure != nullptr);
    int64_t min_off[SM_MAX_NDIM];
    int64_t max_off[SM_MAX_NDIM];
    for (uint32_t d = 0; d < ndim; ++d) {
        min_off[d] = INT64_MAX;
        max_off[d] = INT64_MIN;
    }

    uint32_t fcoord[SM_MAX_NDIM] = {};
    bool all_true = true;
    for (uint64_t idx = 0; idx < total_f; ++idx) {
        const bool active = footprint ? (footprint[idx] != 0) : true;
        if (!active) {
            all_true = false;
        } else {
            MorphTap tap;
            int64_t lin = 0;
            for (uint32_t d = 0; d < ndim; ++d) {
                // Every nonconstant extension maps a singleton axis to its sole sample.
                // Fold it into the tap offset so thin images retain an interior fast path.
                const int64_t off = (shape.dims[d] == 1 && mode != SM_NDIMAGE_CONSTANT)
                                        ? 0 : static_cast<int64_t>(fcoord[d]) + start[d];
                tap.coord[d] = off;
                lin += off * static_cast<int64_t>(in_strides[d]);
                if (off < min_off[d]) min_off[d] = off;
                if (off > max_off[d]) max_off[d] = off;
            }
            tap.linear_offset = lin;
            if (structure) {
                const double sv = structure[idx];
                tap.struct_offset_f64 = sv;
                tap.struct_offset_f32 = static_cast<float>(sv);
                if (!std::isfinite(sv) || std::fabs(sv) >= float32_overflow_threshold) {
                    table.structure_overflow_f32 = true;
                } else {
                    table.structure_max_abs = std::max(table.structure_max_abs, std::fabs(sv));
                    if (std::trunc(sv) != sv) table.structure_non_integral = true;
                }
            }
            table.taps.push_back(tap);
        }
        for (int32_t d = static_cast<int32_t>(ndim) - 1; d >= 0; --d) {
            if (++fcoord[d] < fshape[d]) break;
            fcoord[d] = 0;
        }
    }
    if (table.taps.empty()) throw Error(SM_EINVAL, "all-zero footprint is not supported");
    all_true_out = all_true;

    table.has_interior = true;
    table.all_interior = true;
    for (uint32_t d = 0; d < ndim; ++d) {
        const int64_t dim = static_cast<int64_t>(shape.dims[d]);
        const int64_t lo = std::max<int64_t>(0, -min_off[d]);
        const int64_t hi = std::min<int64_t>(dim, dim - max_off[d]);
        table.interior_lo[d] = lo;
        table.interior_hi[d] = hi;
        if (lo >= hi) table.has_interior = false;
        if (lo != 0 || hi != dim) table.all_interior = false;
    }
    return table;
}

bool taps_fit_gpu(const array::Shape& shape, const MorphTapTable& taps) noexcept {
    if (shape.elements > INT32_MAX) return false;
    if (taps.taps.size() > SM_NDIMAGE_MORPH_MAX_TAPS) return false;
    for (uint32_t d = 0; d < shape.ndim; ++d) {
        if (shape.dims[d] > INT32_MAX) return false;
    }
    for (const MorphTap& tap : taps.taps) {
        if (tap.linear_offset < INT32_MIN || tap.linear_offset > INT32_MAX) return false;
        for (uint32_t d = 0; d < shape.ndim; ++d) {
            if (tap.coord[d] < INT32_MIN || tap.coord[d] > INT32_MAX) return false;
        }
    }
    return true;
}

} // namespace

bool gpu_supports_dense_minmax(const DenseMinMaxPlan& plan) noexcept {
    if (!taps_fit_gpu(plan.shape, plan.taps)) return false;
    uint64_t work = 0;
    if (sm_checked_mul(plan.shape.elements, plan.taps.taps.size(), &work) != SM_OK) return false;
    return work <= gpu_max_morph_work;
}

bool gpu_supports_rank(const RankPlan& plan) noexcept {
    if (!taps_fit_gpu(plan.shape, plan.taps)) return false;
    const size_t k = plan.taps.taps.size();
    if (k < 3 || plan.rank == 0 || plan.rank + 1 >= k) return false;
    const uint32_t s = std::min<uint32_t>(plan.rank, static_cast<uint32_t>(k) - 1u - plan.rank) + 2u;
    if (s > SM_NDIMAGE_RANK_MEDIUM_MAX_S) return false;
    uint64_t work = 0;
    if (sm_checked_mul(plan.shape.elements, static_cast<uint64_t>(k) * s, &work) != SM_OK) return false;
    return work <= gpu_max_morph_work;
}

} // namespace sm::detail::ndimage

using sm::detail::Error;

sm_status sm_ndimage_minmax_filter(sm_context* context, const sm_array* input,
                                   const sm_ndimage_minmax_options* minmax_options,
                                   const sm_route_options* options,
                                   sm_array_result* out, sm_route_info* info) {
    namespace array = sm::detail::array;
    namespace ndimage = sm::detail::ndimage;
    namespace router = sm::detail::router;
    return sm::detail::api(context, [&] {
        if (!out) throw Error(SM_EINVAL, "null output");
        array::check_route_info(info);
        array::check_options(options);
        if (!minmax_options) throw Error(SM_EINVAL, "null minmax filter options");
        if (minmax_options->struct_size != sizeof(sm_ndimage_minmax_options) ||
            minmax_options->abi_version != SM_NDIMAGE_ABI_VERSION)
            throw Error(SM_EINVAL, "minmax filter options size or ABI version mismatch");
        if (minmax_options->reserved0 || minmax_options->reserved[0] || minmax_options->reserved[1])
            throw Error(SM_EINVAL, "minmax filter options reserved fields must be zero");
        if (minmax_options->mode > SM_NDIMAGE_CONSTANT) throw Error(SM_EINVAL, "unknown boundary mode");
        if (minmax_options->structure &&
            reinterpret_cast<uintptr_t>(minmax_options->structure) % alignof(double) != 0)
            throw Error(SM_EINVAL, "morphology structure is not aligned for double");

        const array::Shape shape = array::checked_shape(input);
        if (!array::is_real(input->dtype))
            throw Error(SM_ENOTSUP, "ndimage morphology filters take real arrays");
        if (shape.elements && reinterpret_cast<uintptr_t>(input->data) % array::dtype_size(input->dtype))
            throw Error(SM_EINVAL, "array data is not aligned to its element size");
        const uint32_t out_dtype = ndimage::resolve_morph_out_dtype(input->dtype, minmax_options->out_dtype);
        uint64_t out_bytes = 0;
        if (sm_checked_mul(shape.elements, array::dtype_size(out_dtype), &out_bytes) != SM_OK)
            throw Error(SM_EOVERFLOW, "output array byte size overflows");

        ndimage::DenseMinMaxPlan plan;
        plan.shape = shape;
        plan.in_dtype = input->dtype;
        plan.out_dtype = out_dtype;
        plan.is_max = (minmax_options->is_max != 0);
        plan.mode = minmax_options->mode;
        plan.cval = minmax_options->cval;
        for (uint32_t d = 0; d < shape.ndim; ++d) {
            if (minmax_options->fshape[d] < 1 || minmax_options->fshape[d] > SM_NDIMAGE_MAX_WINDOW)
                throw Error(SM_EINVAL, "footprint extent must be 1 to 2^30");
            const int64_t sz = static_cast<int64_t>(minmax_options->fshape[d]);
            if (minmax_options->origins[d] < -(sz / 2) || minmax_options->origins[d] > (sz - 1) / 2)
                throw Error(SM_EINVAL, "filter origin must satisfy -(size / 2) <= origin <= (size - 1) / 2");
            plan.fshape[d] = minmax_options->fshape[d];
            plan.start[d] = -(sz / 2) - minmax_options->origins[d];
        }
        bool all_true = false;
        plan.taps = ndimage::build_morph_taps(shape, plan.fshape, plan.start, minmax_options->footprint,
                                              minmax_options->structure, plan.mode, all_true);

        sm_route_options route = *options;
        if (!ndimage::gpu_supports_dense_minmax(plan)) {
            if (route.execution == SM_EXECUTION_GPU)
                throw Error(SM_ENOTSUP, "the array or footprint is too large for the GPU engine");
            route.execution = SM_EXECUTION_CPU;
        }
        if (route.precision == SM_PRECISION_STRICT && route.execution == SM_EXECUTION_AUTO)
            route.execution = SM_EXECUTION_CPU;
        if (route.precision == SM_PRECISION_STRICT && route.execution == SM_EXECUTION_GPU &&
            out_dtype != SM_FLOAT32)
            throw Error(SM_ENOTSUP, "strict precision keeps non-float32 outputs on the CPU");

        uint64_t work = 0;
        if (sm_checked_mul(shape.elements, plan.taps.taps.size(), &work) != SM_OK) work = UINT64_MAX;
        const router::Decision decision =
            router::decide(context, ndimage::dense_minmax_family, route, input->dtype, work);

        sm::detail::BufferPtr result;
        uint32_t engine = shape.elements ? decision.engine : uint32_t{SM_EXECUTION_CPU};
        uint32_t compute_dtype = input->dtype;
        std::string reason;
        if (!shape.elements) {
            result = array::allocate(context, 0);
        } else if (decision.engine == SM_EXECUTION_GPU) {
#if defined(SM_ENABLE_METAL)
            const bool allow_fallback = options->execution == SM_EXECUTION_AUTO;
            result = ndimage::run_gpu_dense_minmax(context, plan, *input, allow_fallback, reason);
            if (result) {
                compute_dtype = SM_FLOAT32;
            } else {
                engine = SM_EXECUTION_CPU;
                result = ndimage::run_cpu_dense_minmax(context, plan, input->data);
            }
#else
            throw Error(SM_ENOTSUP, "GPU execution needs a Metal build");
#endif
        } else {
            result = ndimage::run_cpu_dense_minmax(context, plan, input->data);
        }
        array::publish(out, std::move(result), out_dtype, shape);
        array::report(info, engine, compute_dtype, reason.c_str());
    });
}

sm_status sm_ndimage_rank_filter(sm_context* context, const sm_array* input,
                                 const sm_ndimage_rank_options* rank_options,
                                 const sm_route_options* options,
                                 sm_array_result* out, sm_route_info* info) {
    namespace array = sm::detail::array;
    namespace ndimage = sm::detail::ndimage;
    namespace router = sm::detail::router;
    return sm::detail::api(context, [&] {
        if (!out) throw Error(SM_EINVAL, "null output");
        array::check_route_info(info);
        array::check_options(options);
        if (!rank_options) throw Error(SM_EINVAL, "null rank filter options");
        if (rank_options->struct_size != sizeof(sm_ndimage_rank_options) ||
            rank_options->abi_version != SM_NDIMAGE_ABI_VERSION)
            throw Error(SM_EINVAL, "rank filter options size or ABI version mismatch");
        if (rank_options->reserved0 || rank_options->reserved[0] || rank_options->reserved[1])
            throw Error(SM_EINVAL, "rank filter options reserved fields must be zero");
        if (rank_options->mode > SM_NDIMAGE_CONSTANT) throw Error(SM_EINVAL, "unknown boundary mode");

        const array::Shape shape = array::checked_shape(input);
        if (!array::is_real(input->dtype))
            throw Error(SM_ENOTSUP, "ndimage rank filters take real arrays");
        if (shape.elements && reinterpret_cast<uintptr_t>(input->data) % array::dtype_size(input->dtype))
            throw Error(SM_EINVAL, "array data is not aligned to its element size");
        const uint32_t out_dtype = ndimage::resolve_morph_out_dtype(input->dtype, rank_options->out_dtype);
        uint64_t out_bytes = 0;
        if (sm_checked_mul(shape.elements, array::dtype_size(out_dtype), &out_bytes) != SM_OK)
            throw Error(SM_EOVERFLOW, "output array byte size overflows");

        ndimage::RankPlan plan;
        plan.shape = shape;
        plan.in_dtype = input->dtype;
        plan.out_dtype = out_dtype;
        plan.rank = rank_options->rank;
        plan.mode = rank_options->mode;
        plan.cval = rank_options->cval;
        for (uint32_t d = 0; d < shape.ndim; ++d) {
            if (rank_options->fshape[d] < 1 || rank_options->fshape[d] > SM_NDIMAGE_MAX_WINDOW)
                throw Error(SM_EINVAL, "footprint extent must be 1 to 2^30");
            const int64_t sz = static_cast<int64_t>(rank_options->fshape[d]);
            if (rank_options->origins[d] < -(sz / 2) || rank_options->origins[d] > (sz - 1) / 2)
                throw Error(SM_EINVAL, "filter origin must satisfy -(size / 2) <= origin <= (size - 1) / 2");
            plan.fshape[d] = rank_options->fshape[d];
            plan.origins[d] = rank_options->origins[d];
            plan.start[d] = -(sz / 2) - rank_options->origins[d];
        }
        plan.taps = ndimage::build_morph_taps(shape, plan.fshape, plan.start, rank_options->footprint,
                                              nullptr, plan.mode, plan.all_true_footprint);
        const size_t k = plan.taps.taps.size();
        if (plan.rank >= k) throw Error(SM_EINVAL, "rank not within filter footprint size");

        // Delegate extreme ranks (0 or k-1, including k==1) to dense min/max.
        if (plan.rank == 0 || plan.rank + 1 == k) {
            ndimage::DenseMinMaxPlan mm;
            mm.shape = shape;
            mm.in_dtype = input->dtype;
            mm.out_dtype = out_dtype;
            mm.is_max = (plan.rank > 0);
            mm.mode = plan.mode;
            mm.cval = plan.cval;
            for (uint32_t d = 0; d < shape.ndim; ++d) {
                mm.fshape[d] = plan.fshape[d];
                mm.start[d] = plan.start[d];
            }
            mm.taps = std::move(plan.taps);

            sm_route_options route = *options;
            if (!ndimage::gpu_supports_dense_minmax(mm)) {
                if (route.execution == SM_EXECUTION_GPU)
                    throw Error(SM_ENOTSUP, "the array or footprint is too large for the GPU engine");
                route.execution = SM_EXECUTION_CPU;
            }
            if (route.precision == SM_PRECISION_STRICT && route.execution == SM_EXECUTION_AUTO)
                route.execution = SM_EXECUTION_CPU;
            if (route.precision == SM_PRECISION_STRICT && route.execution == SM_EXECUTION_GPU &&
                out_dtype != SM_FLOAT32)
                throw Error(SM_ENOTSUP, "strict precision keeps non-float32 outputs on the CPU");

            uint64_t work = 0;
            if (sm_checked_mul(shape.elements, mm.taps.taps.size(), &work) != SM_OK) work = UINT64_MAX;
            const router::Decision decision =
                router::decide(context, ndimage::dense_minmax_family, route, input->dtype, work);

            sm::detail::BufferPtr result;
            uint32_t engine = shape.elements ? decision.engine : uint32_t{SM_EXECUTION_CPU};
            uint32_t compute_dtype = input->dtype;
            std::string reason;
            if (!shape.elements) {
                result = array::allocate(context, 0);
            } else if (decision.engine == SM_EXECUTION_GPU) {
#if defined(SM_ENABLE_METAL)
                const bool allow_fallback = options->execution == SM_EXECUTION_AUTO;
                result = ndimage::run_gpu_dense_minmax(context, mm, *input, allow_fallback, reason);
                if (result) {
                    compute_dtype = SM_FLOAT32;
                } else {
                    engine = SM_EXECUTION_CPU;
                    result = ndimage::run_cpu_dense_minmax(context, mm, input->data);
                }
#else
                throw Error(SM_ENOTSUP, "GPU execution needs a Metal build");
#endif
            } else {
                result = ndimage::run_cpu_dense_minmax(context, mm, input->data);
            }
            array::publish(out, std::move(result), out_dtype, shape);
            array::report(info, engine, compute_dtype, reason.c_str());
            return;
        }

        sm_route_options route = *options;
        // Wide 1-D windows reuse their overlapping samples on the CPU. The
        // generic Metal kernel revisits every tap for every output.
        if (route.execution == SM_EXECUTION_AUTO && ndimage::use_sliding_rank(plan) &&
            (shape.elements < 65536 || k > 64))
            route.execution = SM_EXECUTION_CPU;
        if (route.execution != SM_EXECUTION_CPU && !ndimage::gpu_supports_rank(plan)) {
            if (route.execution == SM_EXECUTION_GPU)
                throw Error(SM_ENOTSUP, "the array or rank window exceeds GPU rank filter limits");
            route.execution = SM_EXECUTION_CPU;
        }
        if (route.precision == SM_PRECISION_STRICT && route.execution == SM_EXECUTION_AUTO)
            route.execution = SM_EXECUTION_CPU;
        if (route.precision == SM_PRECISION_STRICT && route.execution == SM_EXECUTION_GPU &&
            out_dtype != SM_FLOAT32)
            throw Error(SM_ENOTSUP, "strict precision keeps non-float32 outputs on the CPU");

        uint64_t work = 0;
        if (sm_checked_mul(shape.elements, k, &work) != SM_OK) work = UINT64_MAX;
        const router::Decision decision =
            router::decide(context, ndimage::rank_family, route, input->dtype, work);

        sm::detail::BufferPtr result;
        uint32_t engine = shape.elements ? decision.engine : uint32_t{SM_EXECUTION_CPU};
        uint32_t compute_dtype = SM_FLOAT64;
        std::string reason;
        if (!shape.elements) {
            result = array::allocate(context, 0);
        } else if (decision.engine == SM_EXECUTION_GPU) {
#if defined(SM_ENABLE_METAL)
            const bool allow_fallback = options->execution == SM_EXECUTION_AUTO;
            result = ndimage::run_gpu_rank(context, plan, *input, allow_fallback, reason);
            if (result) {
                compute_dtype = SM_FLOAT32;
            } else {
                engine = SM_EXECUTION_CPU;
                result = ndimage::run_cpu_rank(context, plan, input->data);
            }
#else
            throw Error(SM_ENOTSUP, "GPU execution needs a Metal build");
#endif
        } else {
            result = ndimage::run_cpu_rank(context, plan, input->data);
        }
        array::publish(out, std::move(result), out_dtype, shape);
        array::report(info, engine, compute_dtype, reason.c_str());
    });
}
