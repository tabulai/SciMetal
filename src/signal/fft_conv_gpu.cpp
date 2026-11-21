// GPU FFT and Overlap-Add convolution engine for metal-sci signal.
// Executes pad/OA-chunk -> forward FFT(in1) -> forward FFT(in2) ->
// spectral multiply -> inverse FFT -> OA-combine/crop/widen in ONE gpu::Batch.
#include "fft_conv.hpp"
#include "signal_kernels.h"
#include "core/router.hpp"
#include "fft/engine.hpp"
#include "runtime/metal_compute.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace ms::detail::signal {
namespace {

// Below this many padded FFT elements, MS_EXECUTION_AUTO declines so SciPy
// handles small calls without GPU hand-off overhead. Calibrated via
// bench/signal_convolve.py --calibrate.
constexpr uint64_t fft_gpu_min_elements = uint64_t{1} << 15;

// A5 single-precision FFT magnitude bounds (matching src/fft/fft.cpp) and
// product-overflow ceiling.
constexpr double narrow_ceiling = 0x1p96;
constexpr double narrow_floor = 0x1p-100;
constexpr double product_ceiling = 0x1p124;

struct StagedFft {
    std::vector<fft::Stage> stages;
    bool input_real = false;
    bool result_real = false;
    double applied = 1.0;
};

StagedFft build_stages(const uint64_t* dims, const uint64_t* nsteps, const uint32_t* axes_mask_ptr,
                       uint32_t ndim, bool real_last, bool inverse, const fft::Limits& limits) {
    const uint32_t axis_mask = *axes_mask_ptr;
    const uint32_t last = ndim - 1;
    const bool has_last = (axis_mask & (1u << last)) != 0;

    // Logical dimensions: for axis d in axes, extent is nsteps[d] * len_d.
    // Before last_axis runs (forward), last_axis has fft_len[last]; after r2c, it has fft_len[last]/2 + 1.
    auto compute_inner_tail_and_count = [&](uint32_t ax, uint64_t last_row_elems,
                                            uint64_t& out_count, uint64_t& out_tail) {
        out_count = 1;
        for (uint32_t d = 0; d < ax; ++d) {
            if (ms_checked_mul(out_count, nsteps[d] * dims[d], &out_count) != MS_OK)
                throw Error(MS_ENOTSUP, "the signal FFT exceeds 32-bit kernel indexing");
        }
        if (ms_checked_mul(out_count, nsteps[ax], &out_count) != MS_OK)
            throw Error(MS_ENOTSUP, "the signal FFT exceeds 32-bit kernel indexing");
        out_tail = 1;
        for (uint32_t d = ax + 1; d < ndim; ++d) {
            const uint64_t ext = (d == last) ? (nsteps[d] * last_row_elems) : (nsteps[d] * dims[d]);
            if (ms_checked_mul(out_tail, ext, &out_tail) != MS_OK)
                throw Error(MS_ENOTSUP, "the signal FFT exceeds 32-bit kernel indexing");
        }
    };

    uint64_t rows = 1;
    for (uint32_t d = 0; d < last; ++d) {
        if (ms_checked_mul(rows, nsteps[d] * dims[d], &rows) != MS_OK)
            throw Error(MS_ENOTSUP, "the signal FFT exceeds 32-bit kernel indexing");
    }
    if (ms_checked_mul(rows, nsteps[last], &rows) != MS_OK)
        throw Error(MS_ENOTSUP, "the signal FFT exceeds 32-bit kernel indexing");

    StagedFft staged;
    std::shared_ptr<const fft::Plan> last_plan;
    if (has_last) {
        fft::Request req;
        req.transform = real_last ? (inverse ? fft::Transform::c2r : fft::Transform::r2c) : fft::Transform::c2c;
        req.inverse = (!real_last) && inverse;
        req.n = dims[last];
        req.count = rows;
        fft::Planned planned = fft::plan(req, limits);
        if (!planned.plan)
            throw Error(MS_ENOTSUP, "no GPU plan for signal FFT length " + std::to_string(req.n) + ": " + planned.reason);
        last_plan = std::move(planned.plan);
    }

    const uint64_t before = last_plan ? last_plan->input_row : dims[last];
    const uint64_t after = last_plan ? last_plan->result_row : dims[last];
    const uint64_t inner_row = inverse ? before : after;

    std::vector<fft::Stage> inner;
    for (int32_t ax = static_cast<int32_t>(last) - 1; ax >= 0; --ax) {
        const uint32_t uax = static_cast<uint32_t>(ax);
        if (!(axis_mask & (1u << uax))) continue;
        if (dims[uax] == 1) continue;
        fft::Request req;
        req.transform = fft::Transform::c2c;
        req.inverse = inverse;
        req.n = dims[uax];
        compute_inner_tail_and_count(uax, inner_row, req.count, req.tail);
        fft::Planned planned = fft::plan(req, limits);
        if (!planned.plan)
            throw Error(MS_ENOTSUP, "no GPU plan for signal FFT length " + std::to_string(req.n) +
                                        " along axis " + std::to_string(uax) + ": " + planned.reason);
        inner.push_back({std::move(planned.plan), req.count});
    }

    if (last_plan && !inverse) staged.stages.push_back({last_plan, rows});
    staged.stages.insert(staged.stages.end(), inner.begin(), inner.end());
    if (last_plan && inverse) staged.stages.push_back({last_plan, rows});
    if (staged.stages.empty()) throw Error(MS_ENOTSUP, "a one-point signal FFT needs no transform");

    staged.input_real = last_plan && last_plan->input_real;
    staged.result_real = last_plan && last_plan->result_real;
    for (const fft::Stage& s : staged.stages) {
        if (s.plan->divides_by_n) staged.applied /= static_cast<double>(s.plan->request.n);
    }
    return staged;
}

StagedFft fit_stages(const uint64_t* dims, const uint64_t* nsteps, uint32_t axis_mask,
                     uint32_t ndim, bool real_last, bool inverse, const fft::Limits& limits) {
    StagedFft staged = build_stages(dims, nsteps, &axis_mask, ndim, real_last, inverse, limits);
    std::vector<int32_t> ceilings(staged.stages.size(), limits.group_threads);
    const uint32_t last = ndim - 1;
    const bool has_last = (axis_mask & (1u << last)) != 0;
    constexpr uint32_t max_replan_iters = 64;
    for (uint32_t iter = 0; iter < max_replan_iters; ++iter) {
        fft::Refusal refusal;
        if (!fft::refused(staged.stages, refusal)) {
            for (const fft::Stage& s : staged.stages) {
                if (s.plan->load != fft::Load::direct || s.plan->store != fft::Store::direct) {
                    throw Error(MS_ENOTSUP, "signal FFT requires direct load/store plan");
                }
            }
            return staged;
        }
        const size_t idx = refusal.stage;
        const int32_t next_ceiling = refusal.granted < ceilings[idx] ? refusal.granted : ceilings[idx] / 2;
        ceilings[idx] = next_ceiling;
        const fft::Stage& ref_stage = staged.stages[idx];
        fft::Request req = ref_stage.plan->request;
        req.count = ref_stage.count;
        fft::Planned replacement = fft::replan(req, limits, next_ceiling);
        if (!replacement.plan) {
            throw Error(MS_ENOTSUP, "no GPU plan for signal FFT length " + std::to_string(req.n) + ": " +
                                        replacement.reason);
        }
        if (replacement.plan->load != fft::Load::direct || replacement.plan->store != fft::Store::direct) {
            throw Error(MS_ENOTSUP, "signal FFT requires direct load/store plan");
        }
        staged.stages[idx].plan = std::move(replacement.plan);
        staged.stages[idx].resolved.reset();
        const bool is_last = has_last && ((!inverse && idx == 0) || (inverse && idx + 1 == staged.stages.size()));
        if (is_last) {
            staged.input_real = staged.stages[idx].plan->input_real;
            staged.result_real = staged.stages[idx].plan->result_real;
        }
        staged.applied = 1.0;
        for (const fft::Stage& s : staged.stages) {
            if (s.plan->divides_by_n) staged.applied /= static_cast<double>(s.plan->request.n);
        }
    }
    throw Error(MS_ENOTSUP, "could not fit signal FFT plan to device threadgroup limits");
}

void check_single_range(const array::ConvertReport& rep) {
    if (rep.overflowed || rep.has_nonfinite || !(rep.max_finite <= narrow_ceiling) ||
        (rep.max_finite != 0.0 && rep.max_finite < narrow_floor)) {
        throw Error(MS_ENOTSUP, "input magnitudes do not fit single-precision FFT range; SciPy is used");
    }
}

const char* pad_kernel_name(bool src_complex, bool dst_complex) {
    if (!dst_complex) return "ms_signal_fft_pad_f32_f32";
    if (!src_complex) return "ms_signal_fft_pad_f32_c64";
    return "ms_signal_fft_pad_c64_c64";
}

const char* combine_kernel_name(bool src_complex, uint32_t out_dtype) {
    switch (out_dtype) {
    case MS_FLOAT32: return src_complex ? "ms_signal_oa_combine_c_f32" : "ms_signal_oa_combine_r_f32";
    case MS_FLOAT64: return src_complex ? "ms_signal_oa_combine_c_f64" : "ms_signal_oa_combine_r_f64";
    case MS_COMPLEX64: return "ms_signal_oa_combine_c_c64";
    case MS_COMPLEX128: return "ms_signal_oa_combine_c_c128";
    default: throw Error(MS_EINVAL, "unknown output dtype");
    }
}

void build_logical_spec_shapes(const FftConvPlan& plan, std::vector<uint32_t>& d_out,
                               std::vector<uint32_t>& d_a, std::vector<uint32_t>& d_b) {
    const uint32_t last = plan.ndim - 1;
    for (uint32_t d = 0; d < plan.ndim; ++d) {
        if (plan.axis_mask & (1u << d)) {
            const uint32_t spec_len = static_cast<uint32_t>((d == last && plan.real_last_axis)
                                                                ? (plan.fft_len[d] / 2 + 1)
                                                                : plan.fft_len[d]);
            const uint32_t n1 = static_cast<uint32_t>(plan.nsteps1[d]);
            const uint32_t n2 = static_cast<uint32_t>(plan.nsteps2[d]);
            const uint32_t nout = static_cast<uint32_t>(plan.nsteps_out[d]);
            if (nout > 1) {
                d_out.push_back(nout);
                d_a.push_back(n1);
                d_b.push_back(n2);
            }
            d_out.push_back(spec_len);
            d_a.push_back(spec_len);
            d_b.push_back(spec_len);
        } else {
            const uint32_t s1d = static_cast<uint32_t>(plan.s1.dims[d]);
            const uint32_t s2d = static_cast<uint32_t>(plan.s2.dims[d]);
            const uint32_t sout = std::max(s1d, s2d);
            if (sout > 1 || d_out.empty()) {
                d_out.push_back(sout);
                d_a.push_back(s1d);
                d_b.push_back(s2d);
            }
        }
    }
    // Coalesce adjacent non-broadcast dimensions (both a and b match out) to keep ndim small.
    size_t w = 0;
    for (size_t r = 0; r < d_out.size(); ++r) {
        if (w > 0 && d_a[w - 1] == d_out[w - 1] && d_b[w - 1] == d_out[w - 1] &&
            d_a[r] == d_out[r] && d_b[r] == d_out[r]) {
            d_out[w - 1] *= d_out[r];
            d_a[w - 1] *= d_a[r];
            d_b[w - 1] *= d_b[r];
        } else {
            d_out[w] = d_out[r];
            d_a[w] = d_a[r];
            d_b[w] = d_b[r];
            ++w;
        }
    }
    d_out.resize(w);
    d_a.resize(w);
    d_b.resize(w);
}

} // namespace

BufferPtr run_fft_conv_gpu(ms_context* context, const ms_array& in1_orig, const ms_array& in2_orig,
                           const FftConvPlan& plan_in, const ms_route_options& route_options,
                           uint32_t& out_compute_dtype, uint32_t* out_nf_flags) {
    if (storage_kind(context) != MS_BUFFER_METAL) {
        throw Error(MS_ENOTSUP, "GPU execution needs a Metal-storage context");
    }
    if (route_options.execution == MS_EXECUTION_CPU) {
        throw Error(MS_ENOTSUP, "there is no native CPU FFT convolution; SciPy's is used");
    }
    const bool out_double = array::is_double(plan_in.out_dtype);
    if (out_double && route_options.precision == MS_PRECISION_STRICT) {
        throw Error(MS_ENOTSUP, "strict precision keeps double-precision FFT convolution on SciPy");
    }
    if (plan_in.axis_count == 0) {
        throw Error(MS_ENOTSUP, "no axes to transform; SciPy is used");
    }
    if (route_options.execution == MS_EXECUTION_AUTO && plan_in.effective_fft_elements < fft_gpu_min_elements) {
        throw Error(MS_ENOTSUP, "the signal FFT convolution is below the GPU crossover; SciPy's is used");
    }

    const fft::Limits limits = fft::device_limits();
    FftConvPlan plan = plan_in;
    for (uint32_t d = 0; d < plan.ndim; ++d) {
        if (plan.fft_len[d] == 0 || plan.fft_len[d] > INT32_MAX ||
            plan.nsteps1[d] > INT32_MAX / plan.fft_len[d] ||
            plan.nsteps2[d] > INT32_MAX / plan.fft_len[d] ||
            plan.nsteps_out[d] > INT32_MAX / plan.fft_len[d]) {
            throw Error(MS_ENOTSUP, "the signal FFT convolution exceeds 32-bit kernel indexing");
        }
    }
    if (plan.effective_fft_elements > INT32_MAX) {
        throw Error(MS_ENOTSUP, "the signal FFT convolution exceeds 32-bit kernel indexing");
    }

    uint64_t dims1[MS_MAX_NDIM] = {}, dims2[MS_MAX_NDIM] = {}, dims_out[MS_MAX_NDIM] = {};
    for (uint32_t d = 0; d < plan.ndim; ++d) {
        if (plan.axis_mask & (1u << d)) {
            dims1[d] = dims2[d] = dims_out[d] = plan.fft_len[d];
        } else {
            dims1[d] = plan.s1.dims[d];
            dims2[d] = plan.s2.dims[d];
            dims_out[d] = std::max(plan.s1.dims[d], plan.s2.dims[d]);
        }
    }

    StagedFft fwd1, fwd2, inv;
    for (int attempt = 0; attempt < 2; ++attempt) {
        try {
            fwd1 = fit_stages(dims1, plan.nsteps1, plan.axis_mask, plan.ndim, plan.real_last_axis, false, limits);
            fwd2 = fit_stages(dims2, plan.nsteps2, plan.axis_mask, plan.ndim, plan.real_last_axis, false, limits);
            inv = fit_stages(dims_out, plan.nsteps_out, plan.axis_mask, plan.ndim, plan.real_last_axis, true, limits);
            break;
        } catch (const Error& err) {
            // If real_last_axis non-power-of-two got replanned to non-direct load/store,
            // upgrade last axis to power-of-two once.
            const uint32_t last = plan.ndim - 1;
            if (attempt == 0 && plan.real_last_axis && !std::has_single_bit(plan.fft_len[last]) &&
                plan.fft_len[last] <= (uint64_t{1} << 30)) {
                const uint64_t pow2 = std::bit_ceil(plan.fft_len[last]);
                plan.fft_len[last] = pow2;
                dims1[last] = dims2[last] = dims_out[last] = pow2;
                if (plan.use_oa && plan.overlap[last] > 0) {
                    const uint64_t s1_d = plan.s1.dims[last];
                    const uint64_t s2_d = plan.s2.dims[last];
                    uint64_t full_d = 0;
                    if (ms_checked_add(s1_d, s2_d - 1, &full_d) != MS_OK)
                        throw Error(MS_EOVERFLOW, "convolution output dimension overflows");
                    if (pow2 >= full_d) {
                        plan.overlap[last] = 0;
                        plan.step1[last] = s1_d;
                        plan.step2[last] = s2_d;
                        plan.nsteps1[last] = 1;
                        plan.nsteps2[last] = 1;
                        plan.nsteps_out[last] = 1;
                    } else {
                        const uint64_t small = std::min(s1_d, s2_d);
                        const uint64_t eff_step = pow2 - plan.overlap[last];
                        if (s1_d >= s2_d) {
                            plan.step1[last] = eff_step;
                            plan.step2[last] = small;
                        } else {
                            plan.step1[last] = small;
                            plan.step2[last] = eff_step;
                        }
                        auto calc_n = [&](uint64_t s_d, uint64_t st_d) -> uint64_t {
                            if (s_d <= st_d) return 1;
                            uint64_t n = 1 + s_d / st_d;
                            uint64_t covered = 0;
                            if (ms_checked_mul(eff_step, n, &covered) == MS_OK && covered < full_d) ++n;
                            return n;
                        };
                        plan.nsteps1[last] = calc_n(s1_d, plan.step1[last]);
                        plan.nsteps2[last] = calc_n(s2_d, plan.step2[last]);
                        plan.nsteps_out[last] = std::max(plan.nsteps1[last], plan.nsteps2[last]);
                    }
                }
                if (plan.nsteps1[last] > INT32_MAX / pow2 ||
                    plan.nsteps2[last] > INT32_MAX / pow2 ||
                    plan.nsteps_out[last] > INT32_MAX / pow2) {
                    throw Error(MS_ENOTSUP, "the signal FFT convolution exceeds 32-bit kernel indexing");
                }
                continue;
            }
            throw;
        }
    }

    const ms_array& src1 = plan.swapped ? in2_orig : in1_orig;
    const ms_array& src2 = plan.swapped ? in1_orig : in2_orig;

    // Load inputs into single-precision shared storage and enforce A5 guard rails.
    array::ConvertReport rep1, rep2;
    BufferPtr single1 = array::load_single(context, src1, plan.s1, out_double ? &rep1 : nullptr);
    BufferPtr single2 = array::load_single(context, src2, plan.s2, out_double ? &rep2 : nullptr);

    double total_fft_n = 1.0;
    for (uint32_t i = 0; i < plan.axis_count; ++i) {
        total_fft_n *= static_cast<double>(plan.fft_len[plan.axes[i]]);
    }

    if (out_double) {
        check_single_range(rep1);
        check_single_range(rep2);
        const double peak1 = rep1.max_finite;
        const double peak2 = rep2.max_finite;
        const double c1 = array::is_complex(src1.dtype) ? 2.0 : 1.0;
        const double c2 = array::is_complex(src2.dtype) ? 2.0 : 1.0;
        double n1_blk = 1.0, n2_blk = 1.0;
        for (uint32_t i = 0; i < plan.axis_count; ++i) {
            const uint32_t d = plan.axes[i];
            n1_blk *= static_cast<double>(plan.step1[d]);
            n2_blk *= static_cast<double>(plan.step2[d]);
        }
        // Bound spectral product and unnormalized inverse FFT intermediates in float32.
        const double spec1_max = c1 * peak1 * n1_blk;
        const double spec2_max = c2 * peak2 * n2_blk;
        const double prod_max = 2.0 * spec1_max * spec2_max;
        const double unnorm_inv = total_fft_n;
        if (!std::isfinite(prod_max) || prod_max > product_ceiling ||
            !std::isfinite(prod_max * unnorm_inv) || prod_max * unnorm_inv > product_ceiling ||
            peak1 * peak2 * static_cast<double>(plan.overlap_elements) > product_ceiling) {
            throw Error(MS_ENOTSUP, "the signal convolution product exceeds single-precision range; SciPy is used");
        }
        // Guard against single-precision underflow of tiny finite products.
        if (peak1 != 0.0 && peak2 != 0.0) {
            const double prod_min = peak1 * peak2;
            if (prod_min < narrow_floor || (prod_min / total_fft_n) < 0x1p-116) {
                throw Error(MS_ENOTSUP, "the signal convolution product underflows single-precision range; SciPy is used");
            }
        }
    }

    uint64_t pad_elems1 = 1, pad_elems2 = 1, conv_elems = 1;
    for (uint32_t d = 0; d < plan.ndim; ++d) {
        uint64_t e1 = 0, e2 = 0, eo = 0;
        if (ms_checked_mul(plan.nsteps1[d], dims1[d], &e1) != MS_OK ||
            ms_checked_mul(pad_elems1, e1, &pad_elems1) != MS_OK ||
            ms_checked_mul(plan.nsteps2[d], dims2[d], &e2) != MS_OK ||
            ms_checked_mul(pad_elems2, e2, &pad_elems2) != MS_OK ||
            ms_checked_mul(plan.nsteps_out[d], dims_out[d], &eo) != MS_OK ||
            ms_checked_mul(conv_elems, eo, &conv_elems) != MS_OK) {
            throw Error(MS_ENOTSUP, "the signal FFT convolution exceeds 32-bit kernel indexing");
        }
    }
    if (pad_elems1 > INT32_MAX || pad_elems2 > INT32_MAX || conv_elems > INT32_MAX) {
        throw Error(MS_ENOTSUP, "the signal FFT convolution exceeds 32-bit kernel indexing");
    }

    const bool buf_complex = !plan.real_last_axis;
    const uint64_t scalar_bytes = sizeof(float);
    const uint64_t buf1_bytes = std::max(
        pad_elems1 * (buf_complex ? 2 : 1) * scalar_bytes,
        fwd1.stages.front().plan->slots[0] * fwd1.stages.front().count * scalar_bytes);
    const uint64_t buf2_bytes = std::max(
        pad_elems2 * (buf_complex ? 2 : 1) * scalar_bytes,
        fwd2.stages.front().plan->slots[0] * fwd2.stages.front().count * scalar_bytes);

    const double total_inv_scale = 1.0 / total_fft_n;
    const float remaining_scale = static_cast<float>(total_inv_scale / inv.applied);

    const auto make_pad_params = [&](const array::Shape& s, const uint64_t* nsteps, const uint64_t* step,
                                     const uint64_t* dims, uint64_t dst_elems, bool src_c, bool dst_c,
                                     bool conj_src, uint32_t rev_mask) {
        MSSignalPadParams p{};
        p.ndim = plan.ndim;
        p.dst_elements = static_cast<ms_u32>(dst_elems);
        p.flags = (src_c ? MS_SIGNAL_PAD_SRC_COMPLEX : 0u) |
                  (dst_c ? MS_SIGNAL_PAD_DST_COMPLEX : 0u) |
                  (conj_src ? MS_SIGNAL_PAD_CONJ : 0u);
        p.reverse_mask = rev_mask;
        uint32_t in_str = 1, dst_str = 1;
        for (int32_t d = static_cast<int32_t>(plan.ndim) - 1; d >= 0; --d) {
            p.in_shape[d] = static_cast<ms_u32>(s.dims[d]);
            p.in_stride[d] = in_str;
            p.nsteps[d] = static_cast<ms_u32>(nsteps[d]);
            p.step[d] = static_cast<ms_u32>(step[d]);
            p.fft_len[d] = static_cast<ms_u32>(dims[d]);
            p.dst_stride[d] = dst_str;
            in_str *= static_cast<uint32_t>(s.dims[d]);
            dst_str *= static_cast<uint32_t>(nsteps[d] * dims[d]);
        }
        return p;
    };

    const bool src1_complex = array::is_complex(src1.dtype);
    const bool src2_complex = array::is_complex(src2.dtype);
    const bool rev_src2 = plan.correlate;
    const MSSignalPadParams pad1_params = make_pad_params(
        plan.s1, plan.nsteps1, plan.step1, dims1, pad_elems1, src1_complex, buf_complex,
        false, 0u);
    const MSSignalPadParams pad2_params = make_pad_params(
        plan.s2, plan.nsteps2, plan.step2, dims2, pad_elems2, src2_complex, buf_complex,
        rev_src2, rev_src2 ? plan.raw_axis_mask : 0u);

    std::vector<uint32_t> d_out, d_a, d_b;
    build_logical_spec_shapes(plan, d_out, d_a, d_b);
    const bool same_spec_shape = (d_a == d_b);
    const bool a_is_full = (d_a == d_out);
    const bool b_is_full = (d_b == d_out);
    uint64_t spec_out_elems = 1;
    for (uint32_t v : d_out) spec_out_elems *= v;
    const uint64_t spec_out_bytes = spec_out_elems * 2 * scalar_bytes;

    try {
        GpuOperationScope operation(context, true);
        fft::Scratch scratch(context);
        gpu::Batch batch(context);
        BufferPtr nf_buf = array::allocate(context, sizeof(uint32_t));
        *reinterpret_cast<uint32_t*>(mutable_data(nf_buf.get())) = 0u;

        // 1. Pad in1 -> buf1, forward FFT(buf1) -> spec1.
        ms_buffer* buf1 = scratch.acquire(buf1_bytes);
        batch.dispatch(pad_kernel_name(src1_complex, buf_complex), pad1_params,
                       {gpu::bind(single1.get(), context), gpu::bind(buf1, context),
                        gpu::bind(nf_buf.get(), context)},
                       (pad_elems1 + MS_GPU_THREADS - 1) / MS_GPU_THREADS);
        ms_buffer* spec1 = fft::encode(context, batch, scratch, fwd1.stages, buf1, 1.0f);

        // 2. Pad in2 -> buf2, forward FFT(buf2) -> spec2.
        ms_buffer* buf2 = scratch.acquire(buf2_bytes);
        batch.dispatch(pad_kernel_name(src2_complex, buf_complex), pad2_params,
                       {gpu::bind(single2.get(), context), gpu::bind(buf2, context),
                        gpu::bind(nf_buf.get(), context)},
                       (pad_elems2 + MS_GPU_THREADS - 1) / MS_GPU_THREADS);
        ms_buffer* spec2 = fft::encode(context, batch, scratch, fwd2.stages, buf2, 1.0f);

        // 3. Spectral multiply -> spec_prod.
        ms_buffer* spec_prod = nullptr;
        if (same_spec_shape) {
            MSSignalMulInplaceParams mp{};
            mp.elements = static_cast<ms_u32>(spec_out_elems);
            batch.dispatch("ms_signal_spectral_mul_inplace", mp,
                           {gpu::bind(spec1, context), gpu::bind(spec2, context),
                            gpu::bind(nf_buf.get(), context)},
                           (spec_out_elems + MS_GPU_THREADS - 1) / MS_GPU_THREADS);
            spec_prod = spec1;
            scratch.release(spec2);
        } else {
            MSSignalMulBroadcastParams bp{};
            bp.ndim = static_cast<ms_u32>(d_out.size());
            bp.elements = static_cast<ms_u32>(spec_out_elems);
            uint32_t s_out = 1, s_a = 1, s_b = 1;
            for (int32_t i = static_cast<int32_t>(d_out.size()) - 1; i >= 0; --i) {
                bp.shape[i] = d_out[i];
                bp.out_stride[i] = s_out;
                bp.a_stride[i] = (d_a[i] == 1 && d_out[i] > 1) ? 0u : s_a;
                bp.b_stride[i] = (d_b[i] == 1 && d_out[i] > 1) ? 0u : s_b;
                s_out *= d_out[i];
                s_a *= d_a[i];
                s_b *= d_b[i];
            }
            if (a_is_full) {
                batch.dispatch("ms_signal_spectral_mul_broadcast_inplace", bp,
                               {gpu::bind(spec1, context), gpu::bind(spec2, context),
                                gpu::bind(nf_buf.get(), context)},
                               (spec_out_elems + MS_GPU_THREADS - 1) / MS_GPU_THREADS);
                spec_prod = spec1;
                scratch.release(spec2);
            } else if (b_is_full) {
                std::swap(bp.a_stride, bp.b_stride);
                batch.dispatch("ms_signal_spectral_mul_broadcast_inplace", bp,
                               {gpu::bind(spec2, context), gpu::bind(spec1, context),
                                gpu::bind(nf_buf.get(), context)},
                               (spec_out_elems + MS_GPU_THREADS - 1) / MS_GPU_THREADS);
                spec_prod = spec2;
                scratch.release(spec1);
            } else {
                spec_prod = scratch.acquire(spec_out_bytes);
                batch.dispatch("ms_signal_spectral_mul_broadcast", bp,
                               {gpu::bind(spec1, context), gpu::bind(spec2, context),
                                gpu::bind(spec_prod, context), gpu::bind(nf_buf.get(), context)},
                               (spec_out_elems + MS_GPU_THREADS - 1) / MS_GPU_THREADS);
                scratch.release(spec1);
                scratch.release(spec2);
            }
        }

        // 4. Inverse FFT -> conv_buf -> OA combine / crop / widen.
        bool same_conv_out_shape = !plan.use_oa;
        for (uint32_t d = 0; d < plan.ndim && same_conv_out_shape; ++d) {
            if (dims_out[d] != plan.out_shape.dims[d]) same_conv_out_shape = false;
        }
        const bool inv_src_complex = !inv.result_real;
        const bool direct_publish = same_conv_out_shape && !out_double &&
                                    !(plan.correlate && plan.swapped) &&
                                    (inv_src_complex == plan.is_complex);

        BufferPtr out_buf;
        if (direct_publish) {
            ms_buffer* conv_buf = fft::encode(context, batch, scratch, inv.stages, spec_prod, remaining_scale);
            if (out_nf_flags && plan.out_shape.elements > 0) {
                const uint64_t scalar_count =
                    plan.out_shape.elements * (plan.is_complex ? uint64_t{2} : uint64_t{1});
                const uint64_t groups =
                    std::min<uint64_t>(65536u, (scalar_count + MS_GPU_THREADS - 1) / MS_GPU_THREADS);
                MSSignalFiniteParams fp{};
                fp.count = scalar_count;
                batch.dispatch("ms_signal_check_finite_f32", fp,
                               {gpu::bind(conv_buf, context), gpu::bind(nf_buf.get(), context)},
                               groups);
            }
            batch.commit_and_wait();
            operation.complete();
            out_buf = scratch.take(conv_buf);
        } else {
            ms_buffer* conv_buf = fft::encode(context, batch, scratch, inv.stages, spec_prod, 1.0f);
            uint64_t out_bytes = 0;
            if (ms_checked_mul(plan.out_shape.elements, array::dtype_size(plan.out_dtype), &out_bytes) != MS_OK)
                throw Error(MS_EOVERFLOW, "signal output byte size overflows");
            out_buf = array::allocate(context, out_bytes);

            MSSignalCombineParams cp{};
            cp.ndim = plan.ndim;
            cp.out_elements = static_cast<ms_u32>(plan.out_shape.elements);
            const uint32_t rev_out_mask = (plan.correlate && plan.swapped) ? plan.raw_axis_mask : 0u;
            cp.flags = (inv_src_complex ? MS_SIGNAL_COMBINE_SRC_COMPLEX : 0u) |
                       (plan.is_complex ? MS_SIGNAL_COMBINE_DST_COMPLEX : 0u) |
                       (out_double ? MS_SIGNAL_COMBINE_DST_DOUBLE : 0u) |
                       ((plan.correlate && plan.swapped) ? MS_SIGNAL_COMBINE_REV_CONJ : 0u) |
                       (rev_out_mask << 8u);
            cp.scale = remaining_scale;

            uint32_t out_str = 1, conv_str = 1;
            for (int32_t d = static_cast<int32_t>(plan.ndim) - 1; d >= 0; --d) {
                cp.out_shape[d] = static_cast<ms_u32>(plan.out_shape.dims[d]);
                cp.out_stride[d] = out_str;
                cp.conv_stride[d] = conv_str;
                cp.block_stride[d] = static_cast<ms_u32>(dims_out[d]) * conv_str;
                cp.crop_start[d] = static_cast<ms_u32>(plan.crop_start[d]);
                cp.block_size[d] = static_cast<ms_u32>(dims_out[d]);
                const uint64_t st = (plan.nsteps_out[d] > 1)
                                        ? (plan.fft_len[d] - plan.overlap[d])
                                        : dims_out[d];
                cp.step[d] = static_cast<ms_u32>(st);
                cp.nsteps[d] = static_cast<ms_u32>(plan.nsteps_out[d]);
                out_str *= static_cast<uint32_t>(plan.out_shape.dims[d]);
                conv_str *= static_cast<uint32_t>(plan.nsteps_out[d] * dims_out[d]);
            }

            batch.dispatch(combine_kernel_name(inv_src_complex, plan.out_dtype), cp,
                           {gpu::bind(conv_buf, context), gpu::bind(out_buf.get(), context),
                            gpu::bind(nf_buf.get(), context)},
                           (plan.out_shape.elements + MS_GPU_THREADS - 1) / MS_GPU_THREADS);
            batch.commit_and_wait();
            operation.complete();
        }
        const uint32_t nf_flags = *reinterpret_cast<const uint32_t*>(read(nf_buf.get(), context).data);
        if (out_double && nf_flags != 0) {
            throw Error(MS_ENOTSUP, "float64 signal FFT convolution overflowed single precision; SciPy is used");
        }
        if (out_nf_flags) {
            *out_nf_flags = nf_flags;
        }
        out_compute_dtype = array::single_of(plan.out_dtype);
        return out_buf;
    } catch (const Error& err) {
        if (err.status == MS_EOOM) {
            throw Error(MS_ENOTSUP, std::string("the context's memory budget cannot hold the signal FFT buffers: ") + err.what());
        }
        throw;
    }
}

} // namespace ms::detail::signal
