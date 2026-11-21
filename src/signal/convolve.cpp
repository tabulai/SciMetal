// scipy.signal convolution and correlation: validation, routing, direct path
// (reusing sm::detail::dense_correlate), OA/FFT planning, and choose_conv_method.
#include "fft_conv.hpp"
#include "signal_kernels.h"
#include "core/dense_correlate.hpp"
#include "core/dense_correlate_kernels.h"
#include "core/router.hpp"
#include "fft/plan.hpp"
#include "sm/detail/parallel.hpp"
#if defined(SM_ENABLE_METAL)
#include "fft/engine.hpp"
#include "runtime/metal_compute.hpp"
#endif

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstring>
#include <numbers>
#include <string>
#include <vector>

namespace sm::detail::signal {
namespace {

// Calibrated via bench/signal_convolve.py --calibrate on applegpu_g13d (M1 Ultra).
constexpr router::Family direct_family{"signal.convolve.direct", uint64_t{128} << 20};
constexpr uint64_t fft_gpu_min_elements = uint64_t{1} << 15;
#if defined(SM_ENABLE_METAL)
constexpr double float32_overflow_threshold = 0x1.ffffffp127;

bool overflows_float32(double value) noexcept {
    return std::isfinite(value) && std::fabs(value) >= float32_overflow_threshold;
}
#endif

uint64_t saturating_mul(uint64_t a, uint64_t b) noexcept {
    uint64_t prod = 0;
    return sm_checked_mul(a, b, &prod) == SM_OK ? prod : UINT64_MAX;
}

uint64_t checked_full_len(uint64_t a, uint64_t b) {
    uint64_t out = 0;
    if (sm_checked_add(a, b - 1, &out) != SM_OK)
        throw Error(SM_EOVERFLOW, "convolution output dimension overflows");
    return out;
}

uint64_t oa_step_count(uint64_t s_d, uint64_t st_d, uint64_t eff_step, uint64_t full_d) {
    if (s_d <= st_d) return 1;
    // ceil((s_d + 1) / st_d) without s_d + st_d wraparound: 1 + s_d / st_d.
    uint64_t n = 1 + s_d / st_d;
    uint64_t covered = 0;
    if (sm_checked_mul(eff_step, n, &covered) == SM_OK && covered < full_d) ++n;
    return n;
}

double lambert_wm1(double y) noexcept {
    double w;
    if (y < -0.15) {
        const double p = std::sqrt(std::max(0.0, 2.0 * (std::numbers::e * y + 1.0)));
        w = -1.0 - p;
    } else {
        const double l1 = std::log(-y);
        const double l2 = std::log(-l1);
        w = l1 - l2 + l2 / l1;
    }
    for (int iter = 0; iter < 8; ++iter) {
        const double ew = std::exp(w);
        const double f = w * ew - y;
        const double wp1 = w + 1.0;
        const double denom = ew * wp1 - (w + 2.0) * f / (2.0 * wp1);
        const double delta = f / denom;
        w -= delta;
        if (std::fabs(delta) <= 1e-14 * std::fabs(w)) break;
    }
    return w;
}

void calc_oa_lens(uint64_t s1, uint64_t s2, bool real_last, bool is_last, const fft::Limits& limits,
                  uint64_t& out_block, uint64_t& out_overlap, uint64_t& out_step1, uint64_t& out_step2) {
    if (s1 == s2 || s1 <= 1 || s2 <= 1) {
        out_block = checked_full_len(s1, s2);
        out_overlap = 0;
        out_step1 = s1;
        out_step2 = s2;
        return;
    }
    const bool swapped = s2 > s1;
    const uint64_t big = swapped ? s2 : s1;
    const uint64_t small = swapped ? s1 : s2;
    // Overflow-free equivalent of small >= ceil(big / 2.0): small >= big - big / 2.
    if (small >= big - big / 2) {
        out_block = checked_full_len(s1, s2);
        out_overlap = 0;
        out_step1 = s1;
        out_step2 = s2;
        return;
    }
    const uint64_t overlap = small - 1;
    const double y = -1.0 / (2.0 * std::numbers::e * static_cast<double>(overlap));
    const double opt_size = -static_cast<double>(overlap) * lambert_wm1(y);
    if (!std::isfinite(opt_size) || opt_size >= static_cast<double>(big) ||
        opt_size > static_cast<double>(uint64_t{1} << 62)) {
        out_block = checked_full_len(s1, s2);
        out_overlap = 0;
        out_step1 = s1;
        out_step2 = s2;
        return;
    }
    const uint64_t target = std::max<uint64_t>(small, static_cast<uint64_t>(std::ceil(opt_size)));
    const uint64_t block_size = fft::next_fast_len(target, real_last, is_last, limits);
    if (block_size >= big || block_size < small) {
        out_block = checked_full_len(s1, s2);
        out_overlap = 0;
        out_step1 = s1;
        out_step2 = s2;
        return;
    }
    out_block = block_size;
    out_overlap = overlap;
    if (!swapped) {
        out_step1 = block_size - small + 1;
        out_step2 = small;
    } else {
        out_step1 = small;
        out_step2 = block_size - small + 1;
    }
}

uint32_t to_dense_mode(uint32_t boundary) {
    switch (boundary) {
    case SM_SIGNAL_BOUNDARY_FILL: return dense_correlate::Constant;
    case SM_SIGNAL_BOUNDARY_WRAP: return dense_correlate::Wrap;
    case SM_SIGNAL_BOUNDARY_SYMM: return dense_correlate::Reflect;
    default: throw Error(SM_EINVAL, "unknown signal boundary mode");
    }
}

void copy_kernel_weights(const sm_array& k_arr, const array::Shape& k_sh, bool correlate,
                         std::vector<uint8_t>& out_buf) {
    const uint64_t elems = k_sh.elements;
    const uint64_t esize = array::dtype_size(k_arr.dtype);
    out_buf.resize(elems * esize);
    const uint8_t* src = static_cast<const uint8_t*>(k_arr.data);
    uint8_t* dst = out_buf.data();

    if (!correlate) {
        // Convolution reverses the kernel along all axes (no conjugation).
        if (k_arr.dtype == SM_FLOAT32) {
            const auto* s = reinterpret_cast<const float*>(src);
            auto* d = reinterpret_cast<float*>(dst);
            for (uint64_t i = 0; i < elems; ++i) d[i] = s[(elems - 1) - i];
        } else if (k_arr.dtype == SM_FLOAT64) {
            const auto* s = reinterpret_cast<const double*>(src);
            auto* d = reinterpret_cast<double*>(dst);
            for (uint64_t i = 0; i < elems; ++i) d[i] = s[(elems - 1) - i];
        } else if (k_arr.dtype == SM_COMPLEX64) {
            // SM_COMPLEX64 is aligned to alignof(float) (4 bytes), not 8 bytes.
            const auto* s = reinterpret_cast<const float*>(src);
            auto* d = reinterpret_cast<float*>(dst);
            for (uint64_t i = 0; i < elems; ++i) {
                const uint64_t j = (elems - 1) - i;
                d[2 * i] = s[2 * j];
                d[2 * i + 1] = s[2 * j + 1];
            }
        } else {
            const auto* s = reinterpret_cast<const double*>(src);
            auto* d = reinterpret_cast<double*>(dst);
            for (uint64_t i = 0; i < elems; ++i) {
                const uint64_t j = (elems - 1) - i;
                d[2 * i] = s[2 * j];
                d[2 * i + 1] = s[2 * j + 1];
            }
        }
        return;
    }

    if (!array::is_complex(k_arr.dtype)) {
        std::memcpy(dst, src, elems * esize);
    } else if (k_arr.dtype == SM_COMPLEX64) {
        const auto* s = reinterpret_cast<const float*>(src);
        auto* d = reinterpret_cast<float*>(dst);
        for (uint64_t i = 0; i < elems; ++i) {
            d[2 * i] = s[2 * i];
            d[2 * i + 1] = -s[2 * i + 1];
        }
    } else {
        const auto* s = reinterpret_cast<const double*>(src);
        auto* d = reinterpret_cast<double*>(dst);
        for (uint64_t i = 0; i < elems; ++i) {
            d[2 * i] = s[2 * i];
            d[2 * i + 1] = -s[2 * i + 1];
        }
    }
}

void reverse_conj_cpu(void* data, uint64_t elements, uint32_t dtype, bool do_conj) {
    if (elements == 0) return;
    const uint64_t half = elements / 2;
    if (dtype == SM_FLOAT32) {
        auto* p = static_cast<float*>(data);
        parallel::parallel_for(half, uint64_t{1} << 16, [&](unsigned, uint64_t b, uint64_t e) {
            for (uint64_t i = b; i < e; ++i) std::swap(p[i], p[(elements - 1) - i]);
        });
    } else if (dtype == SM_FLOAT64) {
        auto* p = static_cast<double*>(data);
        parallel::parallel_for(half, uint64_t{1} << 16, [&](unsigned, uint64_t b, uint64_t e) {
            for (uint64_t i = b; i < e; ++i) std::swap(p[i], p[(elements - 1) - i]);
        });
    } else if (dtype == SM_COMPLEX64) {
        auto* p = static_cast<float*>(data);
        parallel::parallel_for(half, uint64_t{1} << 16, [&](unsigned, uint64_t b, uint64_t e) {
            for (uint64_t i = b; i < e; ++i) {
                const uint64_t j = (elements - 1) - i;
                const float ar = p[2 * i], ai = p[2 * i + 1];
                const float br = p[2 * j], bi = p[2 * j + 1];
                p[2 * i] = br;
                p[2 * i + 1] = do_conj ? -bi : bi;
                p[2 * j] = ar;
                p[2 * j + 1] = do_conj ? -ai : ai;
            }
        });
        if (do_conj && (elements & 1u)) p[2 * half + 1] = -p[2 * half + 1];
    } else {
        auto* p = static_cast<double*>(data);
        parallel::parallel_for(half, uint64_t{1} << 16, [&](unsigned, uint64_t b, uint64_t e) {
            for (uint64_t i = b; i < e; ++i) {
                const uint64_t j = (elements - 1) - i;
                const double ar = p[2 * i], ai = p[2 * i + 1];
                const double br = p[2 * j], bi = p[2 * j + 1];
                p[2 * i] = br;
                p[2 * i + 1] = do_conj ? -bi : bi;
                p[2 * j] = ar;
                p[2 * j + 1] = do_conj ? -ai : ai;
            }
        });
        if (do_conj && (elements & 1u)) p[2 * half + 1] = -p[2 * half + 1];
    }
}

bool gpu_available(sm_context* context) {
#if defined(SM_ENABLE_METAL)
    return storage_kind(context) == SM_BUFFER_METAL;
#else
    (void)context;
    return false;
#endif
}

uint32_t choose_conv_method_impl(sm_context* context, uint32_t out_dtype, const array::Shape& s1,
                                 const array::Shape& s2, uint32_t mode, uint32_t host_threads,
                                 const sm_route_options& route_options) {
    const uint32_t ndim = s1.ndim;
    bool ok1 = true, ok2 = true;
    for (uint32_t d = 0; d < ndim; ++d) {
        if (s1.dims[d] < s2.dims[d]) ok1 = false;
        if (s2.dims[d] < s1.dims[d]) ok2 = false;
    }
    if (mode == SM_SIGNAL_VALID && !ok1 && !ok2) {
        throw Error(SM_EINVAL, "for valid mode, one input must be at least as large as the other in every dimension");
    }

    uint64_t out_elems = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
        uint64_t od = 0;
        if (mode == SM_SIGNAL_FULL) od = checked_full_len(s1.dims[d], s2.dims[d]);
        else if (mode == SM_SIGNAL_SAME) od = s1.dims[d];
        else od = ok1 ? (s1.dims[d] - s2.dims[d] + 1) : (s2.dims[d] - s1.dims[d] + 1);
        out_elems = saturating_mul(out_elems, od);
    }

    // For 'same' mode, inputs cannot swap so kernel is s2; otherwise smaller input is kernel.
    const uint64_t tap_elems = (mode == SM_SIGNAL_SAME) ? s2.elements : std::min(s1.elements, s2.elements);
    const uint64_t direct_work = saturating_mul(out_elems, tap_elems);

    const bool is_c = array::is_complex(out_dtype);
    const bool is_d = array::is_double(out_dtype);
    const bool policy_gpu = !is_d || route_options.precision != SM_PRECISION_STRICT;
    const bool can_gpu = gpu_available(context) && policy_gpu && route_options.execution != SM_EXECUTION_CPU;

    // Direct cost model (ns), calibrated on M1 Ultra (applegpu_g13d, 16 P-cores):
    // - CPU: ~0.015 ns/tap-op multi-threaded real (~0.045 complex), ~0.12 ns/tap-op single-threaded.
    // - GPU: ~350 us fixed + ~0.005 ns/tap-op real (~0.010 complex).
    const double cpu_tap_ns = (direct_work >= (uint64_t{1} << 17))
                                  ? (is_c ? 0.045 : (is_d ? 0.018 : 0.014))
                                  : (is_c ? 0.30 : 0.14);
    double direct_cost_ns = 15e3 + static_cast<double>(direct_work) * cpu_tap_ns;
    if (can_gpu && direct_work >= direct_family.gpu_min_work &&
        tap_elems <= SM_DENSE_GPU_MAX_TAPS && out_elems <= INT32_MAX) {
        const double gpu_direct_ns = 350e3 + static_cast<double>(direct_work) * (is_c ? 0.010 : 0.005);
        direct_cost_ns = std::min(direct_cost_ns, gpu_direct_ns);
    }

    // FFT / OA cost model (ns):
    fft::Limits limits{};
#if defined(SM_ENABLE_METAL)
    if (can_gpu) limits = fft::device_limits();
#endif
    const FftConvPlan fft_plan = plan_fft_conv(s1, s2, out_dtype, mode, false, true, false, nullptr, 0, limits);
    if (fft_plan.axis_count == 0) return SM_SIGNAL_METHOD_DIRECT;

    double fft_cost_ns = 0.0;
    const double eff_elems = static_cast<double>(fft_plan.effective_fft_elements);
    const double log2n = std::log2(std::max(eff_elems, 2.0));
    if (can_gpu && fft_plan.effective_fft_elements >= fft_gpu_min_elements &&
        fft_plan.effective_fft_elements <= INT32_MAX) {
        // GPU FFT/OA: ~400 us fixed + ~0.6-1.5 ns/effective_element.
        const double per_elem = (is_d ? 1.3 : 0.7) * (is_c ? 1.5 : 1.0);
        fft_cost_ns = 400e3 + eff_elems * per_elem;
    } else {
        // SciPy pocketfft (threaded via set_workers for large calls).
        const uint32_t threads = std::max<uint32_t>(host_threads, 1);
        const double single_elems = static_cast<double>(fft_plan.single_fft_elements);
        const double single_log = std::log2(std::max(single_elems, 2.0));
        const double eff_threads = (fft_plan.single_fft_elements >= (uint64_t{1} << 17))
                                       ? std::min<double>(threads, 12.0)
                                       : 1.0;
        const double pocket_factor = (is_d ? 2.2 : 1.3) * (is_c ? 1.6 : 1.0);
        fft_cost_ns = 150e3 + single_elems * single_log * 0.25 * pocket_factor / eff_threads;
        (void)log2n;
    }

    return (fft_cost_ns < direct_cost_ns) ? SM_SIGNAL_METHOD_FFT : SM_SIGNAL_METHOD_DIRECT;
}

} // namespace

uint32_t promote_dtype(uint32_t d1, uint32_t d2) {
    (void)array::dtype_size(d1);
    (void)array::dtype_size(d2);
    if (array::is_int_or_bool(d1) || array::is_int_or_bool(d2)) {
        throw Error(SM_ENOTSUP, "signal convolution takes floating-point or complex arrays");
    }
    const bool c = array::is_complex(d1) || array::is_complex(d2);
    const bool dbl = array::is_double(d1) || array::is_double(d2);
    if (c) return dbl ? SM_COMPLEX128 : SM_COMPLEX64;
    return dbl ? SM_FLOAT64 : SM_FLOAT32;
}

FftConvPlan plan_fft_conv(const array::Shape& s1_in, const array::Shape& s2_in,
                          uint32_t out_dtype, uint32_t mode, bool correlate,
                          bool allow_oa, bool force_oa,
                          const uint32_t* axes_in, uint32_t axis_count_in,
                          const fft::Limits& limits, uint32_t center) {
    const uint32_t ndim = s1_in.ndim;
    if (ndim < 1 || ndim > SM_MAX_NDIM || s2_in.ndim != ndim) {
        throw Error(SM_EINVAL, "signal convolution inputs must have the same rank (1..8)");
    }
    if (mode > SM_SIGNAL_VALID) throw Error(SM_EINVAL, "unknown convolution mode");

    uint32_t raw_mask = 0;
    if (axis_count_in == 0) {
        raw_mask = (uint32_t{1} << ndim) - 1u;
    } else {
        if (!axes_in || axis_count_in > ndim) throw Error(SM_EINVAL, "invalid axes");
        for (uint32_t i = 0; i < axis_count_in; ++i) {
            if (axes_in[i] >= ndim) throw Error(SM_EINVAL, "axis out of range");
            if (raw_mask & (1u << axes_in[i])) throw Error(SM_EINVAL, "duplicate axis");
            raw_mask |= (1u << axes_in[i]);
        }
    }

    FftConvPlan plan;
    plan.ndim = ndim;
    plan.out_dtype = out_dtype;
    plan.is_complex = array::is_complex(out_dtype);
    plan.correlate = correlate;
    plan.raw_axis_mask = raw_mask;

    // Eliminate axes where s1[d] == 1 or s2[d] == 1 (SciPy _init_freq_conv_axes).
    for (uint32_t d = 0; d < ndim; ++d) {
        if ((raw_mask & (1u << d)) && s1_in.dims[d] != 1 && s2_in.dims[d] != 1) {
            plan.axis_mask |= (1u << d);
            plan.axes[plan.axis_count++] = d;
        } else {
            const uint64_t a = s1_in.dims[d], b = s2_in.dims[d];
            if (a != b && a != 1 && b != 1) {
                throw Error(SM_EINVAL, "incompatible non-transformed shapes for broadcasting");
            }
        }
    }

    bool ok1 = true, ok2 = true;
    for (uint32_t i = 0; i < plan.axis_count; ++i) {
        const uint32_t d = plan.axes[i];
        if (s1_in.dims[d] < s2_in.dims[d]) ok1 = false;
        if (s2_in.dims[d] < s1_in.dims[d]) ok2 = false;
    }
    if (mode == SM_SIGNAL_VALID) {
        if (!ok1 && !ok2) {
            throw Error(SM_EINVAL, "for valid mode, one input must be at least as large as the other along axes");
        }
        plan.swapped = !ok1;
    }

    plan.s1 = plan.swapped ? s2_in : s1_in;
    plan.s2 = plan.swapped ? s1_in : s2_in;

    const uint32_t last = ndim - 1;
    plan.real_last_axis = !plan.is_complex && ((plan.axis_mask & (1u << last)) != 0);

    uint64_t full_shape[SM_MAX_NDIM] = {};
    plan.out_shape.ndim = ndim;
    plan.out_shape.elements = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
        const uint64_t a = plan.s1.dims[d];
        const uint64_t b = plan.s2.dims[d];
        if (plan.axis_mask & (1u << d)) {
            full_shape[d] = checked_full_len(a, b);
            if (mode == SM_SIGNAL_FULL) plan.out_shape.dims[d] = full_shape[d];
            else if (mode == SM_SIGNAL_SAME) plan.out_shape.dims[d] = a;
            else plan.out_shape.dims[d] = a - b + 1;
            plan.overlap_elements = saturating_mul(plan.overlap_elements, std::min(a, b));
        } else {
            full_shape[d] = std::max(a, b);
            plan.out_shape.dims[d] = (mode == SM_SIGNAL_SAME) ? a : full_shape[d];
        }
        if (mode == SM_SIGNAL_SAME && correlate && center == SM_SIGNAL_CENTER_2D_CORR) {
            plan.crop_start[d] = (full_shape[d] - plan.out_shape.dims[d] + 1) / 2;
        } else {
            plan.crop_start[d] = (full_shape[d] - plan.out_shape.dims[d]) / 2;
        }
        if (sm_checked_mul(plan.out_shape.elements, plan.out_shape.dims[d], &plan.out_shape.elements) != SM_OK) {
            throw Error(SM_EOVERFLOW, "output shape overflows");
        }
    }

    uint64_t single_len[SM_MAX_NDIM] = {};
    plan.single_fft_elements = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
        if (plan.axis_mask & (1u << d)) {
            const bool is_last = (d == last);
            single_len[d] = fft::next_fast_len(full_shape[d], !plan.is_complex && is_last, is_last, limits);
            plan.single_fft_elements = saturating_mul(plan.single_fft_elements, single_len[d]);
        } else {
            single_len[d] = std::max(plan.s1.dims[d], plan.s2.dims[d]);
            plan.single_fft_elements = saturating_mul(plan.single_fft_elements, single_len[d]);
        }
    }

    // Evaluate OA decomposition if allowed.
    bool any_oa = false;
    uint64_t oa_block[SM_MAX_NDIM] = {}, oa_overlap[SM_MAX_NDIM] = {};
    uint64_t oa_step1[SM_MAX_NDIM] = {}, oa_step2[SM_MAX_NDIM] = {};
    uint64_t oa_n1[SM_MAX_NDIM] = {}, oa_n2[SM_MAX_NDIM] = {}, oa_nout[SM_MAX_NDIM] = {};
    uint64_t oa_total_elems = 1;

    if (allow_oa && plan.axis_count > 0) {
        for (uint32_t d = 0; d < ndim; ++d) {
            if (!(plan.axis_mask & (1u << d))) {
                oa_block[d] = std::max(plan.s1.dims[d], plan.s2.dims[d]);
                oa_step1[d] = plan.s1.dims[d];
                oa_step2[d] = plan.s2.dims[d];
                oa_n1[d] = oa_n2[d] = oa_nout[d] = 1;
                oa_total_elems = saturating_mul(oa_total_elems, oa_block[d]);
                continue;
            }
            const bool is_last = (d == last);
            uint64_t bsize = 0, ov = 0, st1 = 0, st2 = 0;
            calc_oa_lens(plan.s1.dims[d], plan.s2.dims[d], !plan.is_complex && is_last, is_last, limits,
                         bsize, ov, st1, st2);
            if (ov > 0) {
                any_oa = true;
                oa_block[d] = bsize;
                oa_overlap[d] = ov;
                oa_step1[d] = st1;
                oa_step2[d] = st2;
                const uint64_t eff_step = bsize - ov;
                oa_n1[d] = oa_step_count(plan.s1.dims[d], st1, eff_step, full_shape[d]);
                oa_n2[d] = oa_step_count(plan.s2.dims[d], st2, eff_step, full_shape[d]);
                oa_nout[d] = std::max(oa_n1[d], oa_n2[d]);
            } else {
                oa_block[d] = single_len[d];
                oa_overlap[d] = 0;
                oa_step1[d] = plan.s1.dims[d];
                oa_step2[d] = plan.s2.dims[d];
                oa_n1[d] = oa_n2[d] = oa_nout[d] = 1;
            }
            oa_total_elems = saturating_mul(oa_total_elems, saturating_mul(oa_nout[d], oa_block[d]));
        }
    }

    plan.oa_fft_elements = any_oa ? oa_total_elems : plan.single_fft_elements;
    plan.use_oa = any_oa && (force_oa || plan.oa_fft_elements < plan.single_fft_elements);
    plan.effective_fft_elements = plan.use_oa ? plan.oa_fft_elements : plan.single_fft_elements;

    for (uint32_t d = 0; d < ndim; ++d) {
        if (plan.use_oa) {
            plan.fft_len[d] = oa_block[d];
            plan.overlap[d] = oa_overlap[d];
            plan.step1[d] = oa_step1[d];
            plan.step2[d] = oa_step2[d];
            plan.nsteps1[d] = oa_n1[d];
            plan.nsteps2[d] = oa_n2[d];
            plan.nsteps_out[d] = oa_nout[d];
        } else {
            plan.fft_len[d] = single_len[d];
            plan.overlap[d] = 0;
            plan.step1[d] = plan.s1.dims[d];
            plan.step2[d] = plan.s2.dims[d];
            plan.nsteps1[d] = 1;
            plan.nsteps2[d] = 1;
            plan.nsteps_out[d] = 1;
        }
    }
    return plan;
}

} // namespace sm::detail::signal

using sm::detail::Error;
namespace array = sm::detail::array;
namespace dense = sm::detail::dense_correlate;
namespace router = sm::detail::router;
namespace sig = sm::detail::signal;

sm_status sm_signal_fftconvolve(sm_context* context, const sm_array* in1, const sm_array* in2,
                                const sm_signal_fftconv_options* options, const sm_route_options* route_options,
                                sm_array_result* out, sm_route_info* info) {
    return sm::detail::api(context, [&] {
        if (!out) throw Error(SM_EINVAL, "null output");
        array::check_route_info(info);
        array::check_options(route_options);
        if (!options) throw Error(SM_EINVAL, "null fftconvolve options");
        if (options->struct_size != sizeof(sm_signal_fftconv_options) ||
            options->abi_version != SM_SIGNAL_ABI_VERSION)
            throw Error(SM_EINVAL, "fftconvolve options size or ABI version mismatch");
        if (options->reserved0 || options->reserved[0] || options->reserved[1])
            throw Error(SM_EINVAL, "fftconvolve options reserved fields must be zero");
        if (options->mode > SM_SIGNAL_VALID) throw Error(SM_EINVAL, "unknown convolution mode");

        const array::Shape s1 = array::checked_shape(in1);
        const array::Shape s2 = array::checked_shape(in2);
        if (s1.ndim != s2.ndim) throw Error(SM_EINVAL, "in1 and in2 must have the same dimensionality");
        if (s1.elements == 0 || s2.elements == 0) throw Error(SM_ENOTSUP, "empty signal input");

        const uint32_t out_dtype = sig::promote_dtype(in1->dtype, in2->dtype);
#if defined(SM_ENABLE_METAL)
        const sm::detail::fft::Limits limits = sm::detail::fft::device_limits();
        const bool allow_oa = options->overlap_add != 0;
        const sig::FftConvPlan plan = sig::plan_fft_conv(
            s1, s2, out_dtype, options->mode, options->correlate != 0,
            allow_oa, allow_oa, options->axes, options->axis_count, limits);
        uint32_t compute_dtype = array::single_of(out_dtype);
        sm::detail::BufferPtr result = sig::run_fft_conv_gpu(context, *in1, *in2, plan, *route_options, compute_dtype);
        array::publish(out, std::move(result), out_dtype, plan.out_shape);
        array::report(info, SM_EXECUTION_GPU, compute_dtype, "");
#else
        const sm::detail::fft::Limits limits{};
        const bool allow_oa = options->overlap_add != 0;
        (void)sig::plan_fft_conv(s1, s2, out_dtype, options->mode, options->correlate != 0,
                                 allow_oa, allow_oa, options->axes, options->axis_count, limits);
        throw Error(SM_ENOTSUP, "this build has no GPU engine and there is no native CPU FFT convolution");
#endif
    });
}

static sm_status signal_convolve(sm_context* context, const sm_array* in1, const sm_array* in2,
                                const sm_signal_conv_options* options, const sm_route_options* route_options,
                                sm_array_result* out, sm_route_info* info, bool scipy_fir) {
    return sm::detail::api(context, [&] {
        if (!out) throw Error(SM_EINVAL, "null output");
        array::check_route_info(info);
        array::check_options(route_options);
        if (!options) throw Error(SM_EINVAL, "null signal conv options");
        if (options->struct_size != sizeof(sm_signal_conv_options) ||
            options->abi_version != SM_SIGNAL_ABI_VERSION)
            throw Error(SM_EINVAL, "signal conv options size or ABI version mismatch");
        if (options->reserved[0] || options->reserved[1])
            throw Error(SM_EINVAL, "signal conv options reserved fields must be zero");
        if (options->mode > SM_SIGNAL_VALID) throw Error(SM_EINVAL, "unknown convolution mode");
        if (options->method > SM_SIGNAL_METHOD_FFT) throw Error(SM_EINVAL, "unknown convolution method");
        if (options->boundary > SM_SIGNAL_BOUNDARY_SYMM) throw Error(SM_EINVAL, "unknown boundary mode");
        if (options->center > SM_SIGNAL_CENTER_2D_CORR) throw Error(SM_EINVAL, "unknown same-mode center");

        const array::Shape s1 = array::checked_shape(in1);
        const array::Shape s2 = array::checked_shape(in2);
        if (s1.ndim != s2.ndim) throw Error(SM_EINVAL, "in1 and in2 must have the same dimensionality");
        if (s1.elements == 0 || s2.elements == 0) throw Error(SM_ENOTSUP, "empty signal input");

        const uint32_t ndim = s1.ndim;
        const uint32_t out_dtype = sig::promote_dtype(in1->dtype, in2->dtype);
        if (scipy_fir &&
            (ndim != 2 || in1->dtype != in2->dtype || (out_dtype != SM_FLOAT32 && out_dtype != SM_FLOAT64) ||
             options->center != SM_SIGNAL_CENTER_2D_CORR || options->method != SM_SIGNAL_METHOD_DIRECT ||
             route_options->precision != SM_PRECISION_STRICT ||
             route_options->execution == SM_EXECUTION_GPU)) {
            throw Error(SM_ENOTSUP, "strict FIR requires direct CPU execution for same-dtype real 2-D inputs");
        }

        if (options->boundary == SM_SIGNAL_BOUNDARY_FILL && !array::is_complex(out_dtype) &&
            options->fillvalue_imag != 0.0) {
            throw Error(SM_EINVAL, "real convolution cannot use a complex fillvalue");
        }
        const bool has_custom_boundary =
            options->boundary != SM_SIGNAL_BOUNDARY_FILL ||
            options->fillvalue_real != 0.0 || options->fillvalue_imag != 0.0;
        if (has_custom_boundary && options->method == SM_SIGNAL_METHOD_FFT) {
            throw Error(SM_EINVAL, "FFT method requires zero-fill boundary");
        }

        // Validate all-dimension dominance for VALID mode ahead of method dispatch
        // (for both Metal and CPU-only builds, and both FFT and direct methods).
        bool ok1 = true, ok2 = true;
        for (uint32_t d = 0; d < ndim; ++d) {
            if (s1.dims[d] < s2.dims[d]) ok1 = false;
            if (s2.dims[d] < s1.dims[d]) ok2 = false;
        }
        if (options->mode == SM_SIGNAL_VALID && !ok1 && !ok2) {
            throw Error(SM_EINVAL, "for valid mode, one input must be at least as large as the other in every dimension");
        }

        uint32_t method = options->method;
        if (method == SM_SIGNAL_METHOD_AUTO) {
            if (has_custom_boundary) {
                method = SM_SIGNAL_METHOD_DIRECT;
            } else {
                method = sig::choose_conv_method_impl(context, out_dtype, s1, s2, options->mode,
                                                      options->host_threads, *route_options);
            }
        }

        if (method == SM_SIGNAL_METHOD_FFT) {
#if defined(SM_ENABLE_METAL)
            const sm::detail::fft::Limits limits = sm::detail::fft::device_limits();
            const bool allow_oa = (options->method == SM_SIGNAL_METHOD_AUTO);
            sig::FftConvPlan plan = sig::plan_fft_conv(
                s1, s2, out_dtype, options->mode, options->correlate != 0,
                allow_oa, /*force_oa=*/false, nullptr, 0, limits, options->center);
            if (plan.axis_count == 0) {
                throw Error(SM_ENOTSUP, "no axes to transform with FFT; SciPy is used");
            }
            uint32_t compute_dtype = array::single_of(out_dtype);
            uint32_t nf_flags = 0;
            sm::detail::BufferPtr result = sig::run_fft_conv_gpu(
                context, *in1, *in2, plan, *route_options, compute_dtype, &nf_flags);
            if (nf_flags != 0 && plan.use_oa) {
                // SciPy convolve/correlate(..., method='auto') routes to single-FFT,
                // propagating NaN/Inf across the full output rather than isolating to an OA block.
                result.reset();
                plan = sig::plan_fft_conv(
                    s1, s2, out_dtype, options->mode, options->correlate != 0,
                    /*allow_oa=*/false, /*force_oa=*/false, nullptr, 0, limits, options->center);
                nf_flags = 0;
                result = sig::run_fft_conv_gpu(
                    context, *in1, *in2, plan, *route_options, compute_dtype, &nf_flags);
            }
            const bool out_nonfinite = (nf_flags & 4u) != 0;
            array::publish(out, std::move(result), out_dtype, plan.out_shape);
            array::report(info, SM_EXECUTION_GPU, compute_dtype, "");
            if (info) {
                info->reserved[0] = SM_SIGNAL_ROUTE_FFT | (out_nonfinite ? SM_SIGNAL_ROUTE_NONFINITE : 0ull);
            }
            return;
#else
            throw Error(SM_ENOTSUP, "this build has no GPU engine and there is no native CPU FFT convolution");
#endif
        }

        // Direct path via shared dense_correlate primitive.
        bool swap = false;
        if (options->mode == SM_SIGNAL_VALID) {
            swap = !ok1;
        } else if (options->center == SM_SIGNAL_CENTER_ND && options->mode == SM_SIGNAL_FULL &&
                   !has_custom_boundary && s2.elements > s1.elements) {
            swap = true;
        }

        const sm_array& in_arr = swap ? *in2 : *in1;
        const sm_array& k_arr = swap ? *in1 : *in2;
        const array::Shape& in_sh = swap ? s2 : s1;
        const array::Shape& k_sh = swap ? s1 : s2;
        const bool rev_conj_out = swap && (options->correlate != 0);

        array::Shape out_shape;
        out_shape.ndim = ndim;
        out_shape.elements = 1;

        dense::StencilSpec spec;
        spec.in_shape = in_sh;
        spec.in_dtype = in_arr.dtype;
        spec.out_dtype = out_dtype;
        spec.set_mode(sig::to_dense_mode(options->boundary));
        spec.cval_real = options->fillvalue_real;
        spec.cval_imag = options->fillvalue_imag;
        spec.cpu_accum = dense::AccumMode::InputDtype;
        spec.prune_zeros = false; // preserve 0.0 * NaN = NaN per SciPy _correlate_nd.cc
        // SciPy 1-D convolve/correlate (full, valid, and same when len(in1) >= len(in2))
        // dispatches to np.convolve/np.correlate, summing only in-bounds overlapping pairs
        // without computing 0.0 * out_of_bounds_in2.
        spec.omit_oob = (ndim == 1) && !has_custom_boundary &&
                        options->center == SM_SIGNAL_CENTER_ND &&
                        (options->mode == SM_SIGNAL_FULL ||
                         (options->mode == SM_SIGNAL_SAME && s1.dims[0] >= s2.dims[0]));
        spec.weights_dtype = k_arr.dtype;

        for (uint32_t d = 0; d < ndim; ++d) {
            if (k_sh.dims[d] > UINT32_MAX) {
                throw Error(SM_ENOTSUP, "kernel dimension exceeds 32 bits");
            }
            spec.wshape[d] = static_cast<uint32_t>(k_sh.dims[d]);
            if (options->mode == SM_SIGNAL_FULL) {
                out_shape.dims[d] = sig::checked_full_len(in_sh.dims[d], k_sh.dims[d]);
                spec.start[d] = -static_cast<int64_t>(k_sh.dims[d] - 1);
            } else if (options->mode == SM_SIGNAL_VALID) {
                out_shape.dims[d] = in_sh.dims[d] - k_sh.dims[d] + 1;
                spec.start[d] = 0;
            } else {
                out_shape.dims[d] = in_sh.dims[d];
                if (options->correlate != 0 && options->center == SM_SIGNAL_CENTER_2D_CORR) {
                    spec.start[d] = -static_cast<int64_t>((k_sh.dims[d] - 1) / 2);
                } else {
                    spec.start[d] = -static_cast<int64_t>(k_sh.dims[d] / 2);
                }
            }
            if (sm_checked_mul(out_shape.elements, out_shape.dims[d], &out_shape.elements) != SM_OK) {
                throw Error(SM_EOVERFLOW, "output element count overflows");
            }
        }
        spec.out_shape = out_shape;

        if (scipy_fir) {
            spec.weights = k_arr.data;
            auto result = dense::run_scipy_fir(context, spec, in_arr.data, options->correlate != 0);
            if (rev_conj_out)
                sig::reverse_conj_cpu(sm::detail::mutable_data(result.get()), out_shape.elements, out_dtype, false);
            array::publish(out, std::move(result), out_dtype, out_shape);
            array::report(info, SM_EXECUTION_CPU, out_dtype, "verified SciPy arm64 FIR accumulation order");
            return;
        }

        std::vector<uint8_t> weight_buf;
        sig::copy_kernel_weights(k_arr, k_sh, options->correlate != 0, weight_buf);
        spec.weights = weight_buf.data();
        const dense::TapTable taps = dense::build_taps(spec);
        // If complex output with real input, GPU path converts real input -> complex64;
        // CPU path also needs complex input if fillvalue has non-zero imaginary part.
        const bool real_in_complex_out = array::is_complex(out_dtype) && !array::is_complex(in_arr.dtype);
        std::vector<uint8_t> promoted_in_buf;
        sm_array cpu_in_arr = in_arr;
        if (options->boundary == SM_SIGNAL_BOUNDARY_FILL && real_in_complex_out && options->fillvalue_imag != 0.0) {
            const uint32_t target_in_dtype = array::is_double(in_arr.dtype) ? SM_COMPLEX128 : SM_COMPLEX64;
            promoted_in_buf.resize(in_sh.elements * array::dtype_size(target_in_dtype));
            if (in_arr.dtype == SM_FLOAT32) {
                const auto* src = static_cast<const float*>(in_arr.data);
                auto* dst = reinterpret_cast<float*>(promoted_in_buf.data());
                for (uint64_t i = 0; i < in_sh.elements; ++i) {
                    dst[2 * i] = src[i];
                    dst[2 * i + 1] = 0.0f;
                }
            } else {
                const auto* src = static_cast<const double*>(in_arr.data);
                auto* dst = reinterpret_cast<double*>(promoted_in_buf.data());
                for (uint64_t i = 0; i < in_sh.elements; ++i) {
                    dst[2 * i] = src[i];
                    dst[2 * i + 1] = 0.0;
                }
            }
            cpu_in_arr.dtype = target_in_dtype;
            cpu_in_arr.data = promoted_in_buf.data();
            spec.in_dtype = target_in_dtype;
        }
        dense::StencilSpec gpu_spec = spec;
        if (real_in_complex_out) gpu_spec.in_dtype = SM_COMPLEX64;

        sm_route_options route = *route_options;
        if (!dense::gpu_supports(gpu_spec, taps)) {
            if (route.execution == SM_EXECUTION_GPU)
                throw Error(SM_ENOTSUP, "the array or kernel is too large for the direct GPU engine");
            route.execution = SM_EXECUTION_CPU;
        }
        if (route.precision == SM_PRECISION_STRICT && route.execution == SM_EXECUTION_AUTO &&
            array::is_double(out_dtype)) {
            route.execution = SM_EXECUTION_CPU;
        }

        const uint64_t work = sig::saturating_mul(out_shape.elements, std::max<uint64_t>(1, taps.taps.size()));
        const router::Decision decision = router::decide(context, sig::direct_family, route, out_dtype, work);

        sm::detail::BufferPtr result;
        uint32_t engine = decision.engine;
        const bool cpu_uses_f32 = (spec.in_dtype == SM_FLOAT32 || spec.in_dtype == SM_COMPLEX64) &&
                                  (out_dtype == SM_FLOAT32 || out_dtype == SM_COMPLEX64);
        uint32_t compute_dtype = cpu_uses_f32 ? out_dtype : (array::is_complex(out_dtype) ? SM_COMPLEX128 : SM_FLOAT64);
        std::string reason;

        if (decision.engine == SM_EXECUTION_GPU) {
#if defined(SM_ENABLE_METAL)
            const bool allow_fallback = route_options->execution == SM_EXECUTION_AUTO;
            bool has_nonfinite_tap = false;
            if (spec.omit_oob) {
                for (const dense::Tap& t : taps.taps) {
                    if (!std::isfinite(t.weight_re_f32) ||
                        (taps.is_complex_weights && !std::isfinite(t.weight_im_f32))) {
                        has_nonfinite_tap = true;
                        break;
                    }
                }
            }
            if (spec.omit_oob && has_nonfinite_tap) {
                reason = "signal direct convolution with non-finite kernel omits out-of-bounds taps; re-routed to CPU";
                if (!allow_fallback) throw Error(SM_ENOTSUP, reason);
                engine = SM_EXECUTION_CPU;
                result = dense::run_cpu(context, spec, taps, cpu_in_arr.data);
                if (rev_conj_out) sig::reverse_conj_cpu(sm::detail::mutable_data(result.get()), out_shape.elements, out_dtype, true);
            } else {
            // Check mixed-dtype (float32 input * float64 kernel) overflow before GPU dispatch.
            bool mixed_overflow = false;
            if (!array::is_double(in_arr.dtype) && array::is_double(out_dtype)) {
                const bool has_fill = (options->boundary == SM_SIGNAL_BOUNDARY_FILL);
                const bool in_is_complex = array::is_complex(in_arr.dtype);
                const bool cval_is_complex = array::is_complex(gpu_spec.in_dtype);
                if (taps.weights_overflow_f32 ||
                    (has_fill && (sig::overflows_float32(spec.cval_real) ||
                                  (cval_is_complex && sig::overflows_float32(spec.cval_imag))))) {
                    mixed_overflow = true;
                } else {
                    double max_in = 0.0;
                    const float* p = static_cast<const float*>(in_arr.data);
                    const uint64_t nscal = in_sh.elements * (in_is_complex ? 2 : 1);
                    for (uint64_t i = 0; i < nscal; ++i) {
                        const double v = std::fabs(static_cast<double>(p[i]));
                        if (std::isfinite(v) && v > max_in) max_in = v;
                    }
                    if (has_fill && std::isfinite(spec.cval_real))
                        max_in = std::max(max_in, std::fabs(static_cast<double>(static_cast<float>(spec.cval_real))));
                    if (has_fill && cval_is_complex && std::isfinite(spec.cval_imag))
                        max_in = std::max(max_in, std::fabs(static_cast<double>(static_cast<float>(spec.cval_imag))));
                    double gain = 0.0;
                    for (const dense::Tap& t : taps.taps) {
                        const double ar = std::fabs(static_cast<double>(t.weight_re_f32));
                        if (std::isfinite(ar)) gain += ar;
                        if (taps.is_complex_weights) {
                            const double ai = std::fabs(static_cast<double>(t.weight_im_f32));
                            if (std::isfinite(ai)) gain += ai;
                        }
                    }
                    constexpr double u = 0x1p-24;
                    const double e = gain * u * max_in + static_cast<double>(taps.taps.size() + 3) * u * gain * max_in;
                    const double bound = max_in * gain + e;
                    if (!std::isfinite(bound) || bound >= sig::float32_overflow_threshold) {
                        mixed_overflow = true;
                    }
                }
                if (mixed_overflow) {
                    reason = "mixed-precision direct filter gain overflowed float32; re-routed to CPU";
                }
            }

            if (mixed_overflow) {
                if (!allow_fallback) throw Error(SM_ENOTSUP, reason);
                engine = SM_EXECUTION_CPU;
                result = dense::run_cpu(context, spec, taps, cpu_in_arr.data);
                if (rev_conj_out) sig::reverse_conj_cpu(sm::detail::mutable_data(result.get()), out_shape.elements, out_dtype, true);
            } else if (!real_in_complex_out && !rev_conj_out) {
                result = dense::run_gpu(context, spec, taps, in_arr, allow_fallback, reason);
                if (result) {
                    compute_dtype = array::single_of(out_dtype);
                } else {
                    engine = SM_EXECUTION_CPU;
                    result = dense::run_cpu(context, spec, taps, cpu_in_arr.data);
                }
            } else {
                // Handle real-input+complex-kernel or swapped-correlate rev+conj in one gpu::Batch.
                sm::detail::BufferPtr single_in;
                array::ConvertReport report;
                const bool in_is_double = array::is_double(in_arr.dtype);
                if (!real_in_complex_out) {
                    single_in = array::load_single(context, in_arr, in_sh, in_is_double ? &report : nullptr);
                } else {
                    sm::detail::BufferPtr tmp_real =
                        array::load_single(context, in_arr, in_sh, in_is_double ? &report : nullptr);
                    single_in = array::allocate(context, in_sh.elements * 2 * sizeof(float));
                    const float* rptr = reinterpret_cast<const float*>(sm::detail::read(tmp_real.get(), context).data);
                    float* cptr = reinterpret_cast<float*>(sm::detail::mutable_data(single_in.get()));
                    sm::detail::parallel::parallel_for(in_sh.elements, uint64_t{1} << 17, [&](unsigned, uint64_t b, uint64_t e) {
                        for (uint64_t i = b; i < e; ++i) {
                            cptr[2 * i] = rptr[i];
                            cptr[2 * i + 1] = 0.0f;
                        }
                    });
                }

                // A5 range checks.
                bool overflowed = taps.weights_overflow_f32 || (in_is_double && report.overflowed);
                if (!overflowed && options->boundary == SM_SIGNAL_BOUNDARY_FILL &&
                    (sig::overflows_float32(spec.cval_real) ||
                     (array::is_complex(gpu_spec.in_dtype) && sig::overflows_float32(spec.cval_imag)))) {
                    overflowed = true;
                }
                if (!overflowed && (in_is_double || array::is_double(out_dtype))) {
                    double m = in_is_double ? std::fabs(static_cast<double>(static_cast<float>(report.max_finite))) : 0.0;
                    if (!in_is_double) {
                        const float* p = reinterpret_cast<const float*>(sm::detail::read(single_in.get(), context).data);
                        const uint64_t nscal = in_sh.elements * (array::is_complex(gpu_spec.in_dtype) ? 2 : 1);
                        for (uint64_t i = 0; i < nscal; ++i) {
                            const double v = std::fabs(static_cast<double>(p[i]));
                            if (std::isfinite(v) && v > m) m = v;
                        }
                    }
                    if (options->boundary == SM_SIGNAL_BOUNDARY_FILL && std::isfinite(spec.cval_real))
                        m = std::max(m, std::fabs(static_cast<double>(static_cast<float>(spec.cval_real))));
                    if (options->boundary == SM_SIGNAL_BOUNDARY_FILL && array::is_complex(gpu_spec.in_dtype) &&
                        std::isfinite(spec.cval_imag))
                        m = std::max(m, std::fabs(static_cast<double>(static_cast<float>(spec.cval_imag))));
                    double gain = 0.0;
                    for (const dense::Tap& t : taps.taps) {
                        const double ar = std::fabs(static_cast<double>(t.weight_re_f32));
                        if (std::isfinite(ar)) gain += ar;
                        if (taps.is_complex_weights) {
                            const double ai = std::fabs(static_cast<double>(t.weight_im_f32));
                            if (std::isfinite(ai)) gain += ai;
                        }
                    }
                    constexpr double u = 0x1p-24;
                    const double e = gain * u * m + static_cast<double>(taps.taps.size() + 3) * u * gain * m;
                    const double bound = m * gain + e;
                    if (!std::isfinite(bound) || bound >= sig::float32_overflow_threshold) overflowed = true;
                }

                if (overflowed) {
                    reason = "float64 values exceed single-precision range; re-routed to CPU";
                    if (!allow_fallback) throw Error(SM_ENOTSUP, reason);
                    single_in.reset();
                    engine = SM_EXECUTION_CPU;
                    result = dense::run_cpu(context, spec, taps, cpu_in_arr.data);
                    if (rev_conj_out)
                        sig::reverse_conj_cpu(sm::detail::mutable_data(result.get()), out_shape.elements, out_dtype, true);
                } else {
                    sm::detail::GpuOperationScope operation(context, true);
                    sm::detail::gpu::Batch batch(context);
                    const uint64_t out_bytes = out_shape.elements * array::dtype_size(out_dtype);
                    if (!rev_conj_out) {
                        result = array::allocate(context, out_bytes);
                        sm::detail::BufferPtr tap_buf =
                            dense::encode_gpu(context, batch, gpu_spec, taps, single_in.get(), result.get());
                        batch.commit_and_wait();
                        operation.complete();
                    } else {
                        sm::detail::BufferPtr tmp = array::allocate(context, out_bytes);
                        result = array::allocate(context, out_bytes);
                        sm::detail::BufferPtr tap_buf =
                            dense::encode_gpu(context, batch, gpu_spec, taps, single_in.get(), tmp.get());
                        SMSignalRevConjParams rp{};
                        rp.elements = static_cast<sm_u32>(out_shape.elements);
                        rp.conj = array::is_complex(out_dtype) ? 1u : 0u;
                        const char* rkernel = out_dtype == SM_FLOAT32 ? "sm_signal_rev_conj_f32"
                                              : out_dtype == SM_FLOAT64 ? "sm_signal_rev_conj_f64"
                                              : out_dtype == SM_COMPLEX64 ? "sm_signal_rev_conj_c64"
                                              : "sm_signal_rev_conj_c128";
                        batch.dispatch(rkernel, rp,
                                       {sm::detail::gpu::bind(tmp.get(), context),
                                        sm::detail::gpu::bind(result.get(), context)},
                                       (out_shape.elements + SM_GPU_THREADS - 1) / SM_GPU_THREADS);
                        batch.commit_and_wait();
                        operation.complete();
                    }
                    compute_dtype = array::single_of(out_dtype);
                }
            }
            }
#else
            throw Error(SM_ENOTSUP, "GPU execution needs a Metal build");
#endif
        } else {
            result = dense::run_cpu(context, spec, taps, cpu_in_arr.data);
            if (rev_conj_out) {
                sig::reverse_conj_cpu(sm::detail::mutable_data(result.get()), out_shape.elements, out_dtype, true);
            }
        }
        array::publish(out, std::move(result), out_dtype, out_shape);
        array::report(info, engine, compute_dtype, reason.c_str());
    });
}

sm_status sm_signal_convolve(sm_context* context, const sm_array* in1, const sm_array* in2,
                             const sm_signal_conv_options* options, const sm_route_options* route_options,
                             sm_array_result* out, sm_route_info* info) {
    return signal_convolve(context, in1, in2, options, route_options, out, info, false);
}

sm_status sm_signal_convolve_scipy_fir(sm_context* context, const sm_array* in1, const sm_array* in2,
                                       const sm_signal_conv_options* options, const sm_route_options* route_options,
                                       sm_array_result* out, sm_route_info* info) {
    return signal_convolve(context, in1, in2, options, route_options, out, info, true);
}

sm_status sm_signal_choose_conv_method(sm_context* context, uint32_t dtype1, uint32_t ndim1, const uint64_t* shape1,
                                       uint32_t dtype2, uint32_t ndim2, const uint64_t* shape2, uint32_t mode,
                                       uint32_t host_threads, const sm_route_options* route_options,
                                       uint32_t* out_method) {
    return sm::detail::api(context, [&] {
        if (!out_method) throw Error(SM_EINVAL, "null out_method");
        array::check_options(route_options);
        if (mode > SM_SIGNAL_VALID) throw Error(SM_EINVAL, "unknown convolution mode");
        if (ndim1 < 1 || ndim1 > SM_MAX_NDIM || ndim1 != ndim2 || !shape1 || !shape2)
            throw Error(SM_EINVAL, "signal inputs must have matching rank 1..8");
        const uint32_t out_dtype = sig::promote_dtype(dtype1, dtype2);

        array::Shape s1, s2;
        s1.ndim = ndim1;
        s2.ndim = ndim2;
        s1.elements = 1;
        s2.elements = 1;
        for (uint32_t d = 0; d < ndim1; ++d) {
            s1.dims[d] = shape1[d];
            s2.dims[d] = shape2[d];
            if (sm_checked_mul(s1.elements, shape1[d], &s1.elements) != SM_OK ||
                sm_checked_mul(s2.elements, shape2[d], &s2.elements) != SM_OK) {
                throw Error(SM_EOVERFLOW, "array element count overflows");
            }
        }
        if (s1.elements == 0 || s2.elements == 0) throw Error(SM_ENOTSUP, "empty signal input");
        *out_method = sig::choose_conv_method_impl(context, out_dtype, s1, s2, mode, host_threads, *route_options);
    });
}
