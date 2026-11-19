// Native tests for signal convolution & correlation (PLAIN and VALIDATED),
// plus fft::next_fast_len and fft::Scratch + fft::encode single-Batch checks.
#include "ms_signal.h"
#include "core/array.hpp"
#include "fft/plan.hpp"
#if defined(MS_TEST_METAL)
#include "fft/engine.hpp"
#include "runtime/metal_compute.hpp"
#endif

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::fprintf(stderr, "signal convolve test failed line %d: %s\n", __LINE__, #x);                          \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

using namespace ms::detail;
using Complex = std::complex<double>;

namespace {

ms_context* ctx = nullptr;
bool have_gpu = false;
bool validated = false;

uint64_t product(const std::vector<uint64_t>& shape) {
    uint64_t n = 1;
    for (uint64_t v : shape) n *= v;
    return n;
}

std::vector<Complex> make_data(uint64_t n, uint32_t seed, bool is_complex) {
    std::vector<Complex> out(n);
    uint64_t state = 0x9E3779B97F4A7C15ull * (seed + 1);
    for (uint64_t i = 0; i < n; ++i) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const double re = static_cast<double>(state >> 11) / 9007199254740992.0 * 2.0 - 1.0 + 0.1;
        double im = 0.0;
        if (is_complex) {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            im = static_cast<double>(state >> 11) / 9007199254740992.0 * 2.0 - 1.0 - 0.05;
        }
        out[i] = {re, im};
    }
    return out;
}

std::vector<uint8_t> pack(const std::vector<Complex>& vals, uint32_t dtype) {
    const uint64_t n = vals.size();
    const uint64_t esize = array::dtype_size(dtype);
    std::vector<uint8_t> bytes(n * esize);
    if (dtype == MS_FLOAT32) {
        auto* p = reinterpret_cast<float*>(bytes.data());
        for (uint64_t i = 0; i < n; ++i) p[i] = static_cast<float>(vals[i].real());
    } else if (dtype == MS_FLOAT64) {
        auto* p = reinterpret_cast<double*>(bytes.data());
        for (uint64_t i = 0; i < n; ++i) p[i] = vals[i].real();
    } else if (dtype == MS_COMPLEX64) {
        auto* p = reinterpret_cast<float*>(bytes.data());
        for (uint64_t i = 0; i < n; ++i) {
            p[2 * i] = static_cast<float>(vals[i].real());
            p[2 * i + 1] = static_cast<float>(vals[i].imag());
        }
    } else {
        auto* p = reinterpret_cast<double*>(bytes.data());
        for (uint64_t i = 0; i < n; ++i) {
            p[2 * i] = vals[i].real();
            p[2 * i + 1] = vals[i].imag();
        }
    }
    return bytes;
}

std::vector<Complex> unpack(const ms_array_result& res) {
    const auto view = read(res.buffer, ctx);
    uint64_t n = 1;
    for (uint32_t d = 0; d < res.ndim; ++d) n *= res.shape[d];
    std::vector<Complex> out(n);
    if (res.dtype == MS_FLOAT32) {
        const auto* p = reinterpret_cast<const float*>(view.data);
        for (uint64_t i = 0; i < n; ++i) out[i] = {p[i], 0.0};
    } else if (res.dtype == MS_FLOAT64) {
        const auto* p = reinterpret_cast<const double*>(view.data);
        for (uint64_t i = 0; i < n; ++i) out[i] = {p[i], 0.0};
    } else if (res.dtype == MS_COMPLEX64) {
        const auto* p = reinterpret_cast<const float*>(view.data);
        for (uint64_t i = 0; i < n; ++i) out[i] = {p[2 * i], p[2 * i + 1]};
    } else {
        const auto* p = reinterpret_cast<const double*>(view.data);
        for (uint64_t i = 0; i < n; ++i) out[i] = {p[2 * i], p[2 * i + 1]};
    }
    return out;
}

ms_array describe(const std::vector<uint64_t>& shape, uint32_t dtype, const void* data) {
    ms_array a{};
    a.struct_size = sizeof(a);
    a.abi_version = MS_ARRAY_ABI_VERSION;
    a.data = data;
    a.dtype = dtype;
    a.ndim = static_cast<uint32_t>(shape.size());
    for (size_t d = 0; d < shape.size(); ++d) a.shape[d] = shape[d];
    return a;
}

ms_route_options route_opts(uint32_t execution, uint32_t precision) {
    ms_route_options o{};
    o.struct_size = sizeof(o);
    o.abi_version = MS_ARRAY_ABI_VERSION;
    o.execution = execution;
    o.precision = precision;
    return o;
}

ms_route_info new_info() {
    ms_route_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = MS_ARRAY_ABI_VERSION;
    return info;
}

ms_execution_info exec_info() {
    ms_execution_info ei{};
    ei.struct_size = sizeof(ei);
    ei.abi_version = MS_RUNTIME_ABI_VERSION;
    CHECK(ms_context_execution_info(ctx, &ei) == MS_OK);
    return ei;
}

int64_t extend_sig(int64_t i, int64_t n, uint32_t boundary) {
    if (i >= 0 && i < n) return i;
    if (boundary == MS_SIGNAL_BOUNDARY_WRAP) {
        int64_t m = i % n;
        return m < 0 ? m + n : m;
    }
    if (boundary == MS_SIGNAL_BOUNDARY_SYMM) {
        const int64_t period = 2 * n;
        int64_t m = i % period;
        if (m < 0) m += period;
        return m < n ? m : period - 1 - m;
    }
    return -1;
}

// Independent reference N-D convolution/correlation with broadcast axes support.
std::vector<Complex> ref_conv(const std::vector<Complex>& a, const std::vector<uint64_t>& s1,
                              const std::vector<Complex>& b, const std::vector<uint64_t>& s2,
                              uint32_t mode, bool correlate, uint32_t boundary,
                              Complex fill, uint32_t center,
                              const std::vector<uint32_t>& axes_in, std::vector<uint64_t>& out_shape) {
    const size_t ndim = s1.size();
    std::vector<bool> is_axis(ndim, false);
    if (axes_in.empty()) {
        std::fill(is_axis.begin(), is_axis.end(), true);
    } else {
        for (uint32_t ax : axes_in) is_axis[ax] = true;
    }
    for (size_t d = 0; d < ndim; ++d) {
        if (s1[d] == 1 || s2[d] == 1) {
            if (!axes_in.empty() || true) {
                // Keep axis in direct mode (when boundary != FILL or axes_in empty),
                // but note for fftconvolve axes_in with unit dims collapse to broadcast.
            }
        }
    }

    // For direct/all-axes reference:
    std::vector<uint64_t> st1(ndim, 1), st2(ndim, 1);
    for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
        st1[d] = st1[d + 1] * s1[d + 1];
        st2[d] = st2[d + 1] * s2[d + 1];
    }

    // Effective axes for FFT mode vs all axes for direct.
    const bool freq_axes_mode = !axes_in.empty();
    std::vector<bool> eff_ax(ndim, false);
    for (size_t d = 0; d < ndim; ++d) {
        eff_ax[d] = is_axis[d] && (!freq_axes_mode || (s1[d] != 1 && s2[d] != 1));
    }
    // Note: if axes_in is empty and we're computing general conv (all dims), eff_ax = true except when
    // freq_axes_mode? Actually for direct/fftconvolve with all axes, s1[d]==1 || s2[d]==1 gives
    // identical full length max(s1,s2) and valid length max-min+1! Only 'same' differs if s1[d]==1 && s2[d]>1:
    // direct keeps s1[d]=1 centered at k/2, and fftconvolve _centered(full, s1) ALSO centers at (s2-1)/2 == s2//2!
    // So both give identical results!

    bool ok1 = true, ok2 = true;
    for (size_t d = 0; d < ndim; ++d) {
        if (eff_ax[d] || !freq_axes_mode) {
            if (s1[d] < s2[d]) ok1 = false;
            if (s2[d] < s1[d]) ok2 = false;
        }
    }
    const bool swapped = (mode == MS_SIGNAL_VALID && !ok1);
    const auto& x = swapped ? b : a;
    const auto& y = swapped ? a : b;
    const auto& sx = swapped ? s2 : s1;
    const auto& sy = swapped ? s1 : s2;
    const auto& stx = swapped ? st2 : st1;
    const auto& sty = swapped ? st1 : st2;

    out_shape.resize(ndim);
    std::vector<uint64_t> full_shape(ndim), crop(ndim);
    for (size_t d = 0; d < ndim; ++d) {
        if (eff_ax[d]) {
            full_shape[d] = sx[d] + sy[d] - 1;
            if (mode == MS_SIGNAL_FULL) out_shape[d] = full_shape[d];
            else if (mode == MS_SIGNAL_SAME) out_shape[d] = sx[d];
            else out_shape[d] = sx[d] - sy[d] + 1;
        } else {
            full_shape[d] = std::max(sx[d], sy[d]);
            out_shape[d] = (mode == MS_SIGNAL_SAME) ? sx[d] : full_shape[d];
        }
        crop[d] = (full_shape[d] - out_shape[d]) / 2;
    }

    const uint64_t out_n = product(out_shape);
    std::vector<Complex> out(out_n, Complex(0.0, 0.0));
    std::vector<uint64_t> p(ndim, 0);

    for (uint64_t oi = 0; oi < out_n; ++oi) {
        uint64_t rem = oi;
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            p[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }
        // Check if eff_ax: convolve/correlate along eff_ax, broadcast along non-eff_ax.
        // Sum over all y coordinates along eff_ax.
        std::vector<uint64_t> yc(ndim, 0);
        uint64_t conv_count = 1;
        for (size_t d = 0; d < ndim; ++d) {
            if (eff_ax[d]) conv_count *= sy[d];
        }
        Complex sum(0.0, 0.0);
        for (uint64_t ci = 0; ci < conv_count; ++ci) {
            uint64_t crem = ci;
            bool ok = true;
            uint64_t x_idx = 0, y_idx = 0;
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                if (eff_ax[d]) {
                    const uint64_t k = crem % sy[d];
                    crem /= sy[d];
                    int64_t pos = 0;
                    if (!correlate) {
                        int64_t start = 0;
                        if (mode == MS_SIGNAL_FULL) start = -static_cast<int64_t>(sy[d] - 1);
                        else if (mode == MS_SIGNAL_VALID) start = 0;
                        else start = -static_cast<int64_t>(sy[d] / 2);
                        const uint64_t yk = (sy[d] - 1) - k;
                        pos = static_cast<int64_t>(p[d]) + static_cast<int64_t>(k) + start;
                        y_idx += yk * sty[d];
                    } else {
                        int64_t start = 0;
                        if (mode == MS_SIGNAL_FULL) start = -static_cast<int64_t>(sy[d] - 1);
                        else if (mode == MS_SIGNAL_VALID) start = 0;
                        else start = -static_cast<int64_t>(center == MS_SIGNAL_CENTER_2D_CORR ? (sy[d] - 1) / 2 : sy[d] / 2);
                        pos = static_cast<int64_t>(p[d]) + static_cast<int64_t>(k) + start;
                        y_idx += k * sty[d];
                    }
                    const int64_t q = extend_sig(pos, static_cast<int64_t>(sx[d]), boundary);
                    if (q < 0) ok = false;
                    else x_idx += static_cast<uint64_t>(q) * stx[d];
                } else {
                    const uint64_t q_full = crop[d] + p[d];
                    const uint64_t xi = (sx[d] == 1) ? 0 : q_full;
                    const uint64_t yi = (sy[d] == 1) ? 0 : q_full;
                    x_idx += xi * stx[d];
                    y_idx += yi * sty[d];
                }
            }
            const Complex vx = ok ? x[x_idx] : fill;
            const Complex vy = correlate ? std::conj(y[y_idx]) : y[y_idx];
            sum += Complex{vx.real() * vy.real() - vx.imag() * vy.imag(),
                           vx.real() * vy.imag() + vx.imag() * vy.real()};
        }
        out[oi] = sum;
    }

    if (swapped && correlate) {
        std::vector<uint64_t> out_st(ndim, 1);
        for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) out_st[d] = out_st[d + 1] * out_shape[d + 1];
        std::vector<Complex> rev(out_n);
        for (uint64_t oi = 0; oi < out_n; ++oi) {
            uint64_t rem = oi, ri = 0;
            for (size_t d = 0; d < ndim; ++d) {
                const uint64_t coord = rem / out_st[d];
                rem %= out_st[d];
                const uint64_t rcoord = eff_ax[d] ? (out_shape[d] - 1 - coord) : coord;
                ri += rcoord * out_st[d];
            }
            rev[ri] = std::conj(out[oi]);
        }
        out = std::move(rev);
    }
    return out;
}

double max_abs_diff(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    CHECK(a.size() == b.size());
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::abs(a[i] - b[i]);
        if (!(d <= worst)) worst = d;
    }
    return worst;
}

double max_norm(const std::vector<Complex>& a) {
    double m = 0.0;
    for (const Complex& v : a) m = std::max(m, std::abs(v));
    return m;
}

void test_next_fast_len() {
    const fft::Limits full{4096, 1024};
    CHECK(fft::next_fast_len(0, true, true, full) == 0);
    CHECK(fft::next_fast_len(1, true, true, full) == 1);
    CHECK(fft::next_fast_len(17, true, true, full) == 18);     // 2 * 3^2
    CHECK(fft::next_fast_len(1025, true, true, full) == 1080); // 2^3 * 3^3 * 5
    CHECK(fft::next_fast_len(4051, true, true, full) == 4096);
    CHECK(fft::next_fast_len(4097, true, true, full) == 8192); // > group_elements -> bit_ceil
    CHECK(fft::next_fast_len(11171, true, true, full) == 16384);
    CHECK(fft::next_fast_len(11171, false, true, full) == 11250); // complex: {2,3,5}-smooth
    CHECK(fft::next_fast_len(11171, true, false, full) == 11250); // off-last axis: {2,3,5}-smooth

    const fft::Limits reduced{2048, 1024};
    CHECK(fft::next_fast_len(2000, true, true, reduced) == 2000);
    CHECK(fft::next_fast_len(2049, true, true, reduced) == 4096);
}

void test_direct_paths() {
    struct Case {
        std::vector<uint64_t> s1, s2;
        uint32_t d1, d2;
        uint32_t mode, boundary, center;
        bool correlate;
        Complex fill;
    };
    const std::vector<Case> cases = {
        {{9, 11}, {3, 4}, MS_FLOAT64, MS_FLOAT64, MS_SIGNAL_FULL, MS_SIGNAL_BOUNDARY_FILL, MS_SIGNAL_CENTER_ND, false, {0, 0}},
        {{9, 11}, {4, 3}, MS_FLOAT64, MS_FLOAT64, MS_SIGNAL_SAME, MS_SIGNAL_BOUNDARY_FILL, MS_SIGNAL_CENTER_ND, true, {0, 0}},
        {{9, 11}, {4, 4}, MS_FLOAT64, MS_FLOAT64, MS_SIGNAL_SAME, MS_SIGNAL_BOUNDARY_FILL, MS_SIGNAL_CENTER_2D_CORR, true, {0, 0}},
        {{5, 4}, {9, 11}, MS_FLOAT64, MS_FLOAT64, MS_SIGNAL_VALID, MS_SIGNAL_BOUNDARY_FILL, MS_SIGNAL_CENTER_ND, true, {0, 0}},
        {{5, 4}, {9, 11}, MS_FLOAT32, MS_FLOAT64, MS_SIGNAL_FULL, MS_SIGNAL_BOUNDARY_FILL, MS_SIGNAL_CENTER_ND, true, {0, 0}},
        {{7, 8}, {5, 6}, MS_FLOAT64, MS_FLOAT64, MS_SIGNAL_SAME, MS_SIGNAL_BOUNDARY_WRAP, MS_SIGNAL_CENTER_ND, false, {0, 0}},
        {{7, 8}, {11, 10}, MS_FLOAT64, MS_FLOAT64, MS_SIGNAL_FULL, MS_SIGNAL_BOUNDARY_SYMM, MS_SIGNAL_CENTER_ND, false, {0, 0}},
        {{7, 8}, {4, 3}, MS_COMPLEX128, MS_COMPLEX128, MS_SIGNAL_FULL, MS_SIGNAL_BOUNDARY_FILL, MS_SIGNAL_CENTER_ND, true, {0.5, -0.25}},
        {{4, 3}, {7, 8}, MS_COMPLEX64, MS_COMPLEX64, MS_SIGNAL_VALID, MS_SIGNAL_BOUNDARY_FILL, MS_SIGNAL_CENTER_ND, true, {0, 0}},
        {{6, 7}, {3, 4}, MS_FLOAT32, MS_COMPLEX64, MS_SIGNAL_SAME, MS_SIGNAL_BOUNDARY_FILL, MS_SIGNAL_CENTER_ND, false, {0.25, -0.5}},
    };

    for (size_t idx = 0; idx < cases.size(); ++idx) {
        const auto& c = cases[idx];
        auto d1 = make_data(product(c.s1), static_cast<uint32_t>(10 + idx), array::is_complex(c.d1));
        auto d2 = make_data(product(c.s2), static_cast<uint32_t>(100 + idx), array::is_complex(c.d2));
        if (!array::is_double(c.d1)) {
            for (Complex& v : d1) v = {static_cast<float>(v.real()), static_cast<float>(v.imag())};
        }
        if (!array::is_double(c.d2)) {
            for (Complex& v : d2) v = {static_cast<float>(v.real()), static_cast<float>(v.imag())};
        }
        const auto b1 = pack(d1, c.d1);
        const auto b2 = pack(d2, c.d2);
        const ms_array a1 = describe(c.s1, c.d1, b1.data());
        const ms_array a2 = describe(c.s2, c.d2, b2.data());

        ms_signal_conv_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = MS_SIGNAL_ABI_VERSION;
        opts.mode = c.mode;
        opts.method = MS_SIGNAL_METHOD_DIRECT;
        opts.correlate = c.correlate ? 1 : 0;
        opts.boundary = c.boundary;
        opts.center = c.center;
        opts.fillvalue_real = c.fill.real();
        opts.fillvalue_imag = c.fill.imag();

        std::vector<uint64_t> ref_shape;
        const auto ref = ref_conv(d1, c.s1, d2, c.s2, c.mode, c.correlate, c.boundary, c.fill, c.center, {}, ref_shape);

        // CPU strict.
        {
            const ms_route_options r = route_opts(MS_EXECUTION_CPU, MS_PRECISION_STRICT);
            ms_array_result out{};
            ms_route_info info = new_info();
            CHECK(ms_signal_convolve(ctx, &a1, &a2, &opts, &r, &out, &info) == MS_OK);
            CHECK(info.engine == MS_EXECUTION_CPU);
            CHECK(out.ndim == ref_shape.size());
            for (size_t d = 0; d < ref_shape.size(); ++d) CHECK(out.shape[d] == ref_shape[d]);
            const auto got = unpack(out);
            ms_buffer_release(out.buffer);
            const double tol = (array::is_double(out.dtype)) ? 1e-12 : 1e-5;
            const double diff = max_abs_diff(got, ref);
            if (!(diff <= tol * std::max(1.0, max_norm(ref)))) {
                std::fprintf(stderr, "direct cpu idx=%zu diff=%.3e norm=%.3e got0=(%.4f,%.4f) ref0=(%.4f,%.4f)\n",
                             idx, diff, max_norm(ref), got[0].real(), got[0].imag(), ref[0].real(), ref[0].imag());
                std::abort();
            }
        }

        // Forced GPU fast.
        if (have_gpu) {
            const auto before = exec_info();
            const ms_route_options r = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);
            ms_array_result out{};
            ms_route_info info = new_info();
            CHECK(ms_signal_convolve(ctx, &a1, &a2, &opts, &r, &out, &info) == MS_OK);
            const auto after = exec_info();
            CHECK(info.engine == MS_EXECUTION_GPU);
            CHECK(after.gpu_command_buffers == before.gpu_command_buffers + 1);
            CHECK(after.gpu_operations == before.gpu_operations + 1);
            const auto got = unpack(out);
            ms_buffer_release(out.buffer);
            const double m1 = std::max(max_norm(d1), std::abs(c.fill));
            const double m2 = max_norm(d2);
            const double bound = 1e-5 * static_cast<double>(product(c.s2)) * m1 * m2 + 1e-5;
            CHECK(max_abs_diff(got, ref) <= bound);
        }
    }

    // Preserve 0.0 * NaN = NaN (prune_zeros = false).
    {
        std::vector<double> x = {1.0, std::numeric_limits<double>::quiet_NaN(), 2.0, 3.0};
        std::vector<double> w = {1.0, 0.0};
        const ms_array a1 = describe({4}, MS_FLOAT64, x.data());
        const ms_array a2 = describe({2}, MS_FLOAT64, w.data());
        ms_signal_conv_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = MS_SIGNAL_ABI_VERSION;
        opts.mode = MS_SIGNAL_SAME;
        opts.method = MS_SIGNAL_METHOD_DIRECT;
        const ms_route_options r = route_opts(MS_EXECUTION_CPU, MS_PRECISION_STRICT);
        ms_array_result out{};
        CHECK(ms_signal_convolve(ctx, &a1, &a2, &opts, &r, &out, nullptr) == MS_OK);
        const auto got = unpack(out);
        ms_buffer_release(out.buffer);
        CHECK(std::isnan(got[2].real()));
    }

    // Bug B regression: 1-D signal.convolve / correlate (full, and same with s1 >= s2) omit OOB
    // taps instead of computing 0.0 * out_of_bounds_in2, whereas 2-D and 1-D same (s1 < s2)
    // preserve 0.0 * nonfinite = NaN.
    {
        const double inf = std::numeric_limits<double>::infinity();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const float inf_f = std::numeric_limits<float>::infinity();
        const float nan_f = std::numeric_limits<float>::quiet_NaN();
        const ms_route_options r_cpu = route_opts(MS_EXECUTION_CPU, MS_PRECISION_STRICT);
        const ms_route_options r_auto = route_opts(MS_EXECUTION_AUTO, MS_PRECISION_FAST);
        const ms_route_options r_gpu = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);

        // 1-D full unswapped: ones(12) * [1, 1, 1, 1, inf] -> [1, 2, 3, 4, inf, ..., inf] (no NaN).
        std::vector<double> ones12(12, 1.0);
        std::vector<double> k_inf = {1.0, 1.0, 1.0, 1.0, inf};
        const ms_array a_ones12 = describe({12}, MS_FLOAT64, ones12.data());
        const ms_array a_kinf = describe({5}, MS_FLOAT64, k_inf.data());
        for (uint32_t corr : {0u, 1u}) {
            for (uint32_t mode : {MS_SIGNAL_FULL, MS_SIGNAL_SAME}) {
                ms_signal_conv_options opts{};
                opts.struct_size = sizeof(opts);
                opts.abi_version = MS_SIGNAL_ABI_VERSION;
                opts.mode = mode;
                opts.method = MS_SIGNAL_METHOD_DIRECT;
                opts.correlate = corr;
                ms_array_result out{};
                ms_route_info info = new_info();
                CHECK(ms_signal_convolve(ctx, &a_ones12, &a_kinf, &opts, &r_cpu, &out, &info) == MS_OK);
                const auto got = unpack(out);
                ms_buffer_release(out.buffer);
                for (const Complex& v : got) CHECK(!std::isnan(v.real()));
                if (corr == 0u && mode == MS_SIGNAL_FULL) {
                    CHECK(got[0].real() == 1.0 && got[1].real() == 2.0 && got[2].real() == 3.0 && got[3].real() == 4.0);
                    CHECK(std::isinf(got[4].real()) && std::isinf(got[15].real()));
                } else if (corr == 0u && mode == MS_SIGNAL_SAME) {
                    CHECK(got[0].real() == 3.0 && got[1].real() == 4.0 && std::isinf(got[2].real()));
                } else if (corr == 1u && mode == MS_SIGNAL_FULL) {
                    CHECK(std::isinf(got[0].real()) && got[12].real() == 4.0 && got[15].real() == 1.0);
                } else {
                    CHECK(std::isinf(got[0].real()) && got[10].real() == 4.0 && got[11].real() == 3.0);
                }
            }
        }

        // 1-D full swapped (s2 > s1) convolve and correlate with Inf in in2 (float32) & zeros + NaN.
        std::vector<float> ones3_f(3, 1.0f);
        std::vector<float> k5_inf_f = {1.0f, 1.0f, 1.0f, 1.0f, inf_f};
        const ms_array a_o3f = describe({3}, MS_FLOAT32, ones3_f.data());
        const ms_array a_k5f = describe({5}, MS_FLOAT32, k5_inf_f.data());
        for (uint32_t corr : {0u, 1u}) {
            ms_signal_conv_options opts{};
            opts.struct_size = sizeof(opts);
            opts.abi_version = MS_SIGNAL_ABI_VERSION;
            opts.mode = MS_SIGNAL_FULL;
            opts.method = MS_SIGNAL_METHOD_DIRECT;
            opts.correlate = corr;
            ms_array_result out{};
            CHECK(ms_signal_convolve(ctx, &a_o3f, &a_k5f, &opts, &r_cpu, &out, nullptr) == MS_OK);
            const auto got = unpack(out);
            ms_buffer_release(out.buffer);
            CHECK(got.size() == 7);
            for (const Complex& v : got) CHECK(!std::isnan(v.real()));
            if (corr == 0u) {
                CHECK(got[0].real() == 1.0 && got[1].real() == 2.0 && got[2].real() == 3.0 && got[3].real() == 3.0);
                CHECK(std::isinf(got[4].real()) && std::isinf(got[6].real()));
            } else {
                CHECK(std::isinf(got[0].real()) && std::isinf(got[2].real()));
                CHECK(got[3].real() == 3.0 && got[4].real() == 3.0 && got[5].real() == 2.0 && got[6].real() == 1.0);
            }
        }

        // 1-D full zeros(12) * [0, 0, 0, 0, nan] -> 12/16 NaNs (4 leading zeros), complex128 as well.
        std::vector<float> zeros12_f(12, 0.0f);
        std::vector<float> knan_f = {0.0f, 0.0f, 0.0f, 0.0f, nan_f};
        const ms_array a_z12f = describe({12}, MS_FLOAT32, zeros12_f.data());
        const ms_array a_knf = describe({5}, MS_FLOAT32, knan_f.data());
        {
            ms_signal_conv_options opts{};
            opts.struct_size = sizeof(opts);
            opts.abi_version = MS_SIGNAL_ABI_VERSION;
            opts.mode = MS_SIGNAL_FULL;
            opts.method = MS_SIGNAL_METHOD_DIRECT;
            ms_array_result out{};
            CHECK(ms_signal_convolve(ctx, &a_z12f, &a_knf, &opts, &r_cpu, &out, nullptr) == MS_OK);
            const auto got = unpack(out);
            ms_buffer_release(out.buffer);
            CHECK(got.size() == 16);
            for (size_t i = 0; i < 4; ++i) CHECK(got[i].real() == 0.0);
            for (size_t i = 4; i < 16; ++i) CHECK(std::isnan(got[i].real()));
        }

        // Complex 1-D full omit_oob.
        std::vector<double> c_ones8(8 * 2, 0.0);
        for (size_t i = 0; i < 8; ++i) c_ones8[2 * i] = 1.0;
        std::vector<double> c_kinf = {1.0, 0.0, 1.0, 0.0, inf, 0.0};
        const ms_array ac1 = describe({8}, MS_COMPLEX128, c_ones8.data());
        const ms_array ac2 = describe({3}, MS_COMPLEX128, c_kinf.data());
        {
            ms_signal_conv_options opts{};
            opts.struct_size = sizeof(opts);
            opts.abi_version = MS_SIGNAL_ABI_VERSION;
            opts.mode = MS_SIGNAL_FULL;
            opts.method = MS_SIGNAL_METHOD_DIRECT;
            opts.correlate = 1;
            ms_array_result out{};
            CHECK(ms_signal_convolve(ctx, &ac1, &ac2, &opts, &r_cpu, &out, nullptr) == MS_OK);
            const auto got = unpack(out);
            ms_buffer_release(out.buffer);
            CHECK(got.size() == 10);
            CHECK(!std::isnan(got[8].real()) && got[8].real() == 2.0 && got[9].real() == 1.0);
        }

        // 2-D convolve2d (fillvalue=0.0) still preserves 0.0 * inf = NaN.
        std::vector<double> ones44(16, 1.0);
        std::vector<double> k12_inf = {1.0, inf};
        const ms_array a2_ones = describe({4, 4}, MS_FLOAT64, ones44.data());
        const ms_array a2_kinf = describe({1, 2}, MS_FLOAT64, k12_inf.data());
        {
            ms_signal_conv_options opts{};
            opts.struct_size = sizeof(opts);
            opts.abi_version = MS_SIGNAL_ABI_VERSION;
            opts.mode = MS_SIGNAL_FULL;
            opts.method = MS_SIGNAL_METHOD_DIRECT;
            opts.boundary = MS_SIGNAL_BOUNDARY_FILL;
            ms_array_result out{};
            CHECK(ms_signal_convolve(ctx, &a2_ones, &a2_kinf, &opts, &r_cpu, &out, nullptr) == MS_OK);
            const auto got = unpack(out);
            ms_buffer_release(out.buffer);
            CHECK(std::isnan(got[0].real()) && std::isinf(got[1].real()));
        }

        // Large 1-D with non-finite tap: AUTO re-routes to CPU (no NaN), forced GPU declines MS_ENOTSUP.
        if (have_gpu) {
            std::vector<float> big1d(200000, 1.0f);
            std::vector<float> k1000_inf(1000, 1.0f);
            k1000_inf[999] = inf_f;
            const ms_array a_big1d = describe({200000}, MS_FLOAT32, big1d.data());
            const ms_array a_k1000 = describe({1000}, MS_FLOAT32, k1000_inf.data());
            ms_signal_conv_options opts{};
            opts.struct_size = sizeof(opts);
            opts.abi_version = MS_SIGNAL_ABI_VERSION;
            opts.mode = MS_SIGNAL_FULL;
            opts.method = MS_SIGNAL_METHOD_DIRECT;
            ms_array_result out{};
            ms_route_info info = new_info();
            CHECK(ms_signal_convolve(ctx, &a_big1d, &a_k1000, &opts, &r_gpu, &out, &info) == MS_ENOTSUP);
            CHECK(out.buffer == nullptr);
            CHECK(ms_signal_convolve(ctx, &a_big1d, &a_k1000, &opts, &r_auto, &out, &info) == MS_OK);
            CHECK(info.engine == MS_EXECUTION_CPU && std::strlen(info.reason) > 0);
            const auto* p = reinterpret_cast<const float*>(read(out.buffer, ctx).data);
            CHECK(p[0] == 1.0f && p[1] == 2.0f && p[998] == 999.0f && std::isinf(p[999]));
            ms_buffer_release(out.buffer);
        }
    }

    // A5 overflow guard rail on direct path.
    if (have_gpu) {
        std::vector<double> x(64, 1.0);
        x[10] = 1e39;
        std::vector<double> w(9, 1.0);
        const ms_array a1 = describe({8, 8}, MS_FLOAT64, x.data());
        const ms_array a2 = describe({3, 3}, MS_FLOAT64, w.data());
        ms_signal_conv_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = MS_SIGNAL_ABI_VERSION;
        opts.mode = MS_SIGNAL_SAME;
        opts.method = MS_SIGNAL_METHOD_DIRECT;

        ms_array_result out{};
        ms_route_info info = new_info();
        const ms_route_options r_gpu = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);
        CHECK(ms_signal_convolve(ctx, &a1, &a2, &opts, &r_gpu, &out, &info) == MS_ENOTSUP);
        CHECK(out.buffer == nullptr);

        // Even under AUTO with large work, overflow re-routes to CPU.
        std::vector<double> big(2048 * 2048, 1.0);
        big[0] = 1e39;
        std::vector<double> kw(7 * 7, 1.0);
        const ms_array abig = describe({2048, 2048}, MS_FLOAT64, big.data());
        const ms_array akw = describe({7, 7}, MS_FLOAT64, kw.data());
        const ms_route_options r_auto = route_opts(MS_EXECUTION_AUTO, MS_PRECISION_FAST);
        CHECK(ms_signal_convolve(ctx, &abig, &akw, &opts, &r_auto, &out, &info) == MS_OK);
        CHECK(info.engine == MS_EXECUTION_CPU);
        CHECK(std::strlen(info.reason) > 0);
        const auto* res_ptr = reinterpret_cast<const double*>(read(out.buffer, ctx).data);
        CHECK(std::isfinite(res_ptr[0]) && res_ptr[0] > 1e38);
        ms_buffer_release(out.buffer);

        // Bug 1 regression: mixed float32 input + float64 kernel + large fillvalue.
        {
            std::vector<float> z_f32(2048 * 2048, 0.0f);
            std::vector<double> k_f64(7 * 7, 1e5);
            const ms_array az = describe({2048, 2048}, MS_FLOAT32, z_f32.data());
            const ms_array ak = describe({7, 7}, MS_FLOAT64, k_f64.data());
            ms_signal_conv_options fopts = opts;
            fopts.boundary = MS_SIGNAL_BOUNDARY_FILL;
            fopts.fillvalue_real = 1e38;
            ms_array_result fout{};
            ms_route_info finfo = new_info();
            CHECK(ms_signal_convolve(ctx, &az, &ak, &fopts, &r_gpu, &fout, &finfo) == MS_ENOTSUP);
            CHECK(fout.buffer == nullptr);
            CHECK(ms_signal_convolve(ctx, &az, &ak, &fopts, &r_auto, &fout, &finfo) == MS_OK);
            CHECK(finfo.engine == MS_EXECUTION_CPU && finfo.compute_dtype == MS_FLOAT64);
            CHECK(std::strlen(finfo.reason) > 0);
            const auto* fp = reinterpret_cast<const double*>(read(fout.buffer, ctx).data);
            CHECK(std::isfinite(fp[0]) && fp[0] > 1e43);
            ms_buffer_release(fout.buffer);
        }

        // Bug 2 regression: real_in_complex_out + zero complex128 kernel + fillvalue=1e39 (no NaN).
        {
            std::vector<double> ones_f64(8 * 8, 1.0);
            std::vector<double> zeros_c128(3 * 3 * 2, 0.0);
            const ms_array a_ones = describe({8, 8}, MS_FLOAT64, ones_f64.data());
            const ms_array a_zker = describe({3, 3}, MS_COMPLEX128, zeros_c128.data());
            ms_signal_conv_options copts = opts;
            copts.boundary = MS_SIGNAL_BOUNDARY_FILL;
            copts.fillvalue_real = 1e39;
            ms_array_result cout{};
            ms_route_info cinfo = new_info();
            CHECK(ms_signal_convolve(ctx, &a_ones, &a_zker, &copts, &r_gpu, &cout, &cinfo) == MS_ENOTSUP);
            CHECK(cout.buffer == nullptr);

            std::vector<double> big_ones(2048 * 2048, 1.0);
            const ms_array a_big = describe({2048, 2048}, MS_FLOAT64, big_ones.data());
            CHECK(ms_signal_convolve(ctx, &a_big, &a_zker, &copts, &r_auto, &cout, &cinfo) == MS_OK);
            CHECK(cinfo.engine == MS_EXECUTION_CPU && cinfo.compute_dtype == MS_COMPLEX128);
            const auto* cp = reinterpret_cast<const double*>(read(cout.buffer, ctx).data);
            CHECK(cp[0] == 0.0 && cp[1] == 0.0);
            ms_buffer_release(cout.buffer);
        }
    }
}

void test_mixed_real_complex_nan_semantics() {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    auto ieee_cmul = [](Complex a, Complex b) -> Complex {
        return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
    };
    auto check_pair = [&](const ms_array& a1, const ms_array& a2, const std::vector<Complex>& d1,
                          const std::vector<uint64_t>& s1, const std::vector<Complex>& d2,
                          const std::vector<uint64_t>& s2, uint32_t mode, bool corr, uint32_t center,
                          const ms_route_options& r, uint32_t expected_engine) {
        ms_signal_conv_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = MS_SIGNAL_ABI_VERSION;
        opts.mode = mode;
        opts.method = MS_SIGNAL_METHOD_DIRECT;
        opts.correlate = corr ? 1u : 0u;
        opts.boundary = MS_SIGNAL_BOUNDARY_FILL;
        opts.center = center;

        ms_array_result out{};
        ms_route_info info = new_info();
        CHECK(ms_signal_convolve(ctx, &a1, &a2, &opts, &r, &out, &info) == MS_OK);
        if (expected_engine != UINT32_MAX) CHECK(info.engine == expected_engine);
        const auto got = unpack(out);
        ms_buffer_release(out.buffer);

        // Compute exact IEEE-754 componentwise reference (without C99 __muldc3 Inf recovery).
        const size_t ndim = s1.size();
        const bool swapped = (mode == MS_SIGNAL_VALID && s1[0] < s2[0]);
        const auto& x = swapped ? d2 : d1;
        const auto& y = swapped ? d1 : d2;
        const auto& sx = swapped ? s2 : s1;
        const auto& sy = swapped ? s1 : s2;
        const bool omit_oob = (ndim == 1) && (center == MS_SIGNAL_CENTER_ND) &&
                              (mode == MS_SIGNAL_FULL || (mode == MS_SIGNAL_SAME && s1[0] >= s2[0]));

        std::vector<uint64_t> out_sh(ndim);
        for (size_t d = 0; d < ndim; ++d) {
            if (mode == MS_SIGNAL_FULL) out_sh[d] = sx[d] + sy[d] - 1;
            else if (mode == MS_SIGNAL_SAME) out_sh[d] = sx[d];
            else out_sh[d] = sx[d] - sy[d] + 1;
        }
        const uint64_t out_n = product(out_sh);
        CHECK(got.size() == out_n);

        // Spot-check first 16 and last 16 elements (all elements for small arrays) to keep 150k fast.
        const uint64_t head_n = std::min<uint64_t>(out_n, 16);
        std::vector<uint64_t> check_indices;
        for (uint64_t i = 0; i < head_n; ++i) check_indices.push_back(i);
        for (uint64_t i = std::max<uint64_t>(head_n, out_n > 16 ? out_n - 16 : 0); i < out_n; ++i)
            check_indices.push_back(i);

        for (uint64_t oi : check_indices) {
            Complex acc{0.0, 0.0};
            if (ndim == 1) {
                const int64_t p0 = static_cast<int64_t>(oi);
                int64_t start = 0;
                if (mode == MS_SIGNAL_FULL) start = -static_cast<int64_t>(sy[0] - 1);
                else if (mode == MS_SIGNAL_VALID) start = 0;
                else start = -static_cast<int64_t>(corr && center == MS_SIGNAL_CENTER_2D_CORR ? (sy[0] - 1) / 2 : sy[0] / 2);
                for (uint64_t k = 0; k < sy[0]; ++k) {
                    const int64_t pos = p0 + static_cast<int64_t>(k) + start;
                    const bool ok = (pos >= 0 && pos < static_cast<int64_t>(sx[0]));
                    if (!ok && omit_oob) continue;
                    const Complex vx = ok ? x[static_cast<size_t>(pos)] : Complex{0.0, 0.0};
                    const Complex raw_y = y[static_cast<size_t>(corr ? k : (sy[0] - 1 - k))];
                    const Complex vy = corr ? std::conj(raw_y) : raw_y;
                    acc += ieee_cmul(vx, vy);
                }
            } else {
                const uint64_t py = oi / out_sh[1], px = oi % out_sh[1];
                int64_t st0 = 0, st1 = 0;
                if (mode == MS_SIGNAL_FULL) {
                    st0 = -static_cast<int64_t>(sy[0] - 1);
                    st1 = -static_cast<int64_t>(sy[1] - 1);
                } else if (mode == MS_SIGNAL_SAME) {
                    st0 = -static_cast<int64_t>(corr && center == MS_SIGNAL_CENTER_2D_CORR ? (sy[0] - 1) / 2 : sy[0] / 2);
                    st1 = -static_cast<int64_t>(corr && center == MS_SIGNAL_CENTER_2D_CORR ? (sy[1] - 1) / 2 : sy[1] / 2);
                }
                for (uint64_t ky = 0; ky < sy[0]; ++ky) {
                    for (uint64_t kx = 0; kx < sy[1]; ++kx) {
                        const int64_t qy = static_cast<int64_t>(py) + static_cast<int64_t>(ky) + st0;
                        const int64_t qx = static_cast<int64_t>(px) + static_cast<int64_t>(kx) + st1;
                        const bool ok = (qy >= 0 && qy < static_cast<int64_t>(sx[0]) &&
                                         qx >= 0 && qx < static_cast<int64_t>(sx[1]));
                        const Complex vx = ok ? x[static_cast<size_t>(qy) * sx[1] + static_cast<size_t>(qx)]
                                              : Complex{0.0, 0.0};
                        const uint64_t yky = corr ? ky : (sy[0] - 1 - ky);
                        const uint64_t ykx = corr ? kx : (sy[1] - 1 - kx);
                        const Complex raw_y = y[static_cast<size_t>(yky) * sy[1] + static_cast<size_t>(ykx)];
                        const Complex vy = corr ? std::conj(raw_y) : raw_y;
                        acc += ieee_cmul(vx, vy);
                    }
                }
            }
            const Complex ref_v = (swapped && corr) ? std::conj(acc) : acc;
            const uint64_t got_oi = (swapped && corr) ? (out_n - 1 - oi) : oi;
            const Complex g = got[static_cast<size_t>(got_oi)];
            CHECK(std::isnan(g.real()) == std::isnan(ref_v.real()));
            CHECK(std::isnan(g.imag()) == std::isnan(ref_v.imag()));
            CHECK(std::isinf(g.real()) == std::isinf(ref_v.real()));
            CHECK(std::isinf(g.imag()) == std::isinf(ref_v.imag()));
            if (std::isfinite(ref_v.real())) CHECK(std::fabs(g.real() - ref_v.real()) <= 1e-4);
            if (std::isfinite(ref_v.imag())) CHECK(std::fabs(g.imag() - ref_v.imag()) <= 1e-4);
        }
    };

    const std::vector<std::pair<uint32_t, uint32_t>> dtype_pairs = {
        {MS_FLOAT32, MS_COMPLEX128},
        {MS_FLOAT64, MS_COMPLEX64},
        {MS_FLOAT32, MS_COMPLEX64},
    };
    const std::vector<Complex> nf_complex = {
        {0.0, nan},
        {inf, 0.0},
        {1.0, inf},
        {inf, inf},
    };
    const std::vector<Complex> ck_kernels = {
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0},
    };

    struct RouteCfg {
        ms_route_options opts;
        uint64_t big_n;
        uint32_t expected_engine;
    };
    std::vector<RouteCfg> routes = {
        {route_opts(MS_EXECUTION_CPU, MS_PRECISION_STRICT), 12, MS_EXECUTION_CPU},
    };
    if (have_gpu) {
        routes.push_back({route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST), 150000, MS_EXECUTION_GPU});
        routes.push_back({route_opts(MS_EXECUTION_AUTO, MS_PRECISION_FAST), 150000, UINT32_MAX});
    }

    for (const auto& rc : routes) {
        for (const auto& [dt_r, dt_c] : dtype_pairs) {
            // 1. Complex input + real kernel (both swapped real+complex and unswapped complex+real).
            for (const Complex& nf : nf_complex) {
                std::vector<Complex> dr(5, {1.0, 0.0});
                std::vector<Complex> dc(rc.big_n, {1.0, 0.0});
                dc[0] = nf;
                const auto br = pack(dr, dt_r);
                const auto bc = pack(dc, dt_c);
                const ms_array ar = describe({5}, dt_r, br.data());
                const ms_array ac = describe({rc.big_n}, dt_c, bc.data());
                for (uint32_t mode : {MS_SIGNAL_FULL, MS_SIGNAL_SAME, MS_SIGNAL_VALID}) {
                    for (bool corr : {false, true}) {
                        check_pair(ar, ac, dr, {5}, dc, {rc.big_n}, mode, corr, MS_SIGNAL_CENTER_ND, rc.opts,
                                   rc.expected_engine);
                        check_pair(ac, ar, dc, {rc.big_n}, dr, {5}, mode, corr, MS_SIGNAL_CENTER_ND, rc.opts,
                                   rc.expected_engine);
                    }
                }
            }

            // 2. Real Inf input + complex kernel (0+1j, 1+0j, 1+1j).
            for (const Complex& ck : ck_kernels) {
                std::vector<Complex> dr(rc.big_n, {1.0, 0.0});
                dr[0] = {inf, 0.0};
                std::vector<Complex> dc(5, {1.0, 0.0});
                dc[0] = ck;
                const auto br = pack(dr, dt_r);
                const auto bc = pack(dc, dt_c);
                const ms_array ar = describe({rc.big_n}, dt_r, br.data());
                const ms_array ac = describe({5}, dt_c, bc.data());
                for (uint32_t mode : {MS_SIGNAL_FULL, MS_SIGNAL_SAME, MS_SIGNAL_VALID}) {
                    for (bool corr : {false, true}) {
                        check_pair(ar, ac, dr, {rc.big_n}, dc, {5}, mode, corr, MS_SIGNAL_CENTER_ND, rc.opts,
                                   rc.expected_engine);
                    }
                }
            }
        }

        // 3. 2-D convolve2d / correlate2d center semantics with mixed complex64 + float32.
        const uint64_t side = (rc.expected_engine == MS_EXECUTION_GPU) ? 64 : 8;
        std::vector<Complex> d2c(side * side, {1.0, 0.0});
        d2c[0] = {inf, 0.0};
        d2c[1] = {1.0, inf};
        std::vector<Complex> d2r(9, {1.0, 0.0});
        const auto b2c = pack(d2c, MS_COMPLEX64);
        const auto b2r = pack(d2r, MS_FLOAT32);
        const ms_array a2c = describe({side, side}, MS_COMPLEX64, b2c.data());
        const ms_array a2r = describe({3, 3}, MS_FLOAT32, b2r.data());
        for (uint32_t mode : {MS_SIGNAL_FULL, MS_SIGNAL_SAME, MS_SIGNAL_VALID}) {
            for (bool corr : {false, true}) {
                check_pair(a2c, a2r, d2c, {side, side}, d2r, {3, 3}, mode, corr, MS_SIGNAL_CENTER_2D_CORR, rc.opts,
                           rc.expected_engine);
            }
        }
    }
}

void test_fft_and_oa_gpu() {
    if (!have_gpu) return;

    struct FreqCase {
        std::vector<uint64_t> s1, s2;
        uint32_t d1, d2;
        uint32_t mode;
        bool correlate;
        bool oa;
        std::vector<uint32_t> axes;
    };
    const std::vector<FreqCase> cases = {
        {{64}, {17}, MS_FLOAT64, MS_FLOAT64, MS_SIGNAL_FULL, false, false, {}},
        {{200}, {13}, MS_FLOAT64, MS_FLOAT64, MS_SIGNAL_SAME, false, true, {}},
        {{13}, {200}, MS_FLOAT64, MS_FLOAT64, MS_SIGNAL_VALID, true, true, {}},
        {{21, 35}, {12, 9}, MS_FLOAT32, MS_FLOAT32, MS_SIGNAL_SAME, false, false, {}},
        {{17, 23}, {9, 14}, MS_COMPLEX128, MS_COMPLEX128, MS_SIGNAL_FULL, true, false, {}},
        // Off-last axis and broadcasting on non-transformed axes.
        {{3, 1, 20}, {1, 4, 7}, MS_FLOAT64, MS_FLOAT64, MS_SIGNAL_FULL, false, false, {2}},
        {{25, 4, 3}, {7, 1, 3}, MS_FLOAT64, MS_FLOAT64, MS_SIGNAL_VALID, false, true, {0}},
        {{2, 19, 3, 25}, {2, 5, 1, 8}, MS_COMPLEX64, MS_COMPLEX64, MS_SIGNAL_FULL, false, true, {1, 3}},
        // C ABI correlate = 1 with subset axes (unswapped FULL and swapped VALID OA).
        {{3, 20}, {3, 7}, MS_COMPLEX128, MS_COMPLEX128, MS_SIGNAL_FULL, true, false, {1}},
        {{3, 12}, {3, 200}, MS_COMPLEX64, MS_COMPLEX64, MS_SIGNAL_VALID, true, true, {1}},
        {{2, 17, 3, 21}, {2, 5, 1, 9}, MS_FLOAT64, MS_FLOAT64, MS_SIGNAL_SAME, true, false, {1, 3}},
        // Larger 1-D OA > group_elements (4096).
        {{25000}, {301}, MS_FLOAT64, MS_FLOAT64, MS_SIGNAL_SAME, false, true, {}},
    };

    for (size_t idx = 0; idx < cases.size(); ++idx) {
        const auto& c = cases[idx];
        const auto d1 = make_data(product(c.s1), static_cast<uint32_t>(500 + idx), array::is_complex(c.d1));
        const auto d2 = make_data(product(c.s2), static_cast<uint32_t>(700 + idx), array::is_complex(c.d2));
        const auto b1 = pack(d1, c.d1);
        const auto b2 = pack(d2, c.d2);
        const ms_array a1 = describe(c.s1, c.d1, b1.data());
        const ms_array a2 = describe(c.s2, c.d2, b2.data());

        ms_signal_fftconv_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = MS_SIGNAL_ABI_VERSION;
        opts.mode = c.mode;
        opts.correlate = c.correlate ? 1 : 0;
        opts.overlap_add = c.oa ? 1 : 0;
        opts.axis_count = static_cast<uint32_t>(c.axes.size());
        for (size_t i = 0; i < c.axes.size(); ++i) opts.axes[i] = c.axes[i];

        std::vector<uint64_t> ref_shape;
        const auto ref = ref_conv(d1, c.s1, d2, c.s2, c.mode, c.correlate, MS_SIGNAL_BOUNDARY_FILL,
                                  {0, 0}, MS_SIGNAL_CENTER_ND, c.axes, ref_shape);

        const auto before = exec_info();
        const ms_route_options r = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);
        ms_array_result out{};
        ms_route_info info = new_info();
        CHECK(ms_signal_fftconvolve(ctx, &a1, &a2, &opts, &r, &out, &info) == MS_OK);
        const auto after = exec_info();
        CHECK(info.engine == MS_EXECUTION_GPU);
        CHECK(after.gpu_command_buffers == before.gpu_command_buffers + 1);
        CHECK(after.gpu_operations == before.gpu_operations + 1);
        CHECK(out.ndim == ref_shape.size());
        for (size_t d = 0; d < ref_shape.size(); ++d) CHECK(out.shape[d] == ref_shape[d]);
        const auto got = unpack(out);
        ms_buffer_release(out.buffer);
        const double worst = max_abs_diff(got, ref);
        const double scale = std::max(1.0, max_norm(ref));
        if (!(worst <= 2e-5 * scale)) {
            std::fprintf(stderr, "freq case %zu failed: worst=%.3e scale=%.3e\n", idx, worst, scale);
            std::abort();
        }
    }

    // Refusals: strict + double, cpu device, below auto crossover, A5 magnitude/product overflow.
    {
        const auto d1 = make_data(64, 1, false);
        const auto d2 = make_data(17, 2, false);
        const auto b1 = pack(d1, MS_FLOAT64);
        const auto b2 = pack(d2, MS_FLOAT64);
        const ms_array a1 = describe({64}, MS_FLOAT64, b1.data());
        const ms_array a2 = describe({17}, MS_FLOAT64, b2.data());
        ms_signal_fftconv_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = MS_SIGNAL_ABI_VERSION;
        opts.mode = MS_SIGNAL_FULL;
        ms_array_result out{};

        auto r = route_opts(MS_EXECUTION_GPU, MS_PRECISION_STRICT);
        CHECK(ms_signal_fftconvolve(ctx, &a1, &a2, &opts, &r, &out, nullptr) == MS_ENOTSUP);
        r = route_opts(MS_EXECUTION_CPU, MS_PRECISION_FAST);
        CHECK(ms_signal_fftconvolve(ctx, &a1, &a2, &opts, &r, &out, nullptr) == MS_ENOTSUP);
        r = route_opts(MS_EXECUTION_AUTO, MS_PRECISION_FAST);
        CHECK(ms_signal_fftconvolve(ctx, &a1, &a2, &opts, &r, &out, nullptr) == MS_ENOTSUP);

        // Magnitude > 2^96
        std::vector<double> huge(64, 1e30);
        const ms_array ahuge = describe({64}, MS_FLOAT64, huge.data());
        r = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);
        CHECK(ms_signal_fftconvolve(ctx, &ahuge, &a2, &opts, &r, &out, nullptr) == MS_ENOTSUP);

        // Magnitude < 2^-100
        std::vector<double> tiny(64, 1e-35);
        const ms_array atiny = describe({64}, MS_FLOAT64, tiny.data());
        CHECK(ms_signal_fftconvolve(ctx, &atiny, &a2, &opts, &r, &out, nullptr) == MS_ENOTSUP);

        // Product overflow: 1e20 * 1e20 * 17 > 2^124 (while each <= 2^96).
        std::vector<double> p1(64, 1e20);
        std::vector<double> p2(17, 1e20);
        const ms_array ap1 = describe({64}, MS_FLOAT64, p1.data());
        const ms_array ap2 = describe({17}, MS_FLOAT64, p2.data());
        CHECK(ms_signal_fftconvolve(ctx, &ap1, &ap2, &opts, &r, &out, nullptr) == MS_ENOTSUP);
    }

    // Buffer reuse across repeated calls.
    if (!validated) {
        const auto d1 = make_data(40000, 7, false);
        const auto d2 = make_data(500, 8, false);
        const auto b1 = pack(d1, MS_FLOAT64);
        const auto b2 = pack(d2, MS_FLOAT64);
        const ms_array a1 = describe({40000}, MS_FLOAT64, b1.data());
        const ms_array a2 = describe({500}, MS_FLOAT64, b2.data());
        ms_signal_fftconv_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = MS_SIGNAL_ABI_VERSION;
        opts.mode = MS_SIGNAL_SAME;
        opts.overlap_add = 1;
        const auto r = route_opts(MS_EXECUTION_AUTO, MS_PRECISION_FAST);
        ms_array_result out1{}, out2{};
        CHECK(ms_signal_fftconvolve(ctx, &a1, &a2, &opts, &r, &out1, nullptr) == MS_OK);
        ms_buffer_release(out1.buffer);
        const auto before = exec_info();
        CHECK(ms_signal_fftconvolve(ctx, &a1, &a2, &opts, &r, &out2, nullptr) == MS_OK);
        const auto after = exec_info();
        ms_buffer_release(out2.buffer);
        CHECK(after.cached_reuses > before.cached_reuses);
    }
}

void test_2d_full_no_swap_and_overhang() {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    struct ShapeCase {
        uint64_t h1, w1, h2, w2;
        uint32_t d1, d2;
        Complex fill;
        bool nan_in_in1;
        bool use_inf;
    };
    const std::vector<ShapeCase> cases = {
        // s2 > s1 overhanging on both dims
        {3, 3, 8, 8, MS_FLOAT32, MS_FLOAT32, {0.0, 0.0}, true, false},
        {3, 3, 8, 8, MS_FLOAT32, MS_FLOAT32, {0.0, 0.0}, false, false},
        {3, 3, 8, 8, MS_FLOAT64, MS_FLOAT64, {0.0, 0.0}, true, true},
        // Asymmetric (3, 10) x (10, 3) (equal elements, overhanging dim 0) and (2, 12) x (9, 3) (s2 > s1)
        {3, 10, 10, 3, MS_FLOAT32, MS_FLOAT32, {0.0, 0.0}, true, false},
        {2, 12, 9, 3, MS_FLOAT32, MS_FLOAT32, {2.5, 0.0}, false, false},
        // Complex and mixed real+complex with non-zero complex fillvalue
        {3, 3, 8, 8, MS_COMPLEX64, MS_COMPLEX64, {0.0, 0.0}, true, false},
        {3, 3, 8, 8, MS_COMPLEX128, MS_COMPLEX128, {1.5, -0.5}, false, false},
        {3, 10, 10, 3, MS_FLOAT32, MS_COMPLEX64, {0.0, 0.0}, true, true},
    };

    std::vector<ms_route_options> routes = {route_opts(MS_EXECUTION_CPU, MS_PRECISION_STRICT)};
    if (have_gpu) {
        routes.push_back(route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST));
        routes.push_back(route_opts(MS_EXECUTION_AUTO, MS_PRECISION_FAST));
    }

    auto run_case = [&](const ShapeCase& c, const ms_route_options& r, uint32_t center, bool corr) {
        const bool c1 = array::is_complex(c.d1);
        const bool c2 = array::is_complex(c.d2);
        std::vector<Complex> d1(c.h1 * c.w1, Complex{1.0, c1 ? 0.25 : 0.0});
        std::vector<Complex> d2(c.h2 * c.w2, Complex{1.0, c2 ? -0.5 : 0.0});
        const Complex nf = c.use_inf ? Complex{inf, 0.0} : Complex{nan, 0.0};
        if (c.nan_in_in1) {
            d1[(c.h1 / 2) * c.w1 + (c.w1 / 2)] = nf;
        } else {
            d2[(c.h2 / 2) * c.w2 + (c.w2 / 2)] = nf;
        }
        const auto b1 = pack(d1, c.d1);
        const auto b2 = pack(d2, c.d2);
        const ms_array a1 = describe({c.h1, c.w1}, c.d1, b1.data());
        const ms_array a2 = describe({c.h2, c.w2}, c.d2, b2.data());

        ms_signal_conv_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = MS_SIGNAL_ABI_VERSION;
        opts.mode = MS_SIGNAL_FULL;
        opts.method = MS_SIGNAL_METHOD_DIRECT;
        opts.correlate = corr ? 1u : 0u;
        opts.boundary = MS_SIGNAL_BOUNDARY_FILL;
        opts.center = center;
        opts.fillvalue_real = c.fill.real();
        opts.fillvalue_imag = c.fill.imag();

        ms_array_result out{};
        ms_route_info info = new_info();
        CHECK(ms_signal_convolve(ctx, &a1, &a2, &opts, &r, &out, &info) == MS_OK);
        const auto got = unpack(out);
        ms_buffer_release(out.buffer);

        // Reference: unswapped for CENTER_2D_CORR (or custom fillvalue), swapped for CENTER_ND when s2 > s1 & fill==0.
        const bool custom_fill = (c.fill.real() != 0.0 || c.fill.imag() != 0.0);
        const bool expect_swap = (center == MS_SIGNAL_CENTER_ND) && !custom_fill && (c.h2 * c.w2 > c.h1 * c.w1);
        std::vector<uint64_t> ref_shape;
        std::vector<Complex> ref;
        if (expect_swap) {
            ref = ref_conv(d2, {c.h2, c.w2}, d1, {c.h1, c.w1}, MS_SIGNAL_FULL, corr,
                           MS_SIGNAL_BOUNDARY_FILL, c.fill, center, {}, ref_shape);
            if (corr) {
                std::reverse(ref.begin(), ref.end());
                for (Complex& v : ref) v = std::conj(v);
            }
        } else {
            ref = ref_conv(d1, {c.h1, c.w1}, d2, {c.h2, c.w2}, MS_SIGNAL_FULL, corr,
                           MS_SIGNAL_BOUNDARY_FILL, c.fill, center, {}, ref_shape);
        }
        const bool out_complex = c1 || c2;
        CHECK(got.size() == ref.size());
        for (size_t i = 0; i < got.size(); ++i) {
            CHECK(std::isnan(got[i].real()) == std::isnan(ref[i].real()));
            CHECK(std::isinf(got[i].real()) == std::isinf(ref[i].real()));
            if (std::isfinite(ref[i].real())) CHECK(std::fabs(got[i].real() - ref[i].real()) <= 1e-4);
            if (out_complex) {
                CHECK(std::isnan(got[i].imag()) == std::isnan(ref[i].imag()));
                CHECK(std::isinf(got[i].imag()) == std::isinf(ref[i].imag()));
                if (std::isfinite(ref[i].imag())) CHECK(std::fabs(got[i].imag() - ref[i].imag()) <= 1e-4);
            } else {
                CHECK(got[i].imag() == 0.0);
            }
        }
    };

    for (const auto& r : routes) {
        for (const auto& c : cases) {
            for (bool corr : {false, true}) {
                run_case(c, r, MS_SIGNAL_CENTER_2D_CORR, corr);
            }
        }
        // Also verify MS_SIGNAL_CENTER_ND keeps swapping when s2 > s1 and fill == 0.
        run_case({3, 3, 8, 8, MS_FLOAT32, MS_FLOAT32, {0.0, 0.0}, true, false}, r, MS_SIGNAL_CENTER_ND, false);
        run_case({3, 3, 8, 8, MS_FLOAT32, MS_FLOAT32, {0.0, 0.0}, false, false}, r, MS_SIGNAL_CENTER_ND, true);
    }

    // Large (3, 3) x (70, 70) directly on GPU with NaN in in1 vs in2.
    if (have_gpu) {
        const auto r_gpu = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);
        for (bool corr : {false, true}) {
            run_case({3, 3, 70, 70, MS_FLOAT32, MS_FLOAT32, {0.0, 0.0}, true, false}, r_gpu, MS_SIGNAL_CENTER_2D_CORR, corr);
            run_case({3, 3, 70, 70, MS_FLOAT32, MS_FLOAT32, {0.0, 0.0}, false, false}, r_gpu, MS_SIGNAL_CENTER_2D_CORR, corr);
            run_case({3, 3, 70, 70, MS_FLOAT32, MS_FLOAT32, {0.0, 0.0}, true, true}, r_gpu, MS_SIGNAL_CENTER_2D_CORR, corr);
            run_case({3, 3, 70, 70, MS_COMPLEX64, MS_COMPLEX64, {0.0, 0.0}, true, false}, r_gpu, MS_SIGNAL_CENTER_2D_CORR, corr);
        }
    }
}

void test_method_fft_disables_oa_nonfinite() {
    if (!have_gpu) return;
    const uint64_t n1 = 40000, n2 = 100;
    std::vector<float> x(n1, 1.0f);
    x[30000] = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> k(n2, 1.0f);
    const ms_array ax = describe({n1}, MS_FLOAT32, x.data());
    const ms_array ak = describe({n2}, MS_FLOAT32, k.data());
    const ms_route_options r_gpu = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);

    // MS_SIGNAL_METHOD_FFT uses single-FFT (allow_oa=false), propagating NaN across entire output.
    for (uint32_t corr : {0u, 1u}) {
        ms_signal_conv_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = MS_SIGNAL_ABI_VERSION;
        opts.mode = MS_SIGNAL_FULL;
        opts.method = MS_SIGNAL_METHOD_FFT;
        opts.correlate = corr;
        ms_array_result out{};
        ms_route_info info = new_info();
        CHECK(ms_signal_convolve(ctx, &ax, &ak, &opts, &r_gpu, &out, &info) == MS_OK);
        CHECK(info.engine == MS_EXECUTION_GPU);
        CHECK(info.reserved[0] == (MS_SIGNAL_ROUTE_FFT | MS_SIGNAL_ROUTE_NONFINITE));
        const auto* p = reinterpret_cast<const float*>(read(out.buffer, ctx).data);
        for (uint64_t i = 0; i < n1 + n2 - 1; ++i) CHECK(std::isnan(p[i]));
        ms_buffer_release(out.buffer);
    }

    // MS_SIGNAL_METHOD_AUTO routing to FFT on non-finite inputs (1-D OA shapes & 2-D complex64)
    // falls back to single-FFT for full-output NaN propagation and flags ROUTE_FFT | ROUTE_NONFINITE.
    for (uint64_t klen : {uint64_t{1000}, uint64_t{2000}}) {
        std::vector<float> ka(klen, 1.0f);
        const ms_array aka = describe({klen}, MS_FLOAT32, ka.data());
        for (uint32_t corr : {0u, 1u}) {
            ms_signal_conv_options opts{};
            opts.struct_size = sizeof(opts);
            opts.abi_version = MS_SIGNAL_ABI_VERSION;
            opts.mode = MS_SIGNAL_FULL;
            opts.method = MS_SIGNAL_METHOD_AUTO;
            opts.correlate = corr;
            ms_array_result out{};
            ms_route_info info = new_info();
            CHECK(ms_signal_convolve(ctx, &ax, &aka, &opts, &r_gpu, &out, &info) == MS_OK);
            CHECK(info.engine == MS_EXECUTION_GPU);
            CHECK(info.reserved[0] == (MS_SIGNAL_ROUTE_FFT | MS_SIGNAL_ROUTE_NONFINITE));
            const auto* p = reinterpret_cast<const float*>(read(out.buffer, ctx).data);
            for (uint64_t i = 0; i < n1 + klen - 1; ++i) CHECK(std::isnan(p[i]));
            ms_buffer_release(out.buffer);
        }
    }

    // 2-D complex64 (256, 256) x (16, 16) under METHOD_AUTO routes to FFT and reports ROUTE_FFT | ROUTE_NONFINITE.
    {
        std::vector<float> a2d(256 * 256 * 2, 1.0f);
        a2d[2 * (200 * 256 + 100)] = std::numeric_limits<float>::infinity();
        std::vector<float> k2d(16 * 16 * 2, 1.0f);
        const ms_array aa2 = describe({256, 256}, MS_COMPLEX64, a2d.data());
        const ms_array ak2 = describe({16, 16}, MS_COMPLEX64, k2d.data());
        ms_signal_conv_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = MS_SIGNAL_ABI_VERSION;
        opts.mode = MS_SIGNAL_FULL;
        opts.method = MS_SIGNAL_METHOD_AUTO;
        ms_array_result out{};
        ms_route_info info = new_info();
        CHECK(ms_signal_convolve(ctx, &aa2, &ak2, &opts, &r_gpu, &out, &info) == MS_OK);
        CHECK(info.engine == MS_EXECUTION_GPU);
        CHECK(info.reserved[0] == (MS_SIGNAL_ROUTE_FFT | MS_SIGNAL_ROUTE_NONFINITE));
        ms_buffer_release(out.buffer);
    }

    // Finite MS_SIGNAL_METHOD_AUTO via FFT reports MS_SIGNAL_ROUTE_FFT without MS_SIGNAL_ROUTE_NONFINITE.
    {
        std::vector<float> xf(n1, 1.0f);
        std::vector<float> kf(1000, 1.0f);
        const ms_array axf = describe({n1}, MS_FLOAT32, xf.data());
        const ms_array akf = describe({1000}, MS_FLOAT32, kf.data());
        ms_signal_conv_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = MS_SIGNAL_ABI_VERSION;
        opts.mode = MS_SIGNAL_FULL;
        opts.method = MS_SIGNAL_METHOD_AUTO;
        ms_array_result out{};
        ms_route_info info = new_info();
        CHECK(ms_signal_convolve(ctx, &axf, &akf, &opts, &r_gpu, &out, &info) == MS_OK);
        CHECK(info.engine == MS_EXECUTION_GPU);
        CHECK(info.reserved[0] == MS_SIGNAL_ROUTE_FFT);
        ms_buffer_release(out.buffer);
    }

    // MS_SIGNAL_METHOD_AUTO routed to direct with NaN does NOT set MS_SIGNAL_ROUTE_FFT.
    {
        std::vector<float> xd(2000, 1.0f);
        xd[500] = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> kd(5, 1.0f);
        const ms_array axd = describe({2000}, MS_FLOAT32, xd.data());
        const ms_array akd = describe({5}, MS_FLOAT32, kd.data());
        ms_signal_conv_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = MS_SIGNAL_ABI_VERSION;
        opts.mode = MS_SIGNAL_FULL;
        opts.method = MS_SIGNAL_METHOD_AUTO;
        ms_array_result out{};
        ms_route_info info = new_info();
        CHECK(ms_signal_convolve(ctx, &axd, &akd, &opts, &r_gpu, &out, &info) == MS_OK);
        CHECK(info.reserved[0] == 0);
        ms_buffer_release(out.buffer);
    }

    // ms_signal_fftconvolve(..., overlap_add=1) keeps OA block isolation on non-finite inputs.
    {
        ms_signal_fftconv_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = MS_SIGNAL_ABI_VERSION;
        opts.mode = MS_SIGNAL_FULL;
        opts.overlap_add = 1;
        ms_array_result out{};
        ms_route_info info = new_info();
        CHECK(ms_signal_fftconvolve(ctx, &ax, &ak, &opts, &r_gpu, &out, &info) == MS_OK);
        CHECK(info.engine == MS_EXECUTION_GPU);
        const auto* p = reinterpret_cast<const float*>(read(out.buffer, ctx).data);
        CHECK(std::isfinite(p[0]) && std::fabs(p[0] - 1.0f) < 1e-3f);
        CHECK(std::isnan(p[30000]));
        ms_buffer_release(out.buffer);
    }
}

void test_direct_publish_inverse_fft_overflow_nonfinite() {
    if (!have_gpu) return;
    const ms_route_options r_gpu = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);

    // float32 (32769 x 32768): finite inputs & spectral products (nf_flags == 0 before inverse FFT),
    // direct_publish == true, out_buf[0] finite, yet inverse FFT overflows at index > 0.
    for (float val : {1e17f, 1e19f}) {
        const uint64_t n1 = 32769, n2 = 32768;
        std::vector<float> x(n1, 0.0f), k(n2, 0.0f);
        x[101] = val;
        k[100] = val;
        const ms_array ax = describe({n1}, MS_FLOAT32, x.data());
        const ms_array ak = describe({n2}, MS_FLOAT32, k.data());

        for (uint32_t method : {MS_SIGNAL_METHOD_AUTO, MS_SIGNAL_METHOD_FFT}) {
            for (uint32_t corr : {0u, 1u}) {
                ms_signal_conv_options opts{};
                opts.struct_size = sizeof(opts);
                opts.abi_version = MS_SIGNAL_ABI_VERSION;
                opts.mode = MS_SIGNAL_FULL;
                opts.method = method;
                opts.correlate = corr;
                ms_array_result out{};
                ms_route_info info = new_info();
                const auto before = exec_info();
                CHECK(ms_signal_convolve(ctx, &ax, &ak, &opts, &r_gpu, &out, &info) == MS_OK);
                const auto after = exec_info();
                CHECK(after.gpu_command_buffers == before.gpu_command_buffers + 1);
                CHECK(info.engine == MS_EXECUTION_GPU);
                CHECK(info.reserved[0] == (MS_SIGNAL_ROUTE_FFT | MS_SIGNAL_ROUTE_NONFINITE));
                const auto* p = reinterpret_cast<const float*>(read(out.buffer, ctx).data);
                if (corr == 0u) CHECK(std::isfinite(p[0]));
                bool any_nf = false;
                for (uint64_t i = (corr == 0u ? 1 : 0); i < n1 + n2 - 1; ++i) {
                    if (!std::isfinite(p[i])) {
                        any_nf = true;
                        break;
                    }
                }
                CHECK(any_nf);
                ms_buffer_release(out.buffer);
            }
        }

        // ms_signal_fftconvolve handles the same shape/input without error and leaves reserved[0] == 0.
        ms_signal_fftconv_options fopts{};
        fopts.struct_size = sizeof(fopts);
        fopts.abi_version = MS_SIGNAL_ABI_VERSION;
        fopts.mode = MS_SIGNAL_FULL;
        ms_array_result fout{};
        ms_route_info finfo = new_info();
        CHECK(ms_signal_fftconvolve(ctx, &ax, &ak, &fopts, &r_gpu, &fout, &finfo) == MS_OK);
        CHECK(finfo.engine == MS_EXECUTION_GPU);
        CHECK(finfo.reserved[0] == 0);
        ms_buffer_release(fout.buffer);
    }

    // complex64 (32806 x 32805) with val = 1e19f: non-pow2 direct_publish inverse-FFT overflow at index > 0.
    {
        const uint64_t n1 = 32806, n2 = 32805;
        std::vector<float> x(n1 * 2, 0.0f), k(n2 * 2, 0.0f);
        x[101 * 2] = 1e19f;
        k[100 * 2] = 1e19f;
        const ms_array ax = describe({n1}, MS_COMPLEX64, x.data());
        const ms_array ak = describe({n2}, MS_COMPLEX64, k.data());
        for (uint32_t method : {MS_SIGNAL_METHOD_AUTO, MS_SIGNAL_METHOD_FFT}) {
            for (uint32_t corr : {0u, 1u}) {
                ms_signal_conv_options opts{};
                opts.struct_size = sizeof(opts);
                opts.abi_version = MS_SIGNAL_ABI_VERSION;
                opts.mode = MS_SIGNAL_FULL;
                opts.method = method;
                opts.correlate = corr;
                ms_array_result out{};
                ms_route_info info = new_info();
                const auto before = exec_info();
                CHECK(ms_signal_convolve(ctx, &ax, &ak, &opts, &r_gpu, &out, &info) == MS_OK);
                const auto after = exec_info();
                CHECK(after.gpu_command_buffers == before.gpu_command_buffers + 1);
                CHECK(info.engine == MS_EXECUTION_GPU);
                CHECK(info.reserved[0] == (MS_SIGNAL_ROUTE_FFT | MS_SIGNAL_ROUTE_NONFINITE));
                const auto* p = reinterpret_cast<const float*>(read(out.buffer, ctx).data);
                if (corr == 0u) CHECK(std::isfinite(p[0]) && std::isfinite(p[1]));
                bool any_nf = false;
                for (uint64_t i = (corr == 0u ? 2 : 0); i < (n1 + n2 - 1) * 2; ++i) {
                    if (!std::isfinite(p[i])) {
                        any_nf = true;
                        break;
                    }
                }
                CHECK(any_nf);
                ms_buffer_release(out.buffer);
            }
        }
    }

    // Finite direct_publish (32769 x 32768) with 1.0f reports MS_SIGNAL_ROUTE_FFT only.
    {
        const uint64_t n1 = 32769, n2 = 32768;
        std::vector<float> x(n1, 0.0f), k(n2, 0.0f);
        x[101] = 1.0f;
        k[100] = 1.0f;
        const ms_array ax = describe({n1}, MS_FLOAT32, x.data());
        const ms_array ak = describe({n2}, MS_FLOAT32, k.data());
        ms_signal_conv_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = MS_SIGNAL_ABI_VERSION;
        opts.mode = MS_SIGNAL_FULL;
        opts.method = MS_SIGNAL_METHOD_AUTO;
        ms_array_result out{};
        ms_route_info info = new_info();
        CHECK(ms_signal_convolve(ctx, &ax, &ak, &opts, &r_gpu, &out, &info) == MS_OK);
        CHECK(info.engine == MS_EXECUTION_GPU);
        CHECK(info.reserved[0] == MS_SIGNAL_ROUTE_FFT);
        ms_buffer_release(out.buffer);
    }
}

void test_choose_conv_method() {
    const uint64_t s1_small[2] = {16, 16}, s2_small[2] = {5, 5};
    const uint64_t s1_big[2] = {2048, 2048}, s2_big[2] = {512, 512};
    const auto r = route_opts(MS_EXECUTION_AUTO, MS_PRECISION_FAST);
    uint32_t method = 99;
    CHECK(ms_signal_choose_conv_method(ctx, MS_FLOAT64, 2, s1_small, MS_FLOAT64, 2, s2_small,
                                       MS_SIGNAL_FULL, 1, &r, &method) == MS_OK);
    CHECK(method == MS_SIGNAL_METHOD_DIRECT);
    CHECK(ms_signal_choose_conv_method(ctx, MS_FLOAT64, 2, s1_big, MS_FLOAT64, 2, s2_big,
                                       MS_SIGNAL_FULL, 1, &r, &method) == MS_OK);
    CHECK(method == MS_SIGNAL_METHOD_FFT);
}

#if defined(MS_TEST_METAL)
void test_fft_scratch_encode() {
    if (!have_gpu) return;
    // Encode two FFT stages in a single gpu::Batch using fft::Scratch + fft::encode.
    const uint64_t n = 256, rows = 4;
    fft::Request req;
    req.transform = fft::Transform::c2c;
    req.n = n;
    req.count = rows;
    const fft::Limits limits = fft::device_limits();
    auto fwd = fft::plan(req, limits);
    req.inverse = true;
    auto inv = fft::plan(req, limits);
    CHECK(fwd.plan && inv.plan);

    const uint64_t bytes = rows * n * 2 * sizeof(float);
    BufferPtr in = array::allocate(ctx, bytes);
    auto* p = reinterpret_cast<float*>(mutable_data(in.get()));
    for (uint64_t i = 0; i < rows * n; ++i) {
        p[2 * i] = static_cast<float>(i % 17) * 0.25f;
        p[2 * i + 1] = -static_cast<float>(i % 11) * 0.125f;
    }

    const auto before = exec_info();
    BufferPtr out_buf;
    {
        GpuOperationScope op(ctx, true);
        fft::Scratch scratch(ctx);
        gpu::Batch batch(ctx);
        ms_buffer* s_in = scratch.adopt(std::move(in));
        ms_buffer* spec = fft::encode(ctx, batch, scratch, {{fwd.plan, rows}}, s_in, 1.0f);
        ms_buffer* round = fft::encode(ctx, batch, scratch, {{inv.plan, rows}}, spec, 1.0f);
        batch.commit_and_wait();
        op.complete();
        out_buf = scratch.take(round);
    }
    const auto after = exec_info();
    CHECK(after.gpu_command_buffers == before.gpu_command_buffers + 1);
    const auto* got = reinterpret_cast<const float*>(read(out_buf.get(), ctx).data);
    for (uint64_t i = 0; i < rows * n; ++i) {
        const float er = static_cast<float>(i % 17) * 0.25f;
        const float ei = -static_cast<float>(i % 11) * 0.125f;
        CHECK(std::fabs(got[2 * i] - er) < 1e-4f);
        CHECK(std::fabs(got[2 * i + 1] - ei) < 1e-4f);
    }
}
#endif

} // namespace

int main() {
    const char* val_env = std::getenv("MTL_SHADER_VALIDATION");
    validated = val_env && *val_env && std::strcmp(val_env, "0") != 0;

    ms_context_options opts{};
    opts.struct_size = sizeof(opts);
    opts.abi_version = MS_RUNTIME_ABI_VERSION;
    opts.flags = MS_CONTEXT_POISON_UNWRITTEN;
    opts.memory_limit_bytes = uint64_t{4} << 30;
#if defined(MS_TEST_METAL)
    opts.storage_kind = MS_BUFFER_METAL;
    have_gpu = true;
#else
    opts.storage_kind = MS_BUFFER_CPU;
    have_gpu = false;
#endif
    CHECK(ms_context_create(&opts, &ctx) == MS_OK);

    test_next_fast_len();
    test_direct_paths();
    test_mixed_real_complex_nan_semantics();
    test_fft_and_oa_gpu();
    test_2d_full_no_swap_and_overhang();
    test_method_fft_disables_oa_nonfinite();
    test_direct_publish_inverse_fft_overflow_nonfinite();
    test_choose_conv_method();
#if defined(MS_TEST_METAL)
    test_fft_scratch_encode();
#endif

    ms_memory_info mem{};
    mem.struct_size = sizeof(mem);
    mem.abi_version = MS_RUNTIME_ABI_VERSION;
    CHECK(ms_context_memory_info(ctx, &mem) == MS_OK);
    CHECK(mem.owned_bytes == 0 && mem.allocation_count == 0);
    ms_context_release(ctx);
    std::printf("signal convolve native tests ok%s\n", validated ? " (validated)" : "");
    return 0;
}
