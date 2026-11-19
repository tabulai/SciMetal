// Native tests for shared ms::detail::dense_correlate, ms_ndimage_dense_filter,
// and ms_ndimage_composite_filter (PLAIN and VALIDATED).
#include "ms_ndimage.h"
#include "core/dense_correlate.hpp"
#if defined(MS_TEST_METAL)
#include "runtime/metal_compute.hpp"
#endif

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#pragma STDC FP_CONTRACT OFF

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::fprintf(stderr, "ndimage dense test failed line %d: %s\n", __LINE__, #x);                             \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

namespace {

ms_context* ctx = nullptr;
bool have_gpu = false;

uint64_t product(const std::vector<uint64_t>& shape) {
    uint64_t n = 1;
    for (uint64_t v : shape) n *= v;
    return n;
}

std::vector<double> make_data(uint64_t n, uint32_t seed) {
    std::vector<double> values(n);
    uint64_t state = 0x9E3779B97F4A7C15ull * (seed + 1);
    for (double& v : values) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        v = static_cast<double>(state >> 11) / 9007199254740992.0 * 2.0 - 1.0 + 0.15;
    }
    return values;
}

int64_t fold(int64_t i, int64_t n, uint32_t mode) {
    if (i >= 0 && i < n) return i;
    switch (mode) {
    case MS_NDIMAGE_NEAREST: return i < 0 ? 0 : n - 1;
    case MS_NDIMAGE_WRAP:
        while (i < 0) i += n;
        while (i >= n) i -= n;
        return i;
    case MS_NDIMAGE_REFLECT:
        while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - 1 - i;
        return i;
    case MS_NDIMAGE_MIRROR:
        if (n == 1) return 0;
        while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
        return i;
    default: return -1;
    }
}

ms_array describe(const std::vector<uint64_t>& shape, uint32_t dtype, const void* data) {
    ms_array array{};
    array.struct_size = sizeof(array);
    array.abi_version = MS_ARRAY_ABI_VERSION;
    array.data = data;
    array.dtype = dtype;
    array.ndim = static_cast<uint32_t>(shape.size());
    for (size_t d = 0; d < shape.size(); ++d) array.shape[d] = shape[d];
    return array;
}

ms_route_options route_opts(uint32_t execution, uint32_t precision) {
    ms_route_options opts{};
    opts.struct_size = sizeof(opts);
    opts.abi_version = MS_ARRAY_ABI_VERSION;
    opts.execution = execution;
    opts.precision = precision;
    return opts;
}

ms_route_info new_info() {
    ms_route_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = MS_ARRAY_ABI_VERSION;
    return info;
}

ms_execution_info counters() {
    ms_execution_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = MS_RUNTIME_ABI_VERSION;
    CHECK(ms_context_execution_info(ctx, &info) == MS_OK);
    return info;
}

double max_diff(const std::vector<double>& a, const std::vector<double>& b) {
    CHECK(a.size() == b.size());
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (!(d <= worst)) worst = d;
    }
    return worst;
}

// Reference N-D correlation for arbitrary out_shape, start, per-axis modes, and prune_zeros.
std::vector<double> ref_dense_real(const std::vector<double>& in, const std::vector<uint64_t>& in_shape,
                                   const std::vector<uint64_t>& out_shape, const std::vector<uint32_t>& wshape,
                                   const std::vector<int64_t>& start, const std::vector<double>& weights,
                                   const std::vector<uint32_t>& modes, double cval, bool prune_zeros) {
    const size_t ndim = in_shape.size();
    std::vector<uint64_t> in_strides(ndim, 1);
    for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) in_strides[d] = in_strides[d + 1] * in_shape[d + 1];

    const uint64_t out_n = product(out_shape);
    const uint64_t w_n = std::accumulate(wshape.begin(), wshape.end(), uint64_t{1}, std::multiplies<uint64_t>());
    std::vector<double> out(out_n, 0.0);
    std::vector<uint64_t> p(ndim, 0), wc(ndim, 0);

    for (uint64_t oi = 0; oi < out_n; ++oi) {
        {
            uint64_t rem = oi;
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                p[d] = rem % out_shape[d];
                rem /= out_shape[d];
            }
        }
        double acc = 0.0;
        std::fill(wc.begin(), wc.end(), 0);
        for (uint64_t wi = 0; wi < w_n; ++wi) {
            const double w = weights[wi];
            if (!prune_zeros || std::fabs(w) > DBL_EPSILON) {
                bool ok = true;
                uint64_t in_idx = 0;
                for (size_t d = 0; d < ndim; ++d) {
                    const int64_t pos = static_cast<int64_t>(p[d]) + static_cast<int64_t>(wc[d]) + start[d];
                    const int64_t q = fold(pos, static_cast<int64_t>(in_shape[d]), modes[d]);
                    if (q < 0) {
                        ok = false;
                        break;
                    }
                    in_idx += static_cast<uint64_t>(q) * in_strides[d];
                }
                const double val = ok ? in[in_idx] : cval;
                acc += w * val;
            }
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                if (++wc[d] < wshape[d]) break;
                wc[d] = 0;
            }
        }
        out[oi] = acc;
    }
    return out;
}

void test_shared_dense_correlate_primitive() {
    namespace dense = ms::detail::dense_correlate;
    namespace array = ms::detail::array;

    // 1. Real N-D with arbitrary out_shape (valid/same/full-like), per-axis modes, prune_zeros=false/true.
    {
        const std::vector<uint64_t> in_shape = {9, 11, 7};
        const std::vector<uint64_t> out_shape = {7, 8, 9};
        const std::vector<uint32_t> wshape = {3, 4, 2};
        const std::vector<int64_t> start = {0, -1, -1};
        const std::vector<uint32_t> modes = {MS_NDIMAGE_REFLECT, MS_NDIMAGE_WRAP, MS_NDIMAGE_CONSTANT};
        std::vector<double> in_data = make_data(product(in_shape), 101);
        std::vector<double> weights = make_data(3 * 4 * 2, 202);
        weights[5] = 0.0;
        weights[11] = 1e-17;

        for (bool prune : {true, false}) {
            dense::StencilSpec spec;
            spec.in_shape.ndim = 3;
            spec.out_shape.ndim = 3;
            spec.in_shape.elements = product(in_shape);
            spec.out_shape.elements = product(out_shape);
            for (int d = 0; d < 3; ++d) {
                spec.in_shape.dims[d] = in_shape[d];
                spec.out_shape.dims[d] = out_shape[d];
                spec.wshape[d] = wshape[d];
                spec.start[d] = start[d];
                spec.mode[d] = modes[d];
            }
            spec.in_dtype = MS_FLOAT64;
            spec.out_dtype = MS_FLOAT64;
            spec.cval_real = -0.5;
            spec.cpu_accum = dense::AccumMode::Double;
            spec.prune_zeros = prune;
            spec.weights_dtype = MS_FLOAT64;
            spec.weights = weights.data();

            const dense::TapTable taps = dense::build_taps(spec);
            const auto expected =
                ref_dense_real(in_data, in_shape, out_shape, wshape, start, weights, modes, -0.5, prune);
            auto cpu_buf = dense::run_cpu(ctx, spec, taps, in_data.data());
            const void* ptr = nullptr;
            uint64_t bytes = 0;
            CHECK(ms_buffer_data(cpu_buf.get(), &ptr, &bytes) == MS_OK);
            CHECK(bytes == expected.size() * sizeof(double));
            std::vector<double> got(expected.size());
            std::memcpy(got.data(), ptr, bytes);
            CHECK(max_diff(got, expected) <= 1e-12);

            if (have_gpu) {
                CHECK(dense::gpu_supports(spec, taps));
                const ms_array in_arr = describe(in_shape, MS_FLOAT64, in_data.data());
                std::string reason;
                auto gpu_buf = dense::run_gpu(ctx, spec, taps, in_arr, false, reason);
                CHECK(gpu_buf && reason.empty());
                CHECK(ms_buffer_data(gpu_buf.get(), &ptr, &bytes) == MS_OK);
                std::vector<double> got_gpu(expected.size());
                std::memcpy(got_gpu.data(), ptr, bytes);
                CHECK(max_diff(got_gpu, expected) <= 1e-4);

#if defined(MS_TEST_METAL)
                // Also verify encode_gpu inside caller's gpu::Batch (1 command buffer).
                const ms_execution_info before = counters();
                auto single = array::load_single(ctx, in_arr, spec.in_shape);
                auto batch_out = array::allocate(ctx, bytes);
                {
                    ms::detail::GpuOperationScope op(ctx, true);
                    ms::detail::gpu::Batch batch(ctx);
                    auto tap_buf = dense::encode_gpu(ctx, batch, spec, taps, single.get(), batch_out.get());
                    batch.commit_and_wait();
                    op.complete();
                }
                const ms_execution_info after = counters();
                CHECK(after.gpu_operations == before.gpu_operations + 1);
                CHECK(after.gpu_command_buffers == before.gpu_command_buffers + 1);
                CHECK(after.gpu_dispatches == before.gpu_dispatches + 1);
                CHECK(ms_buffer_data(batch_out.get(), &ptr, &bytes) == MS_OK);
                std::vector<double> got_batch(expected.size());
                std::memcpy(got_batch.data(), ptr, bytes);
                CHECK(got_batch == got_gpu);
#endif
            }
        }
    }

    // 2. Complex64 / Complex128 with complex weights and AccumMode::InputDtype.
    {
        const std::vector<uint64_t> in_shape = {10, 12};
        const std::vector<uint64_t> out_shape = {8, 9};
        const std::vector<uint32_t> wshape = {3, 4};
        std::vector<double> in_raw = make_data(product(in_shape) * 2, 303);
        std::vector<double> w_raw = make_data(3 * 4 * 2, 404);
        std::vector<std::complex<double>> in_c(product(in_shape));
        std::vector<std::complex<double>> w_c(12);
        for (size_t i = 0; i < in_c.size(); ++i) in_c[i] = {in_raw[2 * i], in_raw[2 * i + 1]};
        for (size_t i = 0; i < w_c.size(); ++i) w_c[i] = {w_raw[2 * i], w_raw[2 * i + 1]};

        std::vector<std::complex<double>> expected(product(out_shape), {0.0, 0.0});
        for (uint64_t oy = 0; oy < out_shape[0]; ++oy) {
            for (uint64_t ox = 0; ox < out_shape[1]; ++ox) {
                std::complex<double> acc{0.0, 0.0};
                for (uint32_t wy = 0; wy < wshape[0]; ++wy) {
                    for (uint32_t wx = 0; wx < wshape[1]; ++wx) {
                        acc += in_c[(oy + wy) * in_shape[1] + (ox + wx)] * w_c[wy * wshape[1] + wx];
                    }
                }
                expected[oy * out_shape[1] + ox] = acc;
            }
        }

        dense::StencilSpec spec;
        spec.in_shape.ndim = spec.out_shape.ndim = 2;
        spec.in_shape.dims[0] = in_shape[0];
        spec.in_shape.dims[1] = in_shape[1];
        spec.in_shape.elements = product(in_shape);
        spec.out_shape.dims[0] = out_shape[0];
        spec.out_shape.dims[1] = out_shape[1];
        spec.out_shape.elements = product(out_shape);
        spec.wshape[0] = wshape[0];
        spec.wshape[1] = wshape[1];
        spec.in_dtype = MS_COMPLEX128;
        spec.out_dtype = MS_COMPLEX128;
        spec.set_mode(dense::Constant);
        spec.prune_zeros = false;
        spec.weights_dtype = MS_COMPLEX128;
        spec.weights = w_c.data();

        const dense::TapTable taps = dense::build_taps(spec);
        CHECK(taps.all_interior);
        auto cpu_buf = dense::run_cpu(ctx, spec, taps, in_c.data());
        const void* ptr = nullptr;
        uint64_t bytes = 0;
        CHECK(ms_buffer_data(cpu_buf.get(), &ptr, &bytes) == MS_OK);
        const auto* got = static_cast<const std::complex<double>*>(ptr);
        for (size_t i = 0; i < expected.size(); ++i) {
            CHECK(std::abs(got[i] - expected[i]) <= 1e-12);
        }

        if (have_gpu) {
            const ms_array in_arr = describe(in_shape, MS_COMPLEX128, in_c.data());
            std::string reason;
            auto gpu_buf = dense::run_gpu(ctx, spec, taps, in_arr, false, reason);
            CHECK(gpu_buf);
            CHECK(ms_buffer_data(gpu_buf.get(), &ptr, &bytes) == MS_OK);
            const auto* got_g = static_cast<const std::complex<double>*>(ptr);
            for (size_t i = 0; i < expected.size(); ++i) {
                CHECK(std::abs(got_g[i] - expected[i]) <= 1e-4);
            }
        }
    }
}

void test_dense_filter_api_and_rank1_cval() {
    // 1. Non-separable dense filter: CPU vs ref, GPU vs CPU, 1 command buffer.
    {
        const std::vector<uint64_t> shape = {64, 80};
        const std::vector<double> data = make_data(product(shape), 501);
        const std::vector<double> weights = {0.1, -0.2, 0.3, 0.4, 0.5, -0.1, 0.2, 0.0, -0.3};
        const ms_array input = describe(shape, MS_FLOAT64, data.data());

        ms_ndimage_dense_options dopts{};
        dopts.struct_size = sizeof(dopts);
        dopts.abi_version = MS_NDIMAGE_ABI_VERSION;
        dopts.mode = MS_NDIMAGE_MIRROR;
        dopts.wshape[0] = 3;
        dopts.wshape[1] = 3;
        dopts.weights = weights.data();

        const auto cpu_opts = route_opts(MS_EXECUTION_CPU, MS_PRECISION_STRICT);
        ms_array_result cpu_res{};
        ms_route_info cpu_info = new_info();
        CHECK(ms_ndimage_dense_filter(ctx, &input, &dopts, &cpu_opts, &cpu_res, &cpu_info) == MS_OK);
        CHECK(cpu_info.engine == MS_EXECUTION_CPU && cpu_info.compute_dtype == MS_FLOAT64);

        const void* ptr = nullptr;
        uint64_t bytes = 0;
        CHECK(ms_buffer_data(cpu_res.buffer, &ptr, &bytes) == MS_OK);
        std::vector<double> cpu_vals(product(shape));
        std::memcpy(cpu_vals.data(), ptr, bytes);
        ms_buffer_release(cpu_res.buffer);

        const auto ref =
            ref_dense_real(data, shape, shape, {3, 3}, {-1, -1}, weights, {MS_NDIMAGE_MIRROR, MS_NDIMAGE_MIRROR}, 0.0, true);
        CHECK(max_diff(cpu_vals, ref) <= 1e-13);

        if (have_gpu) {
            const ms_execution_info before = counters();
            const auto gpu_opts = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);
            ms_array_result gpu_res{};
            ms_route_info gpu_info = new_info();
            CHECK(ms_ndimage_dense_filter(ctx, &input, &dopts, &gpu_opts, &gpu_res, &gpu_info) == MS_OK);
            const ms_execution_info after = counters();
            CHECK(gpu_info.engine == MS_EXECUTION_GPU && gpu_info.compute_dtype == MS_FLOAT32);
            CHECK(after.gpu_operations == before.gpu_operations + 1);
            CHECK(after.gpu_command_buffers == before.gpu_command_buffers + 1);
            CHECK(after.gpu_dispatches == before.gpu_dispatches + 1);

            CHECK(ms_buffer_data(gpu_res.buffer, &ptr, &bytes) == MS_OK);
            std::vector<double> gpu_vals(product(shape));
            std::memcpy(gpu_vals.data(), ptr, bytes);
            ms_buffer_release(gpu_res.buffer);
            CHECK(max_diff(gpu_vals, cpu_vals) <= 1e-5);
        }
    }

    // 2. Rank-1 separable kernel with non-unit weight sum and CONSTANT cval != 0 (cval_k adjustment!).
    if (have_gpu) {
        const std::vector<uint64_t> shape = {48, 60};
        const std::vector<double> data = make_data(product(shape), 601);
        const std::vector<double> v0 = {1.0, 2.0, 1.0};   // sum = 4
        const std::vector<double> v1 = {-1.0, 3.0, 2.0};  // sum = 4
        std::vector<double> weights(9);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) weights[i * 3 + j] = v0[i] * v1[j];

        const ms_array input = describe(shape, MS_FLOAT64, data.data());
        ms_ndimage_dense_options dopts{};
        dopts.struct_size = sizeof(dopts);
        dopts.abi_version = MS_NDIMAGE_ABI_VERSION;
        dopts.mode = MS_NDIMAGE_CONSTANT;
        dopts.cval = 2.75;
        dopts.wshape[0] = 3;
        dopts.wshape[1] = 3;
        dopts.weights = weights.data();

        const auto cpu_opts = route_opts(MS_EXECUTION_CPU, MS_PRECISION_STRICT);
        ms_array_result cpu_res{};
        ms_route_info cpu_info = new_info();
        CHECK(ms_ndimage_dense_filter(ctx, &input, &dopts, &cpu_opts, &cpu_res, &cpu_info) == MS_OK);

        const void* ptr = nullptr;
        uint64_t bytes = 0;
        CHECK(ms_buffer_data(cpu_res.buffer, &ptr, &bytes) == MS_OK);
        std::vector<double> cpu_vals(product(shape));
        std::memcpy(cpu_vals.data(), ptr, bytes);
        ms_buffer_release(cpu_res.buffer);

        const ms_execution_info before = counters();
        const auto gpu_opts = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);
        ms_array_result gpu_res{};
        ms_route_info gpu_info = new_info();
        CHECK(ms_ndimage_dense_filter(ctx, &input, &dopts, &gpu_opts, &gpu_res, &gpu_info) == MS_OK);
        const ms_execution_info after = counters();
        CHECK(gpu_info.engine == MS_EXECUTION_GPU);
        CHECK(after.gpu_operations == before.gpu_operations + 1);
        CHECK(after.gpu_command_buffers == before.gpu_command_buffers + 1);
        // Rank-1 separable runs as 2 1-D dispatches in 1 command buffer!
        CHECK(after.gpu_dispatches == before.gpu_dispatches + 2);

        CHECK(ms_buffer_data(gpu_res.buffer, &ptr, &bytes) == MS_OK);
        std::vector<double> gpu_vals(product(shape));
        std::memcpy(gpu_vals.data(), ptr, bytes);
        ms_buffer_release(gpu_res.buffer);
        CHECK(max_diff(gpu_vals, cpu_vals) <= 1e-4);
    }

    // 3. Overflow guard rails on dense filter: AUTO -> CPU with reason; GPU -> MS_ENOTSUP.
    if (have_gpu) {
        const std::vector<uint64_t> large = {2896, 2896};
        std::vector<double> weights(25, 0.1);
        weights[0] = -0.2;
        std::vector<double> over(product(large), 1.0);
        over[0] = 1e39;
        const ms_array input = describe(large, MS_FLOAT64, over.data());

        ms_ndimage_dense_options dopts{};
        dopts.struct_size = sizeof(dopts);
        dopts.abi_version = MS_NDIMAGE_ABI_VERSION;
        dopts.mode = MS_NDIMAGE_REFLECT;
        dopts.wshape[0] = 5;
        dopts.wshape[1] = 5;
        dopts.weights = weights.data();

        const auto auto_opts = route_opts(MS_EXECUTION_AUTO, MS_PRECISION_FAST);
        const ms_execution_info before = counters();
        ms_array_result auto_res{};
        ms_route_info auto_info = new_info();
        CHECK(ms_ndimage_dense_filter(ctx, &input, &dopts, &auto_opts, &auto_res, &auto_info) == MS_OK);
        const ms_execution_info after = counters();
        CHECK(auto_info.engine == MS_EXECUTION_CPU && auto_info.compute_dtype == MS_FLOAT64);
        CHECK(auto_info.reason[0] != '\0');
        CHECK(after.gpu_operations == before.gpu_operations);
        ms_buffer_release(auto_res.buffer);

        const auto gpu_opts = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);
        ms_array_result gpu_res{};
        CHECK(ms_ndimage_dense_filter(ctx, &input, &dopts, &gpu_opts, &gpu_res, nullptr) == MS_ENOTSUP);
    }

    // 4. High-dynamic-range non-rank-1 kernel (w = {1.0, 0.0, 0.0, 1e-15}) must NOT drop off-pivot tap.
    if (have_gpu) {
        const std::vector<uint64_t> shape = {8, 8};
        std::vector<double> in_vals(product(shape), 0.0);
        in_vals[1 * 8 + 1] = 1e20;
        const std::vector<double> w_hdr = {1.0, 0.0, 0.0, 1e-15};
        const ms_array input = describe(shape, MS_FLOAT64, in_vals.data());

        ms_ndimage_dense_options dopts{};
        dopts.struct_size = sizeof(dopts);
        dopts.abi_version = MS_NDIMAGE_ABI_VERSION;
        dopts.mode = MS_NDIMAGE_CONSTANT;
        dopts.cval = 0.0;
        dopts.wshape[0] = 2;
        dopts.wshape[1] = 2;
        dopts.origins[0] = -1;
        dopts.origins[1] = -1;
        dopts.weights = w_hdr.data();

        const auto gpu_opts = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);
        ms_array_result gpu_res{};
        ms_route_info gpu_info = new_info();
        CHECK(ms_ndimage_dense_filter(ctx, &input, &dopts, &gpu_opts, &gpu_res, &gpu_info) == MS_OK);
        CHECK(gpu_info.engine == MS_EXECUTION_GPU);

        const void* ptr = nullptr;
        uint64_t bytes = 0;
        CHECK(ms_buffer_data(gpu_res.buffer, &ptr, &bytes) == MS_OK);
        std::vector<double> gpu_vals(product(shape));
        std::memcpy(gpu_vals.data(), ptr, bytes);
        ms_buffer_release(gpu_res.buffer);
        // Output[0, 0] correlates window [0..1, 0..1], hitting in[1, 1] * w[1, 1] = 1e20 * 1e-15 = 1e5.
        CHECK(std::fabs(gpu_vals[0] - 1e5) <= 1.0);
    }

    // 5. Rank-1 kernels with zero taps must prune zeros rather than leaking 0.0 * NaN on GPU.
    if (have_gpu) {
        const std::vector<uint64_t> shape = {5, 5};
        std::vector<double> in_vals(product(shape), 1.0);
        in_vals[0] = std::numeric_limits<double>::quiet_NaN();
        const ms_array input = describe(shape, MS_FLOAT64, in_vals.data());

        const std::vector<double> w_delta = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
        const std::vector<double> w_sobel = {-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0};
        for (const auto& w : {w_delta, w_sobel}) {
            ms_ndimage_dense_options dopts{};
            dopts.struct_size = sizeof(dopts);
            dopts.abi_version = MS_NDIMAGE_ABI_VERSION;
            dopts.mode = MS_NDIMAGE_CONSTANT;
            dopts.cval = 0.0;
            dopts.wshape[0] = 3;
            dopts.wshape[1] = 3;
            dopts.weights = w.data();

            const auto cpu_opts = route_opts(MS_EXECUTION_CPU, MS_PRECISION_STRICT);
            const auto gpu_opts = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);
            ms_array_result cpu_res{}, gpu_res{};
            CHECK(ms_ndimage_dense_filter(ctx, &input, &dopts, &cpu_opts, &cpu_res, nullptr) == MS_OK);
            CHECK(ms_ndimage_dense_filter(ctx, &input, &dopts, &gpu_opts, &gpu_res, nullptr) == MS_OK);

            const void *cp = nullptr, *gp = nullptr;
            uint64_t cb = 0, gb = 0;
            CHECK(ms_buffer_data(cpu_res.buffer, &cp, &cb) == MS_OK);
            CHECK(ms_buffer_data(gpu_res.buffer, &gp, &gb) == MS_OK);
            std::vector<double> cv(product(shape)), gv(product(shape));
            std::memcpy(cv.data(), cp, cb);
            std::memcpy(gv.data(), gp, gb);
            ms_buffer_release(cpu_res.buffer);
            ms_buffer_release(gpu_res.buffer);
            for (size_t i = 0; i < cv.size(); ++i) {
                CHECK(std::isnan(cv[i]) == std::isnan(gv[i]));
                if (!std::isnan(cv[i])) CHECK(std::fabs(cv[i] - gv[i]) <= 1e-5);
            }
        }
    }
}

void test_composite_filter() {
    const std::vector<uint64_t> shape = {24, 30, 18};
    const std::vector<double> data = make_data(product(shape), 701);
    const ms_array input = describe(shape, MS_FLOAT64, data.data());

    const std::vector<double> w_lap = {1.0, -2.0, 1.0};
    const std::vector<double> w_sm = {0.25, 0.5, 0.25};

    // Build 3 chains: chain 0 has 1 pass, chain 1 has 0 passes (identity!), chain 2 has 2 passes.
    ms_ndimage_pass p0{}, p2a{}, p2b{};
    for (ms_ndimage_pass* p : {&p0, &p2a, &p2b}) {
        p->struct_size = sizeof(*p);
        p->abi_version = MS_NDIMAGE_ABI_VERSION;
        p->kind = MS_NDIMAGE_CORRELATE;
        p->size = 3;
    }
    p0.axis = 0;
    p0.mode = MS_NDIMAGE_REFLECT;
    p0.weights = w_lap.data();
    p2a.axis = 1;
    p2a.mode = MS_NDIMAGE_CONSTANT;
    p2a.cval = 1.5;
    p2a.weights = w_sm.data();
    p2b.axis = 2;
    p2b.mode = MS_NDIMAGE_WRAP;
    p2b.weights = w_lap.data();

    ms_ndimage_chain chains[3] = {};
    for (auto& c : chains) {
        c.struct_size = sizeof(c);
        c.abi_version = MS_NDIMAGE_ABI_VERSION;
    }
    chains[0].pass_count = 1;
    chains[0].passes = &p0;
    chains[1].pass_count = 0;
    chains[1].passes = nullptr;
    chains[2].pass_count = 2;
    chains[2].passes = new ms_ndimage_pass[2]{p2a, p2b};

    for (uint32_t combine_op : {MS_NDIMAGE_COMBINE_SUM, MS_NDIMAGE_COMBINE_SQRT_SUM_SQ}) {
        const auto cpu_opts = route_opts(MS_EXECUTION_CPU, MS_PRECISION_STRICT);
        ms_array_result cpu_res{};
        ms_route_info cpu_info = new_info();
        CHECK(ms_ndimage_composite_filter(ctx, &input, chains, 3, combine_op, 0, &cpu_opts, &cpu_res, &cpu_info) ==
              MS_OK);
        CHECK(cpu_info.engine == MS_EXECUTION_CPU && cpu_info.compute_dtype == MS_FLOAT64);

        const void* ptr = nullptr;
        uint64_t bytes = 0;
        CHECK(ms_buffer_data(cpu_res.buffer, &ptr, &bytes) == MS_OK);
        std::vector<double> cpu_vals(product(shape));
        std::memcpy(cpu_vals.data(), ptr, bytes);
        ms_buffer_release(cpu_res.buffer);

        // Compare against independent separable calls for each chain!
        auto run_sep = [&](const ms_ndimage_pass* ps, uint32_t cnt) {
            ms_array_result r{};
            CHECK(ms_ndimage_separable_filter(ctx, &input, ps, cnt, &cpu_opts, &r, nullptr) == MS_OK);
            const void* rp = nullptr;
            uint64_t rb = 0;
            CHECK(ms_buffer_data(r.buffer, &rp, &rb) == MS_OK);
            std::vector<double> v(product(shape));
            std::memcpy(v.data(), rp, rb);
            ms_buffer_release(r.buffer);
            return v;
        };
        const auto c0 = run_sep(&p0, 1);
        const auto& c1 = data;
        const auto c2 = run_sep(chains[2].passes, 2);
        std::vector<double> expected(product(shape));
        for (size_t i = 0; i < expected.size(); ++i) {
            if (combine_op == MS_NDIMAGE_COMBINE_SUM) {
                expected[i] = c0[i] + c1[i] + c2[i];
            } else {
                expected[i] = std::sqrt(c0[i] * c0[i] + c1[i] * c1[i] + c2[i] * c2[i]);
            }
        }
        CHECK(cpu_vals == expected);

        if (have_gpu) {
            const ms_execution_info before = counters();
            const auto gpu_opts = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);
            ms_array_result gpu_res{};
            ms_route_info gpu_info = new_info();
            CHECK(ms_ndimage_composite_filter(ctx, &input, chains, 3, combine_op, 0, &gpu_opts, &gpu_res,
                                              &gpu_info) == MS_OK);
            const ms_execution_info after = counters();
            CHECK(gpu_info.engine == MS_EXECUTION_GPU && gpu_info.compute_dtype == MS_FLOAT32);
            CHECK(after.gpu_operations == before.gpu_operations + 1);
            CHECK(after.gpu_command_buffers == before.gpu_command_buffers + 1);

            CHECK(ms_buffer_data(gpu_res.buffer, &ptr, &bytes) == MS_OK);
            std::vector<double> gpu_vals(product(shape));
            std::memcpy(gpu_vals.data(), ptr, bytes);
            ms_buffer_release(gpu_res.buffer);
            CHECK(max_diff(gpu_vals, cpu_vals) <= 1e-4);
        }
    }
    delete[] chains[2].passes;

    // 1-chain 0-pass SQRT_SUM_SQ and 1e20 overflow guard rail.
    if (have_gpu) {
        ms_ndimage_chain id_chain{};
        id_chain.struct_size = sizeof(id_chain);
        id_chain.abi_version = MS_NDIMAGE_ABI_VERSION;
        id_chain.pass_count = 0;

        const std::vector<uint64_t> large = {1536, 2048};
        std::vector<double> big(product(large), 1e20);
        const ms_array big_arr = describe(large, MS_FLOAT64, big.data());
        const auto gpu_opts = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);
        ms_array_result res{};
        CHECK(ms_ndimage_composite_filter(ctx, &big_arr, &id_chain, 1, MS_NDIMAGE_COMBINE_SQRT_SUM_SQ, 0,
                                          &gpu_opts, &res, nullptr) == MS_ENOTSUP);
    }

    // Regression: K=1, SQRT_SUM_SQ, pass_count = 2..6 on >= 256x256 (tests ping/res0 non-aliasing).
    if (have_gpu) {
        const std::vector<uint64_t> large = {256, 256};
        const std::vector<double> large_data = make_data(product(large), 808);
        const ms_array large_arr = describe(large, MS_FLOAT64, large_data.data());
        const std::vector<double> w0 = {0.25, 0.5, 0.25};
        const std::vector<double> w1 = {-0.5, 0.0, 0.5};

        for (uint32_t pass_count = 2; pass_count <= 6; ++pass_count) {
            std::vector<ms_ndimage_pass> passes(pass_count);
            for (uint32_t i = 0; i < pass_count; ++i) {
                passes[i] = {};
                passes[i].struct_size = sizeof(passes[i]);
                passes[i].abi_version = MS_NDIMAGE_ABI_VERSION;
                passes[i].kind = MS_NDIMAGE_CORRELATE;
                passes[i].axis = i % 2;
                passes[i].mode = MS_NDIMAGE_REFLECT;
                passes[i].size = 3;
                passes[i].weights = (i + 1 == pass_count) ? w1.data() : w0.data();
            }
            ms_ndimage_chain chain{};
            chain.struct_size = sizeof(chain);
            chain.abi_version = MS_NDIMAGE_ABI_VERSION;
            chain.pass_count = pass_count;
            chain.passes = passes.data();

            const auto cpu_opts = route_opts(MS_EXECUTION_CPU, MS_PRECISION_STRICT);
            const auto gpu_opts = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);
            ms_array_result cpu_res{}, gpu_res{};
            CHECK(ms_ndimage_composite_filter(ctx, &large_arr, &chain, 1, MS_NDIMAGE_COMBINE_SQRT_SUM_SQ, 0,
                                              &cpu_opts, &cpu_res, nullptr) == MS_OK);
            CHECK(ms_ndimage_composite_filter(ctx, &large_arr, &chain, 1, MS_NDIMAGE_COMBINE_SQRT_SUM_SQ, 0,
                                              &gpu_opts, &gpu_res, nullptr) == MS_OK);

            const void *cp = nullptr, *gp = nullptr;
            uint64_t cb = 0, gb = 0;
            CHECK(ms_buffer_data(cpu_res.buffer, &cp, &cb) == MS_OK);
            CHECK(ms_buffer_data(gpu_res.buffer, &gp, &gb) == MS_OK);
            std::vector<double> cv(product(large)), gv(product(large));
            std::memcpy(cv.data(), cp, cb);
            std::memcpy(gv.data(), gp, gb);
            ms_buffer_release(cpu_res.buffer);
            ms_buffer_release(gpu_res.buffer);
            CHECK(max_diff(gv, cv) <= 1e-4);
        }
    }
}

void test_validation_errors() {
    const std::vector<uint64_t> shape = {4, 5};
    const std::vector<double> data = make_data(20, 999);
    const ms_array input = describe(shape, MS_FLOAT64, data.data());
    const auto opts = route_opts(MS_EXECUTION_CPU, MS_PRECISION_STRICT);

    ms_array_result res{};
    ms_ndimage_dense_options dopts{};
    dopts.struct_size = sizeof(dopts);
    dopts.abi_version = MS_NDIMAGE_ABI_VERSION;
    CHECK(ms_ndimage_dense_filter(ctx, &input, &dopts, &opts, &res, nullptr) == MS_EINVAL);

    ms_ndimage_chain bad_chain{};
    bad_chain.struct_size = sizeof(bad_chain);
    bad_chain.abi_version = MS_NDIMAGE_ABI_VERSION;
    bad_chain.pass_count = 1;
    bad_chain.passes = nullptr;
    CHECK(ms_ndimage_composite_filter(ctx, &input, &bad_chain, 1, MS_NDIMAGE_COMBINE_SUM, 0, &opts, &res, nullptr) ==
          MS_EINVAL);
    CHECK(ms_ndimage_composite_filter(ctx, &input, &bad_chain, 1, 99, 0, &opts, &res, nullptr) == MS_EINVAL);
}

void test_cross_float_out_dtype() {
    const std::vector<uint64_t> shape = {24, 32};
    const std::vector<double> f64_data = make_data(product(shape), 1101);
    std::vector<float> f32_data(f64_data.size());
    std::vector<double> f32_as_f64(f64_data.size());
    for (size_t i = 0; i < f64_data.size(); ++i) {
        f32_data[i] = static_cast<float>(f64_data[i]);
        f32_as_f64[i] = static_cast<double>(f32_data[i]);
    }

    auto read_as_f64 = [](const ms_array_result& r, uint64_t n) {
        const void* ptr = nullptr;
        uint64_t bytes = 0;
        CHECK(ms_buffer_data(r.buffer, &ptr, &bytes) == MS_OK);
        CHECK(bytes == n * (r.dtype == MS_FLOAT32 ? sizeof(float) : sizeof(double)));
        std::vector<double> out(n);
        if (r.dtype == MS_FLOAT32) {
            const float* fp = static_cast<const float*>(ptr);
            for (uint64_t i = 0; i < n; ++i) out[i] = fp[i];
        } else {
            std::memcpy(out.data(), ptr, bytes);
        }
        return out;
    };

    const std::vector<double> w_dense = {0.1, -0.2, 0.3, 0.4, 0.5, -0.1, 0.2, 0.0, -0.3};
    const std::vector<double> w_sep = {1.0, 2.0, 1.0, -1.0, 3.0, 2.0, 1.0, 2.0, 1.0};
    for (uint32_t in_dt : {MS_FLOAT64, MS_FLOAT32}) {
        const ms_array input = describe(shape, in_dt, in_dt == MS_FLOAT64 ? static_cast<const void*>(f64_data.data())
                                                                          : static_cast<const void*>(f32_data.data()));
        const auto& in_ref = (in_dt == MS_FLOAT64) ? f64_data : f32_as_f64;
        for (uint32_t out_dt : {MS_FLOAT64, MS_FLOAT32}) {
            // 1. Dense filter (both non-separable and rank-1 separable).
            for (const auto& w : {w_dense, w_sep}) {
                ms_ndimage_dense_options dopts{};
                dopts.struct_size = sizeof(dopts);
                dopts.abi_version = MS_NDIMAGE_ABI_VERSION;
                dopts.mode = MS_NDIMAGE_CONSTANT;
                dopts.out_dtype = out_dt;
                dopts.cval = 1.25;
                dopts.wshape[0] = 3;
                dopts.wshape[1] = 3;
                dopts.weights = w.data();

                const auto cpu_opts = route_opts(MS_EXECUTION_CPU, MS_PRECISION_STRICT);
                ms_array_result cpu_res{};
                CHECK(ms_ndimage_dense_filter(ctx, &input, &dopts, &cpu_opts, &cpu_res, nullptr) == MS_OK);
                CHECK(cpu_res.dtype == out_dt);
                const auto cpu_vals = read_as_f64(cpu_res, product(shape));
                ms_buffer_release(cpu_res.buffer);

                auto ref = ref_dense_real(in_ref, shape, shape, {3, 3}, {-1, -1}, w,
                                          {MS_NDIMAGE_CONSTANT, MS_NDIMAGE_CONSTANT}, 1.25, true);
                if (out_dt == MS_FLOAT32) {
                    for (double& v : ref) v = static_cast<double>(static_cast<float>(v));
                }
                CHECK(max_diff(cpu_vals, ref) <= (out_dt == MS_FLOAT32 ? 1e-6 : 1e-13));

                if (have_gpu) {
                    const auto gpu_opts = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);
                    ms_array_result gpu_res{};
                    ms_route_info info = new_info();
                    CHECK(ms_ndimage_dense_filter(ctx, &input, &dopts, &gpu_opts, &gpu_res, &info) == MS_OK);
                    CHECK(gpu_res.dtype == out_dt && info.engine == MS_EXECUTION_GPU && info.compute_dtype == MS_FLOAT32);
                    const auto gpu_vals = read_as_f64(gpu_res, product(shape));
                    ms_buffer_release(gpu_res.buffer);
                    CHECK(max_diff(gpu_vals, cpu_vals) <= 1e-4);
                }
            }

            // 2. Composite filter (multi-chain + identity chain, SUM and SQRT_SUM_SQ).
            const std::vector<double> w_lap = {1.0, -2.0, 1.0};
            ms_ndimage_pass p0{}, p1{};
            p0.struct_size = p1.struct_size = sizeof(p0);
            p0.abi_version = p1.abi_version = MS_NDIMAGE_ABI_VERSION;
            p0.kind = p1.kind = MS_NDIMAGE_CORRELATE;
            p0.size = p1.size = 3;
            p0.mode = p1.mode = MS_NDIMAGE_REFLECT;
            p0.weights = p1.weights = w_lap.data();
            p0.axis = 0;
            p1.axis = 1;
            ms_ndimage_chain chains[2] = {};
            chains[0].struct_size = chains[1].struct_size = sizeof(ms_ndimage_chain);
            chains[0].abi_version = chains[1].abi_version = MS_NDIMAGE_ABI_VERSION;
            chains[0].pass_count = 1;
            chains[0].passes = &p0;
            chains[1].pass_count = 1;
            chains[1].passes = &p1;

            for (uint32_t op : {MS_NDIMAGE_COMBINE_SUM, MS_NDIMAGE_COMBINE_SQRT_SUM_SQ}) {
                const auto cpu_opts = route_opts(MS_EXECUTION_CPU, MS_PRECISION_STRICT);
                ms_array_result cpu_res{};
                CHECK(ms_ndimage_composite_filter(ctx, &input, chains, 2, op, out_dt, &cpu_opts, &cpu_res, nullptr) ==
                      MS_OK);
                CHECK(cpu_res.dtype == out_dt);
                const auto cpu_vals = read_as_f64(cpu_res, product(shape));
                ms_buffer_release(cpu_res.buffer);

                if (have_gpu) {
                    const auto gpu_opts = route_opts(MS_EXECUTION_GPU, MS_PRECISION_FAST);
                    ms_array_result gpu_res{};
                    ms_route_info info = new_info();
                    CHECK(ms_ndimage_composite_filter(ctx, &input, chains, 2, op, out_dt, &gpu_opts, &gpu_res, &info) ==
                          MS_OK);
                    CHECK(gpu_res.dtype == out_dt && info.engine == MS_EXECUTION_GPU && info.compute_dtype == MS_FLOAT32);
                    const auto gpu_vals = read_as_f64(gpu_res, product(shape));
                    ms_buffer_release(gpu_res.buffer);
                    CHECK(max_diff(gpu_vals, cpu_vals) <= 1e-4);
                }
            }
        }
    }
}

} // namespace

int main() {
    ms_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = MS_RUNTIME_ABI_VERSION;
#if defined(MS_TEST_METAL)
    options.storage_kind = MS_BUFFER_METAL;
    have_gpu = true;
#else
    options.storage_kind = MS_BUFFER_CPU;
#endif
    options.flags = MS_CONTEXT_POISON_UNWRITTEN;
    options.memory_limit_bytes = uint64_t{2} << 30;
    CHECK(ms_context_create(&options, &ctx) == MS_OK);

    test_shared_dense_correlate_primitive();
    test_dense_filter_api_and_rank1_cval();
    test_composite_filter();
    test_validation_errors();
    test_cross_float_out_dtype();

    ms_context_release(ctx);
    std::printf("ndimage dense/composite ok%s\n", have_gpu ? "" : " [CPU only]");
    return 0;
}
