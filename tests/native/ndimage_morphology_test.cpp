// Native tests for ndimage morphology & rank/median filters (PLAIN and VALIDATED):
//   - SM_NDIMAGE_MINIMUM / SM_NDIMAGE_MAXIMUM separable passes (direct + GPU van Herk)
//   - SM_NDIMAGE_COMBINE_SUB / SM_NDIMAGE_COMBINE_MORPH_LAPLACE / SM_NDIMAGE_COMBINE_XOR
//   - sm_ndimage_minmax_filter (flat & non-flat N-D)
//   - sm_ndimage_rank_filter (3x3, 5x5, 7x7 tiled medians + small/medium Forgetful Selection)
#include "sm_ndimage.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::fprintf(stderr, "ndimage morphology test failed line %d: %s\n", __LINE__, #x);                        \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

namespace {

sm_context* ctx = nullptr;
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
        v = static_cast<double>(state >> 11) / 9007199254740992.0 * 20.0 - 10.0;
    }
    return values;
}

int64_t fold(int64_t i, int64_t n, uint32_t mode) {
    if (i >= 0 && i < n) return i;
    switch (mode) {
    case SM_NDIMAGE_NEAREST: return i < 0 ? 0 : n - 1;
    case SM_NDIMAGE_WRAP:
        while (i < 0) i += n;
        while (i >= n) i -= n;
        return i;
    case SM_NDIMAGE_REFLECT:
        while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - 1 - i;
        return i;
    case SM_NDIMAGE_MIRROR:
        if (n == 1) return 0;
        while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
        return i;
    default: return -1;
    }
}

sm_array describe(const std::vector<uint64_t>& shape, uint32_t dtype, const void* data) {
    sm_array array{};
    array.struct_size = sizeof(array);
    array.abi_version = SM_ARRAY_ABI_VERSION;
    array.data = data;
    array.dtype = dtype;
    array.ndim = static_cast<uint32_t>(shape.size());
    for (size_t d = 0; d < shape.size(); ++d) array.shape[d] = shape[d];
    return array;
}

sm_route_options route_opts(uint32_t execution, uint32_t precision) {
    sm_route_options opts{};
    opts.struct_size = sizeof(opts);
    opts.abi_version = SM_ARRAY_ABI_VERSION;
    opts.execution = execution;
    opts.precision = precision;
    return opts;
}

sm_route_info new_info() {
    sm_route_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = SM_ARRAY_ABI_VERSION;
    return info;
}

sm_execution_info counters() {
    sm_execution_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = SM_RUNTIME_ABI_VERSION;
    CHECK(sm_context_execution_info(ctx, &info) == SM_OK);
    return info;
}

template<class T>
std::vector<T> read_result(const sm_array_result& res, uint64_t n) {
    sm_buffer_view view{};
    view.struct_size = sizeof(view);
    view.abi_version = SM_BUFFER_ABI_VERSION;
    CHECK(sm_buffer_export(res.buffer, &view) == SM_OK);
    std::vector<T> out(n);
    if (n) std::memcpy(out.data(), static_cast<const uint8_t*>(view.host_base) + view.byte_offset, n * sizeof(T));
    return out;
}

// Independent reference 1-D min/max pass.
std::vector<double> ref_minmax1d(const std::vector<double>& in, const std::vector<uint64_t>& shape,
                                 uint32_t axis, uint64_t size, int64_t origin, uint32_t mode,
                                 double cval, bool is_max) {
    const uint64_t length = shape[axis];
    uint64_t outer = 1, inner = 1;
    for (uint32_t d = 0; d < axis; ++d) outer *= shape[d];
    for (uint32_t d = axis + 1; d < shape.size(); ++d) inner *= shape[d];
    const int64_t before = static_cast<int64_t>(size) / 2 + origin;
    std::vector<double> out(in.size());
    for (uint64_t o = 0; o < outer; ++o) {
        for (uint64_t i = 0; i < inner; ++i) {
            const uint64_t base = o * length * inner + i;
            for (int64_t p = 0; p < static_cast<int64_t>(length); ++p) {
                double acc = 0.0;
                for (uint64_t j = 0; j < size; ++j) {
                    const int64_t q = fold(p - before + static_cast<int64_t>(j), static_cast<int64_t>(length), mode);
                    const double v = q < 0 ? cval : in[base + static_cast<uint64_t>(q) * inner];
                    if (j == 0) acc = v;
                    else acc = is_max ? (v > acc ? v : acc) : (v < acc ? v : acc);
                }
                out[base + static_cast<uint64_t>(p) * inner] = acc;
            }
        }
    }
    return out;
}

// Independent reference N-D dense min/max and rank.
std::vector<double> ref_dense_minmax(const std::vector<double>& in, const std::vector<uint64_t>& shape,
                                     const std::vector<uint32_t>& fshape, const std::vector<int64_t>& origins,
                                     const std::vector<uint8_t>& fp, const std::vector<double>* structure,
                                     uint32_t mode, double cval, bool is_max) {
    const size_t ndim = shape.size();
    std::vector<uint64_t> strides(ndim, 1);
    for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) strides[d] = strides[d + 1] * shape[d + 1];
    uint64_t total_f = 1;
    for (uint32_t s : fshape) total_f *= s;
    std::vector<std::vector<int64_t>> offsets;
    std::vector<double> ss;
    std::vector<uint32_t> fc(ndim, 0);
    for (uint64_t idx = 0; idx < total_f; ++idx) {
        if (fp.empty() || fp[idx]) {
            std::vector<int64_t> off(ndim);
            for (size_t d = 0; d < ndim; ++d) {
                off[d] = static_cast<int64_t>(fc[d]) - (static_cast<int64_t>(fshape[d]) / 2) - origins[d];
            }
            offsets.push_back(std::move(off));
            if (structure) ss.push_back((*structure)[idx]);
        }
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++fc[d] < fshape[d]) break;
            fc[d] = 0;
        }
    }
    const uint64_t n = in.size();
    std::vector<double> out(n);
    std::vector<int64_t> coord(ndim, 0);
    for (uint64_t idx = 0; idx < n; ++idx) {
        double acc = 0.0;
        for (size_t t = 0; t < offsets.size(); ++t) {
            uint64_t in_idx = 0;
            bool ok = true;
            for (size_t d = 0; d < ndim; ++d) {
                const int64_t q = fold(coord[d] + offsets[t][d], static_cast<int64_t>(shape[d]), mode);
                if (q < 0) {
                    ok = false;
                    break;
                }
                in_idx += static_cast<uint64_t>(q) * strides[d];
            }
            double v = ok ? in[in_idx] : cval;
            if (structure) v += ss[t];
            if (t == 0) acc = v;
            else acc = is_max ? (v > acc ? v : acc) : (v < acc ? v : acc);
        }
        out[idx] = acc;
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++coord[d] < static_cast<int64_t>(shape[d])) break;
            coord[d] = 0;
        }
    }
    return out;
}

std::vector<double> ref_rank(const std::vector<double>& in, const std::vector<uint64_t>& shape,
                             const std::vector<uint32_t>& fshape, const std::vector<int64_t>& origins,
                             const std::vector<uint8_t>& fp, uint32_t rank, uint32_t mode, double cval) {
    const size_t ndim = shape.size();
    std::vector<uint64_t> strides(ndim, 1);
    for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) strides[d] = strides[d + 1] * shape[d + 1];
    uint64_t total_f = 1;
    for (uint32_t s : fshape) total_f *= s;
    std::vector<std::vector<int64_t>> offsets;
    std::vector<uint32_t> fc(ndim, 0);
    for (uint64_t idx = 0; idx < total_f; ++idx) {
        if (fp.empty() || fp[idx]) {
            std::vector<int64_t> off(ndim);
            for (size_t d = 0; d < ndim; ++d) {
                off[d] = static_cast<int64_t>(fc[d]) - (static_cast<int64_t>(fshape[d]) / 2) - origins[d];
            }
            offsets.push_back(std::move(off));
        }
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++fc[d] < fshape[d]) break;
            fc[d] = 0;
        }
    }
    const uint64_t n = in.size();
    std::vector<double> out(n);
    std::vector<double> buf(offsets.size());
    std::vector<int64_t> coord(ndim, 0);
    for (uint64_t idx = 0; idx < n; ++idx) {
        for (size_t t = 0; t < offsets.size(); ++t) {
            uint64_t in_idx = 0;
            bool ok = true;
            for (size_t d = 0; d < ndim; ++d) {
                const int64_t q = fold(coord[d] + offsets[t][d], static_cast<int64_t>(shape[d]), mode);
                if (q < 0) {
                    ok = false;
                    break;
                }
                in_idx += static_cast<uint64_t>(q) * strides[d];
            }
            buf[t] = ok ? in[in_idx] : cval;
        }
        std::sort(buf.begin(), buf.end());
        out[idx] = buf[rank];
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++coord[d] < static_cast<int64_t>(shape[d])) break;
            coord[d] = 0;
        }
    }
    return out;
}

void test_separable_minmax() {
    // Small/medium windows (direct GPU path) across all 5 modes, even/odd sizes, non-zero origins, 1D/2D/3D.
    const std::vector<std::vector<uint64_t>> shapes = {{37}, {29, 31}, {5, 11, 13}};
    for (const auto& shape : shapes) {
        const uint64_t n = product(shape);
        const std::vector<double> data_f64 = make_data(n, static_cast<uint32_t>(n));
        std::vector<float> data_f32(n);
        for (uint64_t i = 0; i < n; ++i) data_f32[i] = static_cast<float>(data_f64[i]);
        const sm_array arr64 = describe(shape, SM_FLOAT64, data_f64.data());
        const sm_array arr32 = describe(shape, SM_FLOAT32, data_f32.data());

        for (uint32_t mode = 0; mode <= SM_NDIMAGE_CONSTANT; ++mode) {
            for (uint32_t is_max = 0; is_max <= 1; ++is_max) {
                const uint64_t w = (mode % 2 == 0) ? 5 : 4;
                const int64_t origin = (mode == SM_NDIMAGE_WRAP) ? -1 : 0;
                const uint32_t axis = static_cast<uint32_t>(mode % shape.size());
                const double cval = -3.25;

                sm_ndimage_pass pass{};
                pass.struct_size = sizeof(pass);
                pass.abi_version = SM_NDIMAGE_ABI_VERSION;
                pass.kind = is_max ? SM_NDIMAGE_MAXIMUM : SM_NDIMAGE_MINIMUM;
                pass.axis = axis;
                pass.mode = mode;
                pass.origin = origin;
                pass.size = w;
                pass.cval = cval;

                const std::vector<double> ref64 = ref_minmax1d(data_f64, shape, axis, w, origin, mode, cval, is_max != 0);
                std::vector<double> in32_d(n);
                for (uint64_t i = 0; i < n; ++i) in32_d[i] = static_cast<double>(data_f32[i]);
                const std::vector<double> ref32 = ref_minmax1d(in32_d, shape, axis, w, origin, mode, static_cast<float>(cval), is_max != 0);

                // CPU strict f64
                sm_array_result res{};
                sm_route_info info = new_info();
                const sm_route_options cpu_opts = route_opts(SM_EXECUTION_CPU, SM_PRECISION_STRICT);
                CHECK(sm_ndimage_separable_filter(ctx, &arr64, &pass, 1, &cpu_opts, &res, &info) == SM_OK);
                CHECK(info.engine == SM_EXECUTION_CPU);
                const std::vector<double> got_cpu = read_result<double>(res, n);
                sm_buffer_release(res.buffer);
                for (uint64_t i = 0; i < n; ++i) CHECK(got_cpu[i] == ref64[i]);

                if (have_gpu) {
                    const sm_route_options gpu_opts = route_opts(SM_EXECUTION_GPU, SM_PRECISION_FAST);
                    // GPU f32: bit-exact!
                    const sm_execution_info before32 = counters();
                    info = new_info();
                    CHECK(sm_ndimage_separable_filter(ctx, &arr32, &pass, 1, &gpu_opts, &res, &info) == SM_OK);
                    const sm_execution_info after32 = counters();
                    CHECK(info.engine == SM_EXECUTION_GPU);
                    CHECK(after32.gpu_command_buffers == before32.gpu_command_buffers + 1);
                    const std::vector<float> got_f32 = read_result<float>(res, n);
                    sm_buffer_release(res.buffer);
                    for (uint64_t i = 0; i < n; ++i) CHECK(got_f32[i] == static_cast<float>(ref32[i]));

                    // GPU f64: widened from float32 (exact on float32-narrowed values)
                    const sm_execution_info before64 = counters();
                    info = new_info();
                    CHECK(sm_ndimage_separable_filter(ctx, &arr64, &pass, 1, &gpu_opts, &res, &info) == SM_OK);
                    const sm_execution_info after64 = counters();
                    CHECK(info.engine == SM_EXECUTION_GPU);
                    CHECK(after64.gpu_command_buffers == before64.gpu_command_buffers + 1);
                    const std::vector<double> got_f64 = read_result<double>(res, n);
                    sm_buffer_release(res.buffer);
                    for (uint64_t i = 0; i < n; ++i) CHECK(std::fabs(got_f64[i] - ref64[i]) <= 1e-5);
                }
            }
        }
    }

    // Large window (size = 97 > 64) exercising the 2-pass GPU van Herk path along BOTH contiguous & strided axes!
    {
        const std::vector<uint64_t> shape = {256, 256};
        const uint64_t n = product(shape);
        const std::vector<double> data_f64 = make_data(n, 777);
        std::vector<float> data_f32(n);
        for (uint64_t i = 0; i < n; ++i) data_f32[i] = static_cast<float>(data_f64[i]);
        const sm_array arr32 = describe(shape, SM_FLOAT32, data_f32.data());
        const sm_array arr64 = describe(shape, SM_FLOAT64, data_f64.data());

        for (uint32_t axis = 0; axis < 2; ++axis) {
            for (uint32_t is_max = 0; is_max <= 1; ++is_max) {
                sm_ndimage_pass pass{};
                pass.struct_size = sizeof(pass);
                pass.abi_version = SM_NDIMAGE_ABI_VERSION;
                pass.kind = is_max ? SM_NDIMAGE_MAXIMUM : SM_NDIMAGE_MINIMUM;
                pass.axis = axis;
                pass.mode = (axis == 0) ? SM_NDIMAGE_REFLECT : SM_NDIMAGE_CONSTANT;
                pass.origin = 11;
                pass.size = 97;
                pass.cval = 2.5;

                sm_array_result res_cpu{};
                const sm_route_options cpu_opts = route_opts(SM_EXECUTION_CPU, SM_PRECISION_STRICT);
                CHECK(sm_ndimage_separable_filter(ctx, &arr32, &pass, 1, &cpu_opts, &res_cpu, nullptr) == SM_OK);
                const std::vector<float> ref32 = read_result<float>(res_cpu, n);
                sm_buffer_release(res_cpu.buffer);

                if (have_gpu) {
                    const sm_route_options gpu_opts = route_opts(SM_EXECUTION_GPU, SM_PRECISION_FAST);
                    const sm_execution_info before = counters();
                    sm_array_result res_gpu{};
                    sm_route_info info = new_info();
                    CHECK(sm_ndimage_separable_filter(ctx, &arr32, &pass, 1, &gpu_opts, &res_gpu, &info) == SM_OK);
                    const sm_execution_info after = counters();
                    CHECK(info.engine == SM_EXECUTION_GPU);
                    // Both van Herk passes MUST live inside ONE command buffer!
                    CHECK(after.gpu_command_buffers == before.gpu_command_buffers + 1);
                    const std::vector<float> got32 = read_result<float>(res_gpu, n);
                    sm_buffer_release(res_gpu.buffer);
                    for (uint64_t i = 0; i < n; ++i) CHECK(got32[i] == ref32[i]);

                    // Also test f64 output widening on 2-pass van Herk
                    const sm_execution_info before64 = counters();
                    CHECK(sm_ndimage_separable_filter(ctx, &arr64, &pass, 1, &gpu_opts, &res_gpu, &info) == SM_OK);
                    CHECK(counters().gpu_command_buffers == before64.gpu_command_buffers + 1);
                    const std::vector<double> got64 = read_result<double>(res_gpu, n);
                    sm_buffer_release(res_gpu.buffer);
                    for (uint64_t i = 0; i < n; ++i) CHECK(std::fabs(got64[i] - static_cast<double>(ref32[i])) <= 1e-5);
                }
            }
        }
    }
}

void test_composite_morphology() {
    const std::vector<uint64_t> shape = {41, 53};
    const uint64_t n = product(shape);
    const std::vector<double> data = make_data(n, 42);
    const sm_array arr = describe(shape, SM_FLOAT64, data.data());

    sm_ndimage_pass dil_p[2]{};
    sm_ndimage_pass ero_p[2]{};
    for (uint32_t ax = 0; ax < 2; ++ax) {
        dil_p[ax].struct_size = sizeof(sm_ndimage_pass);
        dil_p[ax].abi_version = SM_NDIMAGE_ABI_VERSION;
        dil_p[ax].kind = SM_NDIMAGE_MAXIMUM;
        dil_p[ax].axis = ax;
        dil_p[ax].mode = SM_NDIMAGE_REFLECT;
        dil_p[ax].size = 5;
        ero_p[ax] = dil_p[ax];
        ero_p[ax].kind = SM_NDIMAGE_MINIMUM;
    }
    sm_ndimage_chain chains[2]{};
    chains[0].struct_size = sizeof(sm_ndimage_chain);
    chains[0].abi_version = SM_NDIMAGE_ABI_VERSION;
    chains[0].pass_count = 2;
    chains[0].passes = dil_p;
    chains[1].struct_size = sizeof(sm_ndimage_chain);
    chains[1].abi_version = SM_NDIMAGE_ABI_VERSION;
    chains[1].pass_count = 2;
    chains[1].passes = ero_p;

    const std::vector<double> dil = ref_minmax1d(ref_minmax1d(data, shape, 0, 5, 0, SM_NDIMAGE_REFLECT, 0.0, true),
                                                 shape, 1, 5, 0, SM_NDIMAGE_REFLECT, 0.0, true);
    const std::vector<double> ero = ref_minmax1d(ref_minmax1d(data, shape, 0, 5, 0, SM_NDIMAGE_REFLECT, 0.0, false),
                                                 shape, 1, 5, 0, SM_NDIMAGE_REFLECT, 0.0, false);

    for (uint32_t op : {uint32_t{SM_NDIMAGE_COMBINE_SUB}, uint32_t{SM_NDIMAGE_COMBINE_MORPH_LAPLACE}}) {
        sm_array_result res{};
        const sm_route_options cpu_opts = route_opts(SM_EXECUTION_CPU, SM_PRECISION_STRICT);
        CHECK(sm_ndimage_composite_filter(ctx, &arr, chains, 2, op, 0, &cpu_opts, &res, nullptr) == SM_OK);
        const std::vector<double> got_cpu = read_result<double>(res, n);
        sm_buffer_release(res.buffer);
        for (uint64_t i = 0; i < n; ++i) {
            const double expected = (op == SM_NDIMAGE_COMBINE_SUB) ? (dil[i] - ero[i]) : ((dil[i] + ero[i]) - data[i] - data[i]);
            CHECK(got_cpu[i] == expected);
        }
        if (have_gpu) {
            const sm_route_options gpu_opts = route_opts(SM_EXECUTION_GPU, SM_PRECISION_FAST);
            const sm_execution_info before = counters();
            sm_route_info info = new_info();
            CHECK(sm_ndimage_composite_filter(ctx, &arr, chains, 2, op, 0, &gpu_opts, &res, &info) == SM_OK);
            CHECK(info.engine == SM_EXECUTION_GPU);
            CHECK(counters().gpu_command_buffers == before.gpu_command_buffers + 1);
            const std::vector<double> got_gpu = read_result<double>(res, n);
            sm_buffer_release(res.buffer);
            for (uint64_t i = 0; i < n; ++i) {
                CHECK(std::fabs(got_gpu[i] - got_cpu[i]) <= 1e-4);
            }
        }
    }
}

void test_dense_minmax() {
    // 2D cross footprint + 3D diamond footprint, both flat and non-flat.
    const std::vector<uint64_t> shape = {23, 29};
    const uint64_t n = product(shape);
    const std::vector<double> data = make_data(n, 101);
    std::vector<float> data32(n);
    for (uint64_t i = 0; i < n; ++i) data32[i] = static_cast<float>(data[i]);
    const sm_array arr64 = describe(shape, SM_FLOAT64, data.data());
    const sm_array arr32 = describe(shape, SM_FLOAT32, data32.data());

    const std::vector<uint32_t> fshape = {3, 3};
    const std::vector<uint8_t> fp = {0, 1, 0, 1, 1, 1, 0, 1, 0};
    const std::vector<double> st = {0.0, 0.5, 0.0, -0.25, 1.0, -0.25, 0.0, 0.5, 0.0};
    const std::vector<int64_t> origins = {-1, 1};

    for (uint32_t mode = 0; mode <= SM_NDIMAGE_CONSTANT; ++mode) {
        for (uint32_t is_max = 0; is_max <= 1; ++is_max) {
            for (int nonflat = 0; nonflat <= 1; ++nonflat) {
                const double cval = 1.75;
                const std::vector<double> ref = ref_dense_minmax(data, shape, fshape, origins, fp,
                                                                 nonflat ? &st : nullptr, mode, cval, is_max != 0);
                sm_ndimage_minmax_options opts{};
                opts.struct_size = sizeof(opts);
                opts.abi_version = SM_NDIMAGE_ABI_VERSION;
                opts.is_max = is_max;
                opts.mode = mode;
                opts.fshape[0] = fshape[0];
                opts.fshape[1] = fshape[1];
                opts.origins[0] = origins[0];
                opts.origins[1] = origins[1];
                opts.footprint = fp.data();
                opts.structure = nonflat ? st.data() : nullptr;
                opts.cval = cval;

                sm_array_result res{};
                const sm_route_options cpu_opts = route_opts(SM_EXECUTION_CPU, SM_PRECISION_STRICT);
                CHECK(sm_ndimage_minmax_filter(ctx, &arr64, &opts, &cpu_opts, &res, nullptr) == SM_OK);
                const std::vector<double> got_cpu = read_result<double>(res, n);
                sm_buffer_release(res.buffer);
                for (uint64_t i = 0; i < n; ++i) CHECK(got_cpu[i] == ref[i]);

                if (have_gpu) {
                    const sm_route_options gpu_opts = route_opts(SM_EXECUTION_GPU, SM_PRECISION_FAST);
                    const sm_execution_info before = counters();
                    sm_route_info info = new_info();
                    CHECK(sm_ndimage_minmax_filter(ctx, &arr64, &opts, &gpu_opts, &res, &info) == SM_OK);
                    CHECK(info.engine == SM_EXECUTION_GPU);
                    CHECK(counters().gpu_command_buffers == before.gpu_command_buffers + 1);
                    const std::vector<double> got_gpu = read_result<double>(res, n);
                    sm_buffer_release(res.buffer);
                    for (uint64_t i = 0; i < n; ++i) CHECK(std::fabs(got_gpu[i] - ref[i]) <= 1e-5);

                    // Exact float32 check for flat
                    if (!nonflat) {
                        CHECK(sm_ndimage_minmax_filter(ctx, &arr32, &opts, &gpu_opts, &res, &info) == SM_OK);
                        const std::vector<float> got_f32 = read_result<float>(res, n);
                        sm_buffer_release(res.buffer);
                        CHECK(sm_ndimage_minmax_filter(ctx, &arr32, &opts, &cpu_opts, &res, nullptr) == SM_OK);
                        const std::vector<float> ref_f32 = read_result<float>(res, n);
                        sm_buffer_release(res.buffer);
                        for (uint64_t i = 0; i < n; ++i) CHECK(got_f32[i] == ref_f32[i]);
                    }
                }
            }
        }
    }
}

void test_rank_and_median() {
    // 1. Exhaustive 0-1 Principle verification of GPU 3x3 optimal sorting network (all 512 binary 3x3 patches)!
    if (have_gpu) {
        // Pack all 512 3x3 patches side-by-side into a (3, 3*512) image and check centers!
        const uint64_t H = 3, W = 3 * 512;
        std::vector<float> img(H * W, 0.0f);
        for (uint32_t mask = 0; mask < 512; ++mask) {
            for (uint32_t dy = 0; dy < 3; ++dy) {
                for (uint32_t dx = 0; dx < 3; ++dx) {
                    const uint32_t bit = dy * 3 + dx;
                    img[dy * W + (mask * 3 + dx)] = (mask & (1u << bit)) ? 1.0f : 0.0f;
                }
            }
        }
        const std::vector<uint64_t> shape = {H, W};
        const sm_array arr = describe(shape, SM_FLOAT32, img.data());
        sm_ndimage_rank_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = SM_NDIMAGE_ABI_VERSION;
        opts.rank = 4;
        opts.mode = SM_NDIMAGE_REFLECT;
        opts.fshape[0] = 3;
        opts.fshape[1] = 3;
        const sm_route_options gpu_opts = route_opts(SM_EXECUTION_GPU, SM_PRECISION_FAST);
        sm_array_result res{};
        sm_route_info info = new_info();
        CHECK(sm_ndimage_rank_filter(ctx, &arr, &opts, &gpu_opts, &res, &info) == SM_OK);
        CHECK(info.engine == SM_EXECUTION_GPU);
        const std::vector<float> got = read_result<float>(res, H * W);
        sm_buffer_release(res.buffer);
        for (uint32_t mask = 0; mask < 512; ++mask) {
            const int popcnt = __builtin_popcount(mask);
            const float expected = (popcnt >= 5) ? 1.0f : 0.0f;
            CHECK(got[1 * W + (mask * 3 + 1)] == expected);
        }
    }

    // 2. Dedicated 2-D tiled 3x3, 5x5, 7x7 medians across all 5 boundary modes & odd shapes.
    const std::vector<std::vector<uint64_t>> shapes2d = {{1, 1}, {1, 7}, {7, 1}, {17, 19}, {33, 47}};
    for (const auto& shape : shapes2d) {
        const uint64_t n = product(shape);
        const std::vector<double> data = make_data(n, static_cast<uint32_t>(n + 13));
        std::vector<float> data32(n);
        for (uint64_t i = 0; i < n; ++i) data32[i] = static_cast<float>(data[i]);
        const sm_array arr64 = describe(shape, SM_FLOAT64, data.data());
        const sm_array arr32 = describe(shape, SM_FLOAT32, data32.data());

        for (uint32_t k : {3u, 5u, 7u}) {
            for (uint32_t mode = 0; mode <= SM_NDIMAGE_CONSTANT; ++mode) {
                const uint32_t rank = (k * k) / 2;
                const int64_t oy = (k >= 5 && mode == SM_NDIMAGE_WRAP) ? -2 : 0;
                const int64_t ox = (mode == SM_NDIMAGE_MIRROR) ? 1 : 0;
                const double cval = -4.5;
                const std::vector<double> ref = ref_rank(data, shape, {k, k}, {oy, ox}, {}, rank, mode, cval);

                sm_ndimage_rank_options opts{};
                opts.struct_size = sizeof(opts);
                opts.abi_version = SM_NDIMAGE_ABI_VERSION;
                opts.rank = rank;
                opts.mode = mode;
                opts.fshape[0] = k;
                opts.fshape[1] = k;
                opts.origins[0] = oy;
                opts.origins[1] = ox;
                opts.cval = cval;

                sm_array_result res{};
                const sm_route_options cpu_opts = route_opts(SM_EXECUTION_CPU, SM_PRECISION_STRICT);
                CHECK(sm_ndimage_rank_filter(ctx, &arr64, &opts, &cpu_opts, &res, nullptr) == SM_OK);
                const std::vector<double> got_cpu = read_result<double>(res, n);
                sm_buffer_release(res.buffer);
                for (uint64_t i = 0; i < n; ++i) CHECK(got_cpu[i] == ref[i]);

                if (have_gpu) {
                    const sm_route_options gpu_opts = route_opts(SM_EXECUTION_GPU, SM_PRECISION_FAST);
                    const sm_execution_info before = counters();
                    sm_route_info info = new_info();
                    CHECK(sm_ndimage_rank_filter(ctx, &arr64, &opts, &gpu_opts, &res, &info) == SM_OK);
                    CHECK(info.engine == SM_EXECUTION_GPU);
                    CHECK(counters().gpu_command_buffers == before.gpu_command_buffers + 1);
                    const std::vector<double> got_gpu = read_result<double>(res, n);
                    sm_buffer_release(res.buffer);
                    for (uint64_t i = 0; i < n; ++i) CHECK(std::fabs(got_gpu[i] - ref[i]) <= 1e-5);

                    // Bit-exact on float32!
                    CHECK(sm_ndimage_rank_filter(ctx, &arr32, &opts, &gpu_opts, &res, &info) == SM_OK);
                    const std::vector<float> got_f32 = read_result<float>(res, n);
                    sm_buffer_release(res.buffer);
                    CHECK(sm_ndimage_rank_filter(ctx, &arr32, &opts, &cpu_opts, &res, nullptr) == SM_OK);
                    const std::vector<float> ref_f32 = read_result<float>(res, n);
                    sm_buffer_release(res.buffer);
                    for (uint64_t i = 0; i < n; ++i) CHECK(got_f32[i] == ref_f32[i]);
                }
            }
        }
    }

    // 3. General Forgetful Selection: small (3D 3x3x3 + 2D non-square + arbitrary footprint + asymmetric ranks)
    //    AND medium (9x9 = 81 taps, rank = 40, s = 42; 11x11 = 121 taps, rank = 60, s = 62)!
    {
        // 3D 3x3x3 cube with various ranks (0, 1, 5, 13, 21, 25, 26)
        const std::vector<uint64_t> shape3d = {7, 9, 11};
        const uint64_t n3 = product(shape3d);
        const std::vector<double> data3 = make_data(n3, 303);
        std::vector<float> data3_f32(n3);
        for (uint64_t i = 0; i < n3; ++i) data3_f32[i] = static_cast<float>(data3[i]);
        const sm_array arr3 = describe(shape3d, SM_FLOAT32, data3_f32.data());
        for (uint32_t r : {0u, 1u, 5u, 13u, 21u, 25u, 26u}) {
            sm_ndimage_rank_options opts{};
            opts.struct_size = sizeof(opts);
            opts.abi_version = SM_NDIMAGE_ABI_VERSION;
            opts.rank = r;
            opts.mode = SM_NDIMAGE_REFLECT;
            opts.fshape[0] = 3;
            opts.fshape[1] = 3;
            opts.fshape[2] = 3;
            sm_array_result res_cpu{};
            const sm_route_options cpu_opts = route_opts(SM_EXECUTION_CPU, SM_PRECISION_STRICT);
            CHECK(sm_ndimage_rank_filter(ctx, &arr3, &opts, &cpu_opts, &res_cpu, nullptr) == SM_OK);
            const std::vector<float> ref = read_result<float>(res_cpu, n3);
            sm_buffer_release(res_cpu.buffer);

            if (have_gpu) {
                const sm_route_options gpu_opts = route_opts(SM_EXECUTION_GPU, SM_PRECISION_FAST);
                const sm_execution_info before = counters();
                sm_array_result res_gpu{};
                sm_route_info info = new_info();
                CHECK(sm_ndimage_rank_filter(ctx, &arr3, &opts, &gpu_opts, &res_gpu, &info) == SM_OK);
                CHECK(info.engine == SM_EXECUTION_GPU);
                CHECK(counters().gpu_command_buffers == before.gpu_command_buffers + 1);
                const std::vector<float> got = read_result<float>(res_gpu, n3);
                sm_buffer_release(res_gpu.buffer);
                for (uint64_t i = 0; i < n3; ++i) CHECK(got[i] == ref[i]);
            }
        }
    }

    // Medium Forgetful Selection (9x9 and 11x11) + exceeding GPU limit (13x13 -> s=86 > 64)
    {
        const std::vector<uint64_t> shape = {25, 27};
        const uint64_t n = product(shape);
        const std::vector<double> data = make_data(n, 909);
        std::vector<float> data32(n);
        for (uint64_t i = 0; i < n; ++i) data32[i] = static_cast<float>(data[i]);
        const sm_array arr32 = describe(shape, SM_FLOAT32, data32.data());
        for (uint32_t k : {9u, 11u}) {
            sm_ndimage_rank_options opts{};
            opts.struct_size = sizeof(opts);
            opts.abi_version = SM_NDIMAGE_ABI_VERSION;
            opts.rank = (k * k) / 2;
            opts.mode = SM_NDIMAGE_NEAREST;
            opts.fshape[0] = k;
            opts.fshape[1] = k;
            sm_array_result res_cpu{};
            const sm_route_options cpu_opts = route_opts(SM_EXECUTION_CPU, SM_PRECISION_STRICT);
            CHECK(sm_ndimage_rank_filter(ctx, &arr32, &opts, &cpu_opts, &res_cpu, nullptr) == SM_OK);
            const std::vector<float> ref = read_result<float>(res_cpu, n);
            sm_buffer_release(res_cpu.buffer);

            if (have_gpu) {
                const sm_route_options gpu_opts = route_opts(SM_EXECUTION_GPU, SM_PRECISION_FAST);
                sm_array_result res_gpu{};
                sm_route_info info = new_info();
                CHECK(sm_ndimage_rank_filter(ctx, &arr32, &opts, &gpu_opts, &res_gpu, &info) == SM_OK);
                CHECK(info.engine == SM_EXECUTION_GPU);
                const std::vector<float> got = read_result<float>(res_gpu, n);
                sm_buffer_release(res_gpu.buffer);
                for (uint64_t i = 0; i < n; ++i) CHECK(got[i] == ref[i]);
            }
        }
        // 13x13 median: s = 86 > 64 -> forced GPU returns SM_ENOTSUP, AUTO runs on CPU.
        {
            sm_ndimage_rank_options opts{};
            opts.struct_size = sizeof(opts);
            opts.abi_version = SM_NDIMAGE_ABI_VERSION;
            opts.rank = 84;
            opts.mode = SM_NDIMAGE_REFLECT;
            opts.fshape[0] = 13;
            opts.fshape[1] = 13;
            sm_array_result res{};
            if (have_gpu) {
                const sm_route_options gpu_opts = route_opts(SM_EXECUTION_GPU, SM_PRECISION_FAST);
                CHECK(sm_ndimage_rank_filter(ctx, &arr32, &opts, &gpu_opts, &res, nullptr) == SM_ENOTSUP);
            }
            const sm_route_options auto_opts = route_opts(SM_EXECUTION_AUTO, SM_PRECISION_FAST);
            sm_route_info info = new_info();
            CHECK(sm_ndimage_rank_filter(ctx, &arr32, &opts, &auto_opts, &res, &info) == SM_OK);
            CHECK(info.engine == SM_EXECUTION_CPU);
            sm_buffer_release(res.buffer);
        }
    }
}

void test_validation_and_guardrails() {
    const std::vector<uint64_t> shape = {8, 8};
    const std::vector<double> data(64, 1.0);
    const sm_array arr = describe(shape, SM_FLOAT64, data.data());
    const sm_route_options cpu_opts = route_opts(SM_EXECUTION_CPU, SM_PRECISION_STRICT);
    sm_array_result res{};

    // All-zero footprint -> SM_EINVAL
    {
        const std::vector<uint8_t> zeros(9, 0);
        sm_ndimage_minmax_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = SM_NDIMAGE_ABI_VERSION;
        opts.fshape[0] = 3;
        opts.fshape[1] = 3;
        opts.footprint = zeros.data();
        CHECK(sm_ndimage_minmax_filter(ctx, &arr, &opts, &cpu_opts, &res, nullptr) == SM_EINVAL);
    }
    // rank >= active_taps -> SM_EINVAL
    {
        sm_ndimage_rank_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = SM_NDIMAGE_ABI_VERSION;
        opts.rank = 9;
        opts.fshape[0] = 3;
        opts.fshape[1] = 3;
        CHECK(sm_ndimage_rank_filter(ctx, &arr, &opts, &cpu_opts, &res, nullptr) == SM_EINVAL);
    }
    // Non-null weights on SM_NDIMAGE_MINIMUM -> SM_EINVAL
    {
        double w[3] = {1.0, 1.0, 1.0};
        sm_ndimage_pass pass{};
        pass.struct_size = sizeof(pass);
        pass.abi_version = SM_NDIMAGE_ABI_VERSION;
        pass.kind = SM_NDIMAGE_MINIMUM;
        pass.size = 3;
        pass.weights = w;
        CHECK(sm_ndimage_separable_filter(ctx, &arr, &pass, 1, &cpu_opts, &res, nullptr) == SM_EINVAL);
    }
    // Float64 overflow guard rail on GPU rank filter: forced GPU returns SM_ENOTSUP, AUTO falls back to CPU.
    if (have_gpu) {
        std::vector<double> huge(64, 1e300);
        const sm_array arr_huge = describe(shape, SM_FLOAT64, huge.data());
        sm_ndimage_rank_options opts{};
        opts.struct_size = sizeof(opts);
        opts.abi_version = SM_NDIMAGE_ABI_VERSION;
        opts.rank = 4;
        opts.fshape[0] = 3;
        opts.fshape[1] = 3;
        const sm_route_options gpu_opts = route_opts(SM_EXECUTION_GPU, SM_PRECISION_FAST);
        CHECK(sm_ndimage_rank_filter(ctx, &arr_huge, &opts, &gpu_opts, &res, nullptr) == SM_ENOTSUP);
    }
}

} // namespace

int main() {
    sm_context_options copts{};
    copts.struct_size = sizeof(copts);
    copts.abi_version = SM_RUNTIME_ABI_VERSION;
    copts.storage_kind = (sm_array_capabilities() & SM_CAPABILITY_METAL) ? SM_BUFFER_METAL : SM_BUFFER_CPU;
    copts.flags = SM_CONTEXT_POISON_UNWRITTEN;
    copts.memory_limit_bytes = uint64_t{2} << 30;
    CHECK(sm_context_create(&copts, &ctx) == SM_OK);
    have_gpu = (sm_array_capabilities() & SM_CAPABILITY_METAL) != 0;

    test_separable_minmax();
    test_composite_morphology();
    test_dense_minmax();
    test_rank_and_median();
    test_validation_and_guardrails();

    sm_context_release(ctx);
    std::printf("ndimage morphology native tests passed (gpu=%d)\n", have_gpu ? 1 : 0);
    return 0;
}
