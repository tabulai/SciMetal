// ms_ndimage_separable_filter through the C ABI: the CPU engine against a
// direct evaluation of SciPy's definition, the forced-GPU engine against the
// CPU engine, argument validation, routing and the execution counters.
//
// Runs plain and under the Metal API and shader validators. A build without
// Metal (no MS_TEST_METAL) runs the CPU engine's checks on CPU storage.
#include "ms_ndimage.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "ndimage filters test failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

namespace {
ms_context* ctx = nullptr;
bool have_gpu = false;
unsigned long cpu_cases = 0, gpu_cases = 0;
double worst_cpu = 0.0, worst_gpu = 0.0;  // largest error / bound seen

struct Spec {
    uint32_t kind = MS_NDIMAGE_CORRELATE;
    uint32_t axis = 0;
    uint32_t mode = MS_NDIMAGE_REFLECT;
    int64_t origin = 0;
    std::vector<double> weights;  // correlate
    uint64_t size = 0;            // uniform
    double cval = 0.0;
    uint64_t taps() const { return kind == MS_NDIMAGE_UNIFORM ? size : weights.size(); }
};

uint64_t product(const std::vector<uint64_t>& shape) {
    uint64_t n = 1;
    for (uint64_t extent : shape) n *= extent;
    return n;
}

// Deterministic values in [-1, 1) with an offset, so sums do not cancel to nothing.
std::vector<double> make_data(uint64_t n, uint32_t seed) {
    std::vector<double> values(n);
    uint64_t state = 0x9E3779B97F4A7C15ull * (seed + 1);
    for (double& v : values) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        v = static_cast<double>(state >> 11) / 9007199254740992.0 * 2.0 - 1.0 + 0.25;
    }
    return values;
}

// The line position an extension position repeats (-1: the constant), found
// by folding the position back step by step rather than by the library's
// modular arithmetic.
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

// SciPy's definition of one pass, evaluated tap by tap. Float32 arrays round
// each pass's result to float32, as storing it does.
std::vector<double> reference_pass(const std::vector<double>& in, const std::vector<uint64_t>& shape,
                                   const Spec& spec, bool single) {
    const int64_t length = static_cast<int64_t>(shape[spec.axis]);
    uint64_t inner = 1, outer = 1;
    for (size_t axis = 0; axis < spec.axis; ++axis) outer *= shape[axis];
    for (size_t axis = spec.axis + 1; axis < shape.size(); ++axis) inner *= shape[axis];
    const int64_t taps = static_cast<int64_t>(spec.taps());
    const int64_t first = -(taps / 2) - spec.origin;
    std::vector<double> out(in.size());
    for (uint64_t o = 0; o < outer; ++o) {
        for (uint64_t c = 0; c < inner; ++c) {
            const double* line = in.data() + o * static_cast<uint64_t>(length) * inner + c;
            for (int64_t p = 0; p < length; ++p) {
                double sum = 0.0;
                for (int64_t j = 0; j < taps; ++j) {
                    const int64_t source = fold(p + first + j, length, spec.mode);
                    const double value = source < 0 ? spec.cval : line[static_cast<uint64_t>(source) * inner];
                    sum += spec.kind == MS_NDIMAGE_UNIFORM ? value : spec.weights[static_cast<size_t>(j)] * value;
                }
                if (spec.kind == MS_NDIMAGE_UNIFORM) sum /= static_cast<double>(taps);
                out[o * static_cast<uint64_t>(length) * inner + c + static_cast<uint64_t>(p) * inner] =
                    single ? static_cast<double>(static_cast<float>(sum)) : sum;
            }
        }
    }
    return out;
}

// Error budget of a call in arithmetic with unit roundoff `unit`. A bound m on
// the magnitude of the data and a bound e on its error go through the passes:
//   m' = max(m, |cval|), e' = max(e, unit * |cval|)   for a constant boundary,
//   e  = S * e' + (taps + 4) * unit * S * m',   m = S * m',   S = sum|weights|.
// taps + 2 covers a dot product of `taps` terms in any order with rounded
// weights and a pair sum (Higham, gamma_n ~ n u); one more unit is the rounding
// of a stored float32 result and one is slack for second-order terms. The
// input starts with e = unit * m (its narrowing). `running` adds the growth of
// the CPU engine's uniform running sum along a line (two roundings per step).
double budget(const std::vector<double>& data, const std::vector<uint64_t>& shape, const std::vector<Spec>& specs,
              double unit, bool running) {
    double magnitude = 0.0;
    for (double v : data) magnitude = std::max(magnitude, std::fabs(v));
    double error = unit * magnitude;
    for (const Spec& spec : specs) {
        if (spec.mode == MS_NDIMAGE_CONSTANT) {
            magnitude = std::max(magnitude, std::fabs(spec.cval));
            error = std::max(error, unit * std::fabs(spec.cval));
        }
        double gain = 0.0;
        for (double w : spec.weights) gain += std::fabs(w);
        if (spec.kind == MS_NDIMAGE_UNIFORM) gain = 1.0;
        double terms = static_cast<double>(spec.taps()) + 4.0;
        if (running && spec.kind == MS_NDIMAGE_UNIFORM) terms += 2.0 * static_cast<double>(shape[spec.axis]);
        error = gain * error + terms * unit * gain * magnitude;
        magnitude *= gain;
    }
    return error;
}

struct Outcome {
    ms_status status = MS_OK;
    ms_route_info info{};
    std::vector<double> values;
};

// The passes point at the specs' weights: `specs` must outlive them.
std::vector<ms_ndimage_pass> to_passes(const std::vector<Spec>& specs) {
    std::vector<ms_ndimage_pass> passes(specs.size());
    for (size_t i = 0; i < specs.size(); ++i) {
        ms_ndimage_pass& pass = passes[i];
        pass.struct_size = sizeof(pass);
        pass.abi_version = MS_NDIMAGE_ABI_VERSION;
        pass.kind = specs[i].kind;
        pass.axis = specs[i].axis;
        pass.mode = specs[i].mode;
        pass.origin = specs[i].origin;
        pass.size = specs[i].taps();
        pass.weights = specs[i].kind == MS_NDIMAGE_UNIFORM ? nullptr : specs[i].weights.data();
        pass.cval = specs[i].cval;
    }
    return passes;
}

ms_array describe(const std::vector<uint64_t>& shape, uint32_t dtype, const void* data) {
    ms_array array{};
    array.struct_size = sizeof(array);
    array.abi_version = MS_ARRAY_ABI_VERSION;
    array.data = data;
    array.dtype = dtype;
    array.ndim = static_cast<uint32_t>(shape.size());
    for (size_t axis = 0; axis < shape.size(); ++axis) array.shape[axis] = shape[axis];
    return array;
}

ms_route_options route(uint32_t execution, uint32_t precision) {
    ms_route_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = MS_ARRAY_ABI_VERSION;
    options.execution = execution;
    options.precision = precision;
    return options;
}

// Runs the filter on `data` stored as `dtype` and returns the result as doubles.
Outcome run(const std::vector<double>& data, const std::vector<uint64_t>& shape, uint32_t dtype,
            const std::vector<Spec>& specs, uint32_t execution, uint32_t precision = MS_PRECISION_FAST) {
    std::vector<float> narrow;
    const void* pointer = data.data();
    if (dtype == MS_FLOAT32) {
        narrow.assign(data.begin(), data.end());
        pointer = narrow.data();
    }
    const ms_array input = describe(shape, dtype, pointer);
    const std::vector<ms_ndimage_pass> passes = to_passes(specs);
    const ms_route_options options = route(execution, precision);
    Outcome outcome;
    outcome.info.struct_size = sizeof(outcome.info);
    outcome.info.abi_version = MS_ARRAY_ABI_VERSION;
    ms_array_result result{};
    outcome.status = ms_ndimage_separable_filter(ctx, &input, passes.data(), static_cast<uint32_t>(passes.size()),
                                                 &options, &result, &outcome.info);
    if (outcome.status != MS_OK) return outcome;
    CHECK(result.buffer && result.dtype == dtype && result.ndim == shape.size());
    for (size_t axis = 0; axis < shape.size(); ++axis) CHECK(result.shape[axis] == shape[axis]);
    const void* bytes = nullptr;
    uint64_t size = 0;
    CHECK(ms_buffer_data(result.buffer, &bytes, &size) == MS_OK);
    CHECK(size == data.size() * (dtype == MS_FLOAT32 ? 4 : 8));
    outcome.values.resize(data.size());
    if (dtype == MS_FLOAT32) {
        const float* values = static_cast<const float*>(bytes);
        for (size_t i = 0; i < data.size(); ++i) outcome.values[i] = values[i];
    } else if (!data.empty()) {
        std::memcpy(outcome.values.data(), bytes, static_cast<size_t>(size));
    }
    ms_buffer_release(result.buffer);
    return outcome;
}

double max_difference(const std::vector<double>& a, const std::vector<double>& b) {
    CHECK(a.size() == b.size());
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (!(d <= worst)) worst = d;  // NaN counts as a failure
    }
    return worst;
}

const double unit32 = std::ldexp(1.0, -24), unit64 = std::ldexp(1.0, -53);

// CPU engine against the reference; forced GPU engine against the CPU engine,
// for float32 and float64 storage of the same values.
void compare(const std::vector<uint64_t>& shape, const std::vector<Spec>& specs, uint32_t seed) {
    const std::vector<double> exact = make_data(product(shape), seed);
    for (uint32_t dtype : {MS_FLOAT64, MS_FLOAT32}) {
        const bool single = dtype == MS_FLOAT32;
        std::vector<double> data = exact;
        if (single) for (double& v : data) v = static_cast<double>(static_cast<float>(v));
        std::vector<double> expected = data;
        for (const Spec& spec : specs) expected = reference_pass(expected, shape, spec, single);

        const Outcome cpu = run(data, shape, dtype, specs, MS_EXECUTION_CPU);
        CHECK(cpu.status == MS_OK);
        CHECK(cpu.info.engine == MS_EXECUTION_CPU && cpu.info.compute_dtype == MS_FLOAT64);
        const double cpu_bound = budget(data, shape, specs, single ? unit32 : unit64, true);
        const double cpu_error = max_difference(cpu.values, expected);
        if (!(cpu_error <= cpu_bound)) {
            std::fprintf(stderr, "CPU engine: error %.3g exceeds %.3g (dtype %u, %zu passes, first axis %u mode %u taps %llu)\n",
                         cpu_error, cpu_bound, dtype, specs.size(), specs[0].axis, specs[0].mode,
                         static_cast<unsigned long long>(specs[0].taps()));
            CHECK(false);
        }
        if (cpu_bound > 0) worst_cpu = std::max(worst_cpu, cpu_error / cpu_bound);
        ++cpu_cases;

        if (!have_gpu) continue;
        const Outcome gpu = run(data, shape, dtype, specs, MS_EXECUTION_GPU);
        CHECK(gpu.status == MS_OK);
        CHECK(gpu.info.engine == MS_EXECUTION_GPU && gpu.info.compute_dtype == MS_FLOAT32);
        const double gpu_bound = budget(data, shape, specs, unit32, false);
        const double gpu_error = max_difference(gpu.values, cpu.values);
        if (!(gpu_error <= gpu_bound)) {
            std::fprintf(stderr, "GPU engine: error %.3g exceeds %.3g (dtype %u, %zu passes, first axis %u mode %u taps %llu origin %lld)\n",
                         gpu_error, gpu_bound, dtype, specs.size(), specs[0].axis, specs[0].mode,
                         static_cast<unsigned long long>(specs[0].taps()), static_cast<long long>(specs[0].origin));
            CHECK(false);
        }
        if (gpu_bound > 0) worst_gpu = std::max(worst_gpu, gpu_error / gpu_bound);
        ++gpu_cases;
    }
}

std::vector<double> gaussian(int radius, double sigma, int order) {
    std::vector<double> w(static_cast<size_t>(2 * radius + 1));
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) sum += w[static_cast<size_t>(i + radius)] = std::exp(-0.5 * i * i / (sigma * sigma));
    for (int i = -radius; i <= radius; ++i) {
        double& v = w[static_cast<size_t>(i + radius)];
        v /= sum;
        if (order == 1) v *= -i / (sigma * sigma);  // odd: antisymmetric
    }
    return w;
}

std::vector<double> ramp(size_t n) {  // asymmetric, mixed signs
    std::vector<double> w(n);
    for (size_t i = 0; i < n; ++i) w[i] = (static_cast<double>(i) - 0.3 * static_cast<double>(n)) / static_cast<double>(n) + 0.125;
    return w;
}

// The extreme origins of a window and the centred one.
std::vector<int64_t> origins_of(uint64_t taps) {
    const int64_t size = static_cast<int64_t>(taps);
    std::vector<int64_t> origins = {-(size / 2), 0, (size - 1) / 2};  // ascending
    origins.erase(std::unique(origins.begin(), origins.end()), origins.end());
    return origins;
}

void test_grid() {
    const std::vector<std::vector<uint64_t>> shapes = {
        {1}, {2}, {7}, {1, 1}, {1, 37}, {37, 1}, {5, 3}, {33, 70},        // tiny and thin
        {17, 4, 9}, {3, 40, 5}, {2, 3, 4, 5}, {3, 1, 2, 1, 2},
        {2, 1, 2, 1, 2, 1, 2, 3},                                         // rank 8
        {1031},                                                           // one line, not a multiple of any threadgroup
        {67, 129}, {9, 33, 65},                                           // straddle 32 x 32 and 64-wide groups
    };
    const std::vector<std::vector<double>> weight_sets = {
        {2.5},                        // one tap
        {0.25, 0.5, 0.25},            // symmetric
        {-1.0, 0.0, 1.0},             // antisymmetric
        {0.5, -1.5},                  // even length
        ramp(4), ramp(5),             // asymmetric, even and odd
        gaussian(3, 0.8, 0), gaussian(12, 3.0, 0), gaussian(12, 3.0, 1),
        ramp(41),                     // longer than most axes here
    };
    const uint32_t modes[] = {MS_NDIMAGE_NEAREST, MS_NDIMAGE_WRAP, MS_NDIMAGE_REFLECT, MS_NDIMAGE_MIRROR,
                              MS_NDIMAGE_CONSTANT};
    uint32_t seed = 0;
    for (const auto& shape : shapes) {
        for (uint32_t axis = 0; axis < shape.size(); ++axis) {
            // Every mode with every window on the small shapes; a rotating subset on the rest.
            const bool exhaustive = product(shape) <= 600;
            for (size_t w = 0; w < weight_sets.size(); ++w) {
                for (uint32_t mode : modes) {
                    if (!exhaustive && (w + mode + axis) % 3 != 0) continue;
                    for (int64_t origin : origins_of(weight_sets[w].size())) {
                        Spec spec;
                        spec.axis = axis;
                        spec.mode = mode;
                        spec.origin = origin;
                        spec.weights = weight_sets[w];
                        spec.cval = -1.75;
                        compare(shape, {spec}, ++seed);
                    }
                }
            }
            for (uint64_t size : {uint64_t{1}, uint64_t{2}, uint64_t{3}, uint64_t{8}, uint64_t{51}}) {
                for (uint32_t mode : modes) {
                    if (!exhaustive && (size + mode) % 2 != 0) continue;
                    for (int64_t origin : origins_of(size)) {
                        Spec spec;
                        spec.kind = MS_NDIMAGE_UNIFORM;
                        spec.axis = axis;
                        spec.mode = mode;
                        spec.origin = origin;
                        spec.size = size;
                        spec.cval = 3.5;
                        compare(shape, {spec}, ++seed);
                    }
                }
            }
        }
    }
}

// Several passes in one call: every axis, mixed modes and kinds, as
// gaussian_filter, uniform_filter and sobel issue them.
void test_sequences() {
    const std::vector<std::vector<uint64_t>> shapes = {{40, 50}, {9, 20, 11}, {4, 5, 6, 7}, {130}, {1, 9}};
    uint32_t seed = 1000;
    for (const auto& shape : shapes) {
        std::vector<Spec> gaussians, mixed, sobel, uniforms;
        for (uint32_t axis = 0; axis < shape.size(); ++axis) {
            Spec g;
            g.axis = axis;
            g.mode = axis % 2 ? MS_NDIMAGE_MIRROR : MS_NDIMAGE_REFLECT;
            g.weights = gaussian(8, 2.0, axis == 1 ? 1 : 0);
            gaussians.push_back(g);
            Spec m;
            m.axis = static_cast<uint32_t>(shape.size()) - 1 - axis;
            m.mode = axis % 2 ? MS_NDIMAGE_CONSTANT : MS_NDIMAGE_WRAP;
            m.cval = 0.5;
            if (axis % 2) {
                m.kind = MS_NDIMAGE_UNIFORM;
                m.size = 4;
                m.origin = -1;
            } else {
                m.weights = ramp(6);
                m.origin = 2;
            }
            mixed.push_back(m);
            Spec s;
            s.axis = axis;
            s.mode = MS_NDIMAGE_NEAREST;
            s.weights = axis == 0 ? std::vector<double>{-1.0, 0.0, 1.0} : std::vector<double>{1.0, 2.0, 1.0};
            sobel.push_back(s);
            Spec u;
            u.kind = MS_NDIMAGE_UNIFORM;
            u.axis = axis;
            u.size = 3 + axis;
            uniforms.push_back(u);
        }
        compare(shape, gaussians, ++seed);
        compare(shape, mixed, ++seed);
        compare(shape, sobel, ++seed);
        compare(shape, uniforms, ++seed);
        // The same axis twice in one call.
        compare(shape, {gaussians[0], gaussians[0], uniforms[0]}, ++seed);
    }
}

// Shapes large enough for several lanes and many threadgroups, through both
// CPU layouts (lines along the last axis, tiles along the others).
void test_larger() {
    uint32_t seed = 5000;
    const uint32_t modes[] = {MS_NDIMAGE_REFLECT, MS_NDIMAGE_CONSTANT, MS_NDIMAGE_WRAP};
    for (const std::vector<uint64_t>& shape :
         {std::vector<uint64_t>{311, 517}, {70, 90, 3}, {5, 300, 257}, {200003}, {3, 70001}}) {
        for (uint32_t axis = 0; axis < shape.size(); ++axis) {
            Spec g;
            g.axis = axis;
            g.mode = modes[axis % 3];
            g.cval = 2.0;
            g.weights = gaussian(40, 10.0, 0);  // 40 pairs: SciPy's vectorized prefix and scalar tail
            compare(shape, {g}, ++seed);
            Spec a = g;
            a.weights = ramp(24);
            a.origin = -7;
            compare(shape, {a}, ++seed);
            Spec u;
            u.kind = MS_NDIMAGE_UNIFORM;
            u.axis = axis;
            u.mode = modes[(axis + 1) % 3];
            u.size = 20;
            u.origin = 3;
            compare(shape, {u}, ++seed);
        }
    }
}

// A float64 result is the exact widening of the float32 result for the same
// float32 values: the fused store changes the representation, not the value.
void test_fused_widening_is_exact() {
    if (!have_gpu) return;
    const std::vector<uint64_t> shape = {123, 77};
    std::vector<double> data = make_data(product(shape), 7);
    for (double& v : data) v = static_cast<double>(static_cast<float>(v));
    data[5] = INFINITY;
    data[900] = -0.0;
    data[901] = 1e-42;  // subnormal as a float32
    Spec g0, g1;
    g0.axis = 0;
    g0.weights = gaussian(5, 1.5, 0);
    g1.axis = 1;
    g1.weights = {1.0};  // copies its input: every special value reaches the store
    g1.mode = MS_NDIMAGE_CONSTANT;
    for (const std::vector<Spec>& specs : {std::vector<Spec>{g1}, {g0, g1}, {g1, g0}}) {
        const Outcome single = run(data, shape, MS_FLOAT32, specs, MS_EXECUTION_GPU);
        const Outcome wide = run(data, shape, MS_FLOAT64, specs, MS_EXECUTION_GPU);
        CHECK(single.status == MS_OK && wide.status == MS_OK);
        for (size_t i = 0; i < data.size(); ++i) {
            const bool same = std::memcmp(&single.values[i], &wide.values[i], sizeof(double)) == 0;
            CHECK(same || (std::isnan(single.values[i]) && std::isnan(wide.values[i])));
        }
    }
    // The one-tap pass with weight 1 is an exact copy.
    const Outcome copy = run(data, shape, MS_FLOAT64, {g1}, MS_EXECUTION_GPU);
    CHECK(copy.status == MS_OK);
    for (size_t i = 0; i < data.size(); ++i) {
        // Subnormal float32 values may be flushed to zero by the GPU's float datapath.
        if (std::fabs(data[i]) < 1.2e-38) continue;
        CHECK(std::memcmp(&copy.values[i], &data[i], sizeof(double)) == 0);
    }
}

std::string last_error() {
    char text[512] = {};
    CHECK(ms_context_error_copy(ctx, text, sizeof(text)) == MS_OK);
    return text;
}

void test_validation() {
    const std::vector<uint64_t> shape = {6, 5};
    const std::vector<double> data = make_data(30, 1);
    Spec good;
    good.axis = 1;
    good.weights = {0.25, 0.5, 0.25};
    const std::vector<Spec> one = {good};
    const ms_array input = describe(shape, MS_FLOAT64, data.data());
    const ms_route_options options = route(MS_EXECUTION_CPU, MS_PRECISION_STRICT);

    // Each call must fail with `expected`, describe the failure and leave out and info untouched.
    const auto rejects = [&](ms_status expected, const ms_array* array, const ms_ndimage_pass* pass_pointer,
                             uint32_t count, const ms_route_options* opts, bool with_out = true) {
        ms_array_result result;
        std::memset(&result, 0x5a, sizeof(result));
        const ms_array_result before = result;
        ms_route_info info{};
        info.struct_size = sizeof(info);
        info.abi_version = MS_ARRAY_ABI_VERSION;
        info.engine = 77;
        info.compute_dtype = 78;
        const ms_status status =
            ms_ndimage_separable_filter(ctx, array, pass_pointer, count, opts, with_out ? &result : nullptr, &info);
        CHECK(status == expected);
        CHECK(std::memcmp(&result, &before, sizeof(result)) == 0);
        CHECK(info.engine == 77 && info.compute_dtype == 78);
        CHECK(!last_error().empty());
    };
    const auto with = [&](auto&& change) {
        std::vector<ms_ndimage_pass> passes = to_passes(one);
        change(passes[0]);
        return passes;
    };
    std::vector<ms_ndimage_pass> passes = to_passes(one);
    rejects(MS_EINVAL, nullptr, passes.data(), 1, &options);
    rejects(MS_EINVAL, &input, nullptr, 1, &options);
    rejects(MS_EINVAL, &input, passes.data(), 0, &options);
    rejects(MS_EINVAL, &input, passes.data(), MS_NDIMAGE_MAX_PASSES + 1, &options);
    rejects(MS_EINVAL, &input, passes.data(), 1, nullptr);
    rejects(MS_EINVAL, &input, passes.data(), 1, &options, false);
    ms_route_options bad_options = options;
    bad_options.precision = 9;
    rejects(MS_EINVAL, &input, passes.data(), 1, &bad_options);
    bad_options = options;
    bad_options.execution = 0;
    rejects(MS_EINVAL, &input, passes.data(), 1, &bad_options);

    ms_array bad_input = input;
    bad_input.dtype = MS_COMPLEX128;
    rejects(MS_ENOTSUP, &bad_input, passes.data(), 1, &options);
    bad_input = input;
    bad_input.dtype = 0;
    rejects(MS_EINVAL, &bad_input, passes.data(), 1, &options);
    bad_input = input;
    bad_input.ndim = 0;
    rejects(MS_EINVAL, &bad_input, passes.data(), 1, &options);
    bad_input = input;
    bad_input.ndim = MS_MAX_NDIM + 1;
    rejects(MS_EINVAL, &bad_input, passes.data(), 1, &options);
    bad_input = input;
    bad_input.data = nullptr;
    rejects(MS_EINVAL, &bad_input, passes.data(), 1, &options);
    bad_input = input;
    bad_input.data = reinterpret_cast<const char*>(data.data()) + 1;  // misaligned
    rejects(MS_EINVAL, &bad_input, passes.data(), 1, &options);
    bad_input = input;
    bad_input.struct_size = 8;
    rejects(MS_EINVAL, &bad_input, passes.data(), 1, &options);

    passes = with([](ms_ndimage_pass& p) { p.struct_size = 8; });
    rejects(MS_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](ms_ndimage_pass& p) { p.abi_version = 99; });
    rejects(MS_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](ms_ndimage_pass& p) { p.kind = 0; });
    rejects(MS_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](ms_ndimage_pass& p) { p.axis = 2; });
    rejects(MS_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](ms_ndimage_pass& p) { p.mode = 5; });
    rejects(MS_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](ms_ndimage_pass& p) { p.size = 0; });
    rejects(MS_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](ms_ndimage_pass& p) { p.size = MS_NDIMAGE_MAX_WINDOW + 1; });
    rejects(MS_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](ms_ndimage_pass& p) { p.origin = 2; });   // 3 taps allow -1 .. 1
    rejects(MS_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](ms_ndimage_pass& p) { p.origin = -2; });
    rejects(MS_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](ms_ndimage_pass& p) { p.weights = nullptr; });
    rejects(MS_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](ms_ndimage_pass& p) { p.kind = MS_NDIMAGE_UNIFORM; });  // weights given
    rejects(MS_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](ms_ndimage_pass& p) { p.reserved[1] = 1; });
    rejects(MS_EINVAL, &input, passes.data(), 1, &options);
    // An even window has one fewer position on the right: origin 1 is out for two taps.
    Spec even = good;
    even.weights = {1.0, 2.0};
    even.origin = 1;
    const std::vector<Spec> even_window = {even};
    passes = to_passes(even_window);
    rejects(MS_EINVAL, &input, passes.data(), 1, &options);
    // A bad second pass fails the whole call.
    Spec bad_second = good;
    bad_second.axis = 9;
    const std::vector<Spec> two = {good, bad_second};
    passes = to_passes(two);
    rejects(MS_EINVAL, &input, passes.data(), 2, &options);

    // A wrong route-info header is an error too, and leaves the output alone.
    {
        passes = to_passes(one);
        ms_array_result result;
        std::memset(&result, 0x5a, sizeof(result));
        const ms_array_result before = result;
        ms_route_info info{};
        CHECK(ms_ndimage_separable_filter(ctx, &input, passes.data(), 1, &options, &result, &info) == MS_EINVAL);
        CHECK(std::memcmp(&result, &before, sizeof(result)) == 0);
    }
    // info is optional; a successful call clears the error text.
    {
        passes = to_passes(one);
        ms_array_result result{};
        CHECK(ms_ndimage_separable_filter(ctx, &input, passes.data(), 1, &options, &result, nullptr) == MS_OK);
        CHECK(last_error().empty());
        ms_buffer_release(result.buffer);
    }
    // Empty arrays are valid and come back empty with their shape.
    {
        const std::vector<uint64_t> empty_shape = {0, 5};
        const ms_array empty = describe(empty_shape, MS_FLOAT32, nullptr);
        passes = to_passes(one);
        ms_array_result result{};
        CHECK(ms_ndimage_separable_filter(ctx, &empty, passes.data(), 1, &options, &result, nullptr) == MS_OK);
        CHECK(result.ndim == 2 && result.shape[0] == 0 && result.shape[1] == 5 && result.dtype == MS_FLOAT32);
        const void* bytes = nullptr;
        uint64_t size = 1;
        CHECK(ms_buffer_data(result.buffer, &bytes, &size) == MS_OK && size == 0);
        ms_buffer_release(result.buffer);
    }
}

ms_execution_info counters() {
    ms_execution_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = MS_RUNTIME_ABI_VERSION;
    CHECK(ms_context_execution_info(ctx, &info) == MS_OK);
    return info;
}

void test_routing() {
    const std::vector<uint64_t> small = {32, 32};
    const std::vector<double> small_data = make_data(product(small), 3);
    Spec g0, g1;
    g0.axis = 0;
    g1.axis = 1;
    g0.weights = g1.weights = gaussian(12, 3.0, 0);

    // AUTO keeps a small array on the CPU under every policy; forced CPU is honoured.
    for (uint32_t precision : {MS_PRECISION_FAST, MS_PRECISION_REFINED, MS_PRECISION_STRICT}) {
        for (uint32_t execution : {MS_EXECUTION_AUTO, MS_EXECUTION_CPU}) {
            const Outcome outcome = run(small_data, small, MS_FLOAT64, {g0, g1}, execution, precision);
            CHECK(outcome.status == MS_OK && outcome.info.engine == MS_EXECUTION_CPU);
            CHECK(outcome.info.compute_dtype == MS_FLOAT64);
        }
    }
    // The CPU engine does not depend on scheduling: repeated runs are identical.
    {
        const std::vector<uint64_t> shape = {257, 300};
        const std::vector<double> data = make_data(product(shape), 4);
        const Outcome first = run(data, shape, MS_FLOAT64, {g0, g1}, MS_EXECUTION_CPU);
        for (int repeat = 0; repeat < 3; ++repeat) {
            const Outcome again = run(data, shape, MS_FLOAT64, {g0, g1}, MS_EXECUTION_CPU);
            CHECK(again.status == MS_OK && again.values == first.values);
        }
    }
    if (!have_gpu) {
        // Without a GPU a forced GPU route is refused, not silently rerouted.
        const Outcome refused = run(small_data, small, MS_FLOAT32, {g0, g1}, MS_EXECUTION_GPU);
        CHECK(refused.status == MS_ENOTSUP && !last_error().empty());
        return;
    }
    // Forced GPU: float64 under fast and refined, float32 under every policy;
    // strict refuses float64 instead of losing precision silently.
    for (uint32_t precision : {MS_PRECISION_FAST, MS_PRECISION_REFINED}) {
        const Outcome outcome = run(small_data, small, MS_FLOAT64, {g0, g1}, MS_EXECUTION_GPU, precision);
        CHECK(outcome.status == MS_OK && outcome.info.engine == MS_EXECUTION_GPU);
        CHECK(outcome.info.compute_dtype == MS_FLOAT32);
    }
    CHECK(run(small_data, small, MS_FLOAT32, {g0, g1}, MS_EXECUTION_GPU, MS_PRECISION_STRICT).info.engine == MS_EXECUTION_GPU);
    CHECK(run(small_data, small, MS_FLOAT64, {g0, g1}, MS_EXECUTION_GPU, MS_PRECISION_STRICT).status == MS_ENOTSUP);
    CHECK(!last_error().empty());

    // One call is one GPU operation in one command buffer, one dispatch per pass.
    {
        const ms_execution_info before = counters();
        CHECK(run(small_data, small, MS_FLOAT64, {g0, g1, g0}, MS_EXECUTION_GPU).status == MS_OK);
        const ms_execution_info after = counters();
        CHECK(after.gpu_operations == before.gpu_operations + 1);
        CHECK(after.gpu_command_buffers == before.gpu_command_buffers + 1);
        CHECK(after.gpu_dispatches == before.gpu_dispatches + 3);
    }
    // AUTO on a large array: the GPU under fast and refined, the CPU under
    // strict for both dtypes (SciPy's float32 results come from double sums).
    {
        const std::vector<uint64_t> large = {1536, 2048};
        const std::vector<double> data = make_data(product(large), 5);
        const Outcome fast = run(data, large, MS_FLOAT64, {g0, g1}, MS_EXECUTION_AUTO, MS_PRECISION_FAST);
        CHECK(fast.status == MS_OK && fast.info.engine == MS_EXECUTION_GPU && fast.info.compute_dtype == MS_FLOAT32);
        const Outcome strict = run(data, large, MS_FLOAT64, {g0, g1}, MS_EXECUTION_AUTO, MS_PRECISION_STRICT);
        CHECK(strict.status == MS_OK && strict.info.engine == MS_EXECUTION_CPU);
        CHECK(max_difference(fast.values, strict.values) <= budget(data, large, {g0, g1}, unit32, false));
        const Outcome strict32 = run(data, large, MS_FLOAT32, {g0, g1}, MS_EXECUTION_AUTO, MS_PRECISION_STRICT);
        CHECK(strict32.status == MS_OK && strict32.info.engine == MS_EXECUTION_CPU);
        const Outcome refined = run(data, large, MS_FLOAT32, {g0, g1}, MS_EXECUTION_AUTO, MS_PRECISION_REFINED);
        CHECK(refined.status == MS_OK && refined.info.engine == MS_EXECUTION_GPU);
        // A long uniform window stays on the CPU, whose running sum does not
        // grow with the window, even in a call that is otherwise GPU-sized.
        Spec wide;
        wide.kind = MS_NDIMAGE_UNIFORM;
        wide.axis = 1;
        wide.size = 2001;
        const Outcome uniform = run(data, large, MS_FLOAT64, {g0, wide}, MS_EXECUTION_AUTO, MS_PRECISION_FAST);
        CHECK(uniform.status == MS_OK && uniform.info.engine == MS_EXECUTION_CPU);
    }
    // Released results and scratch return to the buffer cache: steady-state
    // calls allocate nothing.
    {
        const std::vector<uint64_t> shape = {300, 400};
        const std::vector<double> data = make_data(product(shape), 6);
        for (uint32_t execution : {MS_EXECUTION_GPU, MS_EXECUTION_CPU}) {
            CHECK(run(data, shape, MS_FLOAT64, {g0, g1}, execution).status == MS_OK);
            const ms_execution_info before = counters();
            for (int repeat = 0; repeat < 4; ++repeat)
                CHECK(run(data, shape, MS_FLOAT64, {g0, g1}, execution).status == MS_OK);
            const ms_execution_info after = counters();
            CHECK(after.native_allocations == before.native_allocations);
            CHECK(after.cached_reuses > before.cached_reuses);
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
    options.flags = MS_CONTEXT_POISON_UNWRITTEN;  // an element an engine fails to write shows up as garbage
    options.memory_limit_bytes = uint64_t{2} << 30;
    CHECK(ms_context_create(&options, &ctx) == MS_OK);

    test_validation();
    test_grid();
    test_sequences();
    test_larger();
    test_fused_widening_is_exact();
    test_routing();

    ms_context_release(ctx);
    std::printf("ndimage filters ok: %lu CPU cases (worst error %.2g of its bound), %lu GPU cases (worst %.2g)%s\n",
                cpu_cases, worst_cpu, gpu_cases, worst_gpu, have_gpu ? "" : " [built without Metal: CPU engine only]");
    return 0;
}
