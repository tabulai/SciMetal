// sm_ndimage_separable_filter through the C ABI: the CPU engine against a
// direct evaluation of SciPy's definition, the forced-GPU engine against the
// CPU engine, argument validation, routing and the execution counters.
//
// Runs plain and under the Metal API and shader validators. A build without
// Metal (no SM_TEST_METAL) runs the CPU engine's checks on CPU storage.
#include "sm_ndimage.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "ndimage filters test failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

namespace {
sm_context* ctx = nullptr;
bool have_gpu = false;
unsigned long cpu_cases = 0, gpu_cases = 0;
double worst_cpu = 0.0, worst_gpu = 0.0;  // largest error / bound seen

struct Spec {
    uint32_t kind = SM_NDIMAGE_CORRELATE;
    uint32_t axis = 0;
    uint32_t mode = SM_NDIMAGE_REFLECT;
    int64_t origin = 0;
    std::vector<double> weights;  // correlate
    uint64_t size = 0;            // uniform
    double cval = 0.0;
    uint64_t taps() const { return kind == SM_NDIMAGE_UNIFORM ? size : weights.size(); }
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
                    sum += spec.kind == SM_NDIMAGE_UNIFORM ? value : spec.weights[static_cast<size_t>(j)] * value;
                }
                if (spec.kind == SM_NDIMAGE_UNIFORM) sum /= static_cast<double>(taps);
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
        if (spec.mode == SM_NDIMAGE_CONSTANT) {
            magnitude = std::max(magnitude, std::fabs(spec.cval));
            error = std::max(error, unit * std::fabs(spec.cval));
        }
        double gain = 0.0;
        for (double w : spec.weights) gain += std::fabs(w);
        if (spec.kind == SM_NDIMAGE_UNIFORM) gain = 1.0;
        double terms = static_cast<double>(spec.taps()) + 4.0;
        if (running && spec.kind == SM_NDIMAGE_UNIFORM) terms += 2.0 * static_cast<double>(shape[spec.axis]);
        error = gain * error + terms * unit * gain * magnitude;
        magnitude *= gain;
    }
    return error;
}

struct Outcome {
    sm_status status = SM_OK;
    sm_route_info info{};
    std::vector<double> values;
};

// The passes point at the specs' weights: `specs` must outlive them.
std::vector<sm_ndimage_pass> to_passes(const std::vector<Spec>& specs) {
    std::vector<sm_ndimage_pass> passes(specs.size());
    for (size_t i = 0; i < specs.size(); ++i) {
        sm_ndimage_pass& pass = passes[i];
        pass.struct_size = sizeof(pass);
        pass.abi_version = SM_NDIMAGE_ABI_VERSION;
        pass.kind = specs[i].kind;
        pass.axis = specs[i].axis;
        pass.mode = specs[i].mode;
        pass.origin = specs[i].origin;
        pass.size = specs[i].taps();
        pass.weights = specs[i].kind == SM_NDIMAGE_UNIFORM ? nullptr : specs[i].weights.data();
        pass.cval = specs[i].cval;
    }
    return passes;
}

sm_array describe(const std::vector<uint64_t>& shape, uint32_t dtype, const void* data) {
    sm_array array{};
    array.struct_size = sizeof(array);
    array.abi_version = SM_ARRAY_ABI_VERSION;
    array.data = data;
    array.dtype = dtype;
    array.ndim = static_cast<uint32_t>(shape.size());
    for (size_t axis = 0; axis < shape.size(); ++axis) array.shape[axis] = shape[axis];
    return array;
}

sm_route_options route(uint32_t execution, uint32_t precision) {
    sm_route_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = SM_ARRAY_ABI_VERSION;
    options.execution = execution;
    options.precision = precision;
    return options;
}

// Runs the filter on `data` stored as `dtype` with `out_dtype` (0 = same as `dtype`) and returns the result as doubles.
Outcome run_to(const std::vector<double>& data, const std::vector<uint64_t>& shape, uint32_t dtype,
               uint32_t out_dtype, const std::vector<Spec>& specs, uint32_t execution,
               uint32_t precision = SM_PRECISION_FAST) {
    std::vector<float> narrow;
    const void* pointer = data.data();
    if (dtype == SM_FLOAT32) {
        narrow.assign(data.begin(), data.end());
        pointer = narrow.data();
    }
    const sm_array input = describe(shape, dtype, pointer);
    const std::vector<sm_ndimage_pass> passes = to_passes(specs);
    const sm_route_options options = route(execution, precision);
    Outcome outcome;
    outcome.info.struct_size = sizeof(outcome.info);
    outcome.info.abi_version = SM_ARRAY_ABI_VERSION;
    sm_array_result result{};
    outcome.status = (out_dtype == 0)
        ? sm_ndimage_separable_filter(ctx, &input, passes.data(), static_cast<uint32_t>(passes.size()),
                                      &options, &result, &outcome.info)
        : sm_ndimage_separable_filter_to(ctx, &input, passes.data(), static_cast<uint32_t>(passes.size()),
                                         out_dtype, &options, &result, &outcome.info);
    if (outcome.status != SM_OK) return outcome;
    const uint32_t expected_out = out_dtype ? out_dtype : dtype;
    CHECK(result.buffer && result.dtype == expected_out && result.ndim == shape.size());
    for (size_t axis = 0; axis < shape.size(); ++axis) CHECK(result.shape[axis] == shape[axis]);
    const void* bytes = nullptr;
    uint64_t size = 0;
    CHECK(sm_buffer_data(result.buffer, &bytes, &size) == SM_OK);
    CHECK(size == data.size() * (expected_out == SM_FLOAT32 ? 4 : 8));
    outcome.values.resize(data.size());
    if (expected_out == SM_FLOAT32) {
        const float* values = static_cast<const float*>(bytes);
        for (size_t i = 0; i < data.size(); ++i) outcome.values[i] = values[i];
    } else if (!data.empty()) {
        std::memcpy(outcome.values.data(), bytes, static_cast<size_t>(size));
    }
    sm_buffer_release(result.buffer);
    return outcome;
}

Outcome run(const std::vector<double>& data, const std::vector<uint64_t>& shape, uint32_t dtype,
            const std::vector<Spec>& specs, uint32_t execution, uint32_t precision = SM_PRECISION_FAST) {
    return run_to(data, shape, dtype, 0, specs, execution, precision);
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
    for (uint32_t dtype : {SM_FLOAT64, SM_FLOAT32}) {
        const bool single = dtype == SM_FLOAT32;
        std::vector<double> data = exact;
        if (single) for (double& v : data) v = static_cast<double>(static_cast<float>(v));
        std::vector<double> expected = data;
        for (const Spec& spec : specs) expected = reference_pass(expected, shape, spec, single);

        const Outcome cpu = run(data, shape, dtype, specs, SM_EXECUTION_CPU);
        CHECK(cpu.status == SM_OK);
        CHECK(cpu.info.engine == SM_EXECUTION_CPU && cpu.info.compute_dtype == SM_FLOAT64);
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
        const Outcome gpu = run(data, shape, dtype, specs, SM_EXECUTION_GPU);
        CHECK(gpu.status == SM_OK);
        CHECK(gpu.info.engine == SM_EXECUTION_GPU && gpu.info.compute_dtype == SM_FLOAT32);
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
    const uint32_t modes[] = {SM_NDIMAGE_NEAREST, SM_NDIMAGE_WRAP, SM_NDIMAGE_REFLECT, SM_NDIMAGE_MIRROR,
                              SM_NDIMAGE_CONSTANT};
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
                        spec.kind = SM_NDIMAGE_UNIFORM;
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
            g.mode = axis % 2 ? SM_NDIMAGE_MIRROR : SM_NDIMAGE_REFLECT;
            g.weights = gaussian(8, 2.0, axis == 1 ? 1 : 0);
            gaussians.push_back(g);
            Spec m;
            m.axis = static_cast<uint32_t>(shape.size()) - 1 - axis;
            m.mode = axis % 2 ? SM_NDIMAGE_CONSTANT : SM_NDIMAGE_WRAP;
            m.cval = 0.5;
            if (axis % 2) {
                m.kind = SM_NDIMAGE_UNIFORM;
                m.size = 4;
                m.origin = -1;
            } else {
                m.weights = ramp(6);
                m.origin = 2;
            }
            mixed.push_back(m);
            Spec s;
            s.axis = axis;
            s.mode = SM_NDIMAGE_NEAREST;
            s.weights = axis == 0 ? std::vector<double>{-1.0, 0.0, 1.0} : std::vector<double>{1.0, 2.0, 1.0};
            sobel.push_back(s);
            Spec u;
            u.kind = SM_NDIMAGE_UNIFORM;
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
    const uint32_t modes[] = {SM_NDIMAGE_REFLECT, SM_NDIMAGE_CONSTANT, SM_NDIMAGE_WRAP};
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
            u.kind = SM_NDIMAGE_UNIFORM;
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
    g1.mode = SM_NDIMAGE_CONSTANT;
    for (const std::vector<Spec>& specs : {std::vector<Spec>{g1}, {g0, g1}, {g1, g0}}) {
        const Outcome single = run(data, shape, SM_FLOAT32, specs, SM_EXECUTION_GPU);
        const Outcome wide = run(data, shape, SM_FLOAT64, specs, SM_EXECUTION_GPU);
        CHECK(single.status == SM_OK && wide.status == SM_OK);
        for (size_t i = 0; i < data.size(); ++i) {
            const bool same = std::memcmp(&single.values[i], &wide.values[i], sizeof(double)) == 0;
            CHECK(same || (std::isnan(single.values[i]) && std::isnan(wide.values[i])));
        }
    }
    // The one-tap pass with weight 1 is an exact copy.
    const Outcome copy = run(data, shape, SM_FLOAT64, {g1}, SM_EXECUTION_GPU);
    CHECK(copy.status == SM_OK);
    for (size_t i = 0; i < data.size(); ++i) {
        // Subnormal float32 values may be flushed to zero by the GPU's float datapath.
        if (std::fabs(data[i]) < 1.2e-38) continue;
        CHECK(std::memcmp(&copy.values[i], &data[i], sizeof(double)) == 0);
    }
}

std::string last_error() {
    char text[512] = {};
    CHECK(sm_context_error_copy(ctx, text, sizeof(text)) == SM_OK);
    return text;
}

void test_validation() {
    const std::vector<uint64_t> shape = {6, 5};
    const std::vector<double> data = make_data(30, 1);
    Spec good;
    good.axis = 1;
    good.weights = {0.25, 0.5, 0.25};
    const std::vector<Spec> one = {good};
    const sm_array input = describe(shape, SM_FLOAT64, data.data());
    const sm_route_options options = route(SM_EXECUTION_CPU, SM_PRECISION_STRICT);

    // Each call must fail with `expected`, describe the failure and leave out and info untouched.
    const auto rejects = [&](sm_status expected, const sm_array* array, const sm_ndimage_pass* pass_pointer,
                             uint32_t count, const sm_route_options* opts, bool with_out = true) {
        sm_array_result result;
        std::memset(&result, 0x5a, sizeof(result));
        const sm_array_result before = result;
        sm_route_info info{};
        info.struct_size = sizeof(info);
        info.abi_version = SM_ARRAY_ABI_VERSION;
        info.engine = 77;
        info.compute_dtype = 78;
        const sm_status status =
            sm_ndimage_separable_filter(ctx, array, pass_pointer, count, opts, with_out ? &result : nullptr, &info);
        CHECK(status == expected);
        CHECK(std::memcmp(&result, &before, sizeof(result)) == 0);
        CHECK(info.engine == 77 && info.compute_dtype == 78);
        CHECK(!last_error().empty());
    };
    const auto with = [&](auto&& change) {
        std::vector<sm_ndimage_pass> passes = to_passes(one);
        change(passes[0]);
        return passes;
    };
    std::vector<sm_ndimage_pass> passes = to_passes(one);
    rejects(SM_EINVAL, nullptr, passes.data(), 1, &options);
    rejects(SM_EINVAL, &input, nullptr, 1, &options);
    rejects(SM_EINVAL, &input, passes.data(), 0, &options);
    rejects(SM_EINVAL, &input, passes.data(), SM_NDIMAGE_MAX_PASSES + 1, &options);
    rejects(SM_EINVAL, &input, passes.data(), 1, nullptr);
    rejects(SM_EINVAL, &input, passes.data(), 1, &options, false);
    sm_route_options bad_options = options;
    bad_options.precision = 9;
    rejects(SM_EINVAL, &input, passes.data(), 1, &bad_options);
    bad_options = options;
    bad_options.execution = 0;
    rejects(SM_EINVAL, &input, passes.data(), 1, &bad_options);

    sm_array bad_input = input;
    bad_input.dtype = SM_COMPLEX128;
    rejects(SM_ENOTSUP, &bad_input, passes.data(), 1, &options);
    bad_input = input;
    bad_input.dtype = 0;
    rejects(SM_EINVAL, &bad_input, passes.data(), 1, &options);
    bad_input = input;
    bad_input.ndim = 0;
    rejects(SM_EINVAL, &bad_input, passes.data(), 1, &options);
    bad_input = input;
    bad_input.ndim = SM_MAX_NDIM + 1;
    rejects(SM_EINVAL, &bad_input, passes.data(), 1, &options);
    bad_input = input;
    bad_input.data = nullptr;
    rejects(SM_EINVAL, &bad_input, passes.data(), 1, &options);
    bad_input = input;
    bad_input.data = reinterpret_cast<const char*>(data.data()) + 1;  // misaligned
    rejects(SM_EINVAL, &bad_input, passes.data(), 1, &options);
    bad_input = input;
    bad_input.struct_size = 8;
    rejects(SM_EINVAL, &bad_input, passes.data(), 1, &options);

    passes = with([](sm_ndimage_pass& p) { p.struct_size = 8; });
    rejects(SM_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](sm_ndimage_pass& p) { p.abi_version = 99; });
    rejects(SM_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](sm_ndimage_pass& p) { p.kind = 0; });
    rejects(SM_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](sm_ndimage_pass& p) { p.axis = 2; });
    rejects(SM_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](sm_ndimage_pass& p) { p.mode = 5; });
    rejects(SM_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](sm_ndimage_pass& p) { p.size = 0; });
    rejects(SM_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](sm_ndimage_pass& p) { p.size = SM_NDIMAGE_MAX_WINDOW + 1; });
    rejects(SM_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](sm_ndimage_pass& p) { p.origin = 2; });   // 3 taps allow -1 .. 1
    rejects(SM_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](sm_ndimage_pass& p) { p.origin = -2; });
    rejects(SM_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](sm_ndimage_pass& p) { p.weights = nullptr; });
    rejects(SM_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](sm_ndimage_pass& p) { p.kind = SM_NDIMAGE_UNIFORM; });  // weights given
    rejects(SM_EINVAL, &input, passes.data(), 1, &options);
    passes = with([](sm_ndimage_pass& p) { p.reserved[1] = 1; });
    rejects(SM_EINVAL, &input, passes.data(), 1, &options);
    // An even window has one fewer position on the right: origin 1 is out for two taps.
    Spec even = good;
    even.weights = {1.0, 2.0};
    even.origin = 1;
    const std::vector<Spec> even_window = {even};
    passes = to_passes(even_window);
    rejects(SM_EINVAL, &input, passes.data(), 1, &options);
    // A bad second pass fails the whole call.
    Spec bad_second = good;
    bad_second.axis = 9;
    const std::vector<Spec> two = {good, bad_second};
    passes = to_passes(two);
    rejects(SM_EINVAL, &input, passes.data(), 2, &options);

    // A wrong route-info header is an error too, and leaves the output alone.
    {
        passes = to_passes(one);
        sm_array_result result;
        std::memset(&result, 0x5a, sizeof(result));
        const sm_array_result before = result;
        sm_route_info info{};
        CHECK(sm_ndimage_separable_filter(ctx, &input, passes.data(), 1, &options, &result, &info) == SM_EINVAL);
        CHECK(std::memcmp(&result, &before, sizeof(result)) == 0);
    }
    // info is optional; a successful call clears the error text.
    {
        passes = to_passes(one);
        sm_array_result result{};
        CHECK(sm_ndimage_separable_filter(ctx, &input, passes.data(), 1, &options, &result, nullptr) == SM_OK);
        CHECK(last_error().empty());
        sm_buffer_release(result.buffer);
    }
    // Empty arrays are valid and come back empty with their shape.
    {
        const std::vector<uint64_t> empty_shape = {0, 5};
        const sm_array empty = describe(empty_shape, SM_FLOAT32, nullptr);
        passes = to_passes(one);
        sm_array_result result{};
        CHECK(sm_ndimage_separable_filter(ctx, &empty, passes.data(), 1, &options, &result, nullptr) == SM_OK);
        CHECK(result.ndim == 2 && result.shape[0] == 0 && result.shape[1] == 5 && result.dtype == SM_FLOAT32);
        const void* bytes = nullptr;
        uint64_t size = 1;
        CHECK(sm_buffer_data(result.buffer, &bytes, &size) == SM_OK && size == 0);
        sm_buffer_release(result.buffer);
    }
}

sm_execution_info counters() {
    sm_execution_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = SM_RUNTIME_ABI_VERSION;
    CHECK(sm_context_execution_info(ctx, &info) == SM_OK);
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
    for (uint32_t precision : {SM_PRECISION_FAST, SM_PRECISION_REFINED, SM_PRECISION_STRICT}) {
        for (uint32_t execution : {SM_EXECUTION_AUTO, SM_EXECUTION_CPU}) {
            const Outcome outcome = run(small_data, small, SM_FLOAT64, {g0, g1}, execution, precision);
            CHECK(outcome.status == SM_OK && outcome.info.engine == SM_EXECUTION_CPU);
            CHECK(outcome.info.compute_dtype == SM_FLOAT64);
        }
    }
    // The CPU engine does not depend on scheduling: repeated runs are identical.
    {
        const std::vector<uint64_t> shape = {257, 300};
        const std::vector<double> data = make_data(product(shape), 4);
        const Outcome first = run(data, shape, SM_FLOAT64, {g0, g1}, SM_EXECUTION_CPU);
        for (int repeat = 0; repeat < 3; ++repeat) {
            const Outcome again = run(data, shape, SM_FLOAT64, {g0, g1}, SM_EXECUTION_CPU);
            CHECK(again.status == SM_OK && again.values == first.values);
        }
    }
    if (!have_gpu) {
        // Without a GPU a forced GPU route is refused, not silently rerouted.
        const Outcome refused = run(small_data, small, SM_FLOAT32, {g0, g1}, SM_EXECUTION_GPU);
        CHECK(refused.status == SM_ENOTSUP && !last_error().empty());
        return;
    }
    // Forced GPU: float64 under fast and refined, float32 under every policy;
    // strict refuses float64 instead of losing precision silently.
    for (uint32_t precision : {SM_PRECISION_FAST, SM_PRECISION_REFINED}) {
        const Outcome outcome = run(small_data, small, SM_FLOAT64, {g0, g1}, SM_EXECUTION_GPU, precision);
        CHECK(outcome.status == SM_OK && outcome.info.engine == SM_EXECUTION_GPU);
        CHECK(outcome.info.compute_dtype == SM_FLOAT32);
    }
    CHECK(run(small_data, small, SM_FLOAT32, {g0, g1}, SM_EXECUTION_GPU, SM_PRECISION_STRICT).info.engine == SM_EXECUTION_GPU);
    CHECK(run(small_data, small, SM_FLOAT64, {g0, g1}, SM_EXECUTION_GPU, SM_PRECISION_STRICT).status == SM_ENOTSUP);
    CHECK(!last_error().empty());

    // One call is one GPU operation in one command buffer, one dispatch per pass.
    {
        const sm_execution_info before = counters();
        CHECK(run(small_data, small, SM_FLOAT64, {g0, g1, g0}, SM_EXECUTION_GPU).status == SM_OK);
        const sm_execution_info after = counters();
        CHECK(after.gpu_operations == before.gpu_operations + 1);
        CHECK(after.gpu_command_buffers == before.gpu_command_buffers + 1);
        CHECK(after.gpu_dispatches == before.gpu_dispatches + 3);
    }
    // AUTO on a large array: the GPU under fast and refined, the CPU under
    // strict for both dtypes (SciPy's float32 results come from double sums).
    {
        const std::vector<uint64_t> large = {1536, 2048};
        const std::vector<double> data = make_data(product(large), 5);
        const Outcome fast = run(data, large, SM_FLOAT64, {g0, g1}, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        CHECK(fast.status == SM_OK && fast.info.engine == SM_EXECUTION_GPU && fast.info.compute_dtype == SM_FLOAT32);
        const Outcome strict = run(data, large, SM_FLOAT64, {g0, g1}, SM_EXECUTION_AUTO, SM_PRECISION_STRICT);
        CHECK(strict.status == SM_OK && strict.info.engine == SM_EXECUTION_CPU);
        CHECK(max_difference(fast.values, strict.values) <= budget(data, large, {g0, g1}, unit32, false));
        const Outcome strict32 = run(data, large, SM_FLOAT32, {g0, g1}, SM_EXECUTION_AUTO, SM_PRECISION_STRICT);
        CHECK(strict32.status == SM_OK && strict32.info.engine == SM_EXECUTION_CPU);
        const Outcome refined = run(data, large, SM_FLOAT32, {g0, g1}, SM_EXECUTION_AUTO, SM_PRECISION_REFINED);
        CHECK(refined.status == SM_OK && refined.info.engine == SM_EXECUTION_GPU);
        // A long uniform window stays on the CPU, whose running sum does not
        // grow with the window, even in a call that is otherwise GPU-sized.
        Spec wide;
        wide.kind = SM_NDIMAGE_UNIFORM;
        wide.axis = 1;
        wide.size = 2001;
        const Outcome uniform = run(data, large, SM_FLOAT64, {g0, wide}, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        CHECK(uniform.status == SM_OK && uniform.info.engine == SM_EXECUTION_CPU);
    }
    // Released results and scratch return to the buffer cache: steady-state
    // calls allocate nothing.
    {
        const std::vector<uint64_t> shape = {300, 400};
        const std::vector<double> data = make_data(product(shape), 6);
        for (uint32_t execution : {SM_EXECUTION_GPU, SM_EXECUTION_CPU}) {
            CHECK(run(data, shape, SM_FLOAT64, {g0, g1}, execution).status == SM_OK);
            const sm_execution_info before = counters();
            for (int repeat = 0; repeat < 4; ++repeat)
                CHECK(run(data, shape, SM_FLOAT64, {g0, g1}, execution).status == SM_OK);
            const sm_execution_info after = counters();
            CHECK(after.native_allocations == before.native_allocations);
            CHECK(after.cached_reuses > before.cached_reuses);
        }
    }
}

void test_overflow_guard() {
    if (!have_gpu) return;
    const std::vector<uint64_t> large = {1536, 2048};
    const size_t n = product(large);
    Spec g0, g1;
    g0.axis = 0;
    g1.axis = 1;
    g0.weights = g1.weights = gaussian(12, 3.0, 0);

    // Normal data runs on GPU with empty reason.
    {
        const std::vector<double> normal = make_data(n, 7);
        const Outcome ok = run(normal, large, SM_FLOAT64, {g0, g1}, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        CHECK(ok.status == SM_OK && ok.info.engine == SM_EXECUTION_GPU && ok.info.compute_dtype == SM_FLOAT32);
        CHECK(ok.info.reason[0] == '\0');
    }

    // 1. Direct input overflow (1e39): AUTO -> CPU (SM_FLOAT64, non-empty reason, 0 GPU ops); GPU -> SM_ENOTSUP.
    {
        std::vector<double> data(n, 1.0);
        data[0] = 1e39;
        data[n / 2] = -1e39;
        const Outcome cpu_ref = run(data, large, SM_FLOAT64, {g0, g1}, SM_EXECUTION_CPU, SM_PRECISION_FAST);
        CHECK(cpu_ref.status == SM_OK);

        const sm_execution_info before = counters();
        const Outcome auto_out = run(data, large, SM_FLOAT64, {g0, g1}, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        const sm_execution_info after = counters();
        CHECK(auto_out.status == SM_OK);
        CHECK(auto_out.info.engine == SM_EXECUTION_CPU && auto_out.info.compute_dtype == SM_FLOAT64);
        CHECK(auto_out.info.reason[0] != '\0');
        CHECK(after.gpu_operations == before.gpu_operations);
        CHECK(after.gpu_command_buffers == before.gpu_command_buffers);
        CHECK(auto_out.values == cpu_ref.values);
        for (double v : auto_out.values) CHECK(std::isfinite(v));

        const Outcome gpu_out = run(data, large, SM_FLOAT64, {g0, g1}, SM_EXECUTION_GPU, SM_PRECISION_FAST);
        CHECK(gpu_out.status == SM_ENOTSUP && !last_error().empty());
        const sm_execution_info after_gpu = counters();
        CHECK(after_gpu.gpu_operations == after.gpu_operations);
        CHECK(after_gpu.gpu_command_buffers == after.gpu_command_buffers);
    }

    // 2. Gain-amplified 1e38 (each element fits float32, yet 1e38 * 16 = 1.6e39 overflows float32):
    {
        Spec s0, s1;
        s0.axis = 0;
        s1.axis = 1;
        s0.weights = s1.weights = {1.0, 2.0, 1.0};
        // 6 passes of {1, 2, 1} -> gain = 4^6 = 4096, 1536 * 2048 * 48 ~ 151M ops > 80M work.
        const std::vector<Spec> passes = {s0, s1, s0, s1, s0, s1};
        std::vector<double> data(n, 1e38);
        const Outcome cpu_ref = run(data, large, SM_FLOAT64, passes, SM_EXECUTION_CPU, SM_PRECISION_FAST);
        CHECK(cpu_ref.status == SM_OK);

        const sm_execution_info before = counters();
        const Outcome auto_out = run(data, large, SM_FLOAT64, passes, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        const sm_execution_info after = counters();
        CHECK(auto_out.status == SM_OK);
        CHECK(auto_out.info.engine == SM_EXECUTION_CPU && auto_out.info.compute_dtype == SM_FLOAT64);
        CHECK(auto_out.info.reason[0] != '\0');
        CHECK(after.gpu_operations == before.gpu_operations);
        CHECK(auto_out.values == cpu_ref.values);

        const Outcome gpu_out = run(data, large, SM_FLOAT64, passes, SM_EXECUTION_GPU, SM_PRECISION_FAST);
        CHECK(gpu_out.status == SM_ENOTSUP && !last_error().empty());
        const sm_execution_info after_gpu = counters();
        CHECK(after_gpu.gpu_operations == after.gpu_operations);
    }

    // 3. Overflowing constant cval and overflowing weights:
    {
        const std::vector<double> data(n, 1.0);
        Spec cpass = g0;
        cpass.mode = SM_NDIMAGE_CONSTANT;
        cpass.cval = 1e39;
        const sm_execution_info before = counters();
        const Outcome auto_cval = run(data, large, SM_FLOAT64, {cpass, g1}, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        const sm_execution_info after = counters();
        CHECK(auto_cval.status == SM_OK && auto_cval.info.engine == SM_EXECUTION_CPU);
        CHECK(auto_cval.info.compute_dtype == SM_FLOAT64 && auto_cval.info.reason[0] != '\0');
        CHECK(after.gpu_operations == before.gpu_operations);
        CHECK(run(data, large, SM_FLOAT64, {cpass, g1}, SM_EXECUTION_GPU, SM_PRECISION_FAST).status == SM_ENOTSUP);

        Spec wpass = g0;
        wpass.weights = {1e39, 1.0, 1.0};
        const Outcome auto_w = run(data, large, SM_FLOAT64, {wpass, g1, g0, g1}, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        CHECK(auto_w.status == SM_OK && auto_w.info.engine == SM_EXECUTION_CPU && auto_w.info.reason[0] != '\0');
        CHECK(run(data, large, SM_FLOAT64, {wpass}, SM_EXECUTION_GPU, SM_PRECISION_FAST).status == SM_ENOTSUP);
    }

    // 4. Pre-existing [inf, -inf, nan] with cval=inf must still run on the GPU!
    {
        std::vector<double> data = make_data(n, 8);
        data[10] = std::numeric_limits<double>::infinity();
        data[20] = -std::numeric_limits<double>::infinity();
        data[30] = std::numeric_limits<double>::quiet_NaN();
        Spec cpass = g0;
        cpass.mode = SM_NDIMAGE_CONSTANT;
        cpass.cval = std::numeric_limits<double>::infinity();
        const sm_execution_info before = counters();
        const Outcome special = run(data, large, SM_FLOAT64, {cpass, g1}, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        const sm_execution_info after = counters();
        CHECK(special.status == SM_OK && special.info.engine == SM_EXECUTION_GPU);
        CHECK(special.info.compute_dtype == SM_FLOAT32 && special.info.reason[0] == '\0');
        CHECK(after.gpu_operations == before.gpu_operations + 1);
    }

    // 5. mode != CONSTANT with cval = 1e39 stays on GPU; mode == CONSTANT with cval = -1e39 re-routes/declines.
    {
        const std::vector<double> data = make_data(n, 9);
        Spec rpass = g0;
        rpass.mode = SM_NDIMAGE_REFLECT;
        rpass.cval = 1e39;
        const sm_execution_info before = counters();
        const Outcome reflect_ok = run(data, large, SM_FLOAT64, {rpass, g1}, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        const sm_execution_info after = counters();
        CHECK(reflect_ok.status == SM_OK && reflect_ok.info.engine == SM_EXECUTION_GPU);
        CHECK(reflect_ok.info.compute_dtype == SM_FLOAT32 && reflect_ok.info.reason[0] == '\0');
        CHECK(after.gpu_operations == before.gpu_operations + 1);

        Spec neg_cpass = g0;
        neg_cpass.mode = SM_NDIMAGE_CONSTANT;
        neg_cpass.cval = -1e39;
        const Outcome cpu_ref = run(data, large, SM_FLOAT64, {neg_cpass, g1}, SM_EXECUTION_CPU, SM_PRECISION_FAST);
        const Outcome auto_neg = run(data, large, SM_FLOAT64, {neg_cpass, g1}, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        CHECK(auto_neg.status == SM_OK && auto_neg.info.engine == SM_EXECUTION_CPU);
        CHECK(auto_neg.info.compute_dtype == SM_FLOAT64 && auto_neg.info.reason[0] != '\0');
        CHECK(counters().gpu_operations == after.gpu_operations);
        CHECK(auto_neg.values == cpu_ref.values);
        CHECK(run(data, large, SM_FLOAT64, {neg_cpass, g1}, SM_EXECUTION_GPU, SM_PRECISION_FAST).status == SM_ENOTSUP);
    }

    // 6. Symmetric/antisymmetric (below +- above) pre-multiply overflow on 2e38 (uniform + correlate + [0, 1, 0] NaN guard).
    {
        Spec u0, u1;
        u0.kind = SM_NDIMAGE_UNIFORM;
        u0.axis = 0;
        u0.size = 5;
        u1.kind = SM_NDIMAGE_UNIFORM;
        u1.axis = 1;
        u1.size = 5;
        const std::vector<double> normal = make_data(n, 10);
        CHECK(run(normal, large, SM_FLOAT64, {u0, u1}, SM_EXECUTION_AUTO, SM_PRECISION_FAST).info.engine == SM_EXECUTION_GPU);

        const std::vector<double> two_e38(n, 2e38);
        const Outcome u_ref = run(two_e38, large, SM_FLOAT64, {u0, u1}, SM_EXECUTION_CPU, SM_PRECISION_FAST);
        const sm_execution_info before_u = counters();
        const Outcome u_auto = run(two_e38, large, SM_FLOAT64, {u0, u1}, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        CHECK(u_auto.status == SM_OK && u_auto.info.engine == SM_EXECUTION_CPU);
        CHECK(u_auto.info.compute_dtype == SM_FLOAT64 && u_auto.info.reason[0] != '\0');
        CHECK(counters().gpu_operations == before_u.gpu_operations);
        CHECK(u_auto.values == u_ref.values);
        for (double v : u_auto.values) CHECK(std::isfinite(v));
        CHECK(run(two_e38, large, SM_FLOAT64, {u0, u1}, SM_EXECUTION_GPU, SM_PRECISION_FAST).status == SM_ENOTSUP);

        // [0.0, 1.0, 0.0] (symmetric even, gain = 1.0): without the 2*m guard, (2e38 + 2e38) * 0.0 -> inf * 0 = NaN.
        Spec id_sym;
        id_sym.axis = 0;
        id_sym.weights = {0.0, 1.0, 0.0};
        const std::vector<Spec> six_id_sym(6, id_sym);
        CHECK(run(normal, large, SM_FLOAT64, six_id_sym, SM_EXECUTION_AUTO, SM_PRECISION_FAST).info.engine == SM_EXECUTION_GPU);
        const sm_execution_info before_id = counters();
        const Outcome id_auto = run(two_e38, large, SM_FLOAT64, six_id_sym, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        CHECK(id_auto.status == SM_OK && id_auto.info.engine == SM_EXECUTION_CPU);
        CHECK(id_auto.info.compute_dtype == SM_FLOAT64 && id_auto.info.reason[0] != '\0');
        CHECK(counters().gpu_operations == before_id.gpu_operations);
        for (double v : id_auto.values) CHECK(v == 2e38);
        CHECK(run(two_e38, large, SM_FLOAT64, {id_sym}, SM_EXECUTION_GPU, SM_PRECISION_FAST).status == SM_ENOTSUP);

        // Antisymmetric [-0.1, 0.0, 0.1] on row-alternating +-2e38 (gain = 0.2).
        std::vector<double> alt_2e38(n);
        for (size_t i = 0; i < n; ++i) alt_2e38[i] = ((i / large[1]) & 1) ? 2e38 : -2e38;
        Spec anti;
        anti.axis = 0;
        anti.weights = {-0.1, 0.0, 0.1};
        const std::vector<Spec> six_anti(6, anti);
        CHECK(run(normal, large, SM_FLOAT64, six_anti, SM_EXECUTION_AUTO, SM_PRECISION_FAST).info.engine == SM_EXECUTION_GPU);
        const Outcome anti_ref = run(alt_2e38, large, SM_FLOAT64, six_anti, SM_EXECUTION_CPU, SM_PRECISION_FAST);
        const sm_execution_info before_anti = counters();
        const Outcome anti_auto = run(alt_2e38, large, SM_FLOAT64, six_anti, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        CHECK(anti_auto.status == SM_OK && anti_auto.info.engine == SM_EXECUTION_CPU);
        CHECK(anti_auto.info.compute_dtype == SM_FLOAT64 && anti_auto.info.reason[0] != '\0');
        CHECK(counters().gpu_operations == before_anti.gpu_operations);
        CHECK(anti_auto.values == anti_ref.values);
        for (double v : anti_auto.values) CHECK(std::isfinite(v));
        CHECK(run(alt_2e38, large, SM_FLOAT64, {anti}, SM_EXECUTION_GPU, SM_PRECISION_FAST).status == SM_ENOTSUP);
    }

    // 7. Grow-then-shrink intermediate-pass overflow (even-length asymmetric and symmetric+antisymmetric).
    {
        const std::vector<double> one_e38(n, 1e38);
        const std::vector<double> normal = make_data(n, 11);

        // Asymmetric even-length: pass 0 gain = 10 (1e38 -> 1e39 overflows), pass 1 gain = 0.01 (product = 0.1).
        Spec grow, shrink, pad;
        grow.axis = 0;
        grow.weights = {5.0, 5.0};
        shrink.axis = 1;
        shrink.weights = {0.005, 0.005};
        pad.axis = 0;
        pad.weights = {0.5, 0.5};
        const std::vector<Spec> gts_asym = {grow, shrink, pad, pad, pad, pad};
        CHECK(run(normal, large, SM_FLOAT64, gts_asym, SM_EXECUTION_AUTO, SM_PRECISION_FAST).info.engine == SM_EXECUTION_GPU);

        const Outcome gts_ref = run(one_e38, large, SM_FLOAT64, gts_asym, SM_EXECUTION_CPU, SM_PRECISION_FAST);
        const sm_execution_info before_gts = counters();
        const Outcome gts_auto = run(one_e38, large, SM_FLOAT64, gts_asym, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        CHECK(gts_auto.status == SM_OK && gts_auto.info.engine == SM_EXECUTION_CPU);
        CHECK(gts_auto.info.compute_dtype == SM_FLOAT64 && gts_auto.info.reason[0] != '\0');
        CHECK(counters().gpu_operations == before_gts.gpu_operations);
        CHECK(gts_auto.values == gts_ref.values);
        CHECK(run(one_e38, large, SM_FLOAT64, {grow, shrink}, SM_EXECUTION_GPU, SM_PRECISION_FAST).status == SM_ENOTSUP);

        // Symmetric grow (gain = 8) + antisymmetric Gaussian order=1 shrink (sigma = 10, gain ~ 0.08, product ~ 0.64).
        Spec sym_grow;
        sym_grow.axis = 0;
        sym_grow.weights = {2.0, -4.0, 2.0};
        Spec gauss_shrink;
        gauss_shrink.axis = 1;
        gauss_shrink.weights = gaussian(40, 10.0, 1);
        const std::vector<Spec> gts_gauss = {sym_grow, gauss_shrink};
        CHECK(run(normal, large, SM_FLOAT64, gts_gauss, SM_EXECUTION_AUTO, SM_PRECISION_FAST).info.engine == SM_EXECUTION_GPU);

        const Outcome gauss_ref = run(one_e38, large, SM_FLOAT64, gts_gauss, SM_EXECUTION_CPU, SM_PRECISION_FAST);
        const sm_execution_info before_gauss = counters();
        const Outcome gauss_auto = run(one_e38, large, SM_FLOAT64, gts_gauss, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        CHECK(gauss_auto.status == SM_OK && gauss_auto.info.engine == SM_EXECUTION_CPU);
        CHECK(gauss_auto.info.compute_dtype == SM_FLOAT64 && gauss_auto.info.reason[0] != '\0');
        CHECK(counters().gpu_operations == before_gauss.gpu_operations);
        CHECK(gauss_auto.values == gauss_ref.values);
        CHECK(run(one_e38, large, SM_FLOAT64, gts_gauss, SM_EXECUTION_GPU, SM_PRECISION_FAST).status == SM_ENOTSUP);
    }

    // 8. Near-FLT_MAX float32 weight-narrowing and accumulation rounding on Symmetry::none passes.
    {
        const std::vector<uint64_t> wide = {2048, 2560};
        const size_t nw = product(wide);
        const std::vector<double> normal_large = make_data(n, 12);
        const std::vector<double> normal_wide = make_data(nw, 13);

        auto check_flt_max_guard = [&](const std::vector<uint64_t>& shape, const std::vector<double>& data,
                                       const std::vector<double>& normal_data, const std::vector<Spec>& passes) {
            CHECK(run(normal_data, shape, SM_FLOAT64, passes, SM_EXECUTION_AUTO, SM_PRECISION_FAST).info.engine ==
                  SM_EXECUTION_GPU);
            const Outcome ref = run(data, shape, SM_FLOAT64, passes, SM_EXECUTION_CPU, SM_PRECISION_FAST);
            CHECK(ref.status == SM_OK);
            const sm_execution_info before = counters();
            const Outcome auto_out = run(data, shape, SM_FLOAT64, passes, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
            CHECK(auto_out.status == SM_OK && auto_out.info.engine == SM_EXECUTION_CPU);
            CHECK(auto_out.info.compute_dtype == SM_FLOAT64 && auto_out.info.reason[0] != '\0');
            CHECK(counters().gpu_operations == before.gpu_operations);
            CHECK(auto_out.values == ref.values);
            for (double v : auto_out.values) CHECK(std::isfinite(v));
            CHECK(run(data, shape, SM_FLOAT64, passes, SM_EXECUTION_GPU, SM_PRECISION_FAST).status == SM_ENOTSUP);
        };

        const double flt_max = static_cast<double>(FLT_MAX);
        for (double val : {flt_max, 3.402823e38}) {
            const std::vector<double> data_large(n, val);
            const std::vector<double> data_wide(nw, val);
            for (uint64_t sz : {uint64_t{10}, uint64_t{20}, uint64_t{24}}) {
                Spec p0, p1;
                p0.kind = SM_NDIMAGE_UNIFORM;
                p0.axis = 0;
                p0.size = sz;
                p1.kind = SM_NDIMAGE_UNIFORM;
                p1.axis = 1;
                p1.size = sz;
                check_flt_max_guard(large, data_large, normal_large, {p0, p1});
            }
            {
                Spec u1d;
                u1d.kind = SM_NDIMAGE_UNIFORM;
                u1d.axis = 0;
                u1d.size = 10;
                check_flt_max_guard(large, data_large, normal_large, {u1d});
            }
            {
                Spec c8;
                c8.axis = 0;
                c8.weights = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3};
                check_flt_max_guard(wide, data_wide, normal_wide, {c8});
            }
            {
                Spec c6;
                c6.axis = 0;
                c6.weights = {0.1, 0.1, 0.1, 0.1, 0.1, 0.5};
                check_flt_max_guard(wide, data_wide, normal_wide, {c6});
            }
        }

        // Constant cval = FLT_MAX on unit input with even-sized uniform(10).
        {
            const std::vector<double> unit(n, 1.0);
            Spec n0, n1;
            n0.kind = SM_NDIMAGE_UNIFORM;
            n0.axis = 0;
            n0.size = 10;
            n0.mode = SM_NDIMAGE_CONSTANT;
            n0.cval = 0.0;
            n1.kind = SM_NDIMAGE_UNIFORM;
            n1.axis = 1;
            n1.size = 10;
            n1.mode = SM_NDIMAGE_CONSTANT;
            n1.cval = 0.0;
            CHECK(run(unit, large, SM_FLOAT64, {n0, n1}, SM_EXECUTION_AUTO, SM_PRECISION_FAST).info.engine ==
                  SM_EXECUTION_GPU);

            Spec c0 = n0, c1 = n1;
            c0.cval = flt_max;
            c1.cval = flt_max;
            const Outcome ref = run(unit, large, SM_FLOAT64, {c0, c1}, SM_EXECUTION_CPU, SM_PRECISION_FAST);
            CHECK(ref.status == SM_OK);
            const sm_execution_info before = counters();
            const Outcome auto_out = run(unit, large, SM_FLOAT64, {c0, c1}, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
            CHECK(auto_out.status == SM_OK && auto_out.info.engine == SM_EXECUTION_CPU);
            CHECK(auto_out.info.compute_dtype == SM_FLOAT64 && auto_out.info.reason[0] != '\0');
            CHECK(counters().gpu_operations == before.gpu_operations);
            CHECK(auto_out.values == ref.values);
            for (double v : auto_out.values) CHECK(std::isfinite(v));
            CHECK(run(unit, large, SM_FLOAT64, {c0, c1}, SM_EXECUTION_GPU, SM_PRECISION_FAST).status == SM_ENOTSUP);
        }
    }
}

void test_output_dtype_to() {
    // Empty array with explicit out_dtype publishes out_dtype.
    {
        const std::vector<uint64_t> empty_shape = {0, 5};
        const sm_array empty = describe(empty_shape, SM_FLOAT32, nullptr);
        Spec s;
        s.axis = 1;
        s.weights = {0.25, 0.5, 0.25};
        const auto passes = to_passes({s});
        const auto opts = route(SM_EXECUTION_CPU, SM_PRECISION_STRICT);
        sm_array_result result{};
        CHECK(sm_ndimage_separable_filter_to(ctx, &empty, passes.data(), 1, SM_FLOAT64, &opts, &result, nullptr) ==
              SM_OK);
        CHECK(result.dtype == SM_FLOAT64 && result.ndim == 2 && result.shape[0] == 0 && result.shape[1] == 5);
        sm_buffer_release(result.buffer);
    }

    // Invalid out_dtype returns SM_ENOTSUP and leaves result/info untouched.
    {
        const std::vector<uint64_t> shape = {4, 5};
        const std::vector<double> data = make_data(product(shape), 42);
        const sm_array input = describe(shape, SM_FLOAT64, data.data());
        Spec s;
        s.axis = 1;
        s.weights = {1.0};
        const auto passes = to_passes({s});
        const auto opts = route(SM_EXECUTION_CPU, SM_PRECISION_STRICT);
        for (uint32_t bad_dtype : {uint32_t{SM_COMPLEX64}, uint32_t{SM_COMPLEX128}}) {
            sm_array_result res;
            std::memset(&res, 0x5a, sizeof(res));
            const sm_array_result before = res;
            CHECK(sm_ndimage_separable_filter_to(ctx, &input, passes.data(), 1, bad_dtype, &opts, &res, nullptr) ==
                  SM_ENOTSUP);
            CHECK(std::memcmp(&res, &before, sizeof(res)) == 0);
        }
        {
            sm_array_result res;
            std::memset(&res, 0x5a, sizeof(res));
            const sm_array_result before = res;
            CHECK(sm_ndimage_separable_filter_to(ctx, &input, passes.data(), 1, 99, &opts, &res, nullptr) == SM_EINVAL);
            CHECK(std::memcmp(&res, &before, sizeof(res)) == 0);
        }
    }

    // Cross-float combinations (float64->float32, float32->float64) on CPU and GPU.
    const std::vector<uint64_t> shape = {29, 37};
    const std::vector<double> exact = make_data(product(shape), 55);
    Spec c0, c1, u0;
    c0.axis = 0;
    c0.weights = gaussian(4, 1.2, 0);
    c0.mode = SM_NDIMAGE_REFLECT;
    c1.axis = 1;
    c1.weights = ramp(6);
    c1.mode = SM_NDIMAGE_CONSTANT;
    c1.cval = -1.25;
    u0.kind = SM_NDIMAGE_UNIFORM;
    u0.axis = 1;
    u0.size = 5;
    u0.mode = SM_NDIMAGE_WRAP;

    for (const auto& specs : {std::vector<Spec>{c0}, {c0, c1}, {c0, u0, c1}}) {
        for (uint32_t in_dtype : {SM_FLOAT64, SM_FLOAT32}) {
            for (uint32_t out_dtype : {SM_FLOAT64, SM_FLOAT32}) {
                std::vector<double> data = exact;
                if (in_dtype == SM_FLOAT32) {
                    for (double& v : data) v = static_cast<double>(static_cast<float>(v));
                }
                const bool out_single = out_dtype == SM_FLOAT32;
                std::vector<double> expected = data;
                for (const Spec& sp : specs) expected = reference_pass(expected, shape, sp, out_single);

                const Outcome cpu = run_to(data, shape, in_dtype, out_dtype, specs, SM_EXECUTION_CPU, SM_PRECISION_STRICT);
                CHECK(cpu.status == SM_OK && cpu.info.engine == SM_EXECUTION_CPU && cpu.info.compute_dtype == SM_FLOAT64);
                const double cpu_bound = budget(data, shape, specs, out_single ? unit32 : unit64, true);
                CHECK(max_difference(cpu.values, expected) <= cpu_bound);

                if (have_gpu) {
                    const sm_execution_info before = counters();
                    const Outcome gpu = run_to(data, shape, in_dtype, out_dtype, specs, SM_EXECUTION_GPU, SM_PRECISION_FAST);
                    const sm_execution_info after = counters();
                    CHECK(gpu.status == SM_OK && gpu.info.engine == SM_EXECUTION_GPU &&
                          gpu.info.compute_dtype == SM_FLOAT32);
                    CHECK(after.gpu_operations == before.gpu_operations + 1);
                    CHECK(after.gpu_command_buffers == before.gpu_command_buffers + 1);
                    const double gpu_bound = budget(data, shape, specs, unit32, false);
                    CHECK(max_difference(gpu.values, cpu.values) <= gpu_bound);
                }
            }
        }
    }

    // Overflow guard rails with cross-float out_dtype (float64->float32 and float32->float64).
    if (have_gpu) {
        const std::vector<uint64_t> large = {1536, 2048};
        const size_t n = product(large);
        Spec g0, g1;
        g0.axis = 0;
        g1.axis = 1;
        g0.weights = g1.weights = gaussian(12, 3.0, 0);

        std::vector<double> over(n, 1.0);
        over[0] = 1e39;
        const Outcome cpu_ref = run_to(over, large, SM_FLOAT64, SM_FLOAT32, {g0, g1}, SM_EXECUTION_CPU, SM_PRECISION_FAST);
        const sm_execution_info before = counters();
        const Outcome auto_out = run_to(over, large, SM_FLOAT64, SM_FLOAT32, {g0, g1}, SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        CHECK(auto_out.status == SM_OK && auto_out.info.engine == SM_EXECUTION_CPU &&
              auto_out.info.compute_dtype == SM_FLOAT64 && auto_out.info.reason[0] != '\0');
        CHECK(counters().gpu_operations == before.gpu_operations);
        CHECK(auto_out.values == cpu_ref.values);
        CHECK(run_to(over, large, SM_FLOAT64, SM_FLOAT32, {g0, g1}, SM_EXECUTION_GPU, SM_PRECISION_FAST).status ==
              SM_ENOTSUP);

        // float32 -> float64 with cval = 1e39 must also re-route AUTO to CPU and decline GPU.
        const std::vector<double> f32_data = make_data(n, 77);
        Spec cpass = g0;
        cpass.mode = SM_NDIMAGE_CONSTANT;
        cpass.cval = 1e39;
        const Outcome c_auto = run_to(f32_data, large, SM_FLOAT32, SM_FLOAT64, {cpass, g1}, SM_EXECUTION_AUTO,
                                      SM_PRECISION_FAST);
        CHECK(c_auto.status == SM_OK && c_auto.info.engine == SM_EXECUTION_CPU &&
              c_auto.info.compute_dtype == SM_FLOAT64 && c_auto.info.reason[0] != '\0');
        CHECK(run_to(f32_data, large, SM_FLOAT32, SM_FLOAT64, {cpass, g1}, SM_EXECUTION_GPU, SM_PRECISION_FAST).status ==
              SM_ENOTSUP);
    }
}

void test_overhanging_windows() {
    const uint32_t modes[] = {SM_NDIMAGE_NEAREST, SM_NDIMAGE_WRAP, SM_NDIMAGE_REFLECT, SM_NDIMAGE_MIRROR,
                              SM_NDIMAGE_CONSTANT};
    const std::vector<std::vector<uint64_t>> shapes = {{1}, {1, 17}, {17, 1}, {2, 3}, {7, 19}, {3, 5, 4}};
    uint32_t seed = 9000;
    for (const auto& shape : shapes) {
        for (uint32_t axis = 0; axis < shape.size(); ++axis) {
            const uint64_t len = shape[axis];
            for (size_t taps : {size_t{5}, size_t{19}, size_t{65}}) {
                if (taps <= len) continue;
                const std::vector<std::vector<double>> wsets = {
                    ramp(taps),
                    gaussian(static_cast<int>(taps / 2), 4.0, 0),
                    gaussian(static_cast<int>(taps / 2), 4.0, 1),
                };
                for (const auto& w : wsets) {
                    for (uint32_t mode : modes) {
                        for (int64_t origin : origins_of(w.size())) {
                            Spec spec;
                            spec.axis = axis;
                            spec.mode = mode;
                            spec.origin = origin;
                            spec.weights = w;
                            spec.cval = -0.75;
                            compare(shape, {spec}, ++seed);
                        }
                    }
                }
            }
        }
    }

    // Special values (NaN, Inf, -Inf, cval=Inf) on overhanging windows (GPU vs reference).
    if (have_gpu) {
        const std::vector<uint64_t> shape = {8, 5};
        std::vector<double> data = make_data(product(shape), 9500);
        for (double& v : data) v = static_cast<double>(static_cast<float>(v));
        data[0] = std::numeric_limits<double>::quiet_NaN();
        data[7] = std::numeric_limits<double>::infinity();
        data[19] = -std::numeric_limits<double>::infinity();
        const std::vector<double> odd_w = {-0.5, -0.25, 0.0, 0.25, 0.5};
        const std::vector<double> sym_w = {0.1, 0.2, 0.4, 0.2, 0.1};
        const std::vector<double> asym_w = ramp(13);

        for (uint32_t mode : modes) {
            for (const auto& w : {odd_w, sym_w, asym_w}) {
                Spec sp;
                sp.axis = 1;
                sp.mode = mode;
                sp.weights = w;
                sp.cval = std::numeric_limits<double>::infinity();
                const Outcome gpu = run(data, shape, SM_FLOAT32, {sp}, SM_EXECUTION_GPU, SM_PRECISION_FAST);
                CHECK(gpu.status == SM_OK && gpu.info.engine == SM_EXECUTION_GPU);
                const std::vector<double> ref = reference_pass(data, shape, sp, true);
                for (size_t i = 0; i < ref.size(); ++i) {
                    CHECK(std::isnan(gpu.values[i]) == std::isnan(ref[i]));
                    if (!std::isnan(ref[i])) {
                        if (std::isinf(ref[i])) {
                            CHECK(gpu.values[i] == ref[i]);
                        } else {
                            CHECK(std::fabs(gpu.values[i] - ref[i]) <= 1e-4 * std::max(1.0, std::fabs(ref[i])));
                        }
                    }
                }
            }
        }
    }
}

void test_uniform_neon_and_tiles() {
    // Independent exact scalar NI_UniformFilter1D running-sum reference (must match bit-for-bit!).
    auto ref_uniform_running = [](const std::vector<double>& in, const std::vector<uint64_t>& shape,
                                  const Spec& spec, bool out_single) {
        const int64_t length = static_cast<int64_t>(shape[spec.axis]);
        uint64_t inner = 1, outer = 1;
        for (size_t a = 0; a < spec.axis; ++a) outer *= shape[a];
        for (size_t a = spec.axis + 1; a < shape.size(); ++a) inner *= shape[a];
        const int64_t taps = static_cast<int64_t>(spec.size);
        const int64_t before = taps / 2 + spec.origin;
        const double size = static_cast<double>(taps);
        std::vector<double> out(in.size());
        for (uint64_t o = 0; o < outer; ++o) {
            for (uint64_t c = 0; c < inner; ++c) {
                const double* line = in.data() + o * static_cast<uint64_t>(length) * inner + c;
                auto at = [&](int64_t p) {
                    const int64_t q = fold(p, length, spec.mode);
                    return q < 0 ? spec.cval : line[static_cast<uint64_t>(q) * inner];
                };
                double sum = 0.0;
                for (int64_t k = 0; k < taps; ++k) sum = sum + at(k - before);
                out[o * static_cast<uint64_t>(length) * inner + c] =
                    out_single ? static_cast<double>(static_cast<float>(sum / size)) : (sum / size);
                for (int64_t y = 1; y < length; ++y) {
                    sum = sum + (at(y - before + taps - 1) - at(y - 1 - before));
                    const double v = sum / size;
                    out[o * static_cast<uint64_t>(length) * inner + c + static_cast<uint64_t>(y) * inner] =
                        out_single ? static_cast<double>(static_cast<float>(v)) : v;
                }
            }
        }
        return out;
    };

    const uint32_t modes[] = {SM_NDIMAGE_NEAREST, SM_NDIMAGE_WRAP, SM_NDIMAGE_REFLECT, SM_NDIMAGE_MIRROR,
                              SM_NDIMAGE_CONSTANT};
    uint32_t seed = 12000;
    // Test line counts 1..17 across stride==1 (exercises 4-line NEON bundles + 0..3 remainder lines)
    // and overhanging + non-overhanging window sizes.
    for (uint64_t lines : {uint64_t{1}, uint64_t{3}, uint64_t{4}, uint64_t{5}, uint64_t{7}, uint64_t{8},
                           uint64_t{9}, uint64_t{15}, uint64_t{17}}) {
        for (uint64_t len : {uint64_t{7}, uint64_t{33}}) {
            const std::vector<uint64_t> shape = {lines, len};
            const std::vector<double> exact = make_data(product(shape), ++seed);
            for (uint64_t sz : {uint64_t{3}, uint64_t{9}, uint64_t{15}}) {
                const uint32_t mode = modes[(lines + sz) % 5];
                Spec sp;
                sp.kind = SM_NDIMAGE_UNIFORM;
                sp.axis = 1;
                sp.mode = mode;
                sp.size = sz;
                sp.origin = (sz % 2) ? 0 : -1;
                sp.cval = 2.25;
                for (uint32_t in_dt : {SM_FLOAT64, SM_FLOAT32}) {
                    for (uint32_t out_dt : {SM_FLOAT64, SM_FLOAT32}) {
                        std::vector<double> data = exact;
                        if (in_dt == SM_FLOAT32) for (double& v : data) v = static_cast<double>(static_cast<float>(v));
                        const auto expected = ref_uniform_running(data, shape, sp, out_dt == SM_FLOAT32);
                        const Outcome cpu =
                            run_to(data, shape, in_dt, out_dt, {sp}, SM_EXECUTION_CPU, SM_PRECISION_STRICT);
                        CHECK(cpu.status == SM_OK && cpu.values == expected);
                    }
                }
            }
        }
    }

    // Test uniform_tiles scaling across wide inner dimensions (axis=0).
    for (uint64_t inner : {uint64_t{64}, uint64_t{257}, uint64_t{4096}}) {
        const std::vector<uint64_t> shape = {25, inner};
        const std::vector<double> exact = make_data(product(shape), ++seed);
        Spec sp;
        sp.kind = SM_NDIMAGE_UNIFORM;
        sp.axis = 0;
        sp.mode = SM_NDIMAGE_REFLECT;
        sp.size = 9;
        for (uint32_t in_dt : {SM_FLOAT64, SM_FLOAT32}) {
            std::vector<double> data = exact;
            if (in_dt == SM_FLOAT32) for (double& v : data) v = static_cast<double>(static_cast<float>(v));
            const auto expected = ref_uniform_running(data, shape, sp, in_dt == SM_FLOAT32);
            const Outcome cpu = run(data, shape, in_dt, {sp}, SM_EXECUTION_CPU, SM_PRECISION_STRICT);
            CHECK(cpu.status == SM_OK && cpu.values == expected);
        }
    }
}

void test_integer_and_bool_dtypes() {
    auto sat_i32 = [](double x) -> int32_t {
        if (std::isnan(x)) return 0;
        if (x >= 2147483647.0) return INT32_MAX;
        if (x <= -2147483648.0) return INT32_MIN;
        return static_cast<int32_t>(x);
    };
    auto sat_u32 = [](double x) -> uint32_t {
        if (!(x >= 0.0)) return 0u;
        if (x >= 4294967295.0) return UINT32_MAX;
        return static_cast<uint32_t>(x);
    };
    auto sat_i64 = [](double x) -> int64_t {
        if (std::isnan(x)) return 0;
        if (x >= 9223372036854775808.0) return INT64_MAX;
        if (x <= -9223372036854775808.0) return INT64_MIN;
        return static_cast<int64_t>(x);
    };
    auto sat_u64 = [](double x) -> uint64_t {
        if (!(x >= 0.0)) return 0u;
        if (x >= 18446744073709551616.0) return UINT64_MAX;
        return static_cast<uint64_t>(x);
    };
    auto ref_store = [&](double x, uint32_t dtype, void* dst, size_t idx) {
        switch (dtype) {
        case SM_BOOL:
        case SM_UINT8: static_cast<uint8_t*>(dst)[idx] = static_cast<uint8_t>(sat_i32(x)); break;
        case SM_UINT16: static_cast<uint16_t*>(dst)[idx] = static_cast<uint16_t>(sat_i32(x)); break;
        case SM_UINT32: static_cast<uint32_t*>(dst)[idx] = sat_u32(x); break;
        case SM_UINT64: static_cast<uint64_t*>(dst)[idx] = sat_u64(x); break;
        case SM_INT8: static_cast<int8_t*>(dst)[idx] = static_cast<int8_t>(static_cast<uint8_t>(sat_i32(x))); break;
        case SM_INT16: static_cast<int16_t*>(dst)[idx] = static_cast<int16_t>(static_cast<uint16_t>(sat_i32(x))); break;
        case SM_INT32: static_cast<int32_t*>(dst)[idx] = sat_i32(x); break;
        case SM_INT64: static_cast<int64_t*>(dst)[idx] = sat_i64(x); break;
        case SM_FLOAT32: static_cast<float*>(dst)[idx] = static_cast<float>(x); break;
        case SM_FLOAT64: static_cast<double*>(dst)[idx] = x; break;
        }
    };
    auto dtype_bytes = [](uint32_t dtype) -> size_t {
        switch (dtype) {
        case SM_BOOL: case SM_INT8: case SM_UINT8: return 1;
        case SM_INT16: case SM_UINT16: return 2;
        case SM_FLOAT32: case SM_INT32: case SM_UINT32: return 4;
        default: return 8;
        }
    };
    auto load_f64 = [](const void* src, uint32_t dtype, size_t idx) -> double {
        switch (dtype) {
        case SM_BOOL:
        case SM_UINT8: return static_cast<double>(static_cast<const uint8_t*>(src)[idx]);
        case SM_UINT16: return static_cast<double>(static_cast<const uint16_t*>(src)[idx]);
        case SM_UINT32: return static_cast<double>(static_cast<const uint32_t*>(src)[idx]);
        case SM_UINT64: return static_cast<double>(static_cast<const uint64_t*>(src)[idx]);
        case SM_INT8: return static_cast<double>(static_cast<const int8_t*>(src)[idx]);
        case SM_INT16: return static_cast<double>(static_cast<const int16_t*>(src)[idx]);
        case SM_INT32: return static_cast<double>(static_cast<const int32_t*>(src)[idx]);
        case SM_INT64: return static_cast<double>(static_cast<const int64_t*>(src)[idx]);
        case SM_FLOAT32: return static_cast<double>(static_cast<const float*>(src)[idx]);
        default: return static_cast<const double*>(src)[idx];
        }
    };

    const uint32_t int_bool_dtypes[] = {SM_BOOL,   SM_INT8,   SM_INT16, SM_INT32, SM_INT64,
                                        SM_UINT8,  SM_UINT16, SM_UINT32, SM_UINT64};
    const std::vector<uint64_t> shape = {13, 17};
    const size_t n = product(shape);

    // 1. All 9 int/bool dtypes on multi-pass correlate + uniform: bit-exact to SciPy NI_LineBufferToArray.
    Spec s_corr, s_uni;
    s_corr.kind = SM_NDIMAGE_CORRELATE;
    s_corr.axis = 1;
    s_corr.mode = SM_NDIMAGE_CONSTANT;
    s_corr.cval = 2.0;
    s_corr.weights = {-1.0, 2.0, -1.0}; // intentionally produces negative values!
    s_uni.kind = SM_NDIMAGE_UNIFORM;
    s_uni.axis = 0;
    s_uni.mode = SM_NDIMAGE_REFLECT;
    s_uni.size = 3;

    for (uint32_t dt : int_bool_dtypes) {
        std::vector<uint8_t> raw_in(n * dtype_bytes(dt), 0);
        for (size_t i = 0; i < n; ++i) {
            if (dt == SM_BOOL) {
                raw_in[i] = static_cast<uint8_t>(i % 4); // raw non-0/1 bytes preserved across passes
            } else {
                const double v = (dt >= SM_INT8 && dt <= SM_INT64) ? static_cast<double>(static_cast<int>(i % 25) - 10)
                                                                    : static_cast<double>(i % 25);
                ref_store(v, dt, raw_in.data(), i);
            }
        }

        // Reference two-pass evaluation with intermediate quantization to `dt`.
        std::vector<double> work(n);
        for (size_t i = 0; i < n; ++i) work[i] = load_f64(raw_in.data(), dt, i);
        std::vector<uint8_t> ref_bytes(n * dtype_bytes(dt), 0);
        for (const Spec& sp : {s_corr, s_uni}) {
            const auto next_f64 = reference_pass(work, shape, sp, false);
            for (size_t i = 0; i < n; ++i) {
                ref_store(next_f64[i], dt, ref_bytes.data(), i);
                work[i] = load_f64(ref_bytes.data(), dt, i);
            }
        }

        const sm_array in_arr = describe(shape, dt, raw_in.data());
        const std::vector<Spec> specs = {s_corr, s_uni};
        const auto passes = to_passes(specs);
        const auto opts = route(SM_EXECUTION_CPU, SM_PRECISION_STRICT);
        sm_array_result res{};
        sm_route_info info{};
        info.struct_size = sizeof(info);
        info.abi_version = SM_ARRAY_ABI_VERSION;
        CHECK(sm_ndimage_separable_filter(ctx, &in_arr, passes.data(), static_cast<uint32_t>(passes.size()), &opts,
                                          &res, &info) == SM_OK);
        CHECK(info.engine == SM_EXECUTION_CPU && info.compute_dtype == SM_FLOAT64);
        CHECK(res.dtype == dt);
        const void* ptr = nullptr;
        uint64_t bytes = 0;
        CHECK(sm_buffer_data(res.buffer, &ptr, &bytes) == SM_OK);
        CHECK(bytes == ref_bytes.size());
        CHECK(std::memcmp(ptr, ref_bytes.data(), ref_bytes.size()) == 0);
        sm_buffer_release(res.buffer);
        ++cpu_cases;
    }

    // 1b. Regression: uniform_lines_generic NEON 4-line bundle (stride==1, lines>=4) tail boundary loop.
    {
        const std::vector<uint64_t> ushape = {19, 23};
        const size_t un = product(ushape);
        Spec s_uni_x;
        s_uni_x.kind = SM_NDIMAGE_UNIFORM;
        s_uni_x.axis = 1;
        s_uni_x.mode = SM_NDIMAGE_NEAREST;
        s_uni_x.size = 4;
        const std::vector<Spec> u_specs = {s_uni_x};
        const auto u_passes = to_passes(u_specs);
        for (uint32_t dt : {SM_BOOL, SM_INT8, SM_UINT8, SM_INT32}) {
            std::vector<uint8_t> raw_in(un * dtype_bytes(dt), 0);
            std::vector<double> f64_in(un);
            for (size_t i = 0; i < un; ++i) {
                if (dt == SM_BOOL) {
                    raw_in[i] = static_cast<uint8_t>(i % 5);
                } else {
                    const double v = (dt == SM_INT8 || dt == SM_INT32)
                                         ? static_cast<double>(static_cast<int>(i % 101) - 40)
                                         : static_cast<double>(i % 101);
                    ref_store(v, dt, raw_in.data(), i);
                }
                f64_in[i] = load_f64(raw_in.data(), dt, i);
            }
            const auto next_f64 = reference_pass(f64_in, ushape, s_uni_x, false);
            std::vector<uint8_t> u_ref(un * dtype_bytes(dt), 0);
            for (size_t i = 0; i < un; ++i) ref_store(next_f64[i], dt, u_ref.data(), i);

            const sm_array in_arr = describe(ushape, dt, raw_in.data());
            const auto opts = route(SM_EXECUTION_CPU, SM_PRECISION_STRICT);
            sm_array_result res{};
            CHECK(sm_ndimage_separable_filter(ctx, &in_arr, u_passes.data(), 1, &opts, &res, nullptr) == SM_OK);
            const void* ptr = nullptr;
            uint64_t bytes = 0;
            CHECK(sm_buffer_data(res.buffer, &ptr, &bytes) == SM_OK);
            CHECK(bytes == u_ref.size() && std::memcmp(ptr, u_ref.data(), u_ref.size()) == 0);
            sm_buffer_release(res.buffer);
            ++cpu_cases;
        }
    }

    // 2. Explicit negative -> unsigned NI_LineBufferToArray semantics check on single pass.
    {
        const uint8_t in_u8[4] = {0, 2, 0, 0};
        const uint32_t in_u32[4] = {0, 2, 0, 0};
        Spec sp;
        sp.axis = 0;
        sp.mode = SM_NDIMAGE_CONSTANT;
        sp.weights = {0.0, -1.0, 0.0}; // out[1] = -2.0
        const std::vector<Spec> specs = {sp};
        const auto passes = to_passes(specs);
        const auto opts = route(SM_EXECUTION_CPU, SM_PRECISION_STRICT);

        const sm_array a_u8 = describe({4}, SM_UINT8, in_u8);
        sm_array_result r_u8{};
        CHECK(sm_ndimage_separable_filter(ctx, &a_u8, passes.data(), 1, &opts, &r_u8, nullptr) == SM_OK);
        const void* p8 = nullptr;
        uint64_t b8 = 0;
        CHECK(sm_buffer_data(r_u8.buffer, &p8, &b8) == SM_OK);
        CHECK(static_cast<const uint8_t*>(p8)[1] == uint8_t{254}); // (uint8_t)(int32_t)(-2.0) == 254
        sm_buffer_release(r_u8.buffer);

        const sm_array a_u32 = describe({4}, SM_UINT32, in_u32);
        sm_array_result r_u32{};
        CHECK(sm_ndimage_separable_filter(ctx, &a_u32, passes.data(), 1, &opts, &r_u32, nullptr) == SM_OK);
        const void* p32 = nullptr;
        uint64_t b32 = 0;
        CHECK(sm_buffer_data(r_u32.buffer, &p32, &b32) == SM_OK);
        CHECK(static_cast<const uint32_t*>(p32)[1] == uint32_t{0}); // (uint32_t) saturates < 0 to 0
        sm_buffer_release(r_u32.buffer);
    }

    // 3. Cross-dtype out_dtype combinations on CPU STRICT.
    {
        const std::vector<uint8_t> u8_in(n, 5);
        const sm_array a_in = describe(shape, SM_UINT8, u8_in.data());
        Spec sp;
        sp.axis = 1;
        sp.mode = SM_NDIMAGE_REFLECT;
        sp.weights = {0.25, 0.5, 0.25};
        const std::vector<Spec> specs = {sp};
        const auto passes = to_passes(specs);
        const auto opts = route(SM_EXECUTION_CPU, SM_PRECISION_STRICT);
        for (uint32_t out_dt : {SM_FLOAT64, SM_FLOAT32, SM_INT16}) {
            sm_array_result res{};
            sm_route_info info{};
            info.struct_size = sizeof(info);
            info.abi_version = SM_ARRAY_ABI_VERSION;
            CHECK(sm_ndimage_separable_filter_to(ctx, &a_in, passes.data(), 1, out_dt, &opts, &res, &info) == SM_OK);
            CHECK(res.dtype == out_dt && info.engine == SM_EXECUTION_CPU && info.compute_dtype == SM_FLOAT64);
            const void* ptr = nullptr;
            uint64_t bytes = 0;
            CHECK(sm_buffer_data(res.buffer, &ptr, &bytes) == SM_OK);
            CHECK(bytes == n * dtype_bytes(out_dt));
            for (size_t i = 0; i < n; ++i) CHECK(load_f64(ptr, out_dt, i) == 5.0);
            sm_buffer_release(res.buffer);
        }
    }

    // 4. GPU fast mode on uint8/int16 + strict stays on CPU (SM_ENOTSUP on forced GPU) + |x| >= 2^24 guard.
    if (have_gpu) {
        const std::vector<uint64_t> gshape = {29, 37};
        const size_t gn = product(gshape);
        std::vector<uint8_t> u8_in(gn);
        for (size_t i = 0; i < gn; ++i) u8_in[i] = static_cast<uint8_t>((i * 7) % 200);

        Spec sp;
        sp.axis = 1;
        sp.mode = SM_NDIMAGE_REFLECT;
        sp.weights = {0.25, 0.5, 0.25};
        const std::vector<Spec> specs = {sp};
        const auto passes = to_passes(specs);
        const sm_array a_u8 = describe(gshape, SM_UINT8, u8_in.data());

        // STRICT keeps integers on CPU -> forced GPU returns SM_ENOTSUP.
        {
            const auto strict_gpu = route(SM_EXECUTION_GPU, SM_PRECISION_STRICT);
            sm_array_result res{};
            CHECK(sm_ndimage_separable_filter(ctx, &a_u8, passes.data(), 1, &strict_gpu, &res, nullptr) == SM_ENOTSUP);
        }

        // FAST uint8 -> uint8 runs on GPU in 1 command buffer and matches exact dyadic weights bit-exact.
        {
            sm_execution_info before{}, after{};
            before.struct_size = after.struct_size = sizeof(before);
            before.abi_version = after.abi_version = SM_RUNTIME_ABI_VERSION;
            CHECK(sm_context_execution_info(ctx, &before) == SM_OK);

            const auto fast_gpu = route(SM_EXECUTION_GPU, SM_PRECISION_FAST);
            sm_array_result gpu_res{}, cpu_res{};
            sm_route_info info{};
            info.struct_size = sizeof(info);
            info.abi_version = SM_ARRAY_ABI_VERSION;
            CHECK(sm_ndimage_separable_filter(ctx, &a_u8, passes.data(), 1, &fast_gpu, &gpu_res, &info) == SM_OK);
            CHECK(sm_context_execution_info(ctx, &after) == SM_OK);
            CHECK(info.engine == SM_EXECUTION_GPU && info.compute_dtype == SM_FLOAT32);
            CHECK(after.gpu_operations == before.gpu_operations + 1);
            CHECK(after.gpu_command_buffers == before.gpu_command_buffers + 1);

            const auto cpu_opts = route(SM_EXECUTION_CPU, SM_PRECISION_STRICT);
            CHECK(sm_ndimage_separable_filter(ctx, &a_u8, passes.data(), 1, &cpu_opts, &cpu_res, nullptr) == SM_OK);
            const void *gp = nullptr, *cp = nullptr;
            uint64_t gb = 0, cb = 0;
            CHECK(sm_buffer_data(gpu_res.buffer, &gp, &gb) == SM_OK);
            CHECK(sm_buffer_data(cpu_res.buffer, &cp, &cb) == SM_OK);
            CHECK(gb == gn && cb == gn);
            CHECK(std::memcmp(gp, cp, gn) == 0);
            sm_buffer_release(gpu_res.buffer);
            sm_buffer_release(cpu_res.buffer);
            ++gpu_cases;
        }

        // |x| >= 2^24 guard rail on int32 / uint32 / int64 / uint64.
        const std::vector<uint64_t> large = {2896, 2896};
        const size_t ln = product(large);
        Spec g0, g1;
        g0.axis = 0;
        g1.axis = 1;
        g0.weights = g1.weights = {0.25, 0.5, 0.25};
        const std::vector<Spec> lspecs = {g0, g1};
        const auto lpasses = to_passes(lspecs);

        std::vector<int32_t> i32(ln, 12);
        i32[0] = (int32_t{1} << 24) - 1;
        const sm_array a_ok = describe(large, SM_INT32, i32.data());
        {
            const auto auto_fast = route(SM_EXECUTION_AUTO, SM_PRECISION_FAST);
            sm_array_result res{};
            sm_route_info info{};
            info.struct_size = sizeof(info);
            info.abi_version = SM_ARRAY_ABI_VERSION;
            CHECK(sm_ndimage_separable_filter(ctx, &a_ok, lpasses.data(), 2, &auto_fast, &res, &info) == SM_OK);
            CHECK(info.engine == SM_EXECUTION_GPU && info.compute_dtype == SM_FLOAT32);
            sm_buffer_release(res.buffer);
        }

        for (int32_t bad_val : {int32_t{1} << 24, -(int32_t{1} << 24)}) {
            i32[ln / 2 + 7] = bad_val;
            const sm_array a_bad = describe(large, SM_INT32, i32.data());
            sm_execution_info before{}, after{};
            before.struct_size = after.struct_size = sizeof(before);
            before.abi_version = after.abi_version = SM_RUNTIME_ABI_VERSION;
            CHECK(sm_context_execution_info(ctx, &before) == SM_OK);

            const auto auto_fast = route(SM_EXECUTION_AUTO, SM_PRECISION_FAST);
            sm_array_result res{};
            sm_route_info info{};
            info.struct_size = sizeof(info);
            info.abi_version = SM_ARRAY_ABI_VERSION;
            CHECK(sm_ndimage_separable_filter(ctx, &a_bad, lpasses.data(), 2, &auto_fast, &res, &info) == SM_OK);
            CHECK(sm_context_execution_info(ctx, &after) == SM_OK);
            CHECK(info.engine == SM_EXECUTION_CPU && info.compute_dtype == SM_FLOAT64 && info.reason[0] != '\0');
            CHECK(after.gpu_operations == before.gpu_operations);
            sm_buffer_release(res.buffer);

            const auto gpu_fast = route(SM_EXECUTION_GPU, SM_PRECISION_FAST);
            sm_array_result gres{};
            CHECK(sm_ndimage_separable_filter(ctx, &a_bad, lpasses.data(), 2, &gpu_fast, &gres, nullptr) == SM_ENOTSUP);
        }
    }
}

void test_review_regressions() {
    // 1. SM_PRECISION_STRICT + SM_EXECUTION_GPU must reject float32 -> float64 (widening needs double).
    if (have_gpu) {
        const std::vector<uint64_t> shape = {16, 16};
        const std::vector<double> data = make_data(product(shape), 4242);
        Spec sp;
        sp.axis = 1;
        sp.weights = {0.25, 0.5, 0.25};
        const Outcome strict_gpu = run_to(data, shape, SM_FLOAT32, SM_FLOAT64, {sp}, SM_EXECUTION_GPU, SM_PRECISION_STRICT);
        CHECK(strict_gpu.status == SM_ENOTSUP);
    }

    // 2. float32 -> float32 gain guard: {1, -2, 1} on 2e38f cancels to 0 in double, overflows in float32.
    if (have_gpu) {
        const std::vector<uint64_t> large = {1536, 2048};
        const size_t n = product(large);
        std::vector<float> big(n, 2e38f);
        const sm_array a_big = describe(large, SM_FLOAT32, big.data());
        Spec sp;
        sp.axis = 1;
        sp.mode = SM_NDIMAGE_REFLECT;
        sp.weights = {1.0, -2.0, 1.0};
        const std::vector<Spec> specs = {sp};
        const auto passes = to_passes(specs);

        const auto gpu_fast = route(SM_EXECUTION_GPU, SM_PRECISION_FAST);
        sm_array_result gres{};
        CHECK(sm_ndimage_separable_filter(ctx, &a_big, passes.data(), 1, &gpu_fast, &gres, nullptr) == SM_ENOTSUP);

        const auto auto_fast = route(SM_EXECUTION_AUTO, SM_PRECISION_FAST);
        sm_array_result ares{};
        sm_route_info ainfo{};
        ainfo.struct_size = sizeof(ainfo);
        ainfo.abi_version = SM_ARRAY_ABI_VERSION;
        CHECK(sm_ndimage_separable_filter(ctx, &a_big, passes.data(), 1, &auto_fast, &ares, &ainfo) == SM_OK);
        CHECK(ainfo.engine == SM_EXECUTION_CPU && ainfo.compute_dtype == SM_FLOAT64);
        const void* ptr = nullptr;
        uint64_t bytes = 0;
        CHECK(sm_buffer_data(ares.buffer, &ptr, &bytes) == SM_OK);
        CHECK(static_cast<const float*>(ptr)[0] == 0.0f);
        sm_buffer_release(ares.buffer);
    }

    // 3. SM_BOOL raw non-canonical bytes on GPU match CPU bit-for-bit.
    if (have_gpu) {
        const std::vector<uint64_t> shape = {17, 23};
        const size_t n = product(shape);
        std::vector<uint8_t> raw_bool(n);
        for (size_t i = 0; i < n; ++i) raw_bool[i] = static_cast<uint8_t>((i * 5) % 8);
        const sm_array a_bool = describe(shape, SM_BOOL, raw_bool.data());
        Spec sp;
        sp.axis = 1;
        sp.mode = SM_NDIMAGE_REFLECT;
        sp.weights = {0.25, 0.5, 0.25};
        const std::vector<Spec> specs = {sp};
        const auto passes = to_passes(specs);

        const auto cpu_opts = route(SM_EXECUTION_CPU, SM_PRECISION_STRICT);
        const auto gpu_opts = route(SM_EXECUTION_GPU, SM_PRECISION_FAST);
        sm_array_result cres{}, gres{};
        sm_route_info ginfo{};
        ginfo.struct_size = sizeof(ginfo);
        ginfo.abi_version = SM_ARRAY_ABI_VERSION;
        CHECK(sm_ndimage_separable_filter(ctx, &a_bool, passes.data(), 1, &cpu_opts, &cres, nullptr) == SM_OK);
        CHECK(sm_ndimage_separable_filter(ctx, &a_bool, passes.data(), 1, &gpu_opts, &gres, &ginfo) == SM_OK);
        CHECK(ginfo.engine == SM_EXECUTION_GPU);
        const void *cp = nullptr, *gp = nullptr;
        uint64_t cb = 0, gb = 0;
        CHECK(sm_buffer_data(cres.buffer, &cp, &cb) == SM_OK);
        CHECK(sm_buffer_data(gres.buffer, &gp, &gb) == SM_OK);
        CHECK(cb == n && gb == n && std::memcmp(cp, gp, n) == 0);
        sm_buffer_release(cres.buffer);
        sm_buffer_release(gres.buffer);
    }

    // 4. Symmetric weights with NaN in upper half ({2e38, 0, NaN}): NI_Correlate1D classifies as
    //    Symmetry::even and multiplies weights[0] by (below + above), so effective gain is 4e38 -> GPU declines.
    if (have_gpu) {
        const std::vector<uint64_t> shape = {16, 16};
        const std::vector<double> ones(product(shape), 1.0);
        Spec sp;
        sp.axis = 1;
        sp.weights = {2e38, 0.0, std::numeric_limits<double>::quiet_NaN()};
        CHECK(run(ones, shape, SM_FLOAT32, {sp}, SM_EXECUTION_GPU, SM_PRECISION_FAST).status == SM_ENOTSUP);
    }
}
} // namespace

int main() {
    sm_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = SM_RUNTIME_ABI_VERSION;
#if defined(SM_TEST_METAL)
    options.storage_kind = SM_BUFFER_METAL;
    have_gpu = true;
#else
    options.storage_kind = SM_BUFFER_CPU;
#endif
    options.flags = SM_CONTEXT_POISON_UNWRITTEN;  // an element an engine fails to write shows up as garbage
    options.memory_limit_bytes = uint64_t{2} << 30;
    CHECK(sm_context_create(&options, &ctx) == SM_OK);

    test_validation();
    test_grid();
    test_sequences();
    test_larger();
    test_fused_widening_is_exact();
    test_routing();
    test_overflow_guard();
    test_output_dtype_to();
    test_overhanging_windows();
    test_uniform_neon_and_tiles();
    test_integer_and_bool_dtypes();
    test_review_regressions();

    sm_context_release(ctx);
    std::printf("ndimage filters ok: %lu CPU cases (worst error %.2g of its bound), %lu GPU cases (worst %.2g)%s\n",
                cpu_cases, worst_cpu, gpu_cases, worst_gpu, have_gpu ? "" : " [built without Metal: CPU engine only]");
    return 0;
}
