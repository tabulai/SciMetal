// GPU FFTs through the C ABI (ms_fft.h), checked against a double-precision
// reference transform: every plan class, the sizes MLX 0.31.1 gets wrong
// (2^21 .. 2^24), batches, several axes, padding, normalisation, precision
// handling and routing. Also runs under Metal API and shader validation, where
// threadgroup memory is halved and the largest sizes are skipped for time.
#include "ms_fft.h"
#include "core/array.hpp"
#include "fft/engine.hpp"
#include "fft/plan.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "fft gpu test failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

using namespace ms::detail;
using Complex = std::complex<double>;

namespace {
// Single-precision arithmetic through at most a few dozen butterfly stages and
// single-precision twiddles: the transforms measure 1e-7 .. 2.2e-6 relative to
// the output's norm (largest for Rader and Bluestein sizes near 2048).
constexpr double tolerance = 5e-6;

bool validated = false;  // running under shader validation: reduced limits, small sizes only
ms_context* context = nullptr;

// --- Reference transforms, double precision ----------------------------------
Complex unit(int64_t k, int64_t n, bool inverse) {
    k %= n;
    if (k < 0) k += n;
    const double angle = (inverse ? 2.0 : -2.0) * std::numbers::pi * static_cast<double>(k) / static_cast<double>(n);
    return {std::cos(angle), std::sin(angle)};
}

void radix2(std::vector<Complex>& x, bool inverse) {
    const size_t n = x.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
    for (size_t length = 2; length <= n; length <<= 1) {
        std::vector<Complex> roots(length / 2);
        for (size_t k = 0; k < length / 2; ++k) roots[k] = unit(static_cast<int64_t>(k), static_cast<int64_t>(length), inverse);
        for (size_t start = 0; start < n; start += length) {
            for (size_t k = 0; k < length / 2; ++k) {
                const Complex a = x[start + k], b = x[start + k + length / 2] * roots[k];
                x[start + k] = a + b;
                x[start + k + length / 2] = a - b;
            }
        }
    }
}

// Unnormalised DFT with exp(-/+ 2 pi i jk/n): direct for short inputs, radix 2
// for powers of two, Bluestein's algorithm over radix 2 otherwise.
std::vector<Complex> dft(const std::vector<Complex>& x, bool inverse) {
    const int64_t n = static_cast<int64_t>(x.size());
    if (std::has_single_bit(x.size())) {
        std::vector<Complex> y = x;
        radix2(y, inverse);
        return y;
    }
    if (n <= 256) {
        std::vector<Complex> y(x.size());
        for (int64_t k = 0; k < n; ++k) {
            Complex sum = 0;
            for (int64_t j = 0; j < n; ++j) sum += x[static_cast<size_t>(j)] * unit(j * k, n, inverse);
            y[static_cast<size_t>(k)] = sum;
        }
        return y;
    }
    // x[j] w^(jk) = c[k] (x[j] c[j]) conj(c[k - j]) with c[m] = exp(-/+ i pi m^2 / n).
    const size_t m = std::bit_ceil(static_cast<size_t>(2 * n - 1));
    std::vector<Complex> chirp(x.size()), a(m, Complex(0)), b(m, Complex(0));
    for (int64_t j = 0; j < n; ++j) chirp[static_cast<size_t>(j)] = unit(j * j, 2 * n, inverse);
    for (int64_t j = 0; j < n; ++j) {
        a[static_cast<size_t>(j)] = x[static_cast<size_t>(j)] * chirp[static_cast<size_t>(j)];
        b[static_cast<size_t>(j)] = std::conj(chirp[static_cast<size_t>(j)]);
        if (j) b[m - static_cast<size_t>(j)] = b[static_cast<size_t>(j)];
    }
    radix2(a, false);
    radix2(b, false);
    for (size_t i = 0; i < m; ++i) a[i] *= b[i];
    radix2(a, true);
    std::vector<Complex> y(x.size());
    for (int64_t k = 0; k < n; ++k)
        y[static_cast<size_t>(k)] = chirp[static_cast<size_t>(k)] * a[static_cast<size_t>(k)] / static_cast<double>(m);
    return y;
}

// --- Arrays -------------------------------------------------------------------
struct Tensor {
    std::vector<uint64_t> shape;
    std::vector<Complex> data;  // real arrays keep zero imaginary parts
    uint64_t size() const {
        uint64_t total = 1;
        for (const uint64_t extent : shape) total *= extent;
        return total;
    }
};

uint64_t random_state = 0x9E3779B97F4A7C15ull;
double uniform() {  // xorshift64*, in [-1, 1)
    random_state ^= random_state >> 12;
    random_state ^= random_state << 25;
    random_state ^= random_state >> 27;
    return static_cast<double>((random_state * 0x2545F4914F6CDD1Dull) >> 11) / 4503599627370496.0 - 1.0;
}

// Values are rounded to single precision so that both precisions transform
// exactly the same numbers.
Tensor random_tensor(std::vector<uint64_t> shape, bool complex) {
    Tensor tensor;
    tensor.shape = std::move(shape);
    tensor.data.resize(tensor.size());
    for (Complex& value : tensor.data)
        value = {static_cast<float>(uniform()), complex ? static_cast<float>(uniform()) : 0.0f};
    return tensor;
}

// Applies f to every 1-D line of `tensor` along `axis`, which may change length.
template<class Function> Tensor along(const Tensor& tensor, size_t axis, uint64_t new_length, Function&& f) {
    Tensor result;
    result.shape = tensor.shape;
    result.shape[axis] = new_length;
    result.data.resize(result.size());
    uint64_t inner = 1;
    for (size_t i = axis + 1; i < tensor.shape.size(); ++i) inner *= tensor.shape[i];
    const uint64_t length = tensor.shape[axis], outer = tensor.size() / (length * inner);
    std::vector<Complex> line(length);
    for (uint64_t o = 0; o < outer; ++o) {
        for (uint64_t i = 0; i < inner; ++i) {
            for (uint64_t k = 0; k < length; ++k) line[k] = tensor.data[(o * length + k) * inner + i];
            const std::vector<Complex> out = f(line);
            CHECK(out.size() == new_length);
            for (uint64_t k = 0; k < new_length; ++k) result.data[(o * new_length + k) * inner + i] = out[k];
        }
    }
    return result;
}

std::vector<Complex> resized(std::vector<Complex> line, uint64_t length) {
    line.resize(length, Complex(0));
    return line;
}

// scipy.fft's semantics for one call, in double precision.
Tensor reference(const Tensor& input, uint32_t kind, uint32_t norm, const std::vector<uint32_t>& axes, uint64_t n) {
    const size_t last = input.shape.size() - 1;
    const bool inverse = kind == MS_FFT_INVERSE || kind == MS_FFT_C2R;
    const bool has_last = std::find(axes.begin(), axes.end(), static_cast<uint32_t>(last)) != axes.end();
    if (!n) n = kind == MS_FFT_C2R ? 2 * (input.shape[last] - 1) : input.shape[last];
    Tensor work = input;
    double total = 1.0;
    const auto inner_axes = [&] {
        for (const uint32_t axis : axes) {
            if (axis == last) continue;
            total *= static_cast<double>(work.shape[axis]);
            work = along(work, axis, work.shape[axis], [&](const std::vector<Complex>& line) { return dft(line, inverse); });
        }
    };
    if (inverse) inner_axes();
    if (has_last) {
        total *= static_cast<double>(n);
        if (kind == MS_FFT_C2R) {
            work = along(work, last, n, [&](const std::vector<Complex>& line) {
                // The Hermitian spectrum of the first n/2 + 1 points; imaginary
                // parts SciPy ignores (x[0], and x[n/2] for even n) are dropped.
                const std::vector<Complex> half = resized(line, n / 2 + 1);
                std::vector<Complex> full(n);
                for (uint64_t k = 0; k <= n / 2; ++k) full[k] = half[k];
                full[0] = full[0].real();
                if (n % 2 == 0) full[n / 2] = full[n / 2].real();
                for (uint64_t k = 1; k < n - n / 2; ++k) full[n - k] = std::conj(half[k]);
                std::vector<Complex> out = dft(full, true);
                for (Complex& value : out) value = value.real();
                return out;
            });
        } else if (kind == MS_FFT_R2C) {
            work = along(work, last, n / 2 + 1, [&](const std::vector<Complex>& line) {
                return resized(dft(resized(line, n), false), n / 2 + 1);
            });
        } else {
            work = along(work, last, n, [&](const std::vector<Complex>& line) { return dft(resized(line, n), inverse); });
        }
    }
    if (!inverse) inner_axes();
    const double scale = norm == MS_FFT_NORM_ORTHO ? 1.0 / std::sqrt(total)
                         : (norm == MS_FFT_NORM_FORWARD) != inverse ? 1.0 / total : 1.0;
    for (Complex& value : work.data) value *= scale;
    return work;
}

// --- Calling the library -------------------------------------------------------
ms_route_options route_options(uint32_t execution = MS_EXECUTION_GPU, uint32_t precision = MS_PRECISION_FAST) {
    ms_route_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = MS_ARRAY_ABI_VERSION;
    options.execution = execution;
    options.precision = precision;
    return options;
}

ms_fft_options fft_options(uint32_t kind, uint32_t norm, const std::vector<uint32_t>& axes, uint64_t n) {
    ms_fft_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = MS_FFT_ABI_VERSION;
    options.kind = kind;
    options.norm = norm;
    options.axis_count = static_cast<uint32_t>(axes.size());
    for (size_t i = 0; i < axes.size(); ++i) options.axes[i] = axes[i];
    options.n = n;
    return options;
}

bool is_complex(uint32_t dtype) { return dtype == MS_COMPLEX64 || dtype == MS_COMPLEX128; }
bool is_double(uint32_t dtype) { return dtype == MS_FLOAT64 || dtype == MS_COMPLEX128; }

std::vector<uint8_t> pack(const Tensor& tensor, uint32_t dtype) {
    const size_t scalars = tensor.data.size() * (is_complex(dtype) ? 2 : 1);
    std::vector<uint8_t> bytes(scalars * (is_double(dtype) ? 8 : 4));
    for (size_t i = 0; i < tensor.data.size(); ++i) {
        const double parts[2] = {tensor.data[i].real(), tensor.data[i].imag()};
        for (size_t part = 0; part < (is_complex(dtype) ? 2u : 1u); ++part) {
            const size_t index = i * (is_complex(dtype) ? 2 : 1) + part;
            if (is_double(dtype)) std::memcpy(bytes.data() + index * 8, &parts[part], 8);
            else {
                const float value = static_cast<float>(parts[part]);
                std::memcpy(bytes.data() + index * 4, &value, 4);
            }
        }
    }
    return bytes;
}

struct Output {
    ms_status status = MS_OK;
    uint32_t dtype = 0;
    Tensor tensor;
    std::vector<uint8_t> bytes;
    ms_route_info info{};
};

Output call(const Tensor& input, uint32_t dtype, uint32_t kind, uint32_t norm, const std::vector<uint32_t>& axes,
            uint64_t n = 0, ms_route_options route = route_options()) {
    const std::vector<uint8_t> bytes = pack(input, dtype);
    ms_array array{};
    array.struct_size = sizeof(array);
    array.abi_version = MS_ARRAY_ABI_VERSION;
    array.data = bytes.data();
    array.dtype = dtype;
    array.ndim = static_cast<uint32_t>(input.shape.size());
    for (size_t i = 0; i < input.shape.size(); ++i) array.shape[i] = input.shape[i];
    const ms_fft_options options = fft_options(kind, norm, axes, n);
    Output output;
    output.info.struct_size = sizeof(output.info);
    output.info.abi_version = MS_ARRAY_ABI_VERSION;
    ms_array_result result{};
    output.status = ms_fft_transform(context, &array, &options, &route, &result, &output.info);
    if (output.status != MS_OK) {
        CHECK(result.buffer == nullptr && output.info.engine == 0);  // outputs untouched on failure
        return output;
    }
    output.dtype = result.dtype;
    output.tensor.shape.assign(result.shape, result.shape + result.ndim);
    const void* data = nullptr;
    uint64_t size = 0;
    CHECK(ms_buffer_data(result.buffer, &data, &size) == MS_OK);
    const uint64_t elements = output.tensor.size();
    const size_t width = (is_double(result.dtype) ? 8 : 4) * (is_complex(result.dtype) ? 2 : 1);
    CHECK(size == elements * width);
    output.bytes.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
    output.tensor.data.resize(elements);
    for (uint64_t i = 0; i < elements; ++i) {
        double parts[2] = {0.0, 0.0};
        for (size_t part = 0; part < (is_complex(result.dtype) ? 2u : 1u); ++part) {
            const size_t index = i * (is_complex(result.dtype) ? 2 : 1) + part;
            if (is_double(result.dtype)) std::memcpy(&parts[part], output.bytes.data() + index * 8, 8);
            else {
                float value;
                std::memcpy(&value, output.bytes.data() + index * 4, 4);
                parts[part] = value;
            }
        }
        output.tensor.data[i] = {parts[0], parts[1]};
    }
    ms_buffer_release(result.buffer);
    return output;
}

double relative_error(const Tensor& got, const Tensor& expected) {
    CHECK(got.shape == expected.shape);
    long double difference = 0, norm = 0;
    for (size_t i = 0; i < got.data.size(); ++i) {
        difference += std::norm(got.data[i] - expected.data[i]);
        norm += std::norm(expected.data[i]);
    }
    return static_cast<double>(std::sqrt(difference / std::max<long double>(norm, 1e-300L)));
}

std::string last_error() {
    char text[512] = {};
    ms_context_error_copy(context, text, sizeof(text));
    return text;
}

ms_execution_info execution_info() {
    ms_execution_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = MS_RUNTIME_ABI_VERSION;
    CHECK(ms_context_execution_info(context, &info) == MS_OK);
    return info;
}

double worst_error = 0;
int transforms_checked = 0;

// One native transform against the reference, with everything a successful
// call promises: dtype, shape, the route report, one command buffer.
double check(const Tensor& input, uint32_t dtype, uint32_t kind, uint32_t norm, const std::vector<uint32_t>& axes,
             uint64_t n = 0) {
    const ms_execution_info before = execution_info();
    const Output output = call(input, dtype, kind, norm, axes, n);
    if (output.status != MS_OK) {
        std::fprintf(stderr, "transform failed (status %d): %s\n", static_cast<int>(output.status), last_error().c_str());
        std::abort();
    }
    const ms_execution_info after = execution_info();
    CHECK(after.gpu_command_buffers == before.gpu_command_buffers + 1);
    CHECK(after.gpu_operations == before.gpu_operations + 1);
    const uint32_t expected_dtype = kind == MS_FFT_C2R ? (is_double(dtype) ? MS_FLOAT64 : MS_FLOAT32)
                                                       : (is_double(dtype) ? MS_COMPLEX128 : MS_COMPLEX64);
    CHECK(output.dtype == expected_dtype);
    CHECK(output.info.engine == MS_EXECUTION_GPU);
    CHECK(output.info.compute_dtype == (is_complex(dtype) ? MS_COMPLEX64 : MS_FLOAT32));
    const Tensor expected = reference(input, kind, norm, axes, n);
    const double error = relative_error(output.tensor, expected);
    if (!(error < tolerance)) {
        std::string shape;
        for (const uint64_t extent : input.shape) shape += std::to_string(extent) + " ";
        std::fprintf(stderr, "kind %u norm %u dtype %u shape [%s] n %llu: relative error %.3e\n", kind, norm, dtype,
                     shape.c_str(), static_cast<unsigned long long>(n), error);
        std::abort();
    }
    worst_error = std::max(worst_error, error);
    ++transforms_checked;
    return error;
}

// All four transforms of `rows` rows of length n along the last axis.
void check_length(uint64_t n, uint64_t rows, uint32_t precision_double = 0) {
    const uint32_t complex_dtype = precision_double ? MS_COMPLEX128 : MS_COMPLEX64;
    const uint32_t real_dtype = precision_double ? MS_FLOAT64 : MS_FLOAT32;
    const Tensor x = random_tensor({rows, n}, true);
    check(x, complex_dtype, MS_FFT_FORWARD, MS_FFT_NORM_BACKWARD, {1});
    check(x, complex_dtype, MS_FFT_INVERSE, MS_FFT_NORM_BACKWARD, {1});
    check(random_tensor({rows, n}, false), real_dtype, MS_FFT_R2C, MS_FFT_NORM_BACKWARD, {1});
    // General (not Hermitian-consistent) input: imag(x[0]) and imag(x[n/2]) must be ignored.
    check(random_tensor({rows, n / 2 + 1}, true), complex_dtype, MS_FFT_C2R, MS_FFT_NORM_BACKWARD, {1}, n);
}

void test_plan_classes() {
    // Stockham, including every radix and MLX's special cases; Rader; Bluestein.
    for (const uint64_t n : {uint64_t{2}, uint64_t{3}, uint64_t{4}, uint64_t{5}, uint64_t{6}, uint64_t{7}, uint64_t{8},
                             uint64_t{9}, uint64_t{11}, uint64_t{13}, uint64_t{16}, uint64_t{26}, uint64_t{30},
                             uint64_t{64}, uint64_t{77}, uint64_t{100}, uint64_t{128}, uint64_t{143}, uint64_t{243},
                             uint64_t{256}, uint64_t{500}, uint64_t{512}, uint64_t{1000}, uint64_t{1001},
                             uint64_t{1024}, uint64_t{2048}})
        check_length(n, 3);
    for (const uint64_t n : {uint64_t{17}, uint64_t{34}, uint64_t{47}, uint64_t{289}, uint64_t{1009}, uint64_t{1021},
                             uint64_t{2003}})
        check_length(n, 3);
    if (!validated) {
        // One kernel with the full 4096 elements of threadgroup memory, or with
        // more threads than a validated pipeline is granted (1982: Rader on 397).
        for (const uint64_t n : {uint64_t{1982}, uint64_t{2038}, uint64_t{2039}, uint64_t{2187}, uint64_t{3003},
                                 uint64_t{3159}, uint64_t{3645}, uint64_t{3969}, uint64_t{4095}, uint64_t{4096}})
            check_length(n, 3);
        // Sizes whose single kernel asks for more threads than its pipeline is
        // granted run on the replacement plan: Bluestein for Rader (1874, 1965),
        // a strided chain for Stockham (3900, 3960).
        for (const uint64_t n : {uint64_t{1874}, uint64_t{1965}, uint64_t{1983}, uint64_t{1990}, uint64_t{3900}, uint64_t{3960}})
            check_length(n, 2);
    }
    // Multi-pass: MLX's four-step, our strided chains.
    for (const uint64_t n : {uint64_t{5000}, uint64_t{6000}, uint64_t{8192}, uint64_t{19683}, uint64_t{30030},
                             uint64_t{65536}})
        check_length(n, 3);
    if (!validated) {
        for (const uint64_t n : {uint64_t{131072}, uint64_t{1000000}, uint64_t{1048576}}) check_length(n, 2);
    }
    std::printf("plan classes: %d transforms, worst relative error %.2e\n", transforms_checked, worst_error);
}

// The sizes MLX 0.31.1's own GPU FFT raises on (2^21, 2^22) or gets wrong
// (2^23, 2^24).
void test_large_powers_of_two() {
    if (validated) return;
    for (const int log2n : {21, 22, 23, 24}) {
        const uint64_t n = uint64_t{1} << log2n;
        const Tensor x = random_tensor({1, n}, true);
        const double forward = check(x, MS_COMPLEX64, MS_FFT_FORWARD, MS_FFT_NORM_BACKWARD, {1});
        const double inverse = check(x, MS_COMPLEX64, MS_FFT_INVERSE, MS_FFT_NORM_BACKWARD, {1});
        double real = 0, back = 0;
        if (log2n % 2 == 0) {
            real = check(random_tensor({1, n}, false), MS_FLOAT32, MS_FFT_R2C, MS_FFT_NORM_BACKWARD, {1});
            back = check(random_tensor({1, n / 2 + 1}, true), MS_COMPLEX64, MS_FFT_C2R, MS_FFT_NORM_BACKWARD, {1}, n);
        }
        std::printf("2^%d: fft %.2e ifft %.2e rfft %.2e irfft %.2e\n", log2n, forward, inverse, real, back);
    }
    // Two rows of 2^21 through the double-precision path.
    check(random_tensor({2, uint64_t{1} << 21}, true), MS_COMPLEX128, MS_FFT_FORWARD, MS_FFT_NORM_ORTHO, {1});
}

void test_batches() {
    // Odd and even row counts: the real kernels pair rows; short transforms share threadgroups.
    for (const uint64_t rows : {uint64_t{1}, uint64_t{2}, uint64_t{5}, uint64_t{64}, uint64_t{257}}) {
        check_length(8, rows);
        check_length(100, rows);
        check_length(1009, rows);
    }
    check_length(47, 1000);
    check_length(1024, 513);
    check_length(8192, 33);
    check_length(6000, 17);
    if (!validated) {
        check_length(4096, 300);
        check_length(65536, 20);
        // A batch large enough for the short-pass plan of a non-power-of-two length.
        check_length(10000, 500);
    }
    // Leading axes beyond one are part of the batch.
    check(random_tensor({3, 4, 5, 96}, true), MS_COMPLEX64, MS_FFT_FORWARD, MS_FFT_NORM_BACKWARD, {3});
    check(random_tensor({2, 3, 2, 2, 2, 2, 2, 48}, false), MS_FLOAT32, MS_FFT_R2C, MS_FFT_NORM_BACKWARD, {7});
    check(random_tensor({1000}, true), MS_COMPLEX64, MS_FFT_FORWARD, MS_FFT_NORM_BACKWARD, {0});
}

void test_padding_and_norms() {
    for (const uint32_t norm : {MS_FFT_NORM_BACKWARD, MS_FFT_NORM_ORTHO, MS_FFT_NORM_FORWARD}) {
        for (const uint32_t dtype : {MS_COMPLEX64, MS_COMPLEX128}) {
            const uint32_t real_dtype = dtype == MS_COMPLEX64 ? MS_FLOAT32 : MS_FLOAT64;
            const Tensor x = random_tensor({5, 96}, true), r = random_tensor({5, 96}, false);
            for (const uint64_t n : {uint64_t{0}, uint64_t{50}, uint64_t{96}, uint64_t{200}, uint64_t{1021}, uint64_t{6000}}) {
                check(x, dtype, MS_FFT_FORWARD, norm, {1}, n);
                check(x, dtype, MS_FFT_INVERSE, norm, {1}, n);
                check(r, real_dtype, MS_FFT_R2C, norm, {1}, n);
                check(x, dtype, MS_FFT_C2R, norm, {1}, n);
                // A real input is taken as complex by the complex transforms.
                check(r, real_dtype, MS_FFT_FORWARD, norm, {1}, n);
                check(r, real_dtype, MS_FFT_C2R, norm, {1}, n);
            }
            check(x, dtype, MS_FFT_C2R, norm, {1}, 95);   // odd output length
            check(x, dtype, MS_FFT_C2R, norm, {1}, 191);  // exactly 2 m - 1
        }
    }
}

void test_axes() {
    for (const uint32_t dtype : {MS_COMPLEX64, MS_COMPLEX128}) {
        const uint32_t real_dtype = dtype == MS_COMPLEX64 ? MS_FLOAT32 : MS_FLOAT64;
        const Tensor x = random_tensor({6, 10, 36}, true), r = random_tensor({6, 10, 36}, false);
        for (const uint32_t norm : {MS_FFT_NORM_BACKWARD, MS_FFT_NORM_ORTHO}) {
            for (const std::vector<uint32_t>& axes : std::vector<std::vector<uint32_t>>{
                     {0}, {1}, {2}, {0, 1}, {1, 2}, {2, 0}, {0, 1, 2}, {2, 1, 0}}) {
                check(x, dtype, MS_FFT_FORWARD, norm, axes);
                check(x, dtype, MS_FFT_INVERSE, norm, axes);
                check(r, real_dtype, MS_FFT_FORWARD, norm, axes);
                if (std::find(axes.begin(), axes.end(), 2u) != axes.end()) {
                    check(r, real_dtype, MS_FFT_R2C, norm, axes);
                    check(x, dtype, MS_FFT_C2R, norm, axes);
                    check(x, dtype, MS_FFT_C2R, norm, axes, 71);
                    check(r, real_dtype, MS_FFT_R2C, norm, axes, 50);
                    check(x, dtype, MS_FFT_FORWARD, norm, axes, 64);
                }
            }
        }
    }
    // Inner axes of every kind of length the strided chain takes, with tails
    // that the threadgroup width does and does not divide.
    check(random_tensor({1000, 3}, true), MS_COMPLEX64, MS_FFT_FORWARD, MS_FFT_NORM_BACKWARD, {0});
    check(random_tensor({2, 3, 5}, true), MS_COMPLEX64, MS_FFT_INVERSE, MS_FFT_NORM_BACKWARD, {0, 1});
    check(random_tensor({4096, 5}, true), MS_COMPLEX64, MS_FFT_INVERSE, MS_FFT_NORM_BACKWARD, {0});
    check(random_tensor({3, 6000, 7}, true), MS_COMPLEX64, MS_FFT_FORWARD, MS_FFT_NORM_BACKWARD, {1});
    check(random_tensor({143, 77}, true), MS_COMPLEX64, MS_FFT_FORWARD, MS_FFT_NORM_BACKWARD, {0, 1});
    check(random_tensor({256, 258}, false), MS_FLOAT32, MS_FFT_R2C, MS_FFT_NORM_BACKWARD, {0, 1});       // tail 130
    check(random_tensor({256, 129}, true), MS_COMPLEX64, MS_FFT_C2R, MS_FFT_NORM_BACKWARD, {0, 1});
    check(random_tensor({6, 30, 5000}, false), MS_FLOAT32, MS_FFT_R2C, MS_FFT_NORM_BACKWARD, {1, 2});    // real through a complex chain
    check(random_tensor({10, 30, 2501}, true), MS_COMPLEX64, MS_FFT_C2R, MS_FFT_NORM_BACKWARD, {1, 2});
    if (!validated) {
        check(random_tensor({1024, 1024}, true), MS_COMPLEX64, MS_FFT_FORWARD, MS_FFT_NORM_BACKWARD, {0, 1});
        check(random_tensor({4096, 2100}, false), MS_FLOAT64, MS_FFT_R2C, MS_FFT_NORM_ORTHO, {0, 1});     // large: short passes
        check(random_tensor({4096, 1051}, true), MS_COMPLEX128, MS_FFT_C2R, MS_FFT_NORM_ORTHO, {0, 1});
    }
    // One-point axes are identities.
    check(random_tensor({1, 96}, true), MS_COMPLEX64, MS_FFT_FORWARD, MS_FFT_NORM_ORTHO, {0, 1});
}

void test_round_trips() {
    std::vector<uint64_t> lengths = {12, 1000, 1021, 5000, 65536};
    if (!validated) lengths.insert(lengths.end(), {uint64_t{1} << 20, uint64_t{1} << 22, 3000000});
    for (const uint64_t n : lengths) {
        const uint64_t rows = n > 100000 ? 1 : 3;
        const Tensor x = random_tensor({rows, n}, true);
        const Output forward = call(x, MS_COMPLEX64, MS_FFT_FORWARD, MS_FFT_NORM_BACKWARD, {1});
        CHECK(forward.status == MS_OK);
        const Output back = call(forward.tensor, MS_COMPLEX64, MS_FFT_INVERSE, MS_FFT_NORM_BACKWARD, {1});
        CHECK(back.status == MS_OK && relative_error(back.tensor, x) < tolerance);
        const Tensor r = random_tensor({rows, n}, false);
        const Output spectrum = call(r, MS_FLOAT32, MS_FFT_R2C, MS_FFT_NORM_ORTHO, {1});
        CHECK(spectrum.status == MS_OK);
        const Output restored = call(spectrum.tensor, MS_COMPLEX64, MS_FFT_C2R, MS_FFT_NORM_ORTHO, {1}, n);
        CHECK(restored.status == MS_OK && relative_error(restored.tensor, r) < tolerance);
    }
    const Tensor image = random_tensor({96, 100}, false);
    const Output spectrum = call(image, MS_FLOAT64, MS_FFT_R2C, MS_FFT_NORM_BACKWARD, {0, 1});
    CHECK(spectrum.status == MS_OK && spectrum.dtype == MS_COMPLEX128);
    const Output restored = call(spectrum.tensor, MS_COMPLEX128, MS_FFT_C2R, MS_FFT_NORM_BACKWARD, {0, 1}, 100);
    CHECK(restored.status == MS_OK && restored.dtype == MS_FLOAT64 && relative_error(restored.tensor, image) < tolerance);
}

// A double-precision input whose values are single-precision numbers must
// give exactly the widened single-precision result: narrowing and widening
// add nothing.
void test_precision_paths() {
    for (const uint64_t n : {uint64_t{96}, uint64_t{1021}, uint64_t{8192}}) {
        const Tensor x = random_tensor({7, n}, true), r = random_tensor({7, n}, false);
        const uint64_t padded = n == 96 ? 100 : n == 1021 ? 1024 : 10000;  // zero padding takes the row-wise load
        struct Case { const Tensor* input; uint32_t single, wide, kind; uint64_t n; };
        for (const Case& each : {Case{&x, MS_COMPLEX64, MS_COMPLEX128, MS_FFT_FORWARD, 0},
                                 Case{&x, MS_COMPLEX64, MS_COMPLEX128, MS_FFT_INVERSE, 0},
                                 Case{&r, MS_FLOAT32, MS_FLOAT64, MS_FFT_R2C, 0},
                                 Case{&x, MS_COMPLEX64, MS_COMPLEX128, MS_FFT_C2R, n},
                                 Case{&x, MS_COMPLEX64, MS_COMPLEX128, MS_FFT_FORWARD, padded}}) {
            const Output single = call(*each.input, each.single, each.kind, MS_FFT_NORM_BACKWARD, {1}, each.n);
            const Output wide = call(*each.input, each.wide, each.kind, MS_FFT_NORM_BACKWARD, {1}, each.n);
            CHECK(single.status == MS_OK && wide.status == MS_OK);
            CHECK(is_double(wide.dtype) && !is_double(single.dtype) && wide.tensor.shape == single.tensor.shape);
            CHECK(wide.info.compute_dtype == single.info.compute_dtype);
            for (size_t i = 0; i < single.tensor.data.size(); ++i) CHECK(wide.tensor.data[i] == single.tensor.data[i]);
        }
    }
}

void test_routing_and_errors() {
    const Tensor small = random_tensor({4, 64}, true), real = random_tensor({4, 64}, false);
    // No native CPU FFT: forced CPU and AUTO below the crossover decline.
    CHECK(call(small, MS_COMPLEX64, MS_FFT_FORWARD, 0, {1}, 0, route_options(MS_EXECUTION_CPU)).status == MS_ENOTSUP);
    CHECK(call(small, MS_COMPLEX64, MS_FFT_FORWARD, 0, {1}, 0, route_options(MS_EXECUTION_AUTO)).status == MS_ENOTSUP);
    CHECK(last_error().find("crossover") != std::string::npos);
    // Strict precision keeps double inputs off the GPU, forced or not, but not single ones.
    CHECK(call(small, MS_COMPLEX128, MS_FFT_FORWARD, 0, {1}, 0, route_options(MS_EXECUTION_GPU, MS_PRECISION_STRICT)).status == MS_ENOTSUP);
    CHECK(call(small, MS_COMPLEX64, MS_FFT_FORWARD, 0, {1}, 0, route_options(MS_EXECUTION_GPU, MS_PRECISION_STRICT)).status == MS_OK);
    CHECK(call(small, MS_COMPLEX128, MS_FFT_FORWARD, 0, {1}, 0, route_options(MS_EXECUTION_GPU, MS_PRECISION_REFINED)).status == MS_OK);
    // AUTO takes a transform that is large enough, and reports it.
    const Tensor large = random_tensor({1024, 4096}, true);
    const Output automatic = call(large, MS_COMPLEX64, MS_FFT_FORWARD, 0, {1}, 0, route_options(MS_EXECUTION_AUTO));
    CHECK(automatic.status == MS_OK);
    CHECK(automatic.info.engine == MS_EXECUTION_GPU && automatic.info.compute_dtype == MS_COMPLEX64);
    const Output strict_auto = call(large, MS_COMPLEX128, MS_FFT_FORWARD, 0, {1}, 0, route_options(MS_EXECUTION_AUTO, MS_PRECISION_STRICT));
    CHECK(strict_auto.status == MS_ENOTSUP && last_error().find("strict") != std::string::npos);
    CHECK(ms_fft_auto_min_points() > 1024 && ms_fft_auto_min_points() < (uint64_t{1} << 20));
    // A threaded host alternative raises the crossover for batches.
    {
        const std::vector<uint8_t> bytes = pack(large, MS_COMPLEX64);
        ms_array array{};
        array.struct_size = sizeof(array);
        array.abi_version = MS_ARRAY_ABI_VERSION;
        array.data = bytes.data();
        array.dtype = MS_COMPLEX64;
        array.ndim = 2;
        array.shape[0] = large.shape[0];
        array.shape[1] = large.shape[1];
        ms_fft_options options = fft_options(MS_FFT_FORWARD, 0, {1}, 0);
        options.host_threads = 1u << 20;
        const ms_route_options route = route_options(MS_EXECUTION_AUTO);
        ms_array_result result{};
        CHECK(ms_fft_transform(context, &array, &options, &route, &result, nullptr) == MS_ENOTSUP);
        options.host_threads = 0;
        CHECK(ms_fft_transform(context, &array, &options, &route, &result, nullptr) == MS_OK);  // info is optional
        ms_buffer_release(result.buffer);
    }

    // Lengths without a correct plan decline instead of computing something else.
    for (const uint64_t n : {uint64_t{1}, uint64_t{3001}, uint64_t{4099}, uint64_t{100003}}) {
        CHECK(call(random_tensor({2, n}, true), MS_COMPLEX64, MS_FFT_FORWARD, 0, {1}).status == MS_ENOTSUP);
        CHECK(call(random_tensor({2, n}, false), MS_FLOAT32, MS_FFT_R2C, 0, {1}).status == MS_ENOTSUP);
        CHECK(call(random_tensor({2, 40}, true), MS_COMPLEX64, MS_FFT_C2R, 0, {1}, n).status == MS_ENOTSUP);
    }
    CHECK(call(random_tensor({17, 64}, true), MS_COMPLEX64, MS_FFT_FORWARD, 0, {0}).status == MS_ENOTSUP);   // inner axis needs radix factors
    CHECK(call(random_tensor({16, 524}, true), MS_COMPLEX64, MS_FFT_C2R, 0, {0, 1}, 1046).status == MS_ENOTSUP);  // Bluestein c2r with another axis
    CHECK(call(random_tensor({2, 1}, true), MS_COMPLEX64, MS_FFT_C2R, 0, {1}).status == MS_ENOTSUP);         // natural length 0
    Tensor empty;
    empty.shape = {0, 8};
    CHECK(call(empty, MS_COMPLEX64, MS_FFT_FORWARD, 0, {1}).status == MS_ENOTSUP);

    // Double-precision magnitudes single precision cannot carry are declined;
    // what lies beyond the truncation length is not read.
    for (const double factor : {1e30, 1e300, 1e-32, 1e-300}) {
        Tensor scaled = small;
        for (Complex& value : scaled.data) value *= factor;
        CHECK(call(scaled, MS_COMPLEX128, MS_FFT_FORWARD, 0, {1}).status == MS_ENOTSUP);
        CHECK(last_error().find("single precision") != std::string::npos);
        CHECK(call(scaled, MS_FLOAT64, MS_FFT_R2C, 0, {0, 1}).status == MS_ENOTSUP);
    }
    for (const double factor : {1e-28, 1e-9, 1e9, 1e28}) {
        Tensor scaled = small;
        for (Complex& value : scaled.data) value *= factor;
        const Output output = call(scaled, MS_COMPLEX128, MS_FFT_FORWARD, 0, {1});
        CHECK(output.status == MS_OK && relative_error(output.tensor, reference(scaled, MS_FFT_FORWARD, 0, {1}, 0)) < tolerance);
    }
    {
        Tensor special = small;
        special.data[70] = {1.0, std::numeric_limits<double>::infinity()};
        CHECK(call(special, MS_COMPLEX128, MS_FFT_FORWARD, 0, {1}).status == MS_ENOTSUP);
        special.data[70] = {std::numeric_limits<double>::quiet_NaN(), 0.0};
        CHECK(call(special, MS_COMPLEX128, MS_FFT_INVERSE, 0, {1}).status == MS_ENOTSUP);
        Tensor tail = small;
        for (uint64_t row = 0; row < 4; ++row) tail.data[row * 64 + 40] = 1e300;
        CHECK(call(tail, MS_COMPLEX128, MS_FFT_FORWARD, 0, {1}, 32).status == MS_OK);
        CHECK(call(tail, MS_COMPLEX128, MS_FFT_FORWARD, 0, {1}, 48).status == MS_ENOTSUP);
        Tensor zeros = small;
        for (Complex& value : zeros.data) value = 0.0;
        CHECK(call(zeros, MS_COMPLEX128, MS_FFT_FORWARD, 0, {1}).status == MS_OK);
    }

    // Malformed requests.
    CHECK(call(small, MS_COMPLEX64, 9, 0, {1}).status == MS_EINVAL);
    CHECK(call(small, MS_COMPLEX64, MS_FFT_FORWARD, 3, {1}).status == MS_EINVAL);
    CHECK(call(small, MS_COMPLEX64, MS_FFT_FORWARD, 0, {}).status == MS_EINVAL);
    CHECK(call(small, MS_COMPLEX64, MS_FFT_FORWARD, 0, {2}).status == MS_EINVAL);
    CHECK(call(small, MS_COMPLEX64, MS_FFT_FORWARD, 0, {1, 1}).status == MS_EINVAL);
    CHECK(call(small, MS_COMPLEX64, MS_FFT_FORWARD, 0, {0, 1, 0}).status == MS_EINVAL);
    CHECK(call(small, MS_COMPLEX64, MS_FFT_R2C, 0, {1}).status == MS_EINVAL);        // complex input
    CHECK(call(real, MS_FLOAT32, MS_FFT_R2C, 0, {0}).status == MS_EINVAL);           // the real axis must be the last
    CHECK(call(small, MS_COMPLEX64, MS_FFT_C2R, 0, {0}).status == MS_EINVAL);
    CHECK(call(small, MS_COMPLEX64, MS_FFT_FORWARD, 0, {0}, 32).status == MS_EINVAL); // n without the last axis
    {
        ms_fft_options options = fft_options(MS_FFT_FORWARD, 0, {1}, 0);
        const std::vector<uint8_t> bytes = pack(small, MS_COMPLEX64);
        ms_array array{};
        array.struct_size = sizeof(array);
        array.abi_version = MS_ARRAY_ABI_VERSION;
        array.data = bytes.data();
        array.dtype = MS_COMPLEX64;
        array.ndim = 2;
        array.shape[0] = 4;
        array.shape[1] = 64;
        const ms_route_options route = route_options();
        ms_array_result result{};
        CHECK(ms_fft_transform(context, &array, &options, &route, nullptr, nullptr) == MS_EINVAL);
        CHECK(ms_fft_transform(context, &array, nullptr, &route, &result, nullptr) == MS_EINVAL);
        CHECK(ms_fft_transform(context, &array, &options, nullptr, &result, nullptr) == MS_EINVAL);
        options.abi_version = 2;
        CHECK(ms_fft_transform(context, &array, &options, &route, &result, nullptr) == MS_EINVAL);
        options.abi_version = MS_FFT_ABI_VERSION;
        options.reserved[1] = 1;
        CHECK(ms_fft_transform(context, &array, &options, &route, &result, nullptr) == MS_EINVAL);
        CHECK(result.buffer == nullptr);
    }
}

void test_plan_info() {
    const auto describe = [](uint32_t dtype, const std::vector<uint64_t>& shape, uint32_t kind,
                             const std::vector<uint32_t>& axes, ms_route_options route, uint32_t prepare = 0) {
        const ms_fft_options options = fft_options(kind, 0, axes, 0);
        ms_fft_plan_info info{};
        info.struct_size = sizeof(info);
        info.abi_version = MS_FFT_ABI_VERSION;
        CHECK(ms_fft_plan(context, dtype, static_cast<uint32_t>(shape.size()), shape.data(), &options, &route, prepare, &info) == MS_OK);
        return info;
    };
    ms_fft_plan_info info = describe(MS_COMPLEX64, {8, 1000}, MS_FFT_FORWARD, {1}, route_options());
    CHECK(info.status == MS_OK && info.engine == MS_EXECUTION_GPU && info.passes == 1);
    CHECK(std::string(info.description) == "axis 1: stockham 1000");
    CHECK(info.device_bytes == 2 * 8 * 1000 * 8);  // input copy and result
    info = describe(MS_COMPLEX128, {8, 1000}, MS_FFT_FORWARD, {1}, route_options());
    CHECK(info.device_bytes == 2 * 8 * 1000 * 8 + 8 * 1000 * 16);
    info = describe(MS_FLOAT32, {64, 96}, MS_FFT_R2C, {0, 1}, route_options(), 1);
    CHECK(info.status == MS_OK && info.passes == 2 && std::string(info.description) == "axis 1: stockham 96; axis 0: strided 64 tail 49");
    info = describe(MS_COMPLEX64, {64, 49}, MS_FFT_C2R, {0, 1}, route_options());
    CHECK(info.status == MS_OK && std::string(info.description) == "axis 0: strided 64 tail 49; axis 1: stockham 96");
    info = describe(MS_COMPLEX64, {2, 3001}, MS_FFT_FORWARD, {1}, route_options());
    CHECK(info.status == MS_ENOTSUP && info.passes == 0 && std::string(info.description).find("3001") != std::string::npos);
    info = describe(MS_COMPLEX64, {2, 64}, MS_FFT_FORWARD, {1}, route_options(MS_EXECUTION_AUTO));
    CHECK(info.status == MS_ENOTSUP && std::string(info.description).find("crossover") != std::string::npos);
    info = describe(MS_COMPLEX128, {2, 64}, MS_FFT_FORWARD, {1}, route_options(MS_EXECUTION_GPU, MS_PRECISION_STRICT));
    CHECK(info.status == MS_ENOTSUP);
    // 2^31 points cannot be indexed by the kernels.
    info = describe(MS_COMPLEX64, {uint64_t{1} << 19, 4096}, MS_FFT_FORWARD, {1}, route_options());
    CHECK(info.status == MS_ENOTSUP && std::string(info.description).find("32-bit") != std::string::npos);
    // Malformed requests are errors, not declines.
    const ms_fft_options options = fft_options(MS_FFT_FORWARD, 0, {5}, 0);
    const ms_route_options route = route_options();
    const uint64_t shape[2] = {4, 4};
    ms_fft_plan_info raw{};
    raw.struct_size = sizeof(raw);
    raw.abi_version = MS_FFT_ABI_VERSION;
    CHECK(ms_fft_plan(context, MS_COMPLEX64, 2, shape, &options, &route, 0, &raw) == MS_EINVAL);
    raw.struct_size = 4;
    CHECK(ms_fft_plan(context, MS_COMPLEX64, 2, shape, &options, &route, 0, &raw) == MS_EINVAL);
}

// Every factorisation the planner may choose computes the same transform:
// run explicit pass lists through the engine.
void test_factorisations() {
    const fft::Limits limits = fft::device_limits();
    struct Case { uint64_t n, tail; bool inverse; std::vector<std::vector<int32_t>> factors; };
    const std::vector<Case> cases = {
        {65536, 1, false, {{256, 256}, {1024, 64}, {64, 1024}, {16, 16, 16, 16}, {4, 4096, 4}, {32, 64, 32}, {16, 4096}, {2048, 32}}},
        {65536, 1, true, {{256, 256}, {16, 16, 16, 16}, {8, 8192 / 8, 8}}},
        {3000, 1, false, {{30, 100}, {100, 30}, {10, 10, 30}, {3000}, {2, 2, 2, 3, 5, 5, 5}, {5, 600}}},
        {3000, 1, true, {{30, 100}, {3000}, {2, 2, 2, 3, 5, 5, 5}}},
        {2 * 3 * 5 * 7 * 11 * 13, 1, false, {{2, 3, 5, 7, 11, 13}, {13, 11, 7, 5, 3, 2}, {26, 33, 35}, {130, 231}, {65, 462}}},
        {96, 37, false, {{96}, {8, 12}, {12, 8}, {2, 2, 2, 2, 2, 3}, {3, 32}}},
        {96, 37, true, {{96}, {4, 4, 6}}},
        {1024, 130, false, {{1024}, {32, 32}, {4, 4, 4, 4, 4}, {16, 64}}},
        {143, 2, false, {{143}, {11, 13}, {13, 11}}},
    };
    int ran = 0;
    for (const Case& each : cases) {
        const uint64_t count = 3;
        fft::Request request;
        request.n = each.n;
        request.tail = each.tail;
        request.inverse = each.inverse;
        request.count = count;
        const Tensor x = random_tensor({count, each.n, each.tail}, true);
        Tensor expected = along(x, 1, each.n, [&](const std::vector<Complex>& line) { return dft(line, each.inverse); });
        for (const std::vector<int32_t>& factors : each.factors) {
            const fft::Planned planned = fft::build_multipass(request, factors, limits);
            if (!planned.plan) {
                CHECK(validated);  // only the reduced limits rule a pass list out
                continue;
            }
            const std::vector<uint8_t> bytes = pack(x, MS_COMPLEX64);
            BufferPtr input = array::allocate(context, bytes.size());
            std::memcpy(mutable_data(input.get()), bytes.data(), bytes.size());
            BufferPtr result = fft::run(context, {{planned.plan, count}}, std::move(input), 1.0f);
            const ReadView view = read(result.get(), context);
            CHECK(view.size == bytes.size());
            Tensor got;
            got.shape = x.shape;
            got.data.resize(x.data.size());
            const auto* values = reinterpret_cast<const float*>(view.data);
            const double scale = planned.plan->divides_by_n ? static_cast<double>(each.n) : 1.0;
            for (size_t i = 0; i < got.data.size(); ++i) got.data[i] = Complex(values[2 * i], values[2 * i + 1]) * scale;
            const double error = relative_error(got, expected);
            if (!(error < tolerance)) {
                std::fprintf(stderr, "%s: relative error %.3e\n", planned.plan->name.c_str(), error);
                std::abort();
            }
            ++ran;
        }
    }
    CHECK(ran >= 30);
    // The device-side factor: on a plan of ours (in its last pass) and on one of MLX's (a separate pass).
    for (const uint64_t n : {uint64_t{6000}, uint64_t{1000}}) {
        fft::Request request;
        request.n = n;
        request.count = 2;
        const fft::Planned planned = fft::plan(request, limits);
        CHECK(planned.plan && planned.plan->takes_scale == (n == 6000));
        const Tensor x = random_tensor({2, n}, true);
        const std::vector<uint8_t> bytes = pack(x, MS_COMPLEX64);
        BufferPtr input = array::allocate(context, bytes.size());
        std::memcpy(mutable_data(input.get()), bytes.data(), bytes.size());
        BufferPtr result = fft::run(context, {{planned.plan, 2}}, std::move(input), 0.375f);
        const auto* values = reinterpret_cast<const float*>(read(result.get(), context).data);
        Tensor got = x, expected = along(x, 1, n, [&](const std::vector<Complex>& line) { return dft(line, false); });
        for (size_t i = 0; i < got.data.size(); ++i) got.data[i] = Complex(values[2 * i], values[2 * i + 1]);
        for (Complex& value : expected.data) value *= 0.375;
        CHECK(relative_error(got, expected) < tolerance);
    }
}

// A context too small for the buffers declines; nothing leaks or stays charged.
void test_memory_budget() {
    ms_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = MS_RUNTIME_ABI_VERSION;
    options.storage_kind = MS_BUFFER_METAL;
    options.memory_limit_bytes = uint64_t{8} << 20;
    ms_context* small = nullptr;
    CHECK(ms_context_create(&options, &small) == MS_OK);
    ms_context* saved = context;
    context = small;
    CHECK(call(random_tensor({16, 4096}, true), MS_COMPLEX64, MS_FFT_FORWARD, 0, {1}).status == MS_OK);       // 2 x 512 KiB
    CHECK(call(random_tensor({256, 4096}, true), MS_COMPLEX64, MS_FFT_FORWARD, 0, {1}).status == MS_ENOTSUP); // 2 x 8 MiB
    CHECK(last_error().find("memory budget") != std::string::npos);
    ms_memory_info memory{};
    memory.struct_size = sizeof(memory);
    memory.abi_version = MS_RUNTIME_ABI_VERSION;
    CHECK(ms_context_memory_info(small, &memory) == MS_OK && memory.owned_bytes == 0 && memory.allocation_count == 0);
    context = saved;
    ms_context_release(small);
}
} // namespace

int main() {
    const char* validation = std::getenv("MTL_SHADER_VALIDATION");
    validated = validation && *validation && std::strcmp(validation, "0") != 0;
    ms_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = MS_RUNTIME_ABI_VERSION;
    options.storage_kind = MS_BUFFER_METAL;
    options.flags = MS_CONTEXT_POISON_UNWRITTEN;  // output a pass failed to write would not match the reference
    options.memory_limit_bytes = uint64_t{8} << 30;
    CHECK(ms_context_create(&options, &context) == MS_OK);
    CHECK(fft::device_limits().group_elements == (validated ? 2048 : 4096));

    test_plan_classes();
    test_large_powers_of_two();
    test_batches();
    test_padding_and_norms();
    test_axes();
    test_round_trips();
    test_precision_paths();
    test_routing_and_errors();
    test_plan_info();
    test_factorisations();
    test_memory_budget();

    ms_memory_info memory{};
    memory.struct_size = sizeof(memory);
    memory.abi_version = MS_RUNTIME_ABI_VERSION;
    CHECK(ms_context_memory_info(context, &memory) == MS_OK);
    CHECK(memory.owned_bytes == 0 && memory.allocation_count == 0);  // every buffer went back to the cache
    ms_context_release(context);
    std::printf("fft gpu ok: %d transforms against the reference, worst relative error %.2e%s\n", transforms_checked,
                worst_error, validated ? " (validated: reduced limits, small sizes)" : "");
    return 0;
}
