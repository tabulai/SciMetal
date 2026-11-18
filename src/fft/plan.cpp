// FFT planner. See plan.hpp for what it covers.
//
// The single-kernel plans, the two-pass four-step plan and the Rader and
// Bluestein constants are a port of mlx/backend/metal/fft.cpp from MLX 0.31.1:
//
//   Copyright © 2024 Apple Inc.
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all
//   copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//   SOFTWARE.
//
// (third_party/mlx-kernels/LICENSE.mlx). The multi-pass plans, the strided
// passes and everything about batches, slots and host adaptation are ours.
#include "plan.hpp"
#include "fft_kernels.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <complex>
#include <list>
#include <map>
#include <mutex>
#include <numbers>
#include <set>
#include <tuple>

namespace ms::detail::fft {
namespace {
// --- MLX's limits (fft.cpp) -------------------------------------------------
constexpr int32_t max_stockham_size = 4096;  // one transform must fit a threadgroup's memory
constexpr int32_t max_rader_size = 2048;
constexpr int32_t max_bluestein_size = 2048; // the padded transform is the next power of two >= 2n - 1
constexpr int32_t min_group_elements = MS_FFT_MIN_GROUP_ELEMENTS;
// Radices in MLX's order of preference; function constants 4..12 and 13..21
// count the steps of each.
constexpr std::array<int32_t, 9> radices = {13, 11, 8, 7, 6, 5, 4, 3, 2};
// A thread holds at most this many points across the butterflies of one step
// (MAX_OUTPUT_SIZE in the kernels).
constexpr int32_t max_thread_points = 18;

using Steps = std::array<int32_t, 9>;

bool is_radix(int32_t value) { return std::find(radices.begin(), radices.end(), value) != radices.end(); }
bool power_of_two(uint64_t value) { return std::has_single_bit(value); }

// Port of plan_stockham_fft: radix steps for a size whose factors are all radices.
bool plan_stockham(int32_t n, Steps& steps) {
    steps = {};
    const int32_t original = n;
    if (n == 1) return true;
    for (size_t i = 0; i < radices.size(); ++i) {
        const int32_t radix = radices[i];
        // MLX's tuning: small powers of two use radix 4 and 2 only.
        if (power_of_two(static_cast<uint64_t>(original)) && original < 512 && radix > 4) continue;
        while (n % radix == 0) {
            steps[i] += 1;
            n /= radix;
            if (n == 1) return true;
        }
    }
    return false;
}

// Port of FFTPlan / plan_fft for the sizes one kernel handles.
struct Decomposition {
    int32_t n = 0;
    Steps stockham = {};
    Steps rader = {};
    int32_t rader_n = 1;      // the Rader prime, 1 when there is none
    int32_t bluestein_n = 0;  // the padded size, 0 unless Bluestein
};

bool decompose(int32_t n, Decomposition& out) {
    out = Decomposition{};
    out.n = n;
    const auto bluestein = [&] {
        if (n > max_bluestein_size) return false;  // MLX's multi-upload Bluestein is not ported
        out.rader = {};
        out.rader_n = 1;
        out.bluestein_n = static_cast<int32_t>(std::bit_ceil(static_cast<uint32_t>(2 * n - 1)));
        return plan_stockham(out.bluestein_n, out.stockham);
    };
    int32_t remaining = n;
    for (const int32_t factor : mlx::prime_factors(n)) {
        if (is_radix(factor)) continue;
        // One Rader factor at most, and none for large n.
        if (out.rader_n > 1 || n > max_rader_size) return bluestein();
        // Rader's algorithm is not nested: p - 1 must itself be Stockham-plannable.
        for (const int32_t inner : mlx::prime_factors(factor - 1)) {
            if (!is_radix(inner)) return bluestein();
        }
        if (!plan_stockham(factor - 1, out.rader)) return bluestein();
        out.rader_n = factor;
        remaining /= factor;
    }
    return plan_stockham(remaining, out.stockham);
}

// Port of compute_elems_per_thread: MLX's heuristics for how many points of a
// transform each thread owns.
int32_t elems_per_thread(const Decomposition& plan) {
    std::set<int32_t> used;
    for (size_t i = 0; i < radices.size(); ++i) {
        if (plan.stockham[i] > 0 || plan.rader[i] > 0) used.insert(radices[i]);
    }
    const auto has = [&](int32_t radix) { return used.count(radix) != 0; };
    if (has(7) && (has(11) || has(13))) return 7;
    if (has(11) && has(13)) return 11;
    // Sizes MLX found to misbehave with the general rule.
    if (plan.n == 3159) return 13;
    if (plan.n == 3645) return 5;
    if (plan.n == 3969) return 7;
    if (plan.n == 1982) return 5;
    if (used.size() == 1) return *used.begin();
    if (used.size() == 2 && (has(11) || has(13))) return (*used.begin() + *used.rbegin()) / 2;
    return *std::next(used.begin());  // the second smallest radix
}

// The strided passes load whole threads' worth of points, so there the count
// must divide the transform. MLX's choice does for every size MLX's own
// four-step uses; otherwise take the largest radix whose butterflies still fit
// a thread's local arrays.
int32_t strided_elems_per_thread(int32_t n, const Steps& steps) {
    Decomposition plan;
    plan.n = n;
    plan.stockham = steps;
    const int32_t preferred = elems_per_thread(plan);
    if (n % preferred == 0) return preferred;
    for (size_t i = 0; i < radices.size(); ++i) {
        if (steps[i] == 0) continue;
        const int32_t candidate = radices[i];
        bool fits = true;
        for (size_t j = 0; j < radices.size(); ++j) {
            if (steps[j] == 0) continue;
            const int32_t radix = radices[j];
            if ((candidate + radix - 1) / radix * radix > max_thread_points) fits = false;
        }
        if (fits) return candidate;
    }
    return 0;
}

int32_t mod_exp(int32_t x, int32_t y, int32_t n) {
    int32_t out = 1;
    while (y) {
        if (y & 1) out = out * x % n;
        y >>= 1;
        x = x * x % n;
    }
    return out;
}

int32_t primitive_root(int32_t n) {
    const auto factors = mlx::prime_factors(n - 1);
    for (int32_t r = 2; r < n - 1; ++r) {
        bool found = true;
        for (const int32_t factor : factors) {
            if (mod_exp(r, (n - 1) / factor, n) == 1) {
                found = false;
                break;
            }
        }
        if (found) return r;
    }
    return -1;
}

// --- Constant tables --------------------------------------------------------
// MLX computes these in single precision; they are computed in double here
// and rounded once, so they are at least as accurate.
using Table = std::vector<uint8_t>;
using TablePtr = std::shared_ptr<const Table>;
using Complex = std::complex<double>;

Table complex64_table(const std::vector<Complex>& values) {
    Table table(values.size() * 2 * sizeof(float));
    auto* out = reinterpret_cast<float*>(table.data());
    for (size_t i = 0; i < values.size(); ++i) {
        out[2 * i] = static_cast<float>(values[i].real());
        out[2 * i + 1] = static_cast<float>(values[i].imag());
    }
    return table;
}

// exp(-2 pi i k / n) for an exact rational angle.
Complex unit(int64_t k, int64_t n) {
    k %= n;
    if (k < 0) k += n;
    const double angle = -2.0 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(n);
    return {std::cos(angle), std::sin(angle)};
}

// Direct DFT, for the Rader table (at most 2046 points, built once per size).
std::vector<Complex> dft(const std::vector<Complex>& x) {
    const int64_t n = static_cast<int64_t>(x.size());
    std::vector<Complex> roots(x.size()), y(x.size());
    for (int64_t k = 0; k < n; ++k) roots[static_cast<size_t>(k)] = unit(k, n);
    for (int64_t k = 0; k < n; ++k) {
        Complex sum = 0;
        int64_t index = 0;
        for (int64_t j = 0; j < n; ++j) {
            sum += x[static_cast<size_t>(j)] * roots[static_cast<size_t>(index)];
            index += k;
            if (index >= n) index -= n;
        }
        y[static_cast<size_t>(k)] = sum;
    }
    return y;
}

// In-place radix-2 FFT, for the Bluestein table (a power of two up to 4096).
void fft_radix2(std::vector<Complex>& x) {
    const size_t n = x.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
    for (size_t length = 2; length <= n; length <<= 1) {
        for (size_t start = 0; start < n; start += length) {
            for (size_t k = 0; k < length / 2; ++k) {
                const Complex w = unit(static_cast<int64_t>(k), static_cast<int64_t>(length));
                const Complex a = x[start + k], b = x[start + k + length / 2] * w;
                x[start + k] = a + b;
                x[start + k + length / 2] = a - b;
            }
        }
    }
}

struct RaderTables {
    TablePtr b_q, g_q, g_minus_q;
};
struct BluesteinTables {
    TablePtr w_q, w_k;
};

// Port of compute_raders_constants: the permutations generated by a primitive
// root of the prime and the transform of the permuted roots of unity.
RaderTables build_rader_tables(int32_t rader_n) {
    const int32_t root = primitive_root(rader_n);
    const int32_t inverse = mod_exp(root, rader_n - 2, rader_n);  // Fermat's little theorem
    const size_t count = static_cast<size_t>(rader_n - 1);
    std::vector<int16_t> g_q(count), g_minus_q(count);
    std::vector<Complex> b_q(count);
    for (size_t i = 0; i < count; ++i) {
        g_q[i] = static_cast<int16_t>(mod_exp(root, static_cast<int32_t>(i), rader_n));
        g_minus_q[i] = static_cast<int16_t>(mod_exp(inverse, static_cast<int32_t>(i), rader_n));
        b_q[i] = unit(g_minus_q[i], rader_n);
    }
    const auto shorts = [](const std::vector<int16_t>& values) {
        Table table(values.size() * sizeof(int16_t));
        std::copy_n(reinterpret_cast<const uint8_t*>(values.data()), table.size(), table.data());
        return std::make_shared<const Table>(std::move(table));
    };
    return {std::make_shared<const Table>(complex64_table(dft(b_q))), shorts(g_q), shorts(g_minus_q)};
}

// Port of compute_bluestein_constants: the chirp w_k[i] = exp(-i pi i^2 / n)
// and the transform of its reciprocal laid out over (-n, n).
BluesteinTables build_bluestein_tables(int32_t n, int32_t bluestein_n) {
    std::vector<Complex> w_k(static_cast<size_t>(n)), w_q(static_cast<size_t>(bluestein_n), Complex(0));
    for (int64_t i = -n + 1; i < n; ++i) {
        // exp(-i pi i^2 / n) = exp(-2 pi i (i^2 mod 2n) / 2n): keeps the angle exact.
        const Complex chirp = unit(i * i, 2 * static_cast<int64_t>(n));
        w_q[static_cast<size_t>(i + n - 1)] = std::conj(chirp);
        if (i >= 0) w_k[static_cast<size_t>(i)] = chirp;
    }
    fft_radix2(w_q);
    return {std::make_shared<const Table>(complex64_table(w_q)), std::make_shared<const Table>(complex64_table(w_k))};
}

// Small most-recently-used cache: tables and plans are a few KiB each and a
// process works with a handful of sizes.
template<class Key, class Value> class Cache {
public:
    explicit Cache(size_t capacity) : capacity_(capacity) {}
    bool find(const Key& key, Value& out) {
        std::lock_guard lock(mutex_);
        const auto found = index_.find(key);
        if (found == index_.end()) return false;
        order_.splice(order_.begin(), order_, found->second);
        out = found->second->second;
        return true;
    }
    void insert(const Key& key, Value value) {
        std::lock_guard lock(mutex_);
        const auto found = index_.find(key);
        if (found != index_.end()) {
            found->second->second = std::move(value);
            order_.splice(order_.begin(), order_, found->second);
            return;
        }
        order_.emplace_front(key, std::move(value));
        index_[key] = order_.begin();
        if (order_.size() > capacity_) {
            index_.erase(order_.back().first);
            order_.pop_back();
        }
    }

private:
    size_t capacity_;
    std::mutex mutex_;
    std::list<std::pair<Key, Value>> order_;
    std::map<Key, typename std::list<std::pair<Key, Value>>::iterator> index_;
};

RaderTables rader_tables(int32_t rader_n) {
    static Cache<int32_t, RaderTables> cache(64);
    RaderTables tables;
    if (cache.find(rader_n, tables)) return tables;
    tables = build_rader_tables(rader_n);
    cache.insert(rader_n, tables);
    return tables;
}

BluesteinTables bluestein_tables(int32_t n, int32_t bluestein_n) {
    static Cache<int32_t, BluesteinTables> cache(64);
    BluesteinTables tables;
    if (cache.find(n, tables)) return tables;
    tables = build_bluestein_tables(n, bluestein_n);
    cache.insert(n, tables);
    return tables;
}

// --- Plan construction ------------------------------------------------------
Planned unsupported(std::string reason) { return {nullptr, std::move(reason)}; }

bool is_inverse(const Request& request) {
    return request.transform == Transform::c2r || (request.transform == Transform::c2c && request.inverse);
}

Constants constants_for(const Decomposition& plan, int32_t fft_size, int32_t per_thread, bool inverse) {
    Constants constants;
    constants.inverse = inverse;
    constants.power_of_two = power_of_two(static_cast<uint64_t>(fft_size));
    constants.elems_per_thread = per_thread;
    constants.rader_m = plan.n / plan.rader_n;
    std::copy(plan.stockham.begin(), plan.stockham.end(), constants.stockham);
    std::copy(plan.rader.begin(), plan.rader.end(), constants.rader);
    return constants;
}

uint64_t half_spectrum(uint64_t n) { return n / 2 + 1; }

// Transforms per threadgroup for a pass over points `stride` apart: as many
// as `wanted`, within the threadgroup's memory and thread count.
int32_t pass_width(int32_t wanted, int32_t fft_size, int32_t threads, uint64_t stride, const Limits& limits) {
    const int64_t width = std::min<int64_t>({wanted, limits.group_elements / fft_size, limits.group_threads / threads,
                                             static_cast<int64_t>(std::min<uint64_t>(stride, INT32_MAX))});
    return static_cast<int32_t>(std::max<int64_t>(width, 1));
}

int32_t group_elements(int32_t width, int32_t fft_size) {
    return std::max(min_group_elements,
                    static_cast<int32_t>(std::bit_ceil(static_cast<uint32_t>(width) * static_cast<uint32_t>(fft_size))));
}

// One whole transform per threadgroup slice: port of fft_op without four-step.
Planned plan_single_kernel(const Request& request, const Limits& limits) {
    const int32_t n = static_cast<int32_t>(request.n);
    Decomposition decomposition;
    if (!decompose(n, decomposition))
        return unsupported("a length with a prime factor above 13 needs Bluestein's algorithm, which one kernel "
                           "covers up to 2048 points");
    const bool inverse = is_inverse(request);
    Step step;
    int32_t fft_size = 0;
    for (;;) {
        fft_size = decomposition.bluestein_n ? decomposition.bluestein_n : n;
        if (fft_size > limits.group_elements) return unsupported("the transform does not fit a threadgroup's memory");
        const int32_t per_thread = elems_per_thread(decomposition);
        step.constants = constants_for(decomposition, fft_size, per_thread, inverse);
        step.threads = (fft_size + per_thread - 1) / per_thread;
        if (step.threads <= limits.group_threads) break;
        // Bluestein's padded power of two runs on fewer threads than a Rader
        // transform with a small radix does.
        if (decomposition.rader_n > 1 && n <= max_bluestein_size) {
            decomposition = Decomposition{};
            decomposition.n = n;
            decomposition.bluestein_n = static_cast<int32_t>(std::bit_ceil(static_cast<uint32_t>(2 * n - 1)));
            plan_stockham(decomposition.bluestein_n, decomposition.stockham);
            continue;
        }
        return unsupported("the transform needs more threads than a threadgroup has");
    }
    // Short transforms share a threadgroup.
    step.width = std::max(std::min(min_group_elements / fft_size, limits.group_threads / step.threads), 1);
    step.paired = request.transform != Transform::c2c;
    const char* in_type = request.transform == Transform::r2c ? "float" : "float2";
    const char* out_type = request.transform == Transform::c2r ? "float" : "float2";
    step.group_elements = group_elements(step.width, fft_size);
    const std::string suffix = "_mem_" + std::to_string(step.group_elements) + "_" + in_type + "_" + out_type;

    auto plan = std::make_shared<Plan>();
    if (decomposition.bluestein_n) {
        step.kind = Kernel::bluestein;
        step.kernel = "bluestein_fft" + suffix;
        const BluesteinTables tables = bluestein_tables(n, decomposition.bluestein_n);
        plan->tables = {tables.w_q, tables.w_k};
        step.tables = {0, 1};
        step.values = {n, decomposition.bluestein_n, 0};
        step.batch_value = 2;
        plan->clear_dc_imag = request.transform == Transform::c2r;
        plan->name = "bluestein " + std::to_string(n) + " in " + std::to_string(decomposition.bluestein_n);
    } else if (decomposition.rader_n > 1) {
        step.kind = Kernel::rader;
        step.kernel = "rader_fft" + suffix;
        const RaderTables tables = rader_tables(decomposition.rader_n);
        plan->tables = {tables.b_q, tables.g_q, tables.g_minus_q};
        step.tables = {0, 1, 2};
        step.values = {n, 0, decomposition.rader_n};
        step.batch_value = 1;
        plan->name = "rader " + std::to_string(n) + " (prime " + std::to_string(decomposition.rader_n) + ")";
    } else {
        step.kind = Kernel::stockham;
        step.kernel = "fft" + suffix;
        step.values = {n, 0};
        step.batch_value = 1;
        plan->name = "stockham " + std::to_string(n);
    }
    plan->request = request;
    plan->steps.push_back(std::move(step));
    const uint64_t real_row = request.n, half_row = 2 * half_spectrum(request.n), full_row = 2 * request.n;
    switch (request.transform) {
    case Transform::c2c:
        plan->slots = {full_row, full_row};
        plan->input_row = plan->result_row = request.n;
        break;
    case Transform::r2c:
        plan->slots = {real_row, half_row};
        plan->input_real = true;
        plan->input_row = request.n;
        plan->result_row = half_spectrum(request.n);
        break;
    case Transform::c2r:
        plan->slots = {half_row, real_row};
        plan->result_real = true;
        plan->input_row = half_spectrum(request.n);
        plan->result_row = request.n;
        break;
    }
    plan->result_slot = 1;
    plan->divides_by_n = inverse;
    return {std::move(plan), {}};
}

std::string factor_list(const std::vector<int32_t>& factors) {
    std::string text;
    for (const int32_t factor : factors) text += (text.empty() ? "" : "x") + std::to_string(factor);
    return text;
}

bool check_factors(const Request& request, const std::vector<int32_t>& factors, const Limits& limits,
                   std::string& reason) {
    uint64_t product = 1;
    for (const int32_t factor : factors) {
        if (factor < 2 || factor > limits.group_elements || !mlx::smooth(static_cast<uint64_t>(factor))) {
            reason = "every pass length must factor into 2, 3, 5, 7, 11 and 13 and fit a threadgroup's memory";
            return false;
        }
        product *= static_cast<uint64_t>(factor);
        if (product > request.n) break;
    }
    if (factors.empty() || product != request.n) {
        reason = "the pass lengths must multiply to the transform length";
        return false;
    }
    return true;
}

// A strided pass of ours over one factor of the length.
bool strided_step(int32_t n, uint64_t stride, uint64_t outer, uint64_t tail, int32_t wanted_width, const Limits& limits,
                  Step& step, std::string& reason) {
    Steps steps;
    Decomposition decomposition;
    decomposition.n = n;
    if (!plan_stockham(n, steps)) {
        reason = "pass length " + std::to_string(n) + " has no radix decomposition";
        return false;
    }
    decomposition.stockham = steps;
    const int32_t per_thread = strided_elems_per_thread(n, steps);
    if (per_thread == 0) {
        reason = "pass length " + std::to_string(n) + " has no thread layout that divides it";
        return false;
    }
    step = Step{};
    step.kind = Kernel::strided;
    step.constants = constants_for(decomposition, n, per_thread, false);
    step.threads = n / per_thread;
    if (step.threads > limits.group_threads) {
        reason = "pass length " + std::to_string(n) + " needs more threads than a threadgroup has";
        return false;
    }
    int32_t width = pass_width(wanted_width, n, step.threads, stride, limits);
    // A width that divides the stride leaves no partly filled threadgroups;
    // take the largest such width unless it costs more than half the coalescing.
    for (int32_t divisor = width; divisor >= 1; --divisor) {
        if (stride % static_cast<uint64_t>(divisor) == 0) {
            if (2 * divisor >= width) width = divisor;
            break;
        }
    }
    step.width = width;
    step.group_elements = group_elements(width, n);
    step.kernel = "ms_fft_strided_mem_" + std::to_string(step.group_elements);
    step.n = n;
    step.stride = static_cast<int32_t>(stride);
    step.outer = static_cast<int32_t>(outer);
    step.tail = static_cast<int32_t>(tail);
    step.transforms = outer * stride;
    return true;
}

// Width our multi-pass plans ask for: short passes coalesce 16 transforms per
// threadgroup (512 bytes of each stream), measured best in probes/e5_fft.
constexpr int32_t strided_width = 16;

// A complex transform as a chain of our strided passes, one per factor:
// any length that factors into the radices, along any axis. Real transforms
// run as their complex equivalent, with the host expanding or cropping rows.
Planned plan_chain(const Request& request, const std::vector<int32_t>& factors, const Limits& limits) {
    std::string reason;
    if (!check_factors(request, factors, limits, reason)) return unsupported(reason);
    if (request.transform != Transform::c2c && request.tail != 1)
        return unsupported("real transforms run along the last axis only");
    const bool inverse = is_inverse(request);
    auto plan = std::make_shared<Plan>();
    plan->request = request;
    uint64_t outer = 1, stride = request.n * request.tail;
    for (size_t i = 0; i < factors.size(); ++i) {
        stride /= static_cast<uint64_t>(factors[i]);
        Step step;
        if (!strided_step(factors[i], stride, outer, request.tail, strided_width, limits, step, reason))
            return unsupported(reason);
        if (inverse && i == 0) step.flags |= MS_FFT_CONJ_LOAD;
        if (inverse && i + 1 == factors.size()) step.flags |= MS_FFT_CONJ_STORE;
        step.source = static_cast<uint8_t>(i % 2);
        step.target = static_cast<uint8_t>(1 - i % 2);
        plan->steps.push_back(std::move(step));
        outer *= static_cast<uint64_t>(factors[i]);
    }
    const uint64_t full_row = 2 * request.n * request.tail;
    plan->slots = {full_row, full_row};
    plan->result_slot = static_cast<uint8_t>(factors.size() % 2);
    plan->input_row = plan->result_row = request.n;
    plan->takes_scale = true;
    if (request.transform == Transform::r2c) plan->store = Store::half;
    if (request.transform == Transform::c2r) {
        plan->load = Load::half_spectrum;
        plan->store = Store::real_part;
    }
    plan->name = "strided " + factor_list(factors) + (request.tail > 1 ? " tail " + std::to_string(request.tail) : "");
    return {std::move(plan), {}};
}

// A power-of-two transform along the last axis: MLX's four-step kernels for
// the first and last pass (they read real input, rebuild and halve Hermitian
// spectra, and apply the inverse's conjugation and 1/n), our strided pass in
// between. Two passes are exactly MLX's four-step.
Planned plan_power_of_two(const Request& request, const std::vector<int32_t>& factors, int32_t wanted_width,
                          const Limits& limits) {
    std::string reason;
    if (!check_factors(request, factors, limits, reason)) return unsupported(reason);
    if (factors.size() < 2 || request.tail != 1 || !power_of_two(request.n))
        return unsupported("the four-step passes need a power-of-two length along the last axis and two or more passes");
    for (const int32_t factor : factors) {
        if (factor < 4) return unsupported("four-step passes are at least four points long");
    }
    const bool inverse = is_inverse(request);
    const bool real = request.transform != Transform::c2c;
    const size_t count = factors.size();
    auto plan = std::make_shared<Plan>();
    plan->request = request;

    const auto four_step = [&](int32_t fft_size, bool first, uint8_t source, uint8_t target) {
        Decomposition decomposition;
        decomposition.n = fft_size;
        plan_stockham(fft_size, decomposition.stockham);
        const int32_t per_thread = elems_per_thread(decomposition);
        const uint64_t stride = request.n / static_cast<uint64_t>(fft_size);
        Step step;
        step.kind = Kernel::four_step;
        step.constants = constants_for(decomposition, fft_size, per_thread, inverse);
        step.threads = fft_size / per_thread;
        if (step.threads > limits.group_threads) return false;
        // MLX's strided access needs a width that divides the stride: a power of two.
        step.width = static_cast<int32_t>(std::bit_floor(
            static_cast<uint32_t>(pass_width(wanted_width, fft_size, step.threads, stride, limits))));
        const char* in_type = first && request.transform == Transform::r2c ? "float" : "float2";
        const char* out_type = !first && request.transform == Transform::c2r ? "float" : "float2";
        step.group_elements = group_elements(step.width, fft_size);
        step.kernel = "four_step_mem_" + std::to_string(step.group_elements) + "_" + in_type + "_" + out_type + "_" +
                      (first ? "0" : "1") + "_" + (real ? "true" : "false");
        // n1 and n2 as the kernel names them: the first pass transforms n1 points
        // n2 apart, the last pass n2 contiguous points written n1 apart.
        const int32_t other = static_cast<int32_t>(stride);
        step.values = first ? std::vector<int32_t>{fft_size, other, 0} : std::vector<int32_t>{other, fft_size, 0};
        step.batch_value = 2;
        step.transforms = stride;
        step.source = source;
        step.target = target;
        plan->steps.push_back(std::move(step));
        return true;
    };
    const char* too_many_threads = "a four-step pass needs more threads than a threadgroup has";

    // Slots. Complex: the input and one buffer of the same size alternate.
    // Real: the passes in between need full-size complex buffers (one, or two
    // to alternate), and the real side is too small to serve as one. An r2c
    // result goes back into the (slightly enlarged) input slot; a c2r result
    // gets its own slot so that it has exactly the result's size.
    const uint64_t real_row = request.n, half_row = 2 * half_spectrum(request.n), full_row = 2 * request.n;
    std::vector<uint8_t> target(count);
    if (!real) {
        for (size_t i = 0; i < count; ++i) target[i] = static_cast<uint8_t>(1 - i % 2);
        plan->slots = {full_row, full_row};
        plan->input_row = plan->result_row = request.n;
    } else {
        for (size_t i = 0; i + 1 < count; ++i) target[i] = static_cast<uint8_t>(1 + i % 2);
        const bool two_scratch = count > 2;
        if (request.transform == Transform::r2c) {
            plan->slots = {half_row, full_row};
            if (two_scratch) plan->slots.push_back(full_row);
            target[count - 1] = 0;
            plan->input_real = true;
            plan->input_row = request.n;
            plan->result_row = half_spectrum(request.n);
        } else {
            plan->slots = {half_row, full_row};
            if (two_scratch) plan->slots.push_back(full_row);
            target[count - 1] = static_cast<uint8_t>(plan->slots.size());
            plan->slots.push_back(real_row);
            plan->result_real = true;
            plan->input_row = half_spectrum(request.n);
            plan->result_row = request.n;
        }
    }
    plan->result_slot = target[count - 1];

    if (!four_step(factors[0], true, 0, target[0])) return unsupported(too_many_threads);
    uint64_t outer = static_cast<uint64_t>(factors[0]), stride = request.n / outer;
    for (size_t i = 1; i + 1 < count; ++i) {
        stride /= static_cast<uint64_t>(factors[i]);
        Step step;
        if (!strided_step(factors[i], stride, outer, 1, wanted_width, limits, step, reason)) return unsupported(reason);
        step.source = target[i - 1];
        step.target = target[i];
        plan->steps.push_back(std::move(step));
        outer *= static_cast<uint64_t>(factors[i]);
    }
    if (!four_step(factors[count - 1], false, target[count - 2], target[count - 1])) return unsupported(too_many_threads);
    plan->divides_by_n = inverse;
    plan->name = "four-step " + factor_list(factors);
    return {std::move(plan), {}};
}

// --- Choosing the passes ----------------------------------------------------
// Batches of at least this many points do not fit the GPU's caches; there a
// pass is only cheap if each threadgroup touches few streams, so short passes
// beat few long ones (probes/README.md, E5).
constexpr uint64_t large_points = uint64_t{1} << 22;
constexpr int32_t short_pass = 32;   // pass length for large batches
constexpr int32_t long_pass = 128;   // pass length when everything is in cache

bool is_large(const Request& request) {
    const long double points = static_cast<long double>(request.n) * static_cast<long double>(request.tail) *
                               static_cast<long double>(request.count);
    return points >= static_cast<long double>(large_points);
}

// Splits n (factoring into the radices) into as few pass lengths as possible
// that are at most `longest`, as equal as possible, shortest first: the first
// pass has the largest stride and so benefits most from few streams.
std::vector<int32_t> choose_factors(uint64_t n, int32_t longest, const Limits& limits) {
    std::vector<uint64_t> primes;
    for (uint64_t rest = n; rest > 1;) {
        for (const int32_t prime : {2, 3, 5, 7, 11, 13}) {
            if (rest % static_cast<uint64_t>(prime) == 0) {
                primes.push_back(static_cast<uint64_t>(prime));
                rest /= static_cast<uint64_t>(prime);
                break;
            }
        }
    }
    std::sort(primes.rbegin(), primes.rend());
    for (size_t passes = 1; passes <= primes.size(); ++passes) {
        std::vector<uint64_t> groups(passes, 1);
        for (const uint64_t prime : primes) *std::min_element(groups.begin(), groups.end()) *= prime;
        const uint64_t largest = *std::max_element(groups.begin(), groups.end());
        if (largest > static_cast<uint64_t>(longest) && passes < primes.size()) continue;
        std::sort(groups.begin(), groups.end());
        std::vector<int32_t> factors(groups.begin(), groups.end());
        bool usable = true;
        for (const int32_t factor : factors) {
            Steps steps;
            if (factor > limits.group_elements || !plan_stockham(factor, steps)) {
                usable = false;
                continue;
            }
            const int32_t per_thread = strided_elems_per_thread(factor, steps);
            if (per_thread == 0 || factor / per_thread > limits.group_threads) usable = false;
        }
        if (usable) return factors;
    }
    return {};
}

// Pass lengths for a power of two above one kernel's reach.
std::vector<int32_t> power_of_two_factors(uint64_t n, const Limits& limits, int32_t& wanted_width) {
    const int log2n = std::countr_zero(n);
    // The long passes below put four 1024-point (or two 2048-point) transforms
    // in a threadgroup: 4096 elements on 512 threads.
    if (limits.group_elements >= max_stockham_size && limits.group_threads >= 512 && log2n <= 22) {
        wanted_width = 4;  // MLX's MIN_COALESCE_WIDTH
        // MLX's split up to 2^20. Its four-step insists on four transforms per
        // threadgroup, so it stops there; two 2048-point transforms per group
        // carry the same two passes to 2^22.
        if (log2n <= 20) {
            const int32_t n2 = n > 65536 ? 1024 : 64;
            return {static_cast<int32_t>(n / static_cast<uint64_t>(n2)), n2};
        }
        return {log2n == 21 ? 1024 : 2048, 2048};
    }
    // Beyond that the array no longer fits the GPU's caches: 16- and 32-point
    // passes, as equal as possible after a 16-point first pass.
    wanted_width = strided_width;
    const int bits = log2n - 4;
    const int passes = (bits + 4) / 5;
    std::vector<int32_t> factors = {16};
    for (int i = 0; i < passes; ++i) factors.push_back(1 << (bits / passes + (i < bits % passes ? 1 : 0)));
    return factors;
}

struct CacheKey {
    uint8_t transform;
    bool inverse;
    bool large;
    uint64_t n;
    uint64_t tail;
    int32_t group_elements;
    int32_t group_threads;
    auto operator<=>(const CacheKey&) const = default;
};

std::atomic<uint64_t> built_plans{0};
} // namespace

std::string Constants::key() const {
    std::string text = std::to_string(inverse) + std::to_string(power_of_two) + "e" + std::to_string(elems_per_thread) +
                       "m" + std::to_string(rader_m) + "s";
    for (const int32_t steps : stockham) text += std::to_string(steps) + ".";
    text += "r";
    for (const int32_t steps : rader) text += std::to_string(steps) + ".";
    return text;
}

namespace mlx {
std::vector<int32_t> prime_factors(int32_t n) {
    std::vector<int32_t> factors;
    int32_t z = 2;
    while (z * z <= n) {
        if (n % z == 0) {
            factors.push_back(z);
            n /= z;
        } else {
            ++z;
        }
    }
    if (n > 1) factors.push_back(n);
    return factors;
}

bool smooth(uint64_t n) {
    if (n == 0) return false;
    for (const uint64_t prime : {2, 3, 5, 7, 11, 13}) {
        while (n % prime == 0) n /= prime;
    }
    return n == 1;
}
} // namespace mlx

Planned build_multipass(const Request& request, const std::vector<int32_t>& factors, const Limits& limits) {
    if (request.n == 0 || request.tail == 0 || request.n > INT32_MAX / request.tail)
        return unsupported("the transform exceeds the kernels' 32-bit indexing");
    if (request.tail == 1 && power_of_two(request.n) && factors.size() >= 2) {
        bool powers = true;
        for (const int32_t factor : factors) powers = powers && factor >= 4 && power_of_two(static_cast<uint64_t>(factor));
        if (powers) {
            const int32_t width = *std::max_element(factors.begin(), factors.end()) > 64 ? 4 : strided_width;
            return plan_power_of_two(request, factors, width, limits);
        }
    }
    return plan_chain(request, factors, limits);
}

Planned build_plan(const Request& request, const Limits& limits) {
    built_plans.fetch_add(1, std::memory_order_relaxed);
    if (request.n == 0 || request.tail == 0) return unsupported("empty transform");
    if (request.n == 1) return unsupported("a one-point transform is a copy");
    if (request.n > INT32_MAX / request.tail) return unsupported("the transform exceeds the kernels' 32-bit indexing");
    if (request.transform != Transform::c2c && request.tail != 1)
        return unsupported("real transforms run along the last axis only");
    const bool smooth = mlx::smooth(request.n);
    Planned single;
    if (request.tail == 1) {
        if (request.n <= static_cast<uint64_t>(max_stockham_size)) {
            single = plan_single_kernel(request, limits);
            // Without a single kernel (reduced limits), lengths that factor
            // into the radices still have the multi-pass plans below.
            if (single.plan || !smooth) return single;
        }
        if (power_of_two(request.n) && request.n >= 64) {
            int32_t width = 0;
            const std::vector<int32_t> factors = power_of_two_factors(request.n, limits, width);
            Planned planned = plan_power_of_two(request, factors, width, limits);
            if (planned.plan) return planned;
        }
    }
    if (!smooth)
        return unsupported(request.tail == 1
                               ? "a length above 4096 with a prime factor above 13 would need a multi-pass Bluestein transform"
                               : "a length with a prime factor above 13 along an axis that is not the last");
    const std::vector<int32_t> factors = choose_factors(request.n, is_large(request) ? short_pass : long_pass, limits);
    if (factors.empty()) return unsupported("no pass lengths fit a threadgroup");
    return plan_chain(request, factors, limits);
}

namespace {
CacheKey cache_key(const Request& request, const Limits& limits) {
    // Only multi-pass plans differ between small and large batches.
    const bool multipass = request.tail > 1 || request.n > static_cast<uint64_t>(std::min(max_stockham_size, limits.group_elements));
    return {static_cast<uint8_t>(request.transform), is_inverse(request), multipass && is_large(request),
            request.n, request.tail, limits.group_elements, limits.group_threads};
}

Cache<CacheKey, Planned>& plan_cache() {
    static Cache<CacheKey, Planned> cache(256);
    return cache;
}
} // namespace

Planned plan(const Request& request, const Limits& limits) {
    const CacheKey key = cache_key(request, limits);
    Planned planned;
    if (plan_cache().find(key, planned)) return planned;
    planned = build_plan(request, limits);
    plan_cache().insert(key, planned);
    return planned;
}

Planned replan(const Request& request, const Limits& limits, int32_t group_threads) {
    Limits tighter = limits;
    tighter.group_threads = std::min(limits.group_threads, group_threads);
    Planned planned = tighter.group_threads >= 1 ? build_plan(request, tighter)
                                                 : unsupported("the device grants the kernels no threadgroup");
    plan_cache().insert(cache_key(request, limits), planned);
    return planned;
}

uint64_t plans_built() { return built_plans.load(std::memory_order_relaxed); }

bool instantiate(const Plan& plan, uint64_t count, std::vector<Dispatch>& out, std::string& reason) {
    // Every kernel indexes its buffers with 32-bit integers.
    const uint64_t block = plan.request.n * plan.request.tail;
    if (count == 0 || count > INT32_MAX / block) {
        reason = "the batch exceeds the kernels' 32-bit indexing (2^31 points)";
        return false;
    }
    out.clear();
    out.reserve(plan.steps.size());
    for (const Step& step : plan.steps) {
        Dispatch dispatch;
        const uint64_t width = static_cast<uint64_t>(step.width);
        if (step.kind == Kernel::strided) {
            dispatch.blocks = static_cast<int32_t>(count * static_cast<uint64_t>(step.outer));
            const uint64_t stride = static_cast<uint64_t>(step.stride);
            dispatch.groups = count * static_cast<uint64_t>(step.outer) * ((stride + width - 1) / width);
        } else {
            const uint64_t batch = count * step.transforms;
            dispatch.values = step.values;
            dispatch.values[static_cast<size_t>(step.batch_value)] = static_cast<int32_t>(batch);
            dispatch.groups = (batch + width - 1) / width;
            if (step.paired) dispatch.groups = (dispatch.groups + 1) / 2;
        }
        out.push_back(std::move(dispatch));
    }
    return true;
}
} // namespace ms::detail::fft
