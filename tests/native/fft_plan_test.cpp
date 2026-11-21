// The FFT planner without a device: which lengths have a plan, what the plans
// look like, and the invariants every plan must keep for the kernels to be
// used correctly. Also the C ABI on a context without a GPU, where every
// transform declines.
#include "sm_fft.h"
#include "fft/plan.hpp"
#include "fft/fft_kernels.h"

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <set>
#include <string>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "fft plan test failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

using namespace sm::detail::fft;

namespace {
bool is_prime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d) {
        if (n % d == 0) return false;
    }
    return true;
}

Request request(Transform transform, uint64_t n, bool inverse = false, uint64_t tail = 1, uint64_t count = 1) {
    Request result;
    result.transform = transform;
    result.inverse = inverse;
    result.n = n;
    result.tail = tail;
    result.count = count;
    return result;
}

// What the kernels need from any plan, whatever it computes.
void check_invariants(const Plan& plan, const Limits& limits) {
    const Request& rq = plan.request;
    CHECK(!plan.steps.empty() && !plan.name.empty());
    CHECK(plan.result_slot < plan.slots.size());
    const std::set<int> sizes = {256, 512, 1024, 2048, 4096};
    uint64_t strided_product = 1;
    for (const Step& step : plan.steps) {
        CHECK(step.source < plan.slots.size() && step.target < plan.slots.size() && step.source != step.target);
        CHECK(step.width >= 1 && step.threads >= 1);
        CHECK(step.width * step.threads <= limits.group_threads);
        CHECK(sizes.count(step.group_elements) && step.group_elements <= limits.group_elements);
        CHECK(step.constants.elems_per_thread >= 2 && step.constants.elems_per_thread <= 13);
        // The kernel instance named must be the one with that much memory.
        CHECK(step.kernel.find("_mem_" + std::to_string(step.group_elements)) != std::string::npos);
        int32_t fft_size = 0;
        switch (step.kind) {
        case Kernel::stockham:
        case Kernel::rader:
            fft_size = step.values[0];
            CHECK(static_cast<uint64_t>(fft_size) == rq.n && step.batch_value == 1);
            break;
        case Kernel::bluestein:
            fft_size = step.values[1];
            CHECK(static_cast<uint64_t>(step.values[0]) == rq.n && fft_size >= 2 * step.values[0] - 1);
            CHECK(std::has_single_bit(static_cast<uint32_t>(fft_size)) && step.batch_value == 2);
            break;
        case Kernel::four_step: {
            const bool first = step.kernel.find("_0_") != std::string::npos;
            fft_size = first ? step.values[0] : step.values[1];
            const int32_t stride = first ? step.values[1] : step.values[0];
            CHECK(static_cast<uint64_t>(step.values[0]) * static_cast<uint64_t>(step.values[1]) == rq.n);
            // MLX's strided access needs whole threads' worth of points and a
            // width that divides the stride.
            CHECK(fft_size % step.constants.elems_per_thread == 0);
            CHECK(stride % step.width == 0);
            CHECK(step.transforms == static_cast<uint64_t>(stride) && step.batch_value == 2);
            CHECK(std::has_single_bit(rq.n) && rq.tail == 1);
            break;
        }
        case Kernel::strided:
            fft_size = step.n;
            CHECK(fft_size % step.constants.elems_per_thread == 0);
            CHECK(static_cast<uint64_t>(step.tail) == rq.tail && step.stride % step.tail == 0);
            CHECK(static_cast<uint64_t>(step.outer) * static_cast<uint64_t>(step.n) * static_cast<uint64_t>(step.stride) ==
                  rq.n * rq.tail);
            CHECK(static_cast<uint64_t>(step.width) <= static_cast<uint64_t>(step.stride));
            strided_product *= static_cast<uint64_t>(step.n);
            break;
        }
        CHECK(step.threads * step.constants.elems_per_thread >= fft_size);
        CHECK(step.width * fft_size <= step.group_elements);
        CHECK(step.constants.power_of_two == std::has_single_bit(static_cast<uint32_t>(fft_size)));
        // The radix steps multiply to the transform (less the Rader factor).
        uint64_t radix_product = 1;
        const int radices[9] = {13, 11, 8, 7, 6, 5, 4, 3, 2};
        for (int i = 0; i < 9; ++i) {
            for (int s = 0; s < step.constants.stockham[i]; ++s) radix_product *= static_cast<uint64_t>(radices[i]);
        }
        if (step.kind == Kernel::rader) {
            const int32_t prime = step.values[2];
            CHECK(is_prime(static_cast<uint64_t>(prime)) && prime > 13);
            CHECK(radix_product * static_cast<uint64_t>(prime) == rq.n);
            CHECK(step.constants.rader_m == static_cast<int32_t>(rq.n) / prime);
            uint64_t rader_product = 1;
            for (int i = 0; i < 9; ++i) {
                for (int s = 0; s < step.constants.rader[i]; ++s) rader_product *= static_cast<uint64_t>(radices[i]);
            }
            CHECK(rader_product == static_cast<uint64_t>(prime - 1));
            CHECK(plan.tables.size() == 3 && plan.tables[0]->size() == static_cast<size_t>(prime - 1) * 8 &&
                  plan.tables[1]->size() == static_cast<size_t>(prime - 1) * 2);
        } else {
            CHECK(radix_product == static_cast<uint64_t>(fft_size));
        }
        if (step.kind == Kernel::bluestein)
            CHECK(plan.tables.size() == 2 && plan.tables[0]->size() == static_cast<size_t>(fft_size) * 8 &&
                  plan.tables[1]->size() == rq.n * 8);
    }
    // Slot sizes cover what the host loads and reads.
    const uint64_t in_scalars = plan.input_row * rq.tail * (plan.input_real ? 1 : 2);
    const uint64_t out_scalars = plan.result_row * rq.tail * (plan.result_real ? 1 : 2);
    CHECK(plan.slots[0] >= in_scalars);
    CHECK(plan.slots[plan.result_slot] == out_scalars);
    if (plan.takes_scale) CHECK(plan.steps.back().kind == Kernel::strided && !plan.divides_by_n);
    if (strided_product > 1 && plan.steps.front().kind == Kernel::strided) CHECK(strided_product == rq.n);
    // The first step reads the input, the last writes the result, and each
    // step reads what the previous one wrote.
    CHECK(plan.steps.front().source == 0 && plan.steps.back().target == plan.result_slot);
    for (size_t i = 1; i < plan.steps.size(); ++i) CHECK(plan.steps[i].source == plan.steps[i - 1].target);

    std::vector<Dispatch> dispatches;
    std::string reason;
    for (const uint64_t count : {uint64_t{1}, uint64_t{2}, uint64_t{3}, uint64_t{7}}) {
        if (count > INT32_MAX / (rq.n * rq.tail)) continue;
        CHECK(instantiate(plan, count, dispatches, reason) && dispatches.size() == plan.steps.size());
        for (size_t i = 0; i < plan.steps.size(); ++i) {
            const Step& step = plan.steps[i];
            const uint64_t slots = dispatches[i].groups * static_cast<uint64_t>(step.width);
            if (step.kind == Kernel::strided) {
                CHECK(static_cast<uint64_t>(dispatches[i].blocks) == count * static_cast<uint64_t>(step.outer));
                CHECK(slots >= count * static_cast<uint64_t>(step.outer) * static_cast<uint64_t>(step.stride));
            } else {
                const uint64_t batch = count * step.transforms;
                CHECK(static_cast<uint64_t>(dispatches[i].values[static_cast<size_t>(step.batch_value)]) == batch);
                CHECK(slots * (step.paired ? 2 : 1) >= batch);
            }
        }
    }
}

std::shared_ptr<const Plan> must_plan(const Request& rq, const Limits& limits = {}) {
    const Planned planned = build_plan(rq, limits);
    if (!planned.plan) {
        std::fprintf(stderr, "no plan for n=%llu tail=%llu: %s\n", static_cast<unsigned long long>(rq.n),
                     static_cast<unsigned long long>(rq.tail), planned.reason.c_str());
        std::abort();
    }
    check_invariants(*planned.plan, limits);
    return planned.plan;
}

void must_decline(const Request& rq, const Limits& limits = {}) {
    const Planned planned = build_plan(rq, limits);
    if (planned.plan) {
        std::fprintf(stderr, "unexpected plan for n=%llu tail=%llu: %s\n", static_cast<unsigned long long>(rq.n),
                     static_cast<unsigned long long>(rq.tail), planned.plan->name.c_str());
        std::abort();
    }
    CHECK(!planned.reason.empty());
}

std::vector<int32_t> pass_lengths(const Plan& plan) {
    std::vector<int32_t> lengths;
    for (const Step& step : plan.steps) {
        if (step.kind == Kernel::strided) lengths.push_back(step.n);
        else if (step.kind == Kernel::four_step)
            lengths.push_back(step.kernel.find("_0_") != std::string::npos ? step.values[0] : step.values[1]);
    }
    return lengths;
}

void test_factoring() {
    CHECK(mlx::prime_factors(1).empty());
    CHECK((mlx::prime_factors(360) == std::vector<int32_t>{2, 2, 2, 3, 3, 5}));
    CHECK((mlx::prime_factors(1009) == std::vector<int32_t>{1009}));
    CHECK(mlx::smooth(1) && mlx::smooth(4096) && mlx::smooth(13 * 13 * 11 * 7 * 5 * 3 * 2) && mlx::smooth(1000000));
    CHECK(!mlx::smooth(0) && !mlx::smooth(17) && !mlx::smooth(2 * 1009) && !mlx::smooth(3001));
}

// Every length one kernel can reach, for every transform: a plan exactly when
// the length factors into the radices or is at most 2048.
void test_single_kernel_lengths() {
    int stockham = 0, rader = 0, bluestein = 0, declined = 0;
    for (uint64_t n = 2; n <= 4096; ++n) {
        const bool expected = mlx::smooth(n) || n <= 2048;
        for (const Transform transform : {Transform::c2c, Transform::r2c, Transform::c2r}) {
            for (const bool inverse : {false, true}) {
                if (transform != Transform::c2c && inverse) continue;
                const Request rq = request(transform, n, inverse);
                if (!expected) {
                    must_decline(rq);
                    continue;
                }
                const auto plan = must_plan(rq);
                CHECK(plan->steps.size() == 1);
                const Step& step = plan->steps[0];
                CHECK(step.paired == (transform != Transform::c2c));
                CHECK(step.constants.inverse == (inverse || transform == Transform::c2r));
                CHECK(plan->divides_by_n == step.constants.inverse && !plan->takes_scale);
                CHECK(plan->load == Load::direct && plan->store == Store::direct);
                CHECK(plan->input_real == (transform == Transform::r2c) && plan->result_real == (transform == Transform::c2r));
                CHECK(plan->input_row == (transform == Transform::c2r ? n / 2 + 1 : n));
                CHECK(plan->result_row == (transform == Transform::r2c ? n / 2 + 1 : n));
                if (mlx::smooth(n)) CHECK(step.kind == Kernel::stockham);
                else CHECK(step.kind == Kernel::rader || step.kind == Kernel::bluestein);
                // Only the fused Bluestein inverse real kernel needs imag(x[0]) cleared.
                CHECK(plan->clear_dc_imag == (step.kind == Kernel::bluestein && transform == Transform::c2r));
            }
        }
        if (!expected) {
            ++declined;
            continue;
        }
        const auto plan = must_plan(request(Transform::c2c, n));
        stockham += plan->steps[0].kind == Kernel::stockham;
        rader += plan->steps[0].kind == Kernel::rader;
        bluestein += plan->steps[0].kind == Kernel::bluestein;
    }
    // 489 lengths in [2, 4096] factor into the radices, 139 of them above 2048;
    // the rest need Rader or Bluestein, which one kernel covers up to 2048.
    CHECK(stockham == 489 && stockham + rader + bluestein + declined == 4095);
    std::printf("single kernel: %d stockham, %d rader, %d bluestein, %d declined\n", stockham, rader, bluestein, declined);
    CHECK(rader == 944 && bluestein == 753 && declined == 2048 - 139);

    // Known members of each class.
    CHECK(must_plan(request(Transform::c2c, 17))->steps[0].kind == Kernel::rader);
    CHECK(must_plan(request(Transform::c2c, 1009))->steps[0].kind == Kernel::rader);
    CHECK(must_plan(request(Transform::c2c, 2 * 17 * 3))->steps[0].kind == Kernel::rader);
    CHECK(must_plan(request(Transform::c2c, 47))->steps[0].kind == Kernel::bluestein);       // 46 = 2 * 23
    CHECK(must_plan(request(Transform::c2c, 17 * 17))->steps[0].kind == Kernel::bluestein);  // two Rader factors
    CHECK(must_plan(request(Transform::c2c, 1021))->steps[0].kind == Kernel::bluestein);
    must_decline(request(Transform::c2c, 3001));
    must_decline(request(Transform::c2c, 2 * 1031));
    must_decline(request(Transform::c2c, 1));
    must_decline(request(Transform::c2c, 0));
}

void test_powers_of_two() {
    for (int log2n = 13; log2n <= 30; ++log2n) {
        const uint64_t n = uint64_t{1} << log2n;
        for (const Transform transform : {Transform::c2c, Transform::r2c, Transform::c2r}) {
            const auto plan = must_plan(request(transform, n, false));
            const std::vector<int32_t> lengths = pass_lengths(*plan);
            CHECK(lengths.size() == plan->steps.size() && lengths.size() >= 2);
            uint64_t product = 1;
            for (const int32_t length : lengths) product *= static_cast<uint64_t>(length);
            CHECK(product == n);
            CHECK(plan->steps.front().kind == Kernel::four_step && plan->steps.back().kind == Kernel::four_step);
            for (size_t i = 1; i + 1 < plan->steps.size(); ++i) CHECK(plan->steps[i].kind == Kernel::strided);
            if (log2n <= 20) {
                // MLX's own split and coalescing.
                CHECK(lengths.size() == 2 && lengths[1] == (n > 65536 ? 1024 : 64));
                CHECK(plan->steps[0].width == 4 && plan->steps[1].width == 4);
            } else if (log2n <= 22) {
                CHECK(lengths.size() == 2 && lengths[1] == 2048 && plan->steps[1].width == 2);
            } else {
                CHECK(lengths[0] == 16);
                for (const int32_t length : lengths) CHECK(length == 16 || length == 32);
            }
            // Real plans keep full-size complex scratch and the right result layout.
            if (transform == Transform::r2c) {
                CHECK(plan->input_real && !plan->result_real && plan->result_slot == 0 && plan->result_row == n / 2 + 1);
                CHECK(plan->slots[0] == 2 * (n / 2 + 1) && plan->slots[1] == 2 * n);
                CHECK(plan->steps.front().kernel.find("float_float2_0_true") != std::string::npos);
                CHECK(plan->steps.back().kernel.find("float2_float2_1_true") != std::string::npos);
            } else if (transform == Transform::c2r) {
                CHECK(!plan->input_real && plan->result_real && plan->slots[plan->result_slot] == n);
                CHECK(plan->steps.front().kernel.find("float2_float2_0_true") != std::string::npos);
                CHECK(plan->steps.back().kernel.find("float2_float_1_true") != std::string::npos);
                CHECK(plan->divides_by_n);
            } else {
                CHECK(plan->slots.size() == 2 && plan->slots[0] == 2 * n && plan->slots[1] == 2 * n);
            }
        }
    }
    CHECK((pass_lengths(*must_plan(request(Transform::c2c, uint64_t{1} << 16))) == std::vector<int32_t>{1024, 64}));
    CHECK((pass_lengths(*must_plan(request(Transform::c2c, uint64_t{1} << 20))) == std::vector<int32_t>{1024, 1024}));
    CHECK((pass_lengths(*must_plan(request(Transform::c2c, uint64_t{1} << 21))) == std::vector<int32_t>{1024, 2048}));
    CHECK((pass_lengths(*must_plan(request(Transform::c2c, uint64_t{1} << 22))) == std::vector<int32_t>{2048, 2048}));
    CHECK((pass_lengths(*must_plan(request(Transform::c2c, uint64_t{1} << 23))) == std::vector<int32_t>{16, 32, 32, 32, 16}));
    CHECK((pass_lengths(*must_plan(request(Transform::c2c, uint64_t{1} << 24))) == std::vector<int32_t>{16, 32, 32, 32, 32}));
    must_decline(request(Transform::c2c, uint64_t{1} << 31));
}

// Lengths above one kernel's reach that are not powers of two: a chain of
// strided passes when the length factors into the radices, nothing otherwise.
void test_long_lengths() {
    int planned = 0;
    for (const uint64_t n : {uint64_t{4098}, uint64_t{4100}, uint64_t{4374}, uint64_t{5000}, uint64_t{6000}, uint64_t{6561},
                             uint64_t{10000}, uint64_t{19683}, uint64_t{28561}, uint64_t{30030}, uint64_t{65520},
                             uint64_t{100000}, uint64_t{371293}, uint64_t{1000000}, uint64_t{1048576 * 3},
                             uint64_t{9765625}, uint64_t{10000000}, uint64_t{16777216 * 5}, uint64_t{1000000000}}) {
        for (const Transform transform : {Transform::c2c, Transform::r2c, Transform::c2r}) {
            for (const uint64_t count : {uint64_t{1}, uint64_t{4096}}) {
                const Request rq = request(transform, n, false, 1, count);
                if (!mlx::smooth(n)) {
                    must_decline(rq);
                    continue;
                }
                const auto plan = must_plan(rq);
                ++planned;
                const std::vector<int32_t> lengths = pass_lengths(*plan);
                CHECK(lengths.size() == plan->steps.size());
                CHECK(std::accumulate(lengths.begin(), lengths.end(), uint64_t{1},
                                      [](uint64_t a, int32_t b) { return a * static_cast<uint64_t>(b); }) == n);
                CHECK(std::is_sorted(lengths.begin(), lengths.end()));
                // Large batches use short passes; small ones as few passes as fit.
                const bool large = n * count >= (uint64_t{1} << 22);
                for (const int32_t length : lengths) CHECK(length <= (large ? 32 : 128) || lengths.size() == 1);
                // Real transforms run as their complex equivalent with host adaptation.
                CHECK(plan->takes_scale && !plan->divides_by_n && !plan->input_real && !plan->result_real);
                CHECK(plan->input_row == n && plan->result_row == n);
                CHECK(plan->load == (transform == Transform::c2r ? Load::half_spectrum : Load::direct));
                CHECK(plan->store == (transform == Transform::r2c ? Store::half
                                      : transform == Transform::c2r ? Store::real_part : Store::direct));
                const uint32_t inverse = transform == Transform::c2r;
                CHECK((plan->steps.front().flags & SM_FFT_CONJ_LOAD) == (inverse ? SM_FFT_CONJ_LOAD : 0u));
                CHECK((plan->steps.back().flags & SM_FFT_CONJ_STORE) == (inverse ? SM_FFT_CONJ_STORE : 0u));
            }
        }
    }
    CHECK(planned > 60);
    CHECK((pass_lengths(*must_plan(request(Transform::c2c, 1000000))) == std::vector<int32_t>{100, 100, 100}));
    CHECK((pass_lengths(*must_plan(request(Transform::c2c, 6000))) == std::vector<int32_t>{60, 100}));
    for (const uint64_t n : {uint64_t{4099}, uint64_t{5003}, uint64_t{8194}, uint64_t{100003}, uint64_t{2 * 1048576 + 34},
                             uint64_t{17 * 4096}}) {
        for (const Transform transform : {Transform::c2c, Transform::r2c, Transform::c2r}) must_decline(request(transform, n));
    }
    // Every length whose factors are radices, up to 2^17, has a plan.
    int smooth = 0;
    for (uint64_t n = 4097; n <= (uint64_t{1} << 17); ++n) {
        if (!mlx::smooth(n)) continue;
        must_plan(request(Transform::c2c, n, true));
        ++smooth;
    }
    CHECK(smooth > 500);
}

// Transforms along an axis that is not the last.
void test_inner_axes() {
    for (const uint64_t n : {uint64_t{2}, uint64_t{3}, uint64_t{16}, uint64_t{30}, uint64_t{96}, uint64_t{1000},
                             uint64_t{4096}, uint64_t{65536}, uint64_t{1000000}}) {
        for (const uint64_t tail : {uint64_t{2}, uint64_t{7}, uint64_t{129}, uint64_t{2049}}) {
            if (n > INT32_MAX / tail) continue;
            for (const bool inverse : {false, true}) {
                const auto plan = must_plan(request(Transform::c2c, n, inverse, tail, 3));
                for (const Step& step : plan->steps) CHECK(step.kind == Kernel::strided);
                CHECK(plan->takes_scale && plan->slots[0] == 2 * n * tail);
            }
        }
    }
    // One short pass when everything is in cache, short passes for big arrays.
    CHECK(must_plan(request(Transform::c2c, 96, false, 7, 5))->steps.size() == 1);
    CHECK((pass_lengths(*must_plan(request(Transform::c2c, 4096, false, 4096, 1))) == std::vector<int32_t>{16, 16, 16}));
    must_decline(request(Transform::c2c, 17, false, 4));
    must_decline(request(Transform::c2c, 1021, false, 4));
    must_decline(request(Transform::r2c, 16, false, 4));
    must_decline(request(Transform::c2r, 16, false, 4));
    must_decline(request(Transform::c2c, uint64_t{1} << 20, false, uint64_t{1} << 12));
}

// Shader validation halves the threadgroup memory: plans must stay within it
// and still cover what they can.
void test_reduced_limits() {
    Limits limits;
    limits.group_elements = 2048;
    for (uint64_t n = 2; n <= 8192; ++n) {
        const Planned planned = build_plan(request(Transform::c2c, n), limits);
        if (mlx::smooth(n)) CHECK(planned.plan);
        if (planned.plan) check_invariants(*planned.plan, limits);
        // One kernel still reaches every length whose (padded) transform fits.
        if (n <= 1024) CHECK(planned.plan && planned.plan->steps.size() == 1);
    }
    for (int log2n = 12; log2n <= 26; ++log2n) {
        for (const Transform transform : {Transform::c2c, Transform::r2c, Transform::c2r}) {
            const auto plan = must_plan(request(transform, uint64_t{1} << log2n), limits);
            CHECK(plan->steps.size() >= 3);
        }
    }
    must_plan(request(Transform::c2c, 4096, false, 2049), limits);
}

// A pipeline can be granted fewer threads per threadgroup than the 1024 the
// planner assumes; plans built for the smaller grant stay within it.
void test_thread_limits() {
    const auto limited = [](int32_t threads) {
        Limits limits;
        limits.group_threads = threads;
        return limits;
    };
    // Rader over a radix-3 thread layout (625 threads) falls back to Bluestein (512).
    CHECK(must_plan(request(Transform::c2c, 1874))->steps[0].kind == Kernel::rader);
    CHECK(must_plan(request(Transform::c2c, 1874))->steps[0].threads == 625);
    for (const Transform transform : {Transform::c2c, Transform::r2c, Transform::c2r}) {
        const auto plan = must_plan(request(transform, 1874), limited(576));
        CHECK(plan->steps[0].kind == Kernel::bluestein && plan->steps[0].threads == 512);
        CHECK(plan->clear_dc_imag == (transform == Transform::c2r));
    }
    must_decline(request(Transform::c2c, 1874), limited(448));   // Bluestein needs 512
    CHECK(must_plan(request(Transform::c2c, 1009), limited(448))->steps[0].kind == Kernel::rader);
    // A Stockham kernel with too many threads becomes a chain of strided passes.
    CHECK(must_plan(request(Transform::c2c, 3900))->steps[0].threads == 780);
    for (const Transform transform : {Transform::c2c, Transform::r2c, Transform::c2r}) {
        const auto plan = must_plan(request(transform, 3900), limited(704));
        CHECK(plan->steps.size() == 2 && plan->steps[0].kind == Kernel::strided && plan->takes_scale);
        CHECK(plan->store == (transform == Transform::r2c ? Store::half
                              : transform == Transform::c2r ? Store::real_part : Store::direct));
    }
    // Short transforms share a threadgroup with fewer neighbours.
    CHECK(must_plan(request(Transform::c2c, 8))->steps[0].width == 32);
    CHECK(must_plan(request(Transform::c2c, 8), limited(16))->steps[0].width == 8);
    CHECK(must_plan(request(Transform::c2c, 8), limited(2))->steps[0].width == 1);
    // Even one thread per threadgroup leaves a plan: passes of one radix each.
    CHECK(must_plan(request(Transform::c2c, 8), limited(1))->steps.size() == 2);
    // Powers of two: fewer transforms per threadgroup, then shorter passes.
    for (const int32_t threads : {768, 512, 448, 256, 128, 100, 64, 48, 16, 12, 4}) {
        for (const int log2n : {4, 6, 12, 13, 16, 20, 22, 24}) {
            for (const Transform transform : {Transform::c2c, Transform::r2c, Transform::c2r}) {
                const auto plan = must_plan(request(transform, uint64_t{1} << log2n), limited(threads));
                for (const Step& step : plan->steps) CHECK(step.width * step.threads <= threads);
            }
        }
        must_plan(request(Transform::c2c, 1000000), limited(threads));
        must_plan(request(Transform::c2c, 3000, true, 77), limited(threads));
    }
    CHECK(must_plan(request(Transform::c2c, uint64_t{1} << 20), limited(448))->steps.size() > 2);

    // replan replaces what the cache answers for the original limits.
    const Request rq = request(Transform::c2c, 1973);  // prime, 1972 = 4 * 17 * 29: Bluestein from the start
    CHECK(plan(rq).plan->steps[0].kind == Kernel::bluestein);
    const Request rader = request(Transform::r2c, 1874, false, 1, 5);
    CHECK(plan(rader).plan->steps[0].kind == Kernel::rader);
    const Planned replaced = replan(rader, Limits{}, 576);
    CHECK(replaced.plan && replaced.plan->steps[0].kind == Kernel::bluestein);
    CHECK(plan(rader).plan == replaced.plan);
    CHECK(plan(request(Transform::c2c, 1874)).plan->steps[0].kind == Kernel::rader);  // other transforms keep theirs
    CHECK(!replan(rader, Limits{}, 448).plan && !plan(rader).plan);
    CHECK(!replan(rader, Limits{}, 0).plan);
}

void test_explicit_factorisations() {
    const Request rq = request(Transform::c2c, 65536);
    for (const std::vector<int32_t>& factors : std::vector<std::vector<int32_t>>{
             {256, 256}, {1024, 64}, {64, 1024}, {16, 16, 16, 16}, {4, 4096, 4}, {32, 64, 32}, {16, 4096}}) {
        const Planned planned = build_multipass(rq, factors);
        CHECK(planned.plan);
        check_invariants(*planned.plan, {});
        CHECK(pass_lengths(*planned.plan) == factors);
    }
    CHECK(!build_multipass(rq, {256, 128}).plan);        // wrong product
    CHECK(!build_multipass(rq, {8192, 8}).plan);         // a pass longer than a threadgroup holds
    CHECK(!build_multipass(request(Transform::c2c, 34), {2, 17}).plan);  // 17 is not a radix
    const Planned chain = build_multipass(request(Transform::r2c, 3000), {30, 100});
    CHECK(chain.plan && chain.plan->steps[0].kind == Kernel::strided && chain.plan->store == Store::half);
    const Planned single = build_multipass(request(Transform::c2c, 96, true, 5), {96});
    CHECK(single.plan && single.plan->steps.size() == 1 && single.plan->steps[0].flags == (SM_FFT_CONJ_LOAD | SM_FFT_CONJ_STORE));
    check_invariants(*single.plan, {});
}

void test_instantiate_limits() {
    std::vector<Dispatch> dispatches;
    std::string reason;
    const auto plan = must_plan(request(Transform::c2c, 4096));
    CHECK(instantiate(*plan, 524287, dispatches, reason));   // 2^31 - 4096 points
    CHECK(dispatches[0].groups == 524287 && dispatches[0].values[1] == 524287);
    CHECK(!instantiate(*plan, 524288, dispatches, reason) && !reason.empty());  // 2^31 points
    CHECK(!instantiate(*plan, 0, dispatches, reason));
    // A real kernel transforms two rows per slot.
    const auto real = must_plan(request(Transform::r2c, 1000));
    CHECK(real->steps[0].paired && real->steps[0].width == 1);
    CHECK(instantiate(*real, 7, dispatches, reason) && dispatches[0].groups == 4 && dispatches[0].values[1] == 7);
    const auto small = must_plan(request(Transform::c2c, 8));
    CHECK(small->steps[0].width == 32 && instantiate(*small, 100, dispatches, reason) && dispatches[0].groups == 4);
}

void test_cache() {
    const Request rq = request(Transform::c2c, 3 * 5 * 7 * 11);
    const uint64_t before = plans_built();
    const Planned first = plan(rq);
    const Planned again = plan(rq);
    CHECK(first.plan && first.plan == again.plan && plans_built() == before + 1);
    // The batch size only matters to multi-pass plans, and there only by class.
    Request batched = rq;
    batched.count = 1 << 20;
    CHECK(plan(batched).plan == first.plan);
    Request long_small = request(Transform::c2c, 6000), long_large = long_small;
    long_large.count = 4096;
    const Planned small = plan(long_small), large = plan(long_large);
    CHECK(small.plan && large.plan && small.plan != large.plan);
    long_large.count = 8192;
    CHECK(plan(long_large).plan == large.plan);
    // The direction, the transform and the limits are part of the key.
    Request inverse = rq;
    inverse.inverse = true;
    CHECK(plan(inverse).plan != first.plan);
    Limits reduced;
    reduced.group_elements = 2048;
    CHECK(plan(request(Transform::c2c, 4096), reduced).plan != plan(request(Transform::c2c, 4096)).plan);
    // Declines are remembered too.
    const uint64_t built = plans_built();
    CHECK(!plan(request(Transform::c2c, 3001)).plan && !plan(request(Transform::c2c, 3001)).plan);
    CHECK(plans_built() == built + 1);
    // Tables are shared between plans of one length.
    const Planned forward = plan(request(Transform::c2c, 1021)), backward = plan(request(Transform::c2c, 1021, true));
    CHECK(forward.plan->tables[0] == backward.plan->tables[0]);
}

void test_constants_key() {
    const auto a = must_plan(request(Transform::c2c, 1024)), b = must_plan(request(Transform::c2c, 1024, true));
    const auto c = must_plan(request(Transform::c2c, 1000));
    CHECK(a->steps[0].constants.key() != b->steps[0].constants.key());
    CHECK(a->steps[0].constants.key() != c->steps[0].constants.key());
    CHECK(a->steps[0].constants.key() == must_plan(request(Transform::c2c, 1024))->steps[0].constants.key());
}
// On CPU storage (the only kind a build without Metal has) there is no engine
// for an FFT: well-formed requests decline, malformed ones are still errors.
void test_abi_without_gpu() {
    sm_context_options context_options{};
    context_options.struct_size = sizeof(context_options);
    context_options.abi_version = SM_RUNTIME_ABI_VERSION;
    context_options.storage_kind = SM_BUFFER_CPU;
    context_options.memory_limit_bytes = uint64_t{1} << 26;
    sm_context* context = nullptr;
    CHECK(sm_context_create(&context_options, &context) == SM_OK);

    std::vector<float> data(2 * 4 * 4096, 1.0f);
    sm_array array{};
    array.struct_size = sizeof(array);
    array.abi_version = SM_ARRAY_ABI_VERSION;
    array.data = data.data();
    array.dtype = SM_COMPLEX64;
    array.ndim = 2;
    array.shape[0] = 4;
    array.shape[1] = 4096;
    sm_fft_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = SM_FFT_ABI_VERSION;
    options.kind = SM_FFT_FORWARD;
    options.axis_count = 1;
    options.axes[0] = 1;
    sm_route_options route{};
    route.struct_size = sizeof(route);
    route.abi_version = SM_ARRAY_ABI_VERSION;
    route.precision = SM_PRECISION_FAST;
    sm_array_result result{};
    sm_route_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = SM_ARRAY_ABI_VERSION;
    for (const uint32_t execution : {SM_EXECUTION_CPU, SM_EXECUTION_GPU, SM_EXECUTION_AUTO}) {
        route.execution = execution;
        CHECK(sm_fft_transform(context, &array, &options, &route, &result, &info) == SM_ENOTSUP);
        CHECK(result.buffer == nullptr && info.engine == 0);
        sm_fft_plan_info plan_info{};
        plan_info.struct_size = sizeof(plan_info);
        plan_info.abi_version = SM_FFT_ABI_VERSION;
        CHECK(sm_fft_plan(context, SM_COMPLEX64, 2, array.shape, &options, &route, 1, &plan_info) == SM_OK);
        CHECK(plan_info.status == SM_ENOTSUP && plan_info.passes == 0 && plan_info.description[0] != 0);
    }
    options.axes[0] = 2;
    CHECK(sm_fft_transform(context, &array, &options, &route, &result, &info) == SM_EINVAL);
    options.axes[0] = 1;
    options.kind = SM_FFT_R2C;
    CHECK(sm_fft_transform(context, &array, &options, &route, &result, &info) == SM_EINVAL);
    CHECK(sm_fft_auto_min_points() > 1024);

    // Review regressions: C2R 2*(have-1) overflow check, prime_factors(INT32_MAX),
    // ReadWriter width=1 mitigation, Bluestein C2R input_tail_scalars, clear_hermitian_imag.
    {
        const uint64_t overflow_shape[2] = {2, (uint64_t{1} << 63) + 1};
        options.kind = SM_FFT_C2R;
        options.axes[0] = 1;
        sm_fft_plan_info plan_info{};
        plan_info.struct_size = sizeof(plan_info);
        plan_info.abi_version = SM_FFT_ABI_VERSION;
        CHECK(sm_fft_plan(context, SM_COMPLEX64, 2, overflow_shape, &options, &route, 0, &plan_info) == SM_EOVERFLOW);
    }
    CHECK((mlx::prime_factors(INT32_MAX) == std::vector<int32_t>{INT32_MAX}));
    for (uint64_t n : {uint64_t{17}, uint64_t{19}, uint64_t{22}, uint64_t{23}, uint64_t{26}, uint64_t{33}}) {
        const auto p = must_plan(request(Transform::c2c, n));
        CHECK(p->steps[0].width == 1);
    }
    for (uint64_t n : {uint64_t{47}, uint64_t{1021}, uint64_t{1973}}) {
        const auto p = must_plan(request(Transform::c2r, n));
        CHECK(p->steps[0].kind == Kernel::bluestein && p->input_tail_scalars > 0);
    }
    CHECK(must_plan(request(Transform::c2r, 8192))->clear_hermitian_imag);
    CHECK(must_plan(request(Transform::c2r, 6000))->clear_hermitian_imag);

    sm_context_release(context);
}
} // namespace

int main() {
    test_abi_without_gpu();
    test_factoring();
    test_single_kernel_lengths();
    test_powers_of_two();
    test_long_lengths();
    test_inner_axes();
    test_reduced_limits();
    test_thread_limits();
    test_explicit_factorisations();
    test_instantiate_limits();
    test_cache();
    test_constants_key();
    std::puts("fft plan ok");
    return 0;
}
