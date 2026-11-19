#include "core/array.hpp"
#include "ms_array.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(x)                                                                                                      \
    do {                                                                                                              \
        if (!(x)) {                                                                                                   \
            std::fprintf(stderr, "array test failed line %d: %s\n", __LINE__, #x);                                    \
            std::abort();                                                                                             \
        }                                                                                                             \
    } while (0)

using namespace ms::detail;

namespace {
void test_convert_report_edge_cases() {
    // 1. Empty array
    {
        array::ConvertReport report;
        report.overflowed = true;
        report.has_nonfinite = true;
        report.max_finite = 42.0;
        report.max_magnitude = 42.0;
        array::convert(nullptr, MS_FLOAT64, nullptr, MS_FLOAT32, 0, &report);
        CHECK(!report.overflowed && !report.has_nonfinite);
        CHECK(report.max_finite == 0.0 && report.max_magnitude == 0.0);
    }

    // 2. Exact IEEE round-to-nearest-even boundary:
    //    FLT_MAX = 0x1.fffffep127; tie = 0x1.ffffffp127 rounds to +inf.
    const double tie = 0x1.ffffffp127;
    const double below_tie = std::nextafter(tie, 0.0);
    CHECK(std::isfinite(static_cast<float>(below_tie)));
    CHECK(!std::isfinite(static_cast<float>(tie)));

    // Just below tie -> no overflow (tested across multiple sizes including tail and multi-lane)
    for (size_t count : {size_t{1}, size_t{7}, size_t{8}, size_t{15}, size_t{1024}, size_t{300000}}) {
        std::vector<double> src(count, 1.0);
        std::vector<float> dst(count, 0.0f);
        src[count - 1] = (count & 1) ? below_tie : -below_tie;
        array::ConvertReport report;
        array::convert(src.data(), MS_FLOAT64, dst.data(), MS_FLOAT32, count, &report);
        CHECK(!report.overflowed);
        CHECK(!report.has_nonfinite);
        CHECK(report.max_finite == below_tie);
        CHECK(report.max_magnitude == below_tie);
        CHECK(std::fabs(dst[count - 1]) == FLT_MAX);
    }

    // At exact tie -> overflowed = true
    for (size_t pos : {size_t{0}, size_t{5}, size_t{8}, size_t{299999}}) {
        const size_t count = 300000;
        std::vector<double> src(count, 1.5);
        std::vector<float> dst(count, 0.0f);
        src[pos] = (pos & 1) ? -tie : tie;
        array::ConvertReport report;
        array::convert(src.data(), MS_FLOAT64, dst.data(), MS_FLOAT32, count, &report);
        CHECK(report.overflowed);
        CHECK(!report.has_nonfinite);
        CHECK(report.max_finite == tie);
        CHECK(report.max_magnitude == tie);
        CHECK(std::isinf(dst[pos]));
    }

    // 1e39 and -1e39 -> overflowed = true, has_nonfinite = false
    {
        std::vector<double> src(33, -2.5);
        src[17] = 1e39;
        src[31] = -1e39;
        std::vector<float> dst(33, 0.0f);
        array::ConvertReport report;
        array::convert(src.data(), MS_FLOAT64, dst.data(), MS_FLOAT32, src.size(), &report);
        CHECK(report.overflowed);
        CHECK(!report.has_nonfinite);
        CHECK(report.max_finite == 1e39);
        CHECK(report.max_magnitude == 1e39);
    }

    // Pre-existing +-inf and NaN do NOT set overflowed when finite values fit float32
    {
        const double inf = std::numeric_limits<double>::infinity();
        const double qnan = std::numeric_limits<double>::quiet_NaN();
        std::vector<double> src(300005, 1.25);
        src[0] = inf;
        src[7] = -inf;
        src[19] = qnan;
        src[299999] = -2.5;
        std::vector<float> dst(src.size(), 0.0f);
        array::ConvertReport report;
        array::convert(src.data(), MS_FLOAT64, dst.data(), MS_FLOAT32, src.size(), &report);
        CHECK(!report.overflowed);
        CHECK(report.has_nonfinite);
        CHECK(report.max_finite == 2.5);
        CHECK(std::isinf(report.max_magnitude) && report.max_magnitude > 0.0);
    }

    // Finite + NaN (no +-inf) -> max_magnitude == max_finite
    {
        const double qnan = std::numeric_limits<double>::quiet_NaN();
        std::vector<double> src = {1.0, qnan, -3.5, 2.0, qnan, 0.0, -0.0, 0.5, 1.5};
        std::vector<float> dst(src.size(), 0.0f);
        array::ConvertReport report;
        array::convert(src.data(), MS_FLOAT64, dst.data(), MS_FLOAT32, src.size(), &report);
        CHECK(!report.overflowed);
        CHECK(report.has_nonfinite);
        CHECK(report.max_finite == 3.5);
        CHECK(report.max_magnitude == 3.5);
    }

    // All NaNs -> max_finite == 0.0, max_magnitude == 0.0
    {
        const double qnan = std::numeric_limits<double>::quiet_NaN();
        std::vector<double> src(17, qnan);
        std::vector<float> dst(src.size(), 0.0f);
        array::ConvertReport report;
        array::convert(src.data(), MS_FLOAT64, dst.data(), MS_FLOAT32, src.size(), &report);
        CHECK(!report.overflowed);
        CHECK(report.has_nonfinite);
        CHECK(report.max_finite == 0.0);
        CHECK(report.max_magnitude == 0.0);
    }

    // Pre-existing inf + finite 1e39 -> overflowed == true AND has_nonfinite == true
    {
        const double inf = std::numeric_limits<double>::infinity();
        std::vector<double> src = {inf, 1e39, -inf, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<float> dst(src.size(), 0.0f);
        array::ConvertReport report;
        array::convert(src.data(), MS_FLOAT64, dst.data(), MS_FLOAT32, src.size(), &report);
        CHECK(report.overflowed);
        CHECK(report.has_nonfinite);
        CHECK(report.max_finite == 1e39);
        CHECK(std::isinf(report.max_magnitude));
    }

    // Signaling NaNs (+sNaN and -sNaN) must never poison max_magnitude via vmaxnmq_f64
    {
        double snan_pos = 0.0, snan_neg = 0.0;
        const uint64_t snan_pos_bits = 0x7ff0000000000001ULL;
        const uint64_t snan_neg_bits = 0xfff0000000000001ULL;
        std::memcpy(&snan_pos, &snan_pos_bits, sizeof(double));
        std::memcpy(&snan_neg, &snan_neg_bits, sizeof(double));
        const double inf = std::numeric_limits<double>::infinity();

        for (size_t count : {size_t{8}, size_t{9}, size_t{262145}}) {
            for (double snan : {snan_pos, snan_neg}) {
                const std::vector<std::pair<size_t, size_t>> slots = {
                    {0, count - 1},
                    {count - 1, 0},
                    {1, 7},
                };
                for (auto [snan_pos_idx, val_pos_idx] : slots) {
                    // Paired with 1e39
                    {
                        std::vector<double> src(count, 1.0);
                        std::vector<float> dst(count, 0.0f);
                        src[snan_pos_idx] = snan;
                        src[val_pos_idx] = 1e39;
                        array::ConvertReport report;
                        array::convert(src.data(), MS_FLOAT64, dst.data(), MS_FLOAT32, count, &report);
                        CHECK(report.overflowed && report.has_nonfinite);
                        CHECK(report.max_finite == 1e39);
                        CHECK(report.max_magnitude == 1e39);
                        CHECK(std::isnan(dst[snan_pos_idx]));
                    }
                    // Paired with 42.5
                    {
                        std::vector<double> src(count, 1.0);
                        std::vector<float> dst(count, 0.0f);
                        src[snan_pos_idx] = snan;
                        src[val_pos_idx] = 42.5;
                        array::ConvertReport report;
                        array::convert(src.data(), MS_FLOAT64, dst.data(), MS_FLOAT32, count, &report);
                        CHECK(!report.overflowed && report.has_nonfinite);
                        CHECK(report.max_finite == 42.5);
                        CHECK(report.max_magnitude == 42.5);
                        CHECK(std::isnan(dst[snan_pos_idx]));
                    }
                    // Paired with +inf and -inf
                    for (double inf_val : {inf, -inf}) {
                        std::vector<double> src(count, 1.0);
                        std::vector<float> dst(count, 0.0f);
                        src[snan_pos_idx] = snan;
                        src[val_pos_idx] = inf_val;
                        array::ConvertReport report;
                        array::convert(src.data(), MS_FLOAT64, dst.data(), MS_FLOAT32, count, &report);
                        CHECK(!report.overflowed && report.has_nonfinite);
                        CHECK(report.max_finite == 1.0);
                        CHECK(std::isinf(report.max_magnitude) && report.max_magnitude > 0.0);
                        CHECK(std::isnan(dst[snan_pos_idx]));
                    }
                }
                // All sNaNs
                {
                    std::vector<double> src(count, snan);
                    std::vector<float> dst(count, 0.0f);
                    array::ConvertReport report;
                    array::convert(src.data(), MS_FLOAT64, dst.data(), MS_FLOAT32, count, &report);
                    CHECK(!report.overflowed && report.has_nonfinite);
                    CHECK(report.max_finite == 0.0);
                    CHECK(report.max_magnitude == 0.0);
                }
            }
        }
    }

    // complex128 -> complex64: overflow in imaginary part triggers overflowed
    {
        std::vector<double> src(2 * 10, 1.0);
        src[2 * 9 + 1] = -1e39;
        std::vector<float> dst(2 * 10, 0.0f);
        array::ConvertReport report;
        array::convert(src.data(), MS_COMPLEX128, dst.data(), MS_COMPLEX64, 10, &report);
        CHECK(report.overflowed);
        CHECK(!report.has_nonfinite);
        CHECK(report.max_finite == 1e39);
    }

    // Bit-exactness vs scalar cast across special patterns
    {
        const std::vector<double> specials = {
            0.0, -0.0, 1.0, -1.0, 1e-40, -1e-40, 1e-38, 1.5e-45,
            FLT_MAX, -FLT_MAX, below_tie, -below_tie, tie, -tie, 1e39, -1e39,
            std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::quiet_NaN()
        };
        std::vector<double> src(300000 + specials.size());
        for (size_t i = 0; i < src.size(); ++i) src[i] = specials[i % specials.size()];
        std::vector<float> dst(src.size(), 0.0f);
        array::ConvertReport report;
        array::convert(src.data(), MS_FLOAT64, dst.data(), MS_FLOAT32, src.size(), &report);
        CHECK(report.overflowed && report.has_nonfinite && report.max_finite == 1e39);
        for (size_t i = 0; i < src.size(); ++i) {
            const float ref = static_cast<float>(src[i]);
            if (std::isnan(ref)) {
                CHECK(std::isnan(dst[i]));
            } else {
                CHECK(std::memcmp(&dst[i], &ref, sizeof(float)) == 0);
            }
        }
    }
}

void test_route_info_and_report() {
    ms_route_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = MS_ARRAY_ABI_VERSION;
    array::check_route_info(nullptr);
    array::check_route_info(&info);

    array::report(&info, MS_EXECUTION_GPU, MS_FLOAT32, "");
    CHECK(info.engine == MS_EXECUTION_GPU && info.compute_dtype == MS_FLOAT32 && info.reason[0] == '\0');

    array::report(&info, MS_EXECUTION_CPU, MS_FLOAT64, "float64 overflowed");
    CHECK(info.engine == MS_EXECUTION_CPU && info.compute_dtype == MS_FLOAT64);
    CHECK(std::strcmp(info.reason, "float64 overflowed") == 0);

    // Long reason is safely NUL-terminated
    std::string long_reason(400, 'x');
    array::report(&info, MS_EXECUTION_CPU, MS_FLOAT64, long_reason.c_str());
    CHECK(std::strlen(info.reason) == sizeof(info.reason) - 1);

    // Bad version/size throws MS_EINVAL
    ms_route_info bad{};
    bool caught = false;
    try {
        array::check_route_info(&bad);
    } catch (const Error& err) {
        caught = (err.status == MS_EINVAL);
    }
    CHECK(caught);
}

void run_benchmark() {
    for (uint64_t bytes : {uint64_t{256} << 20, uint64_t{1} << 30}) {
        const uint64_t n = bytes / sizeof(double);
        std::vector<double> src(n);
        std::vector<float> dst(n, 0.0f);
        for (uint64_t i = 0; i < n; ++i) src[i] = static_cast<double>((i & 1023u)) * 0.125 - 64.0;

        auto bench_variant = [&](bool with_report) {
            array::ConvertReport report;
            array::ConvertReport* rp = with_report ? &report : nullptr;
            // Warmup
            array::convert(src.data(), MS_FLOAT64, dst.data(), MS_FLOAT32, n, rp);
            constexpr int reps = 7;
            std::vector<double> secs(reps);
            for (int r = 0; r < reps; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                array::convert(src.data(), MS_FLOAT64, dst.data(), MS_FLOAT32, n, rp);
                const auto t1 = std::chrono::steady_clock::now();
                secs[r] = std::chrono::duration<double>(t1 - t0).count();
            }
            std::sort(secs.begin(), secs.end());
            const double med_s = secs[reps / 2];
            const double gbps = static_cast<double>(bytes) / 1e9 / med_s;
            std::printf("array::convert %4llu MB (%s): %.3f ms, %.2f GB/s\n",
                        static_cast<unsigned long long>(bytes >> 20),
                        with_report ? "with_report" : "no_report  ",
                        med_s * 1e3, gbps);
            if (with_report) {
                CHECK(!report.overflowed && !report.has_nonfinite);
                CHECK(report.max_finite > 0.0);
            }
        };
        bench_variant(false);
        bench_variant(true);
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        run_benchmark();
        return 0;
    }
    test_convert_report_edge_cases();
    test_route_info_and_report();
    std::printf("array tests ok\n");
    return 0;
}
