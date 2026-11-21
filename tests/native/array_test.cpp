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

void test_integer_bool_convert_and_inexact_report() {
    // 1. ConvertReport::inexact_int boundary at +-((1<<24)-1) vs +-(1<<24) for 32/64-bit ints.
    auto check_signed_inexact = [](auto tag) {
        using T = decltype(tag);
        const uint32_t dt = std::is_same_v<T, int32_t> ? MS_INT32 : MS_INT64;
        for (size_t count : {size_t{5}, size_t{300000}}) {
            std::vector<T> exact(count, static_cast<T>(-3));
            exact[count - 1] = static_cast<T>((int64_t{1} << 24) - 1);
            exact[0] = static_cast<T>(-((int64_t{1} << 24) - 1));
            std::vector<float> dst(count, 0.0f);
            array::ConvertReport r;
            array::convert(exact.data(), dt, dst.data(), MS_FLOAT32, count, &r);
            CHECK(!r.inexact_int && !r.overflowed && !r.has_nonfinite);
            CHECK(r.max_finite == static_cast<double>((int64_t{1} << 24) - 1));
            CHECK(dst[0] == -16777215.0f && dst[count - 1] == 16777215.0f);

            for (int64_t lim : {int64_t{1} << 24, -(int64_t{1} << 24)}) {
                std::vector<T> big(count, static_cast<T>(7));
                big[count - 1] = static_cast<T>(lim);
                array::ConvertReport rb;
                array::convert(big.data(), dt, dst.data(), MS_FLOAT32, count, &rb);
                CHECK(rb.inexact_int && !rb.overflowed && !rb.has_nonfinite);
                CHECK(rb.max_finite == static_cast<double>(int64_t{1} << 24));
            }
        }
    };
    check_signed_inexact(int32_t{});
    check_signed_inexact(int64_t{});

    auto check_unsigned_inexact = [](auto tag) {
        using T = decltype(tag);
        const uint32_t dt = std::is_same_v<T, uint32_t> ? MS_UINT32 : MS_UINT64;
        for (size_t count : {size_t{5}, size_t{300000}}) {
            std::vector<T> exact(count, static_cast<T>(2));
            exact[count - 1] = static_cast<T>((uint64_t{1} << 24) - 1);
            std::vector<float> dst(count, 0.0f);
            array::ConvertReport r;
            array::convert(exact.data(), dt, dst.data(), MS_FLOAT32, count, &r);
            CHECK(!r.inexact_int && !r.overflowed && !r.has_nonfinite);
            CHECK(r.max_finite == static_cast<double>((uint64_t{1} << 24) - 1));
            CHECK(dst[count - 1] == 16777215.0f);

            std::vector<T> big(count, static_cast<T>(2));
            big[count - 1] = static_cast<T>(uint64_t{1} << 24);
            array::ConvertReport rb;
            array::convert(big.data(), dt, dst.data(), MS_FLOAT32, count, &rb);
            CHECK(rb.inexact_int && !rb.overflowed && !rb.has_nonfinite);
            CHECK(rb.max_finite == static_cast<double>(uint64_t{1} << 24));
        }
    };
    check_unsigned_inexact(uint32_t{});
    check_unsigned_inexact(uint64_t{});

    // Small integers and bool are always exact in float32.
    {
        const uint8_t bvals[] = {0, 1, 2, 255};
        float bf[4]{};
        array::ConvertReport br;
        array::convert(bvals, MS_BOOL, bf, MS_FLOAT32, 4, &br);
        CHECK(!br.inexact_int && !br.overflowed && !br.has_nonfinite && br.max_finite == 1.0);
        CHECK(bf[0] == 0.0f && bf[1] == 1.0f && bf[2] == 1.0f && bf[3] == 1.0f);

        const int16_t i16[] = {-32768, 32767};
        float f16[2]{};
        array::ConvertReport r16;
        array::convert(i16, MS_INT16, f16, MS_FLOAT32, 2, &r16);
        CHECK(!r16.inexact_int && r16.max_finite == 32768.0);
        CHECK(f16[0] == -32768.0f && f16[1] == 32767.0f);
    }

    // 2. Single rounding for 64-bit integers -> float32 (no double-rounding via double).
    {
        const int64_t i64_val = (int64_t{1} << 62) + (int64_t{1} << 38) + 1;
        const float expected_i = static_cast<float>(i64_val);
        const float double_rounded_i = static_cast<float>(static_cast<double>(i64_val));
        CHECK(expected_i != double_rounded_i);
        float got_i = 0.0f;
        array::ConvertReport ri;
        array::convert(&i64_val, MS_INT64, &got_i, MS_FLOAT32, 1, &ri);
        CHECK(ri.inexact_int);
        CHECK(std::memcmp(&got_i, &expected_i, sizeof(float)) == 0);

        const uint64_t u64_val = (uint64_t{1} << 63) + (uint64_t{1} << 39) + 1;
        const float expected_u = static_cast<float>(u64_val);
        const float double_rounded_u = static_cast<float>(static_cast<double>(u64_val));
        CHECK(expected_u != double_rounded_u);
        float got_u = 0.0f;
        array::ConvertReport ru;
        array::convert(&u64_val, MS_UINT64, &got_u, MS_FLOAT32, 1, &ru);
        CHECK(ru.inexact_int);
        CHECK(std::memcmp(&got_u, &expected_u, sizeof(float)) == 0);
    }

    // 3. Exact integer-to-integer conversions including sign extension and >53-bit integers.
    {
        const int8_t neg = -1;
        uint64_t u64 = 0;
        array::convert(&neg, MS_INT8, &u64, MS_UINT64, 1);
        CHECK(u64 == UINT64_MAX);

        const int64_t big_i64 = (int64_t{1} << 60) + 1234567;
        uint64_t mid_u64 = 0;
        int64_t back_i64 = 0;
        array::convert(&big_i64, MS_INT64, &mid_u64, MS_UINT64, 1);
        array::convert(&mid_u64, MS_UINT64, &back_i64, MS_INT64, 1);
        CHECK(back_i64 == big_i64);
    }

    // 4. Float -> int/bool saturation & arm64 truncation semantics.
    {
        const double qnan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();
        const double fvals[] = {-0.0, 0.5, -1.5, 1e20, -1e20, qnan, -inf};
        uint8_t b_out[7]{};
        array::convert(fvals, MS_FLOAT64, b_out, MS_BOOL, 7);
        CHECK(b_out[0] == 0 && b_out[1] == 1 && b_out[2] == 1 && b_out[3] == 1 && b_out[4] == 1 &&
              b_out[5] == 1 && b_out[6] == 1);

        uint8_t u8_out[7]{};
        array::convert(fvals, MS_FLOAT64, u8_out, MS_UINT8, 7);
        CHECK(u8_out[0] == 0 && u8_out[1] == 0 && u8_out[2] == 255 && u8_out[3] == 255 && u8_out[4] == 0 &&
              u8_out[5] == 0 && u8_out[6] == 0);

        int32_t i32_out[7]{};
        array::convert(fvals, MS_FLOAT64, i32_out, MS_INT32, 7);
        CHECK(i32_out[0] == 0 && i32_out[1] == 0 && i32_out[2] == -1 && i32_out[3] == INT32_MAX &&
              i32_out[4] == INT32_MIN && i32_out[5] == 0 && i32_out[6] == INT32_MIN);

        uint64_t u64_out[7]{};
        array::convert(fvals, MS_FLOAT64, u64_out, MS_UINT64, 7);
        CHECK(u64_out[0] == 0 && u64_out[1] == 0 && u64_out[2] == 0 && u64_out[3] == UINT64_MAX &&
              u64_out[4] == 0 && u64_out[5] == 0 && u64_out[6] == 0);
    }

    // 5. ms_array_astype across all 13 dtypes through C ABI.
    ms_context_options copts{};
    copts.struct_size = sizeof(copts);
    copts.abi_version = MS_RUNTIME_ABI_VERSION;
    copts.storage_kind = MS_BUFFER_CPU;
    copts.memory_limit_bytes = uint64_t{64} << 20;
    ms_context* ctx = nullptr;
    CHECK(ms_context_create(&copts, &ctx) == MS_OK);

    const uint32_t real_dtypes[] = {MS_FLOAT32, MS_FLOAT64, MS_BOOL,   MS_INT8,   MS_INT16, MS_INT32,
                                    MS_INT64,   MS_UINT8,   MS_UINT16, MS_UINT32, MS_UINT64};
    uint64_t buf_src[4]{};
    buf_src[0] = 0;
    buf_src[1] = 1;
    for (uint32_t sdt : real_dtypes) {
        ms_array in{};
        in.struct_size = sizeof(in);
        in.abi_version = MS_ARRAY_ABI_VERSION;
        in.data = buf_src;
        in.dtype = sdt;
        in.ndim = 1;
        in.shape[0] = 2;
        for (uint32_t ddt : real_dtypes) {
            ms_array_result res{};
            CHECK(ms_array_astype(ctx, &in, ddt, &res) == MS_OK);
            CHECK(res.dtype == ddt && res.ndim == 1 && res.shape[0] == 2);
            ms_buffer_release(res.buffer);
        }
        ms_array_result bad{};
        CHECK(ms_array_astype(ctx, &in, MS_COMPLEX64, &bad) == MS_EINVAL);
    }
    const double cvals[4] = {1.0, -2.0, 3.0, 4.0};
    ms_array cin{};
    cin.struct_size = sizeof(cin);
    cin.abi_version = MS_ARRAY_ABI_VERSION;
    cin.data = cvals;
    cin.dtype = MS_COMPLEX128;
    cin.ndim = 1;
    cin.shape[0] = 2;
    ms_array_result cres{};
    CHECK(ms_array_astype(ctx, &cin, MS_COMPLEX64, &cres) == MS_OK);
    CHECK(cres.dtype == MS_COMPLEX64);
    ms_buffer_release(cres.buffer);
    CHECK(ms_array_astype(ctx, &cin, MS_INT32, &cres) == MS_EINVAL);

    ms_context_release(ctx);
}

void test_review_regressions() {
    ms_context_options copts{};
    copts.struct_size = sizeof(copts);
    copts.abi_version = MS_RUNTIME_ABI_VERSION;
    copts.storage_kind = MS_BUFFER_CPU;
    copts.memory_limit_bytes = uint64_t{64} << 20;
    ms_context* ctx = nullptr;
    CHECK(ms_context_create(&copts, &ctx) == MS_OK);

    // 1. Scalar alignment check: misaligned pointers rejected, naturally-aligned complex accepted.
    alignas(16) uint8_t storage[64]{};
    {
        ms_array bad{};
        bad.struct_size = sizeof(bad);
        bad.abi_version = MS_ARRAY_ABI_VERSION;
        bad.data = storage + 1;
        bad.dtype = MS_FLOAT64;
        bad.ndim = 1;
        bad.shape[0] = 2;
        ms_array_result res{};
        CHECK(ms_array_astype(ctx, &bad, MS_FLOAT32, &res) == MS_EINVAL);

        bad.data = storage + 2;
        bad.dtype = MS_FLOAT32;
        CHECK(ms_array_astype(ctx, &bad, MS_FLOAT64, &res) == MS_EINVAL);

        bad.data = storage + 4;
        bad.dtype = MS_COMPLEX128;
        CHECK(ms_array_astype(ctx, &bad, MS_COMPLEX64, &res) == MS_EINVAL);

        // Naturally-aligned complex64 (+4) and complex128 (+8) must be accepted.
        ms_array ok_c64 = bad;
        ok_c64.data = storage + 4;
        ok_c64.dtype = MS_COMPLEX64;
        CHECK(ms_array_astype(ctx, &ok_c64, MS_COMPLEX128, &res) == MS_OK);
        ms_buffer_release(res.buffer);

        ms_array ok_c128 = bad;
        ok_c128.data = storage + 8;
        ok_c128.dtype = MS_COMPLEX128;
        CHECK(ms_array_astype(ctx, &ok_c128, MS_COMPLEX64, &res) == MS_OK);
        ms_buffer_release(res.buffer);
    }

    // 2. Non-zero ms_route_options::reserved rejected, and report() clears info->reserved[1].
    {
        ms_route_options ro{};
        ro.struct_size = sizeof(ro);
        ro.abi_version = MS_ARRAY_ABI_VERSION;
        ro.execution = MS_EXECUTION_AUTO;
        ro.precision = MS_PRECISION_STRICT;
        ro.reserved[0] = 1;
        bool caught = false;
        try {
            array::check_options(&ro);
        } catch (const ms::detail::Error& err) {
            caught = (err.status == MS_EINVAL);
        }
        CHECK(caught);

        ms_route_info info{};
        info.struct_size = sizeof(info);
        info.abi_version = MS_ARRAY_ABI_VERSION;
        info.reserved[0] = 0x1234;
        info.reserved[1] = 0xdeadbeef;
        array::report(&info, MS_EXECUTION_CPU, MS_FLOAT64, "ok");
        CHECK(info.reserved[0] == 0 && info.reserved[1] == 0);
    }

    // 3. ms_array_astype byte-size overflow returns MS_EOVERFLOW.
    {
        ms_array huge{};
        huge.struct_size = sizeof(huge);
        huge.abi_version = MS_ARRAY_ABI_VERSION;
        huge.data = storage;
        huge.dtype = MS_UINT8;
        huge.ndim = 1;
        huge.shape[0] = uint64_t{1} << 61;
        ms_array_result res{};
        CHECK(ms_array_astype(ctx, &huge, MS_FLOAT64, &res) == MS_EOVERFLOW);
    }

    // 4. Single-pass float32/complex64 copy+scan report (small and large NEON paths, including sNaN).
    for (size_t count : {size_t{7}, size_t{300000}}) {
        std::vector<float> src(count, 1.5f);
        src[count - 1] = -42.25f;
        std::vector<float> dst(count, 0.0f);
        array::ConvertReport r;
        array::convert(src.data(), MS_FLOAT32, dst.data(), MS_FLOAT32, count, &r);
        CHECK(!r.has_nonfinite && !r.overflowed && !r.inexact_int);
        CHECK(r.max_finite == 42.25);
        CHECK(std::memcmp(src.data(), dst.data(), count * sizeof(float)) == 0);

        const uint32_t snan_bits = 0x7f800001u;
        float snan = 0.0f;
        std::memcpy(&snan, &snan_bits, sizeof(float));
        src[count / 2] = snan;
        array::ConvertReport rn;
        array::convert(src.data(), MS_FLOAT32, dst.data(), MS_FLOAT32, count, &rn);
        CHECK(rn.has_nonfinite && rn.max_finite == 42.25);

        std::vector<float> csrc(count * 2, 0.5f), cdst(count * 2, 0.0f);
        csrc[2 * (count - 1) + 1] = -128.0f;
        array::ConvertReport cr;
        array::convert(csrc.data(), MS_COMPLEX64, cdst.data(), MS_COMPLEX64, count, &cr);
        CHECK(!cr.has_nonfinite && cr.max_finite == 128.0);
        CHECK(std::memcmp(csrc.data(), cdst.data(), count * 2 * sizeof(float)) == 0);
    }

    ms_context_release(ctx);
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
    test_integer_bool_convert_and_inexact_report();
    test_review_regressions();
    std::printf("array tests ok\n");
    return 0;
}
