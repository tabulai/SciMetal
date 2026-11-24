// CPU engines for dense N-D min/max morphology and rank/median/percentile filters.
// Threaded across Apple Silicon P-cores; bit-identical to SciPy 1.18.1 arm64.
#include "morphology.hpp"
#include "filters.hpp"
#include "sm/detail/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>

namespace sm::detail::ndimage {
namespace {

struct BoolTag {};

// Exact replica of SciPy's NI_Select (ni_filters.c lines 675-708) for NaN-safe bit-identity.
double scipy_ni_select(double* buffer, int64_t min_i, int64_t max_i, int64_t rank) noexcept {
    while (min_i != max_i) {
        const double x = buffer[min_i];
        int64_t ii = min_i - 1;
        int64_t jj = max_i + 1;
        for (;;) {
            do {
                --jj;
            } while (buffer[jj] > x);
            do {
                ++ii;
            } while (buffer[ii] < x);
            if (ii < jj) {
                const double t = buffer[ii];
                buffer[ii] = buffer[jj];
                buffer[jj] = t;
            } else {
                break;
            }
        }
        const int64_t left_count = jj - min_i + 1;
        if (rank < left_count) {
            max_i = jj;
        } else {
            min_i = jj + 1;
            rank -= left_count;
        }
    }
    return buffer[min_i];
}

// Optimal 19-comparator median-of-9 for finite doubles (bit-exact order statistic).
inline double median9_finite(double* b) noexcept {
    auto cas = [](double& a, double& c) noexcept {
        if (c < a) {
            const double t = a;
            a = c;
            c = t;
        }
    };
    cas(b[0], b[1]); cas(b[1], b[2]); cas(b[0], b[1]);
    cas(b[3], b[4]); cas(b[4], b[5]); cas(b[3], b[4]);
    cas(b[6], b[7]); cas(b[7], b[8]); cas(b[6], b[7]);
    const double max_min = std::max(std::max(b[0], b[3]), b[6]);
    const double min_max = std::min(std::min(b[2], b[5]), b[8]);
    cas(b[1], b[4]); cas(b[4], b[7]); cas(b[1], b[4]);
    const double med_mid = b[4];
    double f[3] = {max_min, med_mid, min_max};
    cas(f[0], f[1]); cas(f[1], f[2]); cas(f[0], f[1]);
    return f[1];
}

// Reproduces SciPy's `(_type)_cval` cast in CASE_MIN_OR_MAX_POINT and CASE_RANK_POINT on arm64.
// Note: npy_bool in SciPy's C code is unsigned char (uint8_t), not C++ bool.
template<class InT>
inline typename std::conditional_t<std::is_same_v<InT, BoolTag>, uint8_t, InT> cast_cval_to_in(double cval) noexcept {
    if constexpr (std::is_same_v<InT, BoolTag>) {
        return array::cast_double_ndimage<uint8_t>(cval, /*dense_safe=*/false);
    } else {
        return array::cast_double_ndimage<InT>(cval, /*dense_safe=*/false);
    }
}

// Reproduces SciPy's CASE_FILTER_OUT_SAFE (bool/unsigned) and CASE_FILTER_OUT (signed/float).
inline void store_dense_output(uint32_t out_dtype, void* dst, uint64_t idx, double val) noexcept {
    switch (out_dtype) {
    case SM_FLOAT64: static_cast<double*>(dst)[idx] = val; break;
    case SM_FLOAT32: static_cast<float*>(dst)[idx] = static_cast<float>(val); break;
    case SM_BOOL:
    case SM_UINT8:
        static_cast<uint8_t*>(dst)[idx] = array::cast_double_ndimage<uint8_t>(val, /*dense_safe=*/true);
        break;
    case SM_UINT16:
        static_cast<uint16_t*>(dst)[idx] = array::cast_double_ndimage<uint16_t>(val, /*dense_safe=*/true);
        break;
    case SM_UINT32:
        static_cast<uint32_t*>(dst)[idx] = array::cast_double_ndimage<uint32_t>(val, /*dense_safe=*/true);
        break;
    case SM_UINT64:
        static_cast<uint64_t*>(dst)[idx] = array::cast_double_ndimage<uint64_t>(val, /*dense_safe=*/true);
        break;
    case SM_INT8:
        static_cast<int8_t*>(dst)[idx] = array::cast_double_ndimage<int8_t>(val, /*dense_safe=*/false);
        break;
    case SM_INT16:
        static_cast<int16_t*>(dst)[idx] = array::cast_double_ndimage<int16_t>(val, /*dense_safe=*/false);
        break;
    case SM_INT32:
        static_cast<int32_t*>(dst)[idx] = array::cast_double_ndimage<int32_t>(val, /*dense_safe=*/false);
        break;
    case SM_INT64:
        static_cast<int64_t*>(dst)[idx] = array::cast_double_ndimage<int64_t>(val, /*dense_safe=*/false);
        break;
    default: break;
    }
}

// Exact modular addition matching C `_tmp += (_type)_ss[_ii]` in NI_MinOrMaxFilter.
template<class InT>
inline typename std::conditional_t<std::is_same_v<InT, BoolTag>, uint8_t, InT>
add_struct_in_type(typename std::conditional_t<std::is_same_v<InT, BoolTag>, uint8_t, InT> a,
                   typename std::conditional_t<std::is_same_v<InT, BoolTag>, uint8_t, InT> b) noexcept {
    if constexpr (std::is_same_v<InT, BoolTag>) {
        return static_cast<uint8_t>(a + b);
    } else if constexpr (std::is_floating_point_v<InT>) {
        return a + b;
    } else {
        using U = std::make_unsigned_t<InT>;
        return static_cast<InT>(static_cast<U>(static_cast<U>(a) + static_cast<U>(b)));
    }
}

template<class InT, bool IsMax, bool HasStruct>
void dense_minmax_typed(const DenseMinMaxPlan& plan, const void* input_raw, void* output_raw) {
    using MemT = std::conditional_t<std::is_same_v<InT, BoolTag>, uint8_t, InT>;
    const MemT* src = static_cast<const MemT*>(input_raw);
    const uint32_t ndim = plan.shape.ndim;
    const uint64_t last_len = plan.shape.dims[ndim - 1];
    const uint64_t rows = plan.shape.elements / last_len;
    const size_t num_taps = plan.taps.taps.size();
    const MorphTap* taps = plan.taps.taps.data();

    uint64_t strides[SM_MAX_NDIM] = {};
    strides[ndim - 1] = 1;
    for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; --d) {
        strides[d] = strides[d + 1] * plan.shape.dims[d + 1];
    }

    const MemT cval_t = cast_cval_to_in<InT>(plan.cval);
    std::vector<MemT> ss_t;
    double ss0_f64 = 0.0;
    if constexpr (HasStruct) {
        ss_t.resize(num_taps);
        ss0_f64 = taps[0].struct_offset_f64;
        for (size_t k = 0; k < num_taps; ++k) {
            ss_t[k] = cast_cval_to_in<InT>(taps[k].struct_offset_f64);
        }
    }

    const int64_t x_lo = plan.taps.has_interior ? plan.taps.interior_lo[ndim - 1] : static_cast<int64_t>(last_len);
    const int64_t x_hi = plan.taps.has_interior ? plan.taps.interior_hi[ndim - 1] : 0;
    const uint64_t row_work = std::max<uint64_t>(1, last_len * num_taps);
    const uint64_t grain = std::max<uint64_t>(1, (uint64_t{1} << 16) / row_work);

    parallel::parallel_for(rows, grain, [&](unsigned, uint64_t r0, uint64_t r1) {
        int64_t coord[SM_MAX_NDIM] = {};
        for (uint64_t row = r0; row < r1; ++row) {
            uint64_t rem = row;
            for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; --d) {
                coord[d] = static_cast<int64_t>(rem % plan.shape.dims[d]);
                rem /= plan.shape.dims[d];
            }
            bool outer_interior = plan.taps.has_interior;
            if (outer_interior) {
                for (uint32_t d = 0; d + 1 < ndim; ++d) {
                    if (coord[d] < plan.taps.interior_lo[d] || coord[d] >= plan.taps.interior_hi[d]) {
                        outer_interior = false;
                        break;
                    }
                }
            }

            const uint64_t row_base = row * last_len;
            const auto eval_boundary = [&](int64_t x) -> double {
                coord[ndim - 1] = x;
                auto load_tap = [&](size_t t) -> MemT {
                    uint64_t idx = 0;
                    for (uint32_t d = 0; d < ndim; ++d) {
                        const int64_t pos = coord[d] + taps[t].coord[d];
                        const int64_t dim = static_cast<int64_t>(plan.shape.dims[d]);
                        const int64_t q = (pos >= 0 && pos < dim) ? pos : extend(pos, dim, plan.mode);
                        if (q < 0) return cval_t;
                        idx += static_cast<uint64_t>(q) * strides[d];
                    }
                    return src[idx];
                };
                double res = static_cast<double>(load_tap(0));
                if constexpr (HasStruct) res += ss0_f64;
                for (size_t t = 1; t < num_taps; ++t) {
                    MemT tmp = load_tap(t);
                    if constexpr (HasStruct) tmp = add_struct_in_type<InT>(tmp, ss_t[t]);
                    const double td = static_cast<double>(tmp);
                    if constexpr (IsMax) {
                        if (td > res) res = td;
                    } else {
                        if (td < res) res = td;
                    }
                }
                return res;
            };

            const int64_t len = static_cast<int64_t>(last_len);
            if (!outer_interior || x_lo >= x_hi) {
                for (int64_t x = 0; x < len; ++x) {
                    store_dense_output(plan.out_dtype, output_raw, row_base + static_cast<uint64_t>(x),
                                       eval_boundary(x));
                }
            } else {
                for (int64_t x = 0; x < x_lo; ++x) {
                    store_dense_output(plan.out_dtype, output_raw, row_base + static_cast<uint64_t>(x),
                                       eval_boundary(x));
                }
                const int64_t off0 = taps[0].linear_offset;
                for (int64_t x = x_lo; x < x_hi; ++x) {
                    const int64_t base = static_cast<int64_t>(row_base) + x;
                    double res = static_cast<double>(src[base + off0]);
                    if constexpr (HasStruct) res += ss0_f64;
                    for (size_t t = 1; t < num_taps; ++t) {
                        MemT tmp = src[base + taps[t].linear_offset];
                        if constexpr (HasStruct) tmp = add_struct_in_type<InT>(tmp, ss_t[t]);
                        const double td = static_cast<double>(tmp);
                        if constexpr (IsMax) {
                            if (td > res) res = td;
                        } else {
                            if (td < res) res = td;
                        }
                    }
                    store_dense_output(plan.out_dtype, output_raw, static_cast<uint64_t>(base), res);
                }
                for (int64_t x = x_hi; x < len; ++x) {
                    store_dense_output(plan.out_dtype, output_raw, row_base + static_cast<uint64_t>(x),
                                       eval_boundary(x));
                }
            }
        }
    });
}

template<class InT>
void dispatch_dense_minmax_in(const DenseMinMaxPlan& plan, const void* input_raw, void* output_raw) {
    if (plan.is_max) {
        if (plan.taps.has_structure) dense_minmax_typed<InT, true, true>(plan, input_raw, output_raw);
        else dense_minmax_typed<InT, true, false>(plan, input_raw, output_raw);
    } else {
        if (plan.taps.has_structure) dense_minmax_typed<InT, false, true>(plan, input_raw, output_raw);
        else dense_minmax_typed<InT, false, false>(plan, input_raw, output_raw);
    }
}

template<class InT>
void rank_typed(const RankPlan& plan, const void* input_raw, void* output_raw) {
    using MemT = std::conditional_t<std::is_same_v<InT, BoolTag>, uint8_t, InT>;
    const MemT* src = static_cast<const MemT*>(input_raw);
    const uint32_t ndim = plan.shape.ndim;
    const uint64_t last_len = plan.shape.dims[ndim - 1];
    const uint64_t rows = plan.shape.elements / last_len;
    const size_t num_taps = plan.taps.taps.size();
    const MorphTap* taps = plan.taps.taps.data();
    const size_t rank = plan.rank;
    const bool is_float_in = std::is_floating_point_v<InT>;

    uint64_t strides[SM_MAX_NDIM] = {};
    strides[ndim - 1] = 1;
    for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; --d) {
        strides[d] = strides[d + 1] * plan.shape.dims[d + 1];
    }

    const double cval_d = static_cast<double>(cast_cval_to_in<InT>(plan.cval));
    const int64_t x_lo = plan.taps.has_interior ? plan.taps.interior_lo[ndim - 1] : static_cast<int64_t>(last_len);
    const int64_t x_hi = plan.taps.has_interior ? plan.taps.interior_hi[ndim - 1] : 0;
    const uint64_t row_work = std::max<uint64_t>(1, last_len * num_taps * 2);
    const uint64_t grain = std::max<uint64_t>(1, (uint64_t{1} << 16) / row_work);

    parallel::parallel_for(rows, grain, [&](unsigned, uint64_t r0, uint64_t r1) {
        std::vector<double> buf(num_taps);
        int64_t coord[SM_MAX_NDIM] = {};

        auto select_buf = [&](bool has_nan) -> double {
            double sel;
            if (has_nan) {
                sel = scipy_ni_select(buf.data(), 0, static_cast<int64_t>(num_taps) - 1, static_cast<int64_t>(rank));
            } else if (num_taps == 9 && rank == 4) {
                sel = median9_finite(buf.data());
            } else {
                std::nth_element(buf.begin(), buf.begin() + static_cast<ptrdiff_t>(rank), buf.end());
                sel = buf[rank];
            }
            // SciPy CASE_RANK_POINT casts selected double back to input's _type before CASE_FILTER_OUT[_SAFE].
            return static_cast<double>(cast_cval_to_in<InT>(sel));
        };

        for (uint64_t row = r0; row < r1; ++row) {
            uint64_t rem = row;
            for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; --d) {
                coord[d] = static_cast<int64_t>(rem % plan.shape.dims[d]);
                rem /= plan.shape.dims[d];
            }
            bool outer_interior = plan.taps.has_interior;
            if (outer_interior) {
                for (uint32_t d = 0; d + 1 < ndim; ++d) {
                    if (coord[d] < plan.taps.interior_lo[d] || coord[d] >= plan.taps.interior_hi[d]) {
                        outer_interior = false;
                        break;
                    }
                }
            }
            const uint64_t row_base = row * last_len;
            const auto eval_boundary = [&](int64_t x) -> double {
                coord[ndim - 1] = x;
                bool has_nan = false;
                for (size_t t = 0; t < num_taps; ++t) {
                    uint64_t idx = 0;
                    bool in_bounds = true;
                    for (uint32_t d = 0; d < ndim; ++d) {
                        const int64_t pos = coord[d] + taps[t].coord[d];
                        const int64_t dim = static_cast<int64_t>(plan.shape.dims[d]);
                        const int64_t q = (pos >= 0 && pos < dim) ? pos : extend(pos, dim, plan.mode);
                        if (q < 0) {
                            in_bounds = false;
                            break;
                        }
                        idx += static_cast<uint64_t>(q) * strides[d];
                    }
                    const double v = in_bounds ? static_cast<double>(src[idx]) : cval_d;
                    if (is_float_in && std::isnan(v)) has_nan = true;
                    buf[t] = v;
                }
                return select_buf(has_nan);
            };

            const int64_t len = static_cast<int64_t>(last_len);
            if (!outer_interior || x_lo >= x_hi) {
                for (int64_t x = 0; x < len; ++x) {
                    store_dense_output(plan.out_dtype, output_raw, row_base + static_cast<uint64_t>(x),
                                       eval_boundary(x));
                }
            } else {
                for (int64_t x = 0; x < x_lo; ++x) {
                    store_dense_output(plan.out_dtype, output_raw, row_base + static_cast<uint64_t>(x),
                                       eval_boundary(x));
                }
                for (int64_t x = x_lo; x < x_hi; ++x) {
                    const int64_t base = static_cast<int64_t>(row_base) + x;
                    bool has_nan = false;
                    for (size_t t = 0; t < num_taps; ++t) {
                        const double v = static_cast<double>(src[base + taps[t].linear_offset]);
                        if (is_float_in && std::isnan(v)) has_nan = true;
                        buf[t] = v;
                    }
                    store_dense_output(plan.out_dtype, output_raw, static_cast<uint64_t>(base), select_buf(has_nan));
                }
                for (int64_t x = x_hi; x < len; ++x) {
                    store_dense_output(plan.out_dtype, output_raw, row_base + static_cast<uint64_t>(x),
                                       eval_boundary(x));
                }
            }
        }
    });
}

} // namespace

BufferPtr run_cpu_dense_minmax(sm_context* context, const DenseMinMaxPlan& plan, const void* input) {
    uint64_t bytes = 0;
    if (sm_checked_mul(plan.shape.elements, array::dtype_size(plan.out_dtype), &bytes) != SM_OK)
        throw Error(SM_EOVERFLOW, "output array byte size overflows");
    BufferPtr result = array::allocate(context, bytes);
    if (!plan.shape.elements) return result;
    void* out_ptr = mutable_data(result.get());
    switch (plan.in_dtype) {
    case SM_FLOAT64: dispatch_dense_minmax_in<double>(plan, input, out_ptr); break;
    case SM_FLOAT32: dispatch_dense_minmax_in<float>(plan, input, out_ptr); break;
    case SM_BOOL: dispatch_dense_minmax_in<BoolTag>(plan, input, out_ptr); break;
    case SM_UINT8: dispatch_dense_minmax_in<uint8_t>(plan, input, out_ptr); break;
    case SM_UINT16: dispatch_dense_minmax_in<uint16_t>(plan, input, out_ptr); break;
    case SM_UINT32: dispatch_dense_minmax_in<uint32_t>(plan, input, out_ptr); break;
    case SM_UINT64: dispatch_dense_minmax_in<uint64_t>(plan, input, out_ptr); break;
    case SM_INT8: dispatch_dense_minmax_in<int8_t>(plan, input, out_ptr); break;
    case SM_INT16: dispatch_dense_minmax_in<int16_t>(plan, input, out_ptr); break;
    case SM_INT32: dispatch_dense_minmax_in<int32_t>(plan, input, out_ptr); break;
    case SM_INT64: dispatch_dense_minmax_in<int64_t>(plan, input, out_ptr); break;
    default: throw Error(SM_EINVAL, "unsupported input dtype");
    }
    return result;
}

BufferPtr run_cpu_rank(sm_context* context, const RankPlan& plan, const void* input) {
    uint64_t bytes = 0;
    if (sm_checked_mul(plan.shape.elements, array::dtype_size(plan.out_dtype), &bytes) != SM_OK)
        throw Error(SM_EOVERFLOW, "output array byte size overflows");
    BufferPtr result = array::allocate(context, bytes);
    if (!plan.shape.elements) return result;
    void* out_ptr = mutable_data(result.get());
    switch (plan.in_dtype) {
    case SM_FLOAT64: rank_typed<double>(plan, input, out_ptr); break;
    case SM_FLOAT32: rank_typed<float>(plan, input, out_ptr); break;
    case SM_BOOL: rank_typed<BoolTag>(plan, input, out_ptr); break;
    case SM_UINT8: rank_typed<uint8_t>(plan, input, out_ptr); break;
    case SM_UINT16: rank_typed<uint16_t>(plan, input, out_ptr); break;
    case SM_UINT32: rank_typed<uint32_t>(plan, input, out_ptr); break;
    case SM_UINT64: rank_typed<uint64_t>(plan, input, out_ptr); break;
    case SM_INT8: rank_typed<int8_t>(plan, input, out_ptr); break;
    case SM_INT16: rank_typed<int16_t>(plan, input, out_ptr); break;
    case SM_INT32: rank_typed<int32_t>(plan, input, out_ptr); break;
    case SM_INT64: rank_typed<int64_t>(plan, input, out_ptr); break;
    default: throw Error(SM_EINVAL, "unsupported input dtype");
    }
    return result;
}

} // namespace sm::detail::ndimage
