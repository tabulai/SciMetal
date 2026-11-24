// CPU engines for dense N-D min/max morphology and rank/median/percentile filters.
// Threaded across Apple Silicon P-cores; bit-identical to SciPy 1.18.1 arm64.
#include "morphology.hpp"
#include "filters.hpp"
#include "sm/detail/parallel.hpp"

#include <algorithm>
#include <bit>
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
    // Partition pixels, allowing several lanes to share a long row on a thin image.
    const uint64_t grain = std::max<uint64_t>(1, (uint64_t{1} << 14) / num_taps);

    parallel::parallel_for(plan.shape.elements, grain, [&](unsigned, uint64_t begin, uint64_t end) {
        int64_t coord[SM_MAX_NDIM] = {};
        for (uint64_t row = begin / last_len; row <= (end - 1) / last_len; ++row) {
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

            int64_t x = static_cast<int64_t>(std::max(begin, row_base) - row_base);
            const int64_t end_x = static_cast<int64_t>(std::min(last_len, end - row_base));
            const int64_t head_end = outer_interior ? std::min(end_x, x_lo) : end_x;
            for (; x < head_end; ++x) {
                store_dense_output(plan.out_dtype, output_raw, row_base + static_cast<uint64_t>(x), eval_boundary(x));
            }
            const int64_t off0 = taps[0].linear_offset;
            for (; x < std::min(end_x, x_hi); ++x) {
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
            for (; x < end_x; ++x) {
                store_dense_output(plan.out_dtype, output_raw, row_base + static_cast<uint64_t>(x), eval_boundary(x));
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

// Each chunk sorts its samples (including the halo) once. A Fenwick tree
// counts the samples currently in the window by sorted position. Sliding
// changes two counts, and a prefix-count search locates any requested rank.
template<class T>
bool try_sliding_rank(const RankPlan& plan, const T* src, T* dst, T cval) {
    if (!use_sliding_rank(plan)) return false;
    const int64_t length = static_cast<int64_t>(plan.shape.elements);
    if constexpr (std::is_floating_point_v<T>) {
        // NaNs require SciPy's comparison order. Preserve its zero tie
        // handling too when equal values can have different representations.
        const auto special = [](T v) { return std::isnan(v) || (v == 0 && std::signbit(v)); };
        if ((plan.mode == SM_NDIMAGE_CONSTANT && special(cval)) || std::any_of(src, src + length, special))
            return false;
    }

    struct Sample {
        double value;
        int position;
    };
    const int width = static_cast<int>(plan.fshape[0]);
    const int chunk_size = std::max(1024, width);
    const int64_t chunks = (length - 1) / chunk_size + 1;
    parallel::parallel_for(chunks, 1, [&](unsigned, uint64_t first, uint64_t last) {
        std::vector<Sample> samples(chunk_size + width - 1);
        std::vector<int> index(samples.size()), counts(samples.size() + 1);
        for (uint64_t chunk = first; chunk < last; ++chunk) {
            const int64_t start = static_cast<int64_t>(chunk) * chunk_size;
            const int outputs = static_cast<int>(std::min<int64_t>(chunk_size, length - start));
            const int size = outputs + width - 1;
            for (int i = 0; i < size; ++i) {
                int64_t pos = start + plan.start[0] + i;
                if (pos < 0 || pos >= length) pos = extend(pos, length, plan.mode);
                samples[i] = {static_cast<double>(pos < 0 ? cval : src[pos]), i};
            }
            std::sort(samples.begin(), samples.begin() + size,
                      [](const Sample& a, const Sample& b) { return a.value < b.value; });
            for (int i = 0; i < size; ++i) index[samples[i].position] = i + 1;

            std::fill_n(counts.begin(), size + 1, 0);
            const auto update = [&](int position, int delta) {
                for (int j = index[position]; j <= size; j += j & -j) counts[j] += delta;
            };
            for (int i = 0; i < width; ++i) update(i, 1);
            const int top_bit = static_cast<int>(std::bit_floor(static_cast<unsigned>(size)));
            for (int i = 0; i < outputs; ++i) {
                int below = 0, remaining = static_cast<int>(plan.rank);
                for (int bit = top_bit; bit; bit >>= 1) {
                    const int next = below + bit;
                    if (next <= size && counts[next] <= remaining) {
                        remaining -= counts[next];
                        below = next;
                    }
                }
                dst[start + i] = static_cast<T>(samples[below].value);
                if (i + 1 < outputs) {
                    update(i, -1);
                    update(i + width, 1);
                }
            }
        }
    });
    return true;
}

template<class InT>
void rank_typed(const RankPlan& plan, const void* input_raw, void* output_raw) {
    using MemT = std::conditional_t<std::is_same_v<InT, BoolTag>, uint8_t, InT>;
    const MemT* src = static_cast<const MemT*>(input_raw);
    if constexpr (!std::is_same_v<InT, BoolTag> && !std::is_same_v<InT, int64_t> && !std::is_same_v<InT, uint64_t>) {
        if (try_sliding_rank(plan, src, static_cast<InT*>(output_raw), cast_cval_to_in<InT>(plan.cval))) return;
    }
    const uint32_t ndim = plan.shape.ndim;
    const uint64_t last_len = plan.shape.dims[ndim - 1];
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
    const uint64_t grain = std::max<uint64_t>(1, (uint64_t{1} << 13) / num_taps);

    parallel::parallel_for(plan.shape.elements, grain, [&](unsigned, uint64_t begin, uint64_t end) {
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

        for (uint64_t row = begin / last_len; row <= (end - 1) / last_len; ++row) {
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

            int64_t x = static_cast<int64_t>(std::max(begin, row_base) - row_base);
            const int64_t end_x = static_cast<int64_t>(std::min(last_len, end - row_base));
            const int64_t head_end = outer_interior ? std::min(end_x, x_lo) : end_x;
            for (; x < head_end; ++x) {
                store_dense_output(plan.out_dtype, output_raw, row_base + static_cast<uint64_t>(x), eval_boundary(x));
            }
            for (; x < std::min(end_x, x_hi); ++x) {
                const int64_t base = static_cast<int64_t>(row_base) + x;
                bool has_nan = false;
                for (size_t t = 0; t < num_taps; ++t) {
                    const double v = static_cast<double>(src[base + taps[t].linear_offset]);
                    if (is_float_in && std::isnan(v)) has_nan = true;
                    buf[t] = v;
                }
                store_dense_output(plan.out_dtype, output_raw, static_cast<uint64_t>(base), select_buf(has_nan));
            }
            for (; x < end_x; ++x) {
                store_dense_output(plan.out_dtype, output_raw, row_base + static_cast<uint64_t>(x), eval_boundary(x));
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
