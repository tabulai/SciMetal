// Shared dense N-D correlation primitive: tap builder and threaded CPU engine.
#include "dense_correlate.hpp"
#include "ms/detail/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <type_traits>

// Preserve exact FMA contraction matching SciPy's _NI_Correlate on arm64.
#pragma STDC FP_CONTRACT OFF

namespace ms::detail::dense_correlate {
namespace {

constexpr uint64_t min_work_per_lane = uint64_t{1} << 17;
constexpr uint64_t max_col_chunk = 2048;
constexpr uint64_t min_col_chunk = 256;
constexpr double float32_overflow_threshold = 0x1.ffffffp127;

bool overflows_float32(double v) noexcept {
    return std::isfinite(v) && std::fabs(v) >= float32_overflow_threshold;
}

unsigned lanes_for(uint64_t elements, uint64_t taps) {
    uint64_t work = 0;
    if (ms_checked_mul(elements, std::max<uint64_t>(taps, 1), &work) != MS_OK) work = UINT64_MAX;
    return static_cast<unsigned>(std::clamp<uint64_t>(work / min_work_per_lane, 1, parallel::hardware_lanes()));
}

template<class Make, class Function>
void for_each_item(uint64_t items, uint64_t grain, unsigned lanes, Make&& make, Function&& fn) {
    if (!items) return;
    lanes = static_cast<unsigned>(std::min<uint64_t>(lanes, (items + grain - 1) / grain));
    std::atomic<uint64_t> next{0};
    parallel::run_lanes(std::max(1u, lanes), [&](unsigned) {
        auto workspace = make();
        for (;;) {
            const uint64_t begin = next.fetch_add(grain, std::memory_order_relaxed);
            if (begin >= items) break;
            const uint64_t end = std::min(items, begin + grain);
            for (uint64_t item = begin; item < end; ++item) fn(item, workspace);
        }
    });
}

inline int64_t extend(int64_t position, int64_t length, uint32_t mode) noexcept {
    if (position >= 0 && position < length) return position;
    switch (mode) {
    case Nearest:
        return position < 0 ? 0 : length - 1;
    case Wrap: {
        if (position >= -length && position < 0) return position + length;
        if (position >= length && position < 2 * length) return position - length;
        const int64_t m = position % length;
        return m < 0 ? m + length : m;
    }
    case Reflect: {
        if (position >= -length && position < 0) return -1 - position;
        if (position >= length && position < 2 * length) return 2 * length - 1 - position;
        const int64_t period = 2 * length;
        int64_t m = position % period;
        if (m < 0) m += period;
        return m < length ? m : period - 1 - m;
    }
    case Mirror: {
        if (length == 1) return 0;
        if (position > -length && position < 0) return -position;
        if (position >= length && position <= 2 * length - 2) return 2 * length - 2 - position;
        const int64_t period = 2 * length - 2;
        int64_t m = position % period;
        if (m < 0) m += period;
        return m < length ? m : period - m;
    }
    default:
        return -1;
    }
}

inline double fma_step(double a, double b, double c) noexcept { return __builtin_fma(a, b, c); }
inline float fma_step(float a, float b, float c) noexcept { return __builtin_fmaf(a, b, c); }

template<class AccumT, class SrcT>
void sweep_interior_real(AccumT* __restrict acc, const SrcT* const* ptrs, const AccumT* w,
                         size_t num_taps, size_t n, AccumT cval, bool omit_oob) {
    size_t k = 0;
    for (; k + 4 <= num_taps; k += 4) {
        const SrcT *x0 = ptrs[k], *x1 = ptrs[k + 1], *x2 = ptrs[k + 2], *x3 = ptrs[k + 3];
        const AccumT w0 = w[k], w1 = w[k + 1], w2 = w[k + 2], w3 = w[k + 3];
        if (x0 && x1 && x2 && x3) {
            for (size_t c = 0; c < n; ++c) {
                AccumT s = acc[c];
                s = fma_step(static_cast<AccumT>(x0[c]), w0, s);
                s = fma_step(static_cast<AccumT>(x1[c]), w1, s);
                s = fma_step(static_cast<AccumT>(x2[c]), w2, s);
                s = fma_step(static_cast<AccumT>(x3[c]), w3, s);
                acc[c] = s;
            }
        } else {
            for (size_t t = 0; t < 4; ++t) {
                const SrcT* xt = ptrs[k + t];
                const AccumT wt = w[k + t];
                if (xt) {
                    for (size_t c = 0; c < n; ++c) acc[c] = fma_step(static_cast<AccumT>(xt[c]), wt, acc[c]);
                } else if (!omit_oob) {
                    for (size_t c = 0; c < n; ++c) acc[c] = fma_step(cval, wt, acc[c]);
                }
            }
        }
    }
    for (; k < num_taps; ++k) {
        const SrcT* xk = ptrs[k];
        const AccumT wk = w[k];
        if (xk) {
            for (size_t c = 0; c < n; ++c) acc[c] = fma_step(static_cast<AccumT>(xk[c]), wk, acc[c]);
        } else if (!omit_oob) {
            for (size_t c = 0; c < n; ++c) acc[c] = fma_step(cval, wk, acc[c]);
        }
    }
}

template<class SrcT, class DstT, class AccumT>
void run_real_impl(const StencilSpec& spec, const TapTable& taps, const void* input_ptr, void* output_ptr) {
    const SrcT* src = static_cast<const SrcT*>(input_ptr);
    DstT* dst = static_cast<DstT*>(output_ptr);

    const uint32_t ndim = spec.out_shape.ndim;
    const uint64_t inner_u64 = spec.out_shape.dims[ndim - 1];
    const int64_t in_inner = static_cast<int64_t>(spec.in_shape.dims[ndim - 1]);
    const uint64_t outer_rows = spec.out_shape.elements / inner_u64;
    const size_t num_taps = taps.taps.size();

    int64_t in_strides[MS_MAX_NDIM] = {};
    if (ndim > 0) {
        in_strides[ndim - 1] = 1;
        for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; --d) {
            in_strides[d] = in_strides[d + 1] * static_cast<int64_t>(spec.in_shape.dims[d + 1]);
        }
    }

    std::vector<AccumT> weights(num_taps);
    for (size_t k = 0; k < num_taps; ++k) {
        if constexpr (std::is_same_v<AccumT, float>) {
            weights[k] = taps.taps[k].weight_re_f32;
        } else {
            weights[k] = taps.taps[k].weight_re_f64;
        }
    }
    const AccumT cval = static_cast<AccumT>(spec.cval_real);
    const int64_t x_lo = taps.interior_lo[ndim - 1];
    const int64_t x_hi = taps.interior_hi[ndim - 1];
    const uint32_t inner_mode = spec.mode[ndim - 1];

    const unsigned lanes = lanes_for(spec.out_shape.elements, num_taps);
    uint64_t col_chunk = max_col_chunk;
    if (lanes > 1 && outer_rows < uint64_t{4} * lanes) {
        const uint64_t target_chunks = (uint64_t{4} * lanes + outer_rows - 1) / outer_rows;
        col_chunk = std::clamp<uint64_t>(inner_u64 / target_chunks, min_col_chunk, max_col_chunk);
    }
    const uint64_t chunks = (inner_u64 + col_chunk - 1) / col_chunk;
    const uint64_t total_items = outer_rows * chunks;
    const uint64_t grain = outer_rows >= uint64_t{4} * lanes ? std::max<uint64_t>(1, 16384 / std::max<uint64_t>(inner_u64, 1)) : 1;

    struct Workspace {
        std::vector<int64_t> row_bases;
        std::vector<int64_t> inner_offsets;
        std::vector<uint8_t> outer_ok;
        std::vector<const SrcT*> ptrs;
        std::vector<AccumT> acc;
    };

    const auto make = [&] {
        Workspace ws;
        ws.row_bases.resize(num_taps);
        ws.inner_offsets.resize(num_taps);
        ws.outer_ok.resize(num_taps);
        ws.ptrs.resize(num_taps);
        if constexpr (!std::is_same_v<DstT, AccumT>) {
            ws.acc.resize(static_cast<size_t>(std::min<uint64_t>(col_chunk, inner_u64)));
        }
        return ws;
    };

    for_each_item(total_items, grain, lanes, make, [&](uint64_t item, Workspace& ws) {
        const uint64_t r = item / chunks;
        const uint64_t chunk_idx = item % chunks;
        const int64_t c_begin = static_cast<int64_t>(chunk_idx * col_chunk);
        const int64_t c_end = static_cast<int64_t>(std::min<uint64_t>(inner_u64, static_cast<uint64_t>(c_begin) + col_chunk));

        int64_t p_outer[MS_MAX_NDIM] = {};
        if (ndim > 1) {
            uint64_t rem = r;
            for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; --d) {
                const uint64_t extent = spec.out_shape.dims[d];
                p_outer[d] = static_cast<int64_t>(rem % extent);
                rem /= extent;
            }
        }

        for (size_t k = 0; k < num_taps; ++k) {
            const Tap& tap = taps.taps[k];
            int64_t off = 0;
            bool ok = true;
            for (uint32_t d = 0; d + 1 < ndim; ++d) {
                const int64_t q = extend(p_outer[d] + tap.coord[d],
                                         static_cast<int64_t>(spec.in_shape.dims[d]), spec.mode[d]);
                if (q < 0) {
                    ok = false;
                    break;
                }
                off += q * in_strides[d];
            }
            ws.outer_ok[k] = ok ? 1 : 0;
            ws.row_bases[k] = off;
            ws.inner_offsets[k] = off + tap.coord[ndim - 1];
        }

        DstT* __restrict dst_row = dst + r * inner_u64;

        const int64_t left_end = std::min<int64_t>(c_end, x_lo);
        const int64_t mid_begin = std::max<int64_t>(c_begin, x_lo);
        const int64_t mid_end = std::max<int64_t>(mid_begin, std::min<int64_t>(c_end, x_hi));
        const int64_t right_begin = std::max<int64_t>(c_begin, mid_end);

        auto eval_boundary = [&](int64_t begin, int64_t end) {
            for (int64_t c = begin; c < end; ++c) {
                AccumT acc = AccumT{0};
                for (size_t k = 0; k < num_taps; ++k) {
                    if (!ws.outer_ok[k]) {
                        if (!spec.omit_oob) acc = fma_step(cval, weights[k], acc);
                        continue;
                    }
                    const int64_t q = extend(c + taps.taps[k].coord[ndim - 1], in_inner, inner_mode);
                    if (q < 0) {
                        if (!spec.omit_oob) acc = fma_step(cval, weights[k], acc);
                    } else {
                        acc = fma_step(static_cast<AccumT>(src[ws.row_bases[k] + q]), weights[k], acc);
                    }
                }
                dst_row[c] = static_cast<DstT>(acc);
            }
        };

        eval_boundary(c_begin, left_end);

        if (mid_begin < mid_end) {
            const size_t n = static_cast<size_t>(mid_end - mid_begin);
            for (size_t k = 0; k < num_taps; ++k) {
                ws.ptrs[k] = ws.outer_ok[k] ? (src + ws.inner_offsets[k] + mid_begin) : nullptr;
            }
            if constexpr (std::is_same_v<DstT, AccumT>) {
                AccumT* __restrict acc_ptr = dst_row + mid_begin;
                std::fill_n(acc_ptr, n, AccumT{0});
                sweep_interior_real(acc_ptr, ws.ptrs.data(), weights.data(), num_taps, n, cval, spec.omit_oob);
            } else {
                AccumT* __restrict acc_ptr = ws.acc.data();
                std::fill_n(acc_ptr, n, AccumT{0});
                sweep_interior_real(acc_ptr, ws.ptrs.data(), weights.data(), num_taps, n, cval, spec.omit_oob);
                for (size_t c = 0; c < n; ++c) {
                    dst_row[mid_begin + static_cast<int64_t>(c)] = static_cast<DstT>(acc_ptr[c]);
                }
            }
        }

        eval_boundary(right_begin, c_end);
    });
}

template<bool SrcComplex, class SrcScalar, class DstScalar, class AccumScalar>
void run_complex_impl(const StencilSpec& spec, const TapTable& taps, const void* input_ptr, void* output_ptr) {
    const SrcScalar* src = static_cast<const SrcScalar*>(input_ptr);
    DstScalar* dst = static_cast<DstScalar*>(output_ptr);
    constexpr size_t src_step = SrcComplex ? 2 : 1;

    const uint32_t ndim = spec.out_shape.ndim;
    const uint64_t inner_u64 = spec.out_shape.dims[ndim - 1];
    const int64_t in_inner = static_cast<int64_t>(spec.in_shape.dims[ndim - 1]);
    const uint64_t outer_rows = spec.out_shape.elements / inner_u64;
    const size_t num_taps = taps.taps.size();

    int64_t in_strides[MS_MAX_NDIM] = {};
    if (ndim > 0) {
        in_strides[ndim - 1] = 1;
        for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; --d) {
            in_strides[d] = in_strides[d + 1] * static_cast<int64_t>(spec.in_shape.dims[d + 1]);
        }
    }

    std::vector<AccumScalar> w_re(num_taps), w_im(num_taps);
    for (size_t k = 0; k < num_taps; ++k) {
        if constexpr (std::is_same_v<AccumScalar, float>) {
            w_re[k] = taps.taps[k].weight_re_f32;
            w_im[k] = taps.taps[k].weight_im_f32;
        } else {
            w_re[k] = taps.taps[k].weight_re_f64;
            w_im[k] = taps.taps[k].weight_im_f64;
        }
    }
    const AccumScalar cval_re = static_cast<AccumScalar>(spec.cval_real);
    const AccumScalar cval_im = SrcComplex ? static_cast<AccumScalar>(spec.cval_imag) : AccumScalar{0};
    const int64_t x_lo = taps.interior_lo[ndim - 1];
    const int64_t x_hi = taps.interior_hi[ndim - 1];
    const uint32_t inner_mode = spec.mode[ndim - 1];

    const unsigned lanes = lanes_for(spec.out_shape.elements, num_taps);
    uint64_t col_chunk = max_col_chunk;
    if (lanes > 1 && outer_rows < uint64_t{4} * lanes) {
        const uint64_t target_chunks = (uint64_t{4} * lanes + outer_rows - 1) / outer_rows;
        col_chunk = std::clamp<uint64_t>(inner_u64 / target_chunks, min_col_chunk, max_col_chunk);
    }
    const uint64_t chunks = (inner_u64 + col_chunk - 1) / col_chunk;
    const uint64_t total_items = outer_rows * chunks;
    const uint64_t grain = outer_rows >= uint64_t{4} * lanes ? std::max<uint64_t>(1, 16384 / std::max<uint64_t>(inner_u64, 1)) : 1;

    struct Workspace {
        std::vector<int64_t> row_bases;
        std::vector<int64_t> inner_offsets;
        std::vector<uint8_t> outer_ok;
        std::vector<const SrcScalar*> ptrs;
        std::vector<AccumScalar> acc_re;
        std::vector<AccumScalar> acc_im;
    };

    const auto make = [&] {
        Workspace ws;
        ws.row_bases.resize(num_taps);
        ws.inner_offsets.resize(num_taps);
        ws.outer_ok.resize(num_taps);
        ws.ptrs.resize(num_taps);
        const size_t cap = static_cast<size_t>(std::min<uint64_t>(col_chunk, inner_u64));
        ws.acc_re.resize(cap);
        ws.acc_im.resize(cap);
        return ws;
    };

    for_each_item(total_items, grain, lanes, make, [&](uint64_t item, Workspace& ws) {
        const uint64_t r = item / chunks;
        const uint64_t chunk_idx = item % chunks;
        const int64_t c_begin = static_cast<int64_t>(chunk_idx * col_chunk);
        const int64_t c_end = static_cast<int64_t>(std::min<uint64_t>(inner_u64, static_cast<uint64_t>(c_begin) + col_chunk));

        int64_t p_outer[MS_MAX_NDIM] = {};
        if (ndim > 1) {
            uint64_t rem = r;
            for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; --d) {
                const uint64_t extent = spec.out_shape.dims[d];
                p_outer[d] = static_cast<int64_t>(rem % extent);
                rem /= extent;
            }
        }

        for (size_t k = 0; k < num_taps; ++k) {
            const Tap& tap = taps.taps[k];
            int64_t off = 0;
            bool ok = true;
            for (uint32_t d = 0; d + 1 < ndim; ++d) {
                const int64_t q = extend(p_outer[d] + tap.coord[d],
                                         static_cast<int64_t>(spec.in_shape.dims[d]), spec.mode[d]);
                if (q < 0) {
                    ok = false;
                    break;
                }
                off += q * in_strides[d];
            }
            ws.outer_ok[k] = ok ? 1 : 0;
            ws.row_bases[k] = off;
            ws.inner_offsets[k] = off + tap.coord[ndim - 1];
        }

        DstScalar* __restrict dst_row = dst + r * inner_u64 * 2;
        const int64_t left_end = std::min<int64_t>(c_end, x_lo);
        const int64_t mid_begin = std::max<int64_t>(c_begin, x_lo);
        const int64_t mid_end = std::max<int64_t>(mid_begin, std::min<int64_t>(c_end, x_hi));
        const int64_t right_begin = std::max<int64_t>(c_begin, mid_end);

        auto eval_boundary = [&](int64_t begin, int64_t end) {
            for (int64_t c = begin; c < end; ++c) {
                AccumScalar re = AccumScalar{0}, im = AccumScalar{0};
                for (size_t k = 0; k < num_taps; ++k) {
                    AccumScalar vr, vi;
                    if (!ws.outer_ok[k]) {
                        if (spec.omit_oob) continue;
                        vr = cval_re;
                        vi = cval_im;
                    } else {
                        const int64_t q = extend(c + taps.taps[k].coord[ndim - 1], in_inner, inner_mode);
                        if (q < 0) {
                            if (spec.omit_oob) continue;
                            vr = cval_re;
                            vi = cval_im;
                        } else {
                            const size_t idx = static_cast<size_t>(ws.row_bases[k] + q) * src_step;
                            vr = static_cast<AccumScalar>(src[idx]);
                            vi = SrcComplex ? static_cast<AccumScalar>(src[idx + 1]) : AccumScalar{0};
                        }
                    }
                    const AccumScalar wr = w_re[k];
                    const AccumScalar wi = w_im[k];
                    re += vr * wr - vi * wi;
                    im += vr * wi + vi * wr;
                }
                dst_row[c * 2] = static_cast<DstScalar>(re);
                dst_row[c * 2 + 1] = static_cast<DstScalar>(im);
            }
        };

        eval_boundary(c_begin, left_end);

        if (mid_begin < mid_end) {
            const size_t n = static_cast<size_t>(mid_end - mid_begin);
            for (size_t k = 0; k < num_taps; ++k) {
                ws.ptrs[k] = ws.outer_ok[k] ? (src + (ws.inner_offsets[k] + mid_begin) * static_cast<int64_t>(src_step))
                                            : nullptr;
            }
            AccumScalar* __restrict are = ws.acc_re.data();
            AccumScalar* __restrict aim = ws.acc_im.data();
            std::fill_n(are, n, AccumScalar{0});
            std::fill_n(aim, n, AccumScalar{0});
            for (size_t k = 0; k < num_taps; ++k) {
                const SrcScalar* xk = ws.ptrs[k];
                const AccumScalar wr = w_re[k];
                const AccumScalar wi = w_im[k];
                if (xk) {
                    for (size_t c = 0; c < n; ++c) {
                        const AccumScalar vr = static_cast<AccumScalar>(xk[c * src_step]);
                        const AccumScalar vi = SrcComplex ? static_cast<AccumScalar>(xk[c * src_step + 1]) : AccumScalar{0};
                        are[c] += vr * wr - vi * wi;
                        aim[c] += vr * wi + vi * wr;
                    }
                } else if (!spec.omit_oob) {
                    for (size_t c = 0; c < n; ++c) {
                        are[c] += cval_re * wr - cval_im * wi;
                        aim[c] += cval_re * wi + cval_im * wr;
                    }
                }
            }
            for (size_t c = 0; c < n; ++c) {
                dst_row[(mid_begin + static_cast<int64_t>(c)) * 2] = static_cast<DstScalar>(are[c]);
                dst_row[(mid_begin + static_cast<int64_t>(c)) * 2 + 1] = static_cast<DstScalar>(aim[c]);
            }
        }

        eval_boundary(right_begin, c_end);
    });
}

} // namespace

TapTable build_taps(const StencilSpec& spec) {
    const uint32_t ndim = spec.in_shape.ndim;
    if (ndim < 1 || ndim > MS_MAX_NDIM || spec.out_shape.ndim != ndim)
        throw Error(MS_EINVAL, "dense_correlate: invalid rank");
    if (!spec.weights) throw Error(MS_EINVAL, "dense_correlate: null weights");
    (void)array::dtype_size(spec.weights_dtype);

    uint64_t w_total = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
        if (spec.wshape[d] < 1) throw Error(MS_EINVAL, "dense_correlate: weight extent must be >= 1");
        if (ms_checked_mul(w_total, spec.wshape[d], &w_total) != MS_OK)
            throw Error(MS_EOVERFLOW, "dense_correlate: weight count overflows");
    }

    int64_t in_strides[MS_MAX_NDIM] = {};
    in_strides[ndim - 1] = 1;
    for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; --d) {
        in_strides[d] = in_strides[d + 1] * static_cast<int64_t>(spec.in_shape.dims[d + 1]);
    }

    const bool cw = array::is_complex(spec.weights_dtype);
    const bool w_double = array::is_double(spec.weights_dtype);

    TapTable table;
    table.is_complex_weights = cw;
    table.taps.reserve(static_cast<size_t>(std::min<uint64_t>(w_total, 4096)));

    uint32_t w_coord[MS_MAX_NDIM] = {};
    int64_t min_c[MS_MAX_NDIM] = {};
    int64_t max_c[MS_MAX_NDIM] = {};
    bool first_kept = true;

    for (uint64_t idx = 0; idx < w_total; ++idx) {
        double wre = 0.0, wim = 0.0;
        if (cw) {
            if (w_double) {
                const double* p = static_cast<const double*>(spec.weights) + idx * 2;
                wre = p[0];
                wim = p[1];
            } else {
                const float* p = static_cast<const float*>(spec.weights) + idx * 2;
                wre = static_cast<double>(p[0]);
                wim = static_cast<double>(p[1]);
            }
        } else {
            if (w_double) {
                wre = static_cast<const double*>(spec.weights)[idx];
            } else {
                wre = static_cast<double>(static_cast<const float*>(spec.weights)[idx]);
            }
        }

        const bool keep = !spec.prune_zeros ||
                          (cw ? (std::fabs(wre) > DBL_EPSILON || std::fabs(wim) > DBL_EPSILON)
                              : (std::fabs(wre) > DBL_EPSILON));
        if (keep) {
            Tap tap;
            tap.weight_re_f64 = wre;
            tap.weight_im_f64 = wim;
            tap.weight_re_f32 = static_cast<float>(wre);
            tap.weight_im_f32 = static_cast<float>(wim);
            if (overflows_float32(wre) || (cw && overflows_float32(wim))) {
                table.weights_overflow_f32 = true;
            }
            int64_t lin = 0;
            for (uint32_t d = 0; d < ndim; ++d) {
                const int64_t c = static_cast<int64_t>(w_coord[d]) + spec.start[d];
                tap.coord[d] = c;
                lin += c * in_strides[d];
                if (first_kept) {
                    min_c[d] = max_c[d] = c;
                } else {
                    if (c < min_c[d]) min_c[d] = c;
                    if (c > max_c[d]) max_c[d] = c;
                }
            }
            tap.linear_offset = lin;
            first_kept = false;
            table.taps.push_back(tap);
        }

        for (int32_t d = static_cast<int32_t>(ndim) - 1; d >= 0; --d) {
            if (++w_coord[d] < spec.wshape[d]) break;
            w_coord[d] = 0;
        }
    }

    if (table.taps.empty() || spec.out_shape.elements == 0) {
        table.has_interior = spec.out_shape.elements > 0;
        table.all_interior = spec.out_shape.elements > 0;
        for (uint32_t d = 0; d < ndim; ++d) {
            table.interior_lo[d] = 0;
            table.interior_hi[d] = static_cast<int64_t>(spec.out_shape.dims[d]);
        }
        return table;
    }

    table.has_interior = true;
    table.all_interior = true;
    for (uint32_t d = 0; d < ndim; ++d) {
        const int64_t out_d = static_cast<int64_t>(spec.out_shape.dims[d]);
        const int64_t in_d = static_cast<int64_t>(spec.in_shape.dims[d]);
        const int64_t lo = std::clamp<int64_t>(-min_c[d], 0, out_d);
        const int64_t hi = std::clamp<int64_t>(in_d - max_c[d], lo, out_d);
        table.interior_lo[d] = lo;
        table.interior_hi[d] = hi;
        if (lo >= hi) table.has_interior = false;
        if (lo != 0 || hi != out_d) table.all_interior = false;
    }
    return table;
}

BufferPtr run_cpu(ms_context* ctx, const StencilSpec& spec, const TapTable& taps, const void* input) {
    const uint64_t out_bytes = spec.out_shape.elements * array::dtype_size(spec.out_dtype);
    BufferPtr result = array::allocate(ctx, out_bytes);
    if (spec.out_shape.elements == 0) return result;
    void* out_ptr = mutable_data(result.get());
    if (taps.taps.empty()) {
        std::memset(out_ptr, 0, out_bytes);
        return result;
    }
    if (spec.in_shape.elements == 0) {
        // Only possible if CONSTANT mode everywhere; handle via standard path.
        for (uint32_t d = 0; d < spec.in_shape.ndim; ++d) {
            if (spec.in_shape.dims[d] == 0 && spec.mode[d] != Constant)
                throw Error(MS_EINVAL, "dense_correlate: empty input with non-constant boundary");
        }
    }

    const bool in_c = array::is_complex(spec.in_dtype);
    const bool out_c = array::is_complex(spec.out_dtype);
    if (!in_c && !out_c && !taps.is_complex_weights) {
        const bool use_f32_accum =
            spec.cpu_accum == AccumMode::InputDtype && spec.in_dtype == MS_FLOAT32 && spec.out_dtype == MS_FLOAT32;
        if (use_f32_accum) {
            run_real_impl<float, float, float>(spec, taps, input, out_ptr);
        } else if (spec.in_dtype == MS_FLOAT32 && spec.out_dtype == MS_FLOAT32) {
            run_real_impl<float, float, double>(spec, taps, input, out_ptr);
        } else if (spec.in_dtype == MS_FLOAT32 && spec.out_dtype == MS_FLOAT64) {
            run_real_impl<float, double, double>(spec, taps, input, out_ptr);
        } else if (spec.in_dtype == MS_FLOAT64 && spec.out_dtype == MS_FLOAT32) {
            run_real_impl<double, float, double>(spec, taps, input, out_ptr);
        } else {
            run_real_impl<double, double, double>(spec, taps, input, out_ptr);
        }
        return result;
    }

    if (!out_c) throw Error(MS_EINVAL, "dense_correlate: complex input or weights require complex output");
    const bool use_f32_accum = spec.cpu_accum == AccumMode::InputDtype && !array::is_double(spec.in_dtype) &&
                               !array::is_double(spec.out_dtype);
    const bool in_d = array::is_double(spec.in_dtype);
    const bool out_d = array::is_double(spec.out_dtype);
    if (in_c) {
        if (use_f32_accum) {
            run_complex_impl<true, float, float, float>(spec, taps, input, out_ptr);
        } else if (!in_d && !out_d) {
            run_complex_impl<true, float, float, double>(spec, taps, input, out_ptr);
        } else if (!in_d && out_d) {
            run_complex_impl<true, float, double, double>(spec, taps, input, out_ptr);
        } else if (in_d && !out_d) {
            run_complex_impl<true, double, float, double>(spec, taps, input, out_ptr);
        } else {
            run_complex_impl<true, double, double, double>(spec, taps, input, out_ptr);
        }
    } else {
        if (use_f32_accum) {
            run_complex_impl<false, float, float, float>(spec, taps, input, out_ptr);
        } else if (!in_d && !out_d) {
            run_complex_impl<false, float, float, double>(spec, taps, input, out_ptr);
        } else if (!in_d && out_d) {
            run_complex_impl<false, float, double, double>(spec, taps, input, out_ptr);
        } else if (in_d && !out_d) {
            run_complex_impl<false, double, float, double>(spec, taps, input, out_ptr);
        } else {
            run_complex_impl<false, double, double, double>(spec, taps, input, out_ptr);
        }
    }
    return result;
}

#if !defined(MS_ENABLE_METAL)
bool gpu_supports(const StencilSpec&, const TapTable&) noexcept { return false; }
#endif

} // namespace ms::detail::dense_correlate
