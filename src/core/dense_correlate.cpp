// Shared dense N-D correlation primitive: tap builder and threaded CPU engine.
#include "dense_correlate.hpp"
#include "sm/detail/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <type_traits>

// Preserve exact FMA contraction matching SciPy's _NI_Correlate on arm64.
#pragma STDC FP_CONTRACT OFF

namespace sm::detail::dense_correlate {
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
    if (sm_checked_mul(elements, std::max<uint64_t>(taps, 1), &work) != SM_OK) work = UINT64_MAX;
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

#if defined(__APPLE__) && defined(__aarch64__)
void check_fir_fpcr() {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    if (fpcr != 0) throw Error(SM_ENOTSUP, "strict FIR requires the default arm64 floating-point environment");
}

// The verified SciPy wheel rounds term 1, fuses terms 0, 2, and 3 into
// that product, then adds the group to the accumulator separately.
template<class T>
inline T scipy_fir_four(const T* w, T x0, T x1, T x2, T x3) noexcept {
    T group = w[1] * x1;
    group = fma_step(w[0], x0, group);
    group = fma_step(w[2], x2, group);
    return fma_step(w[3], x3, group);
}

template<class T, bool Correlate>
void run_scipy_fir_rows(const StencilSpec& spec, const T* input, T* output) {
    // SciPy visits the original weights in increasing order for both
    // operations. Only the direction of the corresponding samples changes.
    constexpr int64_t step = Correlate ? 1 : -1;
    const int64_t in_height = static_cast<int64_t>(spec.in_shape.dims[0]);
    const int64_t in_width = static_cast<int64_t>(spec.in_shape.dims[1]);
    const int64_t out_width = static_cast<int64_t>(spec.out_shape.dims[1]);
    const int64_t kernel_height = spec.wshape[0];
    const int64_t kernel_width = spec.wshape[1];
    const int64_t max_col_offset = spec.start[1] + kernel_width - 1;
    const int64_t first_row = spec.start[0] + (Correlate ? 0 : kernel_height - 1);
    const int64_t first_col = Correlate ? spec.start[1] : max_col_offset;
    const int64_t interior_begin = std::clamp(-spec.start[1], int64_t{0}, out_width);
    const int64_t interior_end = std::clamp(in_width - max_col_offset, interior_begin, out_width);
    const T* weights = static_cast<const T*>(spec.weights);
    const T fill = static_cast<T>(spec.cval_real);
    const unsigned lanes = lanes_for(spec.out_shape.elements, uint64_t{spec.wshape[0]} * spec.wshape[1]);
    for_each_item(spec.out_shape.dims[0], 4, lanes, [&] {
        check_fir_fpcr();
        return std::vector<const T*>(kernel_height);
    }, [&](uint64_t out_row, auto& rows) {
        for (int64_t j = 0; j < kernel_height; ++j) {
            const int64_t y = extend(static_cast<int64_t>(out_row) + first_row + step * j, in_height, spec.mode[0]);
            rows[j] = y >= 0 ? input + y * in_width : nullptr;
        }

        T* dst = output + out_row * out_width;
        auto border = [&](int64_t begin, int64_t end) {
            for (int64_t c = begin; c < end; ++c) {
                T sum = 0;
                for (int64_t j = 0; j < kernel_height; ++j) {
                    const T* row = rows[j];
                    const T* w = weights + j * kernel_width;
                    const auto pixel = [&](int64_t k) {
                        const int64_t x = extend(c + first_col + step * k, in_width, spec.mode[1]);
                        return row && x >= 0 ? row[x] : fill;
                    };
                    int64_t k = 0;
                    for (; k + 4 <= kernel_width; k += 4) {
                        sum += scipy_fir_four(w + k, pixel(k), pixel(k + 1), pixel(k + 2), pixel(k + 3));
                    }
                    for (; k < kernel_width; ++k) sum = fma_step(w[k], pixel(k), sum);
                }
                dst[c] = sum;
            }
        };
        border(0, interior_begin);
        border(interior_end, out_width);

        // Tile the contiguous columns so the same outputs stay in cache
        // while sweeping over the taps. Each loop can vectorize across pixels.
        constexpr int64_t tile_width = 1024;
        for (int64_t begin = interior_begin; begin < interior_end; begin += tile_width) {
            const int64_t end = std::min(begin + tile_width, interior_end);
            std::fill(dst + begin, dst + end, T{0});
            for (int64_t j = 0; j < kernel_height; ++j) {
                const T* row = rows[j];
                const T* w = weights + j * kernel_width;
                int64_t k = 0;
                for (; k + 4 <= kernel_width; k += 4) {
                    if (row) {
                        const T* x = row + (first_col + step * k + begin);
                        T* __restrict sums = dst + begin;
                        for (int64_t c = 0; c < end - begin; ++c) {
                            sums[c] += scipy_fir_four(w + k, x[c], x[c + step], x[c + 2 * step], x[c + 3 * step]);
                        }
                    } else {
                        const T group = scipy_fir_four(w + k, fill, fill, fill, fill);
                        for (int64_t c = begin; c < end; ++c) dst[c] += group;
                    }
                }
                for (; k < kernel_width; ++k) {
                    if (row) {
                        const T* x = row + (first_col + step * k + begin);
                        T* __restrict sums = dst + begin;
                        const T weight = w[k];
                        for (int64_t c = 0; c < end - begin; ++c) sums[c] = fma_step(weight, x[c], sums[c]);
                    } else {
                        for (int64_t c = begin; c < end; ++c) dst[c] = fma_step(w[k], fill, dst[c]);
                    }
                }
            }
        }

        // SIMD can select a different NaN payload. Let SciPy handle any
        // nonfinite result, including overflow from entirely finite inputs.
        bool finite = true;
        for (int64_t c = 0; c < out_width; ++c) finite &= std::isfinite(dst[c]);
        if (!finite) throw Error(SM_ENOTSUP, "strict FIR delegates nonfinite results to SciPy");
    });
}
#endif

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

inline void compute_row_bases(const StencilSpec& spec, const TapTable& taps, const int64_t* in_strides,
                              uint64_t row_idx, uint8_t* outer_ok, int64_t* row_bases, int64_t* inner_offsets) {
    const uint32_t ndim = spec.out_shape.ndim;
    int64_t p_outer[SM_MAX_NDIM] = {};
    if (ndim > 1) {
        uint64_t rem = row_idx;
        for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; --d) {
            const uint64_t extent = spec.out_shape.dims[d];
            p_outer[d] = static_cast<int64_t>(rem % extent);
            rem /= extent;
        }
    }
    const size_t num_taps = taps.taps.size();
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
        outer_ok[k] = ok ? 1 : 0;
        row_bases[k] = off;
        inner_offsets[k] = off + tap.coord[ndim - 1];
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

    int64_t in_strides[SM_MAX_NDIM] = {};
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
        compute_row_bases(spec, taps, in_strides, r, ws.outer_ok.data(), ws.row_bases.data(), ws.inner_offsets.data());

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
                ws.ptrs[k] = ws.outer_ok[k] ? (src + (ws.inner_offsets[k] + mid_begin)) : nullptr;
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

    int64_t in_strides[SM_MAX_NDIM] = {};
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
        compute_row_bases(spec, taps, in_strides, r, ws.outer_ok.data(), ws.row_bases.data(), ws.inner_offsets.data());

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

void store_dense_row(uint32_t dst_dtype, const double* __restrict__ acc, void* dst, size_t n) {
    auto run = [&](auto* out) {
        using DstT = std::remove_pointer_t<decltype(out)>;
        for (size_t c = 0; c < n; ++c) {
            out[c] = array::cast_double_ndimage<DstT>(acc[c], /*dense_safe=*/true);
        }
    };
    switch (dst_dtype) {
    case SM_FLOAT64: run(static_cast<double*>(dst)); break;
    case SM_FLOAT32: run(static_cast<float*>(dst)); break;
    case SM_BOOL:
    case SM_UINT8: run(static_cast<uint8_t*>(dst)); break;
    case SM_UINT16: run(static_cast<uint16_t*>(dst)); break;
    case SM_UINT32: run(static_cast<uint32_t*>(dst)); break;
    case SM_UINT64: run(static_cast<uint64_t*>(dst)); break;
    case SM_INT8: run(static_cast<int8_t*>(dst)); break;
    case SM_INT16: run(static_cast<int16_t*>(dst)); break;
    case SM_INT32: run(static_cast<int32_t*>(dst)); break;
    case SM_INT64: run(static_cast<int64_t*>(dst)); break;
    default: throw Error(SM_EINVAL, "unknown dense_correlate output dtype");
    }
}

template<class SrcT>
void run_real_int_src(const StencilSpec& spec, const TapTable& taps, const void* input_ptr, void* output_ptr) {
    const SrcT* src = static_cast<const SrcT*>(input_ptr);
    uint8_t* dst = static_cast<uint8_t*>(output_ptr);
    const size_t dst_esize = static_cast<size_t>(array::dtype_size(spec.out_dtype));

    const uint32_t ndim = spec.out_shape.ndim;
    const uint64_t inner_u64 = spec.out_shape.dims[ndim - 1];
    const int64_t in_inner = static_cast<int64_t>(spec.in_shape.dims[ndim - 1]);
    const uint64_t outer_rows = spec.out_shape.elements / inner_u64;
    const size_t num_taps = taps.taps.size();

    int64_t in_strides[SM_MAX_NDIM] = {};
    if (ndim > 0) {
        in_strides[ndim - 1] = 1;
        for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; --d) {
            in_strides[d] = in_strides[d + 1] * static_cast<int64_t>(spec.in_shape.dims[d + 1]);
        }
    }

    std::vector<double> weights(num_taps);
    for (size_t k = 0; k < num_taps; ++k) weights[k] = taps.taps[k].weight_re_f64;
    const double cval = spec.cval_real;
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
    const uint64_t grain =
        outer_rows >= uint64_t{4} * lanes ? std::max<uint64_t>(1, 16384 / std::max<uint64_t>(inner_u64, 1)) : 1;

    struct Workspace {
        std::vector<int64_t> row_bases;
        std::vector<int64_t> inner_offsets;
        std::vector<uint8_t> outer_ok;
        std::vector<const SrcT*> ptrs;
        std::vector<double> acc;
    };
    const auto make = [&] {
        Workspace ws;
        ws.row_bases.resize(num_taps);
        ws.inner_offsets.resize(num_taps);
        ws.outer_ok.resize(num_taps);
        ws.ptrs.resize(num_taps);
        ws.acc.resize(static_cast<size_t>(std::min<uint64_t>(col_chunk, inner_u64)));
        return ws;
    };

    for_each_item(total_items, grain, lanes, make, [&](uint64_t item, Workspace& ws) {
        const uint64_t r = item / chunks;
        const uint64_t chunk_idx = item % chunks;
        const int64_t c_begin = static_cast<int64_t>(chunk_idx * col_chunk);
        const int64_t c_end =
            static_cast<int64_t>(std::min<uint64_t>(inner_u64, static_cast<uint64_t>(c_begin) + col_chunk));
        compute_row_bases(spec, taps, in_strides, r, ws.outer_ok.data(), ws.row_bases.data(), ws.inner_offsets.data());

        uint8_t* dst_row = dst + r * inner_u64 * dst_esize;
        const int64_t left_end = std::min<int64_t>(c_end, x_lo);
        const int64_t mid_begin = std::max<int64_t>(c_begin, x_lo);
        const int64_t mid_end = std::max<int64_t>(mid_begin, std::min<int64_t>(c_end, x_hi));
        const int64_t right_begin = std::max<int64_t>(c_begin, mid_end);

        auto eval_boundary = [&](int64_t begin, int64_t end) {
            if (begin >= end) return;
            const size_t n = static_cast<size_t>(end - begin);
            double* __restrict__ acc_ptr = ws.acc.data();
            for (int64_t c = begin; c < end; ++c) {
                double acc = 0.0;
                for (size_t k = 0; k < num_taps; ++k) {
                    if (!ws.outer_ok[k]) {
                        if (!spec.omit_oob) acc = fma_step(cval, weights[k], acc);
                        continue;
                    }
                    const int64_t q = extend(c + taps.taps[k].coord[ndim - 1], in_inner, inner_mode);
                    if (q < 0) {
                        if (!spec.omit_oob) acc = fma_step(cval, weights[k], acc);
                    } else {
                        acc = fma_step(static_cast<double>(src[ws.row_bases[k] + q]), weights[k], acc);
                    }
                }
                acc_ptr[static_cast<size_t>(c - begin)] = acc;
            }
            store_dense_row(spec.out_dtype, acc_ptr, dst_row + static_cast<size_t>(begin) * dst_esize, n);
        };

        eval_boundary(c_begin, left_end);
        if (mid_begin < mid_end) {
            const size_t n = static_cast<size_t>(mid_end - mid_begin);
            for (size_t k = 0; k < num_taps; ++k) {
                ws.ptrs[k] = ws.outer_ok[k] ? (src + (ws.inner_offsets[k] + mid_begin)) : nullptr;
            }
            double* __restrict__ acc_ptr = ws.acc.data();
            std::fill_n(acc_ptr, n, 0.0);
            sweep_interior_real(acc_ptr, ws.ptrs.data(), weights.data(), num_taps, n, cval, spec.omit_oob);
            store_dense_row(spec.out_dtype, acc_ptr, dst_row + static_cast<size_t>(mid_begin) * dst_esize, n);
        }
        eval_boundary(right_begin, c_end);
    });
}

void run_real_generic_int(const StencilSpec& spec, const TapTable& taps, const void* input_ptr, void* output_ptr) {
    switch (spec.in_dtype) {
    case SM_FLOAT64: run_real_int_src<double>(spec, taps, input_ptr, output_ptr); break;
    case SM_FLOAT32: run_real_int_src<float>(spec, taps, input_ptr, output_ptr); break;
    case SM_BOOL:
    case SM_UINT8: run_real_int_src<uint8_t>(spec, taps, input_ptr, output_ptr); break;
    case SM_UINT16: run_real_int_src<uint16_t>(spec, taps, input_ptr, output_ptr); break;
    case SM_UINT32: run_real_int_src<uint32_t>(spec, taps, input_ptr, output_ptr); break;
    case SM_UINT64: run_real_int_src<uint64_t>(spec, taps, input_ptr, output_ptr); break;
    case SM_INT8: run_real_int_src<int8_t>(spec, taps, input_ptr, output_ptr); break;
    case SM_INT16: run_real_int_src<int16_t>(spec, taps, input_ptr, output_ptr); break;
    case SM_INT32: run_real_int_src<int32_t>(spec, taps, input_ptr, output_ptr); break;
    case SM_INT64: run_real_int_src<int64_t>(spec, taps, input_ptr, output_ptr); break;
    default: throw Error(SM_EINVAL, "unknown dense_correlate input dtype");
    }
}

} // namespace

TapTable build_taps(const StencilSpec& spec) {
    const uint32_t ndim = spec.in_shape.ndim;
    if (ndim < 1 || ndim > SM_MAX_NDIM || spec.out_shape.ndim != ndim)
        throw Error(SM_EINVAL, "dense_correlate: invalid rank");
    if (!spec.weights) throw Error(SM_EINVAL, "dense_correlate: null weights");
    (void)array::dtype_size(spec.weights_dtype);
    if (!array::is_float(spec.weights_dtype) && !array::is_complex(spec.weights_dtype))
        throw Error(SM_EINVAL, "dense_correlate: weights dtype must be floating-point or complex");

    uint64_t w_total = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
        if (spec.wshape[d] < 1) throw Error(SM_EINVAL, "dense_correlate: weight extent must be >= 1");
        if (sm_checked_mul(w_total, spec.wshape[d], &w_total) != SM_OK)
            throw Error(SM_EOVERFLOW, "dense_correlate: weight count overflows");
    }

    int64_t in_strides[SM_MAX_NDIM] = {};
    in_strides[ndim - 1] = 1;
    for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; --d) {
        in_strides[d] = in_strides[d + 1] * static_cast<int64_t>(spec.in_shape.dims[d + 1]);
    }

    const bool cw = array::is_complex(spec.weights_dtype);
    const bool w_double = array::is_double(spec.weights_dtype);

    TapTable table;
    table.is_complex_weights = cw;
    table.taps.reserve(static_cast<size_t>(std::min<uint64_t>(w_total, 4096)));

    uint32_t w_coord[SM_MAX_NDIM] = {};
    int64_t min_c[SM_MAX_NDIM] = {};
    int64_t max_c[SM_MAX_NDIM] = {};
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

BufferPtr run_scipy_fir(sm_context* ctx, const StencilSpec& spec, const void* input, bool correlate) {
#if defined(__APPLE__) && defined(__aarch64__)
    check_fir_fpcr();
    uint64_t out_bytes = 0;
    if (sm_checked_mul(spec.out_shape.elements, array::dtype_size(spec.out_dtype), &out_bytes) != SM_OK)
        throw Error(SM_EOVERFLOW, "strict FIR output byte size overflows");
    auto result = array::allocate(ctx, out_bytes);
    void* out = mutable_data(result.get());
    auto run = [&]<class T>() {
        const auto* src = static_cast<const T*>(input);
        auto* dst = static_cast<T*>(out);
        if (correlate) run_scipy_fir_rows<T, true>(spec, src, dst);
        else run_scipy_fir_rows<T, false>(spec, src, dst);
    };
    if (spec.out_dtype == SM_FLOAT32) run.operator()<float>();
    else run.operator()<double>();
    return result;
#else
    (void)ctx;
    (void)spec;
    (void)input;
    (void)correlate;
    throw Error(SM_ENOTSUP, "strict FIR requires the verified macOS arm64 arithmetic");
#endif
}

BufferPtr run_cpu(sm_context* ctx, const StencilSpec& spec, const TapTable& taps, const void* input) {
    uint64_t out_bytes = 0;
    if (sm_checked_mul(spec.out_shape.elements, array::dtype_size(spec.out_dtype), &out_bytes) != SM_OK)
        throw Error(SM_EOVERFLOW, "dense_correlate: output byte size overflows");
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
                throw Error(SM_EINVAL, "dense_correlate: empty input with non-constant boundary");
        }
    }

    const bool in_c = array::is_complex(spec.in_dtype);
    const bool out_c = array::is_complex(spec.out_dtype);
    if (!in_c && !out_c && !taps.is_complex_weights) {
        if (array::is_int_or_bool(spec.in_dtype) || array::is_int_or_bool(spec.out_dtype)) {
            run_real_generic_int(spec, taps, input, out_ptr);
            return result;
        }
        const bool use_f32_accum =
            spec.cpu_accum == AccumMode::InputDtype && spec.in_dtype == SM_FLOAT32 && spec.out_dtype == SM_FLOAT32;
        if (use_f32_accum) {
            run_real_impl<float, float, float>(spec, taps, input, out_ptr);
        } else if (spec.in_dtype == SM_FLOAT32 && spec.out_dtype == SM_FLOAT32) {
            run_real_impl<float, float, double>(spec, taps, input, out_ptr);
        } else if (spec.in_dtype == SM_FLOAT32 && spec.out_dtype == SM_FLOAT64) {
            run_real_impl<float, double, double>(spec, taps, input, out_ptr);
        } else if (spec.in_dtype == SM_FLOAT64 && spec.out_dtype == SM_FLOAT32) {
            run_real_impl<double, float, double>(spec, taps, input, out_ptr);
        } else {
            run_real_impl<double, double, double>(spec, taps, input, out_ptr);
        }
        return result;
    }

    if (!out_c) throw Error(SM_EINVAL, "dense_correlate: complex input or weights require complex output");
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
        if (!array::is_float(spec.in_dtype))
            throw Error(SM_EINVAL, "dense_correlate: real input for complex output must be floating-point");
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

#if !defined(SM_ENABLE_METAL)
bool gpu_supports(const StencilSpec&, const TapTable&) noexcept { return false; }
#endif

} // namespace sm::detail::dense_correlate
