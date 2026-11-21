// CPU engine of the separable filters: each 1-D pass in double precision on
// the performance cores, with SciPy's operation order.
//
// SciPy (ni_filters.c) copies every line into a double buffer, extends it,
// filters it and casts the result to the array's type. The loops here compute
// the same values without that per-line copy:
//   lines  the filtered axis is the innermost one (or nearly): chunks of a
//          line are gathered into an extended double buffer and filtered with
//          contiguous sweeps;
//   tiles  otherwise: the array is cut into tiles of adjacent lines and the
//          arithmetic is vectorized across them, row by row.
// Results do not depend on the split or on the thread count.
#include "filters.hpp"
#include "ms/detail/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// A product is fused into a sum only where __builtin_fma says so: the
// difference is visible in the last bit, and SciPy's binary fixes which taps
// fuse (see correlate_row).
#pragma STDC FP_CONTRACT OFF

namespace ms::detail::ndimage {
namespace {
// Lines closer together than this are filtered as lines; from here on tiles
// have enough columns for vector loops.
constexpr uint64_t min_tile_columns = 8;
// One lane per this many tap-element operations (about 50 us of work): below
// it waking another core costs more than it saves.
constexpr uint64_t min_work_per_lane = uint64_t{1} << 17;
// Windows up to this many taps keep a ring of rows (tiles); longer ones would
// push the ring out of the caches and are filtered line by line.
constexpr uint64_t max_ring_taps = 4096;
// Lanes share this much workspace at most, so an absurdly long window slows
// down instead of exhausting memory.
constexpr uint64_t workspace_budget_bytes = uint64_t{1} << 30;
// Outputs per step of a line, so its extended copy and accumulator stay in L1.
constexpr int64_t line_chunk = 4096;
// Rows fetched ahead of a tile: its strided reads and writes defeat the
// hardware prefetcher.
constexpr int64_t prefetch_rows = 8;

unsigned lanes_for(uint64_t elements, uint64_t operations_per_element, uint64_t workspace_bytes = 0) {
    uint64_t work = 0;
    if (ms_checked_mul(elements, operations_per_element, &work) != MS_OK) work = UINT64_MAX;
    uint64_t lanes = std::clamp<uint64_t>(work / min_work_per_lane, 1, parallel::hardware_lanes());
    if (workspace_bytes) lanes = std::clamp<uint64_t>(workspace_budget_bytes / workspace_bytes, 1, lanes);
    return static_cast<unsigned>(lanes);
}

// Runs fn(item, workspace) for every item of [0, items) on up to `lanes`
// cores. Lanes claim `grain` items at a time, so a core that is preempted
// simply takes fewer; each lane builds one workspace with `make`.
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

inline void to_double(const double* src, double* dst, size_t n) { std::memcpy(dst, src, n * sizeof(double)); }
inline void to_double(const float* src, double* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(src[i]);
}
inline void prefetch(const void* address, size_t bytes, int write) {
    const char* p = static_cast<const char*>(address);
    for (size_t offset = 0; offset < bytes; offset += 64) {
        if (write) __builtin_prefetch(p + offset, 1, 3);
        else __builtin_prefetch(p + offset, 0, 3);
    }
}

// ---- correlation ----------------------------------------------------------

// One step of SciPy's inner loop for a pair of taps: sum += (a +- b) * w.
template<bool Subtract, bool Fused>
inline double pair_step(double sum, double a, double b, double w) {
    const double t = Subtract ? a - b : a + b;
    return Fused ? __builtin_fma(t, w, sum) : sum + t * w;
}

// acc[c] = pair_step(acc[c], a[k][c], b[k][c], w[k]) for k = 0 .. taps-1, in
// that order for every c: vectorized across c, never across k. Four taps per
// sweep keep acc in registers.
template<bool Subtract, bool Fused>
void pair_sweep(double* __restrict acc, const double* const* a, const double* const* b, const double* w,
                size_t taps, size_t n) {
    size_t k = 0;
    for (; k + 4 <= taps; k += 4) {
        const double *a0 = a[k], *a1 = a[k + 1], *a2 = a[k + 2], *a3 = a[k + 3];
        const double *b0 = b[k], *b1 = b[k + 1], *b2 = b[k + 2], *b3 = b[k + 3];
        const double w0 = w[k], w1 = w[k + 1], w2 = w[k + 2], w3 = w[k + 3];
        for (size_t c = 0; c < n; ++c) {
            double s = acc[c];
            s = pair_step<Subtract, Fused>(s, a0[c], b0[c], w0);
            s = pair_step<Subtract, Fused>(s, a1[c], b1[c], w1);
            s = pair_step<Subtract, Fused>(s, a2[c], b2[c], w2);
            s = pair_step<Subtract, Fused>(s, a3[c], b3[c], w3);
            acc[c] = s;
        }
    }
    for (; k < taps; ++k) {
        const double *ak = a[k], *bk = b[k];
        const double wk = w[k];
        for (size_t c = 0; c < n; ++c) acc[c] = pair_step<Subtract, Fused>(acc[c], ak[c], bk[c], wk);
    }
}

// acc[c] = fma(x[j][c], w[j], acc[c]) for j = 0 .. taps-1, in that order.
void tap_sweep(double* __restrict acc, const double* const* x, const double* w, size_t taps, size_t n) {
    size_t j = 0;
    for (; j + 4 <= taps; j += 4) {
        const double *x0 = x[j], *x1 = x[j + 1], *x2 = x[j + 2], *x3 = x[j + 3];
        const double w0 = w[j], w1 = w[j + 1], w2 = w[j + 2], w3 = w[j + 3];
        for (size_t c = 0; c < n; ++c) {
            double s = acc[c];
            s = __builtin_fma(x0[c], w0, s);
            s = __builtin_fma(x1[c], w1, s);
            s = __builtin_fma(x2[c], w2, s);
            s = __builtin_fma(x3[c], w3, s);
            acc[c] = s;
        }
    }
    for (; j < taps; ++j) {
        const double* xj = x[j];
        const double wj = w[j];
        for (size_t c = 0; c < n; ++c) acc[c] = __builtin_fma(xj[c], wj, acc[c]);
    }
}

// acc[c] = the correlation at n adjacent outputs, where tap(j) points at the
// n source values of tap j. a and b hold `size` pointers each.
//
// The order is NI_Correlate1D's. Symmetric and antisymmetric weights: centre
// tap, then pairs from the outermost inwards, weighted by the lower half.
// Other weights: the last tap, then the rest in order. Which steps are fused
// multiply-adds is a property of the SciPy 1.18.1 arm64 wheel's machine code,
// not of its source: the pair loop is vectorized eight pairs at a time with
// separate multiplies and an in-order sum when there are at least ten pairs,
// and its scalar remainder (every pair when there are fewer than ten) and the
// whole unpaired loop use fmadd. Reproducing that makes float64 results
// identical to that wheel's; any other choice differs by an ulp or two.
template<class Tap>
inline void correlate_row(const Pass& pass, double* __restrict acc, size_t n, Tap&& tap, const double** a,
                          const double** b) {
    const size_t taps = static_cast<size_t>(pass.size);
    const double* w = pass.weights;
    if (pass.symmetry == Symmetry::none) {
        const double* last = tap(taps - 1);
        const double scale = w[taps - 1];
        for (size_t c = 0; c < n; ++c) acc[c] = last[c] * scale;
        for (size_t j = 0; j + 1 < taps; ++j) a[j] = tap(j);
        tap_sweep(acc, a, w, taps - 1, n);
        return;
    }
    const size_t half = taps / 2;
    const double* centre = tap(half);
    const double scale = w[half];
    for (size_t c = 0; c < n; ++c) acc[c] = centre[c] * scale;
    for (size_t k = 0; k < half; ++k) {
        a[k] = tap(k);
        b[k] = tap(taps - 1 - k);
    }
    const size_t unfused = half >= 10 ? half & ~size_t{7} : 0;
    if (pass.symmetry == Symmetry::even) {
        pair_sweep<false, false>(acc, a, b, w, unfused, n);
        pair_sweep<false, true>(acc, a + unfused, b + unfused, w + unfused, half - unfused, n);
    } else {
        pair_sweep<true, false>(acc, a, b, w, unfused, n);
        pair_sweep<true, true>(acc, a + unfused, b + unfused, w + unfused, half - unfused, n);
    }
}

uint64_t correlate_operations(const Pass& pass) {
    return pass.symmetry == Symmetry::none ? pass.size : pass.size / 2 + 1;
}

struct CorrelateWorkspace {
    std::vector<double> window;  // tiles: ring of extended rows; lines: the extended chunk
    std::vector<double> acc;
    std::vector<const double*> a, b;
};

enum class StoreOp : uint32_t {
    write = 0,
    add = 1,
    square = 2,
    square_add = 3,
    square_sqrt = 4,
    square_add_sqrt = 5,
};

template<class DstT, StoreOp Op>
inline void store_val(DstT& dst, double val) noexcept {
    const DstT v = static_cast<DstT>(val);
    if constexpr (Op == StoreOp::write) {
        dst = v;
    } else if constexpr (Op == StoreOp::add) {
        dst = dst + v;
    } else if constexpr (Op == StoreOp::square) {
        dst = v * v;
    } else if constexpr (Op == StoreOp::square_add) {
        dst = dst + v * v;
    } else if constexpr (Op == StoreOp::square_sqrt) {
        dst = std::sqrt(v * v);
    } else {
        dst = std::sqrt(dst + v * v);
    }
}

// Tiles: for one block of the array, columns [c0, c0 + n) and rows [y0, y1)
// along the axis. A ring holds the `size` extended rows the current output row
// reads, as doubles: one contiguous, L1-resident copy of the input instead of
// `size` row segments a whole row apart (which evict each other when the row
// length is a power of two), and float32 rows are widened once, not per tap.
template<class SrcT, class DstT, StoreOp Op = StoreOp::write>
void correlate_tiles(const Pass& pass, const SrcT* src, DstT* dst) {
    const size_t taps = static_cast<size_t>(pass.size);
    const size_t inner = static_cast<size_t>(pass.inner);
    const int64_t length = static_cast<int64_t>(pass.length);
    const int64_t before = static_cast<int64_t>(pass.before), after = static_cast<int64_t>(pass.after);
    // Tile width: a ring of about 64 KiB.
    const size_t tile = std::min(inner, std::clamp<size_t>((65536 / (taps * sizeof(double))) & ~size_t{7}, 32, 2048));
    const uint64_t tiles = (inner + tile - 1) / tile;
    const size_t pitch = (tile + 7) & ~size_t{7};
    const unsigned lanes = lanes_for(pass.outer * pass.length * pass.inner, correlate_operations(pass),
                                     taps * pitch * sizeof(double));
    // Blocks x tiles are the natural work items. The axis is cut too only when
    // they cannot keep every lane busy: each cut reloads size - 1 rows.
    uint64_t chunks = 1;
    const uint64_t wanted = uint64_t{4} * lanes;
    if (lanes > 1 && pass.outer * tiles < wanted) {
        const uint64_t worthwhile = std::max<uint64_t>(1, pass.length / (8 * pass.size));
        chunks = std::min(worthwhile, (wanted + pass.outer * tiles - 1) / (pass.outer * tiles));
    }
    const int64_t rows = static_cast<int64_t>((pass.length + chunks - 1) / chunks);
    chunks = (pass.length + static_cast<uint64_t>(rows) - 1) / static_cast<uint64_t>(rows);
    const auto make = [&] {
        CorrelateWorkspace ws;
        ws.window.resize(taps * pitch);
        if constexpr (!(std::is_same_v<DstT, double> && Op == StoreOp::write)) ws.acc.resize(pitch);
        ws.a.resize(taps);
        ws.b.resize(taps);
        return ws;
    };
    for_each_item(pass.outer * chunks * tiles, 1, lanes, make, [&](uint64_t item, CorrelateWorkspace& ws) {
        const uint64_t block = item / (chunks * tiles);
        const int64_t y0 = static_cast<int64_t>((item / tiles) % chunks) * rows;
        const int64_t y1 = std::min(length, y0 + rows);
        const size_t c0 = static_cast<size_t>(item % tiles) * tile;
        const size_t n = std::min(tile, inner - c0);
        const SrcT* in = src + block * pass.length * pass.inner + c0;
        DstT* out = dst + block * pass.length * pass.inner + c0;
        double* ring = ws.window.data();
        const auto load = [&](int64_t position, size_t slot) {
            double* row = ring + slot * pitch;
            const int64_t source = position >= 0 && position < length ? position : extend(position, length, pass.mode);
            if (source < 0) {
                std::fill_n(row, n, pass.cval);
                return;
            }
            to_double(in + static_cast<size_t>(source) * inner, row, n);
            if (source + prefetch_rows < length)
                prefetch(in + static_cast<size_t>(source + prefetch_rows) * inner, n * sizeof(SrcT), 0);
        };
        // Slot s holds extension row y0 - before + s (mod size).
        for (size_t slot = 0; slot + 1 < taps; ++slot) load(y0 - before + static_cast<int64_t>(slot), slot);
        size_t head = 0, incoming = taps - 1;
        for (int64_t y = y0; y < y1; ++y) {
            load(y + after, incoming);
            incoming = incoming + 1 == taps ? 0 : incoming + 1;
            DstT* target = out + static_cast<size_t>(y) * inner;
            if (y + prefetch_rows < y1) prefetch(target + static_cast<size_t>(prefetch_rows) * inner, n * sizeof(DstT), 1);
            const auto tap = [&](size_t j) -> const double* {
                const size_t slot = head + j;
                return ring + (slot >= taps ? slot - taps : slot) * pitch;
            };
            if constexpr (std::is_same_v<DstT, double> && Op == StoreOp::write) {
                correlate_row(pass, target, n, tap, ws.a.data(), ws.b.data());
            } else {
                double* acc = ws.acc.data();
                correlate_row(pass, acc, n, tap, ws.a.data(), ws.b.data());
                for (size_t c = 0; c < n; ++c) store_val<DstT, Op>(target[c], acc[c]);
            }
            head = head + 1 == taps ? 0 : head + 1;
        }
    });
}

// values[t] = extension position first + t of the line in[0], in[stride], ...
template<class T>
void extended_line(const Pass& pass, const T* in, size_t stride, int64_t first, size_t count, double* values) {
    const int64_t length = static_cast<int64_t>(pass.length);
    const auto outside = [&](int64_t position) {
        const int64_t source = extend(position, length, pass.mode);
        return source < 0 ? pass.cval : static_cast<double>(in[static_cast<size_t>(source) * stride]);
    };
    const int64_t end = first + static_cast<int64_t>(count);
    int64_t position = first;
    size_t t = 0;
    for (; position < std::min<int64_t>(0, end); ++position) values[t++] = outside(position);
    const int64_t inside_end = std::min(length, end);
    if (position < inside_end) {
        const size_t n = static_cast<size_t>(inside_end - position);
        const T* source = in + static_cast<size_t>(position) * stride;
        if (stride == 1) {
            to_double(source, values + t, n);
        } else {
            for (size_t i = 0; i < n; ++i) values[t + i] = static_cast<double>(source[i * stride]);
        }
        t += n;
        position = inside_end;
    }
    for (; position < end; ++position) values[t++] = outside(position);
}

// Lines: every tap of a chunk of one line is the same extended buffer, shifted.
template<class SrcT, class DstT, StoreOp Op = StoreOp::write>
void correlate_lines(const Pass& pass, const SrcT* src, DstT* dst) {
    const size_t taps = static_cast<size_t>(pass.size);
    const size_t stride = static_cast<size_t>(pass.inner);
    const int64_t length = static_cast<int64_t>(pass.length);
    const int64_t before = static_cast<int64_t>(pass.before);
    const uint64_t chunks = (pass.length + line_chunk - 1) / line_chunk;
    const uint64_t lines = pass.outer * pass.inner;
    const size_t widest = static_cast<size_t>(std::min<int64_t>(length, line_chunk));
    const unsigned lanes = lanes_for(lines * pass.length, correlate_operations(pass),
                                     (widest + 3 * taps) * sizeof(double));
    const uint64_t grain = std::max<uint64_t>(1, 16384 / widest);  // short lines: several per claim
    const auto make = [&] {
        CorrelateWorkspace ws;
        ws.window.resize(widest + taps - 1);
        ws.acc.resize(widest);
        ws.a.resize(taps);
        ws.b.resize(taps);
        return ws;
    };
    for_each_item(lines * chunks, grain, lanes, make, [&](uint64_t item, CorrelateWorkspace& ws) {
        const uint64_t line = item / chunks;
        const int64_t begin = static_cast<int64_t>(item % chunks) * line_chunk;
        const size_t n = static_cast<size_t>(std::min(length, begin + line_chunk) - begin);
        const uint64_t offset = (line / pass.inner) * pass.length * pass.inner + line % pass.inner;
        const double* window = ws.window.data();
        extended_line(pass, src + offset, stride, begin - before, n + taps - 1, ws.window.data());
        const auto tap = [&](size_t j) { return window + j; };
        DstT* out = dst + offset + static_cast<size_t>(begin) * stride;
        if constexpr (std::is_same_v<DstT, double> && Op == StoreOp::write) {
            if (stride == 1) {
                correlate_row(pass, out, n, tap, ws.a.data(), ws.b.data());
                return;
            }
        }
        double* acc = ws.acc.data();
        correlate_row(pass, acc, n, tap, ws.a.data(), ws.b.data());
        for (size_t i = 0; i < n; ++i) store_val<DstT, Op>(out[i * stride], acc[i]);
    });
}

// ---- uniform filter -------------------------------------------------------
//
// NI_UniformFilter1D: the first output is the in-order sum of its window
// divided by the size; every later one adds (entering - leaving) to the
// running sum. The recurrence runs the length of a line, so lines (or tiles of
// them) are the unit of parallelism and each is walked from its start.

struct UniformWorkspace {
    std::vector<double> sum, entering, leaving, constant;
};

template<class SrcT, class DstT, StoreOp Op = StoreOp::write>
void uniform_tiles(const Pass& pass, const SrcT* src, DstT* dst) {
    const size_t inner = static_cast<size_t>(pass.inner);
    const int64_t length = static_cast<int64_t>(pass.length), taps = static_cast<int64_t>(pass.size);
    const int64_t before = static_cast<int64_t>(pass.before), after = static_cast<int64_t>(pass.after);
    const unsigned lanes = lanes_for(pass.outer * pass.length * pass.inner, 2);
    size_t tile = std::min<size_t>(inner, 2048);
    if (lanes > 1) {
        const uint64_t initial_tiles = (inner + tile - 1) / tile;
        if (pass.outer * initial_tiles < uint64_t{4} * lanes) {
            const uint64_t wanted_tiles = (uint64_t{4} * lanes + pass.outer - 1) / pass.outer;
            const size_t scaled = (inner / static_cast<size_t>(wanted_tiles)) & ~size_t{7};
            tile = std::clamp<size_t>(scaled, std::min<size_t>(inner, 32), 2048);
        }
    }
    const uint64_t tiles = (inner + tile - 1) / tile;
    const double size = static_cast<double>(pass.size);
    const int64_t inside_begin = std::min(length, before + 1);
    const int64_t inside_end = std::max(inside_begin, length - after);
    const auto make = [&] {
        UniformWorkspace ws;
        ws.sum.resize(tile);
        ws.constant.assign(tile, pass.cval);
        if constexpr (!std::is_same_v<SrcT, double>) {
            ws.entering.resize(tile);
            ws.leaving.resize(tile);
        }
        return ws;
    };
    for_each_item(pass.outer * tiles, 1, lanes, make, [&](uint64_t item, UniformWorkspace& ws) {
        const size_t c0 = static_cast<size_t>(item % tiles) * tile;
        const size_t n = std::min(tile, inner - c0);
        const SrcT* in = src + (item / tiles) * pass.length * pass.inner + c0;
        DstT* out = dst + (item / tiles) * pass.length * pass.inner + c0;
        // The n values of one extended row, as doubles.
        const auto row = [&](int64_t position, std::vector<double>& converted) -> const double* {
            const int64_t source = position >= 0 && position < length ? position : extend(position, length, pass.mode);
            if (source < 0) return ws.constant.data();
            const SrcT* values = in + static_cast<size_t>(source) * inner;
            if constexpr (std::is_same_v<SrcT, double>) {
                (void)converted;
                return values;
            } else {
                to_double(values, converted.data(), n);
                return converted.data();
            }
        };
        double* __restrict sum = ws.sum.data();
        std::fill_n(sum, n, 0.0);
        for (int64_t k = 0; k < taps; ++k) {
            const double* values = row(k - before, ws.entering);
            for (size_t c = 0; c < n; ++c) sum[c] = sum[c] + values[c];
        }
        for (size_t c = 0; c < n; ++c) store_val<DstT, Op>(out[c], sum[c] / size);
        int64_t y = 1;
        for (; y < inside_begin; ++y) {
            const double* entering = row(y + after, ws.entering);
            const double* leaving = row(y - 1 - before, ws.leaving);
            DstT* __restrict target = out + static_cast<size_t>(y) * inner;
            for (size_t c = 0; c < n; ++c) {
                const double s = sum[c] + (entering[c] - leaving[c]);
                sum[c] = s;
                store_val<DstT, Op>(target[c], s / size);
            }
        }
        for (; y < inside_end; ++y) {
            const SrcT* e_src = in + static_cast<size_t>(y + after) * inner;
            const SrcT* l_src = in + static_cast<size_t>(y - 1 - before) * inner;
            if (y + after + prefetch_rows < length)
                prefetch(in + static_cast<size_t>(y + after + prefetch_rows) * inner, n * sizeof(SrcT), 0);
            if (y + prefetch_rows < length)
                prefetch(out + static_cast<size_t>(y + prefetch_rows) * inner, n * sizeof(DstT), 1);
            const double* __restrict entering;
            const double* __restrict leaving;
            if constexpr (std::is_same_v<SrcT, double>) {
                entering = e_src;
                leaving = l_src;
            } else {
                to_double(e_src, ws.entering.data(), n);
                to_double(l_src, ws.leaving.data(), n);
                entering = ws.entering.data();
                leaving = ws.leaving.data();
            }
            DstT* __restrict target = out + static_cast<size_t>(y) * inner;
            for (size_t c = 0; c < n; ++c) {
                const double s = sum[c] + (entering[c] - leaving[c]);
                sum[c] = s;
                store_val<DstT, Op>(target[c], s / size);
            }
        }
        for (; y < length; ++y) {
            const double* entering = row(y + after, ws.entering);
            const double* leaving = row(y - 1 - before, ws.leaving);
            DstT* __restrict target = out + static_cast<size_t>(y) * inner;
            for (size_t c = 0; c < n; ++c) {
                const double s = sum[c] + (entering[c] - leaving[c]);
                sum[c] = s;
                store_val<DstT, Op>(target[c], s / size);
            }
        }
    });
}

#if defined(__aarch64__) && defined(__ARM_NEON)
template<size_t Pairs, class SrcT, class DstT, StoreOp Op>
void uniform_lines_neon_bundle(const Pass& pass, const SrcT* src, DstT* dst, uint64_t line_begin,
                               int64_t inside_begin, int64_t inside_end) {
    constexpr size_t Lines = Pairs * 2;
    const int64_t length = static_cast<int64_t>(pass.length);
    const int64_t taps = static_cast<int64_t>(pass.size);
    const int64_t before = static_cast<int64_t>(pass.before);
    const int64_t after = static_cast<int64_t>(pass.after);
    const double size = static_cast<double>(pass.size);
    const float64x2_t vsize = vdupq_n_f64(size);

    const SrcT* ins[Lines];
    DstT* outs[Lines];
    for (size_t m = 0; m < Lines; ++m) {
        const size_t offset = static_cast<size_t>(line_begin + m) * static_cast<size_t>(pass.length);
        ins[m] = src + offset;
        outs[m] = dst + offset;
    }

    auto load1_pair = [&](size_t pair_idx, int64_t source) -> float64x2_t {
        if (source < 0) return vdupq_n_f64(pass.cval);
        const size_t s = static_cast<size_t>(source);
        const double d0 = static_cast<double>(ins[pair_idx * 2][s]);
        const double d1 = static_cast<double>(ins[pair_idx * 2 + 1][s]);
        return vsetq_lane_f64(d1, vdupq_n_f64(d0), 1);
    };

    auto store1_pair = [&](size_t pair_idx, int64_t y, float64x2_t q) {
        const size_t u = static_cast<size_t>(y);
        store_val<DstT, Op>(outs[pair_idx * 2][u], vgetq_lane_f64(q, 0));
        store_val<DstT, Op>(outs[pair_idx * 2 + 1][u], vgetq_lane_f64(q, 1));
    };

    auto load2 = [](const SrcT* p) -> float64x2_t {
        if constexpr (std::is_same_v<SrcT, double>) {
            return vld1q_f64(p);
        } else {
            return vcvt_f64_f32(vld1_f32(p));
        }
    };

    auto store2 = [](DstT* p, float64x2_t v) {
        if constexpr (Op == StoreOp::write) {
            if constexpr (std::is_same_v<DstT, double>) {
                vst1q_f64(p, v);
            } else {
                vst1_f32(p, vcvt_f32_f64(v));
            }
        } else {
            store_val<DstT, Op>(p[0], vgetq_lane_f64(v, 0));
            store_val<DstT, Op>(p[1], vgetq_lane_f64(v, 1));
        }
    };

    float64x2_t sum[Pairs];
    for (size_t p = 0; p < Pairs; ++p) sum[p] = vdupq_n_f64(0.0);

    for (int64_t k = 0; k < taps; ++k) {
        const int64_t pos = k - before;
        const int64_t source = pos >= 0 && pos < length ? pos : extend(pos, length, pass.mode);
        for (size_t p = 0; p < Pairs; ++p) {
            sum[p] = vaddq_f64(sum[p], load1_pair(p, source));
        }
    }
    for (size_t p = 0; p < Pairs; ++p) {
        store1_pair(p, 0, vdivq_f64(sum[p], vsize));
    }

    int64_t y = 1;
    for (; y < inside_begin; ++y) {
        const int64_t pe = y + after;
        const int64_t pl = y - 1 - before;
        const int64_t se = pe >= 0 && pe < length ? pe : extend(pe, length, pass.mode);
        const int64_t sl = pl >= 0 && pl < length ? pl : extend(pl, length, pass.mode);
        for (size_t p = 0; p < Pairs; ++p) {
            const float64x2_t e = load1_pair(p, se);
            const float64x2_t l = load1_pair(p, sl);
            sum[p] = vaddq_f64(sum[p], vsubq_f64(e, l));
            store1_pair(p, y, vdivq_f64(sum[p], vsize));
        }
    }

    for (; y + 1 < inside_end; y += 2) {
        const size_t oe = static_cast<size_t>(y + after);
        const size_t ol = static_cast<size_t>(y - 1 - before);
        const size_t u = static_cast<size_t>(y);
        for (size_t p = 0; p < Pairs; ++p) {
            const SrcT* ina = ins[p * 2];
            const SrcT* inb = ins[p * 2 + 1];
            const float64x2_t ea = load2(ina + oe);
            const float64x2_t eb = load2(inb + oe);
            const float64x2_t la = load2(ina + ol);
            const float64x2_t lb = load2(inb + ol);
            const float64x2_t d0 = vsubq_f64(vtrn1q_f64(ea, eb), vtrn1q_f64(la, lb));
            const float64x2_t d1 = vsubq_f64(vtrn2q_f64(ea, eb), vtrn2q_f64(la, lb));
            sum[p] = vaddq_f64(sum[p], d0);
            const float64x2_t q0 = vdivq_f64(sum[p], vsize);
            sum[p] = vaddq_f64(sum[p], d1);
            const float64x2_t q1 = vdivq_f64(sum[p], vsize);
            store2(outs[p * 2] + u, vtrn1q_f64(q0, q1));
            store2(outs[p * 2 + 1] + u, vtrn2q_f64(q0, q1));
        }
    }

    for (; y < inside_end; ++y) {
        const int64_t se = y + after;
        const int64_t sl = y - 1 - before;
        for (size_t p = 0; p < Pairs; ++p) {
            const float64x2_t e = load1_pair(p, se);
            const float64x2_t l = load1_pair(p, sl);
            sum[p] = vaddq_f64(sum[p], vsubq_f64(e, l));
            store1_pair(p, y, vdivq_f64(sum[p], vsize));
        }
    }

    for (; y < length; ++y) {
        const int64_t pe = y + after;
        const int64_t pl = y - 1 - before;
        const int64_t se = pe >= 0 && pe < length ? pe : extend(pe, length, pass.mode);
        const int64_t sl = pl >= 0 && pl < length ? pl : extend(pl, length, pass.mode);
        for (size_t p = 0; p < Pairs; ++p) {
            const float64x2_t e = load1_pair(p, se);
            const float64x2_t l = load1_pair(p, sl);
            sum[p] = vaddq_f64(sum[p], vsubq_f64(e, l));
            store1_pair(p, y, vdivq_f64(sum[p], vsize));
        }
    }
}
#endif

template<class SrcT, class DstT, StoreOp Op = StoreOp::write>
void uniform_lines(const Pass& pass, const SrcT* src, DstT* dst) {
    const size_t stride = static_cast<size_t>(pass.inner);
    const int64_t length = static_cast<int64_t>(pass.length), taps = static_cast<int64_t>(pass.size);
    const int64_t before = static_cast<int64_t>(pass.before), after = static_cast<int64_t>(pass.after);
    const uint64_t lines = pass.outer * pass.inner;
    const unsigned lanes = lanes_for(lines * pass.length, 2);
    const double size = static_cast<double>(pass.size);
    // Outputs whose entering and leaving values both lie inside the line.
    const int64_t inside_begin = std::min(length, before + 1);
    const int64_t inside_end = std::max(inside_begin, length - after);

    auto run_scalar_line = [&](uint64_t line) {
        const uint64_t offset = (line / pass.inner) * pass.length * pass.inner + line % pass.inner;
        const SrcT* in = src + offset;
        DstT* out = dst + offset;
        const auto at = [&](int64_t position) -> double {
            const int64_t source = position >= 0 && position < length ? position : extend(position, length, pass.mode);
            return source < 0 ? pass.cval : static_cast<double>(in[static_cast<size_t>(source) * stride]);
        };
        double sum = 0.0;
        for (int64_t k = 0; k < taps; ++k) sum = sum + at(k - before);
        store_val<DstT, Op>(out[0], sum / size);
        int64_t y = 1;
        for (; y < inside_begin; ++y) {
            sum = sum + (at(y + after) - at(y - 1 - before));
            store_val<DstT, Op>(out[static_cast<size_t>(y) * stride], sum / size);
        }
        for (; y < inside_end; ++y) {
            sum = sum + (static_cast<double>(in[static_cast<size_t>(y + after) * stride]) -
                         static_cast<double>(in[static_cast<size_t>(y - 1 - before) * stride]));
            store_val<DstT, Op>(out[static_cast<size_t>(y) * stride], sum / size);
        }
        for (; y < length; ++y) {
            sum = sum + (at(y + after) - at(y - 1 - before));
            store_val<DstT, Op>(out[static_cast<size_t>(y) * stride], sum / size);
        }
    };

#if defined(__aarch64__) && defined(__ARM_NEON)
    if (stride == 1 && lines >= 4) {
        const uint64_t bundles4 = lines / 4;
        const uint64_t bundle_grain = std::max<uint64_t>(1, 4096 / pass.length);
        if (bundles4) {
            for_each_item(bundles4, bundle_grain, lanes, [] { return 0; }, [&](uint64_t bundle, int&) {
                uniform_lines_neon_bundle<2, SrcT, DstT, Op>(pass, src, dst, bundle * 4, inside_begin, inside_end);
            });
        }
        for (uint64_t line = bundles4 * 4; line < lines; ++line) {
            run_scalar_line(line);
        }
        return;
    }
#endif

    const uint64_t grain = std::max<uint64_t>(1, 16384 / pass.length);
    for_each_item(lines, grain, lanes, [] { return 0; }, [&](uint64_t line, int&) {
        run_scalar_line(line);
    });
}

template<class SrcT, class DstT, StoreOp Op = StoreOp::write>
void run_pass(const Pass& pass, const SrcT* src, DstT* dst) {
    const bool tiles = pass.inner >= min_tile_columns;
    if (pass.kind == MS_NDIMAGE_UNIFORM) {
        if (tiles) uniform_tiles<SrcT, DstT, Op>(pass, src, dst);
        else uniform_lines<SrcT, DstT, Op>(pass, src, dst);
    } else {
        if (tiles && pass.size <= max_ring_taps) correlate_tiles<SrcT, DstT, Op>(pass, src, dst);
        else correlate_lines<SrcT, DstT, Op>(pass, src, dst);
    }
}

template<class SrcT, class DstT, StoreOp Op>
void run_pass_op(const Pass& pass, const void* src, void* dst) {
    run_pass<SrcT, DstT, Op>(pass, static_cast<const SrcT*>(src), static_cast<DstT*>(dst));
}

template<class SrcT, class DstT>
void run_pass_with_op(const Pass& pass, const void* src, void* dst, StoreOp op) {
    switch (op) {
    case StoreOp::write: run_pass_op<SrcT, DstT, StoreOp::write>(pass, src, dst); break;
    case StoreOp::add: run_pass_op<SrcT, DstT, StoreOp::add>(pass, src, dst); break;
    case StoreOp::square: run_pass_op<SrcT, DstT, StoreOp::square>(pass, src, dst); break;
    case StoreOp::square_add: run_pass_op<SrcT, DstT, StoreOp::square_add>(pass, src, dst); break;
    case StoreOp::square_sqrt: run_pass_op<SrcT, DstT, StoreOp::square_sqrt>(pass, src, dst); break;
    case StoreOp::square_add_sqrt: run_pass_op<SrcT, DstT, StoreOp::square_add_sqrt>(pass, src, dst); break;
    }
}

template<class SrcT, class DstT, StoreOp Op>
void apply_identity_op(uint64_t elements, const SrcT* src, DstT* dst) {
    parallel::parallel_for(elements, min_work_per_lane, [&](unsigned, uint64_t begin, uint64_t end) {
        for (uint64_t i = begin; i < end; ++i) {
            store_val<DstT, Op>(dst[i], static_cast<double>(src[i]));
        }
    });
}

template<class SrcT, class DstT>
void apply_identity(uint64_t elements, const void* src, void* dst, StoreOp op) {
    const SrcT* s = static_cast<const SrcT*>(src);
    DstT* d = static_cast<DstT*>(dst);
    switch (op) {
    case StoreOp::write: apply_identity_op<SrcT, DstT, StoreOp::write>(elements, s, d); break;
    case StoreOp::add: apply_identity_op<SrcT, DstT, StoreOp::add>(elements, s, d); break;
    case StoreOp::square: apply_identity_op<SrcT, DstT, StoreOp::square>(elements, s, d); break;
    case StoreOp::square_add: apply_identity_op<SrcT, DstT, StoreOp::square_add>(elements, s, d); break;
    case StoreOp::square_sqrt: apply_identity_op<SrcT, DstT, StoreOp::square_sqrt>(elements, s, d); break;
    case StoreOp::square_add_sqrt: apply_identity_op<SrcT, DstT, StoreOp::square_add_sqrt>(elements, s, d); break;
    }
}

void dispatch_identity(uint64_t elements, uint32_t src_dtype, uint32_t dst_dtype, const void* src, void* dst,
                       StoreOp op);

// ---- generic integer / bool support (bit-identical to SciPy 1.18.1 arm64) ---

struct BoolByteTag {};

inline void load_to_double_contig(uint32_t src_dtype, const void* src, double* __restrict__ dst, size_t n) {
    switch (src_dtype) {
    case MS_FLOAT64: std::memcpy(dst, src, n * sizeof(double)); break;
    case MS_FLOAT32: to_double(static_cast<const float*>(src), dst, n); break;
    // SciPy NI_ArrayToLineBuffer reads npy_bool as raw uint8_t cast to double.
    case MS_BOOL:
    case MS_UINT8: {
        const uint8_t* __restrict__ p = static_cast<const uint8_t*>(src);
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
    }
    case MS_UINT16: {
        const uint16_t* __restrict__ p = static_cast<const uint16_t*>(src);
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
    }
    case MS_UINT32: {
        const uint32_t* __restrict__ p = static_cast<const uint32_t*>(src);
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
    }
    case MS_UINT64: {
        const uint64_t* __restrict__ p = static_cast<const uint64_t*>(src);
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
    }
    case MS_INT8: {
        const int8_t* __restrict__ p = static_cast<const int8_t*>(src);
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
    }
    case MS_INT16: {
        const int16_t* __restrict__ p = static_cast<const int16_t*>(src);
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
    }
    case MS_INT32: {
        const int32_t* __restrict__ p = static_cast<const int32_t*>(src);
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
    }
    case MS_INT64: {
        const int64_t* __restrict__ p = static_cast<const int64_t*>(src);
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
    }
    default: break;
    }
}

inline double read_scalar_double(uint32_t src_dtype, const void* base, size_t index) noexcept {
    switch (src_dtype) {
    case MS_FLOAT64: return static_cast<const double*>(base)[index];
    case MS_FLOAT32: return static_cast<double>(static_cast<const float*>(base)[index]);
    case MS_BOOL:
    case MS_UINT8: return static_cast<double>(static_cast<const uint8_t*>(base)[index]);
    case MS_UINT16: return static_cast<double>(static_cast<const uint16_t*>(base)[index]);
    case MS_UINT32: return static_cast<double>(static_cast<const uint32_t*>(base)[index]);
    case MS_UINT64: return static_cast<double>(static_cast<const uint64_t*>(base)[index]);
    case MS_INT8: return static_cast<double>(static_cast<const int8_t*>(base)[index]);
    case MS_INT16: return static_cast<double>(static_cast<const int16_t*>(base)[index]);
    case MS_INT32: return static_cast<double>(static_cast<const int32_t*>(base)[index]);
    case MS_INT64: return static_cast<double>(static_cast<const int64_t*>(base)[index]);
    default: return 0.0;
    }
}

inline void load_to_double_strided(uint32_t src_dtype, const void* base, size_t stride, double* __restrict__ dst,
                                   size_t n) {
    if (stride == 1) {
        load_to_double_contig(src_dtype, base, dst, n);
        return;
    }
    for (size_t i = 0; i < n; ++i) dst[i] = read_scalar_double(src_dtype, base, i * stride);
}

template<class DstT, StoreOp Op>
inline void apply_typed_store_op(typename std::conditional_t<std::is_same_v<DstT, BoolByteTag>, uint8_t, DstT>& dst,
                                 typename std::conditional_t<std::is_same_v<DstT, BoolByteTag>, uint8_t, DstT> v) noexcept {
    if constexpr (std::is_same_v<DstT, BoolByteTag>) {
        if constexpr (Op == StoreOp::write) {
            dst = v;
        } else if constexpr (Op == StoreOp::add || Op == StoreOp::square_add || Op == StoreOp::square_add_sqrt) {
            dst = (dst != 0 || v != 0) ? uint8_t{1} : uint8_t{0};
        } else {
            dst = (v != 0) ? uint8_t{1} : uint8_t{0};
        }
    } else if constexpr (std::is_floating_point_v<DstT>) {
        if constexpr (Op == StoreOp::write) {
            dst = v;
        } else if constexpr (Op == StoreOp::add) {
            dst = dst + v;
        } else if constexpr (Op == StoreOp::square) {
            dst = v * v;
        } else if constexpr (Op == StoreOp::square_add) {
            dst = dst + v * v;
        } else if constexpr (Op == StoreOp::square_sqrt) {
            dst = std::sqrt(v * v);
        } else {
            dst = std::sqrt(dst + v * v);
        }
    } else {
        using RawU = std::make_unsigned_t<DstT>;
        using U = std::conditional_t<(sizeof(DstT) < sizeof(unsigned)), unsigned, RawU>;
        const U ud = static_cast<U>(static_cast<RawU>(dst));
        const U uv = static_cast<U>(static_cast<RawU>(v));
        if constexpr (Op == StoreOp::write) {
            dst = v;
        } else if constexpr (Op == StoreOp::add) {
            dst = static_cast<DstT>(static_cast<RawU>(ud + uv));
        } else if constexpr (Op == StoreOp::square) {
            dst = static_cast<DstT>(static_cast<RawU>(uv * uv));
        } else if constexpr (Op == StoreOp::square_add) {
            dst = static_cast<DstT>(static_cast<RawU>(ud + uv * uv));
        } else if constexpr (Op == StoreOp::square_sqrt) {
            const DstT s = static_cast<DstT>(static_cast<RawU>(uv * uv));
            dst = (std::is_signed_v<DstT> && s < 0) ? DstT{0} : static_cast<DstT>(std::sqrt(static_cast<double>(s)));
        } else {
            const DstT s = static_cast<DstT>(static_cast<RawU>(ud + uv * uv));
            dst = (std::is_signed_v<DstT> && s < 0) ? DstT{0} : static_cast<DstT>(std::sqrt(static_cast<double>(s)));
        }
    }
}

template<class DstT, StoreOp Op>
inline void store_row_double_op(const double* __restrict__ acc, void* base, size_t stride, size_t n) noexcept {
    using MemT = std::conditional_t<std::is_same_v<DstT, BoolByteTag>, uint8_t, DstT>;
    MemT* __restrict__ out = static_cast<MemT*>(base);
    if (stride == 1) {
        for (size_t c = 0; c < n; ++c) {
            const MemT v = array::cast_double_ndimage<MemT>(acc[c], /*dense_safe=*/false);
            apply_typed_store_op<DstT, Op>(out[c], v);
        }
    } else {
        for (size_t c = 0; c < n; ++c) {
            const MemT v = array::cast_double_ndimage<MemT>(acc[c], /*dense_safe=*/false);
            apply_typed_store_op<DstT, Op>(out[c * stride], v);
        }
    }
}

template<class DstT>
inline void store_row_double(StoreOp op, const double* __restrict__ acc, void* base, size_t stride, size_t n) {
    switch (op) {
    case StoreOp::write: store_row_double_op<DstT, StoreOp::write>(acc, base, stride, n); break;
    case StoreOp::add: store_row_double_op<DstT, StoreOp::add>(acc, base, stride, n); break;
    case StoreOp::square: store_row_double_op<DstT, StoreOp::square>(acc, base, stride, n); break;
    case StoreOp::square_add: store_row_double_op<DstT, StoreOp::square_add>(acc, base, stride, n); break;
    case StoreOp::square_sqrt: store_row_double_op<DstT, StoreOp::square_sqrt>(acc, base, stride, n); break;
    case StoreOp::square_add_sqrt: store_row_double_op<DstT, StoreOp::square_add_sqrt>(acc, base, stride, n); break;
    }
}

inline void store_from_double_generic(uint32_t dst_dtype, StoreOp op, const double* __restrict__ acc, void* base,
                                      size_t stride, size_t n) {
    switch (dst_dtype) {
    case MS_FLOAT64: store_row_double<double>(op, acc, base, stride, n); break;
    case MS_FLOAT32: store_row_double<float>(op, acc, base, stride, n); break;
    case MS_BOOL: store_row_double<BoolByteTag>(op, acc, base, stride, n); break;
    case MS_UINT8: store_row_double<uint8_t>(op, acc, base, stride, n); break;
    case MS_UINT16: store_row_double<uint16_t>(op, acc, base, stride, n); break;
    case MS_UINT32: store_row_double<uint32_t>(op, acc, base, stride, n); break;
    case MS_UINT64: store_row_double<uint64_t>(op, acc, base, stride, n); break;
    case MS_INT8: store_row_double<int8_t>(op, acc, base, stride, n); break;
    case MS_INT16: store_row_double<int16_t>(op, acc, base, stride, n); break;
    case MS_INT32: store_row_double<int32_t>(op, acc, base, stride, n); break;
    case MS_INT64: store_row_double<int64_t>(op, acc, base, stride, n); break;
    default: break;
    }
}

void correlate_tiles_generic(const Pass& pass, uint32_t src_dtype, uint32_t dst_dtype, const void* src, void* dst,
                             StoreOp op) {
    const size_t taps = static_cast<size_t>(pass.size);
    const size_t inner = static_cast<size_t>(pass.inner);
    const int64_t length = static_cast<int64_t>(pass.length);
    const int64_t before = static_cast<int64_t>(pass.before), after = static_cast<int64_t>(pass.after);
    const size_t src_esize = static_cast<size_t>(array::dtype_size(src_dtype));
    const size_t dst_esize = static_cast<size_t>(array::dtype_size(dst_dtype));
    const size_t tile = std::min(inner, std::clamp<size_t>((65536 / (taps * sizeof(double))) & ~size_t{7}, 32, 2048));
    const uint64_t tiles = (inner + tile - 1) / tile;
    const size_t pitch = (tile + 7) & ~size_t{7};
    const unsigned lanes = lanes_for(pass.outer * pass.length * pass.inner, correlate_operations(pass),
                                     taps * pitch * sizeof(double));
    uint64_t chunks = 1;
    const uint64_t wanted = uint64_t{4} * lanes;
    if (lanes > 1 && pass.outer * tiles < wanted) {
        const uint64_t worthwhile = std::max<uint64_t>(1, pass.length / (8 * pass.size));
        chunks = std::min(worthwhile, (wanted + pass.outer * tiles - 1) / (pass.outer * tiles));
    }
    const int64_t rows = static_cast<int64_t>((pass.length + chunks - 1) / chunks);
    chunks = (pass.length + static_cast<uint64_t>(rows) - 1) / static_cast<uint64_t>(rows);
    const auto make = [&] {
        CorrelateWorkspace ws;
        ws.window.resize(taps * pitch);
        ws.acc.resize(pitch);
        ws.a.resize(taps);
        ws.b.resize(taps);
        return ws;
    };
    for_each_item(pass.outer * chunks * tiles, 1, lanes, make, [&](uint64_t item, CorrelateWorkspace& ws) {
        const uint64_t block = item / (chunks * tiles);
        const int64_t y0 = static_cast<int64_t>((item / tiles) % chunks) * rows;
        const int64_t y1 = std::min(length, y0 + rows);
        const size_t c0 = static_cast<size_t>(item % tiles) * tile;
        const size_t n = std::min(tile, inner - c0);
        const uint8_t* in = static_cast<const uint8_t*>(src) + (block * pass.length * pass.inner + c0) * src_esize;
        uint8_t* out = static_cast<uint8_t*>(dst) + (block * pass.length * pass.inner + c0) * dst_esize;
        double* ring = ws.window.data();
        const auto load = [&](int64_t position, size_t slot) {
            double* row = ring + slot * pitch;
            const int64_t source = position >= 0 && position < length ? position : extend(position, length, pass.mode);
            if (source < 0) {
                std::fill_n(row, n, pass.cval);
                return;
            }
            load_to_double_contig(src_dtype, in + static_cast<size_t>(source) * inner * src_esize, row, n);
            if (source + prefetch_rows < length)
                prefetch(in + static_cast<size_t>(source + prefetch_rows) * inner * src_esize, n * src_esize, 0);
        };
        for (size_t slot = 0; slot + 1 < taps; ++slot) load(y0 - before + static_cast<int64_t>(slot), slot);
        size_t head = 0, incoming = taps - 1;
        for (int64_t y = y0; y < y1; ++y) {
            load(y + after, incoming);
            incoming = incoming + 1 == taps ? 0 : incoming + 1;
            uint8_t* target = out + static_cast<size_t>(y) * inner * dst_esize;
            if (y + prefetch_rows < y1)
                prefetch(target + static_cast<size_t>(prefetch_rows) * inner * dst_esize, n * dst_esize, 1);
            const auto tap = [&](size_t j) -> const double* {
                const size_t slot = head + j;
                return ring + (slot >= taps ? slot - taps : slot) * pitch;
            };
            double* acc = ws.acc.data();
            correlate_row(pass, acc, n, tap, ws.a.data(), ws.b.data());
            store_from_double_generic(dst_dtype, op, acc, target, 1, n);
            head = head + 1 == taps ? 0 : head + 1;
        }
    });
}

void extended_line_generic(const Pass& pass, uint32_t src_dtype, const uint8_t* in, size_t stride, size_t esize,
                           int64_t first, size_t count, double* values) {
    const int64_t length = static_cast<int64_t>(pass.length);
    const auto outside = [&](int64_t position) {
        const int64_t source = extend(position, length, pass.mode);
        return source < 0 ? pass.cval : read_scalar_double(src_dtype, in, static_cast<size_t>(source) * stride);
    };
    const int64_t end = first + static_cast<int64_t>(count);
    int64_t position = first;
    size_t t = 0;
    for (; position < std::min<int64_t>(0, end); ++position) values[t++] = outside(position);
    const int64_t inside_end = std::min(length, end);
    if (position < inside_end) {
        const size_t n = static_cast<size_t>(inside_end - position);
        const uint8_t* source = in + static_cast<size_t>(position) * stride * esize;
        load_to_double_strided(src_dtype, source, stride, values + t, n);
        t += n;
        position = inside_end;
    }
    for (; position < end; ++position) values[t++] = outside(position);
}

void correlate_lines_generic(const Pass& pass, uint32_t src_dtype, uint32_t dst_dtype, const void* src, void* dst,
                             StoreOp op) {
    const size_t taps = static_cast<size_t>(pass.size);
    const size_t stride = static_cast<size_t>(pass.inner);
    const size_t src_esize = static_cast<size_t>(array::dtype_size(src_dtype));
    const size_t dst_esize = static_cast<size_t>(array::dtype_size(dst_dtype));
    const int64_t length = static_cast<int64_t>(pass.length);
    const int64_t before = static_cast<int64_t>(pass.before);
    const uint64_t chunks = (pass.length + line_chunk - 1) / line_chunk;
    const uint64_t lines = pass.outer * pass.inner;
    const size_t widest = static_cast<size_t>(std::min<int64_t>(length, line_chunk));
    const unsigned lanes = lanes_for(lines * pass.length, correlate_operations(pass),
                                     (widest + 3 * taps) * sizeof(double));
    const uint64_t grain = std::max<uint64_t>(1, 16384 / widest);
    const auto make = [&] {
        CorrelateWorkspace ws;
        ws.window.resize(widest + taps - 1);
        ws.acc.resize(widest);
        ws.a.resize(taps);
        ws.b.resize(taps);
        return ws;
    };
    for_each_item(lines * chunks, grain, lanes, make, [&](uint64_t item, CorrelateWorkspace& ws) {
        const uint64_t line = item / chunks;
        const int64_t begin = static_cast<int64_t>(item % chunks) * line_chunk;
        const size_t n = static_cast<size_t>(std::min(length, begin + line_chunk) - begin);
        const uint64_t offset = (line / pass.inner) * pass.length * pass.inner + line % pass.inner;
        const double* window = ws.window.data();
        extended_line_generic(pass, src_dtype, static_cast<const uint8_t*>(src) + offset * src_esize, stride,
                              src_esize, begin - before, n + taps - 1, ws.window.data());
        const auto tap = [&](size_t j) { return window + j; };
        uint8_t* out = static_cast<uint8_t*>(dst) + (offset + static_cast<size_t>(begin) * stride) * dst_esize;
        double* acc = ws.acc.data();
        correlate_row(pass, acc, n, tap, ws.a.data(), ws.b.data());
        store_from_double_generic(dst_dtype, op, acc, out, stride, n);
    });
}

void uniform_tiles_generic(const Pass& pass, uint32_t src_dtype, uint32_t dst_dtype, const void* src, void* dst,
                           StoreOp op) {
    const size_t inner = static_cast<size_t>(pass.inner);
    const int64_t length = static_cast<int64_t>(pass.length), taps = static_cast<int64_t>(pass.size);
    const int64_t before = static_cast<int64_t>(pass.before), after = static_cast<int64_t>(pass.after);
    const size_t src_esize = static_cast<size_t>(array::dtype_size(src_dtype));
    const size_t dst_esize = static_cast<size_t>(array::dtype_size(dst_dtype));
    const unsigned lanes = lanes_for(pass.outer * pass.length * pass.inner, 2);
    size_t tile = std::min<size_t>(inner, 2048);
    if (lanes > 1) {
        const uint64_t initial_tiles = (inner + tile - 1) / tile;
        if (pass.outer * initial_tiles < uint64_t{4} * lanes) {
            const uint64_t wanted_tiles = (uint64_t{4} * lanes + pass.outer - 1) / pass.outer;
            const size_t scaled = (inner / static_cast<size_t>(wanted_tiles)) & ~size_t{7};
            tile = std::clamp<size_t>(scaled, std::min<size_t>(inner, 32), 2048);
        }
    }
    const uint64_t tiles = (inner + tile - 1) / tile;
    const double size = static_cast<double>(pass.size);
    const int64_t inside_begin = std::min(length, before + 1);
    const int64_t inside_end = std::max(inside_begin, length - after);
    struct GenUniformWS {
        std::vector<double> sum, entering, leaving, constant, acc;
    };
    const auto make = [&] {
        GenUniformWS ws;
        ws.sum.resize(tile);
        ws.acc.resize(tile);
        ws.constant.assign(tile, pass.cval);
        ws.entering.resize(tile);
        ws.leaving.resize(tile);
        return ws;
    };
    for_each_item(pass.outer * tiles, 1, lanes, make, [&](uint64_t item, GenUniformWS& ws) {
        const size_t c0 = static_cast<size_t>(item % tiles) * tile;
        const size_t n = std::min(tile, inner - c0);
        const uint8_t* in = static_cast<const uint8_t*>(src) + ((item / tiles) * pass.length * pass.inner + c0) * src_esize;
        uint8_t* out = static_cast<uint8_t*>(dst) + ((item / tiles) * pass.length * pass.inner + c0) * dst_esize;
        const auto row = [&](int64_t position, std::vector<double>& converted) -> const double* {
            const int64_t source = position >= 0 && position < length ? position : extend(position, length, pass.mode);
            if (source < 0) return ws.constant.data();
            load_to_double_contig(src_dtype, in + static_cast<size_t>(source) * inner * src_esize, converted.data(), n);
            return converted.data();
        };
        double* __restrict__ sum = ws.sum.data();
        double* __restrict__ acc = ws.acc.data();
        std::fill_n(sum, n, 0.0);
        for (int64_t k = 0; k < taps; ++k) {
            const double* values = row(k - before, ws.entering);
            for (size_t c = 0; c < n; ++c) sum[c] = sum[c] + values[c];
        }
        for (size_t c = 0; c < n; ++c) acc[c] = sum[c] / size;
        store_from_double_generic(dst_dtype, op, acc, out, 1, n);
        auto step_row = [&](int64_t y, const double* __restrict__ entering, const double* __restrict__ leaving) {
            uint8_t* target = out + static_cast<size_t>(y) * inner * dst_esize;
            for (size_t c = 0; c < n; ++c) {
                const double s = sum[c] + (entering[c] - leaving[c]);
                sum[c] = s;
                acc[c] = s / size;
            }
            store_from_double_generic(dst_dtype, op, acc, target, 1, n);
        };
        int64_t y = 1;
        for (; y < inside_begin; ++y) {
            step_row(y, row(y + after, ws.entering), row(y - 1 - before, ws.leaving));
        }
        for (; y < inside_end; ++y) {
            if (y + after + prefetch_rows < length)
                prefetch(in + static_cast<size_t>(y + after + prefetch_rows) * inner * src_esize, n * src_esize, 0);
            if (y + prefetch_rows < length)
                prefetch(out + static_cast<size_t>(y + prefetch_rows) * inner * dst_esize, n * dst_esize, 1);
            load_to_double_contig(src_dtype, in + static_cast<size_t>(y + after) * inner * src_esize, ws.entering.data(), n);
            load_to_double_contig(src_dtype, in + static_cast<size_t>(y - 1 - before) * inner * src_esize, ws.leaving.data(), n);
            step_row(y, ws.entering.data(), ws.leaving.data());
        }
        for (; y < length; ++y) {
            step_row(y, row(y + after, ws.entering), row(y - 1 - before, ws.leaving));
        }
    });
}

void uniform_lines_generic(const Pass& pass, uint32_t src_dtype, uint32_t dst_dtype, const void* src, void* dst,
                           StoreOp op) {
    const size_t stride = static_cast<size_t>(pass.inner);
    const size_t src_esize = static_cast<size_t>(array::dtype_size(src_dtype));
    const size_t dst_esize = static_cast<size_t>(array::dtype_size(dst_dtype));
    const int64_t length = static_cast<int64_t>(pass.length), taps = static_cast<int64_t>(pass.size);
    const int64_t before = static_cast<int64_t>(pass.before), after = static_cast<int64_t>(pass.after);
    const uint64_t lines = pass.outer * pass.inner;
    const unsigned lanes = lanes_for(lines * pass.length, 2);
    const double size = static_cast<double>(pass.size);
    const int64_t inside_begin = std::min(length, before + 1);
    const int64_t inside_end = std::max(inside_begin, length - after);

    auto run_scalar_line = [&](uint64_t line) {
        const uint64_t offset = (line / pass.inner) * pass.length * pass.inner + line % pass.inner;
        const uint8_t* in = static_cast<const uint8_t*>(src) + offset * src_esize;
        uint8_t* out = static_cast<uint8_t*>(dst) + offset * dst_esize;
        const auto at = [&](int64_t position) -> double {
            const int64_t source = position >= 0 && position < length ? position : extend(position, length, pass.mode);
            return source < 0 ? pass.cval : read_scalar_double(src_dtype, in, static_cast<size_t>(source) * stride);
        };
        double sum = 0.0;
        for (int64_t k = 0; k < taps; ++k) sum = sum + at(k - before);
        double q = sum / size;
        store_from_double_generic(dst_dtype, op, &q, out, 1, 1);
        int64_t y = 1;
        for (; y < inside_begin; ++y) {
            sum = sum + (at(y + after) - at(y - 1 - before));
            q = sum / size;
            store_from_double_generic(dst_dtype, op, &q, out + static_cast<size_t>(y) * stride * dst_esize, 1, 1);
        }
        for (; y < inside_end; ++y) {
            sum = sum + (read_scalar_double(src_dtype, in, static_cast<size_t>(y + after) * stride) -
                         read_scalar_double(src_dtype, in, static_cast<size_t>(y - 1 - before) * stride));
            q = sum / size;
            store_from_double_generic(dst_dtype, op, &q, out + static_cast<size_t>(y) * stride * dst_esize, 1, 1);
        }
        for (; y < length; ++y) {
            sum = sum + (at(y + after) - at(y - 1 - before));
            q = sum / size;
            store_from_double_generic(dst_dtype, op, &q, out + static_cast<size_t>(y) * stride * dst_esize, 1, 1);
        }
    };

#if defined(__aarch64__) && defined(__ARM_NEON)
    if (stride == 1 && lines >= 4) {
        const uint64_t bundles4 = lines / 4;
        const uint64_t bundle_grain = std::max<uint64_t>(1, 4096 / pass.length);
        const float64x2_t vsize = vdupq_n_f64(size);
        for_each_item(bundles4, bundle_grain, lanes, [] { return 0; }, [&](uint64_t bundle, int&) {
            const uint64_t line_begin = bundle * 4;
            const uint8_t* ins[4];
            uint8_t* outs[4];
            for (size_t m = 0; m < 4; ++m) {
                const size_t off = static_cast<size_t>(line_begin + m) * static_cast<size_t>(pass.length);
                ins[m] = static_cast<const uint8_t*>(src) + off * src_esize;
                outs[m] = static_cast<uint8_t*>(dst) + off * dst_esize;
            }
            auto load1_pair = [&](size_t p, int64_t source) -> float64x2_t {
                if (source < 0) return vdupq_n_f64(pass.cval);
                const size_t s = static_cast<size_t>(source);
                return vsetq_lane_f64(read_scalar_double(src_dtype, ins[p * 2 + 1], s),
                                      vdupq_n_f64(read_scalar_double(src_dtype, ins[p * 2], s)), 1);
            };
            auto store1_pair = [&](size_t p, int64_t y, float64x2_t q) {
                const double v0 = vgetq_lane_f64(q, 0), v1 = vgetq_lane_f64(q, 1);
                const size_t u = static_cast<size_t>(y);
                store_from_double_generic(dst_dtype, op, &v0, outs[p * 2] + u * dst_esize, 1, 1);
                store_from_double_generic(dst_dtype, op, &v1, outs[p * 2 + 1] + u * dst_esize, 1, 1);
            };
            float64x2_t sum[2] = {vdupq_n_f64(0.0), vdupq_n_f64(0.0)};
            for (int64_t k = 0; k < taps; ++k) {
                const int64_t pos = k - before;
                const int64_t source = pos >= 0 && pos < length ? pos : extend(pos, length, pass.mode);
                sum[0] = vaddq_f64(sum[0], load1_pair(0, source));
                sum[1] = vaddq_f64(sum[1], load1_pair(1, source));
            }
            store1_pair(0, 0, vdivq_f64(sum[0], vsize));
            store1_pair(1, 0, vdivq_f64(sum[1], vsize));
            int64_t y = 1;
            for (; y < inside_begin; ++y) {
                const int64_t pe = y + after, pl = y - 1 - before;
                const int64_t se = pe >= 0 && pe < length ? pe : extend(pe, length, pass.mode);
                const int64_t sl = pl >= 0 && pl < length ? pl : extend(pl, length, pass.mode);
                for (size_t p = 0; p < 2; ++p) {
                    sum[p] = vaddq_f64(sum[p], vsubq_f64(load1_pair(p, se), load1_pair(p, sl)));
                    store1_pair(p, y, vdivq_f64(sum[p], vsize));
                }
            }
            constexpr int64_t chunk = 64;
            double eb[4][chunk], lb[4][chunk], ob[4][chunk];
            for (; y < inside_end;) {
                const int64_t cnt = std::min(chunk, inside_end - y);
                const size_t u = static_cast<size_t>(y);
                const size_t oe = static_cast<size_t>(y + after);
                const size_t ol = static_cast<size_t>(y - 1 - before);
                for (size_t m = 0; m < 4; ++m) {
                    load_to_double_contig(src_dtype, ins[m] + oe * src_esize, eb[m], static_cast<size_t>(cnt));
                    load_to_double_contig(src_dtype, ins[m] + ol * src_esize, lb[m], static_cast<size_t>(cnt));
                }
                int64_t t = 0;
                for (; t + 1 < cnt; t += 2) {
                    for (size_t p = 0; p < 2; ++p) {
                        const float64x2_t ea = vld1q_f64(eb[p * 2] + t), ebv = vld1q_f64(eb[p * 2 + 1] + t);
                        const float64x2_t la = vld1q_f64(lb[p * 2] + t), lbv = vld1q_f64(lb[p * 2 + 1] + t);
                        sum[p] = vaddq_f64(sum[p], vsubq_f64(vtrn1q_f64(ea, ebv), vtrn1q_f64(la, lbv)));
                        const float64x2_t q0 = vdivq_f64(sum[p], vsize);
                        sum[p] = vaddq_f64(sum[p], vsubq_f64(vtrn2q_f64(ea, ebv), vtrn2q_f64(la, lbv)));
                        const float64x2_t q1 = vdivq_f64(sum[p], vsize);
                        vst1q_f64(ob[p * 2] + t, vtrn1q_f64(q0, q1));
                        vst1q_f64(ob[p * 2 + 1] + t, vtrn2q_f64(q0, q1));
                    }
                }
                for (; t < cnt; ++t) {
                    for (size_t p = 0; p < 2; ++p) {
                        const float64x2_t e = vsetq_lane_f64(eb[p * 2 + 1][t], vdupq_n_f64(eb[p * 2][t]), 1);
                        const float64x2_t l = vsetq_lane_f64(lb[p * 2 + 1][t], vdupq_n_f64(lb[p * 2][t]), 1);
                        sum[p] = vaddq_f64(sum[p], vsubq_f64(e, l));
                        const float64x2_t q = vdivq_f64(sum[p], vsize);
                        ob[p * 2][t] = vgetq_lane_f64(q, 0);
                        ob[p * 2 + 1][t] = vgetq_lane_f64(q, 1);
                    }
                }
                for (size_t m = 0; m < 4; ++m) {
                    store_from_double_generic(dst_dtype, op, ob[m], outs[m] + u * dst_esize, 1,
                                              static_cast<size_t>(cnt));
                }
                y += cnt;
            }
            for (; y < length; ++y) {
                const int64_t pe = y + after, pl = y - 1 - before;
                const int64_t se = pe >= 0 && pe < length ? pe : extend(pe, length, pass.mode);
                const int64_t sl = pl >= 0 && pl < length ? pl : extend(pl, length, pass.mode);
                for (size_t p = 0; p < 2; ++p) {
                    sum[p] = vaddq_f64(sum[p], vsubq_f64(load1_pair(p, se), load1_pair(p, sl)));
                    store1_pair(p, y, vdivq_f64(sum[p], vsize));
                }
            }
        });
        for (uint64_t line = bundles4 * 4; line < lines; ++line) run_scalar_line(line);
        return;
    }
#endif

    const uint64_t grain = std::max<uint64_t>(1, 16384 / pass.length);
    for_each_item(lines, grain, lanes, [] { return 0; }, [&](uint64_t line, int&) {
        run_scalar_line(line);
    });
}

template<class DstT, StoreOp Op>
void apply_identity_typed_op(const void* src_typed, void* dst_raw, uint64_t count) noexcept {
    using MemT = std::conditional_t<std::is_same_v<DstT, BoolByteTag>, uint8_t, DstT>;
    const MemT* __restrict__ s = static_cast<const MemT*>(src_typed);
    MemT* __restrict__ d = static_cast<MemT*>(dst_raw);
    for (uint64_t i = 0; i < count; ++i) {
        apply_typed_store_op<DstT, Op>(d[i], s[i]);
    }
}

template<class DstT>
void apply_identity_typed(const void* src_typed, void* dst_raw, uint64_t count, StoreOp op) {
    switch (op) {
    case StoreOp::write: apply_identity_typed_op<DstT, StoreOp::write>(src_typed, dst_raw, count); break;
    case StoreOp::add: apply_identity_typed_op<DstT, StoreOp::add>(src_typed, dst_raw, count); break;
    case StoreOp::square: apply_identity_typed_op<DstT, StoreOp::square>(src_typed, dst_raw, count); break;
    case StoreOp::square_add: apply_identity_typed_op<DstT, StoreOp::square_add>(src_typed, dst_raw, count); break;
    case StoreOp::square_sqrt: apply_identity_typed_op<DstT, StoreOp::square_sqrt>(src_typed, dst_raw, count); break;
    case StoreOp::square_add_sqrt:
        apply_identity_typed_op<DstT, StoreOp::square_add_sqrt>(src_typed, dst_raw, count);
        break;
    }
}

void apply_identity_generic(uint64_t elements, uint32_t src_dtype, uint32_t dst_dtype, const void* src, void* dst,
                            StoreOp op) {
    if (op == StoreOp::write) {
        array::convert(src, src_dtype, dst, dst_dtype, elements);
        return;
    }
    const uint64_t src_esize = array::dtype_size(src_dtype);
    const uint64_t dst_esize = array::dtype_size(dst_dtype);
    parallel::parallel_for(elements, min_work_per_lane, [&](unsigned, uint64_t begin, uint64_t end) {
        constexpr uint64_t chunk_elems = 1024;
        uint64_t aligned_buf[chunk_elems];
        for (uint64_t pos = begin; pos < end; pos += chunk_elems) {
            const uint64_t n = std::min(chunk_elems, end - pos);
            const void* src_slice = static_cast<const uint8_t*>(src) + pos * src_esize;
            void* dst_slice = static_cast<uint8_t*>(dst) + pos * dst_esize;
            array::convert(src_slice, src_dtype, aligned_buf, dst_dtype, n);
            switch (dst_dtype) {
            case MS_FLOAT64: apply_identity_typed<double>(aligned_buf, dst_slice, n, op); break;
            case MS_FLOAT32: apply_identity_typed<float>(aligned_buf, dst_slice, n, op); break;
            case MS_BOOL: apply_identity_typed<BoolByteTag>(aligned_buf, dst_slice, n, op); break;
            case MS_UINT8: apply_identity_typed<uint8_t>(aligned_buf, dst_slice, n, op); break;
            case MS_UINT16: apply_identity_typed<uint16_t>(aligned_buf, dst_slice, n, op); break;
            case MS_UINT32: apply_identity_typed<uint32_t>(aligned_buf, dst_slice, n, op); break;
            case MS_UINT64: apply_identity_typed<uint64_t>(aligned_buf, dst_slice, n, op); break;
            case MS_INT8: apply_identity_typed<int8_t>(aligned_buf, dst_slice, n, op); break;
            case MS_INT16: apply_identity_typed<int16_t>(aligned_buf, dst_slice, n, op); break;
            case MS_INT32: apply_identity_typed<int32_t>(aligned_buf, dst_slice, n, op); break;
            case MS_INT64: apply_identity_typed<int64_t>(aligned_buf, dst_slice, n, op); break;
            default: break;
            }
        }
    });
}

void dispatch_pass(const Pass& pass, uint32_t src_dtype, uint32_t dst_dtype, const void* src, void* dst, StoreOp op) {
    if (src_dtype == MS_FLOAT64 && dst_dtype == MS_FLOAT64) {
        run_pass_with_op<double, double>(pass, src, dst, op);
    } else if (src_dtype == MS_FLOAT32 && dst_dtype == MS_FLOAT32) {
        run_pass_with_op<float, float>(pass, src, dst, op);
    } else if (src_dtype == MS_FLOAT32 && dst_dtype == MS_FLOAT64) {
        run_pass_with_op<float, double>(pass, src, dst, op);
    } else if (src_dtype == MS_FLOAT64 && dst_dtype == MS_FLOAT32) {
        run_pass_with_op<double, float>(pass, src, dst, op);
    } else {
        const bool tiles = pass.inner >= min_tile_columns;
        if (pass.kind == MS_NDIMAGE_UNIFORM) {
            if (tiles) uniform_tiles_generic(pass, src_dtype, dst_dtype, src, dst, op);
            else uniform_lines_generic(pass, src_dtype, dst_dtype, src, dst, op);
        } else {
            if (tiles && pass.size <= max_ring_taps) correlate_tiles_generic(pass, src_dtype, dst_dtype, src, dst, op);
            else correlate_lines_generic(pass, src_dtype, dst_dtype, src, dst, op);
        }
    }
}

void dispatch_identity(uint64_t elements, uint32_t src_dtype, uint32_t dst_dtype, const void* src, void* dst,
                       StoreOp op) {
    if (src_dtype == MS_FLOAT64 && dst_dtype == MS_FLOAT64) {
        apply_identity<double, double>(elements, src, dst, op);
    } else if (src_dtype == MS_FLOAT32 && dst_dtype == MS_FLOAT32) {
        apply_identity<float, float>(elements, src, dst, op);
    } else if (src_dtype == MS_FLOAT32 && dst_dtype == MS_FLOAT64) {
        apply_identity<float, double>(elements, src, dst, op);
    } else if (src_dtype == MS_FLOAT64 && dst_dtype == MS_FLOAT32) {
        apply_identity<double, float>(elements, src, dst, op);
    } else {
        apply_identity_generic(elements, src_dtype, dst_dtype, src, dst, op);
    }
}
} // namespace

BufferPtr run_cpu(ms_context* context, const Plan& plan, const void* input) {
    const uint32_t out_dtype = plan.out_dtype ? plan.out_dtype : plan.dtype;
    uint64_t bytes = 0;
    if (ms_checked_mul(plan.shape.elements, array::dtype_size(out_dtype), &bytes) != MS_OK)
        throw Error(MS_EOVERFLOW, "output array byte size overflows");
    const size_t count = plan.passes.size();
    BufferPtr result = array::allocate(context, bytes);
    if (count == 0) {
        if (plan.shape.elements) {
            dispatch_identity(plan.shape.elements, plan.dtype, out_dtype, input, mutable_data(result.get()),
                              StoreOp::write);
        }
        return result;
    }
    BufferPtr scratch = count > 1 ? array::allocate(context, bytes) : BufferPtr();
    const void* src = input;
    uint32_t src_dtype = plan.dtype;
    for (size_t i = 0; i < count; ++i) {
        // Passes alternate between the two buffers so that the last one writes the result.
        void* dst = mutable_data((count - 1 - i) % 2 == 0 ? result.get() : scratch.get());
        dispatch_pass(plan.passes[i], src_dtype, out_dtype, src, dst, StoreOp::write);
        src = dst;
        src_dtype = out_dtype;
    }
    return result;
}

BufferPtr run_cpu_composite(ms_context* context, const CompositePlan& plan, const void* input) {
    const uint64_t elements = plan.shape.elements;
    uint64_t out_bytes = 0;
    if (ms_checked_mul(elements, array::dtype_size(plan.out_dtype), &out_bytes) != MS_OK)
        throw Error(MS_EOVERFLOW, "output array byte size overflows");
    BufferPtr result = array::allocate(context, out_bytes);
    if (!elements) return result;

    size_t max_passes = 0;
    for (const auto& chain : plan.chains) max_passes = std::max(max_passes, chain.size());

    // Chain 0 can alternate with result and scratch[0]; chains >= 1 keep result
    // intact and alternate between scratch[0] and scratch[1] before fusing into result.
    BufferPtr scratch[2];
    if (max_passes >= 2) scratch[0] = array::allocate(context, out_bytes);
    if (plan.chains.size() >= 2 && max_passes >= 3) scratch[1] = array::allocate(context, out_bytes);

    void* result_ptr = mutable_data(result.get());
    const size_t num_chains = plan.chains.size();
    const bool is_grad = plan.combine_op == MS_NDIMAGE_COMBINE_SQRT_SUM_SQ;

    for (size_t k = 0; k < num_chains; ++k) {
        const auto& chain = plan.chains[k];
        const bool is_first = (k == 0);
        const bool is_last = (k + 1 == num_chains);

        StoreOp final_op = StoreOp::write;
        if (is_grad) {
            if (is_first && is_last) final_op = StoreOp::square_sqrt;
            else if (is_first) final_op = StoreOp::square;
            else if (is_last) final_op = StoreOp::square_add_sqrt;
            else final_op = StoreOp::square_add;
        } else {
            final_op = is_first ? StoreOp::write : StoreOp::add;
        }

        if (chain.empty()) {
            dispatch_identity(elements, plan.in_dtype, plan.out_dtype, input, result_ptr, final_op);
            continue;
        }

        const size_t count = chain.size();
        const void* src = input;
        uint32_t src_dtype = plan.in_dtype;
        for (size_t i = 0; i < count; ++i) {
            const bool last_pass = (i + 1 == count);
            void* dst;
            if (last_pass) {
                dst = result_ptr;
            } else if (is_first) {
                dst = mutable_data((count - 1 - i) % 2 == 0 ? result.get() : scratch[0].get());
            } else {
                // Intermediate passes for k >= 1 alternate in scratch[0], scratch[1], ending in
                // scratch[(count - 2 - i) % 2] before the final pass reads it and writes result_ptr.
                dst = mutable_data(scratch[(count - 2 - i) % 2].get());
            }
            dispatch_pass(chain[i], src_dtype, plan.out_dtype, src, dst, last_pass ? final_op : StoreOp::write);
            src = dst;
            src_dtype = plan.out_dtype;
        }
    }
    return result;
}
} // namespace ms::detail::ndimage
