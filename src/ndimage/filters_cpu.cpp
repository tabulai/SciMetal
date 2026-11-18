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
#include <cstring>
#include <type_traits>
#include <vector>

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

// Tiles: for one block of the array, columns [c0, c0 + n) and rows [y0, y1)
// along the axis. A ring holds the `size` extended rows the current output row
// reads, as doubles: one contiguous, L1-resident copy of the input instead of
// `size` row segments a whole row apart (which evict each other when the row
// length is a power of two), and float32 rows are widened once, not per tap.
template<class T>
void correlate_tiles(const Pass& pass, const T* src, T* dst) {
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
        if (!std::is_same_v<T, double>) ws.acc.resize(pitch);
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
        const T* in = src + block * pass.length * pass.inner + c0;
        T* out = dst + block * pass.length * pass.inner + c0;
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
                prefetch(in + static_cast<size_t>(source + prefetch_rows) * inner, n * sizeof(T), 0);
        };
        // Slot s holds extension row y0 - before + s (mod size).
        for (size_t slot = 0; slot + 1 < taps; ++slot) load(y0 - before + static_cast<int64_t>(slot), slot);
        size_t head = 0, incoming = taps - 1;
        for (int64_t y = y0; y < y1; ++y) {
            load(y + after, incoming);
            incoming = incoming + 1 == taps ? 0 : incoming + 1;
            T* target = out + static_cast<size_t>(y) * inner;
            if (y + prefetch_rows < length) prefetch(target + static_cast<size_t>(prefetch_rows) * inner, n * sizeof(T), 1);
            const auto tap = [&](size_t j) -> const double* {
                const size_t slot = head + j;
                return ring + (slot >= taps ? slot - taps : slot) * pitch;
            };
            if constexpr (std::is_same_v<T, double>) {
                correlate_row(pass, target, n, tap, ws.a.data(), ws.b.data());
            } else {
                double* acc = ws.acc.data();
                correlate_row(pass, acc, n, tap, ws.a.data(), ws.b.data());
                for (size_t c = 0; c < n; ++c) target[c] = static_cast<T>(acc[c]);
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
template<class T>
void correlate_lines(const Pass& pass, const T* src, T* dst) {
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
        T* out = dst + offset + static_cast<size_t>(begin) * stride;
        if constexpr (std::is_same_v<T, double>) {
            if (stride == 1) {
                correlate_row(pass, out, n, tap, ws.a.data(), ws.b.data());
                return;
            }
        }
        double* acc = ws.acc.data();
        correlate_row(pass, acc, n, tap, ws.a.data(), ws.b.data());
        for (size_t i = 0; i < n; ++i) out[i * stride] = static_cast<T>(acc[i]);
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

template<class T>
void uniform_tiles(const Pass& pass, const T* src, T* dst) {
    const size_t inner = static_cast<size_t>(pass.inner);
    const int64_t length = static_cast<int64_t>(pass.length), taps = static_cast<int64_t>(pass.size);
    const int64_t before = static_cast<int64_t>(pass.before), after = static_cast<int64_t>(pass.after);
    const size_t tile = std::min<size_t>(inner, 2048);
    const uint64_t tiles = (inner + tile - 1) / tile;
    const unsigned lanes = lanes_for(pass.outer * pass.length * pass.inner, 2);
    const double size = static_cast<double>(pass.size);
    const auto make = [&] {
        UniformWorkspace ws;
        ws.sum.resize(tile);
        ws.constant.assign(tile, pass.cval);
        if (!std::is_same_v<T, double>) {
            ws.entering.resize(tile);
            ws.leaving.resize(tile);
        }
        return ws;
    };
    for_each_item(pass.outer * tiles, 1, lanes, make, [&](uint64_t item, UniformWorkspace& ws) {
        const size_t c0 = static_cast<size_t>(item % tiles) * tile;
        const size_t n = std::min(tile, inner - c0);
        const T* in = src + (item / tiles) * pass.length * pass.inner + c0;
        T* out = dst + (item / tiles) * pass.length * pass.inner + c0;
        // The n values of one extended row, as doubles.
        const auto row = [&](int64_t position, std::vector<double>& converted) -> const double* {
            const int64_t source = position >= 0 && position < length ? position : extend(position, length, pass.mode);
            if (source < 0) return ws.constant.data();
            const T* values = in + static_cast<size_t>(source) * inner;
            if constexpr (std::is_same_v<T, double>) {
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
        for (size_t c = 0; c < n; ++c) out[c] = static_cast<T>(sum[c] / size);
        for (int64_t y = 1; y < length; ++y) {
            const double* entering = row(y + after, ws.entering);
            const double* leaving = row(y - 1 - before, ws.leaving);
            T* __restrict target = out + static_cast<size_t>(y) * inner;
            for (size_t c = 0; c < n; ++c) {
                const double s = sum[c] + (entering[c] - leaving[c]);
                sum[c] = s;
                target[c] = static_cast<T>(s / size);
            }
        }
    });
}

template<class T>
void uniform_lines(const Pass& pass, const T* src, T* dst) {
    const size_t stride = static_cast<size_t>(pass.inner);
    const int64_t length = static_cast<int64_t>(pass.length), taps = static_cast<int64_t>(pass.size);
    const int64_t before = static_cast<int64_t>(pass.before), after = static_cast<int64_t>(pass.after);
    const uint64_t lines = pass.outer * pass.inner;
    const unsigned lanes = lanes_for(lines * pass.length, 2);
    const uint64_t grain = std::max<uint64_t>(1, 16384 / pass.length);
    const double size = static_cast<double>(pass.size);
    // Outputs whose entering and leaving values both lie inside the line.
    const int64_t inside_begin = std::min(length, before + 1);
    const int64_t inside_end = std::max(inside_begin, length - after);
    for_each_item(lines, grain, lanes, [] { return 0; }, [&](uint64_t line, int&) {
        const uint64_t offset = (line / pass.inner) * pass.length * pass.inner + line % pass.inner;
        const T* in = src + offset;
        T* out = dst + offset;
        const auto at = [&](int64_t position) -> double {
            const int64_t source = position >= 0 && position < length ? position : extend(position, length, pass.mode);
            return source < 0 ? pass.cval : static_cast<double>(in[static_cast<size_t>(source) * stride]);
        };
        double sum = 0.0;
        for (int64_t k = 0; k < taps; ++k) sum = sum + at(k - before);
        out[0] = static_cast<T>(sum / size);
        int64_t y = 1;
        for (; y < inside_begin; ++y) {
            sum = sum + (at(y + after) - at(y - 1 - before));
            out[static_cast<size_t>(y) * stride] = static_cast<T>(sum / size);
        }
        for (; y < inside_end; ++y) {
            sum = sum + (static_cast<double>(in[static_cast<size_t>(y + after) * stride]) -
                         static_cast<double>(in[static_cast<size_t>(y - 1 - before) * stride]));
            out[static_cast<size_t>(y) * stride] = static_cast<T>(sum / size);
        }
        for (; y < length; ++y) {
            sum = sum + (at(y + after) - at(y - 1 - before));
            out[static_cast<size_t>(y) * stride] = static_cast<T>(sum / size);
        }
    });
}

template<class T>
void run_pass(const Pass& pass, const T* src, T* dst) {
    const bool tiles = pass.inner >= min_tile_columns;
    if (pass.kind == MS_NDIMAGE_UNIFORM) {
        if (tiles) uniform_tiles(pass, src, dst);
        else uniform_lines(pass, src, dst);
    } else {
        if (tiles && pass.size <= max_ring_taps) correlate_tiles(pass, src, dst);
        else correlate_lines(pass, src, dst);
    }
}
} // namespace

BufferPtr run_cpu(ms_context* context, const Plan& plan, const void* input) {
    const uint64_t bytes = plan.shape.elements * array::dtype_size(plan.dtype);
    const size_t count = plan.passes.size();
    BufferPtr result = array::allocate(context, bytes);
    BufferPtr scratch = count > 1 ? array::allocate(context, bytes) : BufferPtr();
    const void* src = input;
    for (size_t i = 0; i < count; ++i) {
        // Passes alternate between the two buffers so that the last one writes the result.
        void* dst = mutable_data((count - 1 - i) % 2 == 0 ? result.get() : scratch.get());
        if (plan.dtype == MS_FLOAT64)
            run_pass(plan.passes[i], static_cast<const double*>(src), static_cast<double*>(dst));
        else
            run_pass(plan.passes[i], static_cast<const float*>(src), static_cast<float*>(dst));
        src = dst;
    }
    return result;
}
} // namespace ms::detail::ndimage
