// GPU engine of the separable filters: every pass in float32, one dispatch per
// pass, all in one command buffer.
//
// The input is narrowed (or copied) once into pooled storage on the
// performance cores; passes then alternate between two pooled float32
// buffers. Only the last pass differs for float64 arrays: its kernel stores
// the widened result directly, so no pass ever reads a float64 and no separate
// conversion pass runs (probes/README.md, rules 1 and 2).
#include "filters.hpp"
#include "ndimage_kernels.h"
#include "runtime/metal_compute.hpp"

#include <algorithm>
#include <atomic>
#include <vector>

namespace ms::detail::ndimage {
namespace {
// One kernel per summation order (ndimage_kernels.h) and destination type.
constexpr const char* kernels[2][3] = {
    {"ms_ndimage_correlate1d_any_f32", "ms_ndimage_correlate1d_even_f32", "ms_ndimage_correlate1d_odd_f32"},
    {"ms_ndimage_correlate1d_any_f64", "ms_ndimage_correlate1d_even_f64", "ms_ndimage_correlate1d_odd_f64"},
};

// The largest power-of-two threadgroup a kernel's pipeline accepts, at most
// 1024 (shader validation lowers a pipeline's capacity to 512).
uint32_t threadgroup_capacity(const char* kernel, std::atomic<uint32_t>& cached) {
    if (const uint32_t known = cached.load(std::memory_order_relaxed)) return known;
    const uint32_t limit = gpu::max_threads_per_group(kernel);
    uint32_t capacity = 1024;
    while (capacity > 1 && capacity > limit) capacity >>= 1;
    cached.store(capacity, std::memory_order_relaxed);
    return capacity;
}

// The smallest power of two that covers `extent`, at most `limit` (a power of two).
uint32_t covering(uint64_t extent, uint32_t limit) {
    uint32_t threads = 1;
    while (threads < extent && threads < limit) threads <<= 1;
    return threads;
}

// The dispatch grid of one pass and the matching kernel extents.
//
// Probe E4 measured the threadgroup shapes on 4096 x 4096: along the
// contiguous axis flat groups of 256-1024 threads are best (0.85 ms against
// 1.37 ms for 32 x 32); along a strided axis 32 x 32 is best (0.86 ms against
// 1.45 ms flat, and 7.4 ms for one column of 1024 rows). What those shapes
// share is that a group's reads cover a compact run of whole cache lines.
// Extents smaller than the measured shapes keep that property by growing the
// group along the next slower direction instead, so no group is mostly idle.
gpu::Grid grid_for(const Pass& pass, uint32_t capacity, MSNdimageCorrelateParams& params) {
    const uint64_t extent[3] = {pass.inner == 1 ? pass.length : pass.inner,
                                pass.inner == 1 ? pass.outer : pass.length,
                                pass.inner == 1 ? 1 : pass.outer};
    gpu::Grid grid;
    if (pass.inner == 1) {
        grid.threads[0] = covering(extent[0], capacity);
        grid.threads[1] = covering(extent[1], capacity / grid.threads[0]);
    } else {
        grid.threads[0] = covering(extent[0], std::min<uint32_t>(32, capacity));
        grid.threads[1] = covering(extent[1], capacity / grid.threads[0]);
        // A short axis leaves room: widen the group along the rows instead.
        grid.threads[0] = covering(extent[0], capacity / grid.threads[1]);
        grid.threads[2] = covering(extent[2], capacity / (grid.threads[0] * grid.threads[1]));
    }
    for (int axis = 0; axis < 3; ++axis) {
        grid.groups[axis] = (extent[axis] + grid.threads[axis] - 1) / grid.threads[axis];
        params.extent[axis] = static_cast<uint32_t>(extent[axis]);
    }
    params.axis_in_y = pass.inner == 1 ? 0 : 1;
    return grid;
}

uint32_t symmetry_index(Symmetry symmetry) {
    switch (symmetry) {
    case Symmetry::even: return MS_NDIMAGE_SYMMETRY_EVEN;
    case Symmetry::odd: return MS_NDIMAGE_SYMMETRY_ODD;
    default: return MS_NDIMAGE_SYMMETRY_NONE;
    }
}
} // namespace

BufferPtr run_gpu(ms_context* context, const Plan& plan, const ms_array& input) {
    static std::atomic<uint32_t> capacities[2][3] = {};
    GpuOperationScope operation(context, true);
    const size_t count = plan.passes.size();
    const uint64_t elements = plan.shape.elements;
    const bool widen = plan.dtype == MS_FLOAT64;

    // All passes' weights in one small buffer, narrowed like the data.
    uint64_t total_taps = 0;
    for (const Pass& pass : plan.passes) total_taps += pass.size;
    BufferPtr weights = array::allocate(context, total_taps * sizeof(float));
    {
        float* w = reinterpret_cast<float*>(mutable_data(weights.get()));
        for (const Pass& pass : plan.passes) {
            if (pass.kind == MS_NDIMAGE_UNIFORM) {
                std::fill_n(w, pass.size, static_cast<float>(1.0 / static_cast<double>(pass.size)));
            } else {
                for (uint64_t j = 0; j < pass.size; ++j) w[j] = static_cast<float>(pass.weights[j]);
            }
            w += pass.size;
        }
    }

    // Pass i reads buffer i % 2 and writes the other one; a float64 result has
    // its own buffer, written by the last pass.
    BufferPtr single[2];
    single[0] = array::load_single(context, input, plan.shape);
    if (count > 1 || !widen) single[1] = array::allocate(context, elements * sizeof(float));
    BufferPtr wide = widen ? array::allocate(context, elements * sizeof(double)) : BufferPtr();

    gpu::Batch batch(context);
    uint32_t weight_offset = 0;
    for (size_t i = 0; i < count; ++i) {
        const Pass& pass = plan.passes[i];
        const bool last = i + 1 == count;
        const uint32_t wide_store = last && widen, order = symmetry_index(pass.symmetry);
        const char* kernel = kernels[wide_store][order];
        const uint32_t capacity = threadgroup_capacity(kernel, capacities[wide_store][order]);
        MSNdimageCorrelateParams params{};
        const gpu::Grid grid = grid_for(pass, capacity, params);
        params.length = static_cast<uint32_t>(pass.length);
        params.stride = static_cast<uint32_t>(pass.inner);
        params.taps = static_cast<uint32_t>(pass.size);
        params.weight_offset = weight_offset;
        params.before = static_cast<uint32_t>(pass.before);
        params.mode = pass.mode;
        params.cval = static_cast<float>(pass.cval);
        const ms_buffer* destination = last && widen ? wide.get() : single[(i + 1) % 2].get();
        batch.dispatch(kernel, params,
                       {gpu::bind(single[i % 2].get(), context), gpu::bind(destination, context),
                        gpu::bind(weights.get(), context)},
                       grid);
        weight_offset += params.taps;
    }
    batch.commit_and_wait();
    operation.complete();
    return widen ? std::move(wide) : std::move(single[count % 2]);
}
} // namespace ms::detail::ndimage
