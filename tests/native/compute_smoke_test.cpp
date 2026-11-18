// End-to-end check of the vendored runtime with metal-sci's N-D dispatch grids:
// Metal-storage context -> owned shared buffers -> one 2-D dispatch -> host read.
#include "ms_runtime.h"
#include "ms/detail/storage.hpp"
#include "metal_compute.hpp"
#include "../../src/kernels/ms_kernels.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "compute smoke failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

using namespace ms::detail;

int main() {
    ms_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = MS_RUNTIME_ABI_VERSION;
    options.storage_kind = MS_BUFFER_METAL;
    options.flags = MS_CONTEXT_POISON_UNWRITTEN;
    options.memory_limit_bytes = uint64_t{1} << 30;
    ms_context* ctx = nullptr;
    CHECK(ms_context_create(&options, &ctx) == MS_OK);

    // Deliberately not multiples of the 16 x 16 threadgroup.
    const uint64_t width = 1000, height = 37, count = width * height, bytes = count * sizeof(float);
    auto rx = reserve(ctx, padded_capacity(bytes, ctx));
    BufferPtr x = allocate_reserved(ctx, rx, bytes, Fill::unfilled);
    auto ry = reserve(ctx, padded_capacity(bytes, ctx));
    BufferPtr y = allocate_reserved(ctx, ry, bytes, Fill::unfilled);
    auto* xs = reinterpret_cast<float*>(mutable_data(x.get()));
    auto* ys = reinterpret_cast<float*>(mutable_data(y.get()));
    for (uint64_t i = 0; i < count; ++i) {
        xs[i] = static_cast<float>(i % 251) * 0.5f;
        ys[i] = static_cast<float>(i % 17);
    }

    MSSmokeParams params{};
    params.width = width;
    params.height = height;
    params.alpha = 2.0f;
    gpu::Grid grid;
    grid.groups[0] = (width + MS_SMOKE_GROUP - 1) / MS_SMOKE_GROUP;
    grid.groups[1] = (height + MS_SMOKE_GROUP - 1) / MS_SMOKE_GROUP;
    grid.threads[0] = MS_SMOKE_GROUP;
    grid.threads[1] = MS_SMOKE_GROUP;
    {
        gpu::Batch batch(ctx);
        batch.dispatch("ms_smoke_saxpy2d", params, {gpu::bind(x.get(), ctx), gpu::bind(y.get(), ctx)}, grid);
        batch.commit_and_wait();
    }
    const auto view = read(y.get(), ctx);
    CHECK(view.size == bytes);
    const auto* out = reinterpret_cast<const float*>(view.data);
    for (uint64_t i = 0; i < count; ++i) {
        const float expected = 2.0f * (static_cast<float>(i % 251) * 0.5f) + static_cast<float>(i % 17);
        CHECK(out[i] == expected);
    }

    // Indexed bindings on two specializations of one kernel, in one batch.
    {
        const uint32_t n = static_cast<uint32_t>(count);
        const float scale = 3.0f;
        gpu::Specialization plain{"negate=0", {gpu::Constant::of(4, false)}};
        gpu::Specialization negated{"negate=1", {gpu::Constant::of(4, true)}};
        auto ra = reserve(ctx, padded_capacity(bytes, ctx));
        BufferPtr a = allocate_reserved(ctx, ra, bytes, Fill::unfilled);
        auto rb = reserve(ctx, padded_capacity(bytes, ctx));
        BufferPtr b = allocate_reserved(ctx, rb, bytes, Fill::unfilled);
        gpu::Grid line;
        line.groups[0] = (count + 63) / 64;
        line.threads[0] = 64;
        gpu::Batch batch(ctx);
        batch.dispatch_indexed("ms_smoke_scale", &plain, {{3, gpu::bind(x.get(), ctx)}, {5, gpu::bind(a.get(), ctx)}},
                               {{2, &n, sizeof(n)}, {7, &scale, sizeof(scale)}}, line);
        batch.dispatch_indexed("ms_smoke_scale", &negated, {{3, gpu::bind(a.get(), ctx)}, {5, gpu::bind(b.get(), ctx)}},
                               {{2, &n, sizeof(n)}, {7, &scale, sizeof(scale)}}, line);
        batch.commit_and_wait();
        const auto* as = reinterpret_cast<const float*>(read(a.get(), ctx).data);
        const auto* bs = reinterpret_cast<const float*>(read(b.get(), ctx).data);
        for (uint64_t i = 0; i < count; ++i) {
            const float source = static_cast<float>(i % 251) * 0.5f;
            CHECK(as[i] == 3.0f * source);
            CHECK(bs[i] == -3.0f * (3.0f * source));
        }
    }

    ms_execution_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = MS_RUNTIME_ABI_VERSION;
    CHECK(ms_context_execution_info(ctx, &info) == MS_OK);
    CHECK(info.gpu_command_buffers == 2 && info.gpu_dispatches == 3);

    x.reset();
    y.reset();
    ms_context_release(ctx);
    std::puts("compute smoke ok");
    return 0;
}
