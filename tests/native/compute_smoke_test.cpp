// End-to-end check of the vendored runtime with SciMetal's N-D dispatch grids:
// Metal-storage context -> owned shared buffers -> one 2-D dispatch -> host read.
#include "sm_runtime.h"
#include "sm/detail/storage.hpp"
#include "metal_compute.hpp"
#include "../../src/kernels/sm_kernels.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "compute smoke failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

using namespace sm::detail;

int main() {
    sm_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = SM_RUNTIME_ABI_VERSION;
    options.storage_kind = SM_BUFFER_METAL;
    options.flags = SM_CONTEXT_POISON_UNWRITTEN;
    options.memory_limit_bytes = uint64_t{1} << 30;
    sm_context* ctx = nullptr;
    CHECK(sm_context_create(&options, &ctx) == SM_OK);

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

    SMSmokeParams params{};
    params.width = width;
    params.height = height;
    params.alpha = 2.0f;
    gpu::Grid grid;
    grid.groups[0] = (width + SM_SMOKE_GROUP - 1) / SM_SMOKE_GROUP;
    grid.groups[1] = (height + SM_SMOKE_GROUP - 1) / SM_SMOKE_GROUP;
    grid.threads[0] = SM_SMOKE_GROUP;
    grid.threads[1] = SM_SMOKE_GROUP;
    {
        gpu::Batch batch(ctx);
        batch.dispatch("sm_smoke_saxpy2d", params, {gpu::bind(x.get(), ctx), gpu::bind(y.get(), ctx)}, grid);
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
        batch.dispatch_indexed("sm_smoke_scale", &plain, {{3, gpu::bind(x.get(), ctx)}, {5, gpu::bind(a.get(), ctx)}},
                               {{2, &n, sizeof(n)}, {7, &scale, sizeof(scale)}}, line);
        batch.dispatch_indexed("sm_smoke_scale", &negated, {{3, gpu::bind(a.get(), ctx)}, {5, gpu::bind(b.get(), ctx)}},
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

    sm_execution_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = SM_RUNTIME_ABI_VERSION;
    CHECK(sm_context_execution_info(ctx, &info) == SM_OK);
    CHECK(info.gpu_command_buffers == 2 && info.gpu_dispatches == 3);

    // Scratch-slice offset binding: gpu::bind must pass non-zero ArenaSlab offsets to kernels.
    {
        BufferPtr s0 = allocate_scratch(ctx, 128, Fill::zero);
        BufferPtr s1 = allocate_scratch(ctx, 128, Fill::zero);
        BufferPtr s2 = allocate_scratch(ctx, 128, Fill::zero);
        CHECK(is_scratch_slice(s0.get()) && is_scratch_slice(s1.get()) && is_scratch_slice(s2.get()));
        CHECK(device_view(s1.get(), ctx).offset == 256 && device_view(s2.get(), ctx).offset == 512);
        auto* p0 = reinterpret_cast<float*>(mutable_data(s0.get()));
        auto* p1 = reinterpret_cast<float*>(mutable_data(s1.get()));
        for (uint32_t i = 0; i < 32; ++i) {
            p0[i] = 999.0f;
            p1[i] = static_cast<float>(i + 1);
        }
        const uint32_t n32 = 32;
        const float sc = 4.0f;
        gpu::Specialization plain{"negate=0", {gpu::Constant::of(4, false)}};
        gpu::Grid line;
        line.groups[0] = 1;
        line.threads[0] = 32;
        gpu::Batch batch(ctx);
        batch.dispatch_indexed("sm_smoke_scale", &plain, {{3, gpu::bind(s1.get(), ctx)}, {5, gpu::bind(s2.get(), ctx)}},
                               {{2, &n32, sizeof(n32)}, {7, &sc, sizeof(sc)}}, line);
        batch.commit_and_wait();
        const auto* r0 = reinterpret_cast<const float*>(read(s0.get(), ctx).data);
        const auto* r2 = reinterpret_cast<const float*>(read(s2.get(), ctx).data);
        for (uint32_t i = 0; i < 32; ++i) {
            CHECK(r0[i] == 999.0f);
            CHECK(r2[i] == 4.0f * static_cast<float>(i + 1));
        }
    }

    // Batch::dispatch rejects zero-byte or null parameter payloads cleanly.
    {
        gpu::Batch batch(ctx);
        const gpu::Binding bufs[2] = {gpu::bind(x.get(), ctx), gpu::bind(y.get(), ctx)};
        bool caught = false;
        try {
            batch.dispatch("sm_smoke_saxpy2d", nullptr, 0, bufs, 2, grid);
        } catch (const Error& err) {
            caught = (err.status == SM_EBACKEND);
        }
        CHECK(caught);
    }

    // Cross-context zero-copy commit_async pins borrowed source storage until GPU completion.
    {
        sm_context* consumer = nullptr;
        CHECK(sm_context_create(&options, &consumer) == SM_OK);
        sm_buffer_view exported{};
        CHECK(sm_buffer_export(x.get(), &exported) == SM_OK);
        sm_import_result imported{};
        CHECK(sm_buffer_import(consumer, &exported, SM_COPY_NEVER, &imported) == SM_OK);
        CHECK(imported.copied_bytes == 0 && imported.buffer != nullptr);

        auto ro = reserve(consumer, padded_capacity(bytes, consumer));
        BufferPtr out_async = allocate_reserved(consumer, ro, bytes, Fill::unfilled);
        const uint32_t n = static_cast<uint32_t>(count);
        const float scale = 2.5f;
        gpu::Specialization plain{"negate=0", {gpu::Constant::of(4, false)}};
        gpu::Grid line;
        line.groups[0] = (count + 63) / 64;
        line.threads[0] = 64;

        gpu::Batch batch(consumer);
        batch.dispatch_indexed("sm_smoke_scale", &plain,
                               {{3, gpu::bind(imported.buffer, consumer)}, {5, gpu::bind(out_async.get(), consumer)}},
                               {{2, &n, sizeof(n)}, {7, &scale, sizeof(scale)}}, line);
        batch.commit_async({out_async.get()});
        // Release the borrowed import handle and exported descriptor while the GPU batch is in flight.
        sm_buffer_release(imported.buffer);
        exported.release(exported.owner);

        // read() waits for the async GPU batch to complete.
        const auto* got = reinterpret_cast<const float*>(read(out_async.get(), consumer).data);
        for (uint64_t i = 0; i < count; ++i) {
            CHECK(got[i] == 2.5f * (static_cast<float>(i % 251) * 0.5f));
        }
        out_async.reset();
        sm_context_release(consumer);
    }

    x.reset();
    y.reset();
    sm_context_release(ctx);
    std::puts("compute smoke ok");
    return 0;
}
