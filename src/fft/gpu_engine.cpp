#include "engine.hpp"
#include "fft_kernels.h"
#include "core/array.hpp"
#include "runtime/metal_compute.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_map>

namespace sm::detail::fft {
namespace {
gpu::Specialization specialization(const Constants& constants) {
    gpu::Specialization result;
    result.key = constants.key();
    result.constants.reserve(22);
    result.constants.push_back(gpu::Constant::of(0, constants.inverse));
    result.constants.push_back(gpu::Constant::of(1, constants.power_of_two));
    result.constants.push_back(gpu::Constant::of(2, constants.elems_per_thread));
    result.constants.push_back(gpu::Constant::of(3, constants.rader_m));
    for (uint32_t i = 0; i < 9; ++i) {
        result.constants.push_back(gpu::Constant::of(4 + i, constants.stockham[i]));
        result.constants.push_back(gpu::Constant::of(13 + i, constants.rader[i]));
    }
    return result;
}

// Scratch slices live inside a shared slab: bind them at their own offset.
gpu::Binding bind(const sm_buffer* buffer, sm_context* context) {
    gpu::Binding binding = gpu::bind(buffer, context);
    binding.offset = device_view(buffer, context).offset;
    return binding;
}

constexpr uint64_t scalar_bytes = sizeof(float);

struct ResolvedPlan {
    std::vector<MTL::ComputePipelineState*> pipelines;
    bool is_refused = false;
    int32_t granted = 0;
};

constexpr size_t resolved_cache_capacity = 512;

std::shared_ptr<const ResolvedPlan> resolve_plan(const Plan& plan) {
    static std::mutex mutex;
    static std::unordered_map<uint64_t, std::shared_ptr<const ResolvedPlan>> cache;
    static std::deque<uint64_t> order;

    if (plan.id != 0) {
        std::lock_guard lock(mutex);
        const auto found = cache.find(plan.id);
        if (found != cache.end()) return found->second;
    }

    auto resolved = std::make_shared<ResolvedPlan>();
    resolved->pipelines.reserve(plan.steps.size());
    for (const Step& step : plan.steps) {
        const gpu::Specialization constants = specialization(step.constants);
        MTL::ComputePipelineState* state = gpu::resolve_pipeline(step.kernel.c_str(), &constants);
        const uint32_t granted = static_cast<uint32_t>(state->maxTotalThreadsPerThreadgroup());
        if (static_cast<uint32_t>(step.width) * static_cast<uint32_t>(step.threads) > granted) {
            resolved->is_refused = true;
            resolved->granted = static_cast<int32_t>(granted);
            break;
        }
        resolved->pipelines.push_back(state);
    }

    if (plan.id != 0) {
        std::lock_guard lock(mutex);
        auto [it, inserted] = cache.emplace(plan.id, resolved);
        if (!inserted) return it->second;
        order.push_back(plan.id);
        if (order.size() > resolved_cache_capacity) {
            cache.erase(order.front());
            order.pop_front();
        }
    }
    return resolved;
}

MTL::ComputePipelineState* scale_pipeline() {
    static MTL::ComputePipelineState* const state = gpu::resolve_pipeline("sm_fft_scale");
    return state;
}

MTL::ComputePipelineState* c2r_hermitian_pipeline() {
    static MTL::ComputePipelineState* const state = gpu::resolve_pipeline("sm_fft_c2r_hermitian_fix");
    return state;
}

uint64_t checked_slot_bytes(uint64_t slot_scalars, uint64_t count, uint64_t tail_scalars = 0) {
    uint64_t scalars = 0;
    if (sm_checked_mul(slot_scalars, count, &scalars) != SM_OK ||
        sm_checked_add(scalars, tail_scalars, &scalars) != SM_OK)
        throw Error(SM_EOVERFLOW, "FFT device buffer byte size overflows");
    uint64_t bytes = 0;
    if (sm_checked_mul(scalars, scalar_bytes, &bytes) != SM_OK)
        throw Error(SM_EOVERFLOW, "FFT device buffer byte size overflows");
    return bytes;
}
} // namespace

Limits device_limits() {
    static const Limits limits = [] {
        Limits result;
        // Shader validation instruments every pipeline and doubles its static
        // threadgroup memory, so kernels declaring the full 32 KiB cannot be
        // built under it: plan with half.
        const char* validation = std::getenv("MTL_SHADER_VALIDATION");
        if (validation && *validation && std::strcmp(validation, "0") != 0) result.group_elements = 2048;
        return result;
    }();
    return limits;
}

void prepare(const std::vector<Stage>& stages) {
    for (const Stage& stage : stages) (void)resolve_plan(*stage.plan);
    (void)scale_pipeline();
    for (size_t i = 1; i < stages.size(); ++i) {
        if (stages[i].plan->clear_hermitian_imag) {
            (void)c2r_hermitian_pipeline();
            break;
        }
    }
}

bool refused(std::vector<Stage>& stages, Refusal& refusal) {
    for (size_t index = 0; index < stages.size(); ++index) {
        auto resolved = resolve_plan(*stages[index].plan);
        stages[index].resolved = resolved;
        if (resolved->is_refused) {
            refusal = {index, resolved->granted};
            return true;
        }
    }
    return false;
}

Scratch::Scratch(sm_context* context) : context_(context) {}

sm_buffer* Scratch::acquire(uint64_t bytes) {
    for (size_t i = 0; i < idle_.size(); ++i) {
        if (buffer_size(idle_[i]) == bytes) {
            sm_buffer* buffer = idle_[i];
            idle_.erase(idle_.begin() + static_cast<ptrdiff_t>(i));
            return buffer;
        }
    }
    owned_.push_back(array::allocate(context_, bytes));
    return owned_.back().get();
}

sm_buffer* Scratch::adopt(BufferPtr buffer) {
    owned_.push_back(std::move(buffer));
    return owned_.back().get();
}

void Scratch::release(sm_buffer* buffer) {
    if (!buffer) return;
    bool is_owned = false;
    for (const BufferPtr& owned : owned_) {
        if (owned.get() == buffer) {
            is_owned = true;
            break;
        }
    }
    if (!is_owned) return;
    if (std::find(idle_.begin(), idle_.end(), buffer) == idle_.end()) idle_.push_back(buffer);
}

BufferPtr Scratch::take(sm_buffer* buffer) {
    auto it = std::find(idle_.begin(), idle_.end(), buffer);
    if (it != idle_.end()) idle_.erase(it);
    for (size_t i = 0; i < owned_.size(); ++i) {
        if (owned_[i].get() == buffer) {
            BufferPtr out = std::move(owned_[i]);
            owned_.erase(owned_.begin() + static_cast<ptrdiff_t>(i));
            return out;
        }
    }
    throw Error(SM_EBACKEND, "FFT result buffer is not owned by its run");
}

void Scratch::clear() {
    idle_.clear();
    owned_.clear();
}

sm_buffer* encode(sm_context* context, gpu::Batch& batch, Scratch& scratch,
                  const std::vector<Stage>& stages, sm_buffer* input, float scale,
                  Profile* profile) {
    if (stages.empty() || !input) throw Error(SM_EINVAL, "an FFT run needs a stage and an input buffer");
    using Clock = std::chrono::steady_clock;
    const auto elapsed_ns = [](Clock::time_point start, Clock::time_point end) -> uint64_t {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    };

    sm_buffer* carried = input;

    // A plan whose last pass is ours applies the factor there for free; the
    // last such stage does it — except never fuse scale > 1.0f ahead of a later
    // stage that divides by n (multiplying early can overflow float32).
    size_t scaling_stage = stages.size();
    if (scale != 1.0f) {
        const bool amplifies = std::fabs(scale) > 1.0f;
        bool later_divides = false;
        for (size_t i = stages.size(); i-- > 0;) {
            if (stages[i].plan->takes_scale && !(amplifies && later_divides)) {
                scaling_stage = i;
                break;
            }
            if (stages[i].plan->divides_by_n) later_divides = true;
        }
    }

    std::vector<Dispatch> dispatches;
    std::vector<sm_buffer*> slots;
    std::vector<sm_buffer*> tables;
    for (size_t index = 0; index < stages.size(); ++index) {
        Clock::time_point t_alloc;
        if (profile) t_alloc = Clock::now();
        const Plan& plan = *stages[index].plan;
        const std::shared_ptr<const ResolvedPlan> resolved =
            stages[index].resolved ? std::static_pointer_cast<const ResolvedPlan>(stages[index].resolved)
                                   : resolve_plan(plan);
        if (resolved->is_refused || resolved->pipelines.size() != plan.steps.size())
            throw Error(SM_ENOTSUP, "this Metal device's threadgroup capacity is too small for the kernel");

        const uint64_t count = stages[index].count;
        std::string reason;
        if (!instantiate(plan, count, dispatches, reason)) throw Error(SM_ENOTSUP, reason);

        slots.assign(plan.slots.size(), nullptr);
        if (buffer_size(carried, context) < checked_slot_bytes(plan.slots[0], count, plan.input_tail_scalars))
            throw Error(SM_EBACKEND, "FFT input buffer is smaller than its plan's input slot");
        slots[0] = carried;
        for (size_t slot = 1; slot < plan.slots.size(); ++slot)
            slots[slot] = scratch.acquire(checked_slot_bytes(plan.slots[slot], count));
        tables.clear();
        for (const auto& table : plan.tables) {
            BufferPtr device = allocate_scratch(context, table->size(), Fill::unfilled);
            std::memcpy(mutable_data(device.get()), table->data(), table->size());
            tables.push_back(scratch.adopt(std::move(device)));
        }
        Clock::time_point t_enc;
        if (profile) {
            t_enc = Clock::now();
            profile->alloc_scratch_ns += elapsed_ns(t_alloc, t_enc);
        }

        if (index > 0 && plan.clear_hermitian_imag) {
            MSFFTC2RHermitianParams hparams{};
            hparams.rows = static_cast<sm_u32>(count);
            hparams.row_stride = static_cast<sm_u32>(plan.slots[0] / 2);
            hparams.nyquist = (plan.request.n % 2 == 0) ? static_cast<sm_u32>(plan.request.n / 2) : 0u;
            const uint64_t groups = (count + SM_GPU_THREADS - 1) / SM_GPU_THREADS;
            batch.dispatch(c2r_hermitian_pipeline(), hparams, {bind(slots[0], context)}, groups);
        }

        for (size_t i = 0; i < plan.steps.size(); ++i) {
            const Step& step = plan.steps[i];
            const Dispatch& dispatch = dispatches[i];
            gpu::Grid grid;
            grid.groups[0] = dispatch.groups;
            grid.threads[1] = static_cast<uint32_t>(step.width);
            grid.threads[2] = static_cast<uint32_t>(step.threads);

            std::array<gpu::IndexedBuffer, gpu::max_bound_buffers + 1> bound;
            size_t bound_count = 0;
            bound[bound_count++] = {0, bind(slots[step.source], context)};
            bound[bound_count++] = {1, bind(slots[step.target], context)};
            uint32_t next = 2;
            for (const int32_t table : step.tables) {
                if (bound_count >= bound.size()) throw Error(SM_EBACKEND, "GPU dispatch binds too many buffers");
                bound[bound_count++] = {next++, bind(tables[static_cast<size_t>(table)], context)};
            }
            std::array<gpu::IndexedBytes, gpu::max_bound_buffers + 1> bytes;
            size_t bytes_count = 0;
            MSFFTStridedParams params{};
            if (step.kind == Kernel::strided) {
                params.n = step.n;
                params.stride = step.stride;
                params.blocks = dispatch.blocks;
                params.outer = step.outer;
                params.tail = step.tail;
                params.flags = step.flags;
                params.scale = index == scaling_stage && i + 1 == plan.steps.size() ? scale : 1.0f;
                bytes[bytes_count++] = {next, &params, sizeof(params)};
            } else {
                for (const int32_t& value : dispatch.values) {
                    if (bytes_count >= bytes.size()) throw Error(SM_EBACKEND, "GPU dispatch has too many arguments");
                    bytes[bytes_count++] = {next++, &value, sizeof(value)};
                }
            }
            batch.dispatch_indexed(resolved->pipelines[i], bound.data(), bound_count, bytes.data(), bytes_count, grid);
        }
        if (profile) profile->encode_ns += elapsed_ns(t_enc, Clock::now());

        carried = slots[plan.result_slot];
        for (sm_buffer* slot : slots) {
            if (slot != carried) scratch.release(slot);
        }
    }

    if (scale != 1.0f && scaling_stage == stages.size()) {
        Clock::time_point t_scale;
        if (profile) t_scale = Clock::now();
        MSFFTScaleParams params{};
        params.count = buffer_size(carried, context) / scalar_bytes;
        params.scale = scale;
        const uint64_t per_group = uint64_t{SM_GPU_THREADS} * SM_FFT_SCALE_CHUNK;
        batch.dispatch(scale_pipeline(), params, {bind(carried, context)}, (params.count + per_group - 1) / per_group);
        if (profile) profile->encode_ns += elapsed_ns(t_scale, Clock::now());
    }
    return carried;
}

BufferPtr run(sm_context* context, const std::vector<Stage>& stages, BufferPtr input, float scale,
              Profile* profile) {
    if (stages.empty() || !input) throw Error(SM_EINVAL, "an FFT run needs a stage and an input buffer");
    Scratch scratch(context);
    gpu::Batch batch(context);
    sm_buffer* input_buf = scratch.adopt(std::move(input));
    sm_buffer* carried = encode(context, batch, scratch, stages, input_buf, scale, profile);
    gpu::Batch::Timing timing;
    batch.commit_and_wait(profile ? &timing : nullptr);
    if (profile) {
        profile->encode_ns += timing.encode_ns;
        profile->gpu_ns = timing.gpu_ns;
        profile->wait_overhead_ns =
            (timing.host_wait_ns > timing.gpu_ns ? timing.host_wait_ns - timing.gpu_ns : 0) + timing.complete_ns;
    }
    return scratch.take(carried);
}
} // namespace sm::detail::fft
