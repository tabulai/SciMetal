#include "engine.hpp"
#include "fft_kernels.h"
#include "core/array.hpp"
#include "runtime/metal_compute.hpp"

#include <cstdlib>
#include <cstring>

namespace ms::detail::fft {
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
gpu::Binding bind(const ms_buffer* buffer, ms_context* context) {
    gpu::Binding binding = gpu::bind(buffer, context);
    binding.offset = device_view(buffer, context).offset;
    return binding;
}

constexpr uint64_t scalar_bytes = sizeof(float);

// Buffers of one run. Slots released by a stage are reused by later stages
// that need the same size; everything returns to the context's cache when the
// run ends.
class Buffers {
public:
    explicit Buffers(ms_context* context) : context_(context) {}
    ms_buffer* acquire(uint64_t bytes) {
        for (size_t i = 0; i < idle_.size(); ++i) {
            if (buffer_size(idle_[i]) == bytes) {
                ms_buffer* buffer = idle_[i];
                idle_.erase(idle_.begin() + static_cast<ptrdiff_t>(i));
                return buffer;
            }
        }
        owned_.push_back(array::allocate(context_, bytes));
        return owned_.back().get();
    }
    ms_buffer* adopt(BufferPtr buffer) {
        owned_.push_back(std::move(buffer));
        return owned_.back().get();
    }
    // A slot no later pass of its stage reads: free for the stages after it.
    void release(ms_buffer* buffer) { idle_.push_back(buffer); }
    BufferPtr take(ms_buffer* buffer) {
        for (BufferPtr& owned : owned_) {
            if (owned.get() == buffer) return std::move(owned);
        }
        throw Error(MS_EBACKEND, "FFT result buffer is not owned by its run");
    }

private:
    ms_context* context_;
    std::vector<BufferPtr> owned_;
    std::vector<ms_buffer*> idle_;
};
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
    for (const Stage& stage : stages) {
        for (const Step& step : stage.plan->steps) {
            const gpu::Specialization constants = specialization(step.constants);
            gpu::warm_pipeline(step.kernel.c_str(), &constants);
        }
    }
    gpu::warm_pipeline("ms_fft_scale");
}

bool refused(const std::vector<Stage>& stages, Refusal& refusal) {
    for (size_t index = 0; index < stages.size(); ++index) {
        for (const Step& step : stages[index].plan->steps) {
            const gpu::Specialization constants = specialization(step.constants);
            const uint32_t granted = gpu::max_threads_per_group(step.kernel.c_str(), &constants);
            if (static_cast<uint32_t>(step.width) * static_cast<uint32_t>(step.threads) > granted) {
                refusal = {index, static_cast<int32_t>(granted)};
                return true;
            }
        }
    }
    return false;
}

BufferPtr run(ms_context* context, const std::vector<Stage>& stages, BufferPtr input, float scale) {
    if (stages.empty() || !input) throw Error(MS_EINVAL, "an FFT run needs a stage and an input buffer");
    Buffers buffers(context);
    gpu::Batch batch(context);
    ms_buffer* carried = buffers.adopt(std::move(input));

    // A plan whose last pass is ours applies the factor there for free; the
    // last such stage does it.
    size_t scaling_stage = stages.size();
    if (scale != 1.0f) {
        for (size_t i = stages.size(); i-- > 0;) {
            if (stages[i].plan->takes_scale) {
                scaling_stage = i;
                break;
            }
        }
    }

    std::vector<Dispatch> dispatches;
    std::vector<ms_buffer*> slots;
    std::vector<ms_buffer*> tables;
    for (size_t index = 0; index < stages.size(); ++index) {
        const Plan& plan = *stages[index].plan;
        const uint64_t count = stages[index].count;
        std::string reason;
        if (!instantiate(plan, count, dispatches, reason)) throw Error(MS_ENOTSUP, reason);

        slots.assign(plan.slots.size(), nullptr);
        if (buffer_size(carried, context) < plan.slots[0] * count * scalar_bytes)
            throw Error(MS_EBACKEND, "FFT input buffer is smaller than its plan's input slot");
        slots[0] = carried;
        for (size_t slot = 1; slot < plan.slots.size(); ++slot)
            slots[slot] = buffers.acquire(plan.slots[slot] * count * scalar_bytes);
        tables.clear();
        for (const auto& table : plan.tables) {
            BufferPtr device = allocate_scratch(context, table->size(), Fill::unfilled);
            std::memcpy(mutable_data(device.get()), table->data(), table->size());
            tables.push_back(buffers.adopt(std::move(device)));
        }

        for (size_t i = 0; i < plan.steps.size(); ++i) {
            const Step& step = plan.steps[i];
            const Dispatch& dispatch = dispatches[i];
            const gpu::Specialization constants = specialization(step.constants);
            gpu::Grid grid;
            grid.groups[0] = dispatch.groups;
            grid.threads[1] = static_cast<uint32_t>(step.width);
            grid.threads[2] = static_cast<uint32_t>(step.threads);

            std::vector<gpu::IndexedBuffer> bound = {{0, bind(slots[step.source], context)},
                                                     {1, bind(slots[step.target], context)}};
            uint32_t next = 2;
            for (const int32_t table : step.tables) bound.push_back({next++, bind(tables[static_cast<size_t>(table)], context)});
            std::vector<gpu::IndexedBytes> bytes;
            MSFFTStridedParams params{};
            if (step.kind == Kernel::strided) {
                params.n = step.n;
                params.stride = step.stride;
                params.blocks = dispatch.blocks;
                params.outer = step.outer;
                params.tail = step.tail;
                params.flags = step.flags;
                params.scale = index == scaling_stage && i + 1 == plan.steps.size() ? scale : 1.0f;
                bytes.push_back({next, &params, sizeof(params)});
            } else {
                for (const int32_t& value : dispatch.values) bytes.push_back({next++, &value, sizeof(value)});
            }
            batch.dispatch_indexed(step.kernel.c_str(), &constants, bound.data(), bound.size(), bytes.data(),
                                   bytes.size(), grid);
        }

        carried = slots[plan.result_slot];
        for (ms_buffer* slot : slots) {
            if (slot != carried) buffers.release(slot);
        }
    }

    if (scale != 1.0f && scaling_stage == stages.size()) {
        MSFFTScaleParams params{};
        params.count = buffer_size(carried, context) / scalar_bytes;
        params.scale = scale;
        const uint64_t per_group = uint64_t{MS_GPU_THREADS} * MS_FFT_SCALE_CHUNK;
        batch.dispatch("ms_fft_scale", params, {bind(carried, context)}, (params.count + per_group - 1) / per_group);
    }
    batch.commit_and_wait();
    return buffers.take(carried);
}
} // namespace ms::detail::fft
