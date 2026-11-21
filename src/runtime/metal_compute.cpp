#include "metal_compute.hpp"
#include "pipeline_cache.hpp"
#include "../kernels/sm_kernels.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

extern "C" {
extern const unsigned char sm_metallib_data[];
extern const size_t sm_metallib_size;
}

namespace sm::detail::gpu {
namespace {
std::string describe(NS::Error* error) {
    const NS::String* text = error ? error->localizedDescription() : nullptr;
    const char* bytes = text ? text->utf8String() : nullptr;
    return bytes ? bytes : "no Metal error description";
}

struct Runtime {
    std::shared_ptr<DeviceOwner> device;
    MTL::CommandQueue* queue = nullptr;
    MTL::Library* library = nullptr;
    MTL::SharedEvent* event = nullptr;
    std::atomic<uint64_t> next_ticket{1};
    std::mutex mutex;
    std::mutex submit_mutex;
    std::unordered_map<std::string, MTL::ComputePipelineState*> pipelines;

    ~Runtime() {
        if (::getpid() != metal_process) return; // never enter inherited Metal state
        for (auto& entry : pipelines) entry.second->release();
        if (event) event->release();
        if (library) library->release();
        if (queue) queue->release();
    }
};

// Created on the first GPU route: CPU execution never initializes Metal.
Runtime& runtime() {
    require_original_process();
    static const std::unique_ptr<Runtime> instance = [] {
        AutoreleasePool pool;
        auto result = std::make_unique<Runtime>();
        result->device = shared_device();
        result->queue = result->device->device->newCommandQueue();
        if (!result->queue) throw Error(SM_EBACKEND, "Metal command queue creation failed");
        result->event = result->device->device->newSharedEvent();
        if (!result->event) throw Error(SM_EBACKEND, "Metal shared event creation failed");
        dispatch_data_t data = dispatch_data_create(sm_metallib_data, sm_metallib_size, nullptr,
                                                    DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NS::Error* error = nullptr;
        result->library = result->device->device->newLibrary(data, &error);
        dispatch_release(data);
        if (!result->library) throw Error(SM_EBACKEND, "embedded Metal library failed to load: " + describe(error));
        return result;
    }();
    return *instance;
}

MTL::ComputePipelineState* pipeline(const char* kernel, const Specialization* specialization = nullptr) {
    auto& state = runtime();
    // Specializations are distinct pipelines: cache them under kernel#key.
    const std::string cache_key = specialization ? std::string(kernel) + "#" + specialization->key : std::string(kernel);
    const char* name = cache_key.c_str();
    {
        std::lock_guard lock(state.mutex);
        const auto found = state.pipelines.find(cache_key);
        if (found != state.pipelines.end()) return found->second;
    }
    AutoreleasePool pool;
    MTL::Function* function = nullptr;
    NS::Error* error = nullptr;
    if (specialization) {
        MTL::FunctionConstantValues* values = MTL::FunctionConstantValues::alloc()->init();
        if (!values) throw Error(SM_EOOM, "Metal function-constant allocation failed");
        for (const Constant& constant : specialization->constants) {
            static constexpr MTL::DataType types[] = {MTL::DataTypeBool, MTL::DataTypeInt, MTL::DataTypeUInt,
                                                      MTL::DataTypeFloat};
            values->setConstantValue(&constant.value, types[constant.type], constant.index);
        }
        function = state.library->newFunction(NS::String::string(kernel, NS::UTF8StringEncoding), values, &error);
        values->release();
        if (!function)
            throw Error(SM_EBACKEND, std::string("Metal kernel specialization failed for ") + name + ": " + describe(error));
    } else {
        function = state.library->newFunction(NS::String::string(kernel, NS::UTF8StringEncoding));
    }
    if (!function) throw Error(SM_EBACKEND, std::string("Metal kernel is missing: ") + name);
    error = nullptr;
    PipelinePtr<MTL::ComputePipelineState> result(state.device->device->newComputePipelineState(function, &error));
    function->release();
    if (!result) throw Error(SM_EBACKEND, std::string("Metal pipeline creation failed for ") + name + ": " + describe(error));
    // Threadgroup capacity is checked per dispatch against the grid it asks for.
    return cache_pipeline(state.pipelines, state.mutex, name, std::move(result));
}

struct BatchSubmission final : AsyncSync {
    std::shared_ptr<DeviceOwner> device;
    MTL::CommandBuffer* command = nullptr;
    MTL::SharedEvent* event = nullptr;
    uint64_t ticket = 0;
    std::shared_ptr<void> state_handle;
    std::vector<std::shared_ptr<void>> pinned_allocations;
    std::mutex mutex;
    bool retired = false;

    BatchSubmission(std::shared_ptr<DeviceOwner> dev, MTL::CommandBuffer* cmd, MTL::SharedEvent* ev,
                    uint64_t value, std::shared_ptr<void> origin, std::vector<std::shared_ptr<void>> pins)
        : device(std::move(dev)), command(cmd), event(ev), ticket(value), state_handle(std::move(origin)),
          pinned_allocations(std::move(pins)) {
        if (command) command->retain();
        if (event) event->retain();
    }
    ~BatchSubmission() override {
        if (::getpid() != metal_process) return;
        if (!retired && !pinned_allocations.empty() && command &&
            command->status() >= MTL::CommandBufferStatusCommitted) {
            AutoreleasePool pool;
            command->waitUntilCompleted();
        }
        pinned_allocations.clear();
        if (command) command->release();
        if (event) event->release();
    }
    void wait(sm_context* waiter) override {
        require_original_process();
        std::vector<std::shared_ptr<void>> release_pins;
        {
            std::lock_guard lock(mutex);
            if (retired) return;
            AutoreleasePool pool;
            const auto submitted = std::chrono::steady_clock::now();
            command->waitUntilCompleted();
            const uint64_t wait_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - submitted).count());
            const double seconds = command->GPUEndTime() - command->GPUStartTime();
            const uint64_t gpu_ns = seconds > 0 ? static_cast<uint64_t>(seconds * 1e9) : 0;
            retire_async_gpu(state_handle, waiter, gpu_ns, wait_ns);
            retired = true;
            release_pins.swap(pinned_allocations);
            if (command->status() != MTL::CommandBufferStatusCompleted)
                throw Error(SM_EBACKEND, "Metal command buffer failed: " + describe(command->error()));
        }
    }
    bool completed() noexcept override {
        if (::getpid() != metal_process) return false;
        std::lock_guard lock(mutex);
        return retired;
    }
};

struct BatchRollbackGuard {
    PreparedGpuBatch* prepared = nullptr;
    ~BatchRollbackGuard() noexcept {
        if (prepared) rollback_gpu_batch(*prepared);
    }
};

template<class Encoder>
struct EncoderGuard {
    Encoder* enc = nullptr;
    ~EncoderGuard() noexcept {
        if (enc) enc->endEncoding();
    }
};
} // namespace

void require_runtime() { runtime(); }
void warm_pipeline(const char* kernel, const Specialization* specialization) { pipeline(kernel, specialization); }
MTL::ComputePipelineState* resolve_pipeline(const char* kernel, const Specialization* specialization) {
    return pipeline(kernel, specialization);
}
uint32_t max_threads_per_group(const char* kernel, const Specialization* specialization) {
    return static_cast<uint32_t>(pipeline(kernel, specialization)->maxTotalThreadsPerThreadgroup());
}

uint64_t address(const sm_buffer* buffer, sm_context* context) {
    const auto view = device_view(buffer, context);
    return static_cast<MTL::Buffer*>(view.resource)->gpuAddress() + view.offset;
}
MTL::Buffer* resource(const sm_buffer* buffer, sm_context* context) {
    return static_cast<MTL::Buffer*>(device_view(buffer, context).resource);
}

Binding bind(const sm_buffer* buffer, sm_context* context) {
    const auto view = device_view(buffer, context);
    return {static_cast<MTL::Buffer*>(view.resource), is_scratch_slice(buffer) ? view.offset : 0, buffer};
}

Batch::Batch(sm_context* context) : context_(context) {
    require_runtime();
}
Batch::~Batch() {
    // An abandoned batch is never committed: no work is submitted.
    if (::getpid() != metal_process) return;
    if (command_) command_->release();
}
Grid Grid::linear(uint64_t threadgroups) {
    Grid grid;
    grid.groups[0] = threadgroups;
    grid.threads[0] = SM_GPU_THREADS;
    return grid;
}
void Batch::dispatch(MTL::ComputePipelineState* state, const void* params, size_t bytes, const Binding* buffers,
                     size_t count, const Grid& grid) {
    if (committed_) throw Error(SM_ESTATE, "GPU batch used after commit");
    if (!state) throw Error(SM_EBACKEND, "GPU dispatch has a null pipeline");
    if (count > max_bound_buffers) throw Error(SM_EBACKEND, "GPU dispatch binds more than 30 buffers");
    if (count && !buffers) throw Error(SM_EBACKEND, "GPU dispatch has null buffer bindings");
    if (!bytes || !params || bytes % 8 || bytes > max_params_bytes)
        throw Error(SM_EBACKEND, "GPU dispatch parameters must be a non-empty multiple of 8 bytes and at most 4 KiB");
    for (size_t i = 0; i < count; ++i) {
        if (!buffers[i].buffer) throw Error(SM_EBACKEND, "GPU dispatch binds a null Metal buffer");
    }
    if (!grid.groups[0] || !grid.groups[1] || !grid.groups[2]) return;
    for (int axis = 0; axis < 3; ++axis) {
        if (grid.groups[axis] > UINT32_MAX) throw Error(SM_ENOTSUP, "GPU dispatch exceeds the threadgroup grid limit");
        if (!grid.threads[axis]) throw Error(SM_EBACKEND, "GPU dispatch has an empty threadgroup");
    }
    if (grid.threads_per_group() > state->maxTotalThreadsPerThreadgroup())
        throw Error(SM_ENOTSUP, "this Metal device's threadgroup capacity is too small for the kernel");
    const size_t param_offset = param_storage_.size();
    param_storage_.resize(param_offset + bytes);
    std::memcpy(param_storage_.data() + param_offset, params, bytes);
    const size_t binding_offset = binding_storage_.size();
    binding_storage_.insert(binding_storage_.end(), buffers, buffers + count);
    commands_.push_back({state, param_offset, bytes, binding_offset, count, grid, false, 0, 0});
    ++dispatches_;
}
void Batch::dispatch(const char* kernel, const void* params, size_t bytes, const Binding* buffers, size_t count,
                     const Grid& grid) {
    dispatch(pipeline(kernel), params, bytes, buffers, count, grid);
}
void Batch::dispatch_indexed(MTL::ComputePipelineState* state,
                             const IndexedBuffer* buffers, size_t buffer_count,
                             const IndexedBytes* bytes, size_t bytes_count, const Grid& grid) {
    if (committed_) throw Error(SM_ESTATE, "GPU batch used after commit");
    if (!state) throw Error(SM_EBACKEND, "GPU dispatch has a null pipeline");
    if (buffer_count && !buffers) throw Error(SM_EBACKEND, "GPU dispatch has null indexed buffers");
    if (bytes_count && !bytes) throw Error(SM_EBACKEND, "GPU dispatch has null indexed bytes");
    uint64_t used = 0;
    const auto claim = [&](uint32_t index) {
        if (index > max_bound_buffers) throw Error(SM_EBACKEND, "GPU dispatch argument index exceeds the buffer table");
        if (used & (uint64_t{1} << index)) throw Error(SM_EBACKEND, "GPU dispatch binds one argument index twice");
        used |= uint64_t{1} << index;
    };
    for (size_t i = 0; i < buffer_count; ++i) {
        claim(buffers[i].index);
        if (!buffers[i].binding.buffer) throw Error(SM_EBACKEND, "GPU dispatch binds a null Metal buffer");
    }
    for (size_t i = 0; i < bytes_count; ++i) {
        claim(bytes[i].index);
        if (!bytes[i].size || bytes[i].size > max_params_bytes || !bytes[i].data)
            throw Error(SM_EBACKEND, "GPU dispatch by-value arguments must be 1 byte to 4 KiB");
    }
    if (!grid.groups[0] || !grid.groups[1] || !grid.groups[2]) return;
    for (int axis = 0; axis < 3; ++axis) {
        if (grid.groups[axis] > UINT32_MAX) throw Error(SM_ENOTSUP, "GPU dispatch exceeds the threadgroup grid limit");
        if (!grid.threads[axis]) throw Error(SM_EBACKEND, "GPU dispatch has an empty threadgroup");
    }
    if (grid.threads_per_group() > state->maxTotalThreadsPerThreadgroup())
        throw Error(SM_ENOTSUP, "this Metal device's threadgroup capacity is too small for the kernel");
    RecordedDispatch command;
    command.state = state;
    command.grid = grid;
    command.indexed = true;
    command.binding_offset = binding_storage_.size();
    command.binding_count = buffer_count;
    command.slot_offset = slot_storage_.size();
    command.slot_count = buffer_count + bytes_count;
    for (size_t i = 0; i < buffer_count; ++i) {
        slot_storage_.push_back({buffers[i].index, false, binding_storage_.size(), 0});
        binding_storage_.push_back(buffers[i].binding);
    }
    for (size_t i = 0; i < bytes_count; ++i) {
        const size_t offset = param_storage_.size();
        param_storage_.resize(offset + bytes[i].size);
        std::memcpy(param_storage_.data() + offset, bytes[i].data, bytes[i].size);
        slot_storage_.push_back({bytes[i].index, true, offset, bytes[i].size});
    }
    commands_.push_back(command);
    ++dispatches_;
}
void Batch::dispatch_indexed(const char* kernel, const Specialization* specialization,
                             const IndexedBuffer* buffers, size_t buffer_count,
                             const IndexedBytes* bytes, size_t bytes_count, const Grid& grid) {
    dispatch_indexed(pipeline(kernel, specialization), buffers, buffer_count, bytes, bytes_count, grid);
}
void Batch::fill(const sm_buffer* buffer, uint64_t byte_offset, uint64_t bytes, uint8_t value) {
    if (committed_) throw Error(SM_ESTATE, "GPU batch used after commit");
    if (!bytes) return;
    const auto view = device_view(buffer, context_);
    uint64_t end = 0;
    if (sm_checked_add(byte_offset, bytes, &end) != SM_OK || end > view.size)
        throw Error(SM_EINVAL, "GPU fill exceeds logical buffer bounds");
    explicit_fills_.push_back({view.resource, view.offset + byte_offset, bytes, value});
    explicit_buffers_.push_back(buffer);
}
void Batch::use(const std::vector<MTL::Buffer*>& buffers, MTL::ResourceUsage usage) {
    if (committed_) throw Error(SM_ESTATE, "GPU batch used after commit");
    if (buffers.empty()) return;
    uses_.push_back({buffers, usage});
}
void Batch::encode_use(MTL::ComputeCommandEncoder* encoder, const std::vector<MTL::Buffer*>& buffers,
                       MTL::ResourceUsage usage) {
    // Chunks of one column often share a validity or values allocation.
    std::vector<const MTL::Resource*> resources(buffers.begin(), buffers.end());
    std::sort(resources.begin(), resources.end());
    resources.erase(std::unique(resources.begin(), resources.end()), resources.end());
    encoder->useResources(resources.data(), resources.size(), usage);
}
PreparedGpuBatch Batch::encode_batch(const sm_buffer* const* outputs, size_t output_count) {
    std::vector<const sm_buffer*> bound;
    bound.reserve(binding_storage_.size() + explicit_buffers_.size() + output_count);
    for (const auto& b : binding_storage_) {
        if (b.handle) bound.push_back(b.handle);
    }
    for (const sm_buffer* b : explicit_buffers_) {
        if (b) bound.push_back(b);
    }
    for (size_t i = 0; i < output_count; ++i) {
        if (outputs[i]) bound.push_back(outputs[i]);
    }
    std::sort(bound.begin(), bound.end());
    bound.erase(std::unique(bound.begin(), bound.end()), bound.end());

    std::vector<void*> indirect;
    for (const auto& u : uses_) {
        for (auto* b : u.buffers) {
            if (b) indirect.push_back(b);
        }
    }
    std::sort(indirect.begin(), indirect.end());
    indirect.erase(std::unique(indirect.begin(), indirect.end()), indirect.end());

    PreparedGpuBatch prepared = prepare_gpu_batch(context_, bound, indirect);
    BatchRollbackGuard rollback{&prepared};
    for (const auto& wait : prepared.waits)
        command_->encodeWait(static_cast<MTL::Event*>(wait.event), wait.value);

    if (!prepared.fills.empty() || !explicit_fills_.empty()) {
        MTL::BlitCommandEncoder* blit = command_->blitCommandEncoder();
        if (!blit) throw Error(SM_EBACKEND, "Metal blit encoder creation failed");
        EncoderGuard<MTL::BlitCommandEncoder> blit_guard{blit};
        for (const auto& f : prepared.fills) {
            blit->fillBuffer(static_cast<MTL::Buffer*>(f.metal_buffer),
                             NS::Range(static_cast<NS::UInteger>(f.offset), static_cast<NS::UInteger>(f.bytes)),
                             f.value);
        }
        for (const auto& f : explicit_fills_) {
            blit->fillBuffer(static_cast<MTL::Buffer*>(f.metal_buffer),
                             NS::Range(static_cast<NS::UInteger>(f.offset), static_cast<NS::UInteger>(f.bytes)),
                             f.value);
        }
        blit->endEncoding();
        blit_guard.enc = nullptr;
    }

    MTL::ComputeCommandEncoder* encoder = command_->computeCommandEncoder(); // serial dispatch: in order, earlier writes visible
    if (!encoder) throw Error(SM_EBACKEND, "Metal compute encoder creation failed");
    EncoderGuard<MTL::ComputeCommandEncoder> enc_guard{encoder};
    for (const auto& u : uses_) encode_use(encoder, u.buffers, u.usage);
    for (const auto& cmd : commands_) {
        encoder->setComputePipelineState(cmd.state);
        if (cmd.indexed) {
            for (size_t i = 0; i < cmd.slot_count; ++i) {
                const Slot& slot = slot_storage_[cmd.slot_offset + i];
                if (slot.is_bytes) {
                    encoder->setBytes(param_storage_.data() + slot.offset, slot.size, slot.index);
                } else {
                    const Binding& b = binding_storage_[slot.offset];
                    encoder->setBuffer(b.buffer, static_cast<NS::UInteger>(b.offset), slot.index);
                }
            }
        } else {
            encoder->setBytes(param_storage_.data() + cmd.param_offset, cmd.param_bytes, 0);
            const Binding* buffers = binding_storage_.data() + cmd.binding_offset;
            for (size_t index = 0; index < cmd.binding_count; ++index)
                encoder->setBuffer(buffers[index].buffer, static_cast<NS::UInteger>(buffers[index].offset), index + 1);
        }
        encoder->dispatchThreadgroups(
            MTL::Size(cmd.grid.groups[0], cmd.grid.groups[1], cmd.grid.groups[2]),
            MTL::Size(cmd.grid.threads[0], cmd.grid.threads[1], cmd.grid.threads[2]));
    }
    encoder->endEncoding();
    enc_guard.enc = nullptr;
    rollback.prepared = nullptr;
    return prepared;
}
void Batch::commit_and_wait(Timing* timing) {
    if (committed_) throw Error(SM_ESTATE, "GPU batch committed twice");
    auto& state = runtime();
    PreparedGpuBatch prepared;
    std::chrono::steady_clock::time_point entered, submitted;
    if (timing) entered = std::chrono::steady_clock::now();
    {
        AutoreleasePool pool;
        std::lock_guard submit_lock(state.submit_mutex);
        command_ = state.queue->commandBufferWithUnretainedReferences();
        if (!command_) throw Error(SM_EBACKEND, "Metal command buffer creation failed");
        command_->retain();
        prepared = encode_batch(nullptr, 0);
        BatchRollbackGuard rollback{&prepared};
        committed_ = true;
        submitted = std::chrono::steady_clock::now();
        command_->commit();
        rollback.prepared = nullptr;
    }
    {
        AutoreleasePool wait_pool;
        command_->waitUntilCompleted();
        const auto waited = std::chrono::steady_clock::now();
        const uint64_t host_wait_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(waited - submitted).count());
        record_host_wait(context_, host_wait_ns);
        const double seconds = command_->GPUEndTime() - command_->GPUStartTime();
        const uint64_t gpu_ns = seconds > 0 ? static_cast<uint64_t>(seconds * 1e9) : 0;
        record_gpu(context_, 0, 1, dispatches_, gpu_ns);
        if (command_->status() != MTL::CommandBufferStatusCompleted)
            throw Error(SM_EBACKEND, "Metal command buffer failed: " + describe(command_->error()));
        complete_sync_gpu_batch(context_, prepared);
        if (timing) {
            const auto done = std::chrono::steady_clock::now();
            timing->encode_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(submitted - entered).count());
            timing->gpu_ns = gpu_ns;
            timing->host_wait_ns = host_wait_ns;
            timing->complete_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(done - waited).count());
        }
    }
}
void Batch::commit_async(std::initializer_list<const sm_buffer*> outputs) {
    commit_async_impl(outputs.begin(), outputs.size());
}
void Batch::commit_async(const std::vector<const sm_buffer*>& outputs) {
    commit_async_impl(outputs.data(), outputs.size());
}
void Batch::commit_async_impl(const sm_buffer* const* outputs, size_t count) {
    if (committed_) throw Error(SM_ESTATE, "GPU batch committed twice");
    auto& state = runtime();
    std::shared_ptr<BatchSubmission> submission;
    {
        AutoreleasePool pool;
        std::lock_guard submit_lock(state.submit_mutex);
        const uint64_t ticket = state.next_ticket.fetch_add(1, std::memory_order_relaxed);
        if (!ticket) throw Error(SM_EOVERFLOW, "Metal shared event ticket space exhausted");
        command_ = state.queue->commandBuffer();
        if (!command_) throw Error(SM_EBACKEND, "Metal command buffer creation failed");
        command_->retain();
        PreparedGpuBatch prepared = encode_batch(outputs, count);
        BatchRollbackGuard rollback{&prepared};
        command_->encodeSignalEvent(state.event, ticket);
        auto pins = borrowed_pins_for_async(prepared, outputs, count);
        submission = std::make_shared<BatchSubmission>(state.device, command_, state.event, ticket,
                                                       context_state_handle(context_), std::move(pins));
        attach_async_gpu_batch(context_, prepared, outputs, count, state.event, ticket, submission);
        committed_ = true;
        command_->commit();
        rollback.prepared = nullptr;
    }
    record_gpu(context_, 0, 1, dispatches_, 0);
    register_async_gpu_submission(context_, submission);
}
} // namespace sm::detail::gpu
