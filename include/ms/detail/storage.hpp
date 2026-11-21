#pragma once
#include "ms/detail/backend.hpp"
#include <cstdint>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace ms::detail {
struct BufferDeleter {
    void operator()(ms_buffer* buffer) const noexcept { ms_buffer_release(buffer); }
};
using BufferPtr = std::unique_ptr<ms_buffer, BufferDeleter>;
struct ReadView { const uint8_t* data; uint64_t size; };
// What a new owned buffer's logical bytes hold; its padding is always zero.
// `unfilled` is only for producers that write every logical byte before
// publishing: the bytes are uninitialized, or 0xA5 in poisoned contexts.
// `device_zero` and `device_unfilled` defer initialization to a GPU blit
// command encoder when first consumed by a GPU batch (or materialize on CPU
// if read/mutated on the host before any GPU batch claims them).
enum class Fill { zero, unfilled, device_zero, device_unfilled };

class Reservation {
public:
    struct Impl;
    Reservation() noexcept = default;
    explicit Reservation(Impl* value) noexcept : impl_(value) {}
    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;
    Reservation(Reservation&& other) noexcept : impl_(std::exchange(other.impl_, nullptr)) {}
    Reservation& operator=(Reservation&&) noexcept;
    ~Reservation();
private:
    Impl* impl_ = nullptr;
    friend BufferPtr allocate_reserved(ms_context*, Reservation&, uint64_t, Fill);
    friend BufferPtr wrap_immutable(ms_context*, const void*, uint64_t, const void*, uint64_t, std::shared_ptr<void>);
};

uint64_t padded_capacity(uint64_t logical, ms_context* context = nullptr);
Reservation reserve(ms_context*, uint64_t bytes);
// Metal storage may reuse an idle cached buffer of the same capacity (or up to
// 1.5x capacity for allocations >= 1 MiB).
BufferPtr allocate_reserved(ms_context*, Reservation&, uint64_t logical, Fill = Fill::zero);
// Suballocates 256-byte-aligned scratch slices (up to 32 KiB) from persistent
// 256 KiB shared Metal arena slabs when budget/cache headroom permits; otherwise
// falls back to regular page-rounded allocation.
BufferPtr allocate_scratch(ms_context*, uint64_t logical, Fill = Fill::device_zero);
bool is_scratch_slice(const ms_buffer*);
// Only entire VM pages within the producer's witnessed immutable allocation
// may be exposed to Metal. The logical view can start at an interior byte.
BufferPtr wrap_immutable(ms_context*, const void* bytes, uint64_t length, const void* owner_base,
                         uint64_t owner_size, std::shared_ptr<void> producer);
ReadView read(const ms_buffer*, ms_context* expected = nullptr);
// Validates buffer handle and optional context membership without waiting on
// an in-flight GPU producer or materializing deferred device fills.
uint64_t buffer_size(const ms_buffer*, ms_context* expected = nullptr);
uint64_t owned_capacity(const ms_buffer*);
// True only for a complete native allocation's logical span (excluding padding).
bool whole_owned_buffer(const ms_buffer*, uint64_t logical_bytes);
// The native Metal resource behind a buffer: offset is the buffer's first byte
// within that resource. MS_ENOTSUP for storage without a Metal resource.
struct DeviceView { void* resource; uint64_t offset; uint64_t size; };
DeviceView device_view(const ms_buffer*, ms_context* expected);
uint32_t storage_kind(ms_context*);
uint64_t available_headroom_bytes(ms_context*) noexcept;
struct BufferResidency {
    bool metal_owned = false;
    bool async_in_flight = false;
    uint64_t size = 0;
};
BufferResidency buffer_residency(const ms_buffer*) noexcept;

struct BlitFill {
    void* metal_buffer = nullptr;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint8_t value = 0;
};
struct EventWait {
    void* event = nullptr;
    uint64_t value = 0;
};
struct ClaimedPrior {
    std::shared_ptr<void> alloc;
    void* event = nullptr;
    uint64_t value = 0;
};
struct PreparedGpuBatch {
    std::vector<EventWait> waits;
    std::vector<BlitFill> fills;
    std::vector<std::shared_ptr<void>> touched;
    std::vector<std::shared_ptr<void>> claimed;
    std::vector<ClaimedPrior> cleared_priors;
};
PreparedGpuBatch prepare_gpu_batch(ms_context*, const std::vector<const ms_buffer*>& bound_buffers,
                                   const std::vector<void*>& indirect_metal_buffers);
void rollback_gpu_batch(const PreparedGpuBatch&) noexcept;
void complete_sync_gpu_batch(ms_context*, const PreparedGpuBatch&) noexcept;
std::vector<std::shared_ptr<void>> borrowed_pins_for_async(const PreparedGpuBatch&, const ms_buffer* const* outputs,
                                                           size_t output_count);
std::shared_ptr<void> context_state_handle(ms_context*);
void attach_async_gpu_batch(ms_context*, const PreparedGpuBatch&, const ms_buffer* const* outputs,
                            size_t output_count, void* event, uint64_t ticket,
                            const std::shared_ptr<AsyncSync>& sync);
void register_async_gpu_submission(ms_context*, const std::shared_ptr<AsyncSync>& sync);
void retire_async_gpu(const std::shared_ptr<void>& state_handle, ms_context* waiter,
                      uint64_t gpu_ns, uint64_t wait_ns) noexcept;
void flush_async_gpu(ms_context*);

// GPU execution counters for ms_context_execution_info.
// Host time from a command buffer's commit until its wait returned.
void record_host_wait(ms_context*, uint64_t nanoseconds) noexcept;
void record_gpu(ms_context*, uint64_t operations, uint64_t command_buffers, uint64_t dispatches,
                uint64_t gpu_ns) noexcept;
// Scopes a public C ABI call so nested GPU helpers accumulate dispatches and
// command buffers while counting as a single completed gpu_operation on success.
class GpuOperationScope {
public:
    GpuOperationScope(ms_context* context, bool active) noexcept;
    GpuOperationScope(const GpuOperationScope&) = delete;
    GpuOperationScope& operator=(const GpuOperationScope&) = delete;
    ~GpuOperationScope() noexcept;
    void complete() noexcept;
private:
    ms_context* context_ = nullptr;
    bool outermost_ = false;
};
// Only newly allocated, unpublished buffers may be mutated by native builders.
uint8_t* mutable_data(ms_buffer*);
void check_context(ms_context*);
void set_error(ms_context*, ms_status, const char*) noexcept;

template<class Function> ms_status api(ms_context* context, Function&& function) noexcept {
    try {
        check_context(context);
        std::forward<Function>(function)();
        set_error(context, MS_OK, "");
        return MS_OK;
    } catch (const Error& error) {
        set_error(context, error.status, error.what());
        return error.status;
    } catch (const std::bad_alloc&) {
        set_error(context, MS_EOOM, "native allocation failed");
        return MS_EOOM;
    } catch (const std::exception& error) {
        set_error(context, MS_EBACKEND, error.what());
        return MS_EBACKEND;
    } catch (...) {
        set_error(context, MS_EBACKEND, "unknown native exception");
        return MS_EBACKEND;
    }
}
}
