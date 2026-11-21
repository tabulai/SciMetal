#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "metal_device.hpp"
#include "metal_cache_policy.hpp"
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_statistics.h>

#include <chrono>
#include <limits>
#include <memory>
#include <utility>
#include <unistd.h>

namespace sm::detail {
static_assert(VM_PAGE_QUERY_PAGE_PRESENT == 0x1 && VM_PAGE_QUERY_PAGE_PAGED_OUT == 0x10 &&
              VM_PAGE_QUERY_PAGE_REF == 0x4 && VM_PAGE_QUERY_PAGE_DIRTY == 0x8 && VM_PAGE_QUERY_PAGE_COPIED == 0x20);
const std::shared_ptr<DeviceOwner>& shared_device() {
    require_original_process();
    static const auto owner = [] {
        auto result = std::make_shared<DeviceOwner>();
        result->device = MTL::CreateSystemDefaultDevice();
        if (!result->device)
            throw Error(SM_ENOTSUP, "No Metal device is available for shared storage");
        result->registry_id = result->device->registryID();
        result->max_buffer_length = result->device->maxBufferLength();
        if (!result->registry_id || !result->max_buffer_length)
            throw Error(SM_EBACKEND, "Metal device returned invalid allocation capabilities");
        return result;
    }();
    return owner;
}

namespace {
struct BufferOwner {
    std::shared_ptr<DeviceOwner> device;
    MTL::Buffer* buffer = nullptr;

    ~BufferOwner() {
        if (::getpid() == metal_process && buffer) buffer->release();
        // The shared device owner survives the buffer release and any static
        // cache teardown; no Python object, callback or GIL is involved.
    }
};
} // namespace

bool metal_event_complete(void* event, uint64_t value) noexcept {
    if (!event || !value) return true;
    if (::getpid() != metal_process) return false;
    return static_cast<MTL::SharedEvent*>(event)->signaledValue() >= value;
}

uint64_t wait_metal_event(void* event, uint64_t value) {
    require_original_process();
    if (!event || !value) return 0;
    AutoreleasePool pool;
    auto* shared = static_cast<MTL::SharedEvent*>(event);
    if (shared->signaledValue() >= value) return 0;
    const auto start = std::chrono::steady_clock::now();
    if (!shared->waitUntilSignaledValue(value, 60000) && shared->signaledValue() < value)
        throw Error(SM_ETIMEOUT, "timed out waiting on Metal shared event");
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

void verify_metal_import(const sm_buffer_view& view) {
    require_original_process();
    AutoreleasePool pool;
    auto device = shared_device();
    if (view.device_id != device->registry_id)
        throw Error(SM_ENOTSUP, "foreign Metal device is unsupported");
    auto* buffer = static_cast<MTL::Buffer*>(view.metal_buffer);
    if (!buffer->device() || buffer->device()->registryID() != device->registry_id ||
        buffer->length() != view.capacity || buffer->storageMode() != MTL::StorageModeShared)
        throw Error(SM_ENOTSUP, "Metal descriptor requires a shared MTLBuffer on the system device");
    if ((view.flags & SM_BUFFER_HOST_READABLE) && buffer->contents() != view.host_base)
        throw Error(SM_EINVAL, "Metal descriptor host_base does not match MTLBuffer contents");
    if (view.completion_event) {
        auto* event = static_cast<MTL::SharedEvent*>(view.completion_event);
        if (MTL::Device* event_device = event->device();
            event_device && event_device->registryID() != device->registry_id)
            throw Error(SM_ENOTSUP, "Metal completion event belongs to another device");
    }
}

void discard_cached_metal(RawAllocation& raw) {
    require_original_process();
    if (raw.in_flight_sync) {
        raw.in_flight_sync->wait(nullptr);
        raw.in_flight_sync.reset();
        raw.in_flight_event = nullptr;
        raw.in_flight_value = 0;
    } else if (raw.in_flight_event && raw.in_flight_value) {
        (void)wait_metal_event(raw.in_flight_event, raw.in_flight_value);
        raw.in_flight_event = nullptr;
        raw.in_flight_value = 0;
    }
    if (!raw.owner || raw.owner.use_count() != 1 || !raw.metal || !raw.host || !raw.capacity || !raw.device_id)
        throw Error(SM_ESTATE, "discard requires exclusive owned Metal storage");
    auto* buffer = static_cast<MTL::Buffer*>(raw.metal);
    if (buffer->length() != raw.capacity || buffer->device()->registryID() != raw.device_id ||
        buffer->storageMode() != MTL::StorageModeShared ||
        buffer->hazardTrackingMode() != MTL::HazardTrackingModeTracked ||
        buffer->cpuCacheMode() != MTL::CPUCacheModeDefaultCache)
        throw Error(SM_EBACKEND, "cached Metal resource properties changed");
    if (buffer->setPurgeableState(MTL::PurgeableStateKeepCurrent) != MTL::PurgeableStateNonVolatile)
        throw Error(SM_EBACKEND, "cached Metal resource was unexpectedly purgeable");
    // The setter returns the prior state; query the resulting state separately.
    if (buffer->setPurgeableState(MTL::PurgeableStateEmpty) != MTL::PurgeableStateNonVolatile ||
        buffer->setPurgeableState(MTL::PurgeableStateKeepCurrent) != MTL::PurgeableStateEmpty)
        throw Error(SM_EBACKEND, "cached Metal resource did not enter the empty state");
    if (buffer->setPurgeableState(MTL::PurgeableStateNonVolatile) != MTL::PurgeableStateEmpty ||
        buffer->setPurgeableState(MTL::PurgeableStateKeepCurrent) != MTL::PurgeableStateNonVolatile)
        throw Error(SM_EBACKEND, "cached Metal resource did not return to nonvolatile state");
    void* host = buffer->contents();
    if (!host || reinterpret_cast<uintptr_t>(host) % 64 || buffer->length() != raw.capacity ||
        buffer->device()->registryID() != raw.device_id || buffer->allocatedSize() > raw.capacity)
        throw Error(SM_EBACKEND, "restored Metal resource has incompatible storage");
    raw.host = host;
}

void prepare_cached_metal(RawAllocation& raw) {
    require_original_process();
    if (raw.capacity < cached_metal_sample_min_capacity) return;
    if (!metal_event_complete(raw.in_flight_event, raw.in_flight_value)) return;
    if (raw.in_flight_sync) {
        raw.in_flight_sync->wait(nullptr);
        raw.in_flight_sync.reset();
    }
    raw.in_flight_event = nullptr;
    raw.in_flight_value = 0;
    if (!raw.owner || raw.owner.use_count() != 1 || !raw.metal || !raw.host || !raw.device_id)
        throw Error(SM_ESTATE, "cache checkout requires exclusive owned Metal storage");
    const auto queried_page = sysconf(_SC_PAGESIZE);
    if (queried_page < 64 || (queried_page & (queried_page - 1))) return;
    const auto page = static_cast<uint64_t>(queried_page);
    std::array<uint64_t, cached_metal_sample_count> offsets{};
    std::array<int32_t, cached_metal_sample_count> statuses{};
    std::array<uint32_t, cached_metal_sample_count> dispositions{};
    for (size_t index = 0; index < cached_metal_sample_count; ++index) {
        const auto offset = cached_metal_sample_offset(reinterpret_cast<uintptr_t>(raw.host), raw.capacity, page, index);
        if (!offset) return;
        offsets[index] = *offset;
    }
    // Inspect metadata only. Query all samples before deciding: a late failure
    // or ambiguous observation must veto earlier evidence of paged-out storage.
    for (size_t index = 0; index < cached_metal_sample_count; ++index) {
        integer_t flags = 0, references = 0;
        statuses[index] = mach_vm_page_query(mach_task_self(),
            reinterpret_cast<mach_vm_offset_t>(raw.host) + offsets[index], &flags, &references);
        dispositions[index] = static_cast<uint32_t>(flags);
    }
    if (cached_metal_pages_need_reset(statuses, dispositions)) discard_cached_metal(raw);
}

RawAllocation allocate_metal(uint64_t capacity) {
    require_original_process();
    if (!capacity) throw Error(SM_EINVAL, "Metal allocation capacity must be positive");
    if (capacity > std::numeric_limits<NS::UInteger>::max())
        throw Error(SM_EOVERFLOW, "Metal allocation capacity exceeds NSUInteger");

    AutoreleasePool pool;
    auto device = shared_device();
    if (capacity > device->max_buffer_length)
        throw Error(SM_EOOM, "Metal allocation exceeds device maxBufferLength");

    // Allocate the C++ owner before acquiring the native object, so an owner
    // allocation failure cannot leak a retained MTLBuffer.
    auto owner = std::make_shared<BufferOwner>();
    owner->device = std::move(device);
    constexpr MTL::ResourceOptions options = MTL::ResourceStorageModeShared |
        MTL::ResourceHazardTrackingModeTracked | MTL::ResourceCPUCacheModeDefaultCache;
    owner->buffer = owner->device->device->newBuffer(static_cast<NS::UInteger>(capacity), options);
    if (!owner->buffer) throw Error(SM_EOOM, "Metal shared-buffer allocation failed");

    void* host = owner->buffer->contents();
    if (!host || reinterpret_cast<uintptr_t>(host) % 64 ||
        owner->buffer->length() != capacity ||
        owner->buffer->storageMode() != MTL::StorageModeShared ||
        owner->buffer->hazardTrackingMode() != MTL::HazardTrackingModeTracked)
        throw Error(SM_EBACKEND, "Metal returned an incompatible shared-buffer allocation");
    // Metal can round a non-page-aligned length up internally. The runtime
    // reserves page-rounded Metal capacities; never undercharge extra bytes.
    if (owner->buffer->allocatedSize() > capacity)
        throw Error(SM_EBACKEND, "Metal allocatedSize exceeds the reserved capacity");

    return {owner, host, owner->buffer, owner->device->registry_id, capacity, nullptr, 0, nullptr};
}

RawAllocation wrap_metal_pages(const void* base, uint64_t capacity, std::shared_ptr<void> producer) {
    require_original_process();
    const auto first = reinterpret_cast<mach_vm_address_t>(base);
    mach_vm_address_t region = first;
    mach_vm_size_t size = 0;
    natural_t depth = UINT32_MAX;
    vm_region_submap_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
    const auto status = mach_vm_region_recurse(mach_task_self(), &region, &size, &depth,
        reinterpret_cast<vm_region_recurse_info_t>(&info), &count);
    if (status != KERN_SUCCESS || info.is_submap || !(info.protection & VM_PROT_READ) ||
        region > first || first - region > size || capacity > size - (first - region))
        throw Error(SM_ENOTSUP, "no-copy pages must be in one readable VM region");

    AutoreleasePool pool;
    auto owner = std::make_shared<BufferOwner>();
    owner->device = shared_device();
    if (capacity > owner->device->max_buffer_length)
        throw Error(SM_ENOTSUP, "no-copy span exceeds Metal maxBufferLength");
    constexpr auto options = MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;
    // Metal may retain this resource past the last column or descriptor.
    // Its deallocator block keeps the source alive until Metal is actually
    // done. The allocation also pins the producer while it releases the
    // buffer, keeping any synchronous foreign callback outside the state lock.
    owner->buffer = owner->device->device->newBuffer(base, capacity, options,
        ^(void*, NS::UInteger) { (void)producer; });
    if (!owner->buffer || owner->buffer->contents() != base || owner->buffer->length() != capacity ||
        owner->buffer->storageMode() != MTL::StorageModeShared || owner->buffer->allocatedSize() > capacity)
        throw Error(SM_ENOTSUP, "Metal could not wrap these immutable pages without copying");
    return {owner, owner->buffer->contents(), owner->buffer, owner->device->registry_id, capacity, nullptr, 0, nullptr};
}
} // namespace sm::detail
