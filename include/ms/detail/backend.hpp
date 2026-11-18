#pragma once
#include "ms_runtime.h"
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace ms::detail {
class Error : public std::runtime_error {
public:
    ms_status status;
    Error(ms_status code, const std::string& message)
        : std::runtime_error(message), status(code) {}
};
struct AsyncSync {
    virtual ~AsyncSync() = default;
    virtual void wait(ms_context* waiter) = 0;
    virtual bool completed() noexcept = 0;
};
struct RawAllocation {
    std::shared_ptr<void> owner;
    void* host = nullptr;
    void* metal = nullptr;
    uint64_t device_id = 0;
    uint64_t capacity = 0;
    void* in_flight_event = nullptr;
    uint64_t in_flight_value = 0;
    std::shared_ptr<AsyncSync> in_flight_sync;
};
// Called only after checked capacity and budget reservation. Never called for
// CPU storage. A disabled build returns MS_ENOTSUP without driver initialization.
RawAllocation allocate_metal(uint64_t capacity);
// Only for exclusively owned, unpublished Metal storage. Cache checkout calls
// prepare after removing the raw allocation and releasing the cache mutex,
// before any host initialization. Full capacity stays charged throughout.
void prepare_cached_metal(RawAllocation&);
void discard_cached_metal(RawAllocation&);
// Wrap a verified, page-aligned span without transferring ownership of the pages.
RawAllocation wrap_metal_pages(const void* base, uint64_t capacity, std::shared_ptr<void> producer);
// Shared-event synchronization and foreign Metal descriptor verification.
bool metal_event_complete(void* event, uint64_t value) noexcept;
uint64_t wait_metal_event(void* event, uint64_t value);
void verify_metal_import(const ms_buffer_view& view);
}
