#pragma once
// The process-wide Metal device shared by storage and compute. Only
// metal_storage.cpp defines the metal-cpp implementation macros.
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include "sm/detail/backend.hpp"

#include <memory>
#include <unistd.h>

namespace sm::detail {
// Captured when the library loads, without creating a Metal device. Check
// before touching a lazy static's guard, which could be inherited locked.
inline const pid_t metal_process = ::getpid();

inline void require_original_process() {
    if (::getpid() != metal_process) throw Error(SM_ESTATE, "Metal is unavailable after fork; use exec or spawn");
}

struct AutoreleasePool {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    AutoreleasePool() {
        if (!pool) throw Error(SM_EOOM, "Metal autorelease-pool allocation failed");
    }
    ~AutoreleasePool() {
        if (::getpid() == metal_process && pool) pool->release();
    }
    AutoreleasePool(const AutoreleasePool&) = delete;
    AutoreleasePool& operator=(const AutoreleasePool&) = delete;
};

struct DeviceOwner {
    MTL::Device* device = nullptr;
    uint64_t registry_id = 0;
    uint64_t max_buffer_length = 0;

    ~DeviceOwner() {
        // A child must not enter the inherited Objective-C/Metal runtime even
        // during teardown. These inherited resources are reclaimed by exit.
        if (::getpid() == metal_process && device) device->release();
    }
};

// Creates the system default device on first use; SM_ENOTSUP without one.
const std::shared_ptr<DeviceOwner>& shared_device();
} // namespace sm::detail
