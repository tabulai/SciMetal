#pragma once
// Private ownership transfer used by the Metal pipeline cache. Keeping this
// independent of Metal lets tests inject cache allocation failures.
#include <memory>
#include <mutex>

namespace ms::detail::gpu {
template<class Pipeline>
struct PipelineDeleter {
    void operator()(Pipeline* pipeline) const noexcept { pipeline->release(); }
};

template<class Pipeline>
using PipelinePtr = std::unique_ptr<Pipeline, PipelineDeleter<Pipeline>>;

template<class Cache, class Pipeline>
Pipeline* cache_pipeline(Cache& cache, std::mutex& mutex, const char* name, PipelinePtr<Pipeline> result) {
    std::lock_guard lock(mutex);
    const auto [entry, inserted] = cache.emplace(name, result.get());
    // Transfer only after insertion succeeds. A duplicate or an exception
    // leaves the new pipeline owned by result and releases it on scope exit.
    if (inserted) result.release();
    return entry->second;
}
} // namespace ms::detail::gpu
