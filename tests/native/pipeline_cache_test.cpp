#include "../../src/runtime/pipeline_cache.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "CHECK failed: %s:%d: %s\n", __FILE__, __LINE__, #x); std::abort(); } } while (0)

namespace {
struct Pipeline {
    std::atomic<int>& releases;
    void release() noexcept {
        ++releases;
        delete this;
    }
};
using sm::detail::gpu::cache_pipeline;
using Owner = sm::detail::gpu::PipelinePtr<Pipeline>;

struct Allocations {
    int calls = 0;
    int fail_on = -1;
};

template<class T>
struct FailingAllocator {
    using value_type = T;
    Allocations* state;
    explicit FailingAllocator(Allocations& value) noexcept : state(&value) {}
    template<class U>
    FailingAllocator(const FailingAllocator<U>& other) noexcept : state(other.state) {}
    T* allocate(size_t count) {
        if (state->calls++ == state->fail_on) throw std::bad_alloc();
        return std::allocator<T>{}.allocate(count);
    }
    void deallocate(T* pointer, size_t count) noexcept { std::allocator<T>{}.deallocate(pointer, count); }
    template<class U>
    bool operator==(const FailingAllocator<U>& other) const noexcept { return state == other.state; }
};

using Entry = std::pair<const std::string, Pipeline*>;
using Cache = std::unordered_map<std::string, Pipeline*, std::hash<std::string>, std::equal_to<std::string>,
                               FailingAllocator<Entry>>;

void insertion_failure_releases_pipeline() {
    // Discover the implementation's node/bucket allocation count, then fail
    // each allocation in turn to include failures partway through insertion.
    int allocation_count = 0;
    for (int fail_on = -1; fail_on < allocation_count; ++fail_on) {
        Allocations allocations{0, fail_on};
        Cache cache(0, std::hash<std::string>{}, std::equal_to<std::string>{}, FailingAllocator<Entry>(allocations));
        std::mutex mutex;
        std::atomic<int> releases{0};
        auto* candidate = new Pipeline{releases};
        try {
            CHECK(cache_pipeline(cache, mutex, "new_kernel", Owner(candidate)) == candidate);
            CHECK(fail_on == -1 && releases == 0 && cache.size() == 1);
            allocation_count = allocations.calls;
            CHECK(allocation_count > 0);
        } catch (const std::bad_alloc&) {
            CHECK(fail_on >= 0 && releases == 1 && cache.empty());
            // The failed insertion also leaves the lock and cache usable.
            allocations.fail_on = -1;
            candidate = new Pipeline{releases};
            CHECK(cache_pipeline(cache, mutex, "new_kernel", Owner(candidate)) == candidate);
            CHECK(releases == 1 && cache.size() == 1);
        }
        cache.at("new_kernel")->release();
        cache.clear();
        CHECK(releases == (fail_on == -1 ? 1 : 2));
    }
}

void insertion_failure_preserves_existing_pipeline() {
    Allocations allocations;
    Cache cache(0, std::hash<std::string>{}, std::equal_to<std::string>{}, FailingAllocator<Entry>(allocations));
    std::mutex mutex;
    std::atomic<int> releases{0};
    auto* existing = new Pipeline{releases};
    CHECK(cache_pipeline(cache, mutex, "existing", Owner(existing)) == existing);
    allocations.fail_on = allocations.calls;
    try {
        cache_pipeline(cache, mutex, "another", Owner(new Pipeline{releases}));
        CHECK(false);
    } catch (const std::bad_alloc&) {
        CHECK(releases == 1 && cache.size() == 1 && cache.at("existing") == existing);
    }
    existing->release();
    cache.clear();
    CHECK(releases == 2);
}

void concurrent_duplicates_keep_one_pipeline() {
    constexpr size_t count = 8;
    std::unordered_map<std::string, Pipeline*> cache;
    std::mutex cache_mutex, start_mutex;
    std::condition_variable ready;
    size_t waiting = 0;
    std::atomic<int> releases{0};
    std::array<Pipeline*, count> results{};
    std::vector<std::thread> threads;
    for (size_t index = 0; index < count; ++index) {
        threads.emplace_back([&, index] {
            Owner candidate(new Pipeline{releases});
            {
                // Every caller has built its pipeline before any insertion.
                std::unique_lock lock(start_mutex);
                ++waiting;
                ready.notify_all();
                ready.wait(lock, [&] { return waiting == count; });
            }
            results[index] = cache_pipeline(cache, cache_mutex, "same_kernel", std::move(candidate));
        });
    }
    for (auto& thread : threads) thread.join();
    CHECK(cache.size() == 1 && releases == count - 1);
    for (auto* result : results) CHECK(result == cache.at("same_kernel"));
    // Ownership of the winner stays in the cache until its explicit teardown.
    cache.at("same_kernel")->release();
    cache.clear();
    CHECK(releases == count);
}
} // namespace

int main() {
    insertion_failure_releases_pipeline();
    insertion_failure_preserves_existing_pipeline();
    concurrent_duplicates_keep_one_pipeline();
    std::puts("Pipeline cache allocation failure, ownership transfer and concurrent duplicate checks passed");
}
