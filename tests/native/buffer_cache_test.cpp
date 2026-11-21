// The context buffer cache: Metal storage reuses idle buffers of the same
// capacity, re-initializes them like new ones and keeps them charged until
// eviction. CPU storage never caches. Run with --metal for the Metal cases.
#include "sm_runtime.h"
#include "sm/detail/storage.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s failed line %d: %s\n", __FILE__, __LINE__, #x); std::abort(); } } while (0)

namespace {
using Bytes = std::vector<uint8_t>;
const uint64_t page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE)); // Metal capacities round to pages

sm_context* create(uint32_t kind, uint64_t limit, uint32_t flags = 0, uint64_t external = 0) {
    sm_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = SM_RUNTIME_ABI_VERSION;
    options.storage_kind = kind;
    options.flags = flags;
    options.memory_limit_bytes = limit;
    options.external_reservation_bytes = external;
    sm_context* result = nullptr;
    CHECK(sm_context_create(&options, &result) == SM_OK);
    return result;
}
struct Context {
    sm_context* handle;
    explicit Context(uint32_t kind, uint64_t limit = 64 << 20, uint32_t flags = 0) : handle(create(kind, limit, flags)) {}
    ~Context() { sm_context_release(handle); }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
};
sm_memory_info memory(sm_context* ctx) {
    sm_memory_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = SM_RUNTIME_ABI_VERSION;
    CHECK(sm_context_memory_info(ctx, &info) == SM_OK);
    CHECK(info.total_bytes == info.owned_bytes + info.borrowed_bytes + info.reserved_bytes + info.cached_bytes +
                                  info.external_bytes);
    CHECK(info.total_bytes <= info.limit_bytes && info.peak_bytes <= info.limit_bytes);
    return info;
}
sm_execution_info counters(sm_context* ctx) {
    sm_execution_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = SM_RUNTIME_ABI_VERSION;
    CHECK(sm_context_execution_info(ctx, &info) == SM_OK);
    return info;
}
uint64_t cache_limit(sm_context* ctx) {
    uint64_t value = 1;
    CHECK(sm_context_cache_limit(ctx, &value) == SM_OK);
    return value;
}
sm_buffer* zeros(sm_context* ctx, uint64_t bytes) {
    sm_buffer* result = nullptr;
    CHECK(sm_buffer_zeros(ctx, bytes, &result) == SM_OK);
    return result;
}
sm_buffer* copy(sm_context* ctx, const Bytes& data) {
    sm_buffer* result = nullptr;
    CHECK(sm_buffer_copy_from_host(ctx, data.data(), data.size(), &result) == SM_OK);
    return result;
}
Bytes bytes(const sm_buffer* buffer) {
    const void* data = nullptr;
    uint64_t size = 0;
    CHECK(sm_buffer_data(buffer, &data, &size) == SM_OK);
    const auto* begin = static_cast<const uint8_t*>(data);
    return size ? Bytes(begin, begin + size) : Bytes{};
}
// The allocation behind a buffer, read through a short-lived export.
struct Backing {
    const void* metal;
    const uint8_t* host;
    uint64_t capacity, id;
    Bytes all; // every capacity byte, padding included
};
Backing backing(const sm_buffer* buffer) {
    sm_buffer_view view{};
    CHECK(sm_buffer_export(buffer, &view) == SM_OK);
    const auto* host = static_cast<const uint8_t*>(view.host_base);
    Backing result{view.metal_buffer, host, view.capacity, view.allocation_id, Bytes(host, host + view.capacity)};
    view.release(view.owner);
    return result;
}
// Logical sizes whose padded capacity is exactly `pages` pages.
uint64_t logical(uint64_t pages) { return pages * page - 64; }
bool all_equal(const uint8_t* begin, const uint8_t* end, uint8_t value) {
    for (; begin != end; ++begin)
        if (*begin != value) return false;
    return true;
}

void limits_and_options() {
    {
        Context ctx(SM_BUFFER_METAL);
        CHECK(cache_limit(ctx.handle) == (64u << 20) / 4); // a quarter of the budget by default
        CHECK(sm_context_set_cache_limit(ctx.handle, 5 * page) == SM_OK && cache_limit(ctx.handle) == 5 * page);
        CHECK(sm_context_set_cache_limit(ctx.handle, 0) == SM_OK && cache_limit(ctx.handle) == 0);
        CHECK(sm_context_cache_limit(ctx.handle, nullptr) == SM_EINVAL);
    }
    auto* external = create(SM_BUFFER_METAL, 64 << 20, 0, 16 << 20);
    CHECK(cache_limit(external) == (48u << 20) / 4); // the budget left after external headroom
    sm_context_release(external);
    uint64_t value = 0;
    CHECK(sm_context_cache_limit(nullptr, &value) == SM_EINVAL);
    CHECK(sm_context_set_cache_limit(nullptr, 0) == SM_EINVAL);
    CHECK(sm_context_trim(nullptr) == SM_EINVAL);

    sm_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = SM_RUNTIME_ABI_VERSION;
    options.storage_kind = SM_BUFFER_METAL;
    options.memory_limit_bytes = 1 << 20;
    sm_context* rejected = nullptr;
    options.flags = SM_CONTEXT_POISON_UNWRITTEN << 1; // undefined bits are refused
    CHECK(sm_context_create(&options, &rejected) == SM_EINVAL && !rejected);
    options.flags = SM_CONTEXT_POISON_UNWRITTEN;
    CHECK(sm_context_create(&options, &rejected) == SM_OK && rejected);
    sm_context_release(rejected);
}

void automatic_limit_retains_a_multiphase_working_set() {
    Context ctx(SM_BUFFER_METAL, 24 * page);
    CHECK(cache_limit(ctx.handle) == 6 * page);
    const auto before = counters(ctx.handle);
    // Scratch dies before the two outputs are allocated. The maximum live
    // footprint is eight pages, but all eleven pages recur in the next call.
    for (int round = 0; round < 3; ++round) {
        auto* scratch = zeros(ctx.handle, logical(3));
        sm_buffer_release(scratch);
        auto* a = zeros(ctx.handle, logical(4));
        auto* b = zeros(ctx.handle, logical(4));
        CHECK(memory(ctx.handle).owned_bytes == 8 * page);
        CHECK(memory(ctx.handle).cached_bytes == 3 * page);
        CHECK(cache_limit(ctx.handle) == 11 * page);
        sm_buffer_release(a);
        sm_buffer_release(b);
        const auto info = memory(ctx.handle);
        CHECK(info.owned_bytes == 0 && info.cached_bytes == 11 * page);
        CHECK(counters(ctx.handle).native_allocations == before.native_allocations + 3);
        CHECK(counters(ctx.handle).cached_reuses == before.cached_reuses + round * 3);
    }
    CHECK(sm_context_trim(ctx.handle) == SM_OK);
    CHECK(memory(ctx.handle).total_bytes == 0 && cache_limit(ctx.handle) == 11 * page);
    sm_buffer_release(zeros(ctx.handle, logical(3)));
    CHECK(memory(ctx.handle).cached_bytes == 3 * page && cache_limit(ctx.handle) == 11 * page);
}

void automatic_limit_excludes_reservations_and_foreign_storage() {
    Context ctx(SM_BUFFER_METAL, 16 * page);
    CHECK(cache_limit(ctx.handle) == 4 * page);
    {
        // Reservations admit future bytes, not reusable physical storage.
        auto pending = sm::detail::reserve(ctx.handle, 10 * page);
        CHECK(memory(ctx.handle).reserved_bytes == 10 * page);
        auto* small = zeros(ctx.handle, logical(1));
        CHECK(cache_limit(ctx.handle) == 4 * page);
        sm_buffer_release(small);
        CHECK(cache_limit(ctx.handle) == 4 * page);
    }
    CHECK(memory(ctx.handle).reserved_bytes == 0);
    const auto before = counters(ctx.handle);
    auto* sentinel = reinterpret_cast<sm_buffer*>(uintptr_t{0x40});
    sm_buffer* refused = sentinel;
    CHECK(sm_buffer_zeros(ctx.handle, logical(17), &refused) == SM_EOOM && refused == sentinel);
    CHECK(cache_limit(ctx.handle) == 4 * page && memory(ctx.handle).reserved_bytes == 0);
    CHECK(counters(ctx.handle).native_allocations == before.native_allocations);

    // Importing a tiny slice temporarily admits the entire foreign capacity.
    // That borrowed storage must not raise the owned-storage cache target.
    Bytes source(10 * page, 0x31);
    unsigned refs = 1;
    sm_buffer_view view{};
    view.struct_size = sizeof(view);
    view.abi_version = SM_BUFFER_ABI_VERSION;
    view.kind = SM_BUFFER_CPU;
    view.flags = SM_BUFFER_HOST_READABLE | SM_BUFFER_IMMUTABLE | SM_BUFFER_PRODUCER_COMPLETE;
    view.allocation_id = 1000000;
    view.host_base = source.data();
    view.byte_length = 64;
    view.capacity = source.size();
    view.owner = &refs;
    view.retain = [](void* owner) { ++*static_cast<unsigned*>(owner); };
    view.release = [](void* owner) { --*static_cast<unsigned*>(owner); };
    sm_import_result imported{};
    CHECK(sm_buffer_import(ctx.handle, &view, SM_COPY_ALWAYS, &imported) == SM_OK);
    CHECK(refs == 1 && imported.copied_bytes == 64 && imported.borrowed_bytes == 0);
    CHECK(bytes(imported.buffer) == Bytes(64, 0x31));
    CHECK(memory(ctx.handle).owned_bytes == page && memory(ctx.handle).borrowed_bytes == 0);
    CHECK(cache_limit(ctx.handle) == 4 * page);
    sm_buffer_release(imported.buffer);
}

void automatic_ceiling_and_explicit_limits() {
    auto* ctx = create(SM_BUFFER_METAL, 20 * page, 0, 4 * page);
    CHECK(cache_limit(ctx) == 4 * page); // available budget, after external headroom
    auto* large = zeros(ctx, logical(9));
    CHECK(cache_limit(ctx) == 8 * page); // at most half of the sixteen available pages
    sm_buffer_release(large);
    CHECK(memory(ctx).cached_bytes == 0); // one buffer larger than the ceiling

    CHECK(sm_context_set_cache_limit(ctx, 3 * page) == SM_OK);
    sm_buffer_release(zeros(ctx, logical(7)));
    CHECK(cache_limit(ctx) == 3 * page && memory(ctx).cached_bytes == 0);
    CHECK(sm_context_trim(ctx) == SM_OK && cache_limit(ctx) == 3 * page);
    CHECK(sm_context_set_cache_limit(ctx, 0) == SM_OK);
    sm_buffer_release(zeros(ctx, logical(9)));
    CHECK(cache_limit(ctx) == 0 && memory(ctx).cached_bytes == 0);

    // Explicit limits above the automatic ceiling keep their existing meaning.
    CHECK(sm_context_set_cache_limit(ctx, 10 * page) == SM_OK);
    sm_buffer_release(zeros(ctx, logical(9)));
    CHECK(cache_limit(ctx) == 10 * page && memory(ctx).cached_bytes == 9 * page);
    CHECK(sm_context_trim(ctx) == SM_OK && memory(ctx).total_bytes == 4 * page);
    sm_context_release(ctx);
}

void automatic_growth_preserves_live_exports() {
    Context ctx(SM_BUFFER_METAL, 16 * page);
    auto* first = zeros(ctx.handle, logical(5));
    const auto original = backing(first).metal;
    sm_buffer_view exported{};
    CHECK(sm_buffer_export(first, &exported) == SM_OK);
    sm_buffer_release(first);
    CHECK(cache_limit(ctx.handle) == 5 * page && memory(ctx.handle).cached_bytes == 0);
    auto* second = zeros(ctx.handle, logical(5));
    CHECK(backing(second).metal != original && cache_limit(ctx.handle) == 8 * page);
    CHECK(memory(ctx.handle).owned_bytes == 10 * page && memory(ctx.handle).cached_bytes == 0);
    exported.release(exported.owner);
    CHECK(memory(ctx.handle).owned_bytes == 5 * page && memory(ctx.handle).cached_bytes == 5 * page);
    auto* third = zeros(ctx.handle, logical(5));
    CHECK(backing(third).metal == original && memory(ctx.handle).cached_bytes == 0);
    CHECK(memory(ctx.handle).owned_bytes == 10 * page);
    sm_buffer_release(second);
    sm_buffer_release(third);
    CHECK(memory(ctx.handle).cached_bytes == 5 * page && cache_limit(ctx.handle) == 8 * page);
}

void reuse_by_capacity() {
    Context ctx(SM_BUFFER_METAL);
    const auto start = counters(ctx.handle);
    auto* first = zeros(ctx.handle, logical(3));
    const auto original = backing(first);
    CHECK(original.capacity == 3 * page);
    sm_buffer_release(first);
    auto info = memory(ctx.handle);
    CHECK(info.owned_bytes == 0 && info.cached_bytes == 3 * page && info.allocation_count == 0);
    CHECK(info.total_bytes == 3 * page && info.reserved_bytes == 0);

    // The same page-rounded capacity is served from the cache under a new identity.
    auto* second = zeros(ctx.handle, logical(3) - 100);
    const auto reused = backing(second);
    CHECK(reused.metal == original.metal && reused.host == original.host && reused.capacity == 3 * page);
    CHECK(reused.id != original.id);
    auto after = counters(ctx.handle);
    CHECK(after.native_allocations == start.native_allocations + 1 && after.cached_reuses == start.cached_reuses + 1);
    info = memory(ctx.handle);
    CHECK(info.cached_bytes == 0 && info.owned_bytes == 3 * page && info.allocation_count == 1);

    // Another capacity allocates natively, even with a larger buffer cached.
    sm_buffer_release(second);
    auto* smaller = zeros(ctx.handle, logical(2));
    CHECK(backing(smaller).metal != original.metal);
    after = counters(ctx.handle);
    CHECK(after.native_allocations == start.native_allocations + 2 && after.cached_reuses == start.cached_reuses + 1);
    sm_buffer_release(smaller);
    CHECK(memory(ctx.handle).cached_bytes == 5 * page);

    // Among equal capacities, the most recently cached buffer is reused first.
    auto* a = zeros(ctx.handle, logical(1));
    auto* b = zeros(ctx.handle, logical(1));
    const auto newest = backing(b).metal;
    sm_buffer_release(a);
    sm_buffer_release(b);
    auto* c = zeros(ctx.handle, logical(1));
    CHECK(backing(c).metal == newest);
    sm_buffer_release(c);
}

void reused_storage_is_reinitialized() {
    Context ctx(SM_BUFFER_METAL);
    const Bytes dirty(logical(2), 0x5a);
    auto* first = copy(ctx.handle, dirty);
    const auto original = backing(first);
    CHECK(all_equal(original.all.data() + dirty.size(), original.all.data() + original.capacity, 0));
    sm_buffer_release(first);

    // A shorter zeroed request reads zero across the whole capacity, including
    // formerly dirty bytes that are now padding.
    auto* zeroed = zeros(ctx.handle, logical(2) - 5000);
    const auto cleared = backing(zeroed);
    CHECK(cleared.metal == original.metal && all_equal(cleared.all.data(), cleared.all.data() + cleared.capacity, 0));
    sm_buffer_release(zeroed);

    // A shorter copy rewrites its bytes; stale bytes beyond it become zero padding.
    sm_buffer_release(copy(ctx.handle, dirty));
    const Bytes shorter(logical(2) - 5000, 0x11);
    auto* rewritten = copy(ctx.handle, shorter);
    const auto padded = backing(rewritten);
    CHECK(padded.metal == original.metal && padded.capacity == 2 * page);
    CHECK(bytes(rewritten) == shorter);
    CHECK(all_equal(padded.all.data() + shorter.size(), padded.all.data() + padded.capacity, 0));
    sm_buffer_release(rewritten);
}

// Poison fills storage a producer promised to overwrite; correct producers
// hide it completely, and zeroed storage is unaffected.
void poison_is_invisible_to_complete_producers(uint32_t kind) {
    Context ctx(kind, 64 << 20, SM_CONTEXT_POISON_UNWRITTEN);
    const Bytes data(5000, 0x3c);
    for (int round = 0; round < 3; ++round) { // fresh, then reused (Metal)
        auto* copied = copy(ctx.handle, data);
        const auto view = backing(copied);
        CHECK(bytes(copied) == data && all_equal(view.all.data() + data.size(), view.all.data() + view.capacity, 0));
        sm_buffer_release(copied);
        auto* zeroed = zeros(ctx.handle, data.size());
        const auto cleared = backing(zeroed);
        CHECK(all_equal(cleared.all.data(), cleared.all.data() + cleared.capacity, 0));
        sm_buffer_release(zeroed);
    }
}

void eviction_under_budget_pressure() {
    Context ctx(SM_BUFFER_METAL, 10 * page);
    CHECK(sm_context_set_cache_limit(ctx.handle, 10 * page) == SM_OK);
    std::vector<const void*> cached;
    std::vector<sm_buffer*> held;
    for (int index = 0; index < 4; ++index) {
        held.push_back(zeros(ctx.handle, logical(2)));
        cached.push_back(backing(held.back()).metal);
    }
    for (auto* buffer : held) sm_buffer_release(buffer); // cached oldest first: 0, 1, 2, 3
    CHECK(memory(ctx.handle).cached_bytes == 8 * page);

    // Six pages fit only after evicting the two least recently cached buffers.
    auto* big = zeros(ctx.handle, logical(6));
    auto info = memory(ctx.handle);
    CHECK(info.cached_bytes == 4 * page && info.owned_bytes == 6 * page && info.total_bytes == 10 * page);
    // Admission precedes reuse: this reservation evicts buffer 2, then the
    // allocation reuses buffer 3, the most recently cached.
    auto* again = zeros(ctx.handle, logical(2));
    CHECK(backing(again).metal == cached[3]);
    info = memory(ctx.handle);
    CHECK(info.cached_bytes == 0 && info.owned_bytes == 8 * page && info.total_bytes == 8 * page);
    const auto before_next = counters(ctx.handle);
    auto* next = zeros(ctx.handle, logical(2)); // nothing cached: a native allocation
    CHECK(counters(ctx.handle).native_allocations == before_next.native_allocations + 1);
    info = memory(ctx.handle);
    CHECK(info.cached_bytes == 0 && info.owned_bytes == 10 * page);

    // With nothing left to evict, admission fails and changes nothing.
    sm_buffer* refused = reinterpret_cast<sm_buffer*>(uintptr_t{0x10});
    CHECK(sm_buffer_zeros(ctx.handle, 1, &refused) == SM_EOOM && refused == reinterpret_cast<sm_buffer*>(uintptr_t{0x10}));
    info = memory(ctx.handle);
    CHECK(info.owned_bytes == 10 * page && info.reserved_bytes == 0 && info.peak_bytes == 10 * page);
    for (auto* buffer : {big, again, next}) sm_buffer_release(buffer);
    info = memory(ctx.handle);
    CHECK(info.owned_bytes == 0 && info.cached_bytes == 10 * page && info.allocation_count == 0);
}

void cache_limit_bounds_idle_storage() {
    Context ctx(SM_BUFFER_METAL);
    CHECK(sm_context_set_cache_limit(ctx.handle, 3 * page) == SM_OK);
    auto* a = zeros(ctx.handle, logical(2));
    auto* b = zeros(ctx.handle, logical(2));
    const auto kept = backing(b).metal;
    sm_buffer_release(a);
    sm_buffer_release(b); // evicts a to stay within three pages
    CHECK(memory(ctx.handle).cached_bytes == 2 * page);
    auto* reused = zeros(ctx.handle, logical(2));
    CHECK(backing(reused).metal == kept);
    sm_buffer_release(reused);

    // A buffer larger than the limit is freed rather than cached.
    sm_buffer_release(zeros(ctx.handle, logical(4)));
    CHECK(memory(ctx.handle).cached_bytes == 2 * page);
    // Shrinking the limit evicts down to it; zero disables caching.
    CHECK(sm_context_set_cache_limit(ctx.handle, page) == SM_OK);
    CHECK(memory(ctx.handle).cached_bytes == 0 && memory(ctx.handle).total_bytes == 0);
    CHECK(sm_context_set_cache_limit(ctx.handle, 0) == SM_OK);
    const auto before = counters(ctx.handle);
    for (int round = 0; round < 3; ++round) sm_buffer_release(zeros(ctx.handle, logical(1)));
    const auto after = counters(ctx.handle);
    CHECK(after.native_allocations == before.native_allocations + 3 && after.cached_reuses == before.cached_reuses);
    CHECK(memory(ctx.handle).total_bytes == 0);
}

void trim_and_context_release() {
    auto* ctx = create(SM_BUFFER_METAL, 64 << 20);
    for (uint64_t pages = 1; pages <= 3; ++pages) sm_buffer_release(zeros(ctx, logical(pages)));
    CHECK(memory(ctx).cached_bytes == 6 * page);
    CHECK(sm_context_trim(ctx) == SM_OK);
    auto info = memory(ctx);
    CHECK(info.cached_bytes == 0 && info.total_bytes == 0);

    // Releasing the context frees its cache; surviving buffers free directly later.
    const Bytes pattern(3000, 0x77);
    auto* survivor = copy(ctx, pattern);
    sm_buffer_release(zeros(ctx, logical(2)));
    sm_context_release(ctx);
    CHECK(bytes(survivor) == pattern);
    sm_buffer_release(survivor);
}

void live_views_are_never_cached() {
    Context ctx(SM_BUFFER_METAL);
    auto* owner = zeros(ctx.handle, logical(2));
    const auto original = backing(owner).metal;
    sm_buffer_view exported{};
    CHECK(sm_buffer_export(owner, &exported) == SM_OK);
    sm_buffer* slice = nullptr;
    CHECK(sm_buffer_slice(owner, 64, 128, &slice) == SM_OK);
    sm_buffer_release(owner);
    // The export and the slice keep the allocation live: nothing to reuse.
    auto* fresh = zeros(ctx.handle, logical(2));
    CHECK(backing(fresh).metal != original);
    exported.release(exported.owner);
    auto* still = zeros(ctx.handle, logical(2));
    CHECK(backing(still).metal != original);
    sm_buffer_release(slice);
    auto* reused = zeros(ctx.handle, logical(2));
    CHECK(backing(reused).metal == original);
    for (auto* buffer : {fresh, still, reused}) sm_buffer_release(buffer);
}

void concurrent_reuse_never_aliases_live_buffers() {
    Context ctx(SM_BUFFER_METAL, 1ull << 30);
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int thread = 0; thread < 8; ++thread)
        threads.emplace_back([&, thread] {
            sm_buffer* previous = nullptr;
            Bytes expected;
            for (int round = 0; round < 300; ++round) {
                const Bytes data(logical(1 + round % 3), static_cast<uint8_t>(thread * 31 + round));
                sm_buffer* current = nullptr;
                if (sm_buffer_copy_from_host(ctx.handle, data.data(), data.size(), &current) != SM_OK) {
                    failures.fetch_add(1);
                    break;
                }
                // Neither the new buffer nor the one still held may share storage.
                if (bytes(current) != data || (previous && bytes(previous) != expected)) failures.fetch_add(1);
                sm_buffer_release(previous);
                previous = current;
                expected = data;
            }
            sm_buffer_release(previous);
        });
    for (auto& thread : threads) thread.join();
    CHECK(failures.load() == 0);
    const auto info = memory(ctx.handle);
    CHECK(info.owned_bytes == 0 && info.reserved_bytes == 0 && info.allocation_count == 0);
    CHECK(info.cached_bytes <= cache_limit(ctx.handle) && info.total_bytes == info.cached_bytes);
    CHECK(counters(ctx.handle).cached_reuses > 0);
}

void fork_leaves_the_cache_alone() {
    Context ctx(SM_BUFFER_METAL);
    auto* first = zeros(ctx.handle, logical(2));
    const auto original = backing(first).metal;
    sm_buffer_release(first);
    const auto child = fork();
    CHECK(child >= 0);
    if (!child) {
        alarm(10);
        auto* sentinel = reinterpret_cast<sm_buffer*>(uintptr_t{0x20});
        sm_buffer* out = sentinel;
        CHECK(sm_buffer_zeros(ctx.handle, logical(2), &out) == SM_ESTATE && out == sentinel);
        CHECK(sm_context_trim(ctx.handle) == SM_ESTATE);
        CHECK(sm_context_set_cache_limit(ctx.handle, 0) == SM_ESTATE);
        auto* fresh = create(SM_BUFFER_METAL, 1 << 20);
        CHECK(sm_buffer_zeros(fresh, 64, &out) == SM_ESTATE && out == sentinel);
        _exit(0);
    }
    int status = 0;
    CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    auto* reused = zeros(ctx.handle, logical(2));
    CHECK(backing(reused).metal == original);
    sm_buffer_release(reused);
}

// host_fill_bytes counts what the host zero-fills: a zeroed buffer's whole
// capacity, only the padding of a complete producer's buffer, again when
// storage is served from the cache, and never the test-only poison.
void host_fill_bytes_count_zeroing(uint32_t kind) {
    for (const uint32_t flags : {0u, uint32_t{SM_CONTEXT_POISON_UNWRITTEN}}) {
        Context ctx(kind, 64 << 20, flags);
        const uint64_t start = counters(ctx.handle).host_fill_bytes;
        const uint64_t owned = memory(ctx.handle).owned_bytes;
        sm_buffer* zeroed = zeros(ctx.handle, 1000);
        const uint64_t capacity = memory(ctx.handle).owned_bytes - owned;
        CHECK(capacity >= 1000 && counters(ctx.handle).host_fill_bytes == start + capacity);
        const Bytes data(1000, 0x5a);
        sm_buffer* copied = copy(ctx.handle, data);
        CHECK(counters(ctx.handle).host_fill_bytes == start + capacity + (capacity - data.size()));
        sm_buffer_release(zeroed);
        sm_buffer* again = zeros(ctx.handle, 1000); // Metal: the released buffer, from the cache
        CHECK(counters(ctx.handle).host_fill_bytes == start + 2 * capacity + (capacity - data.size()));
        sm_buffer_release(again);
        sm_buffer_release(copied);
    }
}

void cpu_storage_never_caches() {
    Context ctx(SM_BUFFER_CPU);
    CHECK(cache_limit(ctx.handle) == 0);
    CHECK(sm_context_set_cache_limit(ctx.handle, page) == SM_ENOTSUP && cache_limit(ctx.handle) == 0);
    CHECK(sm_context_set_cache_limit(ctx.handle, 0) == SM_OK && sm_context_trim(ctx.handle) == SM_OK);
    const auto before = counters(ctx.handle);
    for (int round = 0; round < 3; ++round) sm_buffer_release(zeros(ctx.handle, 1000));
    const auto after = counters(ctx.handle);
    CHECK(after.native_allocations == before.native_allocations + 3 && after.cached_reuses == 0);
    const auto info = memory(ctx.handle);
    CHECK(info.cached_bytes == 0 && info.total_bytes == 0);
}

void arena_slab_scratch_suballocation() {
    constexpr uint64_t slab_bytes = 256u << 10;
    Context ctx(SM_BUFFER_METAL, 64 << 20);
    const auto before = counters(ctx.handle);
    auto s1 = sm::detail::allocate_scratch(ctx.handle, 12, sm::detail::Fill::zero);
    auto s2 = sm::detail::allocate_scratch(ctx.handle, 300, sm::detail::Fill::zero);
    auto s3 = sm::detail::allocate_scratch(ctx.handle, 4096, sm::detail::Fill::zero);
    CHECK(sm::detail::is_scratch_slice(s1.get()) && sm::detail::is_scratch_slice(s2.get()) &&
          sm::detail::is_scratch_slice(s3.get()));
    const auto v1 = sm::detail::device_view(s1.get(), ctx.handle);
    const auto v2 = sm::detail::device_view(s2.get(), ctx.handle);
    const auto v3 = sm::detail::device_view(s3.get(), ctx.handle);
    CHECK(v1.resource == v2.resource && v2.resource == v3.resource);
    CHECK(v1.offset == 0 && v2.offset == 256 && v3.offset == 768);
    CHECK(v1.offset % 256 == 0 && v2.offset % 256 == 0 && v3.offset % 256 == 0);
    const auto active = memory(ctx.handle);
    CHECK(active.owned_bytes == slab_bytes && active.cached_bytes == 0 && active.allocation_count == 3);
    const auto mid = counters(ctx.handle);
    CHECK(mid.native_allocations == before.native_allocations + 1 && mid.cached_reuses == before.cached_reuses + 2);

    // Releasing partial slices keeps the 256 KiB slab active in owned_bytes.
    s1.reset();
    s2.reset();
    CHECK(memory(ctx.handle).owned_bytes == slab_bytes && memory(ctx.handle).cached_bytes == 0);
    // Releasing the final slice transitions the slab to cached_bytes.
    s3.reset();
    CHECK(memory(ctx.handle).owned_bytes == 0 && memory(ctx.handle).cached_bytes == slab_bytes);

    // Next scratch slice checks out the cached slab and resets bump to offset 0.
    auto s4 = sm::detail::allocate_scratch(ctx.handle, 64, sm::detail::Fill::zero);
    const auto v4 = sm::detail::device_view(s4.get(), ctx.handle);
    CHECK(v4.resource == v1.resource && v4.offset == 0);
    CHECK(counters(ctx.handle).native_allocations == mid.native_allocations &&
          counters(ctx.handle).cached_reuses == mid.cached_reuses + 1);
    s4.reset();

    // Allocations above the 32 KiB slice cap bypass ArenaSlab.
    auto large = sm::detail::allocate_scratch(ctx.handle, 40000, sm::detail::Fill::zero);
    CHECK(!sm::detail::is_scratch_slice(large.get()));
    large.reset();

    // Cache limit below 256 KiB evicts the idle slab and routes scratch to page allocations.
    CHECK(sm_context_set_cache_limit(ctx.handle, 2 * page) == SM_OK);
    auto non_slab = sm::detail::allocate_scratch(ctx.handle, 32, sm::detail::Fill::zero);
    CHECK(!sm::detail::is_scratch_slice(non_slab.get()));
    non_slab.reset();
    CHECK(sm_context_trim(ctx.handle) == SM_OK && memory(ctx.handle).cached_bytes == 0);

    // Tight context (< 4 MiB total headroom) also bypasses ArenaSlab cleanly.
    Context tight(SM_BUFFER_METAL, 2u << 20);
    auto tight_scratch = sm::detail::allocate_scratch(tight.handle, 32, sm::detail::Fill::zero);
    CHECK(!sm::detail::is_scratch_slice(tight_scratch.get()));

    // Fill::unfilled on a reused ArenaSlab slice must still zero its [len..slot_cap) tail padding.
    {
        Context plain(SM_BUFFER_METAL, 64 << 20, 0);
        auto dirty = sm::detail::allocate_scratch(plain.handle, 256, sm::detail::Fill::unfilled);
        CHECK(sm::detail::is_scratch_slice(dirty.get()));
        std::memset(sm::detail::mutable_data(dirty.get()), 0xAA, 256);
        dirty.reset();

        auto reused = sm::detail::allocate_scratch(plain.handle, 12, sm::detail::Fill::unfilled);
        CHECK(sm::detail::is_scratch_slice(reused.get()));
        const auto rv = sm::detail::device_view(reused.get(), plain.handle);
        CHECK(rv.offset == 0);
        const auto* raw = static_cast<const unsigned char*>(sm::detail::mutable_data(reused.get()));
        for (size_t i = 12; i < 256; ++i) CHECK(raw[i] == 0);
    }
}

void bounded_overfit_reuse() {
    Context ctx(SM_BUFFER_METAL, 64 << 20);
    const uint64_t cap_100 = 1u << 20;                  // 1.00 MiB
    const uint64_t cap_125 = 5 * (1u << 18);            // 1.25 MiB (<= 1.5x of 1 MiB)
    const uint64_t cap_1375 = 11 * (1u << 17);          // 1.375 MiB
    const uint64_t cap_175 = 7 * (1u << 18);            // 1.75 MiB (> 1.5x of 1 MiB)
    CHECK(cap_100 % page == 0 && cap_125 % page == 0 && cap_1375 % page == 0 && cap_175 % page == 0);

    auto* b1375 = copy(ctx.handle, Bytes(cap_1375 - 64, 0x33));
    auto* b125_a = copy(ctx.handle, Bytes(cap_125 - 64, 0x44));
    auto* b125_b = copy(ctx.handle, Bytes(cap_125 - 64, 0x55));
    const void* metal_125_b = backing(b125_b).metal;
    sm_buffer_release(b1375);
    sm_buffer_release(b125_a);
    sm_buffer_release(b125_b); // MRU at the smallest >= 1 MiB over-fit bucket (1.25 MiB)

    const auto before = counters(ctx.handle);
    auto* reused = zeros(ctx.handle, cap_100 - 64);
    const auto after = counters(ctx.handle);
    CHECK(after.native_allocations == before.native_allocations && after.cached_reuses == before.cached_reuses + 1);
    CHECK(backing(reused).metal == metal_125_b);
    CHECK(backing(reused).capacity == cap_125);
    // Deterministic logical capacity for Arrow accounting, physical capacity for memory budget.
    CHECK(sm::detail::owned_capacity(reused) == cap_100);
    CHECK(memory(ctx.handle).owned_bytes == cap_125);
    const auto zero_bytes = bytes(reused);
    CHECK(zero_bytes.size() == cap_100 - 64 && all_equal(zero_bytes.data(), zero_bytes.data() + zero_bytes.size(), 0));
    sm_buffer_release(reused);

    // > 1.5x cached buffer (1.75 MiB) must NOT over-fit a 1.0 MiB request.
    CHECK(sm_context_trim(ctx.handle) == SM_OK);
    sm_buffer_release(zeros(ctx.handle, cap_175 - 64));
    const auto before_miss = counters(ctx.handle);
    auto* exact_100 = zeros(ctx.handle, cap_100 - 64);
    const auto after_miss = counters(ctx.handle);
    CHECK(after_miss.native_allocations == before_miss.native_allocations + 1 &&
          after_miss.cached_reuses == before_miss.cached_reuses);
    CHECK(backing(exact_100).capacity == cap_100);
    sm_buffer_release(exact_100);

    // < 1 MiB allocations remain exact-capacity only (768 KiB does not serve 512 KiB).
    CHECK(sm_context_trim(ctx.handle) == SM_OK);
    const uint64_t cap_768k = 3 * (1u << 18), cap_512k = 2 * (1u << 18);
    sm_buffer_release(zeros(ctx.handle, cap_768k - 64));
    const auto before_small = counters(ctx.handle);
    auto* exact_512k = zeros(ctx.handle, cap_512k - 64);
    const auto after_small = counters(ctx.handle);
    CHECK(after_small.native_allocations == before_small.native_allocations + 1 &&
          after_small.cached_reuses == before_small.cached_reuses);
    CHECK(backing(exact_512k).capacity == cap_512k);
    sm_buffer_release(exact_512k);
}

// SciMetal: metal_export_and_import_with_completion_event (an asynchronous GPU
// producer chained across contexts) returns with the first asynchronous route.
} // namespace

int main(int argc, char** argv) {
    const bool metal = argc == 2 && std::strcmp(argv[1], "--metal") == 0;
    cpu_storage_never_caches();
    poison_is_invisible_to_complete_producers(SM_BUFFER_CPU);
    host_fill_bytes_count_zeroing(SM_BUFFER_CPU);
    if (!metal) {
        std::puts("CPU storage never caches buffers; poison stays invisible to complete producers");
        return 0;
    }
    limits_and_options();
    automatic_limit_retains_a_multiphase_working_set();
    automatic_limit_excludes_reservations_and_foreign_storage();
    automatic_ceiling_and_explicit_limits();
    automatic_growth_preserves_live_exports();
    reuse_by_capacity();
    reused_storage_is_reinitialized();
    poison_is_invisible_to_complete_producers(SM_BUFFER_METAL);
    host_fill_bytes_count_zeroing(SM_BUFFER_METAL);
    eviction_under_budget_pressure();
    cache_limit_bounds_idle_storage();
    trim_and_context_release();
    live_views_are_never_cached();
    concurrent_reuse_never_aliases_live_buffers();
    fork_leaves_the_cache_alone();
    arena_slab_scratch_suballocation();
    bounded_overfit_reuse();
    std::puts("Metal buffer cache reuse, re-initialization, accounting, eviction and lifetime checks passed");
}
