#include "ms/detail/storage.hpp"
#include "metal_cache_policy.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(MS_TEST_METAL)
#include <Metal/Metal.hpp>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_statistics.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)
using namespace ms::detail;

namespace {
void page_policy() {
    std::array<int32_t, 8> statuses{};
    std::array<uint32_t, 8> flags{};
    flags.fill(0x1);
    CHECK(!cached_metal_pages_need_reset(statuses, flags));
    flags[0] = 0x10;
    CHECK(!cached_metal_pages_need_reset(statuses, flags));
    flags[1] = 0x10;
    CHECK(cached_metal_pages_need_reset(statuses, flags));
    // A late ambiguity or failed query vetoes two earlier PAGED_OUT samples.
    for (const uint32_t unknown : {0u, 0x11u, 0x2u, 0x80u, 0x40u, 0x101u, 0x80000000u}) {
        flags[7] = unknown;
        CHECK(!cached_metal_pages_need_reset(statuses, flags));
    }
    flags[7] = 0x1;
    statuses[7] = 5;
    CHECK(!cached_metal_pages_need_reset(statuses, flags));
    statuses[7] = 0;
    for (auto& flag : flags) flag |= 0x2c; // REF, DIRTY and COPIED are compatible.
    CHECK(cached_metal_pages_need_reset(statuses, flags));
}

void sample_geometry() {
    for (const uint64_t page : {uint64_t{4096}, uint64_t{16384}}) {
        for (const uintptr_t base : {uintptr_t{0x100000}, uintptr_t{0x100040}}) {
            uint64_t last = 0;
            for (size_t index = 0; index < cached_metal_sample_count; ++index) {
                const auto offset = cached_metal_sample_offset(base, cached_metal_sample_min_capacity, page, index);
                CHECK(offset && *offset > last && (base + *offset) % page == 0);
                CHECK(*offset > page && *offset + page < cached_metal_sample_min_capacity);
                last = *offset;
            }
        }
    }
    CHECK(!cached_metal_sample_offset(0x100000, 240000000, 16384, 0));
    CHECK(!cached_metal_sample_offset(0x100000, cached_metal_sample_min_capacity - 1, 16384, 0));
    CHECK(!cached_metal_sample_offset(0x100001, cached_metal_sample_min_capacity, 16384, 0));
    CHECK(!cached_metal_sample_offset(0x100000, cached_metal_sample_min_capacity, 3, 0));
    CHECK(!cached_metal_sample_offset(0x100000, cached_metal_sample_min_capacity, 16384, 8));
    CHECK(!cached_metal_sample_offset(UINTPTR_MAX - 63, cached_metal_sample_min_capacity, 16384, 0));
}

template<class Function> void rejects(ms_status expected, Function function) {
    bool rejected = false;
    try { function(); }
    catch (const Error& error) { rejected = error.status == expected; }
    CHECK(rejected);
}

#if defined(MS_TEST_METAL)
void reset_guards_and_state() {
    const auto page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    auto raw = allocate_metal(page);
    auto* buffer = static_cast<MTL::Buffer*>(raw.metal);
    std::memset(raw.host, 0x5a, page);
    auto retained_owner = raw.owner;
    rejects(MS_ESTATE, [&] { discard_cached_metal(raw); });
    for (size_t index = 0; index < page; ++index) CHECK(static_cast<uint8_t*>(raw.host)[index] == 0x5a);
    retained_owner.reset();
    raw.capacity += page;
    rejects(MS_EBACKEND, [&] { discard_cached_metal(raw); });
    raw.capacity -= page;
    const auto original_device = raw.device_id;
    raw.device_id = original_device == 1 ? 2 : 1;
    rejects(MS_EBACKEND, [&] { discard_cached_metal(raw); });
    raw.device_id = original_device;
    CHECK(buffer->setPurgeableState(MTL::PurgeableStateEmpty) == MTL::PurgeableStateNonVolatile);
    rejects(MS_EBACKEND, [&] { discard_cached_metal(raw); });
    CHECK(buffer->setPurgeableState(MTL::PurgeableStateNonVolatile) == MTL::PurgeableStateEmpty);
    discard_cached_metal(raw);
    CHECK(buffer->setPurgeableState(MTL::PurgeableStateKeepCurrent) == MTL::PurgeableStateNonVolatile);
    CHECK(raw.host == buffer->contents() && raw.capacity == page && buffer->length() == page);
    CHECK(reinterpret_cast<uintptr_t>(raw.host) % 64 == 0);
    // Contents are unspecified until the producer initializes them again.
    std::memset(raw.host, 0xa5, page);
    for (size_t index = 0; index < page; ++index) CHECK(static_cast<uint8_t*>(raw.host)[index] == 0xa5);
    const auto child = fork();
    CHECK(child >= 0);
    if (!child) {
        try { discard_cached_metal(raw); }
        catch (const Error& error) { _exit(error.status == MS_ESTATE ? 0 : 1); }
        _exit(2);
    }
    int status = -1;
    CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

void actual_owned_page_queries() {
    const auto page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    auto raw = allocate_metal(cached_metal_sample_min_capacity);
    auto retained_owner = raw.owner;
    rejects(MS_ESTATE, [&] { prepare_cached_metal(raw); });
    retained_owner.reset();
    discard_cached_metal(raw);
    auto* buffer = static_cast<MTL::Buffer*>(raw.metal);
    for (const bool touch : {false, true}) {
        std::array<int32_t, cached_metal_sample_count> statuses{};
        std::array<uint32_t, cached_metal_sample_count> flags{};
        for (size_t index = 0; index < cached_metal_sample_count; ++index) {
            const auto offset = cached_metal_sample_offset(reinterpret_cast<uintptr_t>(raw.host), raw.capacity, page, index);
            CHECK(offset);
            if (touch) static_cast<volatile uint8_t*>(raw.host)[*offset] = 0x59;
            integer_t disposition = 0, references = 0;
            statuses[index] = mach_vm_page_query(mach_task_self(),
                reinterpret_cast<mach_vm_offset_t>(raw.host) + *offset, &disposition, &references);
            flags[index] = static_cast<uint32_t>(disposition);
            CHECK(statuses[index] == KERN_SUCCESS);
        }
        // Report actual OS observations without treating demand-zero as a
        // promised physical state or creating artificial memory pressure.
        std::printf("owned Metal %s page flags: %u %u %u %u %u %u %u %u; decision=%d\n",
            touch ? "CPU-touched" : "discarded", flags[0], flags[1], flags[2], flags[3],
            flags[4], flags[5], flags[6], flags[7], int(cached_metal_pages_need_reset(statuses, flags)));
        prepare_cached_metal(raw);
        CHECK(buffer->setPurgeableState(MTL::PurgeableStateKeepCurrent) == MTL::PurgeableStateNonVolatile);
        CHECK(raw.host == buffer->contents() && raw.capacity == cached_metal_sample_min_capacity);
        if (touch) {
            for (size_t index = 0; index < cached_metal_sample_count; ++index) {
                const auto offset = cached_metal_sample_offset(reinterpret_cast<uintptr_t>(raw.host), raw.capacity, page, index);
                CHECK(offset && static_cast<const uint8_t*>(raw.host)[*offset] == 0x59);
            }
        }
    }
}
#endif
} // namespace

int main() {
    page_policy();
    sample_geometry();
#if defined(MS_TEST_METAL)
    reset_guards_and_state();
    actual_owned_page_queries();
#else
    RawAllocation unavailable;
    rejects(MS_ENOTSUP, [&] { prepare_cached_metal(unavailable); });
    rejects(MS_ENOTSUP, [&] { discard_cached_metal(unavailable); });
#endif
    std::puts("Cached Metal page policy, geometry and backend guards passed");
}
