#include "ms_runtime.h"
#include "ms/detail/backend.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/wait.h>
#include <unistd.h>
#endif

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

static ms_context* context(uint32_t kind, uint64_t limit = 1 << 20) {
    ms_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = MS_RUNTIME_ABI_VERSION;
    options.storage_kind = kind;
    options.memory_limit_bytes = limit;
    ms_context* result = nullptr;
    CHECK(ms_context_create(&options, &result) == MS_OK);
    CHECK(result);
    return result;
}

static ms_memory_info memory(ms_context* ctx) {
    ms_memory_info result{};
    result.struct_size = sizeof(result);
    result.abi_version = MS_RUNTIME_ABI_VERSION;
    CHECK(ms_context_memory_info(ctx, &result) == MS_OK);
    return result;
}

static uint64_t page_size() {
#if defined(__unix__) || defined(__APPLE__)
    const auto result = ::sysconf(_SC_PAGESIZE);
    CHECK(result >= 64 && (result & (result - 1)) == 0);
    return static_cast<uint64_t>(result);
#else
    CHECK(false); // Metal tests are enabled only on Apple Silicon.
    return 0;
#endif
}

static void cpu_and_lazy_contexts() {
    auto* cpu = context(MS_BUFFER_CPU);
    ms_buffer* buffer = nullptr;
    CHECK(ms_buffer_zeros(cpu, 7, &buffer) == MS_OK);
    ms_buffer_view exported{};
    CHECK(ms_buffer_export(buffer, &exported) == MS_OK);
    CHECK(exported.kind == MS_BUFFER_CPU && !exported.device_id && !exported.metal_buffer);
    exported.release(exported.owner);
    ms_buffer_release(buffer);
    CHECK(memory(cpu).total_bytes == 0);
    ms_context_release(cpu);

    // Merely choosing Metal storage does not allocate a buffer or require a
    // device, including when linked to the disabled backend.
    auto* lazy = context(MS_BUFFER_METAL);
    CHECK(memory(lazy).total_bytes == 0);
    ms_context_release(lazy);
}

#if defined(__unix__) || defined(__APPLE__)
template<class Function> static void child_check(Function function) {
    const auto child = ::fork();
    CHECK(child >= 0);
    if (child == 0) {
        ::alarm(5); // A mistakenly inherited initialization lock must not hang CTest.
        function();
        ::_exit(0);
    }
    int status = 0;
    CHECK(::waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

static void child_before_device_initialization() {
    // Test the private backend guard as well as the public runtime guards: the
    // backend itself must check the PID before its lazy static guard or Metal.
    child_check([] {
        try {
            auto unexpected = ms::detail::allocate_metal(64);
            (void)unexpected;
            ::_exit(81);
        } catch (const ms::detail::Error& error) {
            CHECK(error.status == MS_ESTATE);
        }
    });
}
#endif

static void concurrent_first_allocations() {
    std::array<uint64_t, 8> devices{};
    std::vector<std::thread> threads;
    for (size_t index = 0; index < devices.size(); ++index) {
        threads.emplace_back([&, index] {
            auto* ctx = context(MS_BUFFER_METAL);
            ms_buffer* buffer = nullptr;
            CHECK(ms_buffer_zeros(ctx, 129, &buffer) == MS_OK);
            ms_buffer_view view{};
            CHECK(ms_buffer_export(buffer, &view) == MS_OK);
            CHECK(view.kind == MS_BUFFER_METAL && view.device_id && view.metal_buffer);
            CHECK(ms_buffer_validate(&view, 64) == MS_OK);
            CHECK(view.capacity >= view.byte_length + 64);
            devices[index] = view.device_id;
            const auto* bytes = static_cast<const unsigned char*>(view.host_base);
            for (uint64_t i = 0; i < view.capacity; ++i) CHECK(bytes[i] == 0);
            view.release(view.owner);
            ms_buffer_release(buffer);
            // Released Metal storage stays charged in the cache until trimmed.
            const auto released = memory(ctx);
            CHECK(released.owned_bytes == 0 && released.allocation_count == 0);
            CHECK(released.total_bytes == released.cached_bytes);
            CHECK(ms_context_trim(ctx) == MS_OK && memory(ctx).total_bytes == 0);
            ms_context_release(ctx);
        });
    }
    for (auto& thread : threads) thread.join();
    for (const auto device : devices) CHECK(device == devices[0]);
}

static void exported_slice_owns_storage() {
    auto* ctx = context(MS_BUFFER_METAL);
    // Cross a runtime page boundary: merely rounding to 64 bytes can undercharge
    // Metal's reported allocatedSize even when MTLBuffer.length is unchanged.
    std::vector<unsigned char> source(page_size() + 1);
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<unsigned char>(i * 17 + 3);
    ms_buffer* buffer = nullptr;
    CHECK(ms_buffer_copy_from_host(ctx, source.data(), source.size(), &buffer) == MS_OK);
    const void* data = nullptr;
    uint64_t length = 0;
    CHECK(ms_buffer_data(buffer, &data, &length) == MS_OK);
    CHECK(length == source.size() && reinterpret_cast<uintptr_t>(data) % 64 == 0);
    CHECK(std::memcmp(data, source.data(), source.size()) == 0);

    ms_buffer_view original{};
    CHECK(ms_buffer_export(buffer, &original) == MS_OK);
    const auto* padded = static_cast<const unsigned char*>(original.host_base);
    for (uint64_t i = source.size(); i < original.capacity; ++i) CHECK(padded[i] == 0);
    CHECK(original.capacity >= source.size() + 64);
    const auto charged = memory(ctx);
    CHECK(charged.owned_bytes == original.capacity && charged.allocation_count == 1);

    ms_buffer* slice = nullptr;
    CHECK(ms_buffer_slice(buffer, 17, 257, &slice) == MS_OK);
    ms_buffer_view view{};
    CHECK(ms_buffer_export(slice, &view) == MS_OK);
    CHECK(view.allocation_id == original.allocation_id && view.device_id == original.device_id);
    CHECK(view.metal_buffer == original.metal_buffer && view.host_base == original.host_base);
    CHECK(view.byte_offset == 17 && view.byte_length == 257);
    CHECK(ms_buffer_validate(&view, 1) == MS_OK);
    CHECK((view.flags & (MS_BUFFER_HOST_READABLE | MS_BUFFER_IMMUTABLE | MS_BUFFER_PRODUCER_COMPLETE)) ==
          (MS_BUFFER_HOST_READABLE | MS_BUFFER_IMMUTABLE | MS_BUFFER_PRODUCER_COMPLETE));
    CHECK(memory(ctx).owned_bytes == original.capacity);

#if defined(__unix__) || defined(__APPLE__)
    child_check([&] {
        const void* unchanged = reinterpret_cast<void*>(uintptr_t(1));
        uint64_t unchanged_size = 123;
        CHECK(ms_buffer_data(slice, &unchanged, &unchanged_size) == MS_ESTATE);
        CHECK(unchanged == reinterpret_cast<void*>(uintptr_t(1)) && unchanged_size == 123);
        auto* unchanged_output = reinterpret_cast<ms_buffer*>(uintptr_t(1));
        CHECK(ms_buffer_zeros(ctx, 64, &unchanged_output) == MS_ESTATE);
        CHECK(unchanged_output == reinterpret_cast<ms_buffer*>(uintptr_t(1)));
        view.release(view.owner);
        ms_buffer_release(slice);
        ms_context_release(ctx);
    });
#endif

    original.release(original.owner);
    ms_buffer_release(buffer);
    ms_buffer_release(slice);
    CHECK(memory(ctx).owned_bytes == view.capacity);
    ms_context_release(ctx);
    // Export owns both the native buffer and allocation/accounting state after
    // every original runtime handle has been destroyed.
    const auto* retained = static_cast<const unsigned char*>(view.host_base) + view.byte_offset;
    CHECK(std::memcmp(retained, source.data() + 17, 257) == 0);
    view.release(view.owner);
}

static void one_page_budget_accounts_for_padding() {
    const auto page = page_size();
    auto* ctx = context(MS_BUFFER_METAL, page);
    for (const auto logical : {page - 63, page}) {
        auto* unchanged = reinterpret_cast<ms_buffer*>(uintptr_t(1));
        // Even page-63 logical bytes need two pages after the required 64-byte
        // padding. The runtime must reject before making a native allocation.
        CHECK(ms_buffer_zeros(ctx, logical, &unchanged) == MS_EOOM);
        CHECK(unchanged == reinterpret_cast<ms_buffer*>(uintptr_t(1)));
        const auto after = memory(ctx);
        CHECK(after.total_bytes == 0 && after.owned_bytes == 0 && after.reserved_bytes == 0);
        CHECK(after.allocation_count == 0);
    }
    ms_buffer* small = nullptr;
    CHECK(ms_buffer_zeros(ctx, 8, &small) == MS_OK);
    ms_buffer_view view{};
    CHECK(ms_buffer_export(small, &view) == MS_OK);
    CHECK(view.byte_length == 8 && view.capacity == page);
    CHECK(memory(ctx).owned_bytes == page);
    view.release(view.owner);
    ms_buffer_release(small);
    CHECK(memory(ctx).total_bytes == 0);
    ms_context_release(ctx);
}

static void sliced_cpu_descriptor_imports_as_metal_snapshot() {
    std::array<unsigned char, 257> source{};
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<unsigned char>(i * 13 + 5);
    constexpr uint64_t offset = 17, length = 129;
    const auto destination_capacity = page_size();
    auto* cpu = context(MS_BUFFER_CPU);
    ms_buffer* original = nullptr;
    ms_buffer* slice = nullptr;
    CHECK(ms_buffer_copy_from_host(cpu, source.data(), source.size(), &original) == MS_OK);
    CHECK(ms_buffer_slice(original, offset, length, &slice) == MS_OK);
    ms_buffer_view input{};
    CHECK(ms_buffer_export(slice, &input) == MS_OK);
    CHECK(input.kind == MS_BUFFER_CPU && input.byte_offset == offset && input.byte_length == length);
    CHECK(input.capacity > length && !input.metal_buffer && !input.device_id);
    ms_buffer_release(slice);
    ms_buffer_release(original);
    ms_context_release(cpu); // The exported descriptor retains its CPU payload.

    std::array<ms_context*, 2> destinations{};
    std::array<ms_import_result, 2> snapshots{};
    const std::array<ms_copy_policy, 2> policies{MS_COPY_AUTO, MS_COPY_ALWAYS};
    for (size_t index = 0; index < policies.size(); ++index) {
        // The destination page alone fits, but the full retained source capacity
        // must coexist with it during copying, including for a short slice.
        auto* tight = context(MS_BUFFER_METAL, input.capacity + destination_capacity - 1);
        ms_import_result unchanged{reinterpret_cast<ms_buffer*>(uintptr_t(1)), 41, 42};
        CHECK(ms_buffer_import(tight, &input, policies[index], &unchanged) == MS_EOOM);
        CHECK(unchanged.buffer == reinterpret_cast<ms_buffer*>(uintptr_t(1)) &&
              unchanged.copied_bytes == 41 && unchanged.borrowed_bytes == 42);
        const auto rolled_back = memory(tight);
        CHECK(rolled_back.total_bytes == 0 && rolled_back.owned_bytes == 0 &&
              rolled_back.borrowed_bytes == 0 && rolled_back.reserved_bytes == 0 &&
              rolled_back.allocation_count == 0);
        ms_buffer* recovery = nullptr;
        CHECK(ms_buffer_zeros(tight, length, &recovery) == MS_OK);
        ms_buffer_release(recovery);
        CHECK(memory(tight).total_bytes == 0);
        ms_context_release(tight);

        auto* destination = context(MS_BUFFER_METAL, input.capacity + destination_capacity);
        destinations[index] = destination;
        CHECK(ms_buffer_import(destination, &input, MS_COPY_NEVER, &unchanged) == MS_ENOTSUP);
        CHECK(unchanged.buffer == reinterpret_cast<ms_buffer*>(uintptr_t(1)) &&
              unchanged.copied_bytes == 41 && unchanged.borrowed_bytes == 42);
        CHECK(memory(destination).total_bytes == 0);
        CHECK(ms_buffer_import(destination, &input, policies[index], &snapshots[index]) == MS_OK);
        CHECK(snapshots[index].copied_bytes == length && snapshots[index].borrowed_bytes == 0);
        const auto accounted = memory(destination);
        CHECK(accounted.owned_bytes == destination_capacity && accounted.borrowed_bytes == 0 &&
              accounted.reserved_bytes == 0 && accounted.total_bytes == destination_capacity &&
              accounted.peak_bytes == input.capacity + destination_capacity && accounted.allocation_count == 1);
        ms_buffer_view output{};
        CHECK(ms_buffer_export(snapshots[index].buffer, &output) == MS_OK);
        CHECK(ms_buffer_validate(&output, 64) == MS_OK);
        CHECK(output.kind == MS_BUFFER_METAL && output.metal_buffer && output.device_id &&
              output.host_base != input.host_base && output.byte_offset == 0 &&
              output.byte_length == length && output.capacity == destination_capacity);
        const auto* bytes = static_cast<const unsigned char*>(output.host_base);
        CHECK(std::memcmp(bytes, source.data() + offset, length) == 0);
        for (uint64_t i = length; i < output.capacity; ++i) CHECK(bytes[i] == 0);
        output.release(output.owner);
    }
    input.release(input.owner); // Each Metal snapshot now survives all source owners.
    for (size_t index = 0; index < snapshots.size(); ++index) {
        const void* data = nullptr;
        uint64_t bytes = 0;
        CHECK(ms_buffer_data(snapshots[index].buffer, &data, &bytes) == MS_OK);
        CHECK(bytes == length && std::memcmp(data, source.data() + offset, length) == 0);
        ms_buffer_release(snapshots[index].buffer);
        CHECK(memory(destinations[index]).total_bytes == 0);
        ms_context_release(destinations[index]);
    }
}

static void maximum_length_failure_rolls_back() {
    auto* ctx = context(MS_BUFFER_METAL, std::numeric_limits<uint64_t>::max());
    auto* output = reinterpret_cast<ms_buffer*>(uintptr_t(1));
    CHECK(ms_buffer_zeros(ctx, std::numeric_limits<uint64_t>::max() / 2, &output) == MS_EOOM);
    CHECK(output == reinterpret_cast<ms_buffer*>(uintptr_t(1)));
    char diagnostic[1024]{};
    CHECK(ms_context_error_copy(ctx, diagnostic, sizeof(diagnostic)) == MS_OK);
    CHECK(std::strstr(diagnostic, "maxBufferLength"));
    const auto after = memory(ctx);
    CHECK(after.total_bytes == 0 && after.owned_bytes == 0 && after.reserved_bytes == 0);
    CHECK(after.allocation_count == 0);
    ms_buffer* small = nullptr;
    CHECK(ms_buffer_zeros(ctx, 8, &small) == MS_OK);
    ms_buffer_release(small);
    const auto released = memory(ctx); // the unbounded budget's cache keeps it
    CHECK(released.owned_bytes == 0 && released.total_bytes == released.cached_bytes);
    ms_context_release(ctx);
}

static void disabled_backend() {
    auto* ctx = context(MS_BUFFER_METAL);
    auto* unchanged = reinterpret_cast<ms_buffer*>(uintptr_t(1));
    CHECK(ms_buffer_zeros(ctx, 64, &unchanged) == MS_ENOTSUP);
    CHECK(unchanged == reinterpret_cast<ms_buffer*>(uintptr_t(1)));
    CHECK(memory(ctx).total_bytes == 0);
    ms_context_release(ctx);
}

int main(int argc, char** argv) {
    cpu_and_lazy_contexts();
    if (argc == 2 && std::strcmp(argv[1], "--disabled") == 0) {
        disabled_backend();
        std::puts("S1 CPU build and explicit disabled-Metal behavior passed");
        return 0;
    }
    CHECK(argc == 1);
#if defined(__unix__) || defined(__APPLE__)
    child_before_device_initialization();
#endif
    concurrent_first_allocations();
    exported_slice_owns_storage();
    one_page_budget_accounts_for_padding();
    sliced_cpu_descriptor_imports_as_metal_snapshot();
    maximum_length_failure_rolls_back();
#if defined(__unix__) || defined(__APPLE__)
    auto native = ms::detail::allocate_metal(64);
    child_check([&] { native.owner.reset(); });
#endif
    std::puts("S1 owned Metal storage, retained views, concurrency, fork guards and rollback passed");
}
