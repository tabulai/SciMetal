#include "sm_runtime.h"
#include "sm/detail/backend.hpp"

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

static sm_context* context(uint32_t kind, uint64_t limit = 1 << 20) {
    sm_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = SM_RUNTIME_ABI_VERSION;
    options.storage_kind = kind;
    options.memory_limit_bytes = limit;
    sm_context* result = nullptr;
    CHECK(sm_context_create(&options, &result) == SM_OK);
    CHECK(result);
    return result;
}

static sm_memory_info memory(sm_context* ctx) {
    sm_memory_info result{};
    result.struct_size = sizeof(result);
    result.abi_version = SM_RUNTIME_ABI_VERSION;
    CHECK(sm_context_memory_info(ctx, &result) == SM_OK);
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
    auto* cpu = context(SM_BUFFER_CPU);
    sm_buffer* buffer = nullptr;
    CHECK(sm_buffer_zeros(cpu, 7, &buffer) == SM_OK);
    sm_buffer_view exported{};
    CHECK(sm_buffer_export(buffer, &exported) == SM_OK);
    CHECK(exported.kind == SM_BUFFER_CPU && !exported.device_id && !exported.metal_buffer);
    exported.release(exported.owner);
    sm_buffer_release(buffer);
    CHECK(memory(cpu).total_bytes == 0);
    sm_context_release(cpu);

    // Merely choosing Metal storage does not allocate a buffer or require a
    // device, including when linked to the disabled backend.
    auto* lazy = context(SM_BUFFER_METAL);
    CHECK(memory(lazy).total_bytes == 0);
    sm_context_release(lazy);
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
            auto unexpected = sm::detail::allocate_metal(64);
            (void)unexpected;
            ::_exit(81);
        } catch (const sm::detail::Error& error) {
            CHECK(error.status == SM_ESTATE);
        }
    });
}
#endif

static void concurrent_first_allocations() {
    std::array<uint64_t, 8> devices{};
    std::vector<std::thread> threads;
    for (size_t index = 0; index < devices.size(); ++index) {
        threads.emplace_back([&, index] {
            auto* ctx = context(SM_BUFFER_METAL);
            sm_buffer* buffer = nullptr;
            CHECK(sm_buffer_zeros(ctx, 129, &buffer) == SM_OK);
            sm_buffer_view view{};
            CHECK(sm_buffer_export(buffer, &view) == SM_OK);
            CHECK(view.kind == SM_BUFFER_METAL && view.device_id && view.metal_buffer);
            CHECK(sm_buffer_validate(&view, 64) == SM_OK);
            CHECK(view.capacity >= view.byte_length + 64);
            devices[index] = view.device_id;
            const auto* bytes = static_cast<const unsigned char*>(view.host_base);
            for (uint64_t i = 0; i < view.capacity; ++i) CHECK(bytes[i] == 0);
            view.release(view.owner);
            sm_buffer_release(buffer);
            // Released Metal storage stays charged in the cache until trimmed.
            const auto released = memory(ctx);
            CHECK(released.owned_bytes == 0 && released.allocation_count == 0);
            CHECK(released.total_bytes == released.cached_bytes);
            CHECK(sm_context_trim(ctx) == SM_OK && memory(ctx).total_bytes == 0);
            sm_context_release(ctx);
        });
    }
    for (auto& thread : threads) thread.join();
    for (const auto device : devices) CHECK(device == devices[0]);
}

static void exported_slice_owns_storage() {
    auto* ctx = context(SM_BUFFER_METAL);
    // Cross a runtime page boundary: merely rounding to 64 bytes can undercharge
    // Metal's reported allocatedSize even when MTLBuffer.length is unchanged.
    std::vector<unsigned char> source(page_size() + 1);
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<unsigned char>(i * 17 + 3);
    sm_buffer* buffer = nullptr;
    CHECK(sm_buffer_copy_from_host(ctx, source.data(), source.size(), &buffer) == SM_OK);
    const void* data = nullptr;
    uint64_t length = 0;
    CHECK(sm_buffer_data(buffer, &data, &length) == SM_OK);
    CHECK(length == source.size() && reinterpret_cast<uintptr_t>(data) % 64 == 0);
    CHECK(std::memcmp(data, source.data(), source.size()) == 0);

    sm_buffer_view original{};
    CHECK(sm_buffer_export(buffer, &original) == SM_OK);
    const auto* padded = static_cast<const unsigned char*>(original.host_base);
    for (uint64_t i = source.size(); i < original.capacity; ++i) CHECK(padded[i] == 0);
    CHECK(original.capacity >= source.size() + 64);
    const auto charged = memory(ctx);
    CHECK(charged.owned_bytes == original.capacity && charged.allocation_count == 1);

    sm_buffer* slice = nullptr;
    CHECK(sm_buffer_slice(buffer, 17, 257, &slice) == SM_OK);
    sm_buffer_view view{};
    CHECK(sm_buffer_export(slice, &view) == SM_OK);
    CHECK(view.allocation_id == original.allocation_id && view.device_id == original.device_id);
    CHECK(view.metal_buffer == original.metal_buffer && view.host_base == original.host_base);
    CHECK(view.byte_offset == 17 && view.byte_length == 257);
    CHECK(sm_buffer_validate(&view, 1) == SM_OK);
    CHECK((view.flags & (SM_BUFFER_HOST_READABLE | SM_BUFFER_IMMUTABLE | SM_BUFFER_PRODUCER_COMPLETE)) ==
          (SM_BUFFER_HOST_READABLE | SM_BUFFER_IMMUTABLE | SM_BUFFER_PRODUCER_COMPLETE));
    CHECK(memory(ctx).owned_bytes == original.capacity);

#if defined(__unix__) || defined(__APPLE__)
    child_check([&] {
        const void* unchanged = reinterpret_cast<void*>(uintptr_t(1));
        uint64_t unchanged_size = 123;
        CHECK(sm_buffer_data(slice, &unchanged, &unchanged_size) == SM_ESTATE);
        CHECK(unchanged == reinterpret_cast<void*>(uintptr_t(1)) && unchanged_size == 123);
        auto* unchanged_output = reinterpret_cast<sm_buffer*>(uintptr_t(1));
        CHECK(sm_buffer_zeros(ctx, 64, &unchanged_output) == SM_ESTATE);
        CHECK(unchanged_output == reinterpret_cast<sm_buffer*>(uintptr_t(1)));
        view.release(view.owner);
        sm_buffer_release(slice);
        sm_context_release(ctx);
    });
#endif

    original.release(original.owner);
    sm_buffer_release(buffer);
    sm_buffer_release(slice);
    CHECK(memory(ctx).owned_bytes == view.capacity);
    sm_context_release(ctx);
    // Export owns both the native buffer and allocation/accounting state after
    // every original runtime handle has been destroyed.
    const auto* retained = static_cast<const unsigned char*>(view.host_base) + view.byte_offset;
    CHECK(std::memcmp(retained, source.data() + 17, 257) == 0);
    view.release(view.owner);
}

static void one_page_budget_accounts_for_padding() {
    const auto page = page_size();
    auto* ctx = context(SM_BUFFER_METAL, page);
    for (const auto logical : {page - 63, page}) {
        auto* unchanged = reinterpret_cast<sm_buffer*>(uintptr_t(1));
        // Even page-63 logical bytes need two pages after the required 64-byte
        // padding. The runtime must reject before making a native allocation.
        CHECK(sm_buffer_zeros(ctx, logical, &unchanged) == SM_EOOM);
        CHECK(unchanged == reinterpret_cast<sm_buffer*>(uintptr_t(1)));
        const auto after = memory(ctx);
        CHECK(after.total_bytes == 0 && after.owned_bytes == 0 && after.reserved_bytes == 0);
        CHECK(after.allocation_count == 0);
    }
    sm_buffer* small = nullptr;
    CHECK(sm_buffer_zeros(ctx, 8, &small) == SM_OK);
    sm_buffer_view view{};
    CHECK(sm_buffer_export(small, &view) == SM_OK);
    CHECK(view.byte_length == 8 && view.capacity == page);
    CHECK(memory(ctx).owned_bytes == page);
    view.release(view.owner);
    sm_buffer_release(small);
    CHECK(memory(ctx).total_bytes == 0);
    sm_context_release(ctx);
}

static void sliced_cpu_descriptor_imports_as_metal_snapshot() {
    std::array<unsigned char, 257> source{};
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<unsigned char>(i * 13 + 5);
    constexpr uint64_t offset = 17, length = 129;
    const auto destination_capacity = page_size();
    auto* cpu = context(SM_BUFFER_CPU);
    sm_buffer* original = nullptr;
    sm_buffer* slice = nullptr;
    CHECK(sm_buffer_copy_from_host(cpu, source.data(), source.size(), &original) == SM_OK);
    CHECK(sm_buffer_slice(original, offset, length, &slice) == SM_OK);
    sm_buffer_view input{};
    CHECK(sm_buffer_export(slice, &input) == SM_OK);
    CHECK(input.kind == SM_BUFFER_CPU && input.byte_offset == offset && input.byte_length == length);
    CHECK(input.capacity > length && !input.metal_buffer && !input.device_id);
    sm_buffer_release(slice);
    sm_buffer_release(original);
    sm_context_release(cpu); // The exported descriptor retains its CPU payload.

    std::array<sm_context*, 2> destinations{};
    std::array<sm_import_result, 2> snapshots{};
    const std::array<sm_copy_policy, 2> policies{SM_COPY_AUTO, SM_COPY_ALWAYS};
    for (size_t index = 0; index < policies.size(); ++index) {
        // The destination page alone fits, but the full retained source capacity
        // must coexist with it during copying, including for a short slice.
        auto* tight = context(SM_BUFFER_METAL, input.capacity + destination_capacity - 1);
        sm_import_result unchanged{reinterpret_cast<sm_buffer*>(uintptr_t(1)), 41, 42};
        CHECK(sm_buffer_import(tight, &input, policies[index], &unchanged) == SM_EOOM);
        CHECK(unchanged.buffer == reinterpret_cast<sm_buffer*>(uintptr_t(1)) &&
              unchanged.copied_bytes == 41 && unchanged.borrowed_bytes == 42);
        const auto rolled_back = memory(tight);
        CHECK(rolled_back.total_bytes == 0 && rolled_back.owned_bytes == 0 &&
              rolled_back.borrowed_bytes == 0 && rolled_back.reserved_bytes == 0 &&
              rolled_back.allocation_count == 0);
        sm_buffer* recovery = nullptr;
        CHECK(sm_buffer_zeros(tight, length, &recovery) == SM_OK);
        sm_buffer_release(recovery);
        CHECK(memory(tight).total_bytes == 0);
        sm_context_release(tight);

        auto* destination = context(SM_BUFFER_METAL, input.capacity + destination_capacity);
        destinations[index] = destination;
        CHECK(sm_buffer_import(destination, &input, SM_COPY_NEVER, &unchanged) == SM_ENOTSUP);
        CHECK(unchanged.buffer == reinterpret_cast<sm_buffer*>(uintptr_t(1)) &&
              unchanged.copied_bytes == 41 && unchanged.borrowed_bytes == 42);
        CHECK(memory(destination).total_bytes == 0);
        CHECK(sm_buffer_import(destination, &input, policies[index], &snapshots[index]) == SM_OK);
        CHECK(snapshots[index].copied_bytes == length && snapshots[index].borrowed_bytes == 0);
        const auto accounted = memory(destination);
        CHECK(accounted.owned_bytes == destination_capacity && accounted.borrowed_bytes == 0 &&
              accounted.reserved_bytes == 0 && accounted.total_bytes == destination_capacity &&
              accounted.peak_bytes == input.capacity + destination_capacity && accounted.allocation_count == 1);
        sm_buffer_view output{};
        CHECK(sm_buffer_export(snapshots[index].buffer, &output) == SM_OK);
        CHECK(sm_buffer_validate(&output, 64) == SM_OK);
        CHECK(output.kind == SM_BUFFER_METAL && output.metal_buffer && output.device_id &&
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
        CHECK(sm_buffer_data(snapshots[index].buffer, &data, &bytes) == SM_OK);
        CHECK(bytes == length && std::memcmp(data, source.data() + offset, length) == 0);
        sm_buffer_release(snapshots[index].buffer);
        CHECK(memory(destinations[index]).total_bytes == 0);
        sm_context_release(destinations[index]);
    }
}

static void maximum_length_failure_rolls_back() {
    auto* ctx = context(SM_BUFFER_METAL, std::numeric_limits<uint64_t>::max());
    auto* output = reinterpret_cast<sm_buffer*>(uintptr_t(1));
    CHECK(sm_buffer_zeros(ctx, std::numeric_limits<uint64_t>::max() / 2, &output) == SM_EOOM);
    CHECK(output == reinterpret_cast<sm_buffer*>(uintptr_t(1)));
    char diagnostic[1024]{};
    CHECK(sm_context_error_copy(ctx, diagnostic, sizeof(diagnostic)) == SM_OK);
    CHECK(std::strstr(diagnostic, "maxBufferLength"));
    const auto after = memory(ctx);
    CHECK(after.total_bytes == 0 && after.owned_bytes == 0 && after.reserved_bytes == 0);
    CHECK(after.allocation_count == 0);
    sm_buffer* small = nullptr;
    CHECK(sm_buffer_zeros(ctx, 8, &small) == SM_OK);
    sm_buffer_release(small);
    const auto released = memory(ctx); // the unbounded budget's cache keeps it
    CHECK(released.owned_bytes == 0 && released.total_bytes == released.cached_bytes);
    sm_context_release(ctx);
}

static void non_host_readable_metal_descriptor_import() {
    auto* producer = context(SM_BUFFER_METAL, 1 << 20);
    auto* consumer = context(SM_BUFFER_METAL, 1 << 20);
    sm_buffer* src = nullptr;
    CHECK(sm_buffer_zeros(producer, 256, &src) == SM_OK);
    sm_buffer_view exported{};
    CHECK(sm_buffer_export(src, &exported) == SM_OK);
    CHECK(exported.kind == SM_BUFFER_METAL && (exported.flags & SM_BUFFER_HOST_READABLE) != 0);

    // Wrap the underlying MTLBuffer in a foreign non-HOST_READABLE descriptor.
    struct ForeignOwner {
        std::atomic<int> refs{1};
        sm_buffer_view inner;
    };
    auto* foreign_owner = new ForeignOwner{{1}, exported};
    auto foreign_retain = [](void* p) noexcept {
        static_cast<ForeignOwner*>(p)->refs.fetch_add(1, std::memory_order_relaxed);
    };
    auto foreign_release = [](void* p) noexcept {
        auto* o = static_cast<ForeignOwner*>(p);
        if (o->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            o->inner.release(o->inner.owner);
            delete o;
        }
    };

    sm_buffer_view view = exported;
    view.flags &= ~uint64_t{SM_BUFFER_HOST_READABLE};
    view.host_base = nullptr;
    view.owner = foreign_owner;
    view.retain = foreign_retain;
    view.release = foreign_release;
    CHECK(sm_buffer_validate(&view, 64) == SM_OK);

    sm_import_result imported{};
    CHECK(sm_buffer_import(consumer, &view, SM_COPY_NEVER, &imported) == SM_OK);
    CHECK(imported.buffer != nullptr && imported.copied_bytes == 0 && imported.borrowed_bytes == view.capacity);

    const void* host_ptr = nullptr;
    uint64_t host_len = 0;
    CHECK(sm_buffer_data(imported.buffer, &host_ptr, &host_len) == SM_ENOTSUP);

    sm_buffer_view reexported{};
    CHECK(sm_buffer_export(imported.buffer, &reexported) == SM_OK);
    CHECK(sm_buffer_validate(&reexported, 64) == SM_OK);
    CHECK((reexported.flags & SM_BUFFER_HOST_READABLE) == 0 && reexported.host_base == nullptr);
    reexported.release(reexported.owner);

    sm_buffer_release(imported.buffer);
    view.release(view.owner);
    sm_buffer_release(src);
    sm_context_release(consumer);
    sm_context_release(producer);
}

static void disabled_backend() {
    auto* ctx = context(SM_BUFFER_METAL);
    auto* unchanged = reinterpret_cast<sm_buffer*>(uintptr_t(1));
    CHECK(sm_buffer_zeros(ctx, 64, &unchanged) == SM_ENOTSUP);
    CHECK(unchanged == reinterpret_cast<sm_buffer*>(uintptr_t(1)));
    CHECK(memory(ctx).total_bytes == 0);
    sm_context_release(ctx);
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
    non_host_readable_metal_descriptor_import();
#if defined(__unix__) || defined(__APPLE__)
    auto native = sm::detail::allocate_metal(64);
    child_check([&] { native.owner.reset(); });
#endif
    std::puts("S1 owned Metal storage, retained views, concurrency, fork guards and rollback passed");
}
