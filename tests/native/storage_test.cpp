#include "ms_runtime.h"
#include "ms/detail/storage.hpp"

#include <array>
#include <atomic>
#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "storage failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

static ms_context* context(uint64_t limit = 1 << 20, uint64_t external = 0) {
    ms_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = MS_RUNTIME_ABI_VERSION;
    options.storage_kind = MS_BUFFER_CPU;
    options.memory_limit_bytes = limit;
    options.external_reservation_bytes = external;
    ms_context* result = nullptr;
    CHECK(ms_context_create(&options, &result) == MS_OK);
    return result;
}
static ms_memory_info memory(ms_context* ctx) {
    ms_memory_info result{};
    result.struct_size = sizeof(result);
    result.abi_version = MS_RUNTIME_ABI_VERSION;
    CHECK(ms_context_memory_info(ctx, &result) == MS_OK);
    CHECK(result.total_bytes == result.owned_bytes + result.borrowed_bytes + result.reserved_bytes + result.external_bytes);
    CHECK(result.total_bytes <= result.limit_bytes && result.peak_bytes <= result.limit_bytes);
    CHECK(result.cached_bytes == 0 && result.scratch_bytes == 0 && !result.external_accounting_complete);
    return result;
}
static void bytes_equal(ms_buffer* buffer, const void* expected, uint64_t size) {
    const void* data = nullptr;
    uint64_t actual = 999;
    CHECK(ms_buffer_data(buffer, &data, &actual) == MS_OK);
    CHECK(actual == size);
    CHECK(!size || std::memcmp(data, expected, static_cast<size_t>(size)) == 0);
}
struct Owner {
    std::atomic<unsigned> refs{1}, retained{0}, released{0};
    ms_context* reenter = nullptr;
};
static void retain(void* pointer) {
    auto& owner = *static_cast<Owner*>(pointer);
    if (owner.reenter) (void)memory(owner.reenter);
    ++owner.refs;
    ++owner.retained;
}
static void release(void* pointer) {
    auto& owner = *static_cast<Owner*>(pointer);
    if (owner.reenter) (void)memory(owner.reenter);
    --owner.refs;
    ++owner.released;
}
static ms_buffer_view foreign(Owner& owner, void* base, uint64_t capacity, uint64_t offset,
                              uint64_t length, uint64_t id = 1000000) {
    ms_buffer_view view{};
    view.struct_size = sizeof(view);
    view.abi_version = MS_BUFFER_ABI_VERSION;
    view.kind = MS_BUFFER_CPU;
    view.flags = MS_BUFFER_HOST_READABLE | MS_BUFFER_IMMUTABLE | MS_BUFFER_PRODUCER_COMPLETE;
    view.allocation_id = id;
    view.host_base = base;
    view.byte_offset = offset;
    view.byte_length = length;
    view.capacity = capacity;
    view.owner = &owner;
    view.retain = retain;
    view.release = release;
    return view;
}

static void owned_storage_and_aliases() {
    auto* ctx = context(512, 64);
    CHECK(memory(ctx).total_bytes == 64);
    std::array<unsigned char, 65> input{};
    for (size_t index = 0; index < input.size(); ++index) input[index] = static_cast<unsigned char>(index + 1);
    ms_buffer* buffer = nullptr;
    CHECK(ms_buffer_copy_from_host(ctx, input.data(), input.size(), &buffer) == MS_OK);
    bytes_equal(buffer, input.data(), input.size());
    ms_buffer_view whole{};
    CHECK(ms_buffer_export(buffer, &whole) == MS_OK);
    CHECK(whole.capacity == 192 && whole.byte_offset == 0);
    CHECK(reinterpret_cast<uintptr_t>(whole.host_base) % 64 == 0);
    CHECK(ms_buffer_validate(&whole, 64) == MS_OK);
    const auto* data = static_cast<const unsigned char*>(whole.host_base);
    for (uint64_t index = input.size(); index < whole.capacity; ++index) CHECK(data[index] == 0);
    auto accounting = memory(ctx);
    CHECK(accounting.owned_bytes == 192 && accounting.allocation_count == 1);
    ms_buffer* slice = nullptr;
    CHECK(ms_buffer_slice(buffer, 3, 7, &slice) == MS_OK);
    ms_buffer* nested = nullptr;
    CHECK(ms_buffer_slice(slice, 2, 4, &nested) == MS_OK);
    bytes_equal(nested, input.data() + 5, 4);
    ms_buffer_view view{};
    CHECK(ms_buffer_export(nested, &view) == MS_OK);
    CHECK(view.allocation_id == whole.allocation_id && view.host_base == whole.host_base);
    CHECK(view.byte_offset == 5 && view.byte_length == 4 && view.capacity == whole.capacity);
    ms_import_result imported{};
    CHECK(ms_buffer_import(ctx, &view, MS_COPY_NEVER, &imported) == MS_OK);
    CHECK(imported.copied_bytes == 0 && imported.borrowed_bytes == 192);
    CHECK(memory(ctx).owned_bytes == 192 && memory(ctx).borrowed_bytes == 0);
    ms_buffer_release(imported.buffer);
    ms_buffer_release(nested);
    ms_buffer_release(slice);
    ms_buffer_release(buffer);
    whole.release(whole.owner);
    CHECK(memory(ctx).owned_bytes == 192); // the last export pins the allocation
    ms_context_release(ctx);
    CHECK(std::memcmp(static_cast<unsigned char*>(view.host_base) + view.byte_offset, input.data() + 5, 4) == 0);
    view.retain(view.owner);
    view.release(view.owner);
    view.release(view.owner);
}

static void foreign_borrow_and_snapshot() {
    std::array<unsigned char, 64> input{};
    for (size_t index = 0; index < input.size(); ++index) input[index] = static_cast<unsigned char>(index + 17);
    auto* ctx = context(256);
    Owner owner;
    owner.reenter = ctx; // callbacks must run outside the accounting mutex
    auto view = foreign(owner, input.data(), input.size(), 7, 9);
    ms_import_result first{}, second{};
    CHECK(ms_buffer_import(ctx, &view, MS_COPY_NEVER, &first) == MS_OK);
    view.byte_offset = 9;
    view.byte_length = 7;
    CHECK(ms_buffer_import(ctx, &view, MS_COPY_AUTO, &second) == MS_OK);
    CHECK(owner.refs == 2 && owner.retained == 1);
    CHECK(first.borrowed_bytes == 64 && first.copied_bytes == 0);
    CHECK(memory(ctx).borrowed_bytes == 64 && memory(ctx).allocation_count == 1);
    bytes_equal(first.buffer, input.data() + 7, 9);
    bytes_equal(second.buffer, input.data() + 9, 7);
    ms_buffer_release(first.buffer);
    ms_buffer_release(second.buffer);
    CHECK(owner.refs == 1 && owner.released == 1);
    CHECK(memory(ctx).total_bytes == 0);

    view.flags &= ~MS_BUFFER_IMMUTABLE;
    ms_import_result unchanged{reinterpret_cast<ms_buffer*>(uintptr_t{17}), 41, 42};
    CHECK(ms_buffer_import(ctx, &view, MS_COPY_NEVER, &unchanged) == MS_ENOTSUP);
    CHECK(unchanged.buffer == reinterpret_cast<ms_buffer*>(uintptr_t{17}) && unchanged.copied_bytes == 41 && unchanged.borrowed_bytes == 42);
    ms_import_result snapshot{};
    CHECK(ms_buffer_import(ctx, &view, MS_COPY_AUTO, &snapshot) == MS_OK);
    CHECK(snapshot.copied_bytes == 7 && snapshot.borrowed_bytes == 0 && owner.refs == 1);
    auto copied = std::array<unsigned char, 7>{};
    std::memcpy(copied.data(), input.data() + 9, copied.size());
    input[9] ^= 0xff;
    bytes_equal(snapshot.buffer, copied.data(), copied.size());
    auto accounting = memory(ctx);
    CHECK(accounting.owned_bytes == 128 && accounting.borrowed_bytes == 0 && accounting.peak_bytes == 192);
    ms_buffer_release(snapshot.buffer);
    CHECK(owner.retained == owner.released && memory(ctx).total_bytes == 0);
    ms_context_release(ctx);
}

static void independent_exports_cross_context() {
    auto* source = context();
    auto* target = context(128);
    const unsigned char input[] = {4, 5, 6};
    ms_buffer* original = nullptr;
    CHECK(ms_buffer_copy_from_host(source, input, sizeof(input), &original) == MS_OK);
    ms_buffer_view left{}, right{};
    CHECK(ms_buffer_export(original, &left) == MS_OK);
    CHECK(ms_buffer_export(original, &right) == MS_OK);
    CHECK(left.owner != right.owner);
    ms_import_result a{}, b{};
    CHECK(ms_buffer_import(target, &left, MS_COPY_NEVER, &a) == MS_OK);
    CHECK(ms_buffer_import(target, &right, MS_COPY_NEVER, &b) == MS_OK);
    CHECK(memory(target).borrowed_bytes == 128 && memory(target).allocation_count == 1);
    left.release(left.owner);
    right.release(right.owner);
    ms_buffer_release(original);
    ms_context_release(source);
    bytes_equal(a.buffer, input, sizeof(input));
    ms_buffer_view propagated{};
    CHECK(ms_buffer_export(a.buffer, &propagated) == MS_OK);
    ms_import_result c{};
    CHECK(ms_buffer_import(target, &propagated, MS_COPY_NEVER, &c) == MS_OK);
    CHECK(memory(target).allocation_count == 1);
    propagated.release(propagated.owner);
    ms_buffer_release(a.buffer);
    ms_buffer_release(b.buffer);
    ms_buffer_release(c.buffer);
    CHECK(memory(target).total_bytes == 0);
    ms_context_release(target);
}

static void rejected_inputs_and_rollback() {
    auto* ctx = context(191);
    std::array<unsigned char, 64> bytes{};
    Owner owner;
    auto view = foreign(owner, bytes.data(), bytes.size(), 0, 3);
    ms_import_result output{reinterpret_cast<ms_buffer*>(uintptr_t{17}), 41, 42};
    CHECK(ms_buffer_import(ctx, &view, MS_COPY_ALWAYS, &output) == MS_EOOM);
    CHECK(output.buffer == reinterpret_cast<ms_buffer*>(uintptr_t{17}) && output.copied_bytes == 41 && output.borrowed_bytes == 42);
    CHECK(owner.refs == 1 && owner.retained == owner.released);
    CHECK(memory(ctx).total_bytes == 0 && memory(ctx).reserved_bytes == 0);

    ms_import_result borrowed{};
    CHECK(ms_buffer_import(ctx, &view, MS_COPY_NEVER, &borrowed) == MS_OK);
    std::array<unsigned char, 64> other{};
    auto collision = view;
    collision.host_base = other.data();
    CHECK(ms_buffer_import(ctx, &collision, MS_COPY_NEVER, &output) == MS_EINVAL);
    Owner other_owner;
    collision = view;
    collision.owner = &other_owner;
    CHECK(ms_buffer_import(ctx, &collision, MS_COPY_NEVER, &output) == MS_EINVAL);
    CHECK(other_owner.retained == 0);
    CHECK(memory(ctx).borrowed_bytes == 64 && owner.refs == 2);
    ms_buffer_release(borrowed.buffer);

    auto metal = view;
    metal.kind = MS_BUFFER_METAL;
    metal.device_id = 1;
    metal.metal_buffer = bytes.data();
    CHECK(ms_buffer_import(ctx, &metal, MS_COPY_ALWAYS, &output) == MS_ENOTSUP);
    view.byte_offset = 63;
    view.byte_length = 2;
    CHECK(ms_buffer_import(ctx, &view, MS_COPY_AUTO, &output) == MS_EINVAL);
    view.byte_offset = UINT64_MAX;
    CHECK(ms_buffer_import(ctx, &view, MS_COPY_AUTO, &output) == MS_EOVERFLOW);

    ms_buffer* sentinel = reinterpret_cast<ms_buffer*>(uintptr_t{17});
    CHECK(ms_buffer_zeros(ctx, UINT64_MAX, &sentinel) == MS_EOVERFLOW);
    CHECK(sentinel == reinterpret_cast<ms_buffer*>(uintptr_t{17}));
    CHECK(ms_buffer_zeros(ctx, 128, &sentinel) == MS_EOOM); // capacity 192
    CHECK(sentinel == reinterpret_cast<ms_buffer*>(uintptr_t{17}));
    char error[256]{};
    CHECK(ms_context_error_copy(ctx, error, sizeof(error)) == MS_OK && error[0]);
    char tiny = 'x';
    CHECK(ms_context_error_copy(ctx, &tiny, 1) == MS_EINVAL && tiny == 'x');
    char again[256]{};
    CHECK(ms_context_error_copy(ctx, again, sizeof(again)) == MS_OK && std::strcmp(error, again) == 0);
    CHECK(memory(ctx).total_bytes == 0);
    ms_context_release(ctx);
}

static void empty_bounds_and_reservations() {
    auto* ctx = context(256);
    ms_buffer* empty = nullptr;
    CHECK(ms_buffer_copy_from_host(ctx, nullptr, 0, &empty) == MS_OK);
    CHECK(memory(ctx).owned_bytes == 64);
    ms_buffer* slice = nullptr;
    CHECK(ms_buffer_slice(empty, 0, 0, &slice) == MS_OK);
    bytes_equal(slice, nullptr, 0);
    ms_buffer* unchanged = empty;
    CHECK(ms_buffer_slice(empty, 0, 1, &unchanged) == MS_EINVAL && unchanged == empty);
    CHECK(ms_buffer_slice(empty, UINT64_MAX, 1, &unchanged) == MS_EOVERFLOW && unchanged == empty);
    const void* pointer = reinterpret_cast<void*>(uintptr_t{17});
    CHECK(ms_buffer_data(empty, &pointer, nullptr) == MS_EINVAL && pointer == reinterpret_cast<void*>(uintptr_t{17}));
    ms_buffer_release(slice);
    ms_buffer_release(empty);
    {
        auto reservation = ms::detail::reserve(ctx, 256);
        CHECK(memory(ctx).reserved_bytes == 256);
        auto first = ms::detail::allocate_reserved(ctx, reservation, 1);
        CHECK(memory(ctx).reserved_bytes == 128 && memory(ctx).owned_bytes == 128);
        try { auto invalid = ms::detail::allocate_reserved(ctx, reservation, 65); CHECK(false); }
        catch (const ms::detail::Error& error) { CHECK(error.status == MS_EINVAL); }
        CHECK(memory(ctx).reserved_bytes == 128 && memory(ctx).owned_bytes == 128);
    }
    CHECK(memory(ctx).total_bytes == 0);
    Owner owner;
    auto view = foreign(owner, nullptr, 0, 0, 0);
    ms_import_result zero{};
    CHECK(ms_buffer_import(ctx, &view, MS_COPY_NEVER, &zero) == MS_OK);
    bytes_equal(zero.buffer, nullptr, 0);
    CHECK(memory(ctx).borrowed_bytes == 0 && memory(ctx).allocation_count == 1);
    ms_buffer_release(zero.buffer);
    CHECK(owner.refs == 1);
    ms_context_release(ctx);
}

static void concurrent_admission_and_aliases() {
    constexpr unsigned count = 24;
    auto* ctx = context(4 * 128 + 17, 17);
    std::barrier barrier(count + 1);
    std::atomic<unsigned> admitted{0};
    std::vector<std::thread> threads;
    for (unsigned index = 0; index < count; ++index) threads.emplace_back([&] {
        ms_buffer* buffer = nullptr;
        const auto status = ms_buffer_zeros(ctx, 1, &buffer);
        CHECK(status == MS_OK || status == MS_EOOM);
        if (status == MS_OK) ++admitted;
        else CHECK(!buffer);
        barrier.arrive_and_wait();
        barrier.arrive_and_wait();
        ms_buffer_release(buffer);
    });
    barrier.arrive_and_wait();
    CHECK(admitted == 4);
    CHECK(memory(ctx).owned_bytes == 512 && memory(ctx).allocation_count == 4);
    barrier.arrive_and_wait();
    for (auto& thread : threads) thread.join();
    CHECK(memory(ctx).total_bytes == 17 && memory(ctx).reserved_bytes == 0);
    ms_context_release(ctx);

    ctx = context(64);
    Owner owner;
    std::array<unsigned char, 64> data{};
    auto view = foreign(owner, data.data(), 64, 2, 4);
    threads.clear();
    for (unsigned index = 0; index < count; ++index) threads.emplace_back([&] {
        ms_import_result result{};
        CHECK(ms_buffer_import(ctx, &view, MS_COPY_NEVER, &result) == MS_OK);
        barrier.arrive_and_wait();
        barrier.arrive_and_wait();
        ms_buffer_release(result.buffer);
    });
    barrier.arrive_and_wait();
    CHECK(memory(ctx).borrowed_bytes == 64 && memory(ctx).allocation_count == 1 && owner.refs == 2);
    barrier.arrive_and_wait();
    for (auto& thread : threads) thread.join();
    CHECK(owner.refs == 1 && owner.retained == owner.released && memory(ctx).total_bytes == 0);
    ms_context_release(ctx);
}

static void fork_guards() {
    auto* ctx = context();
    ms_buffer* buffer = nullptr;
    CHECK(ms_buffer_zeros(ctx, 3, &buffer) == MS_OK);
    ms_buffer_view view{};
    CHECK(ms_buffer_export(buffer, &view) == MS_OK);
    const auto child = fork();
    CHECK(child >= 0);
    if (!child) {
        alarm(5);
        ms_buffer* output = buffer;
        CHECK(ms_buffer_zeros(ctx, 1, &output) == MS_ESTATE && output == buffer);
        const void* data = nullptr;
        uint64_t bytes = 17;
        CHECK(ms_buffer_data(buffer, &data, &bytes) == MS_ESTATE && !data && bytes == 17);
        char error = 'x';
        CHECK(ms_context_error_copy(ctx, &error, 1) == MS_ESTATE && error == 'x');
        auto* fresh = context();
        ms_import_result imported{};
        CHECK(ms_buffer_import(fresh, &view, MS_COPY_ALWAYS, &imported) == MS_ESTATE && !imported.buffer);
        ms_context_release(fresh);
        ms_buffer_retain(buffer);
        ms_buffer_release(buffer);
        ms_context_retain(ctx);
        ms_context_release(ctx);
        view.retain(view.owner);
        view.release(view.owner);
        _exit(0);
    }
    int status = 0;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    view.release(view.owner);
    ms_buffer_release(buffer);
    CHECK(memory(ctx).total_bytes == 0);
    ms_context_release(ctx);
}

int main() {
    owned_storage_and_aliases();
    foreign_borrow_and_snapshot();
    independent_exports_cross_context();
    rejected_inputs_and_rollback();
    empty_bounds_and_reservations();
    concurrent_admission_and_aliases();
    fork_guards();
    std::puts("S1 immutable CPU storage, retained aliases, admission, rollback, callbacks and fork guards passed");
}
