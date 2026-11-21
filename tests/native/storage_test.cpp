#include "sm_runtime.h"
#include "sm/detail/storage.hpp"

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

static sm_context* context(uint64_t limit = 1 << 20, uint64_t external = 0) {
    sm_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = SM_RUNTIME_ABI_VERSION;
    options.storage_kind = SM_BUFFER_CPU;
    options.memory_limit_bytes = limit;
    options.external_reservation_bytes = external;
    sm_context* result = nullptr;
    CHECK(sm_context_create(&options, &result) == SM_OK);
    return result;
}
static sm_memory_info memory(sm_context* ctx) {
    sm_memory_info result{};
    result.struct_size = sizeof(result);
    result.abi_version = SM_RUNTIME_ABI_VERSION;
    CHECK(sm_context_memory_info(ctx, &result) == SM_OK);
    CHECK(result.total_bytes == result.owned_bytes + result.borrowed_bytes + result.reserved_bytes + result.external_bytes);
    CHECK(result.total_bytes <= result.limit_bytes && result.peak_bytes <= result.limit_bytes);
    CHECK(result.cached_bytes == 0 && result.scratch_bytes == 0 && !result.external_accounting_complete);
    return result;
}
static void bytes_equal(sm_buffer* buffer, const void* expected, uint64_t size) {
    const void* data = nullptr;
    uint64_t actual = 999;
    CHECK(sm_buffer_data(buffer, &data, &actual) == SM_OK);
    CHECK(actual == size);
    CHECK(!size || std::memcmp(data, expected, static_cast<size_t>(size)) == 0);
}
struct Owner {
    std::atomic<unsigned> refs{1}, retained{0}, released{0};
    sm_context* reenter = nullptr;
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
static sm_buffer_view foreign(Owner& owner, void* base, uint64_t capacity, uint64_t offset,
                              uint64_t length, uint64_t id = 1000000) {
    sm_buffer_view view{};
    view.struct_size = sizeof(view);
    view.abi_version = SM_BUFFER_ABI_VERSION;
    view.kind = SM_BUFFER_CPU;
    view.flags = SM_BUFFER_HOST_READABLE | SM_BUFFER_IMMUTABLE | SM_BUFFER_PRODUCER_COMPLETE;
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
    sm_buffer* buffer = nullptr;
    CHECK(sm_buffer_copy_from_host(ctx, input.data(), input.size(), &buffer) == SM_OK);
    bytes_equal(buffer, input.data(), input.size());
    sm_buffer_view whole{};
    CHECK(sm_buffer_export(buffer, &whole) == SM_OK);
    CHECK(whole.capacity == 192 && whole.byte_offset == 0);
    CHECK(reinterpret_cast<uintptr_t>(whole.host_base) % 64 == 0);
    CHECK(sm_buffer_validate(&whole, 64) == SM_OK);
    const auto* data = static_cast<const unsigned char*>(whole.host_base);
    for (uint64_t index = input.size(); index < whole.capacity; ++index) CHECK(data[index] == 0);
    auto accounting = memory(ctx);
    CHECK(accounting.owned_bytes == 192 && accounting.allocation_count == 1);
    sm_buffer* slice = nullptr;
    CHECK(sm_buffer_slice(buffer, 3, 7, &slice) == SM_OK);
    sm_buffer* nested = nullptr;
    CHECK(sm_buffer_slice(slice, 2, 4, &nested) == SM_OK);
    bytes_equal(nested, input.data() + 5, 4);
    sm_buffer_view view{};
    CHECK(sm_buffer_export(nested, &view) == SM_OK);
    CHECK(view.allocation_id == whole.allocation_id && view.host_base == whole.host_base);
    CHECK(view.byte_offset == 5 && view.byte_length == 4 && view.capacity == whole.capacity);
    sm_import_result imported{};
    CHECK(sm_buffer_import(ctx, &view, SM_COPY_NEVER, &imported) == SM_OK);
    CHECK(imported.copied_bytes == 0 && imported.borrowed_bytes == 192);
    CHECK(memory(ctx).owned_bytes == 192 && memory(ctx).borrowed_bytes == 0);
    sm_buffer_release(imported.buffer);
    sm_buffer_release(nested);
    sm_buffer_release(slice);
    sm_buffer_release(buffer);
    whole.release(whole.owner);
    CHECK(memory(ctx).owned_bytes == 192); // the last export pins the allocation
    sm_context_release(ctx);
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
    sm_import_result first{}, second{};
    CHECK(sm_buffer_import(ctx, &view, SM_COPY_NEVER, &first) == SM_OK);
    view.byte_offset = 9;
    view.byte_length = 7;
    CHECK(sm_buffer_import(ctx, &view, SM_COPY_AUTO, &second) == SM_OK);
    CHECK(owner.refs == 2 && owner.retained == 1);
    CHECK(first.borrowed_bytes == 64 && first.copied_bytes == 0);
    CHECK(memory(ctx).borrowed_bytes == 64 && memory(ctx).allocation_count == 1);
    bytes_equal(first.buffer, input.data() + 7, 9);
    bytes_equal(second.buffer, input.data() + 9, 7);
    sm_buffer_release(first.buffer);
    sm_buffer_release(second.buffer);
    CHECK(owner.refs == 1 && owner.released == 1);
    CHECK(memory(ctx).total_bytes == 0);

    view.flags &= ~SM_BUFFER_IMMUTABLE;
    sm_import_result unchanged{reinterpret_cast<sm_buffer*>(uintptr_t{17}), 41, 42};
    CHECK(sm_buffer_import(ctx, &view, SM_COPY_NEVER, &unchanged) == SM_ENOTSUP);
    CHECK(unchanged.buffer == reinterpret_cast<sm_buffer*>(uintptr_t{17}) && unchanged.copied_bytes == 41 && unchanged.borrowed_bytes == 42);
    sm_import_result snapshot{};
    CHECK(sm_buffer_import(ctx, &view, SM_COPY_AUTO, &snapshot) == SM_OK);
    CHECK(snapshot.copied_bytes == 7 && snapshot.borrowed_bytes == 0 && owner.refs == 1);
    auto copied = std::array<unsigned char, 7>{};
    std::memcpy(copied.data(), input.data() + 9, copied.size());
    input[9] ^= 0xff;
    bytes_equal(snapshot.buffer, copied.data(), copied.size());
    auto accounting = memory(ctx);
    CHECK(accounting.owned_bytes == 128 && accounting.borrowed_bytes == 0 && accounting.peak_bytes == 192);
    sm_buffer_release(snapshot.buffer);
    CHECK(owner.retained == owner.released && memory(ctx).total_bytes == 0);
    sm_context_release(ctx);
}

static void independent_exports_cross_context() {
    auto* source = context();
    auto* target = context(128);
    const unsigned char input[] = {4, 5, 6};
    sm_buffer* original = nullptr;
    CHECK(sm_buffer_copy_from_host(source, input, sizeof(input), &original) == SM_OK);
    sm_buffer_view left{}, right{};
    CHECK(sm_buffer_export(original, &left) == SM_OK);
    CHECK(sm_buffer_export(original, &right) == SM_OK);
    CHECK(left.owner != right.owner);
    sm_import_result a{}, b{};
    CHECK(sm_buffer_import(target, &left, SM_COPY_NEVER, &a) == SM_OK);
    CHECK(sm_buffer_import(target, &right, SM_COPY_NEVER, &b) == SM_OK);
    CHECK(memory(target).borrowed_bytes == 128 && memory(target).allocation_count == 1);
    left.release(left.owner);
    right.release(right.owner);
    sm_buffer_release(original);
    sm_context_release(source);
    bytes_equal(a.buffer, input, sizeof(input));
    sm_buffer_view propagated{};
    CHECK(sm_buffer_export(a.buffer, &propagated) == SM_OK);
    sm_import_result c{};
    CHECK(sm_buffer_import(target, &propagated, SM_COPY_NEVER, &c) == SM_OK);
    CHECK(memory(target).allocation_count == 1);
    propagated.release(propagated.owner);
    sm_buffer_release(a.buffer);
    sm_buffer_release(b.buffer);
    sm_buffer_release(c.buffer);
    CHECK(memory(target).total_bytes == 0);
    sm_context_release(target);
}

static void rejected_inputs_and_rollback() {
    auto* ctx = context(191);
    std::array<unsigned char, 64> bytes{};
    Owner owner;
    auto view = foreign(owner, bytes.data(), bytes.size(), 0, 3);
    sm_import_result output{reinterpret_cast<sm_buffer*>(uintptr_t{17}), 41, 42};
    CHECK(sm_buffer_import(ctx, &view, SM_COPY_ALWAYS, &output) == SM_EOOM);
    CHECK(output.buffer == reinterpret_cast<sm_buffer*>(uintptr_t{17}) && output.copied_bytes == 41 && output.borrowed_bytes == 42);
    CHECK(owner.refs == 1 && owner.retained == owner.released);
    CHECK(memory(ctx).total_bytes == 0 && memory(ctx).reserved_bytes == 0);

    sm_import_result borrowed{};
    CHECK(sm_buffer_import(ctx, &view, SM_COPY_NEVER, &borrowed) == SM_OK);
    std::array<unsigned char, 64> other{};
    auto collision = view;
    collision.host_base = other.data();
    CHECK(sm_buffer_import(ctx, &collision, SM_COPY_NEVER, &output) == SM_EINVAL);
    Owner other_owner;
    collision = view;
    collision.owner = &other_owner;
    CHECK(sm_buffer_import(ctx, &collision, SM_COPY_NEVER, &output) == SM_EINVAL);
    CHECK(other_owner.retained == 0);
    CHECK(memory(ctx).borrowed_bytes == 64 && owner.refs == 2);
    sm_buffer_release(borrowed.buffer);

    auto metal = view;
    metal.kind = SM_BUFFER_METAL;
    metal.device_id = 1;
    metal.metal_buffer = bytes.data();
    CHECK(sm_buffer_import(ctx, &metal, SM_COPY_ALWAYS, &output) == SM_ENOTSUP);
    view.byte_offset = 63;
    view.byte_length = 2;
    CHECK(sm_buffer_import(ctx, &view, SM_COPY_AUTO, &output) == SM_EINVAL);
    view.byte_offset = UINT64_MAX;
    CHECK(sm_buffer_import(ctx, &view, SM_COPY_AUTO, &output) == SM_EOVERFLOW);

    sm_buffer* sentinel = reinterpret_cast<sm_buffer*>(uintptr_t{17});
    CHECK(sm_buffer_zeros(ctx, UINT64_MAX, &sentinel) == SM_EOVERFLOW);
    CHECK(sentinel == reinterpret_cast<sm_buffer*>(uintptr_t{17}));
    CHECK(sm_buffer_zeros(ctx, 128, &sentinel) == SM_EOOM); // capacity 192
    CHECK(sentinel == reinterpret_cast<sm_buffer*>(uintptr_t{17}));
    char error[256]{};
    CHECK(sm_context_error_copy(ctx, error, sizeof(error)) == SM_OK && error[0]);
    char tiny = 'x';
    CHECK(sm_context_error_copy(ctx, &tiny, 1) == SM_EINVAL && tiny == 'x');
    char again[256]{};
    CHECK(sm_context_error_copy(ctx, again, sizeof(again)) == SM_OK && std::strcmp(error, again) == 0);
    CHECK(memory(ctx).total_bytes == 0);
    sm_context_release(ctx);
}

static void empty_bounds_and_reservations() {
    auto* ctx = context(256);
    sm_buffer* empty = nullptr;
    CHECK(sm_buffer_copy_from_host(ctx, nullptr, 0, &empty) == SM_OK);
    CHECK(memory(ctx).owned_bytes == 64);
    sm_buffer* slice = nullptr;
    CHECK(sm_buffer_slice(empty, 0, 0, &slice) == SM_OK);
    bytes_equal(slice, nullptr, 0);
    sm_buffer* unchanged = empty;
    CHECK(sm_buffer_slice(empty, 0, 1, &unchanged) == SM_EINVAL && unchanged == empty);
    CHECK(sm_buffer_slice(empty, UINT64_MAX, 1, &unchanged) == SM_EOVERFLOW && unchanged == empty);
    const void* pointer = reinterpret_cast<void*>(uintptr_t{17});
    CHECK(sm_buffer_data(empty, &pointer, nullptr) == SM_EINVAL && pointer == reinterpret_cast<void*>(uintptr_t{17}));
    sm_buffer_release(slice);
    sm_buffer_release(empty);
    {
        auto reservation = sm::detail::reserve(ctx, 256);
        CHECK(memory(ctx).reserved_bytes == 256);
        auto first = sm::detail::allocate_reserved(ctx, reservation, 1);
        CHECK(memory(ctx).reserved_bytes == 128 && memory(ctx).owned_bytes == 128);
        try { auto invalid = sm::detail::allocate_reserved(ctx, reservation, 65); CHECK(false); }
        catch (const sm::detail::Error& error) { CHECK(error.status == SM_EINVAL); }
        CHECK(memory(ctx).reserved_bytes == 128 && memory(ctx).owned_bytes == 128);
    }
    CHECK(memory(ctx).total_bytes == 0);
    Owner owner;
    auto view = foreign(owner, nullptr, 0, 0, 0);
    sm_import_result zero{};
    CHECK(sm_buffer_import(ctx, &view, SM_COPY_NEVER, &zero) == SM_OK);
    bytes_equal(zero.buffer, nullptr, 0);
    CHECK(memory(ctx).borrowed_bytes == 0 && memory(ctx).allocation_count == 1);
    sm_buffer_release(zero.buffer);
    CHECK(owner.refs == 1);
    sm_context_release(ctx);
}

static void concurrent_admission_and_aliases() {
    constexpr unsigned count = 24;
    auto* ctx = context(4 * 128 + 17, 17);
    std::barrier barrier(count + 1);
    std::atomic<unsigned> admitted{0};
    std::vector<std::thread> threads;
    for (unsigned index = 0; index < count; ++index) threads.emplace_back([&] {
        sm_buffer* buffer = nullptr;
        const auto status = sm_buffer_zeros(ctx, 1, &buffer);
        CHECK(status == SM_OK || status == SM_EOOM);
        if (status == SM_OK) ++admitted;
        else CHECK(!buffer);
        barrier.arrive_and_wait();
        barrier.arrive_and_wait();
        sm_buffer_release(buffer);
    });
    barrier.arrive_and_wait();
    CHECK(admitted == 4);
    CHECK(memory(ctx).owned_bytes == 512 && memory(ctx).allocation_count == 4);
    barrier.arrive_and_wait();
    for (auto& thread : threads) thread.join();
    CHECK(memory(ctx).total_bytes == 17 && memory(ctx).reserved_bytes == 0);
    sm_context_release(ctx);

    ctx = context(64);
    Owner owner;
    std::array<unsigned char, 64> data{};
    auto view = foreign(owner, data.data(), 64, 2, 4);
    threads.clear();
    for (unsigned index = 0; index < count; ++index) threads.emplace_back([&] {
        sm_import_result result{};
        CHECK(sm_buffer_import(ctx, &view, SM_COPY_NEVER, &result) == SM_OK);
        barrier.arrive_and_wait();
        barrier.arrive_and_wait();
        sm_buffer_release(result.buffer);
    });
    barrier.arrive_and_wait();
    CHECK(memory(ctx).borrowed_bytes == 64 && memory(ctx).allocation_count == 1 && owner.refs == 2);
    barrier.arrive_and_wait();
    for (auto& thread : threads) thread.join();
    CHECK(owner.refs == 1 && owner.retained == owner.released && memory(ctx).total_bytes == 0);
    sm_context_release(ctx);
}

static void fork_guards() {
    auto* ctx = context();
    sm_buffer* buffer = nullptr;
    CHECK(sm_buffer_zeros(ctx, 3, &buffer) == SM_OK);
    sm_buffer_view view{};
    CHECK(sm_buffer_export(buffer, &view) == SM_OK);
    const auto child = fork();
    CHECK(child >= 0);
    if (!child) {
        alarm(5);
        sm_buffer* output = buffer;
        CHECK(sm_buffer_zeros(ctx, 1, &output) == SM_ESTATE && output == buffer);
        const void* data = nullptr;
        uint64_t bytes = 17;
        CHECK(sm_buffer_data(buffer, &data, &bytes) == SM_ESTATE && !data && bytes == 17);
        char error = 'x';
        CHECK(sm_context_error_copy(ctx, &error, 1) == SM_ESTATE && error == 'x');
        auto* fresh = context();
        sm_import_result imported{};
        CHECK(sm_buffer_import(fresh, &view, SM_COPY_ALWAYS, &imported) == SM_ESTATE && !imported.buffer);
        sm_context_release(fresh);
        sm_buffer_retain(buffer);
        sm_buffer_release(buffer);
        sm_context_retain(ctx);
        sm_context_release(ctx);
        view.retain(view.owner);
        view.release(view.owner);
        _exit(0);
    }
    int status = 0;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    view.release(view.owner);
    sm_buffer_release(buffer);
    CHECK(memory(ctx).total_bytes == 0);
    sm_context_release(ctx);
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
