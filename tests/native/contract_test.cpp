#include "ms.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)
static int callbacks;
static void retain(void*) { ++callbacks; }
static void release(void*) { --callbacks; }
int main() {
    alignas(64) char data[128]{};
    ms_buffer_view v{};
    v.struct_size = sizeof(v); v.abi_version = MS_BUFFER_ABI_VERSION;
    v.kind = MS_BUFFER_CPU;
    v.flags = MS_BUFFER_HOST_READABLE | MS_BUFFER_PRODUCER_COMPLETE | MS_BUFFER_IMMUTABLE;
    v.allocation_id = 1; v.host_base = data; v.owner = data;
    v.retain = retain; v.release = release; v.capacity = sizeof(data);
    v.byte_offset = 16; v.byte_length = 32;
    CHECK(ms_buffer_validate(&v, 8) == MS_OK);
    CHECK(callbacks == 0); // Validation must neither acquire nor consume a reference.
    auto bad = v; bad.byte_offset = UINT64_MAX;
    CHECK(ms_buffer_validate(&bad, 1) == MS_EOVERFLOW);
    bad = v; bad.byte_length = 113;
    CHECK(ms_buffer_validate(&bad, 8) == MS_EINVAL);
    bad = v; bad.byte_offset = 17;
    CHECK(ms_buffer_validate(&bad, 8) == MS_EINVAL);
    bad = v; bad.host_base = data + 1;
    CHECK(ms_buffer_validate(&bad, 8) == MS_EINVAL);
    bad.byte_offset = 7; // CPU alignment is on the logical pointer, not a subrange's base.
    CHECK(ms_buffer_validate(&bad, 8) == MS_OK);
    bad = v; bad.release = nullptr;
    CHECK(ms_buffer_validate(&bad, 8) == MS_EINVAL);
    bad = v; bad.struct_size = 8;
    CHECK(ms_buffer_validate(&bad, 8) == MS_EINVAL);
    bad = v; bad.abi_version = 2;
    CHECK(ms_buffer_validate(&bad, 8) == MS_ENOTSUP);
    bad = v; bad.reserved[0] = 1;
    CHECK(ms_buffer_validate(&bad, 8) == MS_ENOTSUP);
    bad = v; bad.byte_length = 0; bad.byte_offset = bad.capacity;
    CHECK(ms_buffer_validate(&bad, 8) == MS_OK);
    bad = v; bad.flags &= ~MS_BUFFER_PRODUCER_COMPLETE;
    CHECK(ms_buffer_validate(&bad, 8) == MS_EINVAL);
    bad.kind = MS_BUFFER_METAL; bad.metal_buffer = data; bad.device_id = 7;
    CHECK(ms_buffer_validate(&bad, 8) == MS_EINVAL);
    bad.completion_event = data; bad.completion_value = 42;
    CHECK(ms_buffer_validate(&bad, 8) == MS_OK);
    bad.completion_value = 0;
    CHECK(ms_buffer_validate(&bad, 8) == MS_EINVAL);
    uint64_t out = 123;
    CHECK(ms_checked_mul(UINT64_MAX, 2, &out) == MS_EOVERFLOW && out == 123);
    CHECK(ms_checked_add(UINT64_MAX, 1, &out) == MS_EOVERFLOW && out == 123);
    CHECK(ms_checked_mul(UINT64_MAX, 0, &out) == MS_OK && out == 0);
    CHECK(ms_checked_mul(UINT64_C(1) << 32, 8, &out) == MS_OK && out == (UINT64_C(1) << 35));
    CHECK(ms_buffer_validate(nullptr, 8) == MS_EINVAL);
    CHECK(ms_buffer_validate(&v, 3) == MS_EINVAL);
    CHECK(ms_checked_add(1, 2, nullptr) == MS_EINVAL);
    std::puts("S0 C ABI, range, ownership and overflow checks passed (no Metal)");
}
