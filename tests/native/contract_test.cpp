#include "sm.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)
static int callbacks;
static void retain(void*) { ++callbacks; }
static void release(void*) { --callbacks; }
int main() {
    alignas(64) char data[128]{};
    sm_buffer_view v{};
    v.struct_size = sizeof(v); v.abi_version = SM_BUFFER_ABI_VERSION;
    v.kind = SM_BUFFER_CPU;
    v.flags = SM_BUFFER_HOST_READABLE | SM_BUFFER_PRODUCER_COMPLETE | SM_BUFFER_IMMUTABLE;
    v.allocation_id = 1; v.host_base = data; v.owner = data;
    v.retain = retain; v.release = release; v.capacity = sizeof(data);
    v.byte_offset = 16; v.byte_length = 32;
    CHECK(sm_buffer_validate(&v, 8) == SM_OK);
    CHECK(callbacks == 0); // Validation must neither acquire nor consume a reference.
    auto bad = v; bad.byte_offset = UINT64_MAX;
    CHECK(sm_buffer_validate(&bad, 1) == SM_EOVERFLOW);
    bad = v; bad.byte_length = 113;
    CHECK(sm_buffer_validate(&bad, 8) == SM_EINVAL);
    bad = v; bad.byte_offset = 17;
    CHECK(sm_buffer_validate(&bad, 8) == SM_EINVAL);
    bad = v; bad.host_base = data + 1;
    CHECK(sm_buffer_validate(&bad, 8) == SM_EINVAL);
    bad.byte_offset = 7; // CPU alignment is on the logical pointer, not a subrange's base.
    CHECK(sm_buffer_validate(&bad, 8) == SM_OK);
    bad = v; bad.release = nullptr;
    CHECK(sm_buffer_validate(&bad, 8) == SM_EINVAL);
    bad = v; bad.struct_size = 8;
    CHECK(sm_buffer_validate(&bad, 8) == SM_EINVAL);
    bad = v; bad.abi_version = 2;
    CHECK(sm_buffer_validate(&bad, 8) == SM_ENOTSUP);
    bad = v; bad.reserved[0] = 1;
    CHECK(sm_buffer_validate(&bad, 8) == SM_ENOTSUP);
    bad = v; bad.byte_length = 0; bad.byte_offset = bad.capacity;
    CHECK(sm_buffer_validate(&bad, 8) == SM_OK);
    bad = v; bad.flags &= ~SM_BUFFER_PRODUCER_COMPLETE;
    CHECK(sm_buffer_validate(&bad, 8) == SM_EINVAL);
    bad.kind = SM_BUFFER_METAL; bad.metal_buffer = data; bad.device_id = 7;
    CHECK(sm_buffer_validate(&bad, 8) == SM_EINVAL);
    bad.completion_event = data; bad.completion_value = 42;
    CHECK(sm_buffer_validate(&bad, 8) == SM_OK);
    bad.completion_value = 0;
    CHECK(sm_buffer_validate(&bad, 8) == SM_EINVAL);
    uint64_t out = 123;
    CHECK(sm_checked_mul(UINT64_MAX, 2, &out) == SM_EOVERFLOW && out == 123);
    CHECK(sm_checked_add(UINT64_MAX, 1, &out) == SM_EOVERFLOW && out == 123);
    CHECK(sm_checked_mul(UINT64_MAX, 0, &out) == SM_OK && out == 0);
    CHECK(sm_checked_mul(UINT64_C(1) << 32, 8, &out) == SM_OK && out == (UINT64_C(1) << 35));
    CHECK(sm_buffer_validate(nullptr, 8) == SM_EINVAL);
    CHECK(sm_buffer_validate(&v, 3) == SM_EINVAL);
    CHECK(sm_checked_add(1, 2, nullptr) == SM_EINVAL);
    std::puts("S0 C ABI, range, ownership and overflow checks passed (no Metal)");
}
