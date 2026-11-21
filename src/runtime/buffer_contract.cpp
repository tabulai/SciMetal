#include "sm.h"
#include <limits>

extern "C" sm_status sm_checked_add(uint64_t a, uint64_t b, uint64_t* out) {
    if (!out) return SM_EINVAL;
    if (b > std::numeric_limits<uint64_t>::max() - a) return SM_EOVERFLOW;
    *out = a + b;
    return SM_OK;
}
extern "C" sm_status sm_checked_mul(uint64_t a, uint64_t b, uint64_t* out) {
    if (!out) return SM_EINVAL;
    if (a && b > std::numeric_limits<uint64_t>::max() / a) return SM_EOVERFLOW;
    *out = a * b;
    return SM_OK;
}
extern "C" sm_status sm_buffer_validate(const sm_buffer_view* v, uint64_t align) {
    if (!v || v->struct_size < sizeof(sm_buffer_view)) return SM_EINVAL;
    if (v->abi_version != SM_BUFFER_ABI_VERSION) return SM_ENOTSUP;
    if (!align || (align & (align - 1))) return SM_EINVAL;
    if (v->flags & ~(SM_BUFFER_HOST_READABLE | SM_BUFFER_IMMUTABLE |
                     SM_BUFFER_PRODUCER_COMPLETE)) return SM_ENOTSUP;
    if (v->reserved[0] || v->reserved[1]) return SM_ENOTSUP;
    if (!v->owner || !v->retain || !v->release || !v->allocation_id) return SM_EINVAL;
    uint64_t end;
    auto status = sm_checked_add(v->byte_offset, v->byte_length, &end);
    if (status != SM_OK) return status;
    if (end > v->capacity) return SM_EINVAL;
    if (v->flags & SM_BUFFER_HOST_READABLE) {
        if (!v->host_base && v->capacity) return SM_EINVAL;
        uint64_t address;
        status = sm_checked_add(reinterpret_cast<uintptr_t>(v->host_base),
                                v->byte_offset, &address);
        if (status != SM_OK) return status;
        if (address % align) return SM_EINVAL;
        status = sm_checked_add(reinterpret_cast<uintptr_t>(v->host_base),
                                v->capacity, &address);
        if (status != SM_OK) return status;
    } else if (v->host_base) {
        return SM_EINVAL;
    }
    if (v->kind == SM_BUFFER_CPU) {
        if (!(v->flags & SM_BUFFER_HOST_READABLE) || v->metal_buffer ||
            v->device_id || v->completion_event || v->completion_value ||
            !(v->flags & SM_BUFFER_PRODUCER_COMPLETE)) return SM_EINVAL;
    } else if (v->kind == SM_BUFFER_METAL) {
        if (!v->metal_buffer || !v->device_id) return SM_EINVAL;
        if (v->byte_offset % align) return SM_EINVAL;
        if (!(v->flags & SM_BUFFER_PRODUCER_COMPLETE) &&
            (!v->completion_event || !v->completion_value)) return SM_EINVAL;
        if (!v->completion_event && v->completion_value) return SM_EINVAL;
    } else {
        return SM_ENOTSUP;
    }
    return SM_OK;
}
