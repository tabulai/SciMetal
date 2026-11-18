#include "ms.h"
#include <limits>

extern "C" ms_status ms_checked_add(uint64_t a, uint64_t b, uint64_t* out) {
    if (!out) return MS_EINVAL;
    if (b > std::numeric_limits<uint64_t>::max() - a) return MS_EOVERFLOW;
    *out = a + b;
    return MS_OK;
}
extern "C" ms_status ms_checked_mul(uint64_t a, uint64_t b, uint64_t* out) {
    if (!out) return MS_EINVAL;
    if (a && b > std::numeric_limits<uint64_t>::max() / a) return MS_EOVERFLOW;
    *out = a * b;
    return MS_OK;
}
extern "C" ms_status ms_buffer_validate(const ms_buffer_view* v, uint64_t align) {
    if (!v || v->struct_size < sizeof(ms_buffer_view)) return MS_EINVAL;
    if (v->abi_version != MS_BUFFER_ABI_VERSION) return MS_ENOTSUP;
    if (!align || (align & (align - 1))) return MS_EINVAL;
    if (v->flags & ~(MS_BUFFER_HOST_READABLE | MS_BUFFER_IMMUTABLE |
                     MS_BUFFER_PRODUCER_COMPLETE)) return MS_ENOTSUP;
    if (v->reserved[0] || v->reserved[1]) return MS_ENOTSUP;
    if (!v->owner || !v->retain || !v->release || !v->allocation_id) return MS_EINVAL;
    uint64_t end;
    auto status = ms_checked_add(v->byte_offset, v->byte_length, &end);
    if (status != MS_OK) return status;
    if (end > v->capacity) return MS_EINVAL;
    if (v->flags & MS_BUFFER_HOST_READABLE) {
        if (!v->host_base && v->capacity) return MS_EINVAL;
        uint64_t address;
        status = ms_checked_add(reinterpret_cast<uintptr_t>(v->host_base),
                                v->byte_offset, &address);
        if (status != MS_OK) return status;
        if (address % align) return MS_EINVAL;
        status = ms_checked_add(reinterpret_cast<uintptr_t>(v->host_base),
                                v->capacity, &address);
        if (status != MS_OK) return status;
    } else if (v->host_base) {
        return MS_EINVAL;
    }
    if (v->kind == MS_BUFFER_CPU) {
        if (!(v->flags & MS_BUFFER_HOST_READABLE) || v->metal_buffer ||
            v->device_id || v->completion_event || v->completion_value ||
            !(v->flags & MS_BUFFER_PRODUCER_COMPLETE)) return MS_EINVAL;
    } else if (v->kind == MS_BUFFER_METAL) {
        if (!v->metal_buffer || !v->device_id) return MS_EINVAL;
        if (v->byte_offset % align) return MS_EINVAL;
        if (!(v->flags & MS_BUFFER_PRODUCER_COMPLETE) &&
            (!v->completion_event || !v->completion_value)) return MS_EINVAL;
        if (!v->completion_event && v->completion_value) return MS_EINVAL;
    } else {
        return MS_ENOTSUP;
    }
    return MS_OK;
}
