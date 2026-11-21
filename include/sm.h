#ifndef SM_H
#define SM_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* S0 descriptor contract, version 1. This is not an engine API.
 * Imports copy descriptor fields and retain the owner; they do not consume it.
 * The caller's descriptor storage may expire after import returns.
 * Callbacks must be thread-safe, must not throw, and must not require the Python GIL.
 * Owners retain all backing resources and the completion event independently.
 * Without IMMUTABLE, import must enforce a no-concurrent-write agreement or
 * take a snapshot. Structural validation alone does not enforce immutability.
 */
#define SM_BUFFER_ABI_VERSION 1u
typedef enum sm_status {
    SM_OK = 0, SM_EINVAL = 1, SM_ENOTSUP = 2, SM_EOVERFLOW = 3,
    SM_EOOM = 4, SM_ETIMEOUT = 5, SM_EBACKEND = 6, SM_ESTATE = 7
} sm_status;
typedef enum sm_buffer_kind {
    SM_BUFFER_CPU = 1, SM_BUFFER_METAL = 2
} sm_buffer_kind;
enum {
    SM_BUFFER_HOST_READABLE = 1u,
    SM_BUFFER_IMMUTABLE = 2u,
    SM_BUFFER_PRODUCER_COMPLETE = 4u
};
typedef void (*sm_owner_callback)(void*);
typedef struct sm_buffer_view {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t kind;
    uint32_t flags;
    uint64_t allocation_id;
    uint64_t device_id;
    void* host_base;
    void* metal_buffer;
    uint64_t byte_offset;
    uint64_t byte_length;
    uint64_t capacity;
    void* owner;
    sm_owner_callback retain;
    sm_owner_callback release;
    void* completion_event;
    uint64_t completion_value;
    uint64_t reserved[2];
} sm_buffer_view;

/* Structural checks only: does not dereference pointers, retain, wait, or
 * initialize Metal. Resource/device/event identity is checked at import.
 * alignment must be a nonzero power of two; zero-length views are allowed.
 */
sm_status sm_buffer_validate(const sm_buffer_view* view, uint64_t alignment);
/* Leaves *out unchanged on failure. */
sm_status sm_checked_add(uint64_t a, uint64_t b, uint64_t* out);
sm_status sm_checked_mul(uint64_t a, uint64_t b, uint64_t* out);

#ifdef __cplusplus
}
#endif
#endif
