#ifndef MS_H
#define MS_H
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
#define MS_BUFFER_ABI_VERSION 1u
typedef enum ms_status {
    MS_OK = 0, MS_EINVAL = 1, MS_ENOTSUP = 2, MS_EOVERFLOW = 3,
    MS_EOOM = 4, MS_ETIMEOUT = 5, MS_EBACKEND = 6, MS_ESTATE = 7
} ms_status;
typedef enum ms_buffer_kind {
    MS_BUFFER_CPU = 1, MS_BUFFER_METAL = 2
} ms_buffer_kind;
enum {
    MS_BUFFER_HOST_READABLE = 1u,
    MS_BUFFER_IMMUTABLE = 2u,
    MS_BUFFER_PRODUCER_COMPLETE = 4u
};
typedef void (*ms_owner_callback)(void*);
typedef struct ms_buffer_view {
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
    ms_owner_callback retain;
    ms_owner_callback release;
    void* completion_event;
    uint64_t completion_value;
    uint64_t reserved[2];
} ms_buffer_view;

/* Structural checks only: does not dereference pointers, retain, wait, or
 * initialize Metal. Resource/device/event identity is checked at import.
 * alignment must be a nonzero power of two; zero-length views are allowed.
 */
ms_status ms_buffer_validate(const ms_buffer_view* view, uint64_t alignment);
/* Leaves *out unchanged on failure. */
ms_status ms_checked_add(uint64_t a, uint64_t b, uint64_t* out);
ms_status ms_checked_mul(uint64_t a, uint64_t b, uint64_t* out);

#ifdef __cplusplus
}
#endif
#endif
