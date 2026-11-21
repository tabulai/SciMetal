#ifndef MS_RUNTIME_H
#define MS_RUNTIME_H
#include "ms.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Experimental S1 ABI: synchronous immutable storage, native CPU operations and
 * forced-GPU routes for the raw primitives, not a complete Arrow/column adapter.
 * Functions returning status leave public outputs unchanged on failure.
 * Retain/release are thread-safe; callers keep handles alive during each call.
 * Inherited handles are unusable after fork (including release); use exec/spawn.
 */
#define MS_RUNTIME_ABI_VERSION 1u
typedef struct ms_context ms_context;
typedef struct ms_buffer ms_buffer;
typedef enum ms_copy_policy {
    MS_COPY_AUTO = 0, MS_COPY_NEVER = 1, MS_COPY_ALWAYS = 2
} ms_copy_policy;
typedef enum ms_execution {
    MS_EXECUTION_CPU = 1, MS_EXECUTION_GPU = 2, MS_EXECUTION_AUTO = 3
} ms_execution;

/* Context flag: storage that native producers promise to overwrite completely
 * is filled with 0xA5 bytes rather than left uninitialized, so tests can detect
 * output a producer failed to write. Padding is zero either way. */
#define MS_CONTEXT_POISON_UNWRITTEN 1u

typedef struct ms_context_options {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t storage_kind; /* MS_BUFFER_CPU or MS_BUFFER_METAL; no eager device */
    uint32_t flags;        /* MS_CONTEXT_* bits; other bits are invalid */
    uint64_t memory_limit_bytes;
    uint64_t external_reservation_bytes;
    uint64_t reserved[2];
} ms_context_options;

typedef struct ms_memory_info {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t limit_bytes;
    uint64_t owned_bytes;       /* physical padded capacities, unique live allocations */
    uint64_t borrowed_bytes;    /* unique declared foreign capacities */
    uint64_t reserved_bytes;    /* pending output/scratch reservations */
    uint64_t cached_bytes;      /* idle Metal storage kept for reuse (see cache below) */
    uint64_t scratch_bytes;     /* subset of owned/reserved, not additive; not yet tracked */
    uint64_t external_bytes;    /* configured conservative headroom */
    uint64_t total_bytes;       /* owned + borrowed + reserved + cached + external */
    uint64_t peak_bytes;
    uint64_t allocation_count;  /* live allocations; cached storage excluded */
    uint32_t external_accounting_complete; /* zero: foreign heaps not intercepted */
    uint32_t reserved;
} ms_memory_info;

typedef struct ms_import_result {
    ms_buffer* buffer;          /* one retained reference on success */
    uint64_t copied_bytes;      /* actual payload copy; padding is separate */
    uint64_t borrowed_bytes;    /* full source capacity retained by result */
} ms_import_result;

ms_status ms_context_create(const ms_context_options*, ms_context** out);
void ms_context_retain(ms_context*);
void ms_context_release(ms_context*);
/* Initialize out.struct_size and out.abi_version before calling. */
ms_status ms_context_memory_info(ms_context*, ms_memory_info* out);
/* Copy this thread's latest diagnostic for the context (falling back to the
 * context's latest diagnostic), including NUL; insufficient capacity returns
 * MS_EINVAL without modifying destination. */
ms_status ms_context_error_copy(ms_context*, char* destination, uint64_t capacity);
/* Monotonic per-context counters, so callers and tests can assert which route
 * ran and where storage came from. gpu_operations counts completed GPU
 * primitive calls; command buffers, dispatches and GPU time (summed
 * GPUStartTime..GPUEndTime) include work that later failed a check.
 * native_allocations counts new CPU or Metal allocations, cached_reuses owned
 * buffers served from the cache. host_wait_ns sums the host time from each
 * command buffer's commit until its wait returned (submission, GPU work and
 * wake-up), so host_wait_ns - gpu_time_ns approximates per-buffer overhead.
 * host_fill_bytes counts bytes storage zero-filled on the host when it handed
 * out a new or reused buffer (whole buffers for zeroed storage, padding
 * otherwise; poisoning excluded). Fills a route makes itself later, such as
 * clearing scratch between checks, are not counted.
 * Initialize struct_size and abi_version. */
typedef struct ms_execution_info {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t gpu_operations;
    uint64_t gpu_command_buffers;
    uint64_t gpu_dispatches;
    uint64_t gpu_time_ns;
    uint64_t native_allocations;
    uint64_t cached_reuses;
    uint64_t host_wait_ns;    /* formerly reserved[0]; the layout is unchanged */
    uint64_t host_fill_bytes; /* formerly reserved[1] */
} ms_execution_info;
ms_status ms_context_execution_info(ms_context*, ms_execution_info* out);

/* Buffer cache. A Metal-storage context keeps owned storage released by its
 * last reference (handle, slice or export) for reuse by a later allocation of
 * the same page-rounded capacity, most recently cached first. Reused storage
 * reads like new storage, under a new allocation ID. On checkout, large owned
 * buffers with explicit VM evidence of paged-out contents may discard those
 * dead contents before normal initialization; live handles, slices and exports
 * are never eligible.
 * Full cached capacities stay charged (cached_bytes) until reused or evicted,
 * least recently cached first: when an admission needs room, when the cache
 * exceeds its limit and when a Metal allocation fails. Admission precedes reuse,
 * so under budget pressure a reservation may evict storage it could have reused.
 * The automatic limit starts at a quarter of
 * (memory_limit_bytes - external_reservation_bytes).
 * Successful owned allocations grow it to the observed owned + cached bytes,
 * at most half that available budget; borrowed bytes and pending reservations
 * do not grow it. set_cache_limit fixes an explicit limit for this context,
 * disabling automatic growth; 0 disables caching. trim frees idle storage
 * without changing the limit or its automatic/explicit mode. Releasing the
 * last context handle frees the cache. CPU storage never
 * caches: its limit is 0 and a nonzero limit returns MS_ENOTSUP. */
ms_status ms_context_cache_limit(ms_context*, uint64_t* out);
ms_status ms_context_set_cache_limit(ms_context*, uint64_t bytes); /* shrinking evicts */
ms_status ms_context_trim(ms_context*); /* frees all cached storage */
const char* ms_status_message(ms_status);

/* Owned storage has 64-byte alignment and zeroed padding from its logical end
 * to its capacity, at least 64 bytes. CPU capacity rounds to 64; Metal capacity rounds
 * to the host VM page size and verifies native allocatedSize fits that charge.
 * Admission charges the rounded capacity. Inputs
 * to copy_from_host must be readable and stable until the synchronous return. */
ms_status ms_buffer_zeros(ms_context*, uint64_t bytes, ms_buffer** out);
ms_status ms_buffer_copy_from_host(ms_context*, const void*, uint64_t bytes, ms_buffer** out);
void ms_buffer_retain(ms_buffer*);
void ms_buffer_release(ms_buffer*);
ms_status ms_buffer_data(const ms_buffer*, const void** data, uint64_t* bytes);
ms_status ms_buffer_slice(const ms_buffer*, uint64_t offset, uint64_t length, ms_buffer** out);
/* Import copies descriptor fields; caller retains its own reference. First
 * slice admits ready CPU descriptors only. Never-copy requires IMMUTABLE and
 * CPU destination storage. Auto borrows eligible CPU inputs, otherwise copies.
 * Snapshot reads require the caller's no-concurrent-write agreement. Foreign
 * Metal descriptors, events and unknown device conventions reject explicitly.
 * Source owner is retained during copying; temporary source capacity and output
 * capacity must both fit admission. No pointer validity is inferred from flags. */
ms_status ms_buffer_import(ms_context*, const ms_buffer_view*, ms_copy_policy, ms_import_result* out);
/* Export acquires one owner reference. Release it exactly once through the
 * returned descriptor's callback. Exported bytes remain immutable by contract.
 * Slice tails are not padding: consumers must obey byte_length exactly. */
ms_status ms_buffer_export(const ms_buffer*, ms_buffer_view* out);

#ifdef __cplusplus
}
#endif
#endif
