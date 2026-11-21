#ifndef SM_RUNTIME_H
#define SM_RUNTIME_H
#include "sm.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Experimental S1 ABI: synchronous immutable storage, native CPU operations and
 * forced-GPU routes for the raw primitives, not a complete Arrow/column adapter.
 * Functions returning status leave public outputs unchanged on failure.
 * Retain/release are thread-safe; callers keep handles alive during each call.
 * Inherited handles are unusable after fork (including release); use exec/spawn.
 */
#define SM_RUNTIME_ABI_VERSION 1u
typedef struct sm_context sm_context;
typedef struct sm_buffer sm_buffer;
typedef enum sm_copy_policy {
    SM_COPY_AUTO = 0, SM_COPY_NEVER = 1, SM_COPY_ALWAYS = 2
} sm_copy_policy;
typedef enum sm_execution {
    SM_EXECUTION_CPU = 1, SM_EXECUTION_GPU = 2, SM_EXECUTION_AUTO = 3
} sm_execution;

/* Context flag: storage that native producers promise to overwrite completely
 * is filled with 0xA5 bytes rather than left uninitialized, so tests can detect
 * output a producer failed to write. Padding is zero either way. */
#define SM_CONTEXT_POISON_UNWRITTEN 1u

typedef struct sm_context_options {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t storage_kind; /* SM_BUFFER_CPU or SM_BUFFER_METAL; no eager device */
    uint32_t flags;        /* SM_CONTEXT_* bits; other bits are invalid */
    uint64_t memory_limit_bytes;
    uint64_t external_reservation_bytes;
    uint64_t reserved[2];
} sm_context_options;

typedef struct sm_memory_info {
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
} sm_memory_info;

typedef struct sm_import_result {
    sm_buffer* buffer;          /* one retained reference on success */
    uint64_t copied_bytes;      /* actual payload copy; padding is separate */
    uint64_t borrowed_bytes;    /* full source capacity retained by result */
} sm_import_result;

sm_status sm_context_create(const sm_context_options*, sm_context** out);
void sm_context_retain(sm_context*);
void sm_context_release(sm_context*);
/* Initialize out.struct_size and out.abi_version before calling. */
sm_status sm_context_memory_info(sm_context*, sm_memory_info* out);
/* Copy this thread's latest diagnostic for the context (falling back to the
 * context's latest diagnostic), including NUL; insufficient capacity returns
 * SM_EINVAL without modifying destination. */
sm_status sm_context_error_copy(sm_context*, char* destination, uint64_t capacity);
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
typedef struct sm_execution_info {
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
} sm_execution_info;
sm_status sm_context_execution_info(sm_context*, sm_execution_info* out);

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
 * caches: its limit is 0 and a nonzero limit returns SM_ENOTSUP. */
sm_status sm_context_cache_limit(sm_context*, uint64_t* out);
sm_status sm_context_set_cache_limit(sm_context*, uint64_t bytes); /* shrinking evicts */
sm_status sm_context_trim(sm_context*); /* frees all cached storage */
const char* sm_status_message(sm_status);

/* Owned storage has 64-byte alignment and zeroed padding from its logical end
 * to its capacity, at least 64 bytes. CPU capacity rounds to 64; Metal capacity rounds
 * to the host VM page size and verifies native allocatedSize fits that charge.
 * Admission charges the rounded capacity. Inputs
 * to copy_from_host must be readable and stable until the synchronous return. */
sm_status sm_buffer_zeros(sm_context*, uint64_t bytes, sm_buffer** out);
sm_status sm_buffer_copy_from_host(sm_context*, const void*, uint64_t bytes, sm_buffer** out);
void sm_buffer_retain(sm_buffer*);
void sm_buffer_release(sm_buffer*);
sm_status sm_buffer_data(const sm_buffer*, const void** data, uint64_t* bytes);
sm_status sm_buffer_slice(const sm_buffer*, uint64_t offset, uint64_t length, sm_buffer** out);
/* Import copies descriptor fields; caller retains its own reference. First
 * slice admits ready CPU descriptors only. Never-copy requires IMMUTABLE and
 * CPU destination storage. Auto borrows eligible CPU inputs, otherwise copies.
 * Snapshot reads require the caller's no-concurrent-write agreement. Foreign
 * Metal descriptors, events and unknown device conventions reject explicitly.
 * Source owner is retained during copying; temporary source capacity and output
 * capacity must both fit admission. No pointer validity is inferred from flags. */
sm_status sm_buffer_import(sm_context*, const sm_buffer_view*, sm_copy_policy, sm_import_result* out);
/* Export acquires one owner reference. Release it exactly once through the
 * returned descriptor's callback. Exported bytes remain immutable by contract.
 * Slice tails are not padding: consumers must obey byte_length exactly. */
sm_status sm_buffer_export(const sm_buffer*, sm_buffer_view* out);

#ifdef __cplusplus
}
#endif
#endif
