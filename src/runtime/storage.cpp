#include "ms/detail/storage.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <unistd.h>

namespace {
using ms::detail::AsyncSync;
using ms::detail::Error;
using ms::detail::RawAllocation;

constexpr uint64_t arena_slab_bytes = 256u << 10;
constexpr uint64_t arena_slice_alignment = 256;
constexpr uint64_t arena_max_slice_bytes = 32u << 10;
constexpr uint64_t arena_max_logical = arena_max_slice_bytes - 64;
constexpr uint64_t arena_min_headroom_bytes = 4u << 20;
constexpr uint64_t overfit_min_capacity = 1u << 20;

struct Key {
    uint32_t kind;
    uint64_t device, id;
    bool operator==(const Key&) const = default;
};
struct KeyHash {
    size_t operator()(const Key& key) const noexcept {
        return std::hash<uint64_t>{}(key.id) ^
               (std::hash<uint64_t>{}(key.device) << 1) ^ key.kind;
    }
};
struct Allocation;
struct ArenaSlab;
struct CacheEntry {
    RawAllocation raw;
    std::shared_ptr<ArenaSlab> slab;
    uint64_t capacity() const noexcept;
};
struct ArenaSlab {
    RawAllocation raw;
    uint64_t bump = 0;
    uint64_t active_slices = 0;
    void* prior_event = nullptr;
    uint64_t prior_value = 0;
    bool in_active_list = false;
    bool in_idle_cache = false;
    std::list<CacheEntry>::iterator cache_it;
};
uint64_t CacheEntry::capacity() const noexcept {
    return slab ? slab->raw.capacity : raw.capacity;
}

struct Entry { std::weak_ptr<Allocation> allocation; uint64_t generation; };
struct State {
    pid_t pid = getpid();
    uint32_t kind;
    bool poison, open = true; // open: a context handle can still reuse cached storage
    uint64_t limit, external, owned = 0, borrowed = 0, pending = 0, cached = 0, peak, count = 0;
    uint64_t cache_limit;
    bool explicit_cache_limit = false;
    std::mutex mutex;
    char error[256]{};
    std::atomic<bool> has_error{false};
    std::unordered_map<Key, Entry, KeyHash> allocations;
    std::unordered_multimap<void*, Allocation*> live_metal;
    // Idle owned Metal storage, least recently cached first, indexed by
    // capacity. It stays charged as cached bytes until reused or evicted.
    std::list<CacheEntry> cache;
    std::multimap<uint64_t, std::list<CacheEntry>::iterator> cache_index;
    std::vector<std::shared_ptr<ArenaSlab>> active_slabs;
    std::vector<std::shared_ptr<ArenaSlab>> idle_slabs;
    std::mutex async_mutex;
    std::vector<std::shared_ptr<AsyncSync>> async_submissions;
    std::atomic<uint64_t> gpu_operations{0}, gpu_command_buffers{0}, gpu_dispatches{0}, gpu_ns{0};
    std::atomic<uint64_t> native_allocations{0}, cached_reuses{0};
    std::atomic<uint64_t> host_wait_ns{0}, host_fill_bytes{0};
    explicit State(const ms_context_options& options)
        : kind(options.storage_kind), poison((options.flags & MS_CONTEXT_POISON_UNWRITTEN) != 0),
          limit(options.memory_limit_bytes), external(options.external_reservation_bytes), peak(external),
          cache_limit(kind == MS_BUFFER_METAL ? (limit - external) / 4 : 0) {}
    uint64_t total() const noexcept { return external + owned + borrowed + pending + cached; }
    // Learn the actual owned-storage footprint, including idle scratch from an
    // earlier phase of a synchronous operation. Pending reservations can count
    // cached storage a second time, and borrowed storage cannot be reused here.
    // An explicit caller limit always wins, including a limit of zero.
    void observe_owned_storage() noexcept {
        if (kind == MS_BUFFER_METAL && !explicit_cache_limit)
            cache_limit = std::max(cache_limit, std::min(owned + cached, (limit - external) / 2));
    }
    // The caller holds mutex. Storage is freed before its bytes leave the budget.
    void evict_oldest() noexcept {
        const auto oldest = cache.begin();
        const uint64_t capacity = oldest->capacity();
        if (oldest->slab) {
            oldest->slab->in_idle_cache = false;
            auto it = std::find(idle_slabs.begin(), idle_slabs.end(), oldest->slab);
            if (it != idle_slabs.end()) idle_slabs.erase(it);
        } else {
            const auto range = cache_index.equal_range(capacity);
            for (auto entry = range.first; entry != range.second; ++entry)
                if (entry->second == oldest) {
                    cache_index.erase(entry);
                    break;
                }
        }
        cache.erase(oldest);
        cached -= capacity;
    }
    void shrink_cache(uint64_t bytes) noexcept {
        while (cached > bytes) evict_oldest();
    }
    // Idle cached storage gives way before admission fails.
    void admit(uint64_t bytes) {
        while (bytes > limit - total() && !cache.empty()) evict_oldest();
        if (bytes > limit - total()) throw Error(MS_EOOM, "context memory limit exceeded");
    }
    void mark_peak() noexcept { peak = std::max(peak, total()); }
};

void insert_live_metal(State& state, void* metal, Allocation* allocation) {
    if (metal) state.live_metal.emplace(metal, allocation);
}
void erase_live_metal(State& state, void* metal, Allocation* allocation) noexcept {
    if (!metal) return;
    const auto range = state.live_metal.equal_range(metal);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second == allocation) {
            state.live_metal.erase(it);
            break;
        }
    }
}

// An identity also includes the producer owner/callback convention. Equal IDs
// with different backing or owners are rejected, rather than treated as aliases.
struct Identity {
    void* owner;
    ms_owner_callback retain, release;
    bool operator==(const Identity&) const = default;
};
std::atomic<uint64_t> next_id{1}, next_generation{1};

struct ForeignHold {
    ms_buffer_view view{};
    bool active = false;
    explicit ForeignHold(const ms_buffer_view& source) : view(source) {
        view.retain(view.owner);
        active = true;
    }
    ~ForeignHold() noexcept {
        if (active) {
            // C owner callbacks must not throw. Contain a misbehaving callback
            // at the ABI boundary; there is no recoverable ownership protocol.
            try { view.release(view.owner); } catch (...) {}
        }
    }
};
struct PendingFill {
    bool active = false;
    bool zero_logical = false;
    bool poison_logical = false;
    uint64_t offset = 0;
    uint64_t logical = 0;
    uint64_t capacity = 0;
};
struct Allocation {
    std::shared_ptr<State> state;
    Key key{};
    uint64_t generation = next_generation.fetch_add(1, std::memory_order_relaxed);
    uint64_t capacity = 0, logical_capacity = 0, owned_logical_bytes = 0;
    void* host = nullptr;
    void* metal = nullptr;
    Identity identity{};
    // Foreign Arrow releases can re-enter the context. Destroy this hold after
    // the MTLBuffer, and outside the state lock held by the destructor body.
    std::shared_ptr<void> producer;
    RawAllocation raw;
    std::shared_ptr<ArenaSlab> arena_slab;
    uint64_t arena_offset = 0, arena_slice_bytes = 0;
    std::unique_ptr<ForeignHold> foreign;
    bool owned = false, charged = false;

    std::mutex sync_mutex;
    std::atomic<bool> host_ready{true};
    PendingFill pending_fill{};
    void* prior_event = nullptr;
    uint64_t prior_value = 0;
    void* producer_event = nullptr;
    uint64_t producer_value = 0;
    std::shared_ptr<AsyncSync> producer_sync;

    ~Allocation() noexcept {
        // Handles and descriptor owners prevent destruction in inherited
        // processes. Uncharged storage was never published: free it directly.
        foreign.reset();
        if (!charged) return;
        std::shared_ptr<ArenaSlab> dead_slab;
        std::shared_ptr<void> dead_owner;
        {
            std::lock_guard lock(state->mutex);
            --state->count;
            auto entry = state->allocations.find(key);
            if (entry != state->allocations.end() && entry->second.generation == generation)
                state->allocations.erase(entry);
            erase_live_metal(*state, metal, this);
            if (arena_slab) {
                if (--arena_slab->active_slices == 0) {
                    state->owned -= arena_slab->raw.capacity;
                    auto active_it = std::find(state->active_slabs.begin(), state->active_slabs.end(), arena_slab);
                    if (active_it != state->active_slabs.end()) state->active_slabs.erase(active_it);
                    arena_slab->in_active_list = false;
                    if (state->open && arena_slab->raw.capacity <= state->cache_limit) {
                        state->shrink_cache(state->cache_limit - arena_slab->raw.capacity);
                        try {
                            CacheEntry item{};
                            item.slab = arena_slab;
                            state->cache.push_back(std::move(item));
                            try {
                                state->idle_slabs.push_back(arena_slab);
                            } catch (...) {
                                state->cache.pop_back();
                                throw;
                            }
                            arena_slab->cache_it = std::prev(state->cache.end());
                            arena_slab->in_idle_cache = true;
                            state->cached += arena_slab->raw.capacity;
                            arena_slab.reset();
                            return;
                        } catch (...) {}
                    }
                }
                dead_slab = std::move(arena_slab);
                return;
            }
            if (owned) state->owned -= capacity;
            else state->borrowed -= capacity;
            // Owned Metal storage returns to an open context's cache when it fits.
            // Anything else is freed before its bytes leave the budget.
            if (owned && raw.metal && state->open && capacity <= state->cache_limit) {
                state->shrink_cache(state->cache_limit - capacity);
                try {
                    CacheEntry item{};
                    item.raw = std::move(raw);
                    state->cache.push_back(std::move(item));
                    try {
                        state->cache_index.emplace(capacity, std::prev(state->cache.end()));
                    } catch (...) {
                        raw = std::move(state->cache.back().raw);
                        state->cache.pop_back();
                        throw;
                    }
                    state->cached += capacity;
                    return;
                } catch (...) {} // no memory for bookkeeping: free instead
            }
            dead_owner = std::move(raw.owner);
        }
    }
};

struct ThreadDiagnostic {
    std::weak_ptr<State> state;
    pid_t pid = 0;
    char error[256]{};
};
thread_local ThreadDiagnostic tls_diagnostic;

void state_error(const std::shared_ptr<State>& state, const char* message) noexcept {
    if (!state || state->pid != getpid()) return;
    const char* text = message ? message : "";
    size_t length = 0;
    while (length + 1 < sizeof(state->error) && text[length]) ++length;
    try {
        tls_diagnostic.state = state;
        tls_diagnostic.pid = state->pid;
        std::memcpy(tls_diagnostic.error, text, length);
        tls_diagnostic.error[length] = '\0';
    } catch (...) {}
    if (length == 0 && !state->has_error.load(std::memory_order_acquire)) return;
    try {
        std::lock_guard lock(state->mutex);
        std::memcpy(state->error, text, length);
        state->error[length] = '\0';
        state->has_error.store(length != 0, std::memory_order_release);
    } catch (...) {}
}

uint64_t checked_add(uint64_t left, uint64_t right) {
    uint64_t result;
    if (ms_checked_add(left, right, &result) != MS_OK)
        throw Error(MS_EOVERFLOW, "buffer size arithmetic overflow");
    return result;
}
} // namespace

struct ms_context {
    std::atomic<uint64_t> refs{1};
    pid_t pid = getpid();
    std::shared_ptr<State> state;
    explicit ms_context(std::shared_ptr<State> value) : state(std::move(value)) {}
    ~ms_context() {
        // Without a handle nothing can reuse cached storage: free it, and let
        // surviving buffers free their storage directly.
        if (pid != getpid()) return;
        std::vector<CacheEntry> evicted;
        {
            std::lock_guard lock(state->mutex);
            state->open = false;
            evicted.reserve(state->cache.size());
            while (!state->cache.empty()) {
                const uint64_t cap = state->cache.front().capacity();
                evicted.push_back(std::move(state->cache.front()));
                state->cache.pop_front();
                state->cached -= cap;
            }
            state->cache_index.clear();
            state->idle_slabs.clear();
        }
    }
};
struct ms_buffer {
    std::atomic<uint64_t> refs{1};
    pid_t pid = getpid();
    std::shared_ptr<Allocation> allocation;
    uint64_t offset, size;
    ms_buffer(std::shared_ptr<Allocation> value, uint64_t start, uint64_t length)
        : allocation(std::move(value)), offset(start), size(length) {}
};

namespace {
void check_buffer(const ms_buffer* buffer) {
    if (!buffer) throw Error(MS_EINVAL, "null buffer");
    if (buffer->pid != getpid()) throw Error(MS_ESTATE, "inherited buffer is unusable after fork");
}
void check_buffer_context(const ms_buffer* buffer, ms_context* expected) {
    check_buffer(buffer);
    if (expected) {
        ms::detail::check_context(expected);
        if (buffer->allocation->state != expected->state)
            throw Error(MS_EINVAL, "buffer must first be admitted to operation context");
    }
}
// Zeroed storage, or for complete producers only zeroed padding. Returns the
// bytes zero-filled (test-only poisoning is not counted).
uint64_t initialize(uint8_t* host, uint64_t logical, uint64_t capacity, ms::detail::Fill fill, bool poison) {
    if (fill == ms::detail::Fill::zero || fill == ms::detail::Fill::device_zero) {
        std::memset(host, 0, static_cast<size_t>(capacity));
        return capacity;
    }
    if (poison && logical) std::memset(host, 0xa5, static_cast<size_t>(logical));
    std::memset(host + logical, 0, static_cast<size_t>(capacity - logical));
    return capacity - logical;
}

void wait_raw_in_flight(RawAllocation& raw, ms_context* waiter) {
    if (raw.in_flight_sync) {
        raw.in_flight_sync->wait(waiter);
        raw.in_flight_sync.reset();
        raw.in_flight_event = nullptr;
        raw.in_flight_value = 0;
    } else if (raw.in_flight_event && raw.in_flight_value) {
        const uint64_t ns = ms::detail::wait_metal_event(raw.in_flight_event, raw.in_flight_value);
        if (waiter && ns) ms::detail::record_host_wait(waiter, ns);
        raw.in_flight_event = nullptr;
        raw.in_flight_value = 0;
    }
}

void ensure_host_ready(Allocation& allocation, ms_context* waiter) {
    if (allocation.host_ready.load(std::memory_order_acquire)) return;
    std::lock_guard lock(allocation.sync_mutex);
    if (allocation.host_ready.load(std::memory_order_relaxed)) return;
    if (allocation.producer_sync) {
        allocation.producer_sync->wait(waiter);
        allocation.producer_sync.reset();
        allocation.producer_event = nullptr;
        allocation.producer_value = 0;
    } else if (allocation.producer_event && allocation.producer_value) {
        const uint64_t ns = ms::detail::wait_metal_event(allocation.producer_event, allocation.producer_value);
        if (waiter && ns) ms::detail::record_host_wait(waiter, ns);
        allocation.producer_event = nullptr;
        allocation.producer_value = 0;
    }
    if (allocation.prior_event && allocation.prior_value) {
        const uint64_t ns = ms::detail::wait_metal_event(allocation.prior_event, allocation.prior_value);
        if (waiter && ns) ms::detail::record_host_wait(waiter, ns);
        allocation.prior_event = nullptr;
        allocation.prior_value = 0;
    }
    if (allocation.pending_fill.active) {
        auto* base = static_cast<uint8_t*>(allocation.host) + allocation.pending_fill.offset;
        const auto fill = allocation.pending_fill.zero_logical ? ms::detail::Fill::zero : ms::detail::Fill::unfilled;
        const uint64_t zeroed = initialize(base, allocation.pending_fill.logical, allocation.pending_fill.capacity,
                                           fill, allocation.pending_fill.poison_logical);
        allocation.state->host_fill_bytes.fetch_add(zeroed, std::memory_order_relaxed);
        allocation.pending_fill.active = false;
    }
    allocation.host_ready.store(true, std::memory_order_release);
}

template<class Function> ms_status buffer_api(const ms_buffer* buffer, Function&& function) noexcept {
    try {
        check_buffer(buffer);
        function();
        state_error(buffer->allocation->state, "");
        return MS_OK;
    } catch (const Error& error) {
        if (buffer && buffer->pid == getpid()) state_error(buffer->allocation->state, error.what());
        return error.status;
    } catch (const std::bad_alloc&) {
        if (buffer && buffer->pid == getpid()) state_error(buffer->allocation->state, "native allocation failed");
        return MS_EOOM;
    } catch (...) {
        if (buffer && buffer->pid == getpid()) state_error(buffer->allocation->state, "native buffer operation failed");
        return MS_EBACKEND;
    }
}
struct ExportOwner {
    std::atomic<uint64_t> refs{1};
    pid_t pid = getpid();
    std::shared_ptr<Allocation> allocation;
    explicit ExportOwner(std::shared_ptr<Allocation> value) : allocation(std::move(value)) {}
};
void export_retain(void* pointer) noexcept {
    auto* owner = static_cast<ExportOwner*>(pointer);
    if (owner && owner->pid == getpid()) owner->refs.fetch_add(1, std::memory_order_relaxed);
}
void export_release(void* pointer) noexcept {
    auto* owner = static_cast<ExportOwner*>(pointer);
    if (owner && owner->pid == getpid() && owner->refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
        delete owner;
}

Identity source_identity(const ms_buffer_view& source) {
    if (source.retain == export_retain && source.release == export_release) {
        const auto* owner = static_cast<const ExportOwner*>(source.owner);
        if (owner->pid != getpid()) throw Error(MS_ESTATE, "inherited export is unusable after fork");
        const auto& allocation = owner->allocation;
        if (allocation->key != Key{source.kind, source.device_id, source.allocation_id} ||
            allocation->host != source.host_base || allocation->metal != source.metal_buffer ||
            allocation->capacity != source.capacity)
            throw Error(MS_EINVAL, "native export backing metadata changed");
        return allocation->identity;
    }
    return {source.owner, source.retain, source.release};
}
bool same_backing(const Allocation& allocation, const ms_buffer_view& source, Identity identity) {
    return allocation.host == source.host_base && allocation.metal == source.metal_buffer &&
           allocation.capacity == source.capacity && allocation.identity == identity;
}

std::shared_ptr<Allocation> admitted_source(ms_context* context, const ms_buffer_view& source) {
    const Identity identity = source_identity(source);
    const Key key{source.kind, source.device_id, source.allocation_id};
    const auto state = context->state;
    std::shared_ptr<Allocation> existing;
    {
        std::lock_guard lock(state->mutex);
        auto entry = state->allocations.find(key);
        if (entry != state->allocations.end()) existing = entry->second.allocation.lock();
    }
    if (existing) {
        if (!same_backing(*existing, source, identity))
            throw Error(MS_EINVAL, "allocation identity collides with different backing");
        if (source.kind == MS_BUFFER_METAL && !(source.flags & MS_BUFFER_PRODUCER_COMPLETE)) {
            std::lock_guard sync_lock(existing->sync_mutex);
            if (!ms::detail::metal_event_complete(source.completion_event, source.completion_value)) {
                existing->producer_event = source.completion_event;
                existing->producer_value = source.completion_value;
                if (source.retain == export_retain && source.release == export_release)
                    existing->producer_sync = static_cast<const ExportOwner*>(source.owner)->allocation->producer_sync;
                existing->host_ready.store(false, std::memory_order_release);
            }
        }
        return existing;
    }
    // Retain/release may reenter this context; neither runs under its mutex.
    auto candidate = std::make_shared<Allocation>();
    candidate->state = state;
    candidate->key = key;
    candidate->capacity = source.capacity;
    candidate->host = source.host_base;
    candidate->metal = source.metal_buffer;
    candidate->identity = identity;
    candidate->foreign = std::make_unique<ForeignHold>(source);
    if (source.kind == MS_BUFFER_METAL && !(source.flags & MS_BUFFER_PRODUCER_COMPLETE)) {
        if (!ms::detail::metal_event_complete(source.completion_event, source.completion_value)) {
            candidate->producer_event = source.completion_event;
            candidate->producer_value = source.completion_value;
            if (source.retain == export_retain && source.release == export_release)
                candidate->producer_sync = static_cast<const ExportOwner*>(source.owner)->allocation->producer_sync;
            candidate->host_ready.store(false, std::memory_order_release);
        }
    }
    std::shared_ptr<Allocation> result;
    {
        std::lock_guard lock(state->mutex);
        auto entry = state->allocations.find(key);
        if (entry != state->allocations.end()) result = entry->second.allocation.lock();
        if (result) {
            if (!same_backing(*result, source, identity))
                throw Error(MS_EINVAL, "allocation identity collides with different backing");
        } else {
            state->admit(source.capacity);
            state->allocations.insert_or_assign(key, Entry{candidate, candidate->generation});
            insert_live_metal(*state, candidate->metal, candidate.get());
            state->borrowed += source.capacity;
            ++state->count;
            candidate->charged = true;
            state->mark_peak();
            result = candidate;
        }
    }
    return result;
}
} // namespace

namespace ms::detail {
struct Reservation::Impl {
    std::shared_ptr<State> state;
    uint64_t remaining;
    pid_t pid = getpid();
    Impl(std::shared_ptr<State> value, uint64_t bytes) : state(std::move(value)), remaining(bytes) {}
};
namespace {
void dispose_reservation(Reservation::Impl* reservation) noexcept {
    if (!reservation || reservation->pid != getpid()) return;
    {
        std::lock_guard lock(reservation->state->mutex);
        reservation->state->pending -= reservation->remaining;
    }
    delete reservation;
}
}
Reservation::~Reservation() { dispose_reservation(impl_); }
Reservation& Reservation::operator=(Reservation&& other) noexcept {
    if (this != &other) {
        dispose_reservation(impl_);
        impl_ = std::exchange(other.impl_, nullptr);
    }
    return *this;
}
void check_context(ms_context* context) {
    if (!context) throw Error(MS_EINVAL, "null context");
    if (context->pid != getpid()) throw Error(MS_ESTATE, "inherited context is unusable after fork");
}
void set_error(ms_context* context, ms_status, const char* message) noexcept {
    if (context && context->pid == getpid()) state_error(context->state, message);
}
uint64_t padded_capacity(uint64_t logical, ms_context* context) {
    uint64_t alignment = 64;
    if (context) {
        check_context(context);
        if (context->state->kind == MS_BUFFER_METAL) {
            const long page_size = sysconf(_SC_PAGESIZE);
            if (page_size < 64 || (page_size & (page_size - 1)))
                throw Error(MS_EBACKEND, "unsupported host page size for Metal admission");
            alignment = static_cast<uint64_t>(page_size);
        }
    }
    const uint64_t size = checked_add(logical, 64);
    const uint64_t capacity = checked_add(size, alignment - 1) & ~(alignment - 1);
    if (capacity > std::numeric_limits<size_t>::max())
        throw Error(MS_EOVERFLOW, "buffer exceeds host address range");
    return capacity;
}
Reservation reserve(ms_context* context, uint64_t bytes) {
    check_context(context);
    auto reservation = std::make_unique<Reservation::Impl>(context->state, bytes);
    {
        std::lock_guard lock(context->state->mutex);
        context->state->admit(bytes);
        context->state->pending += bytes;
        context->state->mark_peak();
    }
    return Reservation(reservation.release());
}
namespace {
// Pending bytes that keep a reused buffer charged between leaving the cache and
// registration; released if allocation fails in between.
struct Held {
    std::shared_ptr<State> state;
    uint64_t bytes = 0;
    ~Held() {
        if (!bytes) return;
        std::lock_guard lock(state->mutex);
        state->pending -= bytes;
    }
};
RawAllocation native_allocation(State& state, uint64_t capacity) {
    if (state.kind == MS_BUFFER_CPU) {
        void* pointer = nullptr;
        if (posix_memalign(&pointer, 64, static_cast<size_t>(capacity)))
            throw Error(MS_EOOM, "aligned CPU allocation failed");
        RawAllocation raw;
        raw.owner = std::shared_ptr<void>(pointer, std::free);
        raw.host = pointer;
        raw.capacity = capacity;
        return raw;
    }
    try {
        return allocate_metal(capacity);
    } catch (const Error& error) {
        if (error.status != MS_EOOM) throw;
        // The device, not the budget, refused: free idle storage and retry once.
        std::vector<CacheEntry> evicted;
        {
            std::lock_guard lock(state.mutex);
            if (state.cache.empty()) throw;
            evicted.reserve(state.cache.size());
            while (!state.cache.empty()) {
                const uint64_t cap = state.cache.front().capacity();
                if (state.cache.front().slab) state.cache.front().slab->in_idle_cache = false;
                evicted.push_back(std::move(state.cache.front()));
                state.cache.pop_front();
                state.cached -= cap;
            }
            state.cache_index.clear();
            state.idle_slabs.clear();
        }
        evicted.clear();
        return allocate_metal(capacity);
    }
}

// Caller holds state.mutex. Foreign IDs come from producers, so skip any
// live conflicting IDs before publishing this allocation.
void register_native(State& state, const std::shared_ptr<Allocation>& allocation, uint64_t device_id) {
    for (;;) {
        const uint64_t id = next_id.fetch_add(1, std::memory_order_relaxed);
        if (!id) throw Error(MS_EOVERFLOW, "allocation ID space exhausted");
        allocation->key = {state.kind, device_id, id};
        auto entry = state.allocations.find(allocation->key);
        if (entry == state.allocations.end() || entry->second.allocation.expired()) break;
    }
    state.allocations.insert_or_assign(allocation->key, Entry{allocation, allocation->generation});
    insert_live_metal(state, allocation->metal, allocation.get());
}

void setup_allocation_fill(State& state, ms_context* context, Allocation& allocation, uint64_t offset,
                           uint64_t logical, uint64_t fill_capacity, Fill fill) {
    const bool defer = state.kind == MS_BUFFER_METAL &&
                       (fill == Fill::device_zero || fill == Fill::device_unfilled);
    if (defer) {
        if (!allocation.arena_slab && allocation.raw.in_flight_event && allocation.raw.in_flight_value &&
            !metal_event_complete(allocation.raw.in_flight_event, allocation.raw.in_flight_value)) {
            allocation.prior_event = allocation.raw.in_flight_event;
            allocation.prior_value = allocation.raw.in_flight_value;
        }
        allocation.pending_fill.active = true;
        allocation.pending_fill.zero_logical = (fill == Fill::device_zero);
        allocation.pending_fill.poison_logical = (fill == Fill::device_unfilled && state.poison);
        allocation.pending_fill.offset = offset;
        allocation.pending_fill.logical = logical;
        allocation.pending_fill.capacity = fill_capacity;
        allocation.host_ready.store(false, std::memory_order_release);
        return;
    }
    if (!allocation.arena_slab) wait_raw_in_flight(allocation.raw, context);
    if (allocation.arena_slab && fill == Fill::unfilled) {
        if (state.poison && logical)
            std::memset(static_cast<uint8_t*>(allocation.host) + offset, 0xa5, static_cast<size_t>(logical));
        return;
    }
    state.host_fill_bytes.fetch_add(
        initialize(static_cast<uint8_t*>(allocation.host) + offset, logical, fill_capacity, fill, state.poison),
        std::memory_order_relaxed);
}
} // namespace

BufferPtr allocate_reserved(ms_context* context, Reservation& reservation, uint64_t logical, Fill fill) {
    check_context(context);
    if (!reservation.impl_ || reservation.impl_->pid != getpid())
        throw Error(MS_ESTATE, "missing or inherited reservation");
    if (reservation.impl_->state != context->state)
        throw Error(MS_EINVAL, "reservation belongs to another context");
    const uint64_t capacity = padded_capacity(logical, context);
    if (capacity > reservation.impl_->remaining)
        throw Error(MS_EINVAL, "allocation exceeds reservation");
    auto& state = *context->state;
    Held held{context->state}; // declared first: storage is freed before its bytes are released
    auto allocation = std::make_shared<Allocation>();
    allocation->state = context->state;
    allocation->owned = true;
    allocation->owned_logical_bytes = logical;
    allocation->logical_capacity = capacity;
    allocation->capacity = capacity;
    if (state.kind == MS_BUFFER_METAL) {
        // Reuse the most recently cached buffer of this capacity, or for >= 1 MiB
        // allocations fall back to a bounded over-fit cached buffer up to 1.5x.
        std::lock_guard lock(state.mutex);
        auto range = state.cache_index.equal_range(capacity);
        if (range.first == range.second && capacity >= overfit_min_capacity) {
            const uint64_t max_overfit = capacity + (capacity >> 1);
            const auto lower = state.cache_index.lower_bound(capacity);
            if (lower != state.cache_index.end() && lower->first <= max_overfit)
                range = state.cache_index.equal_range(lower->first);
        }
        if (range.first != range.second) {
            const auto entry = std::prev(range.second);
            const uint64_t matched = entry->first;
            allocation->raw = std::move(entry->second->raw);
            allocation->capacity = matched;
            state.cache.erase(entry->second);
            state.cache_index.erase(entry);
            state.cached -= matched;
            state.pending += matched;
            held.bytes = matched;
        }
    }
    const bool reused = held.bytes != 0;
    if (!reused) allocation->raw = native_allocation(state, capacity);
    const uint64_t physical = allocation->capacity;
    // Checkout has exclusive storage and has released state.mutex. Any failure
    // frees the uncharged raw allocation before Held releases its pending charge.
    // Initialization below restores the same fill and padding contract as new storage.
    if (reused) {
        const bool defer = state.kind == MS_BUFFER_METAL &&
                           (fill == Fill::device_zero || fill == Fill::device_unfilled);
        if (!defer) wait_raw_in_flight(allocation->raw, context);
        prepare_cached_metal(allocation->raw);
    }
    if (!allocation->raw.owner || !allocation->raw.host || allocation->raw.capacity != physical ||
        (reinterpret_cast<uintptr_t>(allocation->raw.host) & 63u) ||
        (state.kind == MS_BUFFER_METAL && (!allocation->raw.metal || !allocation->raw.device_id)))
        throw Error(MS_EBACKEND, "storage backend returned invalid allocation");
    allocation->host = allocation->raw.host;
    allocation->metal = allocation->raw.metal;
    allocation->identity = {allocation.get(), export_retain, export_release};
    setup_allocation_fill(state, context, *allocation, 0, logical, physical, fill);
    BufferPtr buffer(new ms_buffer(allocation, 0, logical));
    {
        std::lock_guard lock(state.mutex);
        register_native(state, allocation, allocation->raw.device_id);
        // A reused buffer consumes the reservation without new storage.
        state.pending -= capacity + held.bytes;
        held.bytes = 0;
        state.owned += physical;
        ++state.count;
        reservation.impl_->remaining -= capacity;
        allocation->charged = true;
        state.observe_owned_storage();
    }
    (reused ? state.cached_reuses : state.native_allocations).fetch_add(1, std::memory_order_relaxed);
    return buffer;
}

BufferPtr allocate_scratch(ms_context* context, uint64_t logical, Fill fill) {
    check_context(context);
    auto& state = *context->state;
    const uint64_t padded = checked_add(logical, 64);
    const uint64_t slice_bytes = checked_add(padded, arena_slice_alignment - 1) & ~(arena_slice_alignment - 1);
    if (state.kind != MS_BUFFER_METAL || logical > arena_max_logical || slice_bytes > arena_slab_bytes) {
        auto reservation = reserve(context, padded_capacity(logical, context));
        return allocate_reserved(context, reservation, logical, fill);
    }

    auto allocation = std::make_shared<Allocation>();
    allocation->state = context->state;
    allocation->owned = true;
    allocation->owned_logical_bytes = logical;
    allocation->logical_capacity = slice_bytes;

    const bool defer_fill = (fill == Fill::device_zero || fill == Fill::device_unfilled);
    std::shared_ptr<ArenaSlab> chosen;
    uint64_t slice_offset = 0;
    bool reused_slab = false;
    void* slice_prior_event = nullptr;
    uint64_t slice_prior_value = 0;
    RawAllocation wrap_wait;

    {
        std::unique_lock lock(state.mutex);
        if (state.cache_limit < arena_slab_bytes) {
            lock.unlock();
            auto reservation = reserve(context, padded_capacity(logical, context));
            return allocate_reserved(context, reservation, logical, fill);
        }
        auto claim_slab_prior = [&](ArenaSlab& slab) {
            if (!slab.prior_event || !slab.prior_value) return;
            if (metal_event_complete(slab.prior_event, slab.prior_value)) {
                slab.prior_event = nullptr;
                slab.prior_value = 0;
            } else if (defer_fill) {
                slice_prior_event = slab.prior_event;
                slice_prior_value = slab.prior_value;
            } else {
                wrap_wait.in_flight_event = slab.prior_event;
                wrap_wait.in_flight_value = slab.prior_value;
                slab.prior_event = nullptr;
                slab.prior_value = 0;
            }
        };
        // 1. Suballocate from an already-active ArenaSlab if room remains.
        for (auto it = state.active_slabs.rbegin(); it != state.active_slabs.rend(); ++it) {
            auto& slab = *it;
            if (slab->bump + slice_bytes <= slab->raw.capacity) {
                chosen = slab;
                slice_offset = slab->bump;
                slab->bump += slice_bytes;
                ++slab->active_slices;
                claim_slab_prior(*slab);
                reused_slab = true;
                break;
            }
        }
        // 2. Otherwise check out the most recently cached idle ArenaSlab.
        if (!chosen && !state.idle_slabs.empty()) {
            chosen = state.idle_slabs.back();
            state.idle_slabs.pop_back();
            if (chosen->in_idle_cache) {
                state.cache.erase(chosen->cache_it);
                chosen->in_idle_cache = false;
                state.cached -= chosen->raw.capacity;
            }
            if (metal_event_complete(chosen->raw.in_flight_event, chosen->raw.in_flight_value)) {
                chosen->raw.in_flight_event = nullptr;
                chosen->raw.in_flight_value = 0;
                chosen->raw.in_flight_sync.reset();
                chosen->prior_event = nullptr;
                chosen->prior_value = 0;
                chosen->bump = 0;
            } else if (chosen->bump + slice_bytes <= chosen->raw.capacity &&
                       chosen->raw.capacity - chosen->bump >= (arena_slab_bytes / 2)) {
                claim_slab_prior(*chosen);
            } else if (defer_fill) {
                chosen->prior_event = chosen->raw.in_flight_event;
                chosen->prior_value = chosen->raw.in_flight_value;
                slice_prior_event = chosen->prior_event;
                slice_prior_value = chosen->prior_value;
                chosen->raw.in_flight_event = nullptr;
                chosen->raw.in_flight_value = 0;
                chosen->raw.in_flight_sync.reset();
                chosen->bump = 0;
            } else {
                wrap_wait.in_flight_event = chosen->raw.in_flight_event;
                wrap_wait.in_flight_value = chosen->raw.in_flight_value;
                wrap_wait.in_flight_sync = std::move(chosen->raw.in_flight_sync);
                chosen->raw.in_flight_event = nullptr;
                chosen->raw.in_flight_value = 0;
                chosen->prior_event = nullptr;
                chosen->prior_value = 0;
                chosen->bump = 0;
            }
            slice_offset = chosen->bump;
            chosen->bump += slice_bytes;
            chosen->active_slices = 1;
            chosen->in_active_list = true;
            state.active_slabs.push_back(chosen);
            state.owned += chosen->raw.capacity;
            state.mark_peak();
            reused_slab = true;
        }
        // 3. Otherwise allocate a new 256 KiB ArenaSlab if ample budget headroom exists.
        if (!chosen) {
            const uint64_t non_cached = state.total() - state.cached;
            if (state.limit - non_cached < arena_min_headroom_bytes) {
                lock.unlock();
                auto reservation = reserve(context, padded_capacity(logical, context));
                return allocate_reserved(context, reservation, logical, fill);
            }
            state.admit(arena_slab_bytes);
            state.pending += arena_slab_bytes;
            state.mark_peak();
            Held held{context->state, arena_slab_bytes};
            lock.unlock();

            RawAllocation raw = native_allocation(state, arena_slab_bytes);
            if (!raw.owner || !raw.host || raw.capacity != arena_slab_bytes ||
                (reinterpret_cast<uintptr_t>(raw.host) & 63u) || !raw.metal || !raw.device_id)
                throw Error(MS_EBACKEND, "storage backend returned invalid arena slab");
            auto fresh = std::make_shared<ArenaSlab>();
            fresh->raw = std::move(raw);
            fresh->bump = slice_bytes;
            fresh->active_slices = 1;
            fresh->in_active_list = true;

            lock.lock();
            state.pending -= held.bytes;
            held.bytes = 0;
            state.owned += arena_slab_bytes;
            state.active_slabs.push_back(fresh);
            state.observe_owned_storage();
            state.mark_peak();
            chosen = std::move(fresh);
            slice_offset = 0;
            reused_slab = false;
        }
    }

    wait_raw_in_flight(wrap_wait, context);

    allocation->arena_slab = chosen;
    allocation->arena_offset = slice_offset;
    allocation->arena_slice_bytes = slice_bytes;
    allocation->capacity = chosen->raw.capacity;
    allocation->host = chosen->raw.host;
    allocation->metal = chosen->raw.metal;
    allocation->identity = {allocation.get(), export_retain, export_release};
    if (slice_prior_event && slice_prior_value) {
        allocation->prior_event = slice_prior_event;
        allocation->prior_value = slice_prior_value;
    }
    setup_allocation_fill(state, context, *allocation, slice_offset, logical, slice_bytes, fill);
    BufferPtr buffer(new ms_buffer(allocation, slice_offset, logical));
    {
        std::lock_guard lock(state.mutex);
        register_native(state, allocation, chosen->raw.device_id);
        ++state.count;
        allocation->charged = true;
    }
    (reused_slab ? state.cached_reuses : state.native_allocations).fetch_add(1, std::memory_order_relaxed);
    return buffer;
}

bool is_scratch_slice(const ms_buffer* buffer) {
    check_buffer(buffer);
    return buffer->allocation->arena_slab != nullptr;
}

BufferPtr wrap_immutable(ms_context* context, const void* bytes, uint64_t length, const void* owner_base,
                         uint64_t owner_size, std::shared_ptr<void> producer) {
    if (storage_kind(context) != MS_BUFFER_METAL || !bytes || !length || !owner_base || !producer)
        throw Error(MS_ENOTSUP, "no-copy import requires Metal and an immutable buffer owner");
    const auto page_size = sysconf(_SC_PAGESIZE);
    if (page_size < 64 || (page_size & (page_size - 1)))
        throw Error(MS_EBACKEND, "unsupported host page size for Metal import");
    const auto mask = static_cast<uint64_t>(page_size - 1);
    const auto first = reinterpret_cast<uintptr_t>(bytes);
    const auto base = first & ~mask;
    const auto end = checked_add(checked_add(first, length), mask) & ~mask;
    const auto capacity = end - base;
    const auto witness = reinterpret_cast<uintptr_t>(owner_base);
    if (base < witness || end > checked_add(witness, owner_size))
        throw Error(MS_ENOTSUP, "page-rounded buffer lies outside its immutable owner");

    auto reservation = reserve(context, capacity);
    auto allocation = std::make_shared<Allocation>();
    allocation->state = context->state;
    allocation->producer = std::move(producer);
    allocation->capacity = capacity;
    allocation->raw = wrap_metal_pages(reinterpret_cast<const void*>(base), capacity, allocation->producer);
    allocation->host = allocation->raw.host;
    allocation->metal = allocation->raw.metal;
    allocation->identity = {allocation.get(), export_retain, export_release};
    BufferPtr buffer(new ms_buffer(allocation, first - base, length));
    auto& state = *context->state;
    {
        std::lock_guard lock(state.mutex);
        register_native(state, allocation, allocation->raw.device_id);
        state.pending -= capacity;
        reservation.impl_->remaining -= capacity;
        state.borrowed += capacity;
        ++state.count;
        allocation->charged = true;
    }
    return buffer;
}
ReadView read(const ms_buffer* buffer, ms_context* expected) {
    check_buffer_context(buffer, expected);
    ensure_host_ready(*buffer->allocation, expected);
    const auto* base = static_cast<const uint8_t*>(buffer->allocation->host);
    return {base ? base + buffer->offset : nullptr, buffer->size};
}
uint64_t buffer_size(const ms_buffer* buffer, ms_context* expected) {
    check_buffer_context(buffer, expected);
    return buffer->size;
}
uint64_t owned_capacity(const ms_buffer* buffer) {
    check_buffer(buffer);
    return buffer->allocation->owned ? buffer->allocation->logical_capacity : 0;
}
bool whole_owned_buffer(const ms_buffer* buffer, uint64_t logical_bytes) {
    check_buffer(buffer);
    return buffer->allocation->owned && !buffer->allocation->arena_slab &&
           buffer->offset == 0 && buffer->size == logical_bytes &&
           buffer->allocation->owned_logical_bytes == logical_bytes;
}
DeviceView device_view(const ms_buffer* buffer, ms_context* expected) {
    check_buffer_context(buffer, expected);
    if (!buffer->allocation->metal) throw Error(MS_ENOTSUP, "buffer has no Metal resource");
    return {buffer->allocation->metal, buffer->offset, buffer->size};
}
uint32_t storage_kind(ms_context* context) {
    check_context(context);
    return context->state->kind;
}
uint64_t available_headroom_bytes(ms_context* context) noexcept {
    if (!context || context->pid != getpid() || !context->state) return 0;
    auto& state = *context->state;
    std::lock_guard lock(state.mutex);
    const uint64_t committed = state.total() - state.cached;
    return state.limit > committed ? state.limit - committed : 0;
}
BufferResidency buffer_residency(const ms_buffer* buffer) noexcept {
    if (!buffer || buffer->pid != getpid() || !buffer->allocation) return {};
    const auto& alloc = *buffer->allocation;
    BufferResidency res;
    res.size = buffer->size;
    res.metal_owned = alloc.owned && alloc.metal != nullptr;
    if (!alloc.host_ready.load(std::memory_order_acquire)) {
        res.async_in_flight = alloc.producer_sync != nullptr || alloc.producer_event != nullptr;
    }
    return res;
}

PreparedGpuBatch prepare_gpu_batch(ms_context* context, const std::vector<const ms_buffer*>& bound_buffers,
                                   const std::vector<void*>& indirect_metal_buffers) {
    check_context(context);
    PreparedGpuBatch result;
    if (bound_buffers.empty() && indirect_metal_buffers.empty()) return result;
    auto& state = *context->state;
    std::vector<std::shared_ptr<Allocation>> matched;
    matched.reserve(bound_buffers.size() + indirect_metal_buffers.size());
    for (const ms_buffer* buf : bound_buffers) {
        if (buf && buf->allocation) matched.push_back(buf->allocation);
    }
    if (!indirect_metal_buffers.empty()) {
        std::lock_guard lock(state.mutex);
        for (void* metal : indirect_metal_buffers) {
            const auto range = state.live_metal.equal_range(metal);
            for (auto it = range.first; it != range.second; ++it) {
                if (it->second->arena_slab) continue;
                auto entry = state.allocations.find(it->second->key);
                if (entry != state.allocations.end()) {
                    if (auto locked = entry->second.allocation.lock()) matched.push_back(std::move(locked));
                }
            }
        }
    }
    std::sort(matched.begin(), matched.end());
    matched.erase(std::unique(matched.begin(), matched.end()), matched.end());
    result.touched.reserve(matched.size());
    for (auto& alloc : matched) {
        std::lock_guard sync_lock(alloc->sync_mutex);
        if (alloc->producer_event && alloc->producer_value &&
            !metal_event_complete(alloc->producer_event, alloc->producer_value)) {
            result.waits.push_back({alloc->producer_event, alloc->producer_value});
        }
        if (alloc->prior_event && alloc->prior_value) {
            if (!metal_event_complete(alloc->prior_event, alloc->prior_value))
                result.waits.push_back({alloc->prior_event, alloc->prior_value});
            alloc->prior_event = nullptr;
            alloc->prior_value = 0;
        }
        if (alloc->pending_fill.active) {
            const auto& pf = alloc->pending_fill;
            if (pf.zero_logical) {
                if (pf.capacity) result.fills.push_back({alloc->metal, pf.offset, pf.capacity, 0x00});
            } else {
                if (pf.poison_logical && pf.logical)
                    result.fills.push_back({alloc->metal, pf.offset, pf.logical, 0xa5});
                if (pf.capacity > pf.logical)
                    result.fills.push_back({alloc->metal, pf.offset + pf.logical, pf.capacity - pf.logical, 0x00});
            }
            alloc->pending_fill.active = false;
            result.claimed.push_back(alloc);
        }
        result.touched.push_back(alloc);
    }
    if (result.waits.size() > 1) {
        std::sort(result.waits.begin(), result.waits.end(), [](const EventWait& a, const EventWait& b) {
            return a.event != b.event ? a.event < b.event : a.value < b.value;
        });
        size_t write = 0;
        for (size_t i = 0; i < result.waits.size(); ++i) {
            if (write > 0 && result.waits[write - 1].event == result.waits[i].event)
                result.waits[write - 1].value = std::max(result.waits[write - 1].value, result.waits[i].value);
            else
                result.waits[write++] = result.waits[i];
        }
        result.waits.resize(write);
    }
    return result;
}

void complete_sync_gpu_batch(ms_context* context, const PreparedGpuBatch& prepared) noexcept {
    for (const auto& opaque : prepared.claimed) {
        auto& alloc = *static_cast<Allocation*>(opaque.get());
        std::lock_guard sync_lock(alloc.sync_mutex);
        if (!alloc.pending_fill.active &&
            (!alloc.producer_event || metal_event_complete(alloc.producer_event, alloc.producer_value))) {
            alloc.producer_event = nullptr;
            alloc.producer_value = 0;
            alloc.producer_sync.reset();
            alloc.host_ready.store(true, std::memory_order_release);
        }
    }
    if (context && context->state && !prepared.touched.empty()) {
        std::vector<std::shared_ptr<AsyncSync>> cleared_syncs;
        {
            std::lock_guard lock(context->state->mutex);
            for (const auto& opaque : prepared.touched) {
                auto& alloc = *static_cast<Allocation*>(opaque.get());
                if (alloc.arena_slab) {
                    if (metal_event_complete(alloc.arena_slab->raw.in_flight_event,
                                             alloc.arena_slab->raw.in_flight_value)) {
                        alloc.arena_slab->raw.in_flight_event = nullptr;
                        alloc.arena_slab->raw.in_flight_value = 0;
                        if (alloc.arena_slab->raw.in_flight_sync)
                            cleared_syncs.push_back(std::move(alloc.arena_slab->raw.in_flight_sync));
                    }
                } else if (metal_event_complete(alloc.raw.in_flight_event, alloc.raw.in_flight_value)) {
                    alloc.raw.in_flight_event = nullptr;
                    alloc.raw.in_flight_value = 0;
                    if (alloc.raw.in_flight_sync)
                        cleared_syncs.push_back(std::move(alloc.raw.in_flight_sync));
                }
            }
        }
    }
}

std::shared_ptr<void> context_state_handle(ms_context* context) {
    check_context(context);
    return context->state;
}

void attach_async_gpu_batch(ms_context* context, const PreparedGpuBatch& prepared, const ms_buffer* const* outputs,
                            size_t output_count, void* event, uint64_t ticket,
                            const std::shared_ptr<AsyncSync>& sync) {
    if (context && context->state && !prepared.touched.empty()) {
        std::vector<std::shared_ptr<AsyncSync>> replaced_syncs;
        {
            std::lock_guard lock(context->state->mutex);
            for (const auto& opaque : prepared.touched) {
                auto& alloc = *static_cast<Allocation*>(opaque.get());
                if (alloc.arena_slab) {
                    if (!alloc.arena_slab->raw.in_flight_event || ticket >= alloc.arena_slab->raw.in_flight_value) {
                        if (alloc.arena_slab->raw.in_flight_sync)
                            replaced_syncs.push_back(std::move(alloc.arena_slab->raw.in_flight_sync));
                        alloc.arena_slab->raw.in_flight_event = event;
                        alloc.arena_slab->raw.in_flight_value = ticket;
                        alloc.arena_slab->raw.in_flight_sync = sync;
                    }
                } else if (alloc.owned) {
                    if (!alloc.raw.in_flight_event || ticket >= alloc.raw.in_flight_value) {
                        if (alloc.raw.in_flight_sync)
                            replaced_syncs.push_back(std::move(alloc.raw.in_flight_sync));
                        alloc.raw.in_flight_event = event;
                        alloc.raw.in_flight_value = ticket;
                        alloc.raw.in_flight_sync = sync;
                    }
                }
            }
        }
    }
    auto mark_producer = [&](Allocation& alloc) {
        std::lock_guard sync_lock(alloc.sync_mutex);
        if (!alloc.producer_event || ticket >= alloc.producer_value) {
            alloc.producer_event = event;
            alloc.producer_value = ticket;
            alloc.producer_sync = sync;
            alloc.host_ready.store(false, std::memory_order_release);
        }
    };
    for (const auto& opaque : prepared.claimed) mark_producer(*static_cast<Allocation*>(opaque.get()));
    for (size_t i = 0; i < output_count; ++i) {
        if (outputs[i]) mark_producer(*outputs[i]->allocation);
    }
}

void register_async_gpu_submission(ms_context* context, const std::shared_ptr<AsyncSync>& sync) {
    auto& state = *context->state;
    std::lock_guard lock(state.async_mutex);
    state.async_submissions.erase(
        std::remove_if(state.async_submissions.begin(), state.async_submissions.end(),
                       [](const std::shared_ptr<AsyncSync>& item) { return !item || item->completed(); }),
        state.async_submissions.end());
    state.async_submissions.push_back(sync);
}

void retire_async_gpu(const std::shared_ptr<void>& state_handle, ms_context* waiter,
                      uint64_t gpu_ns, uint64_t wait_ns) noexcept {
    if (state_handle) {
        auto& origin = *static_cast<State*>(state_handle.get());
        if (origin.pid == getpid()) {
            origin.gpu_ns.fetch_add(gpu_ns, std::memory_order_relaxed);
            if (!waiter || waiter->state.get() == &origin)
                origin.host_wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
        }
    }
    if (waiter && waiter->pid == getpid() && (!state_handle || waiter->state.get() != state_handle.get()))
        waiter->state->host_wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
}

void flush_async_gpu(ms_context* context) {
    check_context(context);
    std::vector<std::shared_ptr<AsyncSync>> pending;
    {
        std::lock_guard lock(context->state->async_mutex);
        pending.swap(context->state->async_submissions);
    }
    for (auto& item : pending) {
        if (item) item->wait(context);
    }
}

void record_host_wait(ms_context* context, uint64_t nanoseconds) noexcept {
    if (!context || context->pid != getpid()) return;
    context->state->host_wait_ns.fetch_add(nanoseconds, std::memory_order_relaxed);
}
namespace {
thread_local const ms_context* active_gpu_op_context = nullptr;
} // namespace

void record_gpu(ms_context* context, uint64_t operations, uint64_t command_buffers, uint64_t dispatches,
                uint64_t gpu_ns) noexcept {
    if (!context || context->pid != getpid()) return;
    if (active_gpu_op_context == context) operations = 0;
    auto& state = *context->state;
    state.gpu_operations.fetch_add(operations, std::memory_order_relaxed);
    state.gpu_command_buffers.fetch_add(command_buffers, std::memory_order_relaxed);
    state.gpu_dispatches.fetch_add(dispatches, std::memory_order_relaxed);
    state.gpu_ns.fetch_add(gpu_ns, std::memory_order_relaxed);
}
GpuOperationScope::GpuOperationScope(ms_context* context, bool active) noexcept {
    if (active && context && active_gpu_op_context != context) {
        context_ = context;
        outermost_ = true;
        active_gpu_op_context = context;
    }
}
GpuOperationScope::~GpuOperationScope() noexcept {
    if (outermost_) active_gpu_op_context = nullptr;
}
void GpuOperationScope::complete() noexcept {
    if (outermost_) {
        active_gpu_op_context = nullptr;
        outermost_ = false;
        record_gpu(context_, 1, 0, 0, 0);
    }
}
uint8_t* mutable_data(ms_buffer* buffer) {
    check_buffer(buffer);
    if (!buffer->allocation->owned) throw Error(MS_ESTATE, "borrowed buffers are immutable");
    ensure_host_ready(*buffer->allocation, nullptr);
    return static_cast<uint8_t*>(buffer->allocation->host) + buffer->offset;
}
} // namespace ms::detail

extern "C" {
ms_status ms_context_create(const ms_context_options* options, ms_context** out) {
    try {
        if (!options || !out || options->struct_size < sizeof(*options) ||
            options->abi_version != MS_RUNTIME_ABI_VERSION || (options->flags & ~MS_CONTEXT_POISON_UNWRITTEN) ||
            options->reserved[0] || options->reserved[1] ||
            (options->storage_kind != MS_BUFFER_CPU && options->storage_kind != MS_BUFFER_METAL) ||
            !options->memory_limit_bytes || options->external_reservation_bytes > options->memory_limit_bytes)
            return MS_EINVAL;
        auto state = std::make_shared<State>(*options);
        auto context = std::make_unique<ms_context>(std::move(state));
        *out = context.release();
        return MS_OK;
    } catch (const std::bad_alloc&) { return MS_EOOM; }
    catch (...) { return MS_EBACKEND; }
}
void ms_context_retain(ms_context* context) {
    if (context && context->pid == getpid()) context->refs.fetch_add(1, std::memory_order_relaxed);
}
void ms_context_release(ms_context* context) {
    if (context && context->pid == getpid() && context->refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
        delete context;
}
ms_status ms_context_memory_info(ms_context* context, ms_memory_info* out) {
    return ms::detail::api(context, [&] {
        if (!out || out->struct_size < sizeof(*out) || out->abi_version != MS_RUNTIME_ABI_VERSION)
            throw Error(MS_EINVAL, "memory info requires initialized size and version");
        ms_memory_info result{};
        result.struct_size = sizeof(result);
        result.abi_version = MS_RUNTIME_ABI_VERSION;
        {
            std::lock_guard lock(context->state->mutex);
            result.limit_bytes = context->state->limit;
            result.owned_bytes = context->state->owned;
            result.borrowed_bytes = context->state->borrowed;
            result.reserved_bytes = context->state->pending;
            result.cached_bytes = context->state->cached;
            result.external_bytes = context->state->external;
            result.total_bytes = context->state->total();
            result.peak_bytes = context->state->peak;
            result.allocation_count = context->state->count;
        }
        *out = result;
    });
}
ms_status ms_context_execution_info(ms_context* context, ms_execution_info* out) {
    return ms::detail::api(context, [&] {
        if (!out || out->struct_size < sizeof(*out) || out->abi_version != MS_RUNTIME_ABI_VERSION)
            throw Error(MS_EINVAL, "execution info requires initialized size and version");
        ms::detail::flush_async_gpu(context);
        const auto& state = *context->state;
        ms_execution_info result{};
        result.struct_size = sizeof(result);
        result.abi_version = MS_RUNTIME_ABI_VERSION;
        result.gpu_operations = state.gpu_operations.load(std::memory_order_relaxed);
        result.gpu_command_buffers = state.gpu_command_buffers.load(std::memory_order_relaxed);
        result.gpu_dispatches = state.gpu_dispatches.load(std::memory_order_relaxed);
        result.gpu_time_ns = state.gpu_ns.load(std::memory_order_relaxed);
        result.native_allocations = state.native_allocations.load(std::memory_order_relaxed);
        result.cached_reuses = state.cached_reuses.load(std::memory_order_relaxed);
        result.host_wait_ns = state.host_wait_ns.load(std::memory_order_relaxed);
        result.host_fill_bytes = state.host_fill_bytes.load(std::memory_order_relaxed);
        *out = result;
    });
}
ms_status ms_context_cache_limit(ms_context* context, uint64_t* out) {
    return ms::detail::api(context, [&] {
        if (!out) throw Error(MS_EINVAL, "cache limit output pointer is null");
        std::lock_guard lock(context->state->mutex);
        *out = context->state->cache_limit;
    });
}
ms_status ms_context_set_cache_limit(ms_context* context, uint64_t bytes) {
    return ms::detail::api(context, [&] {
        auto& state = *context->state;
        if (bytes && state.kind != MS_BUFFER_METAL) throw Error(MS_ENOTSUP, "only Metal storage caches buffers");
        std::vector<CacheEntry> evicted;
        {
            std::lock_guard lock(state.mutex);
            state.explicit_cache_limit = true;
            state.cache_limit = bytes;
            while (state.cached > bytes && !state.cache.empty()) {
                const uint64_t cap = state.cache.front().capacity();
                if (state.cache.front().slab) {
                    state.cache.front().slab->in_idle_cache = false;
                    auto it = std::find(state.idle_slabs.begin(), state.idle_slabs.end(), state.cache.front().slab);
                    if (it != state.idle_slabs.end()) state.idle_slabs.erase(it);
                } else {
                    const auto range = state.cache_index.equal_range(cap);
                    for (auto entry = range.first; entry != range.second; ++entry)
                        if (entry->second == state.cache.begin()) {
                            state.cache_index.erase(entry);
                            break;
                        }
                }
                evicted.push_back(std::move(state.cache.front()));
                state.cache.pop_front();
                state.cached -= cap;
            }
        }
    });
}
ms_status ms_context_trim(ms_context* context) {
    return ms::detail::api(context, [&] {
        std::vector<CacheEntry> evicted;
        {
            std::lock_guard lock(context->state->mutex);
            evicted.reserve(context->state->cache.size());
            while (!context->state->cache.empty()) {
                const uint64_t cap = context->state->cache.front().capacity();
                if (context->state->cache.front().slab) context->state->cache.front().slab->in_idle_cache = false;
                evicted.push_back(std::move(context->state->cache.front()));
                context->state->cache.pop_front();
                context->state->cached -= cap;
            }
            context->state->cache_index.clear();
            context->state->idle_slabs.clear();
        }
    });
}
ms_status ms_context_error_copy(ms_context* context, char* destination, uint64_t capacity) {
    try {
        ms::detail::check_context(context);
        if (!destination) return MS_EINVAL;
        if (tls_diagnostic.pid == getpid() && tls_diagnostic.state.lock() == context->state) {
            const size_t needed = std::strlen(tls_diagnostic.error) + 1;
            if (capacity < needed) return MS_EINVAL;
            std::memcpy(destination, tls_diagnostic.error, needed);
            return MS_OK;
        }
        std::lock_guard lock(context->state->mutex);
        const size_t needed = std::strlen(context->state->error) + 1;
        if (capacity < needed) return MS_EINVAL;
        std::memcpy(destination, context->state->error, needed);
        return MS_OK;
    } catch (const Error& error) { return error.status; }
    catch (...) { return MS_EBACKEND; }
}
const char* ms_status_message(ms_status status) {
    switch (status) {
        case MS_OK: return "success";
        case MS_EINVAL: return "invalid argument";
        case MS_ENOTSUP: return "unsupported operation";
        case MS_EOVERFLOW: return "integer overflow";
        case MS_EOOM: return "memory admission or allocation failed";
        case MS_ETIMEOUT: return "operation timed out";
        case MS_EBACKEND: return "backend failure";
        case MS_ESTATE: return "invalid process or object state";
        default: return "unknown status";
    }
}
ms_status ms_buffer_zeros(ms_context* context, uint64_t bytes, ms_buffer** out) {
    return ms::detail::api(context, [&] {
        if (!out) throw Error(MS_EINVAL, "null output");
        auto reservation = ms::detail::reserve(context, ms::detail::padded_capacity(bytes, context));
        auto buffer = ms::detail::allocate_reserved(context, reservation, bytes);
        *out = buffer.release();
    });
}
ms_status ms_buffer_copy_from_host(ms_context* context, const void* source, uint64_t bytes, ms_buffer** out) {
    return ms::detail::api(context, [&] {
        if (!out || (!source && bytes)) throw Error(MS_EINVAL, "null source or output");
        auto reservation = ms::detail::reserve(context, ms::detail::padded_capacity(bytes, context));
        auto buffer = ms::detail::allocate_reserved(context, reservation, bytes, ms::detail::Fill::unfilled);
        if (bytes) std::memcpy(ms::detail::mutable_data(buffer.get()), source, static_cast<size_t>(bytes));
        *out = buffer.release();
    });
}
void ms_buffer_retain(ms_buffer* buffer) {
    if (buffer && buffer->pid == getpid()) buffer->refs.fetch_add(1, std::memory_order_relaxed);
}
void ms_buffer_release(ms_buffer* buffer) {
    if (buffer && buffer->pid == getpid() && buffer->refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
        delete buffer;
}
ms_status ms_buffer_data(const ms_buffer* buffer, const void** data, uint64_t* bytes) {
    return buffer_api(buffer, [&] {
        if (!data || !bytes) throw Error(MS_EINVAL, "null data output");
        const auto result = ms::detail::read(buffer);
        *data = result.data;
        *bytes = result.size;
    });
}
ms_status ms_buffer_slice(const ms_buffer* buffer, uint64_t offset, uint64_t length, ms_buffer** out) {
    return buffer_api(buffer, [&] {
        if (!out) throw Error(MS_EINVAL, "null slice output");
        if (checked_add(offset, length) > buffer->size) throw Error(MS_EINVAL, "slice exceeds logical bounds");
        auto result = std::make_unique<ms_buffer>(buffer->allocation, checked_add(buffer->offset, offset), length);
        *out = result.release();
    });
}
ms_status ms_buffer_import(ms_context* context, const ms_buffer_view* source, ms_copy_policy policy, ms_import_result* out) {
    return ms::detail::api(context, [&] {
        if (!source || !out || (policy != MS_COPY_AUTO && policy != MS_COPY_NEVER && policy != MS_COPY_ALWAYS))
            throw Error(MS_EINVAL, "invalid import arguments");
        const ms_status valid = ms_buffer_validate(source, 1);
        if (valid != MS_OK) throw Error(valid, "invalid import descriptor");
        const ms_buffer_view descriptor = *source;
        if (descriptor.kind != MS_BUFFER_CPU && descriptor.kind != MS_BUFFER_METAL)
            throw Error(MS_ENOTSUP, "unsupported import buffer kind");
        if (descriptor.kind == MS_BUFFER_METAL)
            ms::detail::verify_metal_import(descriptor);
        const bool eligible = (descriptor.flags & MS_BUFFER_IMMUTABLE) &&
                              descriptor.kind == context->state->kind;
        if (policy == MS_COPY_NEVER && !eligible)
            throw Error(MS_ENOTSUP, "never-copy requires immutable source matching destination storage");
        if ((!eligible || policy == MS_COPY_ALWAYS) && !(descriptor.flags & MS_BUFFER_HOST_READABLE))
            throw Error(MS_ENOTSUP, "snapshot import requires host-readable source");
        auto input = admitted_source(context, descriptor);
        ms_import_result result{};
        if (eligible && policy != MS_COPY_ALWAYS) {
            auto buffer = std::make_unique<ms_buffer>(input, descriptor.byte_offset, descriptor.byte_length);
            result.buffer = buffer.release();
            result.borrowed_bytes = descriptor.capacity;
        } else {
            ensure_host_ready(*input, context);
            auto reservation = ms::detail::reserve(context, ms::detail::padded_capacity(descriptor.byte_length, context));
            auto buffer = ms::detail::allocate_reserved(context, reservation, descriptor.byte_length,
                                                        ms::detail::Fill::unfilled);
            if (descriptor.byte_length) {
                const auto* data = static_cast<const uint8_t*>(input->host) + descriptor.byte_offset;
                std::memcpy(ms::detail::mutable_data(buffer.get()), data, static_cast<size_t>(descriptor.byte_length));
            }
            result.buffer = buffer.release();
            result.copied_bytes = descriptor.byte_length;
        }
        *out = result;
    });
}
ms_status ms_buffer_export(const ms_buffer* buffer, ms_buffer_view* out) {
    return buffer_api(buffer, [&] {
        if (!out) throw Error(MS_EINVAL, "null export output");
        const auto& allocation = buffer->allocation;
        void* completion_event = nullptr;
        uint64_t completion_value = 0;
        bool producer_complete = true;
        {
            std::unique_lock sync_lock(allocation->sync_mutex);
            if (allocation->pending_fill.active) {
                // No GPU batch has claimed this deferred fill; materialize it on CPU before export.
                sync_lock.unlock();
                ensure_host_ready(*allocation, nullptr);
            } else if (allocation->producer_event && allocation->producer_value) {
                if (ms::detail::metal_event_complete(allocation->producer_event, allocation->producer_value)) {
                    allocation->producer_event = nullptr;
                    allocation->producer_value = 0;
                    allocation->producer_sync.reset();
                    allocation->host_ready.store(true, std::memory_order_release);
                } else {
                    producer_complete = false;
                    completion_event = allocation->producer_event;
                    completion_value = allocation->producer_value;
                }
            }
        }
        auto owner = std::make_unique<ExportOwner>(allocation);
        ms_buffer_view result{};
        result.struct_size = sizeof(result);
        result.abi_version = MS_BUFFER_ABI_VERSION;
        result.kind = allocation->key.kind;
        result.flags = MS_BUFFER_HOST_READABLE | MS_BUFFER_IMMUTABLE |
                       (producer_complete ? MS_BUFFER_PRODUCER_COMPLETE : 0u);
        result.allocation_id = allocation->key.id;
        result.device_id = allocation->key.device;
        result.host_base = allocation->host;
        result.metal_buffer = allocation->metal;
        result.byte_offset = buffer->offset;
        result.byte_length = buffer->size;
        result.capacity = allocation->capacity;
        result.owner = owner.get();
        result.retain = export_retain;
        result.release = export_release;
        result.completion_event = completion_event;
        result.completion_value = completion_value;
        *out = result;
        owner.release();
    });
}
} // extern C
