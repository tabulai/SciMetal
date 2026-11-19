#pragma once
// Metal compute runtime: the embedded kernel library, a pipeline cache and
// synchronous command batches. One queue per process; serial compute encoders
// order dispatches and make earlier writes visible to later ones (all buffers
// are hazard-tracked). The CPU reads results only after commit_and_wait().
#include "metal_device.hpp"
#include "ms/detail/storage.hpp"

#include <initializer_list>
#include <string>
#include <type_traits>
#include <vector>

namespace ms::detail::gpu {
// Limits every dispatch keeps (docs/s1/kernel-conventions.md): the buffer
// argument table has 31 entries and parameters take index 0; setBytes takes at
// most 4 KiB. (Kernels also keep static threadgroup memory within 16 KiB, half
// the device limit, because shader validation doubles static allocations: under
// MTL_SHADER_VALIDATION a pipeline reports twice its plain size, so the kernel
// inventory test checks that convention in a plain run rather than this runtime.)
inline constexpr size_t max_bound_buffers = 30;
inline constexpr size_t max_params_bytes = 4096;

// A kernel buffer argument: input views bind at offset 0 (element and bit
// offsets travel in the kernel parameters); suballocated 256B-aligned scratch
// slices bind at their slice byte offset.
struct Binding {
    MTL::Buffer* buffer = nullptr;
    uint64_t offset = 0;
    const ms_buffer* handle = nullptr;
};
Binding bind(const ms_buffer*, ms_context*);

// Loads the compute runtime on first use: MS_ESTATE after fork, MS_EBACKEND
// when the embedded kernel library cannot load.
void require_runtime();

// For chunk tables: the GPU address of a buffer's first byte, and the
// resource a batch must make resident to read through that address.
uint64_t address(const ms_buffer*, ms_context*);
MTL::Buffer* resource(const ms_buffer*, ms_context*);

// The dispatch geometry: `groups` threadgroups of `threads` threads along each
// axis. Kernels written to the 1-D convention use Grid::linear, which keeps
// metal-arrow's MS_GPU_THREADS-thread threadgroups.
struct Grid {
    uint64_t groups[3] = {0, 1, 1};
    uint32_t threads[3] = {1, 1, 1};
    static Grid linear(uint64_t threadgroups);
    uint64_t total_groups() const { return groups[0] * groups[1] * groups[2]; }
    uint64_t threads_per_group() const { return uint64_t{threads[0]} * threads[1] * threads[2]; }
};

// A function-constant specialization of a kernel. `key` must identify the
// constant values uniquely: pipelines are cached by kernel name and key.
struct Constant {
    enum Type : uint32_t { Bool, Int, UInt, Float };
    uint32_t index = 0;
    Type type = Int;
    union { bool b; int32_t i; uint32_t u; float f; } value = {};
    static Constant of(uint32_t index, bool v) { Constant c; c.index = index; c.type = Bool; c.value.b = v; return c; }
    static Constant of(uint32_t index, int32_t v) { Constant c; c.index = index; c.type = Int; c.value.i = v; return c; }
    static Constant of(uint32_t index, uint32_t v) { Constant c; c.index = index; c.type = UInt; c.value.u = v; return c; }
    static Constant of(uint32_t index, float v) { Constant c; c.index = index; c.type = Float; c.value.f = v; return c; }
};
struct Specialization {
    std::string key;
    std::vector<Constant> constants;
};
// Arguments bound at explicit indices, for kernels that do not follow the
// "parameters at 0, buffers from 1" convention (the vendored MLX kernels).
struct IndexedBuffer {
    uint32_t index = 0;
    Binding binding;
};
struct IndexedBytes {
    uint32_t index = 0;
    const void* data = nullptr;
    size_t size = 0;
};
// Builds (or finds) the pipeline ahead of time, so its one-off compile cost is
// not paid inside a timed or latency-sensitive call.
void warm_pipeline(const char* kernel, const Specialization* specialization = nullptr);
// Resolves and caches a pipeline state pointer (valid for process lifetime).
MTL::ComputePipelineState* resolve_pipeline(const char* kernel, const Specialization* specialization = nullptr);
// The most threads one threadgroup of this kernel's pipeline may hold. It is a
// property of the compiled pipeline, not of the device: shader validation, for
// one, lowers it. Builds the pipeline if needed.
uint32_t max_threads_per_group(const char* kernel, const Specialization* specialization = nullptr);

class Batch {
public:
    explicit Batch(ms_context*);
    ~Batch();
    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;
    // Dispatches `grid`; buffers bind at indices 1, 2, ... after the parameters
    // at 0. More than max_bound_buffers buffers, parameters that are not a
    // multiple of 8 bytes or exceed max_params_bytes, or a threadgroup larger
    // than the pipeline allows are MS_EBACKEND before anything is encoded.
    void dispatch(MTL::ComputePipelineState* state, const void* params, size_t bytes, const Binding* buffers,
                  size_t count, const Grid& grid);
    void dispatch(const char* kernel, const void* params, size_t bytes, const Binding* buffers, size_t count,
                  const Grid& grid);
    // Dispatches with every argument at an explicit index: buffers (with their
    // byte offsets) and small by-value blobs (each at most max_params_bytes),
    // optionally on a function-constant specialization of the kernel. Indices
    // must be distinct and at most max_bound_buffers.
    void dispatch_indexed(MTL::ComputePipelineState* state,
                          const IndexedBuffer* buffers, size_t buffer_count,
                          const IndexedBytes* bytes, size_t bytes_count, const Grid& grid);
    void dispatch_indexed(const char* kernel, const Specialization* specialization,
                          const IndexedBuffer* buffers, size_t buffer_count,
                          const IndexedBytes* bytes, size_t bytes_count, const Grid& grid);
    void dispatch_indexed(const char* kernel, const Specialization* specialization,
                          std::initializer_list<IndexedBuffer> buffers, std::initializer_list<IndexedBytes> bytes,
                          const Grid& grid) {
        dispatch_indexed(kernel, specialization, buffers.begin(), buffers.size(), bytes.begin(), bytes.size(), grid);
    }
    // The 1-D convention: `threadgroups` groups of MS_GPU_THREADS threads.
    void dispatch(MTL::ComputePipelineState* state, const void* params, size_t bytes, const Binding* buffers,
                  size_t count, uint64_t threadgroups) {
        dispatch(state, params, bytes, buffers, count, Grid::linear(threadgroups));
    }
    template<class Params>
    void dispatch(MTL::ComputePipelineState* state, const Params& params, std::initializer_list<Binding> buffers,
                  uint64_t threadgroups) {
        check_params<Params>();
        dispatch(state, &params, sizeof(Params), buffers.begin(), buffers.size(), threadgroups);
    }
    void dispatch(const char* kernel, const void* params, size_t bytes, const Binding* buffers, size_t count,
                  uint64_t threadgroups) {
        dispatch(kernel, params, bytes, buffers, count, Grid::linear(threadgroups));
    }
    template<class Params>
    void dispatch(const char* kernel, const Params& params, std::initializer_list<Binding> buffers,
                  uint64_t threadgroups) {
        check_params<Params>();
        dispatch(kernel, &params, sizeof(Params), buffers.begin(), buffers.size(), threadgroups);
    }
    template<class Params>
    void dispatch(const char* kernel, const Params& params, const std::vector<Binding>& buffers,
                  uint64_t threadgroups) {
        check_params<Params>();
        dispatch(kernel, &params, sizeof(Params), buffers.data(), buffers.size(), threadgroups);
    }
    template<class Params>
    void dispatch(const char* kernel, const Params& params, std::initializer_list<Binding> buffers,
                  const Grid& grid) {
        check_params<Params>();
        dispatch(kernel, &params, sizeof(Params), buffers.begin(), buffers.size(), grid);
    }
    // Encodes an explicit device-side byte fill in the batch's blit pass before
    // its compute dispatches run.
    void fill(const ms_buffer* buffer, uint64_t byte_offset, uint64_t bytes, uint8_t value);
    // Makes buffers reached only through GPU addresses (chunk tables, address
    // tables) resident for later dispatches; duplicates are declared once.
    // Buffers that kernels write through addresses need ResourceUsageWrite.
    void use(const std::vector<MTL::Buffer*>&, MTL::ResourceUsage usage = MTL::ResourceUsageRead);
    // Commits and waits; command errors become MS_EBACKEND. Records counters.
    struct Timing {
        uint64_t encode_ns = 0;     // submit_mutex + commandBuffer + encode_batch + commit
        uint64_t gpu_ns = 0;        // GPUEndTime - GPUStartTime
        uint64_t host_wait_ns = 0;  // commit -> waitUntilCompleted
        uint64_t complete_ns = 0;   // complete_sync_gpu_batch
    };
    void commit_and_wait(Timing* timing = nullptr);
    // Commits without waiting, signaling the runtime MTLSharedEvent and marking
    // `outputs` with that producer event so downstream GPU batches chain via
    // encodeWait and host reads synchronize automatically.
    void commit_async(std::initializer_list<const ms_buffer*> outputs = {});
    void commit_async(const std::vector<const ms_buffer*>& outputs);

private:
    template<class Params> static constexpr void check_params() {
        static_assert(std::is_trivially_copyable_v<Params>, "kernel parameters are copied with setBytes");
        static_assert(sizeof(Params) % 8 == 0, "kernel parameter structs are a multiple of 8 bytes");
        static_assert(sizeof(Params) <= max_params_bytes, "setBytes takes at most 4 KiB");
    }
    struct RecordedDispatch {
        MTL::ComputePipelineState* state = nullptr;
        size_t param_offset = 0;
        size_t param_bytes = 0;
        size_t binding_offset = 0;
        size_t binding_count = 0;
        Grid grid;
        // Indexed dispatches: slots [slot_offset, slot_offset + slot_count)
        // say where each binding and blob goes; param_* is unused.
        bool indexed = false;
        size_t slot_offset = 0;
        size_t slot_count = 0;
    };
    struct Slot {
        uint32_t index = 0;
        bool is_bytes = false;
        size_t offset = 0; // into binding_storage_ or param_storage_
        size_t size = 0;   // blob bytes
    };
    struct RecordedUse {
        std::vector<MTL::Buffer*> buffers;
        MTL::ResourceUsage usage = MTL::ResourceUsageRead;
    };
    void encode_use(const std::vector<MTL::Buffer*>& buffers, MTL::ResourceUsage usage);
    PreparedGpuBatch encode_batch(const ms_buffer* const* outputs, size_t output_count);
    void commit_async_impl(const ms_buffer* const* outputs, size_t count);

    ms_context* context_;
    AutoreleasePool pool_;
    MTL::CommandBuffer* command_ = nullptr;
    MTL::ComputeCommandEncoder* encoder_ = nullptr;
    std::vector<uint8_t> param_storage_;
    std::vector<Binding> binding_storage_;
    std::vector<Slot> slot_storage_;
    std::vector<RecordedDispatch> commands_;
    std::vector<RecordedUse> uses_;
    std::vector<BlitFill> explicit_fills_;
    std::vector<const ms_buffer*> explicit_buffers_;
    uint64_t dispatches_ = 0;
    bool committed_ = false;
};
} // namespace ms::detail::gpu
