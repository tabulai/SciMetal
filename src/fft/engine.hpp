#pragma once
// GPU engine for FFT plans: encodes the passes of one or more plans into a
// single command buffer over pooled buffers.
#include "plan.hpp"
#include "ms/detail/storage.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace ms::detail::gpu {
class Batch;
}

namespace ms::detail::fft {
struct Profile {
    uint64_t parse_route_ns = 0;
    uint64_t stage_ns = 0;
    uint64_t refused_ns = 0;
    uint64_t replan_ns = 0;
    uint32_t replan_iters = 0;
    uint64_t alloc_in_ns = 0;
    uint64_t copy_in_ns = 0;
    uint64_t alloc_scratch_ns = 0;
    uint64_t encode_ns = 0;
    uint64_t gpu_ns = 0;
    uint64_t wait_overhead_ns = 0;
    uint64_t widen_alloc_ns = 0;
    uint64_t widen_store_ns = 0;
    uint64_t publish_ns = 0;
    uint64_t total_ns = 0;
    bool narrowed = false;
};

inline bool profile_enabled() noexcept {
    static const bool enabled = []() noexcept {
        const char* value = std::getenv("METAL_SCI_FFT_PROFILE");
        return value && *value && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

// One plan applied to `count` blocks of its request.
struct Stage {
    std::shared_ptr<const Plan> plan;
    uint64_t count = 0;
    std::shared_ptr<const void> resolved = {};
};

// The limits plans must be built for on this process's device.
Limits device_limits();

// A pipeline is granted fewer threads per threadgroup the more per-thread
// state its specialisation keeps (see Limits), which is known only once it is
// built. True when a step of the stages asks for more than its pipeline
// holds, with the first such stage and its grant; the caller replans that
// stage within the grant (replan) and asks again. Builds the stages'
// pipelines.
struct Refusal {
    size_t stage = 0;     // index of the refused stage
    int32_t granted = 0;  // threads per threadgroup its pipeline holds
};
bool refused(std::vector<Stage>& stages, Refusal& refusal);

// Owns scratch, intermediate, and constant-table buffers across one or more
// FFT stages encoded into a gpu::Batch. Callers must keep the Scratch alive
// until `batch.commit_and_wait()` finishes.
class Scratch {
public:
    explicit Scratch(ms_context* context);
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
    Scratch(Scratch&&) noexcept = default;
    Scratch& operator=(Scratch&&) noexcept = default;

    ms_buffer* acquire(uint64_t bytes);
    ms_buffer* adopt(BufferPtr buffer);
    void release(ms_buffer* buffer);
    BufferPtr take(ms_buffer* buffer);
    void clear();

private:
    ms_context* context_;
    std::vector<BufferPtr> owned_;
    std::vector<ms_buffer*> idle_;
};

// Encodes the stages in order into `batch` without committing. `input` is the
// first stage's slot 0 (which may be scratch-owned or caller-owned; if owned by
// `scratch` and not equal to the output slot, it is released to `scratch` for
// reuse by subsequent stages). `scale` multiplies the final result on the
// device (pass 1 when the caller applies its own factor). Returns the last
// stage's result buffer.
ms_buffer* encode(ms_context* context, gpu::Batch& batch, Scratch& scratch,
                  const std::vector<Stage>& stages, ms_buffer* input, float scale,
                  Profile* profile = nullptr);

// Runs the stages in order; each stage's result is the next stage's input.
// `input` is the first stage's slot 0, loaded by the caller and holding at
// least the bytes that slot needs. `scale` multiplies the final result on the
// device (pass 1 when the caller applies its own factor). Returns the last
// stage's result buffer, whose size is exactly the result's.
// MS_ENOTSUP when a batch exceeds what the kernels can index or a pipeline
// holds (refused).
BufferPtr run(ms_context* context, const std::vector<Stage>& stages, BufferPtr input, float scale,
              Profile* profile = nullptr);

// Builds every pipeline a run of the stages can need, so that the first run
// does not pay for them (tens of milliseconds per new kernel specialisation).
void prepare(const std::vector<Stage>& stages);
} // namespace ms::detail::fft
