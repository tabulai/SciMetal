#pragma once
// GPU engine for FFT plans: encodes the passes of one or more plans into a
// single command buffer over pooled buffers.
#include "plan.hpp"
#include "ms/detail/storage.hpp"

#include <memory>
#include <vector>

namespace ms::detail::fft {
// One plan applied to `count` blocks of its request.
struct Stage {
    std::shared_ptr<const Plan> plan;
    uint64_t count = 0;
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
bool refused(const std::vector<Stage>& stages, Refusal& refusal);

// Runs the stages in order; each stage's result is the next stage's input.
// `input` is the first stage's slot 0, loaded by the caller and holding at
// least the bytes that slot needs. `scale` multiplies the final result on the
// device (pass 1 when the caller applies its own factor). Returns the last
// stage's result buffer, whose size is exactly the result's.
// MS_ENOTSUP when a batch exceeds what the kernels can index or a pipeline
// holds (refused).
BufferPtr run(ms_context* context, const std::vector<Stage>& stages, BufferPtr input, float scale);

// Builds every pipeline a run of the stages can need, so that the first run
// does not pay for them (tens of milliseconds per new kernel specialisation).
void prepare(const std::vector<Stage>& stages);
} // namespace ms::detail::fft
