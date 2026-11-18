// E8 probe instrumentation: process-wide counters of Metal command-buffer
// activity, installed by swizzling the concrete Metal classes MLX uses.
// Used only to collect EVIDENCE about commits / host syncs; never installed
// while timing.

#pragma once

#include <cstdint>
#include <string>

namespace e8 {

struct MetalCounters {
  uint64_t cb_commandBuffer = 0;
  uint64_t cb_commandBufferWithUnretainedReferences = 0;
  uint64_t cb_commandBufferWithDescriptor = 0;
  uint64_t commit = 0;
  uint64_t waitUntilCompleted = 0;
  uint64_t waitUntilScheduled = 0;
  uint64_t addCompletedHandler = 0;
  uint64_t encodeSignalEvent = 0;
  uint64_t computeCommandEncoder = 0;
  uint64_t endEncoding = 0;
  uint64_t dispatchThreads = 0;
  uint64_t dispatchThreadgroups = 0;
  uint64_t sharedEventWait = 0;
};

// Returns an empty string on success, else a description of what failed.
std::string install_metal_counters();
bool metal_counters_installed();
MetalCounters metal_counters(bool reset);

} // namespace e8
