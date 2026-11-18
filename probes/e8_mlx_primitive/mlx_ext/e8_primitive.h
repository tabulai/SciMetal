// E8 probe: out-of-tree MLX primitive that encodes OUR offline-compiled Metal
// kernel into MLX's current command encoder.  Pinned to MLX 0.31.1.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mlx/array.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace e8 {

namespace mx = mlx::core;

// Directory + base name of the metallib (file is <dir>/<name>.metallib).
void set_metallib(std::string dir, std::string name);
std::string metallib_path();

// passes == 1: one dispatch.  passes == 2: the kernel is applied twice inside
// ONE primitive through a scratch array -- the shape a multi-step metal-sci
// plan takes (several dispatches + temporaries behind a single graph node).
class Correlate1DReflect : public mx::Primitive {
 public:
  explicit Correlate1DReflect(mx::Stream stream, int passes = 1)
      : mx::Primitive(stream), passes_(passes) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "E8Correlate1DReflect";
  }
  bool is_equivalent(const mx::Primitive& other) const override {
    return passes_ == static_cast<const Correlate1DReflect&>(other).passes_;
  }
  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  int passes_;
};

// Lazy op: 1-D correlation of `x` along its last axis with `weights`
// (scipy.ndimage.correlate1d(..., mode='reflect', origin=0) semantics),
// float32.  `guard` inserts mx::contiguous() nodes for inputs whose layout is
// not yet known to be row contiguous.
mx::array correlate1d_reflect(
    const mx::array& x,
    const mx::array& weights,
    mx::StreamOrDevice s = {},
    bool guard = true,
    int passes = 1);

// Lightweight instrumentation of the primitive's eval_gpu.
struct EvalGpuRecord {
  uintptr_t command_buffer;   // MLX stream command buffer the op encoded into
  bool needs_commit_after;    // Device::command_buffer_needs_commit after encode
  uint64_t encode_ns;         // wall time spent inside eval_gpu
};

struct EvalStats {
  uint64_t gpu_calls = 0;
  uint64_t gpu_total_ns = 0;
  uint64_t cpu_calls = 0;
  std::vector<EvalGpuRecord> recent; // most recent eval_gpu calls, oldest first
};

EvalStats eval_stats(bool reset);

} // namespace e8
