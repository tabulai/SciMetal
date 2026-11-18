// E8 probe: MLX encoder for the e8_correlate1d_reflect_f32 kernel.
//
// eval_gpu loads the SAME offline-compiled metallib that the standalone
// runner uses (via metal::Device::get_library(name, dir)) and encodes one
// dispatch into MLX's current command encoder for the primitive's stream.
// eval_cpu computes the same thing on the CPU.

#include "e8_primitive.h"

#include <dlfcn.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/metal/device.h"
#include "mlx/ops.h"

namespace e8 {

namespace {

// Must match struct E8CorrelateParams in kernels/e8_correlate1d.metal.
struct E8CorrelateParams {
  uint32_t rows;
  uint32_t n;
  uint32_t nw;
  int32_t shift;
};

constexpr const char* kKernelName = "e8_correlate1d_reflect_f32";

std::mutex g_mtx;
std::string g_lib_dir;
std::string g_lib_name = "e8_kernels";
// Pipeline state resolved once through MLX's library/kernel caches; MLX keeps
// both alive for the lifetime of its (process-wide) Metal device.
std::atomic<MTL::ComputePipelineState*> g_kernel{nullptr};

// Ring buffer of the most recent eval_gpu records.
constexpr size_t kRecent = 8192;
EvalStats g_stats;
std::vector<EvalGpuRecord> g_ring(kRecent);
size_t g_ring_count = 0; // total records written since the last reset

std::string module_dir() {
  Dl_info info;
  if (dladdr(reinterpret_cast<const void*>(&module_dir), &info) &&
      info.dli_fname) {
    std::string p(info.dli_fname);
    auto pos = p.find_last_of('/');
    return pos == std::string::npos ? std::string(".") : p.substr(0, pos);
  }
  return ".";
}

inline uint64_t now_ns() {
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

struct Dims {
  uint32_t rows;
  uint32_t n;
  uint32_t nw;
};

Dims check_inputs(const mx::array& x, const mx::array& w) {
  if (x.dtype() != mx::float32 || w.dtype() != mx::float32) {
    throw std::invalid_argument("[e8.correlate1d] inputs must be float32");
  }
  if (!x.flags().row_contiguous || !w.flags().row_contiguous) {
    throw std::invalid_argument(
        "[e8.correlate1d] inputs must be row contiguous (use guard=True)");
  }
  size_t n = x.shape(-1);
  size_t rows = n == 0 ? 0 : x.size() / n;
  return {static_cast<uint32_t>(rows),
          static_cast<uint32_t>(n),
          static_cast<uint32_t>(w.size())};
}

inline int reflect_index(int idx, int n) {
  int period = 2 * n;
  int m = idx % period;
  if (m < 0) {
    m += period;
  }
  if (m >= n) {
    m = period - 1 - m;
  }
  return m;
}

// Same arithmetic, same summation order as the Metal kernel.
void correlate1d_cpu(
    const float* in,
    const float* w,
    float* out,
    uint32_t rows,
    uint32_t n_,
    uint32_t nw) {
  const int n = static_cast<int>(n_);
  const int shift = static_cast<int>(nw / 2);
  for (uint32_t r = 0; r < rows; ++r) {
    const float* row = in + static_cast<size_t>(r) * n_;
    float* orow = out + static_cast<size_t>(r) * n_;
    for (int i = 0; i < n; ++i) {
      const int start = i - shift;
      float acc = 0.0f;
      if (start >= 0 && start + static_cast<int>(nw) <= n) {
        for (uint32_t j = 0; j < nw; ++j) {
          acc += w[j] * row[start + j];
        }
      } else {
        for (uint32_t j = 0; j < nw; ++j) {
          acc += w[j] * row[reflect_index(start + static_cast<int>(j), n)];
        }
      }
      orow[i] = acc;
    }
  }
}

} // namespace

void set_metallib(std::string dir, std::string name) {
  std::lock_guard<std::mutex> lk(g_mtx);
  g_lib_dir = std::move(dir);
  g_lib_name = std::move(name);
  g_kernel.store(nullptr);
}

std::string metallib_path() {
  std::lock_guard<std::mutex> lk(g_mtx);
  if (g_lib_dir.empty()) {
    g_lib_dir = module_dir();
  }
  return g_lib_dir + "/" + g_lib_name + ".metallib";
}

EvalStats eval_stats(bool reset) {
  std::lock_guard<std::mutex> lk(g_mtx);
  EvalStats copy = g_stats;
  const size_t have = std::min(g_ring_count, kRecent);
  copy.recent.reserve(have);
  for (size_t i = g_ring_count - have; i < g_ring_count; ++i) {
    copy.recent.push_back(g_ring[i % kRecent]);
  }
  if (reset) {
    g_stats = EvalStats{};
    g_ring_count = 0;
  }
  return copy;
}

void Correlate1DReflect::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const uint64_t t0 = now_ns();
  auto& x = inputs[0];
  auto& w = inputs[1];
  auto& out = outputs[0];
  const Dims dims = check_inputs(x, w);

  out.set_data(mx::allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  auto* kernel = g_kernel.load(std::memory_order_acquire);
  if (!kernel) {
    std::string dir, name;
    {
      std::lock_guard<std::mutex> lk(g_mtx);
      if (g_lib_dir.empty()) {
        g_lib_dir = module_dir();
      }
      dir = g_lib_dir;
      name = g_lib_name;
    }
    // Loads <dir>/<name>.metallib -- the very file the standalone runner
    // uses.  MLX caches the MTL::Library by `name` and the pipeline state by
    // kernel name.
    auto lib = d.get_library(name, dir);
    kernel = d.get_kernel(kKernelName, lib);
    g_kernel.store(kernel, std::memory_order_release);
  }

  // Identical dispatch rule to the standalone runner.
  const size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
  const size_t gw = std::min<size_t>(dims.n, max_threads);
  const size_t gh =
      std::min<size_t>(dims.rows, std::max<size_t>(1, max_threads / gw));
  const MTL::Size grid(dims.n, dims.rows, 1);
  const MTL::Size group(gw, gh, 1);
  E8CorrelateParams p{
      dims.rows, dims.n, dims.nw, static_cast<int32_t>(dims.nw / 2)};

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  if (passes_ == 1) {
    enc.set_input_array(x, 0);
    enc.set_output_array(out, 1);
    enc.set_input_array(w, 2);
    enc.set_bytes(p, 3);
    enc.dispatch_threads(grid, group);
  } else {
    // Two dispatches through a scratch array.  Binding the scratch with
    // set_output_array / set_input_array (rather than a raw set_buffer) is
    // what lets MLX insert the memory barrier between the two dispatches of
    // its concurrent compute encoder; add_temporary keeps the scratch alive
    // until the command buffer that references it has completed.
    mx::array tmp(out.shape(), mx::float32, nullptr, {});
    tmp.set_data(mx::allocator::malloc(tmp.nbytes()));
    enc.set_input_array(x, 0);
    enc.set_output_array(tmp, 1);
    enc.set_input_array(w, 2);
    enc.set_bytes(p, 3);
    enc.dispatch_threads(grid, group);
    enc.set_input_array(tmp, 0);
    enc.set_output_array(out, 1);
    enc.dispatch_threads(grid, group);
    d.add_temporary(std::move(tmp), s.index);
  }

  // Instrumentation only (not needed for correctness).
  EvalGpuRecord rec;
  rec.command_buffer =
      reinterpret_cast<uintptr_t>(d.get_command_buffer(s.index));
  rec.needs_commit_after = d.command_buffer_needs_commit(s.index);
  rec.encode_ns = now_ns() - t0;
  std::lock_guard<std::mutex> lk(g_mtx);
  g_stats.gpu_calls++;
  g_stats.gpu_total_ns += rec.encode_ns;
  g_ring[g_ring_count++ % kRecent] = rec;
}

void Correlate1DReflect::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& x = inputs[0];
  auto& w = inputs[1];
  auto& out = outputs[0];
  const Dims dims = check_inputs(x, w);

  out.set_data(mx::allocator::malloc(out.nbytes()));

  // CPU evaluation in MLX is asynchronous too: the work is dispatched onto
  // the stream's thread; MLX keeps the input/output buffers alive until then.
  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_output_array(out);
  encoder.dispatch([x_ptr = x.data<float>(),
                    w_ptr = w.data<float>(),
                    out_ptr = out.data<float>(),
                    dims,
                    passes = passes_]() {
    if (passes == 1) {
      correlate1d_cpu(x_ptr, w_ptr, out_ptr, dims.rows, dims.n, dims.nw);
    } else {
      std::vector<float> tmp(static_cast<size_t>(dims.rows) * dims.n);
      correlate1d_cpu(x_ptr, w_ptr, tmp.data(), dims.rows, dims.n, dims.nw);
      correlate1d_cpu(tmp.data(), w_ptr, out_ptr, dims.rows, dims.n, dims.nw);
    }
  });

  std::lock_guard<std::mutex> lk(g_mtx);
  g_stats.cpu_calls++;
}

mx::array correlate1d_reflect(
    const mx::array& x_,
    const mx::array& w_,
    mx::StreamOrDevice s_,
    bool guard,
    int passes) {
  if (passes != 1 && passes != 2) {
    throw std::invalid_argument("[e8.correlate1d] passes must be 1 or 2");
  }
  if (x_.ndim() < 1) {
    throw std::invalid_argument("[e8.correlate1d] x must have ndim >= 1");
  }
  if (w_.ndim() != 1 || w_.size() < 1) {
    throw std::invalid_argument(
        "[e8.correlate1d] weights must be a non-empty 1-D array");
  }
  auto s = mx::to_stream(s_);
  mx::array x = x_;
  mx::array w = w_;
  if (x.dtype() != mx::float32) {
    x = mx::astype(x, mx::float32, s);
  }
  if (w.dtype() != mx::float32) {
    w = mx::astype(w, mx::float32, s);
  }
  if (guard) {
    // The kernel wants dense row-major inputs.  The layout of a lazy array is
    // unknown until it is evaluated, and MLX's internal copy_gpu() helper is
    // not exported from libmlx, so the portable way to guarantee it is a
    // graph-level Contiguous node (a buffer share, not a copy, when the
    // input already is row contiguous).  Skip it when the layout is known.
    auto ensure = [&s](mx::array a) {
      const bool known = a.status() != mx::array::Status::unscheduled &&
          a.flags().row_contiguous;
      return known ? a : mx::contiguous(a, false, s);
    };
    x = ensure(std::move(x));
    w = ensure(std::move(w));
  }
  auto shape = x.shape();
  return mx::array(
      std::move(shape),
      mx::float32,
      std::make_shared<Correlate1DReflect>(s, passes),
      {std::move(x), std::move(w)});
}

} // namespace e8
