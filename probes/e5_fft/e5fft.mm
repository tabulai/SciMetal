// SciMetal probe E5: standalone Metal runtime that drives MLX's GPU FFT
// kernels (vendored, MIT) WITHOUT linking libmlx.
//
// The plan/dispatch logic below (prime_factors .. fft_op) is a port of
// mlx/backend/metal/fft.cpp from MLX 0.31.1 (Copyright (c) 2024 Apple Inc.,
// MIT licence), with mlx::core::array replaced by (slot, dtype, rows, len)
// descriptors over our own shared-storage MTLBuffers.
//
// Plain C entry points, driven from Python through ctypes (run.py).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define EXPORT extern "C" __attribute__((visibility("default")))

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static id<MTLDevice> g_dev = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_lib_fft = nil; // vendored MLX kernels (or mlx.metallib)
static id<MTLLibrary> g_lib_ext = nil; // our additions
static std::map<std::string, id<MTLComputePipelineState>> g_pso;
static std::string g_err;
static int g_lib_gen = 0;
static bool g_untracked = true;
static double g_pso_build_s = 0;
static int g_pso_built = 0;

static double now_s() {
  static mach_timebase_info_data_t tb;
  if (tb.denom == 0) {
    mach_timebase_info(&tb);
  }
  return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

EXPORT const char* e5_last_error() {
  return g_err.c_str();
}

static id<MTLLibrary> load_lib(const char* path) {
  NSError* err = nil;
  NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
  id<MTLLibrary> lib = [g_dev newLibraryWithURL:url error:&err];
  if (!lib) {
    g_err = std::string("newLibraryWithURL failed for ") + path + ": " +
        (err ? err.localizedDescription.UTF8String : "?");
  }
  return lib;
}

// fft_metallib: metallib holding MLX's fft kernels (ours, compiled from the
// vendored headers -- or the wheel's mlx.metallib for A/B comparison).
EXPORT int e5_init(const char* fft_metallib, const char* ext_metallib) {
  @autoreleasepool {
    if (!g_dev) {
      g_dev = MTLCreateSystemDefaultDevice();
      if (!g_dev) {
        g_err = "no Metal device";
        return -1;
      }
      g_queue = [g_dev newCommandQueue];
    }
    g_pso.clear();
    g_lib_gen++;
    g_lib_fft = load_lib(fft_metallib);
    if (!g_lib_fft) {
      return -1;
    }
    if (ext_metallib && ext_metallib[0]) {
      g_lib_ext = load_lib(ext_metallib);
      if (!g_lib_ext) {
        return -1;
      }
    }
    return 0;
  }
}

EXPORT void e5_set_untracked(int v) {
  g_untracked = v != 0;
}

EXPORT int e5_device_name(char* buf, int len) {
  if (!g_dev) {
    return -1;
  }
  snprintf(buf, len, "%s", g_dev.name.UTF8String);
  return 0;
}

EXPORT double e5_pso_build_seconds() {
  return g_pso_build_s;
}
EXPORT int e5_pso_built() {
  return g_pso_built;
}

// ---------------------------------------------------------------------------
// Port of MLX's plan logic (mlx/backend/metal/fft.cpp)
// ---------------------------------------------------------------------------
#define MAX_STOCKHAM_FFT_SIZE 4096
#define MAX_RADER_FFT_SIZE 2048
#define MAX_BLUESTEIN_FFT_SIZE 2048
#define MIN_THREADGROUP_MEM_SIZE 256
#define MIN_COALESCE_WIDTH 4

static inline bool is_power_of_2(int v) {
  return (v != 0) && ((v & (v - 1)) == 0);
}
static inline int next_power_of_2(int v) {
  if (is_power_of_2(v)) {
    return v;
  }
  return (int)pow(2, (int)log2(v) + 1);
}

static const std::vector<int> supported_radices() {
  return {13, 11, 8, 7, 6, 5, 4, 3, 2};
}

static std::vector<int> prime_factors(int n) {
  int z = 2;
  std::vector<int> factors;
  while (z * z <= n) {
    if (n % z == 0) {
      factors.push_back(z);
      n /= z;
    } else {
      z++;
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  return factors;
}

struct FourStepParams {
  bool required = false;
  bool first_step = true;
  int n1 = 0;
  int n2 = 0;
};

struct FFTPlan {
  int n = 0;
  std::vector<int> stockham;
  std::vector<int> rader;
  int rader_n = 1;
  int bluestein_n = -1;
  bool four_step = false;
  int n1 = 0;
  int n2 = 0;
};

static int next_fast_n(int n) {
  return next_power_of_2(n);
}

static std::vector<int> plan_stockham_fft(int n) {
  auto radices = supported_radices();
  std::vector<int> plan(radices.size(), 0);
  int orig_n = n;
  if (n == 1) {
    return plan;
  }
  for (size_t i = 0; i < radices.size(); i++) {
    int radix = radices[i];
    if (is_power_of_2(orig_n) && orig_n < 512 && radix > 4) {
      continue;
    }
    while (n % radix == 0) {
      plan[i] += 1;
      n /= radix;
      if (n == 1) {
        return plan;
      }
    }
  }
  throw std::runtime_error("Unplannable");
}

static FFTPlan plan_fft(int n) {
  auto radices = supported_radices();
  std::set<int> radices_set(radices.begin(), radices.end());

  FFTPlan plan;
  plan.n = n;
  plan.rader = std::vector<int>(radices.size(), 0);
  auto factors = prime_factors(n);
  int remaining_n = n;

  if (n > MAX_STOCKHAM_FFT_SIZE && is_power_of_2(n)) {
    plan.four_step = true;
    plan.n2 = n > 65536 ? 1024 : 64;
    plan.n1 = n / plan.n2;
    return plan;
  } else if (n > MAX_STOCKHAM_FFT_SIZE) {
    plan.four_step = true;
    plan.bluestein_n = next_fast_n(2 * n - 1);
    return plan;
  }

  for (int factor : factors) {
    if (radices_set.find(factor) == radices_set.end()) {
      if (plan.rader_n > 1 || n > MAX_RADER_FFT_SIZE) {
        plan.four_step = n > MAX_BLUESTEIN_FFT_SIZE;
        plan.bluestein_n = next_fast_n(2 * n - 1);
        plan.stockham = plan_stockham_fft(plan.bluestein_n);
        plan.rader = std::vector<int>(radices.size(), 0);
        return plan;
      }
      auto rader_factors = prime_factors(factor - 1);
      for (int rf : rader_factors) {
        if (radices_set.find(rf) == radices_set.end()) {
          plan.four_step = n > MAX_BLUESTEIN_FFT_SIZE;
          plan.bluestein_n = next_fast_n(2 * n - 1);
          plan.stockham = plan_stockham_fft(plan.bluestein_n);
          plan.rader = std::vector<int>(radices.size(), 0);
          return plan;
        }
      }
      plan.rader = plan_stockham_fft(factor - 1);
      plan.rader_n = factor;
      remaining_n /= factor;
    }
  }

  plan.stockham = plan_stockham_fft(remaining_n);
  return plan;
}

static int compute_elems_per_thread(FFTPlan plan) {
  auto n = plan.n;

  std::vector<int> steps;
  auto radices = supported_radices();
  steps.insert(steps.end(), plan.stockham.begin(), plan.stockham.end());
  steps.insert(steps.end(), plan.rader.begin(), plan.rader.end());
  std::set<int> used_radices;
  for (size_t i = 0; i < steps.size(); i++) {
    int radix = radices[i % radices.size()];
    if (steps[i] > 0) {
      used_radices.insert(radix);
    }
  }

  if (used_radices.find(7) != used_radices.end() &&
      (used_radices.find(11) != used_radices.end() ||
       used_radices.find(13) != used_radices.end())) {
    return 7;
  } else if (
      used_radices.find(11) != used_radices.end() &&
      used_radices.find(13) != used_radices.end()) {
    return 11;
  }

  if (n == 3159)
    return 13;
  if (n == 3645)
    return 5;
  if (n == 3969)
    return 7;
  if (n == 1982)
    return 5;

  if (used_radices.size() == 1) {
    return *(used_radices.begin());
  }
  if (used_radices.size() == 2) {
    if (used_radices.find(11) != used_radices.end() ||
        used_radices.find(13) != used_radices.end()) {
      return std::accumulate(used_radices.begin(), used_radices.end(), 0) / 2;
    }
    std::vector<int> radix_vec(used_radices.begin(), used_radices.end());
    return radix_vec[1];
  }
  std::vector<int> radix_vec(used_radices.begin(), used_radices.end());
  return radix_vec[1];
}

// Rader
static int mod_exp(int x, int y, int n) {
  int out = 1;
  while (y) {
    if (y & 1) {
      out = out * x % n;
    }
    y >>= 1;
    x = x * x % n;
  }
  return out;
}

static int primitive_root(int n) {
  auto factors = prime_factors(n - 1);
  for (int r = 2; r < n - 1; r++) {
    bool found = true;
    for (int factor : factors) {
      if (mod_exp(r, (n - 1) / factor, n) == 1) {
        found = false;
        break;
      }
    }
    if (found) {
      return r;
    }
  }
  return -1;
}

// Host-side DFT for the (small) Rader / Bluestein constant tables. MLX uses
// single-precision pocketfft here; we use a direct double-precision DFT, so
// these tables can differ from MLX's by an ulp or so.
static std::vector<std::complex<double>> naive_dft(
    const std::vector<std::complex<double>>& x) {
  size_t n = x.size();
  std::vector<std::complex<double>> w(n), y(n);
  for (size_t k = 0; k < n; k++) {
    double th = -2.0 * M_PI * (double)k / (double)n;
    w[k] = {cos(th), sin(th)};
  }
  for (size_t k = 0; k < n; k++) {
    std::complex<double> acc = 0;
    size_t idx = 0;
    for (size_t j = 0; j < n; j++) {
      acc += x[j] * w[idx];
      idx += k;
      if (idx >= n) {
        idx -= n;
      }
    }
    y[k] = acc;
  }
  return y;
}

static id<MTLBuffer> new_buffer(size_t bytes) {
  MTLResourceOptions opt = MTLResourceStorageModeShared;
  if (g_untracked) {
    opt |= MTLResourceHazardTrackingModeUntracked;
  }
  return [g_dev newBufferWithLength:std::max<size_t>(bytes, 16) options:opt];
}

static void compute_raders_constants(
    int rader_n,
    id<MTLBuffer> __strong& b_q_buf,
    id<MTLBuffer> __strong& g_q_buf,
    id<MTLBuffer> __strong& g_minus_q_buf) {
  int proot = primitive_root(rader_n);
  int inv = mod_exp(proot, rader_n - 2, rader_n);
  std::vector<short> g_q(rader_n - 1);
  std::vector<short> g_minus_q(rader_n - 1);
  for (int i = 0; i < rader_n - 1; i++) {
    g_q[i] = mod_exp(proot, i, rader_n);
    g_minus_q[i] = mod_exp(inv, i, rader_n);
  }
  std::vector<std::complex<double>> b_q(rader_n - 1);
  for (int i = 0; i < rader_n - 1; i++) {
    float pi_i = (float)g_minus_q[i] * -2.0 * M_PI / rader_n;
    b_q[i] = std::exp(std::complex<double>(0, pi_i));
  }
  auto b_q_fft = naive_dft(b_q);

  b_q_buf = new_buffer((rader_n - 1) * 8);
  g_q_buf = new_buffer((rader_n - 1) * 2);
  g_minus_q_buf = new_buffer((rader_n - 1) * 2);
  float* b = (float*)b_q_buf.contents;
  for (int i = 0; i < rader_n - 1; i++) {
    b[2 * i] = (float)b_q_fft[i].real();
    b[2 * i + 1] = (float)b_q_fft[i].imag();
  }
  memcpy(g_q_buf.contents, g_q.data(), (rader_n - 1) * 2);
  memcpy(g_minus_q_buf.contents, g_minus_q.data(), (rader_n - 1) * 2);
}

static void compute_bluestein_constants(
    int n,
    int bluestein_n,
    id<MTLBuffer> __strong& w_k_buf,
    id<MTLBuffer> __strong& w_q_buf) {
  std::vector<std::complex<double>> w_q_vec(bluestein_n, 0);
  w_k_buf = new_buffer((size_t)n * 8);
  w_q_buf = new_buffer((size_t)bluestein_n * 8);
  float* wk = (float*)w_k_buf.contents;
  for (int i = -n + 1; i < n; i++) {
    double theta = pow(i, 2) * M_PI / (double)n;
    // MLX stores these as complex<float> before transforming
    std::complex<double> v = std::exp(std::complex<double>(0, theta));
    w_q_vec[i + n - 1] = {(double)(float)v.real(), (double)(float)v.imag()};
    if (i >= 0) {
      std::complex<double> c = std::exp(std::complex<double>(0, -theta));
      wk[2 * i] = (float)c.real();
      wk[2 * i + 1] = (float)c.imag();
    }
  }
  auto w_q = naive_dft(w_q_vec);
  float* wq = (float*)w_q_buf.contents;
  for (int i = 0; i < bluestein_n; i++) {
    wq[2 * i] = (float)w_q[i].real();
    wq[2 * i + 1] = (float)w_q[i].imag();
  }
}

// ---------------------------------------------------------------------------
// Pipelines (specialised with function constants, exactly like MLX)
// ---------------------------------------------------------------------------
struct FuncConsts {
  bool inv = false;
  bool pow2 = true;
  int ept = 0;
  int rader_m = 0;
  int stockham[9] = {0};
  int rader[9] = {0};
};

static id<MTLComputePipelineState> get_kernel(
    id<MTLLibrary> lib,
    const std::string& base_name,
    const std::string& hash_name,
    FuncConsts fc) {
  std::string key = std::to_string(g_lib_gen) + "/" + hash_name;
  auto it = g_pso.find(key);
  if (it != g_pso.end()) {
    return it->second;
  }
  double t0 = now_s();
  MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
  [cv setConstantValue:&fc.inv type:MTLDataTypeBool atIndex:0];
  [cv setConstantValue:&fc.pow2 type:MTLDataTypeBool atIndex:1];
  [cv setConstantValue:&fc.ept type:MTLDataTypeInt atIndex:2];
  [cv setConstantValue:&fc.rader_m type:MTLDataTypeInt atIndex:3];
  for (int i = 0; i < 9; i++) {
    [cv setConstantValue:&fc.stockham[i] type:MTLDataTypeInt atIndex:4 + i];
    [cv setConstantValue:&fc.rader[i] type:MTLDataTypeInt atIndex:13 + i];
  }
  MTLFunctionDescriptor* desc = [MTLFunctionDescriptor functionDescriptor];
  desc.name = [NSString stringWithUTF8String:base_name.c_str()];
  desc.specializedName = [NSString stringWithUTF8String:hash_name.c_str()];
  desc.constantValues = cv;
  NSError* err = nil;
  id<MTLFunction> fn = [lib newFunctionWithDescriptor:desc error:&err];
  if (!fn) {
    g_err = "Unable to load function " + base_name + ": " +
        (err ? err.localizedDescription.UTF8String : "not found");
    return nil;
  }
  id<MTLComputePipelineState> pso =
      [g_dev newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) {
    g_err = "Unable to build pipeline " + hash_name + ": " +
        (err ? err.localizedDescription.UTF8String : "?");
    return nil;
  }
  g_pso_build_s += now_s() - t0;
  g_pso_built++;
  g_pso[key] = pso;
  return pso;
}

// ---------------------------------------------------------------------------
// Plans = a list of kernel passes over pre-allocated shared buffers
// ---------------------------------------------------------------------------
struct Pass {
  id<MTLComputePipelineState> pso;
  std::string name;
  int in_slot = 0;
  int out_slot = 1;
  std::vector<id<MTLBuffer>> cbufs; // extra constant buffers from index 2
  std::vector<int32_t> ints; // scalar args following the buffers
  MTLSize grid;
  MTLSize group;
  int fft_size = 0;
  int ept = 0;
};

struct Arr {
  int slot;
  bool is_float;
  int64_t rows;
  int64_t len;
  int64_t size() const {
    return rows * len;
  }
};

struct Plan {
  int kind = 0; // 0 c2c, 1 r2c, 2 c2r
  int rows = 1;
  int n = 0;
  bool inverse = false;
  std::string mode;
  std::vector<id<MTLBuffer>> bufs; // 0 = in, 1 = out, 2.. = temporaries
  std::vector<size_t> buf_bytes;
  std::vector<Pass> passes;
  // SciMetal extension (multi-step plans only): requested coalesce width
  // (= FFTs per threadgroup) for strided passes. 0 = MLX's policy.
  int coalesce = 0;
  bool split_cb = false; // one command buffer per pass (diagnostic)
  double last_wall_s = 0;
  double last_gpu_s = 0;
  std::vector<double> pass_gpu_s;
};

static int new_slot(Plan& P, size_t bytes) {
  P.bufs.push_back(new_buffer(bytes));
  P.buf_bytes.push_back(bytes);
  return (int)P.bufs.size() - 1;
}

static bool fft_op(
    Plan& P,
    const Arr& in,
    const Arr& out,
    bool inverse,
    bool real,
    const FourStepParams fsp);

// Port of four_step_fft(). MLX's non-power-of-two branch
// (multi_upload_bluestein_fft, built from MLX's generic elementwise /
// pad / slice ops) is NOT ported.
static bool four_step_fft(
    Plan& P,
    const Arr& in,
    const Arr& out,
    bool inverse,
    bool real,
    FFTPlan& plan) {
  if (plan.bluestein_n == -1) {
    FourStepParams fsp = {true, true, plan.n1, plan.n2};
    const Arr& shape = (real && inverse) ? out : in;
    Arr temp = {
        new_slot(P, (size_t)shape.size() * 8), false, shape.rows, shape.len};
    if (!fft_op(P, in, temp, inverse, real, fsp)) {
      return false;
    }
    fsp.first_step = false;
    return fft_op(P, temp, out, inverse, real, fsp);
  }
  g_err = "multi-upload Bluestein (non power-of-two n > 2048 with unsupported "
          "factors, or any non power-of-two n > 4096) is not ported";
  return false;
}

// Port of fft_op(): one kernel dispatch (or recursion into four_step_fft).
static bool fft_op(
    Plan& P,
    const Arr& in,
    const Arr& out,
    bool inverse,
    bool real,
    const FourStepParams fsp) {
  int64_t n64 = out.is_float ? out.len : in.len;
  if (n64 == 1) {
    g_err = "n == 1 not handled by the probe";
    return false;
  }
  int n = (int)n64;
  if (fsp.required) {
    n = fsp.first_step ? fsp.n1 : fsp.n2;
  }

  FFTPlan plan;
  try {
    plan = plan_fft(n);
  } catch (std::exception& e) {
    g_err = e.what();
    return false;
  }
  if (plan.four_step) {
    return four_step_fft(P, in, out, inverse, real, plan);
  }

  int fft_size = plan.bluestein_n > 0 ? plan.bluestein_n : n;
  bool power_of_2 = is_power_of_2(fft_size);

  FuncConsts fc;
  fc.inv = inverse;
  fc.pow2 = power_of_2;
  for (int i = 0; i < 9; i++) {
    fc.stockham[i] = plan.stockham[i];
    fc.rader[i] = plan.rader[i];
  }
  int elems_per_thread = compute_elems_per_thread(plan);
  fc.ept = elems_per_thread;
  int rader_m = n / plan.rader_n;
  fc.rader_m = rader_m;

  int64_t size = out.is_float ? out.size() : in.size();
  if (real && inverse && fsp.required) {
    size = out.size();
  }
  int total_batch_size = (int)(size / n);
  int threads_per_fft = (fft_size + elems_per_thread - 1) / elems_per_thread;

  int threadgroup_batch_size = std::max(MIN_THREADGROUP_MEM_SIZE / fft_size, 1);
  if (fsp.required) {
    if (P.coalesce > 0) {
      // SciMetal extension: MLX hard-codes "at least 4", which (a) caps
      // four-step sub-FFTs at 1024 and so n at 2^20 and (b) leaves strided
      // passes with only 32 contiguous bytes per row. We let the planner pick
      // any width that fits 32KB of threadgroup memory / 1024 threads.
      int stride = fsp.first_step ? fsp.n2 : fsp.n1;
      threadgroup_batch_size = std::max(
          1,
          std::min(
              {P.coalesce,
               MAX_STOCKHAM_FFT_SIZE / fft_size,
               1024 / threads_per_fft,
               stride}));
    } else {
      threadgroup_batch_size =
          std::max(threadgroup_batch_size, MIN_COALESCE_WIDTH);
    }
  }
  int threadgroup_mem_size = next_power_of_2(threadgroup_batch_size * fft_size);
  if (P.coalesce > 0) {
    // kernels are only instantiated for 256..4096
    threadgroup_mem_size =
        std::max(threadgroup_mem_size, MIN_THREADGROUP_MEM_SIZE);
  }

  int batch_size =
      (total_batch_size + threadgroup_batch_size - 1) / threadgroup_batch_size;
  if (real && !fsp.required) {
    batch_size = (batch_size + 2 - 1) / 2;
  }

  const char* in_type_str = in.is_float ? "float" : "float2";
  const char* out_type_str = out.is_float ? "float" : "float2";
  int step = -1;
  std::ostringstream kname;
  std::string real_string = real ? "true" : "false";
  if (plan.bluestein_n > 0) {
    kname << "bluestein_fft_mem_" << threadgroup_mem_size << "_" << in_type_str
          << "_" << out_type_str;
  } else if (plan.rader_n > 1) {
    kname << "rader_fft_mem_" << threadgroup_mem_size << "_" << in_type_str
          << "_" << out_type_str;
  } else if (fsp.required) {
    step = fsp.first_step ? 0 : 1;
    kname << "four_step_mem_" << threadgroup_mem_size << "_" << in_type_str
          << "_" << out_type_str << "_" << step << "_" << real_string;
  } else {
    kname << "fft_mem_" << threadgroup_mem_size << "_" << in_type_str << "_"
          << out_type_str;
  }
  std::string base_name = kname.str();
  kname << "_n" << fft_size << "_inv_" << inverse;
  std::string hash_name = kname.str();

  Pass pass;
  pass.pso = get_kernel(g_lib_fft, base_name, hash_name, fc);
  if (!pass.pso) {
    return false;
  }
  pass.name = hash_name;
  pass.in_slot = in.slot;
  pass.out_slot = out.slot;
  pass.fft_size = fft_size;
  pass.ept = elems_per_thread;

  if (plan.bluestein_n > 0) {
    id<MTLBuffer> w_k, w_q;
    compute_bluestein_constants(n, plan.bluestein_n, w_k, w_q);
    pass.cbufs = {w_q, w_k};
    pass.ints = {n, plan.bluestein_n, total_batch_size};
  } else if (plan.rader_n > 1) {
    id<MTLBuffer> b_q, g_q, g_minus_q;
    compute_raders_constants(plan.rader_n, b_q, g_q, g_minus_q);
    pass.cbufs = {b_q, g_q, g_minus_q};
    pass.ints = {n, total_batch_size, plan.rader_n};
  } else if (fsp.required) {
    pass.ints = {fsp.n1, fsp.n2, total_batch_size};
  } else {
    pass.ints = {n, total_batch_size};
  }

  pass.group = MTLSizeMake(1, threadgroup_batch_size, threads_per_fft);
  pass.grid = MTLSizeMake(batch_size, threadgroup_batch_size, threads_per_fft);
  if (pass.pso.maxTotalThreadsPerThreadgroup <
      (NSUInteger)(threadgroup_batch_size * threads_per_fft)) {
    g_err = "threadgroup too large for " + hash_name;
    return false;
  }
  P.passes.push_back(pass);
  return true;
}

// SciMetal extension: the transposing middle pass (see e5_ext.metal).
//   in layout  [row][K (outer_n)][a_j (nj)][rest (stride)]
//   out layout [row][k_j][K][rest]
static bool emit_mid_pass(
    Plan& P,
    const Arr& in,
    const Arr& out,
    int nj,
    int stride,
    int outer_n) {
  if (!g_lib_ext) {
    g_err = "ext metallib not loaded";
    return false;
  }
  FFTPlan plan;
  try {
    plan = plan_fft(nj);
  } catch (std::exception& e) {
    g_err = e.what();
    return false;
  }
  if (plan.four_step || plan.bluestein_n > 0 || plan.rader_n > 1) {
    g_err = "multi-step factors must be Stockham sizes <= 4096";
    return false;
  }
  FuncConsts fc;
  fc.inv = false;
  fc.pow2 = is_power_of_2(nj);
  for (int i = 0; i < 9; i++) {
    fc.stockham[i] = plan.stockham[i];
  }
  int ept = compute_elems_per_thread(plan);
  fc.ept = ept;
  fc.rader_m = nj;
  int total_batch = (int)(in.size() / nj);
  int threads_per_fft = (nj + ept - 1) / ept;
  int want = P.coalesce > 0 ? P.coalesce : MIN_COALESCE_WIDTH;
  int tgb = std::max(
      1,
      std::min(
          {want, MAX_STOCKHAM_FFT_SIZE / nj, 1024 / threads_per_fft, stride}));
  int tg_mem = next_power_of_2(std::max(tgb * nj, MIN_THREADGROUP_MEM_SIZE));
  std::string base = "sm_mid_mem_" + std::to_string(tg_mem);
  std::string hash = base + "_n" + std::to_string(nj);
  Pass pass;
  pass.pso = get_kernel(g_lib_ext, base, hash, fc);
  if (!pass.pso) {
    return false;
  }
  if (pass.pso.maxTotalThreadsPerThreadgroup <
      (NSUInteger)(tgb * threads_per_fft)) {
    g_err = "threadgroup too large for " + hash;
    return false;
  }
  pass.name = hash;
  pass.in_slot = in.slot;
  pass.out_slot = out.slot;
  pass.ints = {nj, stride, total_batch, outer_n};
  pass.fft_size = nj;
  pass.ept = ept;
  pass.group = MTLSizeMake(1, tgb, threads_per_fft);
  pass.grid = MTLSizeMake((total_batch + tgb - 1) / tgb, tgb, threads_per_fft);
  P.passes.push_back(pass);
  return true;
}

// SciMetal extension: k-pass FFT for N = f_1 * ... * f_k (k >= 2).
//   pass 1      vendored four_step_fft step 0   (n1 = f_1, n2 = N / f_1)
//   pass 2..k-1 sm_mid_fft                      (transposing, e5_ext.metal)
//   pass k      vendored four_step_fft step 1   (n1 = N / f_k, n2 = f_k)
// k = 2 is exactly MLX's four-step (with a free choice of split and coalesce
// width). rfft / irfft reuse MLX's real four-step read/write variants for the
// first and last pass; the middle passes are always complex.
// Temporaries: c2c needs one full-size complex buffer for any k (passes
// ping-pong between `out` and the temporary so that the last one lands in
// `out`); real transforms need one (k = 2) or two (k >= 3) because `out` is
// too small to serve as scratch.
static bool build_multistep(
    Plan& P,
    const Arr& a_in,
    const Arr& a_out,
    const std::vector<int>& f) {
  bool real = P.kind != 0;
  int k = (int)f.size();
  int64_t prod = 1;
  for (int v : f) {
    if (v < 2 || v > MAX_STOCKHAM_FFT_SIZE) {
      g_err = "multi-step factors must be in [2, 4096]";
      return false;
    }
    prod *= v;
  }
  if (k < 2 || prod != P.n) {
    g_err = "multi-step factors must multiply to n (k >= 2)";
    return false;
  }
  int N = P.n;
  size_t full = (size_t)P.rows * N * 8;
  Arr tmp_a = {new_slot(P, full), false, P.rows, N};
  Arr tmp_b = tmp_a;
  if (real && k >= 3) {
    tmp_b = {new_slot(P, full), false, P.rows, N};
  }
  auto target = [&](int j) -> const Arr& {
    if (j == k) {
      return a_out;
    }
    if (real) {
      return (j % 2 == 1) ? tmp_a : tmp_b;
    }
    return ((k - j) % 2 == 0) ? a_out : tmp_a;
  };

  FourStepParams first = {true, true, f[0], N / f[0]};
  if (!fft_op(P, a_in, target(1), P.inverse, real, first)) {
    return false;
  }
  int outer = f[0];
  int rest = N / f[0];
  for (int j = 2; j <= k - 1; j++) {
    rest /= f[j - 1];
    if (!emit_mid_pass(P, target(j - 1), target(j), f[j - 1], rest, outer)) {
      return false;
    }
    outer *= f[j - 1];
  }
  FourStepParams last = {true, false, N / f[k - 1], f[k - 1]};
  return fft_op(P, target(k - 1), target(k), P.inverse, real, last);
}

static Plan* new_plan(int kind, int rows, int n, int inverse, Arr& in, Arr& out) {
  if (!g_dev || !g_lib_fft) {
    g_err = "e5_init not called";
    return nullptr;
  }
  if (kind != 0 && (kind == 1) == (inverse != 0)) {
    g_err = "rfft must be forward and irfft inverse";
    return nullptr;
  }
  Plan* P = new Plan();
  P->kind = kind;
  P->rows = rows;
  P->n = n;
  P->inverse = inverse != 0;
  int64_t half = n / 2 + 1;
  if (kind == 0) {
    in = {new_slot(*P, (size_t)rows * n * 8), false, rows, n};
    out = {new_slot(*P, (size_t)rows * n * 8), false, rows, n};
  } else if (kind == 1) {
    in = {new_slot(*P, (size_t)rows * n * 4), true, rows, n};
    out = {new_slot(*P, (size_t)rows * half * 8), false, rows, half};
  } else {
    in = {new_slot(*P, (size_t)rows * half * 8), false, rows, half};
    out = {new_slot(*P, (size_t)rows * n * 4), true, rows, n};
  }
  return P;
}

// Faithful port of MLX 0.31.1's plan, including its limits.
// kind: 0 = complex->complex, 1 = real->complex (rfft), 2 = complex->real
EXPORT void* e5_plan_create(int kind, int rows, int n, int inverse) {
  @autoreleasepool {
    Arr in, out;
    Plan* P = new_plan(kind, rows, n, inverse, in, out);
    if (!P) {
      return nullptr;
    }
    P->mode = "mlx";
    if (!fft_op(*P, in, out, P->inverse, kind != 0, FourStepParams())) {
      delete P;
      return nullptr;
    }
    return P;
  }
}

// SciMetal multi-step plan: n = factors[0] * ... * factors[k-1].
// coalesce = requested FFTs per threadgroup for the strided passes (clamped
// to what fits 32KB threadgroup memory / 1024 threads); 4 = MLX's value.
EXPORT void* e5_plan_create_ms(
    int kind,
    int rows,
    int n,
    int inverse,
    const int* factors,
    int k,
    int coalesce) {
  @autoreleasepool {
    Arr in, out;
    Plan* P = new_plan(kind, rows, n, inverse, in, out);
    if (!P) {
      return nullptr;
    }
    P->mode = "ms";
    P->coalesce = coalesce > 0 ? coalesce : MIN_COALESCE_WIDTH;
    if (!build_multistep(*P, in, out, std::vector<int>(factors, factors + k))) {
      delete P;
      return nullptr;
    }
    return P;
  }
}

EXPORT void e5_plan_destroy(void* h) {
  @autoreleasepool {
    delete (Plan*)h;
  }
}

EXPORT void* e5_plan_in(void* h) {
  return ((Plan*)h)->bufs[0].contents;
}
EXPORT void* e5_plan_out(void* h) {
  return ((Plan*)h)->bufs[1].contents;
}
EXPORT int e5_plan_num_passes(void* h) {
  return (int)((Plan*)h)->passes.size();
}
EXPORT int e5_plan_num_temps(void* h) {
  return (int)((Plan*)h)->bufs.size() - 2;
}
EXPORT void e5_plan_set_split_cb(void* h, int v) {
  ((Plan*)h)->split_cb = v != 0;
}
EXPORT double e5_plan_last_gpu_seconds(void* h) {
  return ((Plan*)h)->last_gpu_s;
}

// Per-pass GPU seconds of the last exec in split_cb mode.
EXPORT double e5_plan_pass_gpu_seconds(void* h, int i) {
  Plan* P = (Plan*)h;
  return i < (int)P->pass_gpu_s.size() ? P->pass_gpu_s[i] : -1.0;
}

EXPORT int e5_plan_describe(void* h, char* buf, int len) {
  Plan* P = (Plan*)h;
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < P->passes.size(); i++) {
    auto& p = P->passes[i];
    if (i) {
      os << ", ";
    }
    os << "{\"kernel\": \"" << p.name << "\", \"in\": " << p.in_slot
       << ", \"out\": " << p.out_slot << ", \"fft_size\": " << p.fft_size
       << ", \"elems_per_thread\": " << p.ept << ", \"grid\": ["
       << p.grid.width << ", " << p.grid.height << ", " << p.grid.depth
       << "], \"group\": [" << p.group.width << ", " << p.group.height << ", "
       << p.group.depth << "], \"ints\": [";
    for (size_t j = 0; j < p.ints.size(); j++) {
      os << (j ? ", " : "") << p.ints[j];
    }
    os << "], \"const_buffers\": " << p.cbufs.size() << "}";
  }
  os << "]";
  snprintf(buf, len, "%s", os.str().c_str());
  return (int)os.str().size();
}

static void encode_pass(
    Plan* P,
    id<MTLComputeCommandEncoder> enc,
    Pass& p) {
  [enc setComputePipelineState:p.pso];
  [enc setBuffer:P->bufs[p.in_slot] offset:0 atIndex:0];
  [enc setBuffer:P->bufs[p.out_slot] offset:0 atIndex:1];
  int idx = 2;
  for (id<MTLBuffer> b : p.cbufs) {
    [enc setBuffer:b offset:0 atIndex:idx++];
  }
  for (int32_t& v : p.ints) {
    [enc setBytes:&v length:sizeof(int32_t) atIndex:idx++];
  }
  [enc dispatchThreads:p.grid threadsPerThreadgroup:p.group];
}

// Timed region = encode + commit + waitUntilCompleted. Returns seconds, or a
// negative value on a command-buffer error.
EXPORT double e5_plan_exec(void* h) {
  Plan* P = (Plan*)h;
  @autoreleasepool {
    double t0 = now_s();
    double gpu = 0;
    bool failed = false;
    if (!P->split_cb) {
      id<MTLCommandBuffer> cb = [g_queue commandBufferWithUnretainedReferences];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      for (auto& p : P->passes) {
        encode_pass(P, enc, p);
      }
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      failed = cb.status == MTLCommandBufferStatusError;
      if (failed) {
        g_err = cb.error ? cb.error.localizedDescription.UTF8String : "cb error";
      }
      gpu = cb.GPUEndTime - cb.GPUStartTime;
    } else {
      P->pass_gpu_s.clear();
      for (auto& p : P->passes) {
        id<MTLCommandBuffer> cb =
            [g_queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encode_pass(P, enc, p);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
          failed = true;
          g_err =
              cb.error ? cb.error.localizedDescription.UTF8String : "cb error";
        }
        gpu += cb.GPUEndTime - cb.GPUStartTime;
        P->pass_gpu_s.push_back(cb.GPUEndTime - cb.GPUStartTime);
      }
    }
    double t1 = now_s();
    P->last_wall_s = t1 - t0;
    P->last_gpu_s = gpu;
    return failed ? -1.0 : (t1 - t0);
  }
}

// An empty command buffer round trip: the floor of the fixed overhead.
EXPORT double e5_empty_roundtrip() {
  @autoreleasepool {
    double t0 = now_s();
    id<MTLCommandBuffer> cb = [g_queue commandBufferWithUnretainedReferences];
    [cb commit];
    [cb waitUntilCompleted];
    return now_s() - t0;
  }
}

// ---------------------------------------------------------------------------
// Accelerate vDSP baselines (complex single precision, power-of-two n)
// ---------------------------------------------------------------------------
struct VDSP {
  int log2n = 0;
  size_t n = 0;
  FFTSetup fft = nullptr;
  vDSP_DFT_Setup dft = nullptr;
  vDSP_DFT_Interleaved_Setup dfti = nullptr;
  float *ir = nullptr, *ii = nullptr, *outr = nullptr, *outi = nullptr;
};

static float* alloc_f(size_t n) {
  void* p = nullptr;
  posix_memalign(&p, 16384, n * sizeof(float));
  memset(p, 0, n * sizeof(float));
  return (float*)p;
}

EXPORT void* e5_vdsp_create(int log2n) {
  VDSP* v = new VDSP();
  v->log2n = log2n;
  v->n = (size_t)1 << log2n;
  v->fft = vDSP_create_fftsetup(log2n, kFFTRadix2);
  v->dft = vDSP_DFT_zop_CreateSetup(nullptr, v->n, vDSP_DFT_FORWARD);
  if (@available(macOS 12.0, *)) {
    v->dfti = vDSP_DFT_Interleaved_CreateSetup(
        nullptr, v->n, vDSP_DFT_FORWARD, vDSP_DFT_Interleaved_ComplextoComplex);
  }
  v->ir = alloc_f(v->n);
  v->ii = alloc_f(v->n);
  v->outr = alloc_f(v->n);
  v->outi = alloc_f(v->n);
  return v;
}

EXPORT int e5_vdsp_available(void* h, int variant) {
  VDSP* v = (VDSP*)h;
  switch (variant) {
    case 0:
    case 1:
      return v->fft != nullptr;
    case 2:
      return v->dft != nullptr;
    case 3:
      return v->dfti != nullptr;
  }
  return 0;
}

// variant 0: vDSP_fft_zop on split-complex data (conversion NOT timed)
// variant 1: vDSP_ctoz + vDSP_fft_zop + vDSP_ztoc (interleaved in/out, timed)
// variant 2: vDSP_DFT_Execute (zop) on split-complex data (conversion NOT timed)
// variant 3: vDSP_DFT_Interleaved_Execute (interleaved in/out)
// in / out: interleaved complex64, n elements. Returns seconds for one
// forward transform, negative if unavailable.
EXPORT double e5_vdsp_run(void* h, int variant, const float* in, float* out) {
  VDSP* v = (VDSP*)h;
  size_t n = v->n;
  DSPSplitComplex si = {v->ir, v->ii};
  DSPSplitComplex so = {v->outr, v->outi};
  double t0, t1;
  if (variant == 0 || variant == 2) {
    if ((variant == 0 && !v->fft) || (variant == 2 && !v->dft)) {
      return -1;
    }
    vDSP_ctoz((const DSPComplex*)in, 2, &si, 1, n);
    t0 = now_s();
    if (variant == 0) {
      vDSP_fft_zop(v->fft, &si, 1, &so, 1, v->log2n, kFFTDirection_Forward);
    } else {
      vDSP_DFT_Execute(v->dft, v->ir, v->ii, v->outr, v->outi);
    }
    t1 = now_s();
    vDSP_ztoc(&so, 1, (DSPComplex*)out, 2, n);
  } else if (variant == 1) {
    if (!v->fft) {
      return -1;
    }
    t0 = now_s();
    vDSP_ctoz((const DSPComplex*)in, 2, &si, 1, n);
    vDSP_fft_zop(v->fft, &si, 1, &so, 1, v->log2n, kFFTDirection_Forward);
    vDSP_ztoc(&so, 1, (DSPComplex*)out, 2, n);
    t1 = now_s();
  } else {
    if (!v->dfti) {
      return -1;
    }
    if (@available(macOS 12.0, *)) {
      t0 = now_s();
      vDSP_DFT_Interleaved_Execute(
          v->dfti, (const DSPComplex*)in, (DSPComplex*)out);
      t1 = now_s();
    } else {
      return -1;
    }
  }
  return t1 - t0;
}

// Batched rows x n along the last axis with the interleaved DFT (variant 3)
// or ctoz+fft_zop+ztoc (variant 1); nthreads > 1 uses dispatch_apply over
// row chunks (vDSP itself is single threaded).
EXPORT double e5_vdsp_run_batch(
    void* h,
    int variant,
    int rows,
    int nthreads,
    const float* in,
    float* out) {
  VDSP* v = (VDSP*)h;
  size_t n = v->n;
  if ((variant == 3 && !v->dfti) || (variant == 1 && !v->fft)) {
    return -1;
  }
  if (variant != 1 && variant != 3) {
    return -1;
  }
  int chunks = std::max(1, std::min(nthreads, rows));
  // per-chunk split scratch for variant 1
  std::vector<float*> scratch(chunks * 4, nullptr);
  if (variant == 1) {
    for (auto& s : scratch) {
      s = alloc_f(n);
    }
  }
  float** sc = scratch.data();
  double t0 = now_s();
  auto body = ^(size_t c) {
    int r0 = (int)((int64_t)rows * c / chunks);
    int r1 = (int)((int64_t)rows * (c + 1) / chunks);
    for (int r = r0; r < r1; r++) {
      const float* ip = in + (size_t)r * n * 2;
      float* op = out + (size_t)r * n * 2;
      if (variant == 3) {
        if (@available(macOS 12.0, *)) {
          vDSP_DFT_Interleaved_Execute(
              v->dfti, (const DSPComplex*)ip, (DSPComplex*)op);
        }
      } else {
        DSPSplitComplex si = {sc[c * 4], sc[c * 4 + 1]};
        DSPSplitComplex so = {sc[c * 4 + 2], sc[c * 4 + 3]};
        vDSP_ctoz((const DSPComplex*)ip, 2, &si, 1, n);
        vDSP_fft_zop(v->fft, &si, 1, &so, 1, v->log2n, kFFTDirection_Forward);
        vDSP_ztoc(&so, 1, (DSPComplex*)op, 2, n);
      }
    }
  };
  if (chunks == 1) {
    body(0);
  } else {
    dispatch_apply(
        chunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), body);
  }
  double t1 = now_s();
  for (auto s : scratch) {
    free(s);
  }
  return t1 - t0;
}

EXPORT void e5_vdsp_destroy(void* h) {
  VDSP* v = (VDSP*)h;
  if (v->fft) {
    vDSP_destroy_fftsetup(v->fft);
  }
  if (v->dft) {
    vDSP_DFT_DestroySetup(v->dft);
  }
  if (v->dfti) {
    if (@available(macOS 12.0, *)) {
      vDSP_DFT_Interleaved_DestroySetup(v->dfti);
    }
  }
  free(v->ir);
  free(v->ii);
  free(v->outr);
  free(v->outi);
  delete v;
}
