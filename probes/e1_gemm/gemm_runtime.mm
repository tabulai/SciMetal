// E1 probe runtime: own MTLDevice / queue / shared buffers, MLX steel GEMM.
// Build with -fobjc-arc. No libmlx.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

#include "mlx/backend/metal/kernels/steel/gemm/params.h"  // GEMMParams, GEMMAddMMParams

#include "gemm_runtime.h"

using mlx::steel::GEMMAddMMParams;
using mlx::steel::GEMMParams;

namespace {

struct Runtime {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  std::string name, arch;
  char devc = 'd';
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
  std::string last_kernel;
  int last_dispatch[5] = {0, 0, 0, 0, 0};
};

Runtime* g = nullptr;

struct Variant {
  int bm, bn, bk, wm, wn;
};
const Variant kVariants[] = {
    {0, 0, 0, 0, 0},     // 0: auto
    {64, 64, 16, 2, 2},  // 1
    {64, 32, 32, 2, 2},  // 2
    {32, 64, 16, 1, 2},  // 3
    {32, 32, 16, 2, 2},  // 4
};

double now_s() {
  static mach_timebase_info_data_t tb = {0, 0};
  if (tb.denom == 0) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

// MLX matmul.cpp GEMM_TPARAM_MACRO, float32 branch only.
Variant mlx_choice(char devc, bool ta, bool tb, int M, int N) {
  Variant v = {64, 64, 16, 2, 2};
  if (devc == 'g' || devc == 'p') {  // small device
    if (!ta && tb) v = {64, 32, 32, 2, 2};
  } else if (devc == 'd') {  // large device
    if ((size_t)M * (size_t)N >= (1ul << 20)) {
      // float takes default
    } else {
      if (!ta && tb)
        v = {32, 64, 16, 1, 2};
      else
        v = {64, 32, 32, 2, 2};
    }
  }
  return v;
}

id<MTLComputePipelineState> get_pipeline(const Variant& v, bool ta, bool tb,
                                         bool use_out_source, bool do_axpby,
                                         bool align_M, bool align_N,
                                         bool align_K) {
  char base[160];
  snprintf(base, sizeof(base),
           "steel_gemm_fused_%c%c_float32_float32_bm%d_bn%d_bk%d_wm%d_wn%d",
           ta ? 't' : 'n', tb ? 't' : 'n', v.bm, v.bn, v.bk, v.wm, v.wn);
  char full[256];
  snprintf(full, sizeof(full),
           "%s_has_batch_n_use_out_source_%c_do_axpby_%c_align_M_%c_align_N_%c_align_K_%c",
           base, use_out_source ? 't' : 'n', do_axpby ? 't' : 'n',
           align_M ? 't' : 'n', align_N ? 't' : 'n', align_K ? 't' : 'n');
  g->last_kernel = full;
  auto it = g->pipelines.find(full);
  if (it != g->pipelines.end()) return it->second;

  MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
  bool has_batch = false;
  [cv setConstantValue:&has_batch type:MTLDataTypeBool atIndex:10];
  [cv setConstantValue:&use_out_source type:MTLDataTypeBool atIndex:100];
  [cv setConstantValue:&do_axpby type:MTLDataTypeBool atIndex:110];
  [cv setConstantValue:&align_M type:MTLDataTypeBool atIndex:200];
  [cv setConstantValue:&align_N type:MTLDataTypeBool atIndex:201];
  [cv setConstantValue:&align_K type:MTLDataTypeBool atIndex:202];

  MTLFunctionDescriptor* fd = [MTLFunctionDescriptor functionDescriptor];
  fd.name = [NSString stringWithUTF8String:base];
  fd.constantValues = cv;
  NSError* err = nil;
  id<MTLFunction> fn = [g->library newFunctionWithDescriptor:fd error:&err];
  if (!fn) {
    fprintf(stderr, "[e1] no function %s: %s\n", base,
            err ? err.localizedDescription.UTF8String : "?");
    return nil;
  }
  id<MTLComputePipelineState> ps =
      [g->device newComputePipelineStateWithFunction:fn error:&err];
  if (!ps) {
    fprintf(stderr, "[e1] pipeline %s failed: %s\n", full,
            err ? err.localizedDescription.UTF8String : "?");
    return nil;
  }
  g->pipelines[full] = ps;
  return ps;
}

bool encode_one(id<MTLComputeCommandEncoder> enc, const msg_gemm_args* a) {
  const bool ta = a->ta != 0, tb = a->tb != 0;
  Variant v = (a->variant > 0 && a->variant <= 4)
                  ? kVariants[a->variant]
                  : mlx_choice(g->devc, ta, tb, a->M, a->N);
  const int M = a->M, N = a->N, K = a->K;
  // Same semantics as MLX's AddMM path: the out-source (C) epilogue is used
  // whenever a C is supplied and (alpha, beta) != (0, 1).
  const bool use_out_source = (a->C != nullptr) && (a->alpha != 0.0f || a->beta != 1.0f);
  const bool do_axpby = use_out_source && (a->alpha != 1.0f || a->beta != 1.0f);
  const bool align_M = (M % v.bm) == 0;
  const bool align_N = (N % v.bn) == 0;
  const bool align_K = (K % v.bk) == 0;

  id<MTLComputePipelineState> ps =
      get_pipeline(v, ta, tb, use_out_source, do_axpby, align_M, align_N, align_K);
  if (!ps) return false;

  int tn = (N + v.bn - 1) / v.bn;
  int tm = (M + v.bm - 1) / v.bm;
  const int swizzle_log = 0;  // MLX: fixed to 0 for the non-NAX path
  GEMMParams params{M, N, K, a->lda, a->ldb, a->ldd, tn, tm,
                    /*batch_stride_a*/ 0, /*batch_stride_b*/ 0,
                    /*batch_stride_d*/ (int64_t)M * N,
                    swizzle_log, K / v.bk, /*batch_ndim*/ 1};
  const int tile = 1 << swizzle_log;
  tm = (tm + tile - 1) / tile;
  tn = tn * tile;

  [enc setComputePipelineState:ps];
  [enc setBuffer:(__bridge id<MTLBuffer>)a->A offset:a->a_off * sizeof(float) atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)a->B offset:a->b_off * sizeof(float) atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)a->D offset:a->d_off * sizeof(float) atIndex:3];
  [enc setBytes:&params length:sizeof(params) atIndex:4];
  if (use_out_source) {
    GEMMAddMMParams ap{a->ldc, /*fdc*/ 1, /*batch_stride_c*/ 0, a->alpha, a->beta};
    [enc setBuffer:(__bridge id<MTLBuffer>)a->C offset:a->c_off * sizeof(float) atIndex:2];
    [enc setBytes:&ap length:sizeof(ap) atIndex:5];
  }
  [enc dispatchThreadgroups:MTLSizeMake(tn, tm, 1)
      threadsPerThreadgroup:MTLSizeMake(32, v.wn, v.wm)];
  g->last_dispatch[0] = tn;
  g->last_dispatch[1] = tm;
  g->last_dispatch[2] = 32;
  g->last_dispatch[3] = v.wn;
  g->last_dispatch[4] = v.wm;
  return true;
}

}  // namespace

extern "C" {

double msg_now(void) { return now_s(); }

// Raise the calling thread's QoS class (0 = leave, 1 = user-initiated,
// 2 = user-interactive). The thread that blocks in waitUntilCompleted is woken
// sooner on a busy machine when it is not at default/utility QoS.
int msg_set_qos(int level) {
  if (level == 1) return pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
  if (level == 2) return pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  if (level == -1) return pthread_set_qos_class_self_np(QOS_CLASS_DEFAULT, 0);
  return 0;
}

int msg_init(const char* metallib_path, char* err, int errlen) {
  @autoreleasepool {
    if (g) return 0;
    Runtime* r = new Runtime();
    r->device = MTLCreateSystemDefaultDevice();
    if (!r->device) {
      if (err) snprintf(err, errlen, "no Metal device");
      return -1;
    }
    r->queue = [r->device newCommandQueue];
    NSError* e = nil;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path]];
    r->library = [r->device newLibraryWithURL:url error:&e];
    if (!r->library) {
      if (err)
        snprintf(err, errlen, "newLibraryWithURL failed: %s",
                 e ? e.localizedDescription.UTF8String : "?");
      return -2;
    }
    r->name = r->device.name.UTF8String;
    if (@available(macOS 14.0, *)) {
      r->arch = r->device.architecture.name.UTF8String;
    } else {
      r->arch = "unknown_d";
    }
    r->devc = r->arch.empty() ? 'd' : r->arch.back();
    g = r;
    return 0;
  }
}

const char* msg_device_name(void) { return g ? g->name.c_str() : ""; }
const char* msg_device_arch(void) { return g ? g->arch.c_str() : ""; }
const char* msg_last_kernel(void) { return g ? g->last_kernel.c_str() : ""; }
void msg_last_dispatch(int out[5]) {
  for (int i = 0; i < 5; i++) out[i] = g ? g->last_dispatch[i] : 0;
}

void* msg_alloc_padded(size_t nbytes, size_t pad_bytes) {
  @autoreleasepool {
    // MSG_UNTRACKED=1 selects MTLResourceHazardTrackingModeUntracked (what MLX's
    // allocator uses); default is Metal's default tracking mode.
    static const bool untracked = getenv("MSG_UNTRACKED") != nullptr;
    MTLResourceOptions opt = MTLResourceStorageModeShared;
    if (untracked) opt |= MTLResourceHazardTrackingModeUntracked;
    id<MTLBuffer> b = [g->device newBufferWithLength:nbytes + pad_bytes options:opt];
    if (!b) return nullptr;
    return (void*)CFBridgingRetain(b);
  }
}

void* msg_alloc(size_t nbytes) {
  // Pad: the steel "safe" epilogue (apply_epilogue_safe) reads C for every row
  // of an edge tile's fragments (only columns are bound-checked), i.e. up to
  // (BM-1) rows = (BM-1)*ldc elements past the last valid row. Reads only, and
  // the values are discarded, but keep them inside the allocation. Callers
  // whose destination view ends at the end of the buffer should use
  // msg_alloc_padded with pad >= 64 * ldc * sizeof(float).
  return msg_alloc_padded(nbytes, 64 * 1024);
}
void* msg_contents(void* buf) { return [(__bridge id<MTLBuffer>)buf contents]; }
size_t msg_length(void* buf) { return [(__bridge id<MTLBuffer>)buf length]; }
void msg_free(void* buf) {
  if (buf) CFBridgingRelease(buf);
}

void* msg_gemm_submit_many(const msg_gemm_args* a, int count) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [g->queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    bool ok = true;
    for (int i = 0; i < count && ok; i++) ok = encode_one(enc, &a[i]);
    [enc endEncoding];
    if (!ok) return nullptr;
    [cb commit];
    return (void*)CFBridgingRetain(cb);
  }
}

void* msg_gemm_submit(const msg_gemm_args* a) { return msg_gemm_submit_many(a, 1); }

int msg_done(void* h) {
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)h;
  return cb.status >= MTLCommandBufferStatusCompleted ? 1 : 0;
}

int msg_wait(void* h, double* gpu_seconds) {
  @autoreleasepool {
    if (!h) return -1;
    id<MTLCommandBuffer> cb = CFBridgingRelease(h);
    [cb waitUntilCompleted];
    if (gpu_seconds) *gpu_seconds = cb.GPUEndTime - cb.GPUStartTime;
    if (cb.status != MTLCommandBufferStatusCompleted) {
      fprintf(stderr, "[e1] command buffer failed: %s\n",
              cb.error ? cb.error.localizedDescription.UTF8String : "?");
      return -2;
    }
    return 0;
  }
}

int msg_gemm(const msg_gemm_args* a, double* times) {
  @autoreleasepool {
    double t0 = now_s();
    id<MTLCommandBuffer> cb = [g->queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    bool ok = encode_one(enc, a);
    [enc endEncoding];
    if (!ok) return -1;
    double t1 = now_s();
    [cb commit];
    double t2 = now_s();
    [cb waitUntilCompleted];
    double t3 = now_s();
    if (times) {
      times[0] = t1 - t0;
      times[1] = t2 - t1;
      times[2] = t3 - t2;
      times[3] = cb.GPUEndTime - cb.GPUStartTime;
    }
    if (cb.status != MTLCommandBufferStatusCompleted) {
      fprintf(stderr, "[e1] command buffer failed: %s\n",
              cb.error ? cb.error.localizedDescription.UTF8String : "?");
      return -2;
    }
    return 0;
  }
}

// Same as msg_gemm but busy-polls the command buffer status instead of
// blocking in waitUntilCompleted (probe: is the blocking wake-up the cost?).
int msg_gemm_spin(const msg_gemm_args* a, double* times) {
  @autoreleasepool {
    double t0 = now_s();
    id<MTLCommandBuffer> cb = [g->queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    bool ok = encode_one(enc, a);
    [enc endEncoding];
    if (!ok) return -1;
    double t1 = now_s();
    [cb commit];
    double t2 = now_s();
    while (cb.status < MTLCommandBufferStatusCompleted) {
    }
    double t3 = now_s();
    if (times) {
      times[0] = t1 - t0;
      times[1] = t2 - t1;
      times[2] = t3 - t2;
      times[3] = cb.GPUEndTime - cb.GPUStartTime;
    }
    return cb.status == MTLCommandBufferStatusCompleted ? 0 : -2;
  }
}

}  // extern "C"
