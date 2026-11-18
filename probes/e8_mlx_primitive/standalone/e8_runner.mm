// E8 probe: standalone encoder for the e8_correlate1d_reflect_f32 kernel.
//
// Own MTLDevice / MTLCommandQueue / shared MTLBuffers, kernel loaded from our
// offline-compiled metallib.  Does NOT link libmlx.  Exposed as a C API so
// that Python can drive it through ctypes.
//
// Build: clang++ -std=c++17 -O2 -fobjc-arc -dynamiclib e8_runner.mm \
//          -framework Metal -framework Foundation -o libe8_runner.dylib

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <time.h>

namespace {

struct E8CorrelateParams {
  uint32_t rows;
  uint32_t n;
  uint32_t nw;
  int32_t shift;
};

struct Runner {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> pso = nil;
  id<MTLBuffer> in = nil;
  id<MTLBuffer> out = nil;
  id<MTLBuffer> weights = nil;
  uint32_t rows = 0, n = 0, nw = 0;
};

inline uint64_t now_ns() {
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

void set_err(char* err, int errlen, NSString* msg) {
  if (err && errlen > 0) {
    snprintf(err, (size_t)errlen, "%s", msg.UTF8String);
  }
}

// Shared by both encoders (the MLX primitive uses the same rule) so that the
// two hosts issue an identical dispatch.
MTLSize group_dims(id<MTLComputePipelineState> pso, uint32_t rows, uint32_t n) {
  NSUInteger max_threads = pso.maxTotalThreadsPerThreadgroup;
  NSUInteger gw = std::min<NSUInteger>(n, max_threads);
  NSUInteger gh =
      std::min<NSUInteger>(rows, std::max<NSUInteger>(1, max_threads / gw));
  return MTLSizeMake(gw, gh, 1);
}

int run_impl(
    Runner* r,
    id<MTLBuffer> in,
    uint64_t in_off,
    id<MTLBuffer> w,
    uint64_t w_off,
    id<MTLBuffer> out,
    uint64_t out_off,
    uint32_t rows,
    uint32_t n,
    uint32_t nw,
    uint64_t* t_ns) {
  @autoreleasepool {
    uint64_t t0 = now_ns();
    id<MTLCommandBuffer> cb = [r->queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:r->pso];
    [enc setBuffer:in offset:in_off atIndex:0];
    [enc setBuffer:out offset:out_off atIndex:1];
    [enc setBuffer:w offset:w_off atIndex:2];
    E8CorrelateParams p{rows, n, nw, (int32_t)(nw / 2)};
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, rows, 1)
        threadsPerThreadgroup:group_dims(r->pso, rows, n)];
    [enc endEncoding];
    uint64_t t1 = now_ns();
    [cb commit];
    uint64_t t2 = now_ns();
    [cb waitUntilCompleted];
    uint64_t t3 = now_ns();
    if (t_ns) {
      t_ns[0] = t1 - t0;  // create command buffer + encode
      t_ns[1] = t2 - t1;  // commit
      t_ns[2] = t3 - t2;  // wait until completed
      // Pure GPU execution time of the command buffer, as reported by Metal.
      t_ns[3] = (uint64_t)((cb.GPUEndTime - cb.GPUStartTime) * 1e9);
    }
    if (cb.status != MTLCommandBufferStatusCompleted) {
      return -1;
    }
  }
  return 0;
}

} // namespace

extern "C" {

__attribute__((visibility("default"))) void*
e8r_create(const char* metallib_path, char* err, int errlen) {
  @autoreleasepool {
    Runner* r = new Runner();
    r->device = MTLCreateSystemDefaultDevice();
    if (!r->device) {
      set_err(err, errlen, @"no Metal device");
      delete r;
      return nullptr;
    }
    r->queue = [r->device newCommandQueue];
    NSError* error = nil;
    NSURL* url = [NSURL fileURLWithPath:@(metallib_path)];
    r->library = [r->device newLibraryWithURL:url error:&error];
    if (!r->library) {
      set_err(err, errlen, error.localizedDescription);
      delete r;
      return nullptr;
    }
    id<MTLFunction> fn =
        [r->library newFunctionWithName:@"e8_correlate1d_reflect_f32"];
    if (!fn) {
      set_err(err, errlen, @"kernel function not found in metallib");
      delete r;
      return nullptr;
    }
    r->pso = [r->device newComputePipelineStateWithFunction:fn error:&error];
    if (!r->pso) {
      set_err(err, errlen, error.localizedDescription);
      delete r;
      return nullptr;
    }
    return r;
  }
}

__attribute__((visibility("default"))) void e8r_destroy(void* h) {
  delete static_cast<Runner*>(h);
}

// (Re)allocate shared buffers for a rows x n problem with nw weights.
// Lengths are rounded up to a whole number of pages so that the host memory
// can also be handed to other consumers that require page granularity.
__attribute__((visibility("default"))) int
e8r_prepare(void* h, uint32_t rows, uint32_t n, uint32_t nw) {
  @autoreleasepool {
    Runner* r = static_cast<Runner*>(h);
    const size_t page = (size_t)getpagesize();
    auto round_up = [page](size_t bytes) {
      return std::max(page, (bytes + page - 1) / page * page);
    };
    size_t data_bytes = round_up((size_t)rows * n * sizeof(float));
    size_t w_bytes = round_up((size_t)nw * sizeof(float));
    r->in = [r->device newBufferWithLength:data_bytes
                                   options:MTLResourceStorageModeShared];
    r->out = [r->device newBufferWithLength:data_bytes
                                    options:MTLResourceStorageModeShared];
    r->weights = [r->device newBufferWithLength:w_bytes
                                        options:MTLResourceStorageModeShared];
    if (!r->in || !r->out || !r->weights) {
      return -1;
    }
    r->rows = rows;
    r->n = n;
    r->nw = nw;
    return 0;
  }
}

static id<MTLBuffer> pick(Runner* r, int which) {
  return which == 0 ? r->in : (which == 1 ? r->out : r->weights);
}

// Host pointer (-[MTLBuffer contents]) of buffer 0=in, 1=out, 2=weights.
__attribute__((visibility("default"))) void* e8r_host_ptr(void* h, int which) {
  return pick(static_cast<Runner*>(h), which).contents;
}

__attribute__((visibility("default"))) uint64_t
e8r_buffer_length(void* h, int which) {
  return pick(static_cast<Runner*>(h), which).length;
}

// Unretained id<MTLBuffer> / id<MTLDevice> as opaque pointers (for hand-off
// probes; the runner keeps ownership).
__attribute__((visibility("default"))) void*
e8r_mtl_buffer(void* h, int which) {
  return (__bridge void*)pick(static_cast<Runner*>(h), which);
}

__attribute__((visibility("default"))) void* e8r_mtl_device(void* h) {
  return (__bridge void*)static_cast<Runner*>(h)->device;
}

// Encode + commit + waitUntilCompleted on the runner's own buffers.
// t_ns (optional, 4 entries): encode, commit, wait, GPU execution time in
// nanoseconds.
__attribute__((visibility("default"))) int e8r_run(void* h, uint64_t* t_ns) {
  Runner* r = static_cast<Runner*>(h);
  return run_impl(
      r, r->in, 0, r->weights, 0, r->out, 0, r->rows, r->n, r->nw, t_ns);
}

// Same, but on caller-provided MTLBuffers (e.g. buffers owned by MLX arrays).
__attribute__((visibility("default"))) int e8r_run_external(
    void* h,
    void* in_buf,
    uint64_t in_off,
    void* w_buf,
    uint64_t w_off,
    void* out_buf,
    uint64_t out_off,
    uint32_t rows,
    uint32_t n,
    uint32_t nw,
    uint64_t* t_ns) {
  Runner* r = static_cast<Runner*>(h);
  return run_impl(
      r,
      (__bridge id<MTLBuffer>)in_buf,
      in_off,
      (__bridge id<MTLBuffer>)w_buf,
      w_off,
      (__bridge id<MTLBuffer>)out_buf,
      out_off,
      rows,
      n,
      nw,
      t_ns);
}

} // extern "C"
