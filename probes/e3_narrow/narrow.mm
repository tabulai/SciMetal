// E3 probe: float64 <-> float32 conversion cost.
//   (a) CPU: vDSP_vdpsp / vDSP_vspdp chunked over the performance cores (GCD).
//   (b) GPU: integer-only MSL kernels that read / write float64 bit patterns as ulong.
// Plain C entry points, driven from Python via ctypes.
#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#include <unistd.h>

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_narrow, g_widen, g_narrow_f, g_widen_f;
static char g_err[2048];
static int g_pcores = 8;

struct Params {
    uint64_t n;
    uint32_t width;
    uint32_t chunk;
};

static double now_s() {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * (double)tb.numer / (double)tb.denom * 1e-9;
}

static id<MTLComputePipelineState> make_pso(id<MTLLibrary> lib, NSString* name) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        snprintf(g_err, sizeof g_err, "function %s not found", name.UTF8String);
        return nil;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [g_dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) snprintf(g_err, sizeof g_err, "pipeline %s: %s", name.UTF8String, err.localizedDescription.UTF8String);
    return pso;
}

// Run fn(chunk_index) for chunk_index in [0, nchunks) on at most `workers` GCD threads at
// user-initiated QoS.  Workers pull chunks from a shared counter, so one slow / preempted
// thread does not stall the rest.
template <typename F>
static void parallel_chunks(size_t nchunks, int workers, F fn) {
    if (nchunks == 0) return;
    if (workers > (int)nchunks) workers = (int)nchunks;
    if (workers <= 1) {
        for (size_t i = 0; i < nchunks; ++i) fn(i);
        return;
    }
    std::atomic<size_t> next{0};
    std::atomic<size_t>* nextp = &next;
    F* fnp = &fn;
    dispatch_apply((size_t)workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t) {
        for (;;) {
            const size_t i = nextp->fetch_add(1, std::memory_order_relaxed);
            if (i >= nchunks) break;
            (*fnp)(i);
        }
    });
}

extern "C" {

const char* e3_last_error() { return g_err; }
int e3_perf_cores() { return g_pcores; }

int e3_init(const char* metallib_path) {
    @autoreleasepool {
        int n = 0;
        size_t sz = sizeof n;
        if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &sz, nullptr, 0) == 0 && n > 0) g_pcores = n;
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) {
            snprintf(g_err, sizeof g_err, "no Metal device");
            return -1;
        }
        g_queue = [g_dev newCommandQueue];
        NSError* err = nil;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path]];
        id<MTLLibrary> lib = [g_dev newLibraryWithURL:url error:&err];
        if (!lib) {
            snprintf(g_err, sizeof g_err, "metallib: %s", err.localizedDescription.UTF8String);
            return -2;
        }
        g_narrow = make_pso(lib, @"narrow_f64_to_f32");
        g_widen = make_pso(lib, @"widen_f32_to_f64");
        g_narrow_f = make_pso(lib, @"narrow_through_float");
        g_widen_f = make_pso(lib, @"widen_from_float");
        if (!g_narrow || !g_widen || !g_narrow_f || !g_widen_f) return -3;
        return 0;
    }
}

// ---- CPU -------------------------------------------------------------------------------

// chunk_elems: elements per work item (0 -> n / workers, one chunk per worker).
void e3_cpu_narrow(const double* src, float* dst, uint64_t n, int workers, uint64_t chunk_elems) {
    if (workers <= 0) workers = g_pcores;
    if (!chunk_elems) chunk_elems = (n + (uint64_t)workers - 1) / (uint64_t)workers;
    const size_t nchunks = (size_t)((n + chunk_elems - 1) / chunk_elems);
    parallel_chunks(nchunks, workers, [=](size_t c) {
        const uint64_t i0 = (uint64_t)c * chunk_elems;
        const uint64_t m = i0 + chunk_elems <= n ? chunk_elems : n - i0;
        vDSP_vdpsp(src + i0, 1, dst + i0, 1, (vDSP_Length)m);
    });
}

void e3_cpu_widen(const float* src, double* dst, uint64_t n, int workers, uint64_t chunk_elems) {
    if (workers <= 0) workers = g_pcores;
    if (!chunk_elems) chunk_elems = (n + (uint64_t)workers - 1) / (uint64_t)workers;
    const size_t nchunks = (size_t)((n + chunk_elems - 1) / chunk_elems);
    parallel_chunks(nchunks, workers, [=](size_t c) {
        const uint64_t i0 = (uint64_t)c * chunk_elems;
        const uint64_t m = i0 + chunk_elems <= n ? chunk_elems : n - i0;
        vDSP_vspdp(src + i0, 1, dst + i0, 1, (vDSP_Length)m);
    });
}

// Plain C++ loops (what the compiler makes of a cast), same threading, for comparison.
void e3_cpu_narrow_loop(const double* __restrict src, float* __restrict dst, uint64_t n, int workers,
                        uint64_t chunk_elems) {
    if (workers <= 0) workers = g_pcores;
    if (!chunk_elems) chunk_elems = (n + (uint64_t)workers - 1) / (uint64_t)workers;
    const size_t nchunks = (size_t)((n + chunk_elems - 1) / chunk_elems);
    parallel_chunks(nchunks, workers, [=](size_t c) {
        const uint64_t i0 = (uint64_t)c * chunk_elems;
        const uint64_t i1 = i0 + chunk_elems <= n ? i0 + chunk_elems : n;
        for (uint64_t i = i0; i < i1; ++i) dst[i] = (float)src[i];
    });
}

void e3_cpu_widen_loop(const float* __restrict src, double* __restrict dst, uint64_t n, int workers,
                       uint64_t chunk_elems) {
    if (workers <= 0) workers = g_pcores;
    if (!chunk_elems) chunk_elems = (n + (uint64_t)workers - 1) / (uint64_t)workers;
    const size_t nchunks = (size_t)((n + chunk_elems - 1) / chunk_elems);
    parallel_chunks(nchunks, workers, [=](size_t c) {
        const uint64_t i0 = (uint64_t)c * chunk_elems;
        const uint64_t i1 = i0 + chunk_elems <= n ? i0 + chunk_elems : n;
        for (uint64_t i = i0; i < i1; ++i) dst[i] = (double)src[i];
    });
}

// reps timed calls of one CPU variant.  which: 0 vDSP narrow, 1 vDSP widen, 2 loop narrow, 3 loop widen
void e3_time_cpu(int which, const void* src, void* dst, uint64_t n, int workers, uint64_t chunk_elems, int reps,
                 double* t) {
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_s();
        switch (which) {
            case 0: e3_cpu_narrow((const double*)src, (float*)dst, n, workers, chunk_elems); break;
            case 1: e3_cpu_widen((const float*)src, (double*)dst, n, workers, chunk_elems); break;
            case 2: e3_cpu_narrow_loop((const double*)src, (float*)dst, n, workers, chunk_elems); break;
            default: e3_cpu_widen_loop((const float*)src, (double*)dst, n, workers, chunk_elems); break;
        }
        t[r] = now_s() - t0;
    }
}

// ---- GPU -------------------------------------------------------------------------------

void* e3_wrap(uint64_t addr, uint64_t len) {
    @autoreleasepool {
        id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void*)addr
                                                   length:(NSUInteger)len
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        return b ? (__bridge_retained void*)b : nullptr;
    }
}

void* e3_alloc(uint64_t len) {
    @autoreleasepool {
        id<MTLBuffer> b = [g_dev newBufferWithLength:(NSUInteger)len options:MTLResourceStorageModeShared];
        return b ? (__bridge_retained void*)b : nullptr;
    }
}

uint64_t e3_contents(void* h) { return (uint64_t)((__bridge id<MTLBuffer>)h).contents; }

void e3_release(void* h) {
    @autoreleasepool {
        id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)h;
        (void)b;
    }
}

// which: 0 narrow (integer only), 1 widen (integer only), 2 narrow through float, 3 widen from float
// chunk: elements per GPU thread; tg_w/tg_h: threadgroup shape (0 -> execution width x 1).
// Times encode + commit + waitUntilCompleted.
static int gpu_run(int which, id<MTLBuffer> src, uint64_t src_off, id<MTLBuffer> dst, uint64_t dst_off, uint64_t n,
                   uint32_t chunk, uint32_t tg_w, uint32_t tg_h, double* secs) {
    @autoreleasepool {
        id<MTLComputePipelineState> pso = which == 0 ? g_narrow : which == 1 ? g_widen : which == 2 ? g_narrow_f : g_widen_f;
        if (!chunk) chunk = 1;
        const uint64_t threads = (n + chunk - 1) / chunk;
        Params p{};
        p.n = n;
        p.chunk = chunk;
        p.width = threads < 16384 ? (uint32_t)(threads ? threads : 1) : 16384u;
        const uint64_t rows = (threads + p.width - 1) / p.width;
        if (!tg_w) tg_w = (uint32_t)pso.threadExecutionWidth;
        if (!tg_h) tg_h = 1;
        // maxTotalThreadsPerThreadgroup is per pipeline (and shrinks under shader validation): clamp.
        const uint32_t limit = (uint32_t)pso.maxTotalThreadsPerThreadgroup;
        if (tg_w > limit) tg_w = limit;
        while (tg_h > 1 && tg_w * tg_h > limit) tg_h /= 2;
        const double t0 = now_s();
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:src offset:(NSUInteger)src_off atIndex:0];
        [enc setBuffer:dst offset:(NSUInteger)dst_off atIndex:1];
        [enc setBytes:&p length:sizeof p atIndex:2];
        [enc dispatchThreads:MTLSizeMake(p.width, (NSUInteger)rows, 1) threadsPerThreadgroup:MTLSizeMake(tg_w, tg_h, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (secs) *secs = now_s() - t0;
        if (cb.status != MTLCommandBufferStatusCompleted) {
            snprintf(g_err, sizeof g_err, "command buffer failed: %s",
                     cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
            return -1;
        }
        return 0;
    }
}

int e3_gpu_convert(int which, void* hsrc, uint64_t src_off, void* hdst, uint64_t dst_off, uint64_t n, uint32_t chunk,
                   uint32_t tg_w, uint32_t tg_h, double* secs) {
    return gpu_run(which, (__bridge id<MTLBuffer>)hsrc, src_off, (__bridge id<MTLBuffer>)hdst, dst_off, n, chunk, tg_w,
                   tg_h, secs);
}

// reps timed conversions on long-lived (warm) buffers.
int e3_time_gpu(int which, void* hsrc, void* hdst, uint64_t n, uint32_t chunk, uint32_t tg_w, uint32_t tg_h, int reps,
                double* t) {
    for (int r = 0; r < reps; ++r) {
        const int rc = gpu_run(which, (__bridge id<MTLBuffer>)hsrc, 0, (__bridge id<MTLBuffer>)hdst, 0, n, chunk, tg_w,
                               tg_h, &t[r]);
        if (rc != 0) return rc;
    }
    return 0;
}

// reps x { wrap src ; wrap dst ; convert ; sync ; release both }: the cost when both ends are
// NumPy arrays that have never been seen by Metal (wiring of both wraps included).
int e3_time_gpu_cold(int which, uint64_t src_addr, uint64_t src_len, uint64_t dst_addr, uint64_t dst_len, uint64_t n,
                     uint32_t chunk, uint32_t tg_w, uint32_t tg_h, int reps, double* t) {
    for (int r = 0; r < reps; ++r) {
        @autoreleasepool {
            const double t0 = now_s();
            id<MTLBuffer> s = [g_dev newBufferWithBytesNoCopy:(void*)src_addr length:(NSUInteger)src_len options:MTLResourceStorageModeShared deallocator:nil];
            id<MTLBuffer> d = [g_dev newBufferWithBytesNoCopy:(void*)dst_addr length:(NSUInteger)dst_len options:MTLResourceStorageModeShared deallocator:nil];
            if (!s || !d) return -3;
            const int rc = gpu_run(which, s, 0, d, 0, n, chunk, tg_w, tg_h, nullptr);
            if (rc != 0) return rc;
            s = nil;
            d = nil;
            t[r] = now_s() - t0;
        }
    }
    return 0;
}

}  // extern "C"
