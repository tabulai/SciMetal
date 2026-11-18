// E4 probe: separable Gaussian filter (scipy.ndimage.gaussian_filter semantics, mode='reflect').
//   * GPU float32 path: two 1-D correlate passes in one command buffer, with several ways of
//     getting float64 NumPy memory in and out (no-copy wrap + fused in-kernel conversion,
//     separate GPU conversion pass, CPU vDSP conversion, pooled shared output buffer).
//   * CPU "strict" float64 path: the same separable passes as vectorisable loops over the P-cores.
// Plain C entry points, driven from Python via ctypes.
#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/mach_vm.h>
#include <mutex>
#include <sys/sysctl.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
enum { K_F32_F32, K_F64_F32, K_F32_F64, K_F64FAST_F32, K_SPAN_F32_F32, K_SPAN_F64_F32, K_SPAN_F32_F64, K_NARROW, K_WIDEN, K_COUNT };
static id<MTLComputePipelineState> g_pso[K_COUNT];
static char g_err[2048];
static int g_pcores = 8;
static uint64_t g_page = 16384;

struct Params {
    uint32_t W, H, radius, axis, span, pad0, pad1, pad2;
};

static double now_s() {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * (double)tb.numer / (double)tb.denom * 1e-9;
}

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

// ---- buffer pool (shared-storage MTLBuffers, recycled warm) ------------------------------------
static std::mutex g_pool_mu;
static std::unordered_map<uint64_t, std::vector<id<MTLBuffer>>> g_pool;
static long g_pool_fresh = 0;

static id<MTLBuffer> pool_get(uint64_t len) {
    {
        std::lock_guard<std::mutex> lk(g_pool_mu);
        auto it = g_pool.find(len);
        if (it != g_pool.end() && !it->second.empty()) {
            id<MTLBuffer> b = it->second.back();
            it->second.pop_back();
            return b;
        }
        ++g_pool_fresh;
    }
    return [g_dev newBufferWithLength:(NSUInteger)len options:MTLResourceStorageModeShared];
}

static void pool_put(id<MTLBuffer> b) {
    if (!b) return;
    std::lock_guard<std::mutex> lk(g_pool_mu);
    g_pool[(uint64_t)b.length].push_back(b);
}

// ---- no-copy wrap of a host range (recipe from probe E2) ---------------------------------------
// Enclosing page span must be mapped, readable (+ writable when need_write) in consecutive VM entries.
static bool span_ok(uint64_t first, uint64_t last, bool need_write) {
    uint64_t cur = first;
    while (cur < last) {
        mach_vm_address_t region = (mach_vm_address_t)cur;
        mach_vm_size_t size = 0;
        natural_t depth = UINT32_MAX;
        vm_region_submap_info_data_64_t info{};
        mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
        if (mach_vm_region_recurse(mach_task_self(), &region, &size, &depth, (vm_region_recurse_info_t)&info, &count) !=
            KERN_SUCCESS)
            return false;
        if (region > cur || info.is_submap || !(info.protection & VM_PROT_READ) ||
            (need_write && !(info.protection & VM_PROT_WRITE)))
            return false;
        cur = region + size;
    }
    return true;
}

static id<MTLBuffer> wrap_span(uint64_t addr, uint64_t len, bool need_write, uint64_t* offset) {
    const uint64_t first = addr & ~(g_page - 1);
    const uint64_t last = (addr + len + g_page - 1) & ~(g_page - 1);
    if (!span_ok(first, last, need_write)) return nil;
    *offset = addr - first;
    return [g_dev newBufferWithBytesNoCopy:(void*)first
                                    length:(NSUInteger)(last - first)
                                   options:MTLResourceStorageModeShared
                               deallocator:nil];
}

static id<MTLBuffer> g_cached_wrap;
static uint64_t g_cached_addr = 0, g_cached_len = 0, g_cached_off = 0;

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

// ---- CPU conversions (vDSP over the P-cores, 2 MiB-ish dynamic chunks) -------------------------
static void cpu_narrow(const double* src, float* dst, uint64_t n) {
    const uint64_t ce = 1u << 18;
    parallel_chunks((size_t)((n + ce - 1) / ce), g_pcores, [=](size_t c) {
        const uint64_t i0 = (uint64_t)c * ce;
        vDSP_vdpsp(src + i0, 1, dst + i0, 1, (vDSP_Length)(i0 + ce <= n ? ce : n - i0));
    });
}
static void cpu_widen(const float* src, double* dst, uint64_t n) {
    const uint64_t ce = 1u << 18;
    parallel_chunks((size_t)((n + ce - 1) / ce), g_pcores, [=](size_t c) {
        const uint64_t i0 = (uint64_t)c * ce;
        vDSP_vspdp(src + i0, 1, dst + i0, 1, (vDSP_Length)(i0 + ce <= n ? ce : n - i0));
    });
}
static void cpu_copy(const void* src, void* dst, uint64_t nbytes) {
    const uint64_t cb = 1u << 21;
    parallel_chunks((size_t)((nbytes + cb - 1) / cb), g_pcores, [=](size_t c) {
        const uint64_t o = (uint64_t)c * cb;
        memcpy((char*)dst + o, (const char*)src + o, (size_t)(o + cb <= nbytes ? cb : nbytes - o));
    });
}

// ---- CPU separable correlate (strict float64, also instantiated for float32) -------------------
static inline int reflect_idx(int i, int n) {
    const int period = 2 * n;
    i %= period;
    if (i < 0) i += period;
    return i >= n ? period - 1 - i : i;
}

// How `acc += (a + b) * w` is evaluated.  SciPy's summation ORDER is fixed (centre tap, then
// pairs from the outermost inwards) but whether each step is fused depends on how SciPy's C was
// compiled, so it is a parameter here:
//   FMA_OFF         every step is a separate multiply and add (portable IEEE result)
//   FMA_ON          every step is a fused multiply-add
//   FMA_SCIPY_BUILD what the installed SciPy 1.18.1 arm64 wheel was observed to do: its inner
//                   loop is auto-vectorised 8 taps at a time with unfused mul+add, and the
//                   scalar remainder (r mod 8 innermost pairs; all pairs when r < 10) is fused.
enum { FMA_OFF = 0, FMA_ON = 1, FMA_SCIPY_BUILD = 2 };

static inline double fma_t(double a, double b, double c) { return __builtin_fma(a, b, c); }
static inline float fma_t(float a, float b, float c) { return __builtin_fmaf(a, b, c); }

// acc[c] = step(acc[c], (a_k[c] + b_k[c]), w_k) for k = 0..nt-1, in that order for every c;
// vectorised across c (the outputs are independent), never across k.
template <typename T>
static inline void sweep_mul(T* __restrict acc, const T* const* a, const T* const* b, const T* w, int nt, size_t n) {
#pragma clang fp contract(off)
    if (nt == 4) {
        const T *a0 = a[0], *a1 = a[1], *a2 = a[2], *a3 = a[3], *b0 = b[0], *b1 = b[1], *b2 = b[2], *b3 = b[3];
        const T w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
#pragma clang loop vectorize(enable) interleave(enable)
        for (size_t c = 0; c < n; ++c) {
            T s = acc[c];
            s = s + (a0[c] + b0[c]) * w0;
            s = s + (a1[c] + b1[c]) * w1;
            s = s + (a2[c] + b2[c]) * w2;
            s = s + (a3[c] + b3[c]) * w3;
            acc[c] = s;
        }
    } else {
        for (int k = 0; k < nt; ++k) {
            const T *ak = a[k], *bk = b[k];
            const T wk = w[k];
#pragma clang loop vectorize(enable) interleave(enable)
            for (size_t c = 0; c < n; ++c) acc[c] = acc[c] + (ak[c] + bk[c]) * wk;
        }
    }
}

template <typename T>
static inline void sweep_fma(T* __restrict acc, const T* const* a, const T* const* b, const T* w, int nt, size_t n) {
    if (nt == 4) {
        const T *a0 = a[0], *a1 = a[1], *a2 = a[2], *a3 = a[3], *b0 = b[0], *b1 = b[1], *b2 = b[2], *b3 = b[3];
        const T w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
#pragma clang loop vectorize(enable) interleave(enable)
        for (size_t c = 0; c < n; ++c) {
            T s = acc[c];
            s = fma_t(a0[c] + b0[c], w0, s);
            s = fma_t(a1[c] + b1[c], w1, s);
            s = fma_t(a2[c] + b2[c], w2, s);
            s = fma_t(a3[c] + b3[c], w3, s);
            acc[c] = s;
        }
    } else {
        for (int k = 0; k < nt; ++k) {
            const T *ak = a[k], *bk = b[k];
            const T wk = w[k];
#pragma clang loop vectorize(enable) interleave(enable)
            for (size_t c = 0; c < n; ++c) acc[c] = fma_t(ak[c] + bk[c], wk, acc[c]);
        }
    }
}

static int g_prefetch_rows = 8;

static inline int unfused_taps(int r, int fma_mode) {
    if (fma_mode == FMA_OFF) return r;
    if (fma_mode == FMA_ON) return 0;
    return r >= 10 ? (r / 8) * 8 : 0;
}

// A[k], B[k]: the two source lines at distance d = r - k (k = 0 .. r-1); wt[k] their weight.
template <typename T>
static inline void run_taps(T* __restrict acc, size_t n, const T* const* A, const T* const* B, const T* wt, int r,
                            int n_unfused) {
    int k = 0;
    while (k < n_unfused) {
        const int nt = n_unfused - k < 4 ? n_unfused - k : 4;
        sweep_mul<T>(acc, A + k, B + k, wt + k, nt, n);
        k += nt;
    }
    while (k < r) {
        const int nt = r - k < 4 ? r - k : 4;
        sweep_fma<T>(acc, A + k, B + k, wt + k, nt, n);
        k += nt;
    }
}

// wc points at the centre weight; wc[-d] is the weight at distance d (kernel is symmetric).
//
// Axis 0 (vertical).  Work items are column tiles; inside a tile we walk down the rows.  The
// (2r+1) source row segments a row needs are 8*W bytes apart in the image: for power-of-two
// widths they all land in the same L1 sets and evict each other.  So each tile keeps its sliding
// window in a small contiguous ring buffer (one copy of the input, L1-resident, no aliasing).
template <typename T>
static void corr_axis0(const T* __restrict in, T* __restrict out, int H, int W, const T* wc, int r, int workers, int tile,
                       int fma_mode, int use_ring) {
    const int K = 2 * r + 1;
    if (tile <= 0) {  // auto: window of K segments ~ 64 KiB
        tile = (int)(65536 / ((size_t)K * sizeof(T))) & ~7;
        if (tile < 32) tile = 32;
        if (tile > 2048) tile = 2048;
    }
    if (H < K) use_ring = 0;
    const size_t ntiles = (size_t)((W + tile - 1) / tile);
    const int n_unfused = unfused_taps(r, fma_mode);
    parallel_chunks(ntiles, workers, [=](size_t t) {
        const int c0 = (int)t * tile;
        const size_t n = (size_t)((c0 + tile <= W ? tile : W - c0));
        const size_t pitch = (n + 7) & ~(size_t)7;
        std::vector<T> ringv(use_ring ? (size_t)K * pitch + 16 : 16);
        T* ring = ringv.data();
        std::vector<const T*> Av((size_t)r + 1), Bv((size_t)r + 1);
        std::vector<T> wtv((size_t)r + 1);
        const T** A = Av.data();
        const T** B = Bv.data();
        T* wt = wtv.data();
        for (int k = 0; k < r; ++k) wt[k] = wc[-(r - k)];
        int loaded = -1;
        auto line = [&](int row) -> const T* {
            return use_ring ? ring + (size_t)(row % K) * pitch : in + (size_t)row * W + c0;
        };
        const int pf = g_prefetch_rows;  // rows ahead to software-prefetch (strided reads/writes defeat the HW prefetcher)
        for (int y = 0; y < H; ++y) {
            if (use_ring) {
                const int need = y + r < H ? y + r : H - 1;
                while (loaded < need) {
                    ++loaded;
                    memcpy(ring + (size_t)(loaded % K) * pitch, in + (size_t)loaded * W + c0, n * sizeof(T));
                    if (pf > 0 && loaded + pf < H) {
                        const char* p = (const char*)(in + (size_t)(loaded + pf) * W + c0);
                        for (size_t o = 0; o < n * sizeof(T); o += 64) __builtin_prefetch(p + o, 0, 3);
                    }
                }
            }
            if (pf > 0 && y + pf < H) {
                const char* p = (const char*)(out + (size_t)(y + pf) * W + c0);
                for (size_t o = 0; o < n * sizeof(T); o += 64) __builtin_prefetch(p + o, 1, 3);
            }
            T* __restrict acc = out + (size_t)y * W + c0;
            const T* ctr = line(y);
            const T w0 = wc[0];
#pragma clang loop vectorize(enable) interleave(enable)
            for (size_t c = 0; c < n; ++c) acc[c] = ctr[c] * w0;
            for (int k = 0; k < r; ++k) {
                const int d = r - k;
                A[k] = line(reflect_idx(y - d, H));
                B[k] = line(reflect_idx(y + d, H));
            }
            run_taps<T>(acc, n, A, B, wt, r, n_unfused);
        }
    });
}

// Axis 1 (horizontal): work items are blocks of rows; each row is reflect-padded into a small
// line buffer (as SciPy does) and correlated with contiguous, L1-resident sweeps.
template <typename T>
static void corr_axis1(const T* __restrict in, T* __restrict out, int H, int W, const T* wc, int r, int workers,
                       int rows_per_item, int fma_mode) {
    const size_t nitems = (size_t)((H + rows_per_item - 1) / rows_per_item);
    const int n_unfused = unfused_taps(r, fma_mode);
    parallel_chunks(nitems, workers, [=](size_t it) {
        std::vector<T> padv((size_t)W + 2 * (size_t)r + 16);
        T* pad = padv.data();
        std::vector<const T*> Av((size_t)r + 1), Bv((size_t)r + 1);
        std::vector<T> wtv((size_t)r + 1);
        const T** A = Av.data();
        const T** B = Bv.data();
        T* wt = wtv.data();
        const T* ctr = pad + r;
        for (int k = 0; k < r; ++k) {
            wt[k] = wc[-(r - k)];
            A[k] = ctr - (r - k);
            B[k] = ctr + (r - k);
        }
        const int y0 = (int)it * rows_per_item;
        const int y1 = y0 + rows_per_item <= H ? y0 + rows_per_item : H;
        for (int y = y0; y < y1; ++y) {
            const T* row = in + (size_t)y * W;
            for (int k = 0; k < r; ++k) pad[k] = row[reflect_idx(k - r, W)];
            memcpy(pad + r, row, (size_t)W * sizeof(T));
            for (int k = 0; k < r; ++k) pad[r + W + k] = row[reflect_idx(W + k, W)];
            T* __restrict acc = out + (size_t)y * W;
            const T w0 = wc[0];
#pragma clang loop vectorize(enable) interleave(enable)
            for (int x = 0; x < W; ++x) acc[x] = ctr[x] * w0;
            run_taps<T>(acc, (size_t)W, A, B, wt, r, n_unfused);
        }
    });
}

static void* g_scratch = nullptr;
static size_t g_scratch_bytes = 0;
static void* scratch(size_t bytes) {
    if (bytes > g_scratch_bytes) {
        free(g_scratch);
        g_scratch = nullptr;
        if (posix_memalign(&g_scratch, 16384, bytes) != 0) return nullptr;
        memset(g_scratch, 0, bytes);
        g_scratch_bytes = bytes;
    }
    return g_scratch;
}

extern "C" {

const char* e4_last_error() { return g_err; }
int e4_perf_cores() { return g_pcores; }
long e4_pool_fresh_allocations() { return g_pool_fresh; }

int e4_init(const char* metallib_path) {
    @autoreleasepool {
        int n = 0;
        size_t sz = sizeof n;
        if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &sz, nullptr, 0) == 0 && n > 0) g_pcores = n;
        g_page = (uint64_t)sysconf(_SC_PAGESIZE);
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
        NSArray<NSString*>* names = @[
            @"corr_f32_f32", @"corr_f64_f32", @"corr_f32_f64", @"corr_f64fast_f32", @"corr_span_f32_f32",
            @"corr_span_f64_f32", @"corr_span_f32_f64", @"narrow_pass", @"widen_pass"
        ];
        for (int i = 0; i < K_COUNT; ++i) {
            g_pso[i] = make_pso(lib, names[(NSUInteger)i]);
            if (!g_pso[i]) return -3;
        }
        return 0;
    }
}

// kernel index: 0 f32->f32, 1 f64->f32, 2 f32->f64, 3 f64(cheap)->f32, 4..6 sliding-window variants, 7 narrow, 8 widen
int e4_max_threads_per_group(int k) { return k >= 0 && k < K_COUNT ? (int)g_pso[k].maxTotalThreadsPerThreadgroup : -1; }
int e4_thread_execution_width(int k) { return k >= 0 && k < K_COUNT ? (int)g_pso[k].threadExecutionWidth : -1; }

struct E4Job {
    uint64_t in_addr;    // host pointer of the input image (row-major H x W)
    uint64_t out_addr;   // host pointer of the output image, or 0 when out_mode == 1
    uint32_t H, W, radius;
    uint32_t in_f64;     // 0: float32 input, 1: float64 input
    uint32_t out_f64;    // 0: float32 output, 1: float64 output
    // in_mode : 0 wrap no-copy (float64: fused narrowing inside pass 0)
    //           1 CPU into a pooled float32 buffer (float64: vDSP narrow, float32: parallel memcpy)
    //           2 wrap no-copy + separate GPU narrow pass (float64 only)
    uint32_t in_mode;
    // out_mode: 0 wrap out_addr no-copy (float64: fused widening inside pass 1)
    //           1 pooled shared MTLBuffer of the final dtype, handed back to the caller (zero copy)
    //           2 GPU -> pooled float32 buffer, then CPU into out_addr (vDSP widen / memcpy)
    //           3 wrap out_addr no-copy + separate GPU widen pass (float64 only)
    uint32_t out_mode;
    uint32_t tg_w0, tg_h0;  // threadgroup shape, pass along axis 0
    uint32_t tg_w1, tg_h1;  // threadgroup shape, pass along axis 1
    uint32_t span;          // <=1: one output per thread; >1: sliding-window kernel, outputs per thread
    uint32_t split;         // 1: one command buffer per stage (per-stage GPU times; slower end to end)
    uint32_t fast_narrow;   // fused float64 read uses narrow_fast instead of the exact narrowing
    uint32_t cache_in_wrap; // keep the input wrap alive and reuse it when the same range comes back
    uint32_t spin_wait;     // 1: poll the command buffer status instead of sleeping in waitUntilCompleted
};

// bd[0] wrap input      bd[1] CPU input conversion/copy   bd[2] output wrap / pool checkout
// bd[3] encode + commit bd[4] wait for completion (sync)  bd[5] CPU output conversion/copy
// bd[6] release         bd[7] GPU execution time (GPUEndTime - GPUStartTime, summed)
// bd[8] total wall      bd[9..12] per-stage GPU time when split: narrow, pass0, pass1, widen
// bd[13] kernel time (kernelEndTime - kernelStartTime, summed)
int e4_gpu_gauss(const E4Job* j, const float* w32, double* bd, void** out_handle) {
    @autoreleasepool {
        for (int i = 0; i < 16; ++i) bd[i] = 0;
        const double t_begin = now_s();
        const uint64_t npix = (uint64_t)j->H * j->W;
        const uint64_t in_bytes = npix * (j->in_f64 ? 8 : 4);
        const uint64_t out_bytes = npix * (j->out_f64 ? 8 : 4);
        const uint64_t f32_bytes = (npix * 4 + g_page - 1) & ~(g_page - 1);
        const bool spanv = j->span > 1;
        if (spanv && j->span > 8) {
            snprintf(g_err, sizeof g_err, "span too large");
            return -10;
        }
        if (spanv && j->radius > 64) {
            snprintf(g_err, sizeof g_err, "radius too large for the span kernel");
            return -10;
        }

        // ---------------- input ----------------
        id<MTLBuffer> src = nil;      // what pass 0 reads
        uint64_t src_off = 0;
        bool src_is_f64 = false;
        id<MTLBuffer> in_wrap = nil;  // wrapped float64/float32 host memory
        uint64_t in_wrap_off = 0;
        id<MTLBuffer> bufA = nil;     // pooled float32 copy / narrowed input
        double t0 = now_s();
        if (j->in_mode == 0 || j->in_mode == 2) {
            if (j->cache_in_wrap && g_cached_wrap && g_cached_addr == j->in_addr && g_cached_len == in_bytes) {
                in_wrap = g_cached_wrap;
                in_wrap_off = g_cached_off;
            } else {
                in_wrap = wrap_span(j->in_addr, in_bytes, false, &in_wrap_off);
                if (!in_wrap) {
                    snprintf(g_err, sizeof g_err, "input cannot be wrapped without a copy");
                    return -4;
                }
                if (j->cache_in_wrap) {
                    g_cached_wrap = in_wrap;
                    g_cached_addr = j->in_addr;
                    g_cached_len = in_bytes;
                    g_cached_off = in_wrap_off;
                }
            }
            bd[0] = now_s() - t0;
            if (j->in_mode == 2 && j->in_f64) {
                bufA = pool_get(f32_bytes);
                src = bufA;
            } else {
                src = in_wrap;
                src_off = in_wrap_off;
                src_is_f64 = j->in_f64 != 0;
            }
        } else {
            bufA = pool_get(f32_bytes);
            if (j->in_f64)
                cpu_narrow((const double*)j->in_addr, (float*)bufA.contents, npix);
            else
                cpu_copy((const void*)j->in_addr, bufA.contents, npix * 4);
            bd[1] = now_s() - t0;
            src = bufA;
        }
        if (!src) return -5;

        // ---------------- output ----------------
        t0 = now_s();
        id<MTLBuffer> dst = nil;       // what pass 1 writes
        uint64_t dst_off = 0;
        bool dst_is_f64 = false;
        id<MTLBuffer> out_wrap = nil;
        uint64_t out_wrap_off = 0;
        id<MTLBuffer> bufC = nil;      // pooled float32 result (modes 2, 3)
        id<MTLBuffer> out_pooled = nil;
        if (j->out_mode == 0 || j->out_mode == 3) {
            out_wrap = wrap_span(j->out_addr, out_bytes, true, &out_wrap_off);
            if (!out_wrap) {
                snprintf(g_err, sizeof g_err, "output cannot be wrapped without a copy");
                return -6;
            }
            if (j->out_mode == 3 && j->out_f64) {
                bufC = pool_get(f32_bytes);
                dst = bufC;
            } else {
                dst = out_wrap;
                dst_off = out_wrap_off;
                dst_is_f64 = j->out_f64 != 0;
            }
        } else if (j->out_mode == 1) {
            out_pooled = pool_get((out_bytes + g_page - 1) & ~(g_page - 1));
            dst = out_pooled;
            dst_is_f64 = j->out_f64 != 0;
        } else {
            bufC = pool_get(f32_bytes);
            dst = bufC;
        }
        id<MTLBuffer> bufB = pool_get(f32_bytes);  // intermediate between the two passes
        bd[2] = now_s() - t0;
        if (!dst || !bufB) return -7;

        // ---------------- encode ----------------
        t0 = now_s();
        const int r = (int)j->radius;
        Params p0{j->W, j->H, j->radius, 0, spanv ? j->span : 1, 0, 0, 0};
        Params p1 = p0;
        p1.axis = 1;
        id<MTLComputePipelineState> pso0 = spanv ? g_pso[src_is_f64 ? K_SPAN_F64_F32 : K_SPAN_F32_F32]
                                                : g_pso[src_is_f64 ? (j->fast_narrow ? K_F64FAST_F32 : K_F64_F32) : K_F32_F32];
        id<MTLComputePipelineState> pso1 = spanv ? g_pso[dst_is_f64 ? K_SPAN_F32_F64 : K_SPAN_F32_F32]
                                                : g_pso[dst_is_f64 ? K_F32_F64 : K_F32_F32];
        const bool do_narrow = j->in_mode == 2 && j->in_f64;
        const bool do_widen = j->out_mode == 3 && j->out_f64;
        const NSUInteger wlen = (NSUInteger)(2 * r + 1) * sizeof(float);

        NSMutableArray<id<MTLCommandBuffer>>* cbs = [NSMutableArray array];
        id<MTLCommandBuffer> cb = nil;
        id<MTLComputeCommandEncoder> enc = nil;
        int stage_cb[4] = {-1, -1, -1, -1};
        auto begin_stage = [&](int stage) {
            if (!cb || j->split) {
                if (enc) {
                    [enc endEncoding];
                    [cb commit];
                }
                cb = [g_queue commandBuffer];
                [cbs addObject:cb];
                enc = [cb computeCommandEncoder];
            }
            stage_cb[stage] = (int)cbs.count - 1;
        };
        // Requested threadgroup shape, clamped to the grid and to THIS pipeline's limit
        // (maxTotalThreadsPerThreadgroup differs per kernel: 1024 for the plain ones, less for
        // kernels with more live state; exceeding it is a validation error).
        auto tg = [&](id<MTLComputePipelineState> pso, uint32_t w, uint32_t h, NSUInteger gw, NSUInteger gh) {
            const uint32_t limit = (uint32_t)pso.maxTotalThreadsPerThreadgroup;
            if (!w) w = (uint32_t)pso.threadExecutionWidth;
            if (w > gw) w = (uint32_t)gw;
            if (w > limit) w = limit;
            if (!h) h = limit / w;
            if (h > gh) h = (uint32_t)gh;
            while (h > 1 && w * h > limit) h /= 2;
            if (w * h > limit) h = 1;
            return MTLSizeMake(w, h, 1);
        };
        auto tg_flat = [&](id<MTLComputePipelineState> pso, NSUInteger gw) {
            NSUInteger w = pso.maxTotalThreadsPerThreadgroup;
            if (w > 1024) w = 1024;
            if (w > gw) w = gw;
            return MTLSizeMake(w, 1, 1);
        };
        if (do_narrow) {
            begin_stage(0);
            [enc setComputePipelineState:g_pso[K_NARROW]];
            [enc setBuffer:in_wrap offset:(NSUInteger)in_wrap_off atIndex:0];
            [enc setBuffer:bufA offset:0 atIndex:1];
            [enc setBytes:&p0 length:sizeof p0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(j->W, j->H, 1) threadsPerThreadgroup:tg_flat(g_pso[K_NARROW], j->W)];
        }
        {
            begin_stage(1);
            [enc setComputePipelineState:pso0];
            [enc setBuffer:src offset:(NSUInteger)src_off atIndex:0];
            [enc setBuffer:bufB offset:0 atIndex:1];
            [enc setBytes:w32 length:wlen atIndex:2];
            [enc setBytes:&p0 length:sizeof p0 atIndex:3];
            const NSUInteger gw = j->W, gh = spanv ? (j->H + j->span - 1) / j->span : j->H;
            [enc dispatchThreads:MTLSizeMake(gw, gh, 1) threadsPerThreadgroup:tg(pso0, j->tg_w0, j->tg_h0, gw, gh)];
        }
        {
            begin_stage(2);
            [enc setComputePipelineState:pso1];
            [enc setBuffer:bufB offset:0 atIndex:0];
            [enc setBuffer:dst offset:(NSUInteger)dst_off atIndex:1];
            [enc setBytes:w32 length:wlen atIndex:2];
            [enc setBytes:&p1 length:sizeof p1 atIndex:3];
            const NSUInteger gw = spanv ? (j->W + j->span - 1) / j->span : j->W, gh = j->H;
            [enc dispatchThreads:MTLSizeMake(gw, gh, 1) threadsPerThreadgroup:tg(pso1, j->tg_w1, j->tg_h1, gw, gh)];
        }
        if (do_widen) {
            begin_stage(3);
            [enc setComputePipelineState:g_pso[K_WIDEN]];
            [enc setBuffer:bufC offset:0 atIndex:0];
            [enc setBuffer:out_wrap offset:(NSUInteger)out_wrap_off atIndex:1];
            [enc setBytes:&p0 length:sizeof p0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(j->W, j->H, 1) threadsPerThreadgroup:tg_flat(g_pso[K_WIDEN], j->W)];
        }
        [enc endEncoding];
        [cb commit];
        bd[3] = now_s() - t0;

        // ---------------- sync ----------------
        t0 = now_s();
        if (j->spin_wait) {
            // Busy-poll: avoids the sleep/wake-up latency of waitUntilCompleted at the cost of one spinning core.
            while (cb.status < MTLCommandBufferStatusCompleted) {
                for (int k = 0; k < 64; ++k) __builtin_arm_yield();
            }
        }
        [cb waitUntilCompleted];
        bd[4] = now_s() - t0;
        for (id<MTLCommandBuffer> c in cbs) {
            if (c.status != MTLCommandBufferStatusCompleted) {
                snprintf(g_err, sizeof g_err, "command buffer failed: %s",
                         c.error ? c.error.localizedDescription.UTF8String : "unknown");
                return -1;
            }
            bd[7] += c.GPUEndTime - c.GPUStartTime;
            bd[13] += c.kernelEndTime - c.kernelStartTime;
        }
        if (j->split)
            for (int s = 0; s < 4; ++s)
                if (stage_cb[s] >= 0) bd[9 + s] = cbs[(NSUInteger)stage_cb[s]].GPUEndTime - cbs[(NSUInteger)stage_cb[s]].GPUStartTime;

        // ---------------- CPU output conversion ----------------
        t0 = now_s();
        if (j->out_mode == 2) {
            if (j->out_f64)
                cpu_widen((const float*)bufC.contents, (double*)j->out_addr, npix);
            else
                cpu_copy(bufC.contents, (void*)j->out_addr, npix * 4);
        }
        bd[5] = now_s() - t0;

        // ---------------- release ----------------
        t0 = now_s();
        pool_put(bufA);
        pool_put(bufB);
        pool_put(bufC);
        if (out_handle) *out_handle = nullptr;
        if (out_pooled) {
            if (out_handle)
                *out_handle = (__bridge_retained void*)out_pooled;
            else
                pool_put(out_pooled);
        }
        if (!j->cache_in_wrap) in_wrap = nil;
        out_wrap = nil;
        src = nil;
        dst = nil;
        cb = nil;
        enc = nil;
        [cbs removeAllObjects];
        bd[6] = now_s() - t0;
        bd[8] = now_s() - t_begin;
        return 0;
    }
}

void e4_set_prefetch_rows(int rows) { g_prefetch_rows = rows; }

uint64_t e4_handle_contents(void* h) { return (uint64_t)((__bridge id<MTLBuffer>)h).contents; }

// Give a pooled output buffer back (called when the NumPy array that views it is collected).
void e4_pool_put(void* h) {
    @autoreleasepool {
        id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)h;
        pool_put(b);
    }
}

void e4_pool_clear() {
    @autoreleasepool {
        std::lock_guard<std::mutex> lk(g_pool_mu);
        g_pool.clear();
        g_cached_wrap = nil;
        g_cached_addr = g_cached_len = 0;
    }
}

// CPU separable Gaussian.  w64: 2r+1 float64 weights (correlate order).  is_f32 = 0: strict float64
// arithmetic on float64 data; is_f32 = 1: float32 arithmetic on float32 data (context only).
// tile: column tile of the vertical pass (0 = auto); use_ring: see corr_axis0; fma_mode: FMA_*.
// bd[0] pass axis 0, bd[1] pass axis 1, bd[2] total.
int e4_cpu_gauss(const void* in, void* out, uint32_t H, uint32_t W, const double* w64, uint32_t radius, int workers,
                 int tile, uint32_t rows_per_item, int is_f32, int fma_mode, int use_ring, double* bd) {
    if (workers <= 0) workers = g_pcores;
    if (!rows_per_item) rows_per_item = 16;
    const int r = (int)radius;
    const size_t npix = (size_t)H * W;
    const double t0 = now_s();
    if (is_f32) {
        std::vector<float> wf(2 * (size_t)r + 1);
        for (int i = 0; i < 2 * r + 1; ++i) wf[(size_t)i] = (float)w64[i];
        float* tmp = (float*)scratch(npix * sizeof(float));
        if (!tmp) return -1;
        corr_axis0<float>((const float*)in, tmp, (int)H, (int)W, wf.data() + r, r, workers, tile, fma_mode, use_ring);
        const double t1 = now_s();
        corr_axis1<float>(tmp, (float*)out, (int)H, (int)W, wf.data() + r, r, workers, (int)rows_per_item, fma_mode);
        bd[0] = t1 - t0;
        bd[1] = now_s() - t1;
    } else {
        double* tmp = (double*)scratch(npix * sizeof(double));
        if (!tmp) return -1;
        corr_axis0<double>((const double*)in, tmp, (int)H, (int)W, w64 + r, r, workers, tile, fma_mode, use_ring);
        const double t1 = now_s();
        corr_axis1<double>(tmp, (double*)out, (int)H, (int)W, w64 + r, r, workers, (int)rows_per_item, fma_mode);
        bd[0] = t1 - t0;
        bd[1] = now_s() - t1;
    }
    bd[2] = now_s() - t0;
    return 0;
}

}  // extern "C"
