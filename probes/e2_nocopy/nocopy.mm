// E2 probe: zero-copy between host (NumPy) memory and Metal shared buffers.
// Plain C entry points, driven from Python via ctypes.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_muladd, g_fill, g_copy, g_inc64;
static std::atomic<long> g_dealloc_calls{0};
static std::atomic<long> g_release_calls{0};
static char g_err[2048];
static char g_name[256];

struct Params {
    uint64_t n;
    uint32_t mul;
    uint32_t add;
    uint32_t width;
    uint32_t pad;
};

static double now_s() {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * (double)tb.numer / (double)tb.denom * 1e-9;
}

static int perf_cores() {
    int n = 0;
    size_t sz = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &sz, nullptr, 0) != 0 || n <= 0) n = 8;
    return n;
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

extern "C" {

const char* e2_last_error() { return g_err; }

int e2_init(const char* metallib_path) {
    @autoreleasepool {
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
        g_muladd = make_pso(lib, @"muladd_u32");
        g_fill = make_pso(lib, @"fill_u32");
        g_copy = make_pso(lib, @"copy_muladd_u32");
        g_inc64 = make_pso(lib, @"inc_u64");
        if (!g_muladd || !g_fill || !g_copy || !g_inc64) return -3;
        snprintf(g_name, sizeof g_name, "%s", g_dev.name.UTF8String);
        return 0;
    }
}

const char* e2_device_name() { return g_name; }
uint64_t e2_max_buffer_length() { return (uint64_t)g_dev.maxBufferLength; }
uint64_t e2_recommended_working_set() { return (uint64_t)g_dev.recommendedMaxWorkingSetSize; }
uint64_t e2_page_size() { return (uint64_t)sysconf(_SC_PAGESIZE); }
long e2_dealloc_count() { return g_dealloc_calls.load(); }
long e2_release_count() { return g_release_calls.load(); }

// VM region containing addr (same query metal-arrow uses).  Returns 0 on success.
int e2_region(uint64_t addr, uint64_t* r_start, uint64_t* r_size, int* prot, int* max_prot, int* share_mode,
              int* user_tag, int* is_submap) {
    mach_vm_address_t region = (mach_vm_address_t)addr;
    mach_vm_size_t size = 0;
    natural_t depth = UINT32_MAX;
    vm_region_submap_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
    kern_return_t kr = mach_vm_region_recurse(mach_task_self(), &region, &size, &depth,
                                              (vm_region_recurse_info_t)&info, &count);
    if (kr != KERN_SUCCESS) return (int)kr;
    if (region > addr) return -1000;  // addr is in a hole; the call returned the next region
    *r_start = region;
    *r_size = size;
    *prot = info.protection;
    *max_prot = info.max_protection;
    *share_mode = info.share_mode;
    *user_tag = (int)info.user_tag;
    *is_submap = info.is_submap;
    return 0;
}

// Production-style decision: can [addr, addr+len) be wrapped without a copy?
// Expands to the enclosing page span and requires every page of it to be mapped and
// readable (and writable if need_write).  With allow_multi == 0 the span must sit in ONE
// VM map entry (the metal-arrow rule); with allow_multi == 1 it may cross consecutive
// entries with no hole (large anonymous allocations are split into <= 128 MiB entries).
// Returns the number of map entries the span covers (0 = not wrappable).
int e2_wrappable_span(uint64_t addr, uint64_t len, int need_write, int allow_multi, uint64_t* span_start,
                      uint64_t* span_len, uint64_t* offset) {
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    if (!len) return 0;
    const uint64_t first = addr & ~(page - 1);
    const uint64_t last = (addr + len + page - 1) & ~(page - 1);
    uint64_t cur = first;
    int entries = 0;
    while (cur < last) {
        uint64_t rs = 0, rz = 0;
        int prot = 0, mprot = 0, sm = 0, tag = 0, sub = 0;
        if (e2_region(cur, &rs, &rz, &prot, &mprot, &sm, &tag, &sub) != 0) return 0;  // hole or error
        if (sub || !(prot & VM_PROT_READ) || (need_write && !(prot & VM_PROT_WRITE))) return 0;
        if (rs > cur) return 0;
        ++entries;
        if (!allow_multi && rs + rz < last) return 0;
        cur = rs + rz;
    }
    *span_start = first;
    *span_len = last - first;
    *offset = addr - first;
    return entries;
}

typedef void (*dealloc_cb_t)(void*);

// Raw wrap: exactly what was asked, no rounding.  Returns a retained handle or NULL.
void* e2_wrap(uint64_t addr, uint64_t len, dealloc_cb_t cb, void* ctx) {
    @autoreleasepool {
        id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void*)addr
                                                   length:(NSUInteger)len
                                                  options:MTLResourceStorageModeShared
                                              deallocator:^(void*, NSUInteger) {
                                                  g_dealloc_calls++;
                                                  if (cb) cb(ctx);
                                              }];
        if (!b) return nullptr;
        return (__bridge_retained void*)b;
    }
}

// Same wrap but passing deallocator:nil (caller guarantees lifetime).
void* e2_wrap_nodealloc(uint64_t addr, uint64_t len) {
    @autoreleasepool {
        id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void*)addr
                                                   length:(NSUInteger)len
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        if (!b) return nullptr;
        return (__bridge_retained void*)b;
    }
}

void* e2_alloc(uint64_t len) {
    @autoreleasepool {
        id<MTLBuffer> b = [g_dev newBufferWithLength:(NSUInteger)len options:MTLResourceStorageModeShared];
        if (!b) return nullptr;
        return (__bridge_retained void*)b;
    }
}

void e2_release(void* h) {
    @autoreleasepool {
        id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)h;
        (void)b;  // ARC releases at scope exit
        g_release_calls++;
    }
}

uint64_t e2_contents(void* h) { return (uint64_t)((__bridge id<MTLBuffer>)h).contents; }
uint64_t e2_length(void* h) { return (uint64_t)((__bridge id<MTLBuffer>)h).length; }
uint64_t e2_allocated_size(void* h) { return (uint64_t)((__bridge id<MTLBuffer>)h).allocatedSize; }
int e2_storage_mode(void* h) { return (int)((__bridge id<MTLBuffer>)h).storageMode; }

// which: 0 muladd_u32 (in place), 1 fill_u32, 3 inc_u64 (in place)
static int run1(id<MTLComputePipelineState> pso, id<MTLBuffer> buf, uint64_t byte_offset, id<MTLBuffer> buf2,
                uint64_t n, uint32_t mul, uint32_t add, double* secs) {
    @autoreleasepool {
        Params p{};
        p.n = n;
        p.mul = mul;
        p.add = add;
        p.width = n < 65536 ? (uint32_t)(n ? n : 1) : 65536u;
        const uint64_t rows = (n + p.width - 1) / p.width;
        const double t0 = now_s();
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf offset:(NSUInteger)byte_offset atIndex:0];
        [enc setBytes:&p length:sizeof p atIndex:1];
        if (buf2) [enc setBuffer:buf2 offset:0 atIndex:2];
        const NSUInteger tw = pso.threadExecutionWidth;
        const NSUInteger th = pso.maxTotalThreadsPerThreadgroup / tw;
        [enc dispatchThreads:MTLSizeMake(p.width, (NSUInteger)rows, 1)
            threadsPerThreadgroup:MTLSizeMake(tw, th < rows ? th : (NSUInteger)rows, 1)];
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

int e2_run_muladd_u32(void* h, uint64_t byte_offset, uint64_t n, uint32_t mul, uint32_t add, double* secs) {
    return run1(g_muladd, (__bridge id<MTLBuffer>)h, byte_offset, nil, n, mul, add, secs);
}
int e2_run_fill_u32(void* h, uint64_t n, uint32_t mul, uint32_t add, double* secs) {
    return run1(g_fill, (__bridge id<MTLBuffer>)h, 0, nil, n, mul, add, secs);
}
int e2_run_copy_muladd_u32(void* hsrc, uint64_t src_offset, void* hdst, uint64_t n, uint32_t mul, uint32_t add,
                           double* secs) {
    return run1(g_copy, (__bridge id<MTLBuffer>)hsrc, src_offset, (__bridge id<MTLBuffer>)hdst, n, mul, add, secs);
}
int e2_run_inc_u64(void* h, uint64_t n, double* secs) {
    return run1(g_inc64, (__bridge id<MTLBuffer>)h, 0, nil, n, 0, 0, secs);
}

// ---- timing helpers (all timing done natively) -------------------------------------------

// reps x { wrap ; release }.  t_wrap[i], t_release[i] in seconds.  Returns number of
// reps in which the deallocator had already run when release returned (synchronous teardown).
int e2_time_wrap_release(uint64_t addr, uint64_t len, int reps, double* t_wrap, double* t_release) {
    int sync_dealloc = 0;
    for (int r = 0; r < reps; ++r) {
        const long before = g_dealloc_calls.load();
        double t0, t1, t2;
        @autoreleasepool {
            t0 = now_s();
            id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void*)addr
                                                       length:(NSUInteger)len
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:^(void*, NSUInteger) {
                                                      g_dealloc_calls++;
                                                  }];
            t1 = now_s();
            if (!b) return -1;
            b = nil;
        }
        t2 = now_s();
        t_wrap[r] = t1 - t0;
        t_release[r] = t2 - t1;
        if (g_dealloc_calls.load() > before) ++sync_dealloc;
    }
    return sync_dealloc;
}

// reps x { wrap ; inc_u64 kernel over the whole buffer ; wait ; release }: the cost of a
// freshly wrapped buffer's FIRST use, as one number.  n64 = number of ulong elements.
int e2_time_wrap_use_release(uint64_t addr, uint64_t len, uint64_t n64, int reps, double* t_total, double* t_kernel) {
    for (int r = 0; r < reps; ++r) {
        @autoreleasepool {
            const double t0 = now_s();
            id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void*)addr
                                                       length:(NSUInteger)len
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
            if (!b) return -1;
            double tk = 0;
            if (run1(g_inc64, b, 0, nil, n64, 0, 0, &tk) != 0) return -2;
            b = nil;
            t_kernel[r] = tk;
            t_total[r] = now_s() - t0;  // release of the buffer handle included; pool drain after
        }
    }
    return 0;
}

// reps x { alloc owned shared buffer ; release }
int e2_time_alloc_release(uint64_t len, int reps, double* t_alloc, double* t_release) {
    for (int r = 0; r < reps; ++r) {
        double t0, t1, t2;
        @autoreleasepool {
            t0 = now_s();
            id<MTLBuffer> b = [g_dev newBufferWithLength:(NSUInteger)len options:MTLResourceStorageModeShared];
            t1 = now_s();
            if (!b) return -1;
            b = nil;
        }
        t2 = now_s();
        t_alloc[r] = t1 - t0;
        t_release[r] = t2 - t1;
    }
    return 0;
}

// First-use anatomy on an EXISTING (already faulted-in) host range.
//   kind 0: wrap (tracked)   -> inc_u64 over n64 elements -> release
//   kind 1: wrap (untracked) -> inc_u64 over n64 elements -> release
//   kind 2: wrap (tracked)   -> inc_u64 over n64 elements TWICE (2nd timed in t_kernel2) -> release
// t_create / t_kernel / t_kernel2 / t_release are per-rep seconds.
int e2_time_wrap_anatomy(int kind, uint64_t addr, uint64_t len, uint64_t n64, int reps, double* t_create,
                         double* t_kernel, double* t_kernel2, double* t_release) {
    for (int r = 0; r < reps; ++r) {
        double t0, t1, t3;
        @autoreleasepool {
            t0 = now_s();
            const MTLResourceOptions opt = MTLResourceStorageModeShared |
                                           (kind == 1 ? MTLResourceHazardTrackingModeUntracked : 0);
            id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void*)addr length:(NSUInteger)len options:opt deallocator:nil];
            t1 = now_s();
            if (!b) return -1;
            double tk = 0, tk2 = 0;
            if (run1(g_inc64, b, 0, nil, n64, 0, 0, &tk) != 0) return -2;
            if (kind == 2 && run1(g_inc64, b, 0, nil, n64, 0, 0, &tk2) != 0) return -2;
            t_kernel[r] = tk;
            t_kernel2[r] = tk2;
            t3 = now_s();
            b = nil;
        }
        t_create[r] = t1 - t0;
        t_release[r] = now_s() - t3;
    }
    return 0;
}

// Fresh OUTPUT buffer anatomy: where do never-touched pages get faulted in, and what does it cost?
//   kind 0: mmap(fresh) -> wrap -> fill_u32 kernel (GPU is first toucher) -> release -> munmap
//   kind 1: newBufferWithLength (fresh owned) -> fill_u32 kernel -> release
//   kind 2: mmap(fresh) -> memset on 1 thread (CPU is first toucher) -> munmap
//   kind 3: mmap(fresh) -> memset split over the P-cores -> munmap
//   kind 4: malloc(fresh) -> wrap -> fill_u32 kernel -> release -> free   (what np.empty gives us)
//   kind 5: malloc(fresh) -> memset 1 thread -> free                      (what NumPy/SciPy pay)
// t_create: mmap/malloc/alloc (+wrap); t_fill: kernel or memset; t_release: release (+munmap/free)
int e2_time_fresh_output(int kind, uint64_t len, int reps, double* t_create, double* t_fill, double* t_release) {
    const uint64_t n32 = len / 4;
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    for (int r = 0; r < reps; ++r) {
        double t0 = now_s(), t1 = 0, t2 = 0;
        @autoreleasepool {
            void* mem = nullptr;
            id<MTLBuffer> b = nil;
            if (kind == 0 || kind == 2 || kind == 3) {
                mem = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
                if (mem == MAP_FAILED) return -1;
            } else if (kind == 4 || kind == 5) {
                mem = malloc(len);
                if (!mem) return -1;
            }
            if (kind == 0 || kind == 4) {
                b = [g_dev newBufferWithBytesNoCopy:mem length:(NSUInteger)len options:MTLResourceStorageModeShared deallocator:nil];
                if (!b) return -3;
            } else if (kind == 1) {
                b = [g_dev newBufferWithLength:(NSUInteger)len options:MTLResourceStorageModeShared];
                if (!b) return -3;
            }
            t1 = now_s();
            if (b) {
                if (run1(g_fill, b, 0, nil, n32, 1, 3, nullptr) != 0) return -2;
            } else if (kind == 2 || kind == 5) {
                memset(mem, 1, len);
            } else {
                const int nt = perf_cores();
                const uint64_t chunk = (len / (uint64_t)nt + 16383) & ~16383ull;
                dispatch_apply((size_t)((len + chunk - 1) / chunk), q, ^(size_t i) {
                    const uint64_t off = (uint64_t)i * chunk;
                    memset((char*)mem + off, 1, (size_t)(off + chunk <= len ? chunk : len - off));
                });
            }
            t2 = now_s();
            b = nil;
            if (kind == 0 || kind == 2 || kind == 3) munmap(mem, len);
            if (kind == 4 || kind == 5) free(mem);
        }
        t_create[r] = t1 - t0;
        t_fill[r] = t2 - t1;
        t_release[r] = now_s() - t2;
    }
    return 0;
}

void e2_time_memcpy(uint64_t dst, uint64_t src, uint64_t len, int reps, double* t) {
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_s();
        memcpy((void*)dst, (const void*)src, (size_t)len);
        t[r] = now_s() - t0;
    }
}

// memcpy split over the performance cores (best-case copy).
void e2_time_memcpy_mt(uint64_t dst, uint64_t src, uint64_t len, int reps, double* t) {
    const int nt = perf_cores();
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    uint64_t chunk = ((len / (uint64_t)nt) + page - 1) & ~(page - 1);
    if (chunk < (1u << 20)) chunk = 1u << 20;
    const size_t nchunks = (size_t)((len + chunk - 1) / chunk);
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_s();
        dispatch_apply(nchunks, q, ^(size_t i) {
            const uint64_t off = (uint64_t)i * chunk;
            const uint64_t m = off + chunk <= len ? chunk : len - off;
            memcpy((void*)(dst + off), (const void*)(src + off), (size_t)m);
        });
        t[r] = now_s() - t0;
    }
}

int e2_perf_cores() { return perf_cores(); }

// Two adjacent anonymous mappings that are separate VM map entries (second one has a
// different protection so the kernel cannot coalesce them).  Returns base or 0.
uint64_t e2_make_split_mapping(uint64_t pages_each, int second_readonly) {
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    void* base = mmap(nullptr, 2 * pages_each * page, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (base == MAP_FAILED) return 0;
    memset(base, 0, 2 * pages_each * page);
    void* second = (char*)base + pages_each * page;
    // Re-map the second half as its own object.
    void* got = mmap(second, pages_each * page, second_readonly ? PROT_READ : (PROT_READ | PROT_WRITE),
                     MAP_ANON | MAP_SHARED | MAP_FIXED, -1, 0);
    if (got != second) return 0;
    return (uint64_t)base;
}

// A mapping followed by an unmapped hole: returns base of `pages` mapped pages whose
// successor page is guaranteed unmapped.
uint64_t e2_make_mapping_with_hole(uint64_t pages) {
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    void* base = mmap(nullptr, (pages + 1) * page, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (base == MAP_FAILED) return 0;
    memset(base, 0, pages * page);
    munmap((char*)base + pages * page, page);
    return (uint64_t)base;
}

}  // extern "C"
