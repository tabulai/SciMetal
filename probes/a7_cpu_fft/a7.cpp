// A7 probe: Apple Accelerate vDSP FFT / DFT vs SciPy pocketfft on Apple Silicon.
// Compares both end-to-end interleaved (ctoz + transform + ztoc) and split-only
// across 1-D (2^8..2^24 + f*2^n) and batched transforms on 1 and 16 P-core threads.
#include <Accelerate/Accelerate.h>
#include <pthread.h>
#include <sys/qos.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

inline double now_s() noexcept {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Persistent P-core worker pool matching SciMetal's QOS_CLASS_USER_INTERACTIVE scheduling.
class Pool {
public:
    static constexpr size_t max_threads = 16;

    Pool() {
        workers_.reserve(max_threads - 1);
        for (size_t i = 1; i < max_threads; ++i) {
            workers_.emplace_back([this, i] {
                pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
                worker_loop(i);
            });
        }
    }

    ~Pool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
            ++generation_;
        }
        cv_work_.notify_all();
        for (auto& t : workers_) t.join();
    }

    template<class Fn>
    void parallel_for(size_t count, size_t nthreads, Fn&& fn) {
        const size_t active = std::max<size_t>(1, std::min({nthreads, max_threads, count}));
        if (active == 1 || count <= 1) {
            fn(0, 0, count);
            return;
        }
        fn_ = [&](size_t tid) {
            const size_t base = count / active;
            const size_t rem = count % active;
            const size_t begin = tid * base + std::min(tid, rem);
            const size_t end = begin + base + (tid < rem ? 1 : 0);
            fn(tid, begin, end);
        };
        {
            std::lock_guard<std::mutex> lock(mu_);
            active_ = active;
            remaining_.store(active - 1, std::memory_order_relaxed);
            ++generation_;
        }
        cv_work_.notify_all();
        fn_(0);
        // Brief spin before sleeping for minimal barrier latency.
        while (remaining_.load(std::memory_order_acquire) != 0) {
            std::unique_lock<std::mutex> lock(mu_done_);
            if (remaining_.load(std::memory_order_acquire) != 0) {
                cv_done_.wait(lock, [&] { return remaining_.load(std::memory_order_acquire) == 0; });
            }
        }
    }

private:
    void worker_loop(size_t tid) {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_work_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
            }
            if (tid < active_) fn_(tid);
            if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(mu_done_);
                cv_done_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::function<void(size_t)> fn_;
    std::mutex mu_, mu_done_;
    std::condition_variable cv_work_, cv_done_;
    uint64_t generation_ = 0;
    size_t active_ = 1;
    std::atomic<size_t> remaining_{0};
    bool stop_ = false;
};

Pool& pool() {
    static Pool instance;
    return instance;
}

bool is_pow2(size_t n) { return n >= 8 && (n & (n - 1)) == 0; }

unsigned ilog2(size_t n) {
    unsigned k = 0;
    while ((size_t{1} << k) < n) ++k;
    return k;
}

struct ProbePlan {
    size_t batch = 1;
    size_t n = 0;
    int is_double = 0;
    int use_dft = 0;  // 0 = vDSP_fft_zop[D], 1 = vDSP_DFT_Execute[D]
    unsigned log2n = 0;
    FFTSetup fft32 = nullptr;
    FFTSetupD fft64 = nullptr;
    vDSP_DFT_Setup dft32 = nullptr;
    vDSP_DFT_SetupD dft64 = nullptr;

    // Pre-allocated per-worker scratch buffers for interleaved mode (each holds 4*n floats or doubles: ir, ii, or_, oi).
    size_t worker_stride = 0;
    std::vector<float> scratch32;
    std::vector<double> scratch64;

    ~ProbePlan() {
        if (fft32) vDSP_destroy_fftsetup(fft32);
        if (fft64) vDSP_destroy_fftsetupD(fft64);
        if (dft32) vDSP_DFT_DestroySetup(dft32);
        if (dft64) vDSP_DFT_DestroySetupD(dft64);
    }
};

template<int Double, int Dft>
void run_interleaved_chunk(const ProbePlan& h, const void* in_ptr, void* out_ptr, size_t tid, size_t b0, size_t b1) {
    const size_t n = h.n;
    if constexpr (!Double) {
        const float* in = static_cast<const float*>(in_ptr);
        float* out = static_cast<float*>(out_ptr);
        float* s = const_cast<float*>(h.scratch32.data()) + tid * h.worker_stride;
        DSPSplitComplex sp_in{s, s + n};
        DSPSplitComplex sp_out{s + 2 * n, s + 3 * n};
        for (size_t b = b0; b < b1; ++b) {
            const DSPComplex* row_in = reinterpret_cast<const DSPComplex*>(in + 2 * b * n);
            DSPComplex* row_out = reinterpret_cast<DSPComplex*>(out + 2 * b * n);
            vDSP_ctoz(row_in, 2, &sp_in, 1, n);
            if constexpr (Dft) {
                vDSP_DFT_Execute(h.dft32, sp_in.realp, sp_in.imagp, sp_out.realp, sp_out.imagp);
            } else {
                vDSP_fft_zop(h.fft32, &sp_in, 1, &sp_out, 1, h.log2n, kFFTDirection_Forward);
            }
            vDSP_ztoc(&sp_out, 1, row_out, 2, n);
        }
    } else {
        const double* in = static_cast<const double*>(in_ptr);
        double* out = static_cast<double*>(out_ptr);
        double* s = const_cast<double*>(h.scratch64.data()) + tid * h.worker_stride;
        DSPDoubleSplitComplex sp_in{s, s + n};
        DSPDoubleSplitComplex sp_out{s + 2 * n, s + 3 * n};
        for (size_t b = b0; b < b1; ++b) {
            const DSPDoubleComplex* row_in = reinterpret_cast<const DSPDoubleComplex*>(in + 2 * b * n);
            DSPDoubleComplex* row_out = reinterpret_cast<DSPDoubleComplex*>(out + 2 * b * n);
            vDSP_ctozD(row_in, 2, &sp_in, 1, n);
            if constexpr (Dft) {
                vDSP_DFT_ExecuteD(h.dft64, sp_in.realp, sp_in.imagp, sp_out.realp, sp_out.imagp);
            } else {
                vDSP_fft_zopD(h.fft64, &sp_in, 1, &sp_out, 1, h.log2n, kFFTDirection_Forward);
            }
            vDSP_ztocD(&sp_out, 1, row_out, 2, n);
        }
    }
}

template<int Double, int Dft>
void run_split_chunk(const ProbePlan& h, const void* in_re_ptr, const void* in_im_ptr,
                     void* out_re_ptr, void* out_im_ptr, size_t b0, size_t b1) {
    const size_t n = h.n;
    if constexpr (!Double) {
        float* ir = const_cast<float*>(static_cast<const float*>(in_re_ptr));
        float* ii = const_cast<float*>(static_cast<const float*>(in_im_ptr));
        float* orp = static_cast<float*>(out_re_ptr);
        float* oip = static_cast<float*>(out_im_ptr);
        for (size_t b = b0; b < b1; ++b) {
            const size_t off = b * n;
            DSPSplitComplex sp_in{ir + off, ii + off};
            DSPSplitComplex sp_out{orp + off, oip + off};
            if constexpr (Dft) {
                vDSP_DFT_Execute(h.dft32, sp_in.realp, sp_in.imagp, sp_out.realp, sp_out.imagp);
            } else {
                vDSP_fft_zop(h.fft32, &sp_in, 1, &sp_out, 1, h.log2n, kFFTDirection_Forward);
            }
        }
    } else {
        double* ir = const_cast<double*>(static_cast<const double*>(in_re_ptr));
        double* ii = const_cast<double*>(static_cast<const double*>(in_im_ptr));
        double* orp = static_cast<double*>(out_re_ptr);
        double* oip = static_cast<double*>(out_im_ptr);
        for (size_t b = b0; b < b1; ++b) {
            const size_t off = b * n;
            DSPDoubleSplitComplex sp_in{ir + off, ii + off};
            DSPDoubleSplitComplex sp_out{orp + off, oip + off};
            if constexpr (Dft) {
                vDSP_DFT_ExecuteD(h.dft64, sp_in.realp, sp_in.imagp, sp_out.realp, sp_out.imagp);
            } else {
                vDSP_fft_zopD(h.fft64, &sp_in, 1, &sp_out, 1, h.log2n, kFFTDirection_Forward);
            }
        }
    }
}

} // namespace

extern "C" {

void* a7_create(size_t batch, size_t n, int is_double, int use_dft) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    auto h = std::make_unique<ProbePlan>();
    h->batch = batch;
    h->n = n;
    h->is_double = is_double;
    h->use_dft = use_dft;
    if (!use_dft) {
        if (!is_pow2(n)) return nullptr;
        h->log2n = ilog2(n);
        if (is_double) {
            h->fft64 = vDSP_create_fftsetupD(h->log2n, kFFTRadix2);
            if (!h->fft64) return nullptr;
        } else {
            h->fft32 = vDSP_create_fftsetup(h->log2n, kFFTRadix2);
            if (!h->fft32) return nullptr;
        }
    } else {
        if (is_double) {
            h->dft64 = vDSP_DFT_zop_CreateSetupD(nullptr, n, vDSP_DFT_FORWARD);
            if (!h->dft64) return nullptr;
        } else {
            h->dft32 = vDSP_DFT_zop_CreateSetup(nullptr, n, vDSP_DFT_FORWARD);
            if (!h->dft32) return nullptr;
        }
    }
    const size_t active = std::min<size_t>(Pool::max_threads, batch);
    h->worker_stride = 4 * n;
    if (is_double) {
        h->scratch64.resize(active * h->worker_stride);
    } else {
        h->scratch32.resize(active * h->worker_stride);
    }
    return h.release();
}

void a7_destroy(void* ptr) {
    delete static_cast<ProbePlan*>(ptr);
}

void a7_run_interleaved(void* ptr, const void* in_ptr, void* out_ptr, size_t nthreads) {
    const ProbePlan& h = *static_cast<const ProbePlan*>(ptr);
    pool().parallel_for(h.batch, nthreads, [&](size_t tid, size_t b0, size_t b1) {
        if (!h.is_double && !h.use_dft) run_interleaved_chunk<0, 0>(h, in_ptr, out_ptr, tid, b0, b1);
        else if (!h.is_double && h.use_dft) run_interleaved_chunk<0, 1>(h, in_ptr, out_ptr, tid, b0, b1);
        else if (h.is_double && !h.use_dft) run_interleaved_chunk<1, 0>(h, in_ptr, out_ptr, tid, b0, b1);
        else run_interleaved_chunk<1, 1>(h, in_ptr, out_ptr, tid, b0, b1);
    });
}

void a7_run_split(void* ptr, const void* in_re, const void* in_im, void* out_re, void* out_im, size_t nthreads) {
    const ProbePlan& h = *static_cast<const ProbePlan*>(ptr);
    pool().parallel_for(h.batch, nthreads, [&](size_t, size_t b0, size_t b1) {
        if (!h.is_double && !h.use_dft) run_split_chunk<0, 0>(h, in_re, in_im, out_re, out_im, b0, b1);
        else if (!h.is_double && h.use_dft) run_split_chunk<0, 1>(h, in_re, in_im, out_re, out_im, b0, b1);
        else if (h.is_double && !h.use_dft) run_split_chunk<1, 0>(h, in_re, in_im, out_re, out_im, b0, b1);
        else run_split_chunk<1, 1>(h, in_re, in_im, out_re, out_im, b0, b1);
    });
}

double a7_bench_interleaved(void* ptr, const void* in_ptr, void* out_ptr, size_t nthreads, int warm, int reps) {
    for (int i = 0; i < warm; ++i) a7_run_interleaved(ptr, in_ptr, out_ptr, nthreads);
    std::vector<double> ts(reps);
    for (int i = 0; i < reps; ++i) {
        const double t0 = now_s();
        a7_run_interleaved(ptr, in_ptr, out_ptr, nthreads);
        ts[i] = now_s() - t0;
    }
    std::sort(ts.begin(), ts.end());
    return ts[reps / 2] * 1e3; // median ms
}

double a7_bench_split(void* ptr, const void* in_re, const void* in_im, void* out_re, void* out_im,
                      size_t nthreads, int warm, int reps) {
    for (int i = 0; i < warm; ++i) a7_run_split(ptr, in_re, in_im, out_re, out_im, nthreads);
    std::vector<double> ts(reps);
    for (int i = 0; i < reps; ++i) {
        const double t0 = now_s();
        a7_run_split(ptr, in_re, in_im, out_re, out_im, nthreads);
        ts[i] = now_s() - t0;
    }
    std::sort(ts.begin(), ts.end());
    return ts[reps / 2] * 1e3; // median ms
}

} // extern "C"
