// E6 probe: hybrid float32 LU (CPU panels + GPU trailing updates on a shared
// MTLBuffer) with float64 iterative refinement. Plain C entry points.
//
// Layout. The caller's A is float64, ROW-major (NumPy C order). It is narrowed
// 1:1 into the shared buffer, so the buffer holds A row-major == M := A^T
// column-major with lda = n. We factor M = P L U LAPACK-style (column-major,
// partial pivoting on M's rows) and solve A x = M^T x = b with transposed
// triangular solves:  x = P (L^-T (U^-T b)).
//
// The steel GEMM kernel is row-major. The column-major trailing update
//     M22 <- M22 - M21 * M12
// is dispatched on the very same memory as the row-major product
//     M22^T <- M22^T - M12^T * M21^T          (ld = lda for all three)
// i.e. kernel A = U12^T, kernel B = L21^T, kernel C = D = M22^T. No copies.
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../e1_gemm/gemm_runtime.h"
#include "../e7_recursive_cpu/rfact.hpp"

namespace {

using rf::now_s;

struct Ctx {
  int n_max = 0;
  void* buf = nullptr;  // MTLBuffer handle
  float* M = nullptr;   // its contents
  std::vector<int> ipiv;
  std::vector<int> pstart;  // column where each factored block starts (+ n at the end)
  std::vector<float> ws;
  std::vector<double> r, xprev;
};

dispatch_queue_t pq() { return dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0); }

// ---------------------------------------------------------------- narrowing
struct NarrowCtx {
  const double* A;
  float* M;
  int n, chunk;
  double* rowmax;  // per chunk: max abs row sum
};
void narrow_worker(void* v, size_t idx) {
  const NarrowCtx& c = *(const NarrowCtx*)v;
  const int i0 = (int)idx * c.chunk, i1 = std::min(c.n, i0 + c.chunk);
  double mx = 0;
  for (int i = i0; i < i1; ++i) {
    const double* a = c.A + (size_t)i * c.n;
    float* m = c.M + (size_t)i * c.n;
    double s = 0;
    for (int k = 0; k < c.n; ++k) {
      m[k] = (float)a[k];
      s += std::fabs(a[k]);
    }
    mx = std::max(mx, s);
  }
  c.rowmax[idx] = mx;
}
// M <- float(A) in parallel; returns ||A||_inf.
double narrow(const double* A, float* M, int n) {
  int chunk = std::max(8, n / 64);
  size_t nch = (size_t)((n + chunk - 1) / chunk);
  std::vector<double> rowmax(nch, 0.0);
  NarrowCtx c{A, M, n, chunk, rowmax.data()};
  dispatch_apply_f(nch, pq(), &c, narrow_worker);
  double mx = 0;
  for (double v : rowmax) mx = std::max(mx, v);
  return mx;
}

// ------------------------------------------------------- float64 residual
struct ResidCtx {
  const double* A;
  const double* x;
  double* r;
  int n, chunk;
};
void resid_worker(void* v, size_t idx) {
  const ResidCtx& c = *(const ResidCtx*)v;
  const int i0 = (int)idx * c.chunk, i1 = std::min(c.n, i0 + c.chunk);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, i1 - i0, c.n, -1.0, c.A + (size_t)i0 * c.n, c.n,
              c.x, 1, 1.0, c.r + i0, 1);
}
// r <- b - A x (A row-major). par = 0: one cblas_dgemv; par = 1: cblas_dgemv on
// row blocks under dispatch_apply.
void residual(const double* A, const double* x, const double* b, double* r, int n, int par) {
  memcpy(r, b, (size_t)n * sizeof(double));
  if (!par || n < 512) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, -1.0, A, n, x, 1, 1.0, r, 1);
    return;
  }
  int chunk = std::max(64, n / 32);
  ResidCtx c{A, x, r, n, chunk};
  dispatch_apply_f((size_t)((n + chunk - 1) / chunk), pq(), &c, resid_worker);
}

// ---------------------------------------------- row/column-split CPU BLAS3
// The CPU pieces on the critical path of a step are "skinny" (one dimension is
// the panel width), a shape for which Accelerate mostly stays on one thread.
// They are trivially separable, so split them across a few GCD workers, each
// making an ordinary (small) cblas call.
struct SplitCtx {
  int m, n, k, lda, chunk;
  const float* A;
  const float* B;
  float* C;
};
void gemm_rows_worker(void* v, size_t idx) {
  const SplitCtx& c = *(const SplitCtx*)v;
  const int i0 = (int)idx * c.chunk, i1 = std::min(c.m, i0 + c.chunk);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, i1 - i0, c.n, c.k, -1.0f, c.A + i0,
              c.lda, c.B, c.lda, 1.0f, c.C + i0, c.lda);
}
// C(m x n) -= A(m x k) B(k x n), all column-major with the same lda.
void gemm_rows(int m, int n, int k, const float* A, const float* B, float* C, int lda,
               int parts) {
  if (parts <= 1 || m < 1024) {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, -1.0f, A, lda, B, lda, 1.0f,
                C, lda);
    return;
  }
  SplitCtx c{m, n, k, lda, (m + parts - 1) / parts, A, B, C};
  dispatch_apply_f((size_t)parts, pq(), &c, gemm_rows_worker);
}
void trsm_cols_worker(void* v, size_t idx) {
  const SplitCtx& c = *(const SplitCtx*)v;
  const int j0 = (int)idx * c.chunk, j1 = std::min(c.n, j0 + c.chunk);
  if (j1 <= j0) return;
  cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, c.k, j1 - j0, 1.0f,
              c.A, c.lda, c.C + (size_t)j0 * c.lda, c.lda);
}
// B(k x n) <- L^{-1} B, L unit lower k x k; right-hand-side columns split.
void trsm_cols(int k, int n, const float* L, float* B, int lda, int parts) {
  if (parts <= 1 || n < 1024) {
    cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, k, n, 1.0f, L,
                lda, B, lda);
    return;
  }
  SplitCtx c{0, n, k, lda, (n + parts - 1) / parts, L, nullptr, B};
  dispatch_apply_f((size_t)parts, pq(), &c, trsm_cols_worker);
}

double norm_inf(const double* v, int n) {
  double m = 0;
  for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(v[i]));
  return m;
}

}  // namespace

extern "C" {

struct e6_opts {
  int nb;         // panel width
  int cpu_tail;   // finish on the CPU once the trailing matrix order is <= this
  int lookahead;  // 0: none (wait for the GPU, then factor the next panel)
                  // 1: next panel's columns updated by CPU sgemm, panel factored
                  //    while the GPU updates the rest
                  // 2: next panel's columns updated by a first, small GPU command
                  //    buffer; panel factored while a second one updates the rest
  int panel_leaf; // unblocked leaf width of the recursive panel
  int par_resid;  // 1: residual via dgemv on row blocks under dispatch_apply
  int max_iter;   // refinement step cap
  double tol;     // stop when the normwise backward error is <= tol
  int gpu;        // 0: never dispatch to the GPU (pure-CPU float32 + refinement)
  int cpu_parts;  // split the skinny CPU GEMM / TRSM of each step this many ways
  int par_trsv;   // 1: blocked triangular solves with the off-diagonal GEMV split
                  //    across GCD workers; 0: two plain cblas_strsv calls
  int left_swaps; // 1: apply the deferred interchanges to the left part so the
                  //    factor is LAPACK-form; 0: leave L in LINPACK form (the
                  //    solve interleaves the interchanges block by block)
};

struct e6_stats {
  double t_narrow, t_factor, t_solve0, t_refine, t_total;
  // factor breakdown
  double t_panel, t_swap, t_trsm, t_lookahead_gemm, t_submit, t_wait, t_gpu_exec, t_tail,
      t_left_swap;
  // solve/refine breakdown
  double t_resid, t_trsv;
  double normA;
  int n_gpu, n_panels, iters, info;
  double berr[40];  // backward error after 0, 1, 2, ... corrections
  int n_berr;
};

void* e6_create(int n_max) {
  Ctx* c = new Ctx();
  c->n_max = n_max;
  const size_t bytes = (size_t)n_max * n_max * sizeof(float);
  // pad by 64 leading-dimension strides: see msg_alloc in gemm_runtime.mm
  c->buf = msg_alloc_padded(bytes, (size_t)64 * n_max * sizeof(float) + 65536);
  if (!c->buf) {
    delete c;
    return nullptr;
  }
  c->M = (float*)msg_contents(c->buf);
  memset(c->M, 0, bytes);  // fault the pages in once
  c->ipiv.resize(n_max);
  c->ws.resize(n_max);
  c->r.resize(n_max);
  c->xprev.resize(n_max);
  return c;
}

void e6_destroy(void* h) {
  Ctx* c = (Ctx*)h;
  if (!c) return;
  msg_free(c->buf);
  delete c;
}

float* e6_buffer(void* h) { return ((Ctx*)h)->M; }
int* e6_ipiv(void* h) { return ((Ctx*)h)->ipiv.data(); }

// Hybrid LU of the column-major n x n matrix in the shared buffer (lda = n).
// ipiv: 0-based absolute row indices. Fills the factor part of st.
int e6_factor(void* h, int n, const e6_opts* o, e6_stats* st) {
  Ctx* c = (Ctx*)h;
  float* M = c->M;
  const int lda = n;
  int* ipiv = c->ipiv.data();
  rf::Opts& ro = rf::opts();
  ro.lu_leaf = o->panel_leaf;
  ro.trsm_leaf = 0;
  ro.swap_par = 1;
  ro.timers = 0;
  int info = 0;
  const int nb = std::max(8, o->nb);
  std::vector<int>& pstart = c->pstart;  // block starts
  pstart.clear();

  auto panel = [&](int j, int jb) {
    const double t0 = now_s();
    int pinfo = rf::getrf_rec<float>(n - j, jb, M + j + (size_t)j * lda, lda, ipiv + j);
    if (info == 0 && pinfo > 0) info = pinfo + j;
    for (int i = j; i < j + jb; ++i) ipiv[i] += j;
    st->t_panel += now_s() - t0;
    st->n_panels++;
    pstart.push_back(j);
  };

  int j = 0, jb = std::min(nb, n);
  if (o->gpu == 0 || n <= o->cpu_tail) {
    // pure CPU: recursive LU on the whole matrix
    const double t0 = now_s();
    info = rf::getrf_rec<float>(n, n, M, lda, ipiv);
    st->t_tail += now_s() - t0;
    pstart.push_back(0);
    pstart.push_back(n);
    return info;
  }

  panel(0, jb);
  for (;;) {
    const int j2 = j + jb;  // first trailing row/column
    const int nr = n - j2;
    if (nr <= 0) break;
    float* L11 = M + j + (size_t)j * lda;
    float* L21 = L11 + jb;
    float* U12 = M + j + (size_t)j2 * lda;
    float* M22 = U12 + jb;

    // 1. this panel's interchanges on the columns to its right
    double t0 = now_s();
    rf::laswp_cols<float>(nr, M + (size_t)j2 * lda, lda, j, j2, ipiv);
    double t1 = now_s();
    st->t_swap += t1 - t0;
    // 2. U12 = L11^{-1} A12
    trsm_cols(jb, nr, L11, U12, lda, o->cpu_parts);
    double t2 = now_s();
    st->t_trsm += t2 - t1;

    if (nr <= o->cpu_tail) {
      // 3'. finish on the CPU
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nr, nr, jb, -1.0f, L21, lda, U12,
                  lda, 1.0f, M22, lda);
      int pinfo = rf::getrf_rec<float>(nr, nr, M22, lda, ipiv + j2);
      if (info == 0 && pinfo > 0) info = pinfo + j2;
      for (int i = j2; i < n; ++i) ipiv[i] += j2;
      pstart.push_back(j2);
      st->t_tail += now_s() - t2;
      break;
    }

    const int jb2 = std::min(nb, nr);
    const int la = o->lookahead ? jb2 : 0;  // columns of the next panel, done first
    const int c0 = j2 + la;                 // first column of the bulk GPU update
    // GPU: M[j2:, ca:cb] -= L21 * U12[:, ca-j2:cb-j2]   (row-major transposed view)
    auto submit = [&](int ca, int cb_) -> void* {
      msg_gemm_args g;
      g.A = c->buf; g.a_off = (size_t)ca * lda + j;  g.lda = lda; g.ta = 0;  // U12^T: (cb-ca) x jb
      g.B = c->buf; g.b_off = (size_t)j * lda + j2;  g.ldb = lda; g.tb = 0;  // L21^T: jb x nr
      g.C = c->buf; g.c_off = (size_t)ca * lda + j2; g.ldc = lda;
      g.D = c->buf; g.d_off = g.c_off;               g.ldd = lda;
      g.M = cb_ - ca; g.N = nr; g.K = jb;
      g.alpha = -1.0f; g.beta = 1.0f; g.variant = 0;
      st->n_gpu++;
      return msg_gemm_submit(&g);
    };
    void* cb_la = nullptr;
    void* cb = nullptr;
    if (la > 0 && o->lookahead == 2) {
      cb_la = submit(j2, c0);
      if (!cb_la) return -1000;
    }
    if (n - c0 > 0) {
      cb = submit(c0, n);
      if (!cb) return -1000;
    }
    double t3 = now_s();
    st->t_submit += t3 - t2;
    if (la > 0) {
      // 3a. look-ahead: bring the next panel's columns up to date ...
      if (cb_la) {
        double gpu_s = 0;
        if (msg_wait(cb_la, &gpu_s) != 0) return -1001;
        st->t_gpu_exec += gpu_s;
      } else {
        gemm_rows(nr, la, jb, L21, U12, M22, lda, o->cpu_parts);
      }
      double t4 = now_s();
      st->t_lookahead_gemm += t4 - t3;
      // 4. ... and factor it while the GPU works on the rest
      panel(j2, jb2);
    }
    // 5. sync
    double t5 = now_s();
    if (cb) {
      double gpu_s = 0;
      if (msg_wait(cb, &gpu_s) != 0) return -1001;
      st->t_gpu_exec += gpu_s;
    }
    st->t_wait += now_s() - t5;
    if (la == 0) panel(j2, jb2);
    j = j2;
    jb = jb2;
  }
  pstart.push_back(n);

  if (!o->left_swaps) return info;
  // deferred interchanges for the columns left of each panel
  double t0 = now_s();
  for (size_t p = 0; p + 2 < pstart.size(); ++p) {
    const int a = pstart[p], b = pstart[p + 1];
    rf::laswp_cols<float>(b - a, M + (size_t)a * lda, lda, b, n, ipiv);
  }
  st->t_left_swap += now_s() - t0;
  return info;
}

// out[c] -= M[r0:r0+nrows, c]^T v[r0:r0+nrows] for columns c in [c0, c0+ncols)
struct GemvCtx {
  const float* M;
  int lda, r0, nrows, c0, ncols, chunk;
  float* v;
};
// NEON dot product, 4 accumulators (Accelerate's sgemv('T') on a few tall
// columns streams at only ~5 GB/s per thread here; this does ~4x that).
static inline float dot_neon(const float* a, const float* b, int n) {
  float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}
static void gemvT_worker(void* p, size_t idx) {
  const GemvCtx& g = *(const GemvCtx*)p;
  const int a = g.c0 + (int)idx * g.chunk, b = std::min(g.c0 + g.ncols, a + g.chunk);
  if (b <= a) return;
  static const bool use_blas = getenv("E6_TRSV_SGEMV") != nullptr;
  if (use_blas) {
    cblas_sgemv(CblasColMajor, CblasTrans, g.nrows, b - a, -1.0f,
                g.M + g.r0 + (size_t)a * g.lda, g.lda, g.v + g.r0, 1, 1.0f, g.v + a, 1);
    return;
  }
  const float* x = g.v + g.r0;
  for (int c = a; c < b; ++c)
    g.v[c] -= dot_neon(g.M + g.r0 + (size_t)c * g.lda, x, g.nrows);
}
static void gemvT(const float* M, int lda, int r0, int nrows, int c0, int ncols, float* v,
                  int par) {
  if (nrows <= 0 || ncols <= 0) return;
  GemvCtx g{M, lda, r0, nrows, c0, ncols, ncols, v};
  if (!par || (double)nrows * ncols < 5.0e5) {
    gemvT_worker(&g, 0);
    return;
  }
  static const int maxp = getenv("E6_TRSV_PARTS") ? atoi(getenv("E6_TRSV_PARTS")) : 8;
  const int parts = std::min(maxp, std::max(2, ncols / 16));
  g.chunk = (ncols + parts - 1) / parts;
  dispatch_apply_f((size_t)parts, pq(), &g, gemvT_worker);
}

// Solve A x = M^T x = rhs with the float32 factors; result in float64.
//   x = P_0^-1 Lh_0^-T ... P_{K-1}^-1 Lh_{K-1}^-T U^-T rhs
// where block k = columns [pstart[k], pstart[k+1]) and P_k its interchanges.
// With left_swaps the L factor is LAPACK-form and the interchanges commute to
// the end; without, they are applied block by block (LINPACK form). The same
// loop serves both.
static void solve32(Ctx* c, int n, const double* rhs, double* x, const e6_opts* o) {
  float* w = c->ws.data();
  const float* M = c->M;
  const int lda = n;
  const int* ipiv = c->ipiv.data();
  for (int i = 0; i < n; ++i) w[i] = (float)rhs[i];
  const bool blocked = o->par_trsv && n > 1024;
  // U^T y = b : forward substitution, column blocks
  if (!blocked) {
    cblas_strsv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit, n, M, lda, w, 1);
  } else {
    static const int bs = getenv("E6_TRSV_BS") ? atoi(getenv("E6_TRSV_BS")) : 1024;
    for (int i0 = 0; i0 < n; i0 += bs) {
      const int ib = std::min(bs, n - i0);
      gemvT(M, lda, 0, i0, i0, ib, w, 1);
      cblas_strsv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit, ib,
                  M + i0 + (size_t)i0 * lda, lda, w + i0, 1);
    }
  }
  // L^T (with interchanges): backward over the factored blocks
  const std::vector<int>& ps = c->pstart;
  const bool lapack_form = o->left_swaps != 0;
  for (int k = (int)ps.size() - 2; k >= 0; --k) {
    const int a = ps[k], b = ps[k + 1];
    if (blocked || !lapack_form || ps.size() > 2) gemvT(M, lda, b, n - b, a, b - a, w, blocked);
    cblas_strsv(CblasColMajor, CblasLower, CblasTrans, CblasUnit, b - a, M + a + (size_t)a * lda,
                lda, w + a, 1);
    if (!lapack_form)
      for (int i = b - 1; i >= a; --i) {
        const int p = ipiv[i];
        if (p != i) std::swap(w[i], w[p]);
      }
  }
  if (lapack_form)
    for (int i = n - 1; i >= 0; --i) {
      const int p = ipiv[i];
      if (p != i) std::swap(w[i], w[p]);
    }
  for (int i = 0; i < n; ++i) x[i] = (double)w[i];
}

// Full solve: narrow, factor, solve, refine. A: float64 row-major n x n.
int e6_solve(void* h, int n, const double* A, const double* b, double* x, const e6_opts* o,
             e6_stats* st) {
  Ctx* c = (Ctx*)h;
  memset(st, 0, sizeof(*st));
  if (n > c->n_max) return -1;
  const double T0 = now_s();
  st->normA = narrow(A, c->M, n);
  const double T1 = now_s();
  st->t_narrow = T1 - T0;
  int info = e6_factor(h, n, o, st);
  st->info = info;
  const double T2 = now_s();
  st->t_factor = T2 - T1;
  if (info < 0) return info;

  double* r = c->r.data();
  double* xp = c->xprev.data();
  const double nb_ = norm_inf(b, n);
  solve32(c, n, b, x, o);
  const double T3 = now_s();
  st->t_solve0 = T3 - T2;
  st->t_trsv += T3 - T2;
  double eta_prev = INFINITY;
  for (int it = 0;; ++it) {
    double ta = now_s();
    residual(A, x, b, r, n, o->par_resid);
    const double eta = norm_inf(r, n) / (st->normA * norm_inf(x, n) + nb_);
    st->t_resid += now_s() - ta;
    if (st->n_berr < 40) st->berr[st->n_berr++] = eta;
    if (eta <= o->tol) break;                 // at the float64 roundoff floor
    if (!(eta < 0.5 * eta_prev)) {            // stopped improving
      if (eta > eta_prev) {                   // last correction made it worse
        memcpy(x, xp, (size_t)n * sizeof(double));
        st->iters--;
      }
      break;
    }
    if (it >= o->max_iter) break;
    eta_prev = eta;
    memcpy(xp, x, (size_t)n * sizeof(double));
    ta = now_s();
    solve32(c, n, r, r, o);                   // d = A^{-1} r (float32 factors)
    for (int i = 0; i < n; ++i) x[i] += r[i];
    st->t_trsv += now_s() - ta;
    st->iters++;
  }
  const double T4 = now_s();
  st->t_refine = T4 - T3;
  st->t_total = T4 - T0;
  return info;
}

// Triangular solves alone on the factors currently in the context (probe aid).
double e6_solve_only(void* h, int n, const double* rhs, double* x, const e6_opts* o) {
  const double t0 = now_s();
  solve32((Ctx*)h, n, rhs, x, o);
  return now_s() - t0;
}
// Residual alone (probe aid): r = b - A x; returns seconds.
double e6_resid_only(int n, const double* A, const double* x, const double* b, double* r,
                     int par) {
  const double t0 = now_s();
  residual(A, x, b, r, n, par);
  return now_s() - t0;
}

// ------------------------------------------------------------ LAPACK baselines
// All take column-major (Fortran-order) inputs and overwrite them.
// Return info; *seconds = wall time of the LAPACK call alone.
int e6_dgesv(int n, double* A, double* b, int* ipiv, double* seconds) {
  __LAPACK_int n_ = n, one = 1, info = 0;
  const double t0 = now_s();
  dgesv_(&n_, &one, A, &n_, ipiv, b, &n_, &info);
  *seconds = now_s() - t0;
  return (int)info;
}
int e6_sgesv(int n, float* A, float* b, int* ipiv, double* seconds) {
  __LAPACK_int n_ = n, one = 1, info = 0;
  const double t0 = now_s();
  sgesv_(&n_, &one, A, &n_, ipiv, b, &n_, &info);
  *seconds = now_s() - t0;
  return (int)info;
}
// The same solve through the explicit xGETRF + xGETRS pair. On this Accelerate
// build the xGESV driver is markedly slower than xGETRF + xGETRS on the same
// matrix (it does not appear to use the same factorisation path), so both are
// reported.
int e6_dgetrf_dgetrs(int n, double* A, double* b, int* ipiv, double* seconds) {
  __LAPACK_int n_ = n, one = 1, info = 0;
  const double t0 = now_s();
  dgetrf_(&n_, &n_, A, &n_, ipiv, &info);
  if (info == 0) dgetrs_("N", &n_, &one, A, &n_, ipiv, b, &n_, &info);
  *seconds = now_s() - t0;
  return (int)info;
}
int e6_sgetrf_sgetrs(int n, float* A, float* b, int* ipiv, double* seconds) {
  __LAPACK_int n_ = n, one = 1, info = 0;
  const double t0 = now_s();
  sgetrf_(&n_, &n_, A, &n_, ipiv, &info);
  if (info == 0) sgetrs_("N", &n_, &one, A, &n_, ipiv, b, &n_, &info);
  *seconds = now_s() - t0;
  return (int)info;
}
// dsgesv: mixed-precision with iterative refinement. iter > 0: refinement steps
// used; iter < 0: fell back to a full float64 factorisation.
int e6_dsgesv(int n, double* A, const double* b, double* x, int* ipiv, int* iter,
              double* seconds) {
  __LAPACK_int n_ = n, one = 1, info = 0, it = 0;
  std::vector<double> work((size_t)n), bb(b, b + n);
  std::vector<float> swork((size_t)n * (n + 1));
  const double t0 = now_s();
  dsgesv_(&n_, &one, A, &n_, ipiv, bb.data(), &n_, x, &n_, work.data(), swork.data(), &it,
          &info);
  *seconds = now_s() - t0;
  *iter = (int)it;
  return (int)info;
}

// ------------------------------------------------------------- test matrices
// A = U diag(sigma) V^T, row-major, with U = (1/n) H S1 H S2 and
// V^T = (1/n) S3 H S4 H (H = Walsh-Hadamard, S = random +-1 diagonals): exactly
// orthogonal factors that are dense and unstructured for LU purposes, built in
// O(n^2 log n). sigma_i = kappa^(-i/(n-1)) (geometric, "randsvd mode 3").
// n must be a power of two.
namespace {
void fwht(double* v, int n) {
  for (int h = 1; h < n; h <<= 1)
    for (int i = 0; i < n; i += h << 1)
      for (int k = i; k < i + h; ++k) {
        const double a = v[k], b = v[k + h];
        v[k] = a + b;
        v[k + h] = a - b;
      }
}
struct GenCtx {
  double* A;
  int n;
  const double* sigma;
  const signed char *s1, *s2, *s3, *s4;
  int strip;
};
void gen_rows(void* v, size_t i) {  // row i of diag(sigma) S3 H S4 H / n, times S2
  const GenCtx& c = *(const GenCtx*)v;
  const int n = c.n;
  double* row = c.A + i * (size_t)n;
  const double d = c.sigma[i] * c.s3[i];
  for (int k = 0; k < n; ++k)
    row[k] = (__builtin_popcountll((unsigned long long)(i & (size_t)k)) & 1 ? -d : d) * c.s4[k];
  fwht(row, n);
  const double sc = (double)c.s2[i] / n;
  for (int k = 0; k < n; ++k) row[k] *= sc;
}
void gen_cols(void* v, size_t idx) {  // apply (1/n) H S1 H down the columns of a strip
  const GenCtx& c = *(const GenCtx*)v;
  const int n = c.n;
  const int c0 = (int)idx * c.strip, c1 = std::min(n, c0 + c.strip);
  std::vector<double> col(n);
  for (int k = c0; k < c1; ++k) {
    for (int i = 0; i < n; ++i) col[i] = c.A[(size_t)i * n + k];
    fwht(col.data(), n);
    for (int i = 0; i < n; ++i) col[i] *= c.s1[i];
    fwht(col.data(), n);
    const double sc = 1.0 / n;
    for (int i = 0; i < n; ++i) c.A[(size_t)i * n + k] = col[i] * sc;
  }
}
}  // namespace

void e6_gen_randsvd(int n, double kappa, unsigned long long seed, double* A) {
  std::vector<double> sigma(n);
  for (int i = 0; i < n; ++i) sigma[i] = std::pow(kappa, -(double)i / (double)(n - 1));
  std::vector<signed char> s((size_t)4 * n);
  unsigned long long x = seed * 0x9E3779B97F4A7C15ull + 0xD1B54A32D192ED03ull;
  for (auto& e : s) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    e = (x >> 40) & 1 ? 1 : -1;
  }
  GenCtx c{A, n, sigma.data(), s.data(), s.data() + n, s.data() + 2 * n, s.data() + 3 * n, 16};
  dispatch_apply_f((size_t)n, pq(), &c, gen_rows);
  dispatch_apply_f((size_t)((n + c.strip - 1) / c.strip), pq(), &c, gen_cols);
}

}  // extern "C"
