// E7 probe: recursive LU (partial pivoting) and recursive Cholesky built on
// Accelerate's cblas_?gemm / cblas_?trsm. Header-only, float + double.
// Column-major (LAPACK) storage throughout. No threads of our own.
#pragma once

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace rf {

inline double now_s() {
  static mach_timebase_info_data_t tb = {0, 0};
  if (tb.denom == 0) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

// ---------------------------------------------------------------- tunables
struct Opts {
  int lu_leaf = 8;        // columns at/below which the LU panel is unblocked
  int trsm_leaf = 0;      // 0: call cblas_?trsm directly; >0: recursive TRSM on
                          //    GEMM with cblas_?trsm below this size
  int chol_leaf = 32;     // order at/below which Cholesky is unblocked
  int chol_update = 0;    // 0: SYRK, 1: full GEMM (clobbers the strict upper
                          //    triangle), 2: recursive SYRK on GEMM with SYRK
                          //    on the diagonal leaves, 3: recursive SYRK on
                          //    GEMM with GEMM-into-scratch on the diagonal
                          //    leaves (no SYRK at all, upper triangle intact)
  int rsyrk_leaf = 256;   // base order for the recursive SYRK
  int timers = 1;         // collect per-category wall time
  int swap_par = 0;       // 1: row interchanges over columns run under
                          //    dispatch_apply (our only own threading; never
                          //    wrapped around an Accelerate call)
};

struct Stats {
  double gemm = 0, trsm = 0, swap = 0, leaf = 0, syrk = 0;
  long n_gemm = 0, n_trsm = 0, n_swap = 0, n_leaf = 0, n_syrk = 0;
  // GEMM wall time / flops bucketed by floor(log2(K)) of the inner dimension
  double gemm_lvl_s[20] = {0};
  double gemm_lvl_flops[20] = {0};
  void clear() { *this = Stats(); }
};

inline Opts& opts() {
  static Opts o;
  return o;
}
inline Stats& stats() {
  static Stats s;
  return s;
}

inline int ilog2(int k) {
  int l = 0;
  while (k > 1) {
    k >>= 1;
    ++l;
  }
  return l;
}

// GEMM timer that also buckets by inner dimension K.
struct GemmTick {
  int lvl;
  double fl;
  double t0;
  GemmTick(int m, int n, int k)
      : lvl(ilog2(k)), fl(2.0 * m * n * k), t0(opts().timers ? now_s() : 0) {}
  ~GemmTick();
};

struct Tick {
  double* acc;
  long* cnt;
  double t0;
  Tick(double& a, long& c) : acc(&a), cnt(&c), t0(opts().timers ? now_s() : 0) {}
  ~Tick() {
    if (opts().timers) *acc += now_s() - t0;
    ++*cnt;
  }
};

inline GemmTick::~GemmTick() {
  Stats& s = stats();
  if (opts().timers) {
    const double dt = now_s() - t0;
    s.gemm += dt;
    s.gemm_lvl_s[lvl] += dt;
  }
  s.gemm_lvl_flops[lvl] += fl;
  ++s.n_gemm;
}

// ------------------------------------------------------------ BLAS wrappers
inline void gemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int m, int n, int k,
                 double alpha, const double* A, int lda, const double* B, int ldb,
                 double beta, double* C, int ldc) {
  cblas_dgemm(CblasColMajor, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline void gemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int m, int n, int k,
                 float alpha, const float* A, int lda, const float* B, int ldb,
                 float beta, float* C, int ldc) {
  cblas_sgemm(CblasColMajor, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline void trsm(CBLAS_SIDE s, CBLAS_UPLO u, CBLAS_TRANSPOSE t, CBLAS_DIAG d,
                 int m, int n, double alpha, const double* A, int lda, double* B, int ldb) {
  cblas_dtrsm(CblasColMajor, s, u, t, d, m, n, alpha, A, lda, B, ldb);
}
inline void trsm(CBLAS_SIDE s, CBLAS_UPLO u, CBLAS_TRANSPOSE t, CBLAS_DIAG d,
                 int m, int n, float alpha, const float* A, int lda, float* B, int ldb) {
  cblas_strsm(CblasColMajor, s, u, t, d, m, n, alpha, A, lda, B, ldb);
}
inline void syrk(CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k, double alpha,
                 const double* A, int lda, double beta, double* C, int ldc) {
  cblas_dsyrk(CblasColMajor, u, t, n, k, alpha, A, lda, beta, C, ldc);
}
inline void syrk(CBLAS_UPLO u, CBLAS_TRANSPOSE t, int n, int k, float alpha,
                 const float* A, int lda, float beta, float* C, int ldc) {
  cblas_ssyrk(CblasColMajor, u, t, n, k, alpha, A, lda, beta, C, ldc);
}
inline int iamax(int n, const double* x) { return (int)cblas_idamax(n, x, 1); }
inline int iamax(int n, const float* x) { return (int)cblas_isamax(n, x, 1); }
inline void scal(int n, double a, double* x) { cblas_dscal(n, a, x, 1); }
inline void scal(int n, float a, float* x) { cblas_sscal(n, a, x, 1); }
inline void ger(int m, int n, double alpha, const double* x, const double* y, int incy,
                double* A, int lda) {
  cblas_dger(CblasColMajor, m, n, alpha, x, 1, y, incy, A, lda);
}
inline void ger(int m, int n, float alpha, const float* x, const float* y, int incy,
                float* A, int lda) {
  cblas_sger(CblasColMajor, m, n, alpha, x, 1, y, incy, A, lda);
}

// ------------------------------------------------------------------ helpers
// Apply the row interchanges ipiv[k1..k2) (0-based, absolute row indices
// relative to A's row 0) to n columns of A, one column at a time. A column of
// a tall panel fits in L1/L2, so this is far friendlier than the row-wise
// reference xLASWP, which strides across columns (one TLB/cache miss per
// element for large lda).
template <typename T>
inline void laswp_range(int j0, int j1, T* A, int lda, int k1, int k2, const int* ipiv) {
  for (int j = j0; j < j1; ++j) {
    T* c = A + (size_t)j * lda;
    for (int i = k1; i < k2; ++i) {
      const int p = ipiv[i];
      const T t = c[i];
      c[i] = c[p];
      c[p] = t;
    }
  }
}

template <typename T>
struct SwapCtx {
  T* A;
  int lda, k1, k2, n, chunk;
  const int* ipiv;
};

template <typename T>
void laswp_worker(void* vctx, size_t idx) {
  const SwapCtx<T>* c = (const SwapCtx<T>*)vctx;
  const int j0 = (int)idx * c->chunk;
  const int j1 = std::min(c->n, j0 + c->chunk);
  laswp_range(j0, j1, c->A, c->lda, c->k1, c->k2, c->ipiv);
}

template <typename T>
inline void laswp_cols(int n, T* A, int lda, int k1, int k2, const int* ipiv) {
  if (n <= 0 || k2 <= k1) return;
  Tick tk(stats().swap, stats().n_swap);
  if (opts().swap_par && n >= 32 && (double)n * (k2 - k1) >= 1.0e5) {
    int chunk = std::max(8, n / 64);
    SwapCtx<T> ctx{A, lda, k1, k2, n, chunk, ipiv};
    dispatch_apply_f((size_t)((n + chunk - 1) / chunk),
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &ctx,
                     &laswp_worker<T>);
  } else {
    laswp_range(0, n, A, lda, k1, k2, ipiv);
  }
}

// Recursive TRSM on GEMM:  B <- L^{-1} B,  L unit lower triangular (n x n).
template <typename T>
void trsm_llnu(int n, int nrhs, const T* L, int ldl, T* B, int ldb) {
  if (n <= 0 || nrhs <= 0) return;
  const int leaf = opts().trsm_leaf;
  if (leaf <= 0 || n <= leaf) {
    Tick tk(stats().trsm, stats().n_trsm);
    trsm(CblasLeft, CblasLower, CblasNoTrans, CblasUnit, n, nrhs, T(1), L, ldl, B, ldb);
    return;
  }
  const int n1 = n / 2, n2 = n - n1;
  trsm_llnu(n1, nrhs, L, ldl, B, ldb);
  {
    GemmTick tk(n2, nrhs, n1);
    gemm(CblasNoTrans, CblasNoTrans, n2, nrhs, n1, T(-1), L + n1, ldl, B, ldb, T(1),
         B + n1, ldb);
  }
  trsm_llnu(n2, nrhs, L + n1 + (size_t)n1 * ldl, ldl, B + n1, ldb);
}

// Recursive TRSM on GEMM:  B <- B L^{-T},  L lower triangular, non-unit.
template <typename T>
void trsm_rltn(int m, int n, const T* L, int ldl, T* B, int ldb) {
  if (n <= 0 || m <= 0) return;
  const int leaf = opts().trsm_leaf;
  if (leaf <= 0 || n <= leaf) {
    Tick tk(stats().trsm, stats().n_trsm);
    trsm(CblasRight, CblasLower, CblasTrans, CblasNonUnit, m, n, T(1), L, ldl, B, ldb);
    return;
  }
  const int n1 = n / 2, n2 = n - n1;
  trsm_rltn(m, n1, L, ldl, B, ldb);
  {
    GemmTick tk(m, n2, n1);
    // B2 -= X1 * Lb^T, Lb = L[n1:, :n1]
    gemm(CblasNoTrans, CblasTrans, m, n2, n1, T(-1), B, ldb, L + n1, ldl, T(1),
         B + (size_t)n1 * ldb, ldb);
  }
  trsm_rltn(m, n2, L + n1 + (size_t)n1 * ldl, ldl, B + (size_t)n1 * ldb, ldb);
}

// ---------------------------------------------------------------- LU leaves
// Unblocked right-looking LU of an m x n panel (n small). ipiv is 0-based.
template <typename T>
int getf2(int m, int n, T* A, int lda, int* ipiv) {
  Tick tk(stats().leaf, stats().n_leaf);
  int info = 0;
  const int mn = std::min(m, n);
  for (int j = 0; j < mn; ++j) {
    T* cj = A + (size_t)j * lda;
    const int p = j + iamax(m - j, cj + j);
    ipiv[j] = p;
    if (cj[p] != T(0)) {
      if (p != j) {
        for (int c = 0; c < n; ++c) {
          T* col = A + (size_t)c * lda;
          const T t = col[j];
          col[j] = col[p];
          col[p] = t;
        }
      }
      if (j + 1 < m) scal(m - j - 1, T(1) / cj[j], cj + j + 1);
    } else if (info == 0) {
      info = j + 1;
    }
    if (j + 1 < n && j + 1 < m) {
      // A[j+1:, j+1:] -= A[j+1:, j] * A[j, j+1:]
      ger(m - j - 1, n - j - 1, T(-1), cj + j + 1, A + j + (size_t)(j + 1) * lda, lda,
          A + (j + 1) + (size_t)(j + 1) * lda, lda);
    }
  }
  return info;
}

// --------------------------------------------------------------- recursive LU
// Toledo-style column recursion, in place, m x n. ipiv: 0-based row indices
// relative to row 0 of A (length min(m,n)). Returns LAPACK-style info.
template <typename T>
int getrf_rec(int m, int n, T* A, int lda, int* ipiv) {
  if (m <= 0 || n <= 0) return 0;
  const int mn = std::min(m, n);
  if (n <= opts().lu_leaf || mn == 1) return getf2(m, n, A, lda, ipiv);

  const int n1 = mn / 2, n2 = n - n1;
  T* A12 = A + (size_t)n1 * lda;
  T* A21 = A + n1;
  T* A22 = A12 + n1;

  int info = getrf_rec(m, n1, A, lda, ipiv);       // [A11; A21] = P1 L U11
  laswp_cols(n2, A12, lda, 0, n1, ipiv);           // apply P1 to the right half
  trsm_llnu(n1, n2, A, lda, A12, lda);             // U12 = L11^{-1} A12
  if (m > n1) {
    {
      GemmTick tk(m - n1, n2, n1);                 // A22 -= L21 U12
      gemm(CblasNoTrans, CblasNoTrans, m - n1, n2, n1, T(-1), A21, lda, A12, lda, T(1),
           A22, lda);
    }
    int info2 = getrf_rec(m - n1, n2, A22, lda, ipiv + n1);
    if (info == 0 && info2 > 0) info = info2 + n1;
    for (int i = n1; i < mn; ++i) ipiv[i] += n1;
    laswp_cols(n1, A, lda, n1, mn, ipiv);          // apply P2 to the left half
  }
  return info;
}

// ---------------------------------------------- blocked right-looking variant
// Right-looking blocked LU, block size nb, with the m x nb panel factored by
// the recursion above. Trailing updates are full-width rank-nb GEMMs (the
// shape Accelerate's GEMM handles best), TRSM is only nb/n of the work, and
// the interchanges for the already-factored left part are deferred and applied
// once per column at the end.
template <typename T>
int getrf_blk(int n, T* A, int lda, int* ipiv, int nb) {
  int info = 0;
  for (int j = 0; j < n; j += nb) {
    const int jb = std::min(nb, n - j);
    T* Ajj = A + j + (size_t)j * lda;
    int pinfo = getrf_rec(n - j, jb, Ajj, lda, ipiv + j);
    if (info == 0 && pinfo > 0) info = pinfo + j;
    const int nr = n - j - jb;
    if (nr > 0) {
      T* A12 = Ajj + (size_t)jb * lda;
      laswp_cols(nr, A12, lda, 0, jb, ipiv + j);   // local indices, rows from j
      trsm_llnu(jb, nr, Ajj, lda, A12, lda);
      GemmTick tk(nr, nr, jb);
      gemm(CblasNoTrans, CblasNoTrans, nr, nr, jb, T(-1), Ajj + jb, lda, A12, lda, T(1),
           A12 + jb, lda);
    }
    for (int i = j; i < j + jb; ++i) ipiv[i] += j;  // make absolute
  }
  // deferred interchanges for the columns left of each panel
  for (int j = 0; j + nb < n; j += nb) laswp_cols(nb, A + (size_t)j * lda, lda, j + nb, n, ipiv);
  return info;
}

// ------------------------------------------------------------- Cholesky bits
// Unblocked lower Cholesky (right-looking by columns), n small.
template <typename T>
int potf2(int n, T* A, int lda) {
  Tick tk(stats().leaf, stats().n_leaf);
  for (int j = 0; j < n; ++j) {
    T* cj = A + (size_t)j * lda;
    T d = cj[j];
    if (!(d > T(0))) return j + 1;
    d = std::sqrt(d);
    cj[j] = d;
    const T r = T(1) / d;
    for (int i = j + 1; i < n; ++i) cj[i] *= r;
    for (int c = j + 1; c < n; ++c) {
      T* cc = A + (size_t)c * lda;
      const T f = cj[c];
      for (int i = c; i < n; ++i) cc[i] -= cj[i] * f;
    }
  }
  return 0;
}

// C(lower) <- C - A A^T using GEMM only on off-diagonal blocks (never touches
// the strict upper triangle), SYRK on small diagonal blocks.
template <typename T>
void rsyrk_ln(int n, int k, const T* A, int lda, T* C, int ldc) {
  if (n <= opts().rsyrk_leaf) {
    if (opts().chol_update == 3) {
      // diagonal leaf through GEMM into scratch, then fold the lower triangle
      static thread_local std::vector<T> tmp;
      if (tmp.size() < (size_t)n * n) tmp.resize((size_t)n * n);
      {
        GemmTick tk(n, n, k);
        gemm(CblasNoTrans, CblasTrans, n, n, k, T(1), A, lda, A, lda, T(0), tmp.data(), n);
      }
      for (int j = 0; j < n; ++j) {
        T* c = C + (size_t)j * ldc;
        const T* t = tmp.data() + (size_t)j * n;
        for (int i = j; i < n; ++i) c[i] -= t[i];
      }
      return;
    }
    Tick tk(stats().syrk, stats().n_syrk);
    syrk(CblasLower, CblasNoTrans, n, k, T(-1), A, lda, T(1), C, ldc);
    return;
  }
  const int n1 = n / 2, n2 = n - n1;
  rsyrk_ln(n1, k, A, lda, C, ldc);
  {
    GemmTick tk(n2, n1, k);
    gemm(CblasNoTrans, CblasTrans, n2, n1, k, T(-1), A + n1, lda, A, lda, T(1), C + n1, ldc);
  }
  rsyrk_ln(n2, k, A + n1, lda, C + n1 + (size_t)n1 * ldc, ldc);
}

// Recursive lower Cholesky, in place. With chol_update == 1 (full GEMM) the
// strict upper triangle of A is overwritten with garbage.
template <typename T>
int potrf_rec(int n, T* A, int lda) {
  if (n <= 0) return 0;
  if (n <= opts().chol_leaf) return potf2(n, A, lda);
  const int n1 = n / 2, n2 = n - n1;
  T* A21 = A + n1;
  T* A22 = A + n1 + (size_t)n1 * lda;
  int info = potrf_rec(n1, A, lda);
  if (info) return info;
  trsm_rltn(n2, n1, A, lda, A21, lda);  // L21 = A21 L11^{-T}
  switch (opts().chol_update) {
    case 0: {
      Tick tk(stats().syrk, stats().n_syrk);
      syrk(CblasLower, CblasNoTrans, n2, n1, T(-1), A21, lda, T(1), A22, lda);
      break;
    }
    case 1: {
      GemmTick tk(n2, n2, n1);
      gemm(CblasNoTrans, CblasTrans, n2, n2, n1, T(-1), A21, lda, A21, lda, T(1), A22, lda);
      break;
    }
    default:
      rsyrk_ln(n2, n1, A21, lda, A22, lda);
  }
  info = potrf_rec(n2, A22, lda);
  return info ? info + n1 : 0;
}

// Blocked right-looking lower Cholesky: recursive factorisation of the nb x nb
// diagonal block, one TRSM, and a full-width rank-nb trailing update (SYRK,
// full GEMM or recursive SYRK according to chol_update).
template <typename T>
int potrf_blk(int n, T* A, int lda, int nb) {
  for (int j = 0; j < n; j += nb) {
    const int jb = std::min(nb, n - j);
    T* Ajj = A + j + (size_t)j * lda;
    int info = potrf_rec(jb, Ajj, lda);
    if (info) return info + j;
    const int nr = n - j - jb;
    if (nr <= 0) break;
    T* A21 = Ajj + jb;
    T* A22 = A21 + (size_t)jb * lda;
    trsm_rltn(nr, jb, Ajj, lda, A21, lda);
    switch (opts().chol_update) {
      case 0: {
        Tick tk(stats().syrk, stats().n_syrk);
        syrk(CblasLower, CblasNoTrans, nr, jb, T(-1), A21, lda, T(1), A22, lda);
        break;
      }
      case 1: {
        GemmTick tk(nr, nr, jb);
        gemm(CblasNoTrans, CblasTrans, nr, nr, jb, T(-1), A21, lda, A21, lda, T(1), A22, lda);
        break;
      }
      default:
        rsyrk_ln(nr, jb, A21, lda, A22, lda);
    }
  }
  return 0;
}

}  // namespace rf
