// E7 probe: C entry points (ctypes) around rfact.hpp + Accelerate baselines.
#include <mach/mach.h>
#include <sys/resource.h>

#include <cstdio>
#include <cstdlib>

#include "rfact.hpp"

namespace {

// ---- Accelerate LAPACK (new interface), typed overloads
inline void lapack_getrf(int n, double* A, int lda, int* ipiv, int* info) {
  __LAPACK_int n_ = n, lda_ = lda, info_ = 0;
  dgetrf_(&n_, &n_, A, &lda_, ipiv, &info_);
  *info = (int)info_;
}
inline void lapack_getrf(int n, float* A, int lda, int* ipiv, int* info) {
  __LAPACK_int n_ = n, lda_ = lda, info_ = 0;
  sgetrf_(&n_, &n_, A, &lda_, ipiv, &info_);
  *info = (int)info_;
}
inline void lapack_potrf(int n, double* A, int lda, int* info) {
  __LAPACK_int n_ = n, lda_ = lda, info_ = 0;
  dpotrf_("L", &n_, A, &lda_, &info_);
  *info = (int)info_;
}
inline void lapack_potrf(int n, float* A, int lda, int* info) {
  __LAPACK_int n_ = n, lda_ = lda, info_ = 0;
  spotrf_("L", &n_, A, &lda_, &info_);
  *info = (int)info_;
}

double cpu_seconds() {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec +
         ru.ru_stime.tv_usec * 1e-6;
}

// out[0] = wall seconds, out[1] = process CPU seconds (user+sys) during the call
template <typename F>
int timed(F&& f, double* out) {
  const double c0 = cpu_seconds();
  const double t0 = rf::now_s();
  int info = f();
  const double t1 = rf::now_s();
  const double c1 = cpu_seconds();
  if (out) {
    out[0] = t1 - t0;
    out[1] = c1 - c0;
  }
  return info;
}

template <typename T>
int getrf_lapack(int n, T* A, int lda, int* ipiv, double* out) {
  return timed([&] {
    int info = 0;
    lapack_getrf(n, A, lda, ipiv, &info);
    return info;
  }, out);
}

template <typename T>
int getrf_rec(int m, int n, T* A, int lda, int* ipiv, double* out) {
  // ipiv is returned 1-based (LAPACK convention).
  int info = timed([&] { return rf::getrf_rec(m, n, A, lda, ipiv); }, out);
  const int mn = m < n ? m : n;
  for (int i = 0; i < mn; ++i) ipiv[i] += 1;
  return info;
}

template <typename T>
int getrf_blk(int n, T* A, int lda, int* ipiv, int nb, double* out) {
  int info = timed([&] { return rf::getrf_blk(n, A, lda, ipiv, nb); }, out);
  for (int i = 0; i < n; ++i) ipiv[i] += 1;
  return info;
}

template <typename T>
int potrf_lapack(int n, T* A, int lda, double* out) {
  return timed([&] {
    int info = 0;
    lapack_potrf(n, A, lda, &info);
    return info;
  }, out);
}

template <typename T>
int potrf_rec(int n, T* A, int lda, double* out) {
  return timed([&] { return rf::potrf_rec(n, A, lda); }, out);
}

template <typename T>
int potrf_blk(int n, T* A, int lda, int nb, double* out) {
  return timed([&] { return rf::potrf_blk(n, A, lda, nb); }, out);
}

// ||P A0 - L U||_F / ||A0||_F  (ipiv 1-based). Uses 3 n x n temporaries.
// The product is formed with the inner index reversed ((L J)(J U)) so that the
// rounding in this check is not correlated with the k-order of the GEMM calls
// inside the factorisation being checked.
template <typename T>
double lu_resid(int n, const T* A0, int lda0, const T* LU, int ldlu, const int* ipiv) {
  const size_t N = (size_t)n * n;
  T* L = (T*)calloc(N, sizeof(T));
  T* U = (T*)calloc(N, sizeof(T));
  T* R = (T*)malloc(N * sizeof(T));
  for (int j = 0; j < n; ++j) {
    const T* c = LU + (size_t)j * ldlu;
    const int jr = n - 1 - j;  // reversed inner index
    for (int i = 0; i <= j; ++i) U[(n - 1 - i) + (size_t)j * n] = c[i];   // (J U)
    L[j + (size_t)jr * n] = T(1);                                          // (L J)
    for (int i = j + 1; i < n; ++i) L[i + (size_t)jr * n] = c[i];
  }
  // R = P A0
  std::vector<int> perm(n);
  for (int i = 0; i < n; ++i) perm[i] = i;
  for (int i = 0; i < n; ++i) std::swap(perm[i], perm[ipiv[i] - 1]);
  double na = 0;
  for (int j = 0; j < n; ++j) {
    const T* c = A0 + (size_t)j * lda0;
    T* r = R + (size_t)j * n;
    for (int i = 0; i < n; ++i) {
      r[i] = c[perm[i]];
      na += (double)c[i] * (double)c[i];
    }
  }
  rf::gemm(CblasNoTrans, CblasNoTrans, n, n, n, T(-1), L, n, U, n, T(1), R, n);
  double nr = 0;
  for (size_t i = 0; i < N; ++i) nr += (double)R[i] * (double)R[i];
  free(L);
  free(U);
  free(R);
  return std::sqrt(nr) / std::sqrt(na);
}

// ||A0 - L L^T||_F / ||A0||_F  (A0 full symmetric; only the lower part of Lm is read).
// Inner index reversed, as in lu_resid.
template <typename T>
double chol_resid(int n, const T* A0, int lda0, const T* Lm, int ldl) {
  const size_t N = (size_t)n * n;
  T* L = (T*)calloc(N, sizeof(T));
  T* R = (T*)malloc(N * sizeof(T));
  double na = 0;
  for (int j = 0; j < n; ++j) {
    const T* c = Lm + (size_t)j * ldl;
    for (int i = j; i < n; ++i) L[i + (size_t)(n - 1 - j) * n] = c[i];
    const T* a = A0 + (size_t)j * lda0;
    T* r = R + (size_t)j * n;
    for (int i = 0; i < n; ++i) {
      r[i] = a[i];
      na += (double)a[i] * (double)a[i];
    }
  }
  rf::gemm(CblasNoTrans, CblasTrans, n, n, n, T(-1), L, n, L, n, T(1), R, n);
  double nr = 0;
  for (size_t i = 0; i < N; ++i) nr += (double)R[i] * (double)R[i];
  free(L);
  free(R);
  return std::sqrt(nr) / std::sqrt(na);
}

// ---------------------------------------------------------------------------
// Trusted residual estimate, independent of BLAS summation order.
//
// A residual formed with GEMM in working precision has a rounding floor of the
// same size as the backward error being measured, and that floor is strongly
// correlated with the k-order of the GEMMs inside the factorisation (checking a
// GEMM-built factor with GEMM in the same order reports ~1e-16, in reversed
// order ~1.7e-15, for the very same factor). So the reported number is a
// Hutchinson estimate  ||R||_F ~= ||R Z||_F / sqrt(q),  Z = n x q of +-1,
// with every product accumulated in double-double (error-free TwoSum/TwoProd),
// i.e. R Z is evaluated to ~1e-30 relative. Statistical error ~ 1/sqrt(2q).
// ---------------------------------------------------------------------------
static inline void acc_dd(double& hi, double& lo, double x) {
  const double s = hi + x;
  const double bb = s - hi;
  lo += (hi - (s - bb)) + (x - bb);
  hi = s;
}
static inline void acc_prod_dd(double& hi, double& lo, double a, double b) {
  const double p = a * b;
  lo += std::fma(a, b, -p);
  acc_dd(hi, lo, p);
}

struct ProbeCtx {
  int n, q, lda0, ldf;
  const void* A0;
  const void* F;
  const int* ipiv;  // LU: 1-based pivots; Cholesky: nullptr
  int is_double;
  double* out;      // per-probe sum of squares
};

template <typename T>
void probe_worker(void* vctx, size_t idx) {
  const ProbeCtx& c = *(const ProbeCtx*)vctx;
  const int n = c.n;
  const T* A0 = (const T*)c.A0;
  const T* F = (const T*)c.F;
  std::vector<double> z(n), uh(n, 0.0), ul(n, 0.0), wh(n, 0.0), wl(n, 0.0), yh(n, 0.0), yl(n, 0.0);
  uint64_t x = 0x9E3779B97F4A7C15ull * (idx + 1) + 0xD1B54A32D192ED03ull;
  for (int i = 0; i < n; ++i) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    z[i] = (x >> 33) & 1 ? 1.0 : -1.0;
  }
  // y = A0 z
  for (int j = 0; j < n; ++j) {
    const T* col = A0 + (size_t)j * c.lda0;
    const double zj = z[j];
    for (int i = 0; i < n; ++i) acc_dd(yh[i], yl[i], zj * (double)col[i]);
  }
  if (c.ipiv) {
    // u = U z (U = upper triangle of F), w = L u (L unit lower), y <- P y
    for (int j = 0; j < n; ++j) {
      const T* col = F + (size_t)j * c.ldf;
      const double zj = z[j];
      for (int i = 0; i <= j; ++i) acc_dd(uh[i], ul[i], zj * (double)col[i]);
    }
    for (int i = 0; i < n; ++i) { wh[i] = uh[i]; wl[i] = ul[i]; }
    for (int j = 0; j < n; ++j) {
      const T* col = F + (size_t)j * c.ldf;
      const double a = uh[j], b = ul[j];
      for (int i = j + 1; i < n; ++i) {
        const double l = (double)col[i];
        acc_prod_dd(wh[i], wl[i], l, a);
        wl[i] += l * b;
      }
    }
    for (int i = 0; i < n; ++i) {
      const int p = c.ipiv[i] - 1;
      std::swap(yh[i], yh[p]);
      std::swap(yl[i], yl[p]);
    }
  } else {
    // u = L^T z, w = L u (L = lower triangle of F, non-unit)
    for (int j = 0; j < n; ++j) {
      const T* col = F + (size_t)j * c.ldf;
      double h = 0, l = 0;
      for (int i = j; i < n; ++i) acc_dd(h, l, z[i] * (double)col[i]);
      uh[j] = h; ul[j] = l;
    }
    for (int j = 0; j < n; ++j) {
      const T* col = F + (size_t)j * c.ldf;
      const double a = uh[j], b = ul[j];
      for (int i = j; i < n; ++i) {
        const double l = (double)col[i];
        acc_prod_dd(wh[i], wl[i], l, a);
        wl[i] += l * b;
      }
    }
  }
  double ss = 0;
  for (int i = 0; i < n; ++i) {
    const double d = (yh[i] - wh[i]) + (yl[i] - wl[i]);
    ss += d * d;
  }
  c.out[idx] = ss;
}

template <typename T>
double resid_probe(int n, const T* A0, int lda0, const T* F, int ldf, const int* ipiv, int q) {
  std::vector<double> ss(q, 0.0);
  ProbeCtx ctx{n, q, lda0, ldf, A0, F, ipiv, sizeof(T) == 8, ss.data()};
  dispatch_apply_f((size_t)q, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &ctx,
                   &probe_worker<T>);
  double nr = 0, na = 0;
  for (double v : ss) nr += v;
  for (int j = 0; j < n; ++j) {
    const T* col = A0 + (size_t)j * lda0;
    for (int i = 0; i < n; ++i) na += (double)col[i] * (double)col[i];
  }
  return std::sqrt(nr / q) / std::sqrt(na);
}

template <typename T>
void gemm_bench(int n, const T* A, const T* B, T* C, double* out) {
  timed([&] {
    rf::gemm(CblasNoTrans, CblasNoTrans, n, n, n, T(1), A, n, B, n, T(0), C, n);
    return 0;
  }, out);
}

// Individual BLAS3 pieces at the shapes the recursion uses (n x n operands).
template <typename T>
void trsm_bench(int n, const T* L, T* B, double* out) {
  timed([&] {
    rf::trsm(CblasLeft, CblasLower, CblasNoTrans, CblasUnit, n, n, T(1), L, n, B, n);
    return 0;
  }, out);
}
template <typename T>
void syrk_bench(int n, const T* A, T* C, double* out) {
  timed([&] {
    rf::syrk(CblasLower, CblasNoTrans, n, n, T(-1), A, n, T(1), C, n);
    return 0;
  }, out);
}

// GEMM rate at an arbitrary shape: C(m x n) -= A(m x k) B(k x n).
// out[0] = median wall seconds, out[1] = min wall seconds, out[2] = cpu/wall.
template <typename T>
void gemm_shape(int m, int n, int k, int reps, double* out) {
  std::vector<T> A((size_t)m * k), B((size_t)k * n), C((size_t)m * n);
  for (size_t i = 0; i < A.size(); ++i) A[i] = T((i * 2654435761u % 1000) * 1e-3 - 0.5);
  for (size_t i = 0; i < B.size(); ++i) B[i] = T((i * 40503u % 1000) * 1e-3 - 0.5);
  for (size_t i = 0; i < C.size(); ++i) C[i] = T(1);
  std::vector<double> ts;
  double cpu = 0, wall = 0;
  for (int r = 0; r < reps + 1; ++r) {
    double o[2];
    timed([&] {
      rf::gemm(CblasNoTrans, CblasNoTrans, m, n, k, T(-1), A.data(), m, B.data(), k, T(1),
               C.data(), m);
      return 0;
    }, o);
    if (r == 0) continue;  // warm-up
    ts.push_back(o[0]);
    wall += o[0];
    cpu += o[1];
  }
  std::sort(ts.begin(), ts.end());
  out[0] = ts[ts.size() / 2];
  out[1] = ts[0];
  out[2] = cpu / wall;
}

// Rank-k symmetric update C(n x n) -= A(n x k) A^T: SYRK (lower) vs full GEMM.
// out[0] = median wall seconds, out[1] = min, out[2] = cpu/wall.
void rankk_d(int n, int k, int use_syrk, int reps, double* out) {
  std::vector<double> A((size_t)n * k), C((size_t)n * n, 1.0);
  for (size_t i = 0; i < A.size(); ++i) A[i] = (i * 2654435761u % 1000) * 1e-6;
  std::vector<double> ts;
  double cpu = 0, wall = 0;
  for (int r = 0; r < reps + 1; ++r) {
    double o[2];
    timed([&] {
      if (use_syrk)
        cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, n, k, -1.0, A.data(), n, 1.0,
                    C.data(), n);
      else
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, k, -1.0, A.data(), n,
                    A.data(), n, 1.0, C.data(), n);
      return 0;
    }, o);
    if (r == 0) continue;
    ts.push_back(o[0]);
    wall += o[0];
    cpu += o[1];
  }
  std::sort(ts.begin(), ts.end());
  out[0] = ts[ts.size() / 2];
  out[1] = ts[0];
  out[2] = cpu / wall;
}

}  // namespace

extern "C" {

void e7_rankk_d(int n, int k, int use_syrk, int reps, double* out) { rankk_d(n, k, use_syrk, reps, out); }

void e7_gemm_shape_d(int m, int n, int k, int reps, double* out) { gemm_shape<double>(m, n, k, reps, out); }
void e7_gemm_shape_s(int m, int n, int k, int reps, double* out) { gemm_shape<float>(m, n, k, reps, out); }


void e7_set_swap_par(int on) { rf::opts().swap_par = on; }

void e7_set_opts(int lu_leaf, int trsm_leaf, int chol_leaf, int chol_update, int rsyrk_leaf,
                 int timers) {
  rf::Opts& o = rf::opts();
  o.lu_leaf = lu_leaf;
  o.trsm_leaf = trsm_leaf;
  o.chol_leaf = chol_leaf;
  o.chol_update = chol_update;
  o.rsyrk_leaf = rsyrk_leaf;
  o.timers = timers;
}
void e7_clear_stats(void) { rf::stats().clear(); }
// out: gemm, trsm, swap, leaf, syrk seconds, then the five call counts
void e7_get_stats(double* out) {
  const rf::Stats& s = rf::stats();
  out[0] = s.gemm; out[1] = s.trsm; out[2] = s.swap; out[3] = s.leaf; out[4] = s.syrk;
  out[5] = (double)s.n_gemm; out[6] = (double)s.n_trsm; out[7] = (double)s.n_swap;
  out[8] = (double)s.n_leaf; out[9] = (double)s.n_syrk;
}

// out[0..19]: GEMM seconds per floor(log2(K)); out[20..39]: flops per level
void e7_get_gemm_levels(double* out) {
  const rf::Stats& s = rf::stats();
  for (int i = 0; i < 20; ++i) {
    out[i] = s.gemm_lvl_s[i];
    out[20 + i] = s.gemm_lvl_flops[i];
  }
}

int e7_thread_count(void) {
  thread_act_array_t list;
  mach_msg_type_number_t count = 0;
  if (task_threads(mach_task_self(), &list, &count) != KERN_SUCCESS) return -1;
  for (mach_msg_type_number_t i = 0; i < count; ++i)
    mach_port_deallocate(mach_task_self(), list[i]);
  vm_deallocate(mach_task_self(), (vm_address_t)list, count * sizeof(thread_act_t));
  return (int)count;
}

#define E7_EXPORT(sfx, T)                                                              \
  int e7_getrf_lapack_##sfx(int n, T* A, int lda, int* ipiv, double* out) {            \
    return getrf_lapack<T>(n, A, lda, ipiv, out);                                      \
  }                                                                                    \
  int e7_getrf_rec_##sfx(int m, int n, T* A, int lda, int* ipiv, double* out) {        \
    return getrf_rec<T>(m, n, A, lda, ipiv, out);                                      \
  }                                                                                    \
  int e7_getrf_blk_##sfx(int n, T* A, int lda, int* ipiv, int nb, double* out) {       \
    return getrf_blk<T>(n, A, lda, ipiv, nb, out);                                     \
  }                                                                                    \
  int e7_potrf_lapack_##sfx(int n, T* A, int lda, double* out) {                       \
    return potrf_lapack<T>(n, A, lda, out);                                            \
  }                                                                                    \
  int e7_potrf_blk_##sfx(int n, T* A, int lda, int nb, double* out) {                  \
    return potrf_blk<T>(n, A, lda, nb, out);                                           \
  }                                                                                    \
  int e7_potrf_rec_##sfx(int n, T* A, int lda, double* out) {                          \
    return potrf_rec<T>(n, A, lda, out);                                               \
  }                                                                                    \
  double e7_lu_resid_##sfx(int n, const T* A0, int lda0, const T* LU, int ldlu,        \
                           const int* ipiv) {                                          \
    return lu_resid<T>(n, A0, lda0, LU, ldlu, ipiv);                                   \
  }                                                                                    \
  double e7_chol_resid_##sfx(int n, const T* A0, int lda0, const T* L, int ldl) {      \
    return chol_resid<T>(n, A0, lda0, L, ldl);                                         \
  }                                                                                    \
  double e7_lu_resid_probe_##sfx(int n, const T* A0, int lda0, const T* LU, int ldlu,  \
                                 const int* ipiv, int q) {                             \
    return resid_probe<T>(n, A0, lda0, LU, ldlu, ipiv, q);                          \
  }                                                                                    \
  double e7_chol_resid_probe_##sfx(int n, const T* A0, int lda0, const T* L, int ldl,  \
                                   int q) {                                            \
    return resid_probe<T>(n, A0, lda0, L, ldl, nullptr, q);                                \
  }                                                                                    \
  void e7_gemm_##sfx(int n, const T* A, const T* B, T* C, double* out) {               \
    gemm_bench<T>(n, A, B, C, out);                                                    \
  }                                                                                    \
  void e7_trsm_##sfx(int n, const T* L, T* B, double* out) {                           \
    trsm_bench<T>(n, L, B, out);                                                       \
  }                                                                                    \
  void e7_syrk_##sfx(int n, const T* A, T* C, double* out) {                           \
    syrk_bench<T>(n, A, C, out);                                                       \
  }

E7_EXPORT(d, double)
E7_EXPORT(s, float)

}  // extern "C"
