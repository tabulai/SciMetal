// E1 probe: Accelerate CBLAS references (new LAPACK/CBLAS interface).
#include <Accelerate/Accelerate.h>

extern "C" {

// Row-major C = alpha*A*B + beta*C, with leading dimensions.
void e1_sgemm(int M, int N, int K, float alpha, const float* A, int lda,
              const float* B, int ldb, float beta, float* C, int ldc) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda,
              B, ldb, beta, C, ldc);
}

void e1_dgemm(int M, int N, int K, double alpha, const double* A, int lda,
              const double* B, int ldb, double beta, double* C, int ldc) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda,
              B, ldb, beta, C, ldc);
}

}  // extern "C"
