// E1 probe: a minimal standalone Metal runtime that dispatches MLX's steel
// fused GEMM kernel from our own metallib. Plain C entry points (ctypes).
//
// All matrices are ROW-MAJOR as seen by the kernel, addressed as
// (MTLBuffer, element offset, leading dimension) so that sub-matrix views of a
// larger buffer can be used as operands and as the in-place destination.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns 0 on success. err (optional) receives a message on failure.
int msg_init(const char* metallib_path, char* err, int errlen);
const char* msg_device_name(void);
const char* msg_device_arch(void);

// Shared-storage MTLBuffer. `pad_bytes` extra bytes are allocated past nbytes
// (the steel epilogue may read a few rows past an unaligned edge tile).
void* msg_alloc(size_t nbytes);
void* msg_alloc_padded(size_t nbytes, size_t pad_bytes);
void* msg_contents(void* buf);
size_t msg_length(void* buf);
void msg_free(void* buf);

typedef struct {
  void* A; size_t a_off; int lda; int ta;  // op(A): M x K
  void* B; size_t b_off; int ldb; int tb;  // op(B): K x N
  void* C; size_t c_off; int ldc;          // optional addend, M x N (NULL -> none)
  void* D; size_t d_off; int ldd;          // destination, M x N (may alias C)
  int M, N, K;
  float alpha, beta;                        // D = alpha*op(A)*op(B) + beta*C
  int variant;                              // 0 = MLX's choice for this device
} msg_gemm_args;

// variant ids (bm,bn,bk,wm,wn):
//  1: 64,64,16,2,2   2: 64,32,32,2,2   3: 32,64,16,1,2   4: 32,32,16,2,2

// Encode + commit, do not wait. Returns an opaque command-buffer handle.
void* msg_gemm_submit(const msg_gemm_args* a);
// Encode several GEMMs into ONE command buffer + commit.
void* msg_gemm_submit_many(const msg_gemm_args* a, int count);
// waitUntilCompleted + release. Returns 0 on success; gpu_seconds (optional)
// receives GPUEndTime - GPUStartTime.
int msg_wait(void* cb, double* gpu_seconds);
// Non-blocking: 1 if the command buffer has completed (or failed).
int msg_done(void* cb);

// Synchronous convenience: encode + commit + waitUntilCompleted.
// times (optional, 4 doubles): encode, commit, wait, gpu (seconds).
int msg_gemm(const msg_gemm_args* a, double* times);

// Probe variant: busy-poll the status instead of waitUntilCompleted.
int msg_gemm_spin(const msg_gemm_args* a, double* times);

// Last pipeline chosen (for reporting): kernel name incl. specialisation.
const char* msg_last_kernel(void);
// Grid / threadgroup of the last dispatch: tn, tm, 32, wn, wm
void msg_last_dispatch(int out[5]);

double msg_now(void);
// QoS of the calling thread: -1 default, 1 user-initiated, 2 user-interactive.
int msg_set_qos(int level);

#ifdef __cplusplus
}
#endif
