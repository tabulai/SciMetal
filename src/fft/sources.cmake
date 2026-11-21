# FFT family: planner and C ABI in every configuration, the GPU engine and the
# kernels (vendored MLX kernels plus our strided pass) with Metal.
list(APPEND SM_FAMILY_SOURCES src/fft/plan.cpp src/fft/fft.cpp)
list(APPEND SM_FAMILY_GPU_SOURCES src/fft/gpu_engine.cpp)
list(APPEND SM_METAL_SOURCES src/fft/fft_kernels.metal)
list(APPEND SM_METAL_INCLUDE_DIRS third_party/mlx-kernels src/fft)
