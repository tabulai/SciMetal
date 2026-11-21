# scipy.signal: N-D convolution and correlation (direct + GPU FFT/OA).
list(APPEND SM_FAMILY_SOURCES src/signal/convolve.cpp)
list(APPEND SM_FAMILY_GPU_SOURCES src/signal/fft_conv_gpu.cpp)
list(APPEND SM_METAL_SOURCES src/signal/signal_kernels.metal)
list(APPEND SM_METAL_INCLUDE_DIRS src/signal)
