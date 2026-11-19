# scipy.signal: N-D convolution and correlation (direct + GPU FFT/OA).
list(APPEND MS_FAMILY_SOURCES src/signal/convolve.cpp)
list(APPEND MS_FAMILY_GPU_SOURCES src/signal/fft_conv_gpu.cpp)
list(APPEND MS_METAL_SOURCES src/signal/signal_kernels.metal)
list(APPEND MS_METAL_INCLUDE_DIRS src/signal)
