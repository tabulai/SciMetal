# scipy.ndimage: separable N-D filters.
list(APPEND SM_FAMILY_SOURCES src/ndimage/filters.cpp src/ndimage/filters_cpu.cpp)
list(APPEND SM_FAMILY_GPU_SOURCES src/ndimage/filters_gpu.cpp)
list(APPEND SM_METAL_SOURCES src/ndimage/ndimage_filters.metal)
list(APPEND SM_METAL_INCLUDE_DIRS src/ndimage)
