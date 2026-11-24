# scipy.ndimage: separable, dense, composite, morphology, and rank filters.
list(APPEND SM_FAMILY_SOURCES
  src/ndimage/filters.cpp src/ndimage/filters_cpu.cpp
  src/ndimage/morphology.cpp src/ndimage/morphology_cpu.cpp)
list(APPEND SM_FAMILY_GPU_SOURCES src/ndimage/filters_gpu.cpp src/ndimage/morphology_gpu.cpp)
list(APPEND SM_METAL_SOURCES src/ndimage/ndimage_filters.metal src/ndimage/morphology_kernels.metal)
list(APPEND SM_METAL_INCLUDE_DIRS src/ndimage)
