# scipy.ndimage native tests. They use the C ABI only, so they build in every
# configuration; MS_TEST_METAL adds the GPU-engine comparisons.
add_executable(ms_ndimage_filters_test tests/native/ndimage_filters_test.cpp)
target_link_libraries(ms_ndimage_filters_test PRIVATE mscore Threads::Threads)
if(MS_ENABLE_METAL)
  target_compile_definitions(ms_ndimage_filters_test PRIVATE MS_TEST_METAL=1)
endif()
ms_add_gpu_test(ndimage_filters ms_ndimage_filters_test PLAIN VALIDATED)
