# scipy.ndimage native tests. They use the C ABI only, so they build in every
# configuration; SM_TEST_METAL adds the GPU-engine comparisons.
add_executable(sm_ndimage_filters_test tests/native/ndimage_filters_test.cpp)
target_link_libraries(sm_ndimage_filters_test PRIVATE smcore Threads::Threads)
if(SM_ENABLE_METAL)
  target_compile_definitions(sm_ndimage_filters_test PRIVATE SM_TEST_METAL=1)
endif()
sm_add_gpu_test(ndimage_filters sm_ndimage_filters_test PLAIN VALIDATED)

add_executable(sm_ndimage_dense_test tests/native/ndimage_dense_test.cpp)
target_include_directories(sm_ndimage_dense_test PRIVATE src)
target_link_libraries(sm_ndimage_dense_test PRIVATE mscore_test Threads::Threads)
if(SM_ENABLE_METAL)
  target_compile_definitions(sm_ndimage_dense_test PRIVATE SM_TEST_METAL=1 METALCPP_SYMBOL_VISIBILITY_HIDDEN=1)
  target_include_directories(sm_ndimage_dense_test SYSTEM PRIVATE "${SM_METAL_CPP_DIR}")
endif()
sm_add_gpu_test(ndimage_dense sm_ndimage_dense_test PLAIN VALIDATED)
