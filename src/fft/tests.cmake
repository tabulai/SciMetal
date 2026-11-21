# FFT tests: the planner (no device needed) and the GPU transforms against a
# double-precision reference, also under Metal API and shader validation.
add_executable(sm_fft_plan_test tests/native/fft_plan_test.cpp)
target_include_directories(sm_fft_plan_test PRIVATE src)
target_link_libraries(sm_fft_plan_test PRIVATE mscore_test)
sm_add_gpu_test(fft_plan sm_fft_plan_test PLAIN)

if(SM_ENABLE_METAL)
  add_executable(sm_fft_gpu_test tests/native/fft_gpu_test.cpp)
  target_include_directories(sm_fft_gpu_test PRIVATE src)
  target_link_libraries(sm_fft_gpu_test PRIVATE mscore_test Threads::Threads)
  sm_add_gpu_test(fft_gpu sm_fft_gpu_test PLAIN VALIDATED)
  set_tests_properties(fft_gpu fft_gpu_validated PROPERTIES TIMEOUT 600)
endif()
