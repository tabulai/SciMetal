# scipy.signal native tests (PLAIN and VALIDATED).
add_executable(sm_signal_convolve_test tests/native/signal_convolve_test.cpp)
target_include_directories(sm_signal_convolve_test PRIVATE src)
target_link_libraries(sm_signal_convolve_test PRIVATE mscore_test Threads::Threads)
if(SM_ENABLE_METAL)
  target_compile_definitions(sm_signal_convolve_test PRIVATE SM_TEST_METAL=1 METALCPP_SYMBOL_VISIBILITY_HIDDEN=1)
  target_include_directories(sm_signal_convolve_test SYSTEM PRIVATE "${SM_METAL_CPP_DIR}")
endif()
sm_add_gpu_test(signal_convolve sm_signal_convolve_test PLAIN VALIDATED)
