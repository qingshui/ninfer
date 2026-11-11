# T3 / F3 tests. Include from ../tests.cmake with:
#   include("${CMAKE_CURRENT_LIST_DIR}/t3/tests.cmake")
ninfer_add_op_test(ninfer_t3_causal_swa_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_t3_causal_swa.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_t3_full_h8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_t3_full_h8.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../causal/cache.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../causal/device_cache.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../causal/oracle.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../causal/runner.cpp"
  LIBRARIES ninfer_ops)
