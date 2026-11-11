ninfer_add_test(ninfer_spark2_5_config_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/load/test_config.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_spark2_5_bindings_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/load/test_bindings.cpp"
  LIBRARIES ninfer_model_loading)

# Reads the artifact named by NINFER_SPARK2_5_ARTIFACT and skips (77) when it is unset.
ninfer_add_test(ninfer_spark2_5_definition_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/load/test_definition.cpp"
  LIBRARIES ninfer_model_loading)
set_tests_properties(ninfer_spark2_5_definition_test PROPERTIES
  SKIP_RETURN_CODE 77 ENVIRONMENT "CUDA_VISIBLE_DEVICES=")

# Frontend, execution and engine tests are declared beside their sources.
include("${CMAKE_CURRENT_LIST_DIR}/frontend/tests.cmake" OPTIONAL)
include("${CMAKE_CURRENT_LIST_DIR}/execution/tests.cmake" OPTIONAL)
include("${CMAKE_CURRENT_LIST_DIR}/engine/tests.cmake" OPTIONAL)
