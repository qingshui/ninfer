# Text execution against independent FP64 references. The 2-layer fixture is built once from the
# checkpoints named by NINFER_SPARK_ROOT (exit 77 when absent) and shared by the cases below.
set(ninfer_spark2_5_fixture_dir "${CMAKE_CURRENT_BINARY_DIR}/spark2_5_execution_fixture")

add_executable(ninfer_spark2_5_execution_test "${CMAKE_CURRENT_LIST_DIR}/test_execution.cpp")
ninfer_test_includes(ninfer_spark2_5_execution_test)
target_link_libraries(ninfer_spark2_5_execution_test PRIVATE
  ninfer_model_execution ninfer_model_loading ninfer_core ninfer_op_test_support)
ninfer_op_oracle_options(ninfer_spark2_5_execution_test)

add_test(NAME ninfer_spark2_5_execution_fixture_build
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/make_fixture.py"
    "${ninfer_spark2_5_fixture_dir}")
set_tests_properties(ninfer_spark2_5_execution_fixture_build PROPERTIES
  FIXTURES_SETUP ninfer_spark2_5_execution_fixture
  SKIP_RETURN_CODE 77
  TIMEOUT 900)

foreach(case IN ITEMS layers_bf16 layers_fp8 lengths_bf16 lengths_fp8 window)
  if(case STREQUAL "layers_bf16")
    set(arguments layers --variant bf16)
  elseif(case STREQUAL "layers_fp8")
    set(arguments layers --variant fp8)
  elseif(case STREQUAL "lengths_bf16")
    set(arguments lengths --variant bf16)
  elseif(case STREQUAL "lengths_fp8")
    set(arguments lengths --variant fp8)
  else()
    set(arguments window)
  endif()
  add_test(NAME ninfer_spark2_5_execution_${case}_test
    COMMAND ninfer_spark2_5_execution_test ${arguments})
  set_tests_properties(ninfer_spark2_5_execution_${case}_test PROPERTIES
    FIXTURES_REQUIRED ninfer_spark2_5_execution_fixture
    ENVIRONMENT "NINFER_SPARK2_5_FIXTURE=${ninfer_spark2_5_fixture_dir}"
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
    RUN_SERIAL TRUE
    LABELS "gpu")
endforeach()

# A converted checkpoint named by NINFER_SPARK2_5_ARTIFACT; skipped (77) when unset.
add_test(NAME ninfer_spark2_5_execution_real_test
  COMMAND ninfer_spark2_5_execution_test real)
set_tests_properties(ninfer_spark2_5_execution_real_test PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 1800
  RUN_SERIAL TRUE
  LABELS "gpu;real")

# Model-level parity with the HF fp32 forward on the two full checkpoints. The fixture converts both
# checkpoints with the official recipes and writes both references (exit 77 when absent).
set(ninfer_spark2_5_hf_fixture_dir "${CMAKE_CURRENT_BINARY_DIR}/spark2_5_hf_fixture")

add_test(NAME ninfer_spark2_5_hf_fixture_build
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/../numerics/build_hf_fixture.py"
    "${ninfer_spark2_5_hf_fixture_dir}")
set_tests_properties(ninfer_spark2_5_hf_fixture_build PROPERTIES
  FIXTURES_SETUP ninfer_spark2_5_hf_fixture
  SKIP_RETURN_CODE 77
  TIMEOUT 3600)

foreach(variant IN ITEMS bf16 fp8)
  add_test(NAME ninfer_spark2_5_execution_hf_${variant}_test
    COMMAND ninfer_spark2_5_execution_test hf --variant ${variant})
  set_tests_properties(ninfer_spark2_5_execution_hf_${variant}_test PROPERTIES
    FIXTURES_REQUIRED ninfer_spark2_5_hf_fixture
    ENVIRONMENT
      "NINFER_SPARK2_5_HF_REFERENCE=${ninfer_spark2_5_hf_fixture_dir}/reference;NINFER_SPARK2_5_HF_ARTIFACTS=${ninfer_spark2_5_hf_fixture_dir}/artifacts"
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
    RUN_SERIAL TRUE
    LABELS "gpu;real")
endforeach()
