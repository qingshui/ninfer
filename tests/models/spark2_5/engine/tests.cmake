# Engine-level Spark2_5 cases. They share the 2-layer fixture of the execution tests; the capacity
# case needs the full-depth fixture so planned KV and State geometry equal the real model's.
set(ninfer_spark2_5_full_fixture_dir "${CMAKE_CURRENT_BINARY_DIR}/spark2_5_engine_full_fixture")

add_executable(ninfer_spark2_5_engine_test "${CMAKE_CURRENT_LIST_DIR}/test_engine.cpp")
ninfer_test_includes(ninfer_spark2_5_engine_test)
target_link_libraries(ninfer_spark2_5_engine_test PRIVATE ninfer_engine)

add_test(NAME ninfer_spark2_5_engine_full_fixture_build
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/make_full_fixture.py"
    "${ninfer_spark2_5_full_fixture_dir}")
set_tests_properties(ninfer_spark2_5_engine_full_fixture_build PROPERTIES
  FIXTURES_SETUP ninfer_spark2_5_engine_full_fixture
  SKIP_RETURN_CODE 77
  TIMEOUT 1800)

foreach(case IN ITEMS graph16 capacity scoring)
  add_test(NAME ninfer_spark2_5_engine_${case}_test
    COMMAND ninfer_spark2_5_engine_test ${case})
  set(fixtures ninfer_spark2_5_execution_fixture)
  if(case STREQUAL "capacity")
    set(fixtures ninfer_spark2_5_engine_full_fixture)
  endif()
  set_tests_properties(ninfer_spark2_5_engine_${case}_test PROPERTIES
    FIXTURES_REQUIRED "${fixtures}"
    ENVIRONMENT
      "NINFER_SPARK2_5_FIXTURE=${ninfer_spark2_5_fixture_dir};NINFER_SPARK2_5_FULL_FIXTURE=${ninfer_spark2_5_full_fixture_dir}"
    SKIP_RETURN_CODE 77
    TIMEOUT 900
    RUN_SERIAL TRUE
    LABELS "gpu")
endforeach()
