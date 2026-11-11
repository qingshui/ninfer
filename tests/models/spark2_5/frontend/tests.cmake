ninfer_add_test(ninfer_spark2_5_chat_template_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_chat_template.cpp"
  LIBRARIES ninfer_model_control ninfer_jinja ninfer::json
  NEEDS_SOURCE_DIR)

# Needs the checkpoint tokenizer named by NINFER_SPARK2_5_RESOURCES; skipped (77) otherwise.
ninfer_add_test(ninfer_spark2_5_token_golden_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_token_golden.cpp"
  LIBRARIES ninfer_model_control ninfer::json
  NEEDS_SOURCE_DIR)
set_tests_properties(ninfer_spark2_5_token_golden_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_spark2_5_preparation_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_preparation.cpp"
  LIBRARIES ninfer_model_control ninfer::json
  NEEDS_SOURCE_DIR)

ninfer_add_test(ninfer_spark2_5_output_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_output.cpp"
  LIBRARIES ninfer_model_control ninfer::json
  NEEDS_SOURCE_DIR)

ninfer_add_test(ninfer_spark2_5_protocol_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_protocol.cpp"
  LIBRARIES ninfer_model_control ninfer_serve ninfer::json
  NEEDS_SOURCE_DIR)

ninfer_add_test(ninfer_spark2_5_t5_f6_frontend_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_t5_f6_frontend.cpp"
  LIBRARIES ninfer_model_control ninfer::json
  NEEDS_SOURCE_DIR)
