# T05 session contract: the public sample-fork declaration stays never-forks. No GPU needed.
ninfer_add_test(ninfer_spark2_5_session_contract_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_session_contract.cpp"
  LIBRARIES ninfer_model_execution)
