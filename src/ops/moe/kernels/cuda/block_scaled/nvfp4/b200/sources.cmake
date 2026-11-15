target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/plan.cpp")
target_sources(ninfer_ops_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shared.cu")
