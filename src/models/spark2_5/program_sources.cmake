target_sources(ninfer_model_execution PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/profile.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/startup.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/state/state_image.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/buffers.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/execution.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/graph_preparation.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/graphs.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/planning.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/resources.cpp")
