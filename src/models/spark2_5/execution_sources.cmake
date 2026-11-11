target_sources(ninfer_model_execution PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/resident.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/text.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/workspace.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/state/kv_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/state/window_state.cpp")
