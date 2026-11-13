# Resources are read by the Definition; everything else belongs to the control plane.
target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/frontend/resources.cpp")

target_sources(ninfer_model_control PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/frontend/chat_template.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/prompt_layout.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/frontend.cpp")
