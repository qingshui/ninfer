target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n13312_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/linear_geglu.cpp"
)
