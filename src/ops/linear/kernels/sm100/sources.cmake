# tcgen05 dense GEMM. Only the B200 target (sm_100a) compiles these translation units.
target_sources(ninfer_ops_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_gemm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16_gemm.cu"
)
