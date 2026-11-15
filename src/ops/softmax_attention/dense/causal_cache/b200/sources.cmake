# B200 causal attention: the tcgen05 FMHA (shared with no other target) and its host plan.
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/plan.cpp"
)

target_sources(ninfer_ops_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/../../../kernels/causal/sm100/launch.cu"
)
