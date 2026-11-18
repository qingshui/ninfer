target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/linear_add.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n5120_k17408.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n5120_k17408.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q5/shapes/n5120_k17408.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q5/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n2048_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n2048_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n5120_k17408.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n5120_k6144.cu"
)

target_sources(ninfer_ops_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n5120_k17408_tma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n5120_k6144_tma.cu"
)

# The 122B o_proj / GDN out shapes have B200 variants that run on tcgen05 at every token count.
if(NINFER_TARGET STREQUAL "b200")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/bf16/b200/shapes/n3072_k8192.cu"
    "${CMAKE_CURRENT_LIST_DIR}/fp8/b200/shapes/n3072_k8192.cu"
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4/b200/shapes/n3072_k8192.cu")
else()
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/bf16/shapes/n3072_k8192.cu"
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n3072_k8192.cu")
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n3072_k8192_tma.cu")
endif()
