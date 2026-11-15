target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bias.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/shapes/n3072_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n14336_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n14336_k5120/a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n14336_k5120/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n16384_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n16384_k5120/a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n16384_k5120/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n3072_k2048/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n3072_k2048/a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n3072_k2048/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/linear_split.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n14336_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n14336_k5120/a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n14336_k5120/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n16384_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n16384_k5120/a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n16384_k5120/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n17408_k3072/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n17408_k3072/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n4096_n12288_k5120/mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n4096_n12288_k5120/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n4096_n12288_k5120/small.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n7168_n7168_k5120/mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n7168_n7168_k5120/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n7168_n7168_k5120/small.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n12288_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n6144_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n6144_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n9216_k2048.cu"
)

target_sources(ninfer_ops_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n14336_k5120/a4_tma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n16384_k5120/a4_tma.cu"
)

# Full-attention qkvz (n17408_k3072): B200 quantizes into the tcgen05 scale layout and runs the
# tcgen05 GEMM at every token count; the other targets keep the mma.sync / TMA routes.
if(NINFER_TARGET STREQUAL "b200")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n17408_k3072/b200/tcgen05.cu")
else()
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n17408_k3072/a4.cu")
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n17408_k3072/a4_tma.cu")
endif()

# GDN qkvz (n20480_k3072) runs on tcgen05 on B200; the other targets keep the mma.sync ladder.
if(NOT NINFER_TARGET STREQUAL "b200")
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/bf16/shapes/n20480_k3072.cu")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/bf16/${NINFER_TARGET}/sources.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/q8/${NINFER_TARGET}/sources.cmake")

if(NINFER_TARGET STREQUAL "dgx_spark")
  include("${CMAKE_CURRENT_LIST_DIR}/mxfp8/dgx_spark/sources.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/fp8_block_bf16/dgx_spark/sources.cmake")
endif()
