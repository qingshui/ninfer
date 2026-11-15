target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n256_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n3072_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n8_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2048_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n13312_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n131072_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2048_k6656.cu"
)

# GDN out (n3072_k8192) runs on tcgen05 on B200; the other targets keep the mma.sync ladder.
if(NOT NINFER_TARGET STREQUAL "b200")
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n3072_k8192.cu")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/${NINFER_TARGET}/sources.cmake")
