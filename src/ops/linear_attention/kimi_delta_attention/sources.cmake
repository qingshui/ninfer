target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/input_conv/input_conv.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/chunked/prepare.cu"
  "${CMAKE_CURRENT_LIST_DIR}/chunked/recurrence.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kimi_delta_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/replay/fold.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/replay/fold.cu"
  "${CMAKE_CURRENT_LIST_DIR}/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/recurrent.cu"
)

target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/${NINFER_TARGET}/select.cpp")

if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/input_conv.cu")
  target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/mxfp8_input_conv.cu")
else()
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/input_conv/launch.cu")
  # The MXFP8 A8 kernels are an SM120 block-scaled mma.sync family and are not part of the B200
  # build; the input-conv dispatch reports the mxfp8 family as not available on b200.
  if(NOT NINFER_TARGET STREQUAL "b200")
    target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/input_conv/mxfp8.cu")
  endif()
endif()
