target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/tail_extract.cu"
  "${CMAKE_CURRENT_LIST_DIR}/prepare/prepare.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/prepare/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/indexer/indexer.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/indexer/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/sparse_gqa.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/launch.cu"
)

# The sparse QSA kernels are an SM120 mma.sync family and are not part of the B200 build; the
# shared dispatch reports the indexer / sparse_gqa families as not available on b200.
if(NINFER_TARGET STREQUAL "b200")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/indexer/unsupported.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/unsupported.cpp")
else()
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/indexer/${NINFER_TARGET}/plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/${NINFER_TARGET}/plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/${NINFER_TARGET}/launch.cu")
  if(NINFER_TARGET STREQUAL "dgx_spark")
    target_sources(ninfer_ops_non_rdc PRIVATE
      "${CMAKE_CURRENT_LIST_DIR}/indexer/dgx_spark/launch.cu"
      "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/dgx_spark/grouped.cu")
  else()
    target_sources(ninfer_ops PRIVATE
      "${CMAKE_CURRENT_LIST_DIR}/indexer/${NINFER_TARGET}/launch.cu")
  endif()
endif()
