# Native target-private MTP proposal/verification microbenchmark. It enters through the target
# Session and Executor and executes the production round schedule over a real .ninfer artifact.
add_executable(ninfer_qwen3_5_mtp_round_bench
  "${CMAKE_CURRENT_LIST_DIR}/mtp_round_bench.cpp")
ninfer_internal_includes(ninfer_qwen3_5_mtp_round_bench)
target_link_libraries(ninfer_qwen3_5_mtp_round_bench PRIVATE
  ninfer_engine ninfer_model_execution ninfer_core)

# Complete production DFlash steady round: confirmed-context append, next proposal, target
# verification/acceptance, and host publication over a real 35B artifact.
add_executable(ninfer_qwen3_5_dflash_round_bench
  "${CMAKE_CURRENT_LIST_DIR}/dflash_round_bench.cpp")
ninfer_internal_includes(ninfer_qwen3_5_dflash_round_bench)
target_link_libraries(ninfer_qwen3_5_dflash_round_bench PRIVATE
  ninfer_engine ninfer_model_execution ninfer_core)

# Decode step-time benchmark over a real artifact: one complete decode step per measured round
# for a fixed batch, in captured-graph replay and eager dispatch, with phase decomposition
# (attention/GDN/MoE) from the opt-in execution phase clock. Prints one JSON object per
# (batch, mode) pair.
add_executable(ninfer_qwen3_5_decode_step_bench
  "${CMAKE_CURRENT_LIST_DIR}/decode_step_bench.cpp")
ninfer_internal_includes(ninfer_qwen3_5_decode_step_bench)
target_link_libraries(ninfer_qwen3_5_decode_step_bench PRIVATE
  ninfer_engine ninfer_model_execution ninfer_model_loading ninfer_core ninfer::json)

if(NINFER_TARGET STREQUAL "rtx5090")
  add_executable(ninfer_qwen3_5_moe_execution_bench
    "${CMAKE_CURRENT_LIST_DIR}/moe_execution_bench.cpp")
  ninfer_internal_includes(ninfer_qwen3_5_moe_execution_bench)
  target_link_libraries(ninfer_qwen3_5_moe_execution_bench PRIVATE ninfer_model_execution)
endif()

# Packed Main prefill against the same windows executed one by one, through the production
# RankProgram over a real artifact. It reuses the native rank fixture of the model tests.
add_executable(ninfer_qwen3_5_packed_prefill_bench
  "${CMAKE_CURRENT_LIST_DIR}/packed_prefill_bench.cpp")
ninfer_internal_includes(ninfer_qwen3_5_packed_prefill_bench)
target_include_directories(ninfer_qwen3_5_packed_prefill_bench PRIVATE
  "${PROJECT_SOURCE_DIR}/tests")
target_link_libraries(ninfer_qwen3_5_packed_prefill_bench PRIVATE
  ninfer_engine ninfer_model_execution ninfer_model_loading ninfer_core)
