#include "ops/moe/routed_swiglu/cuda/dispatch.h"
#include "ops/moe/operands.h"
#include "core/platform.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::moe_routed {

const ShapePolicy& shape_dispatch(const RoutedSwiGLUProfile& profile) {
    validate_expert_profile(profile);
#if NINFER_TARGET_SPARK
    if (profile.ffn.hidden == 4096) {
        switch (profile.ffn.down_type) {
        case QType::NVFP4:
            return profile.ffn.intermediate == 1024 ? nvfp4_h4096_i1024_e288_k8()
                                                    : nvfp4_h4096_i2048_e288_k8();
        case QType::NVFP4_MUL:
            return profile.ffn.intermediate == 1024 ? nvfp4_mul_h4096_i1024_e288_k8()
                                                    : nvfp4_mul_h4096_i2048_e288_k8();
        case QType::FP8_E4M3FN_BLOCK128X128_BF16:
            return profile.ffn.intermediate == 1024 ? fp8_h4096_i1024_e288_k8()
                                                    : fp8_h4096_i2048_e288_k8();
        default:
            throw std::invalid_argument("unsupported clamped MoE expert format");
        }
    }
    if (profile.ffn.intermediate == 320)
        return profile.ffn.down_type == QType::NVFP4 ? nvfp4_h2560_i320_e512_k10()
                                                     : fp8_h2560_i320_e512_k10();
#endif
    switch (profile.ffn.down_type) {
    case QType::FP8_E4M3FN_BLOCK128X128_BF16:
#if NINFER_TARGET_B200
        throw std::invalid_argument("moe routed swiglu: fp8 is not available on b200,"
                                    " hidden=" +
                                    std::to_string(profile.ffn.hidden) + " intermediate=" +
                                    std::to_string(profile.ffn.intermediate));
#else
        return fp8_h2560_i640_e512_k10();
#endif
    case QType::NVFP4:
        if (profile.ffn.hidden == 3072) {
#if !NINFER_TARGET_SPARK
            return nvfp4_h3072_i1024_e256_k8();
#else
            throw std::invalid_argument("unsupported MoE expert geometry on this target");
#endif
        }
#if NINFER_TARGET_B200
        // The 2560/640/512/10 profile ran block-scaled mma.sync, an SM120 instruction.
        throw std::invalid_argument("moe routed swiglu: mma.sync is not available on b200,"
                                    " hidden=" +
                                    std::to_string(profile.ffn.hidden) + " intermediate=" +
                                    std::to_string(profile.ffn.intermediate));
#else
        return nvfp4_h2560_i640_e512_k10();
#endif
    case QType::Q5_G64_FP16:
        return q4_q5_h2048_i512_e256_k8();
    case QType::Q6_G64_FP16:
        return q4_q6_h2048_i512_e256_k8();
    case QType::Q8_G32_FP16:
        return q8_q8_h2048_i512_e256_k8();
    default:
        throw std::invalid_argument("unsupported MoE expert format");
    }
}

} // namespace ninfer::ops::detail::moe_routed
