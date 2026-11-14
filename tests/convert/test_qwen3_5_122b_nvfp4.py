"""T05 / F06 acceptance: the qwen3_5_122b_nvfp4 official recipe on a synthetic checkpoint.

The checkpoint mirrors the ModelOpt NVFP4 layout of the Qwen3.5-MoE source: per-expert
U8 tensors with weight_scale/weight_scale_2/input_scale, nested text_config and the
flat ``model.`` tensor prefix.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest
from safetensors.torch import save_file
import torch

from tools.convert.official_recipes import RECIPES, qwen3_5_122b_nvfp4
from tools.convert.qwen3_5 import build_model
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource

H, HEADS, KV, D = 128, 2, 1, 64
Q, KVW = HEADS * D, KV * D
I, SHARED, EXPERTS, VOCAB = 128, 128, 4, 16
LAYERS = ["linear_attention", "full_attention"]
EXPERT_INPUT_SCALES = ((4.0, 8.0), (16.0, 32.0))  # (gate/up, down) per layer


def _config() -> dict:
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "hidden_size": H,
            "vocab_size": VOCAB,
            "num_hidden_layers": len(LAYERS),
            "max_position_embeddings": 128,
            "layer_types": list(LAYERS),
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV,
            "head_dim": D,
            "rope_parameters": {
                "partial_rotary_factor": 0.5,
                "mrope_section": [8, 4, 4],
            },
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 4,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 3,
            "num_experts": EXPERTS,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": I,
            "shared_expert_intermediate_size": SHARED,
        },
    }


def _nvfp4(tensors, prefix, n, k, *, weight_scale_2=1.0, input_scale=2.0):
    tensors[prefix + ".weight"] = torch.full((n, k // 2), 0x21, dtype=torch.uint8)
    tensors[prefix + ".weight_scale"] = torch.ones(n, k // 16).to(torch.float8_e4m3fn)
    tensors[prefix + ".weight_scale_2"] = torch.tensor(
        weight_scale_2, dtype=torch.float32
    )
    tensors[prefix + ".input_scale"] = torch.tensor(input_scale, dtype=torch.float32)


def _tensors(expert_overrides=None) -> dict:
    tensors = {
        "model.embed_tokens.weight": torch.zeros(VOCAB, H, dtype=torch.bfloat16),
        "lm_head.weight": torch.ones(VOCAB, H, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(H, dtype=torch.bfloat16),
    }
    for layer in range(len(LAYERS)):
        s = f"model.layers.{layer}."
        tensors[s + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        tensors[s + "post_attention_layernorm.weight"] = torch.ones(
            H, dtype=torch.bfloat16
        )
        if LAYERS[layer] == "linear_attention":
            a = s + "linear_attn."
            tensors[a + "A_log"] = torch.zeros(2, dtype=torch.float32)
            tensors[a + "dt_bias"] = torch.ones(2, dtype=torch.float32)
            tensors[a + "conv1d.weight"] = torch.ones(16, 1, 3)
            tensors[a + "in_proj_a.weight"] = torch.ones(2, H, dtype=torch.bfloat16)
            tensors[a + "in_proj_b.weight"] = torch.ones(2, H, dtype=torch.bfloat16)
            tensors[a + "in_proj_qkv.weight"] = torch.ones(16, H, dtype=torch.bfloat16)
            tensors[a + "in_proj_z.weight"] = torch.ones(8, H, dtype=torch.bfloat16)
            tensors[a + "norm.weight"] = torch.ones(4, dtype=torch.bfloat16)
            tensors[a + "out_proj.weight"] = torch.ones(H, 8, dtype=torch.bfloat16)
        else:
            a = s + "self_attn."
            for role, n in (("q", 2 * Q), ("k", KVW), ("v", KVW), ("o", H)):
                _nvfp4(tensors, a + role + "_proj", n, H if role != "o" else Q)
            tensors[a + "q_norm.weight"] = torch.ones(D, dtype=torch.bfloat16)
            tensors[a + "k_norm.weight"] = torch.ones(D, dtype=torch.bfloat16)
        tensors[s + "mlp.gate.weight"] = torch.ones(EXPERTS, H, dtype=torch.bfloat16)
        tensors[s + "mlp.shared_expert_gate.weight"] = torch.ones(
            1, H, dtype=torch.bfloat16
        )
        for role in ("gate", "up"):
            _nvfp4(tensors, s + f"mlp.shared_expert.{role}_proj", SHARED, H)
        _nvfp4(tensors, s + "mlp.shared_expert.down_proj", H, SHARED)
        gate_up_scale, down_scale = EXPERT_INPUT_SCALES[layer]
        for expert in range(EXPERTS):
            override = (expert_overrides or {}).get((layer, expert), {})
            for role, n, k in (
                ("gate", I, H),
                ("up", I, H),
                ("down", H, I),
            ):
                _nvfp4(
                    tensors,
                    s + f"mlp.experts.{expert}.{role}_proj",
                    n,
                    k,
                    weight_scale_2=1.0 + 0.25 * expert,
                    input_scale=override.get(
                        "input_scale",
                        down_scale if role == "down" else gate_up_scale,
                    ),
                )
    return tensors


def _checkpoint(path: Path, tensors: dict) -> SafetensorsSource:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(_config()))
    save_file(tensors, path / "model.safetensors")
    for role, value in {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
        "preprocessor_config.json": {"patch_size": 16, "temporal_patch_size": 2},
        "video_preprocessor_config.json": {"patch_size": 16, "temporal_patch_size": 2},
    }.items():
        (path / role).write_text(json.dumps(value))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    return SafetensorsSource(path)


def _prepare(tmp_path, tensors=None):
    with _checkpoint(tmp_path / "source", tensors or _tensors()) as source:
        model = build_model(source)
        recipe = Recipe(model)
        qwen3_5_122b_nvfp4(model, recipe, {"base": source})
        return model, recipe, recipe.prepare(device="cpu")


def _job(prepared, name):
    return next(job for job in prepared.weights if name in job.parameters)


def test_recipe_is_registered_by_name():
    assert RECIPES["qwen3_5_122b_nvfp4"] is qwen3_5_122b_nvfp4


def test_expert_banks_are_expert_major_with_batched_nvfp4_layout(tmp_path):
    model, _, prepared = _prepare(tmp_path)
    assert set(model.components) == {"text"}
    assert all(name.startswith("text/") for name in prepared.bindings)
    for layer in range(len(LAYERS)):
        p = f"text/layers/{layer}/moe/experts/"
        gate_up = _job(prepared, p + "0/gate")
        down = _job(prepared, p + "0/down")
        for job, shape in ((gate_up, (EXPERTS, 2 * I, H)), (down, (EXPERTS, H, I))):
            assert job.spec.format == "nvfp4" and job.method_name == "import_encoded"
            assert job.spec.shape == shape
            assert job.spec.layout == "batched_block_scale_k16_m128x4_v1"
        # Gate/up interleave per expert: [e0 gate, e0 up, e1 gate, e1 up, ...].
        assert gate_up.parameters == tuple(
            p + f"{expert}/{role}"
            for expert in range(EXPERTS)
            for role in ("gate", "up")
        )
        assert down.parameters == tuple(
            p + f"{expert}/down" for expert in range(EXPERTS)
        )
        width = I * H
        for expert in range(EXPERTS):
            for role, begin in (("gate", 2 * expert), ("up", 2 * expert + 1)):
                parts = prepared.bindings[p + f"{expert}/{role}"]["parts"]
                assert parts == [
                    {
                        "object": gate_up.spec.id,
                        "range": [begin * width, (begin + 1) * width],
                    }
                ]
            assert prepared.bindings[p + f"{expert}/down"]["parts"] == [
                {
                    "object": down.spec.id,
                    "range": [expert * H * I, (expert + 1) * H * I],
                }
            ]


def test_expert_uses_share_one_divisor_tensor_per_layer(tmp_path):
    _, _, prepared = _prepare(tmp_path)
    shared = [
        (spec, data)
        for spec, data in prepared.auxiliaries
        if spec.id == "auxiliary/shared/expert_activation_divisors"
    ]
    assert len(shared) == 1
    spec, data = shared[0]
    assert spec.format == "fp32" and spec.shape == (len(LAYERS), 2)
    assert data == b"".join(
        struct.pack("<ff", *(1.0 / value for value in scales))
        for scales in EXPERT_INPUT_SCALES
    )
    uses = [use for use in prepared.uses if "/moe/experts/" in use["parameter"]]
    assert len(uses) == len(LAYERS) * EXPERTS * 3
    for use in uses:
        layer = int(use["parameter"].split("/")[2])
        is_down = use["parameter"].endswith("/down")
        offset = 2 * layer + int(is_down)
        assert use["activation_policy"] == "AllowA4"
        assert use["auxiliaries"]["activation_input_divisor"] == {
            "parts": [
                {
                    "object": "auxiliary/shared/expert_activation_divisors",
                    "range": [offset, offset + 1],
                }
            ]
        }
    # Every remaining auxiliary is a scalar of a non-expert Use, never an expert's.
    others = [
        spec for spec, _ in prepared.auxiliaries
        if spec.id != "auxiliary/shared/expert_activation_divisors"
    ]
    non_expert_uses = {
        use["parameter"]
        for use in prepared.uses
        if use.get("auxiliaries") and "/moe/experts/" not in use["parameter"]
    }
    assert len(others) == len(non_expert_uses)
    assert all(spec.shape == () for spec in others)


def test_divergent_expert_input_scales_within_a_layer_are_rejected(tmp_path):
    tensors = _tensors({(1, 2): {"input_scale": 99.0}})
    with pytest.raises(ValueError, match="input_scale differs within the layer"):
        _prepare(tmp_path, tensors)


def test_attention_qkv_scale_mismatch_is_rejected_and_o_proj_is_independent(tmp_path):
    tensors = _tensors()
    tensors["model.layers.1.self_attn.q_proj.weight_scale_2"] = torch.tensor(
        2.0, dtype=torch.float32
    )
    with pytest.raises(
        ValueError, match="attention q/k/v weight_scale_2 or input_scale differ"
    ):
        _prepare(tmp_path, tensors)
    # o_proj is handled independently: its published scales may differ from q/k/v.
    tensors = _tensors()
    tensors["model.layers.1.self_attn.o_proj.weight_scale_2"] = torch.tensor(
        2.0, dtype=torch.float32
    )
    _, _, prepared = _prepare(tmp_path / "independent", tensors)
    assert _job(prepared, "text/layers/1/attention/output").spec.format == "nvfp4"


def test_unquantized_roles_stay_cast_direct_and_vocabulary_uses_q8(tmp_path):
    _, _, prepared = _prepare(tmp_path)
    for name in (
        "text/layers/0/gdn/query",
        "text/layers/0/gdn/a_projection",
        "text/layers/0/gdn/output",
        "text/layers/0/gdn/norm",
        "text/layers/0/gdn/convolution",
        "text/layers/0/moe/router",
        "text/layers/0/moe/shared_score",
        "text/layers/0/input_norm",
        "text/layers/0/post_attention_norm",
        "text/final_norm",
    ):
        assert _job(prepared, name).method_name == "cast_direct", name
    for name in ("text/token_embedding", "text/output_head"):
        assert _job(prepared, name).spec.format == "q8_g32_fp16", name
