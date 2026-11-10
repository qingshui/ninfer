"""T2 / F2: Spark2_5 converter adaptation and official recipes (spec-level checks)."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from tools.convert.official_recipes import RECIPES
from tools.convert.spark2_5 import layer_rope, text_config

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("NINFER_SPARK_ROOT", "/home/users/humingqing/qwen"))
CHECKPOINTS = {
    "spark_x2_6_17b_bf16": ROOT / "Spark-X2.6-1.7B",
    "spark_x2_6_17b_fp8": ROOT / "Spark-X2.6-1.7B-FP8",
}
GIB = 1 << 30


def hf_config(fp8_shape: bool = False, rope_as_float: bool = False) -> dict:
    full, sliding = (5e6, 1e4) if rope_as_float else (5000000, 10000)
    rope = {
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": full},
        "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": sliding},
    }
    if fp8_shape:
        rope["rope_theta"] = 10000.0
        rope["rope_type"] = "default"
    return {
        "architectures": ["Spark2_5ForCausalLM"],
        "model_type": "spark2_5",
        "attention_bias": False,
        "mlp_bias": False,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "gate_attn_act_mode": "sigmoid",
        "headwise_attn_output_gate": True,
        "hidden_act": "gelu",
        "hidden_size": 2048,
        "intermediate_size": 6656,
        "head_dim": 256,
        "layer_types": (["sliding_attention"] * 3 + ["full_attention"]) * 7,
        "max_position_embeddings": 1048576,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "num_hidden_layers": 28,
        "rms_norm_eps": 1e-6,
        "rope_parameters": rope,
        "sliding_window": 512,
        "tie_word_embeddings": True,
        "vocab_size": 131072,
    }


# ---- acceptance 1: rope shapes, layer pattern, sliding_window, invalid configs ----


@pytest.mark.parametrize("fp8_shape", [False, True])
@pytest.mark.parametrize("rope_as_float", [False, True])
def test_rope_json_shapes_resolve_to_same_per_layer_values(fp8_shape, rope_as_float):
    source = hf_config(fp8_shape, rope_as_float)
    assert layer_rope(source, "full_attention") == (64, 5e6)
    assert layer_rope(source, "sliding_attention") == (256, 1e4)
    config = text_config(source)
    for i, kind in enumerate(config["layer_types"]):
        want = (64, 5e6) if kind == "full_attention" else (256, 1e4)
        got = config["rope"][kind]
        assert (got["rotary_dim"], float(got["theta"])) == want, i
    assert text_config(source) == text_config(hf_config())


def test_layer_pattern_and_sliding_window_pass_through():
    config = text_config(hf_config())
    types = config["layer_types"]
    assert len(types) == config["num_hidden_layers"] == 28
    assert types.count("sliding_attention") == 21
    assert types.count("full_attention") == 7
    assert [i for i, t in enumerate(types) if t == "full_attention"] == list(range(3, 28, 4))
    assert config["sliding_window"] == 512
    assert config["architectures"] == ["Spark2_5ForCausalLM"]
    assert config["model_type"] == "spark2_5_text"
    assert config["hidden_act"] == "gelu"
    assert config["attention_gate"] == "headwise_sigmoid"
    assert config["tie_word_embeddings"] is True
    assert config["rms_norm_eps"] == pytest.approx(1e-6, rel=1e-5)


@pytest.mark.parametrize(
    "change",
    [
        {"layer_types": ["full_attention"] * 27},
        {"layer_types": ["full_attention"] * 29},
        {"layer_types": ["sliding_attention"] * 27 + ["linear_attention"]},
        {"gate_attn_act_mode": "softmax"},
        {"hidden_act": "silu"},
        {"attention_bias": True},
        {"mlp_bias": True},
    ],
)
def test_invalid_configs_raise_value_error(change):
    source = deepcopy(hf_config())
    source.update(change)
    with pytest.raises(ValueError) as info:
        text_config(source)
    if "layer_types" not in change or "linear_attention" in change["layer_types"]:
        assert "spark2_5:" in str(info.value)


# ---- acceptance 2: recipes registered ----


def test_official_recipes_registered():
    assert "spark_x2_6_17b_bf16" in RECIPES
    assert "spark_x2_6_17b_fp8" in RECIPES
