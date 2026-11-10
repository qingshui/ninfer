"""T2 / F2 acceptance: Spark2_5 converter on tiny synthetic checkpoints.

Complements the spec-level checks by driving the documented CLI contract
(``python3 -m tools.convert --model --recipe --out``) end to end on small
BF16 and FP8 checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.reader import Artifact
from tools.convert.official_recipes import RECIPES

REPO = Path(__file__).resolve().parents[2]
H, HEADS, KV, D, FF, V = 16, 2, 1, 8, 24, 8
Q, KVW = HEADS * D, KV * D
LAYERS = ["sliding_attention", "full_attention"]
RECIPE = {"bf16": "spark_x2_6_17b_bf16", "fp8": "spark_x2_6_17b_fp8"}


def make_config(fp8: bool) -> dict:
    rope = {
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
        "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
    }
    if fp8:
        rope.update(rope_theta=10000.0, rope_type="default")
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
        "hidden_size": H,
        "intermediate_size": FF,
        "head_dim": D,
        "layer_types": list(LAYERS),
        "max_position_embeddings": 4096,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-6,
        "rope_parameters": rope,
        "sliding_window": 4,
        "tie_word_embeddings": True,
        "vocab_size": V,
    }


def make_tensors(fp8: bool) -> dict:
    gen = torch.Generator().manual_seed(11)

    def bf16(*shape):
        return torch.randn(*shape, generator=gen).to(torch.bfloat16)

    def matrix(name, n, k):
        if not fp8:
            return {name + ".weight": bf16(n, k)}
        codes = torch.randint(0, 0x7E, (n, k), generator=gen).to(torch.uint8)
        return {
            name + ".weight": codes.view(torch.float8_e4m3fn),
            name + ".weight_scale": torch.rand(n, 1, generator=gen) + 0.25,
        }

    tensors = {"model.embedding.weight": bf16(V, H), "model.norm.weight": bf16(H)}
    for i in range(2):
        p = f"model.layers.{i}."
        tensors[p + "input_layernorm.weight"] = bf16(H)
        tensors[p + "post_attention_layernorm.weight"] = bf16(H)
        tensors[p + "self_attn.g_proj.weight"] = bf16(HEADS, H)
        tensors.update(matrix(p + "self_attn.q_k_v_proj", Q + 2 * KVW, H))
        tensors.update(matrix(p + "self_attn.out_proj", H, Q))
        for role in ("gate", "up"):
            tensors.update(matrix(p + f"mlp.{role}_proj", FF, H))
        tensors.update(matrix(p + "mlp.down_proj", H, FF))
    return tensors


def make_checkpoint(path: Path, fp8: bool) -> Path:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(make_config(fp8)))
    save_file(make_tensors(fp8), path / "model.safetensors")
    resources = {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
    }
    for name, value in resources.items():
        (path / name).write_text(json.dumps(value))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    return path


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("f2_acceptance")
    return {
        "bf16": make_checkpoint(root / "bf16", False),
        "fp8": make_checkpoint(root / "fp8", True),
    }


def snapshot(directory: Path) -> dict:
    return {
        p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(directory.iterdir())
    }


def run_cli(model: Path, recipe: str, out: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, "-m", "tools.convert", "--model", str(model),
         "--recipe", recipe, "--out", str(out), "--device", "cpu"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=240,
    )


@pytest.fixture(scope="module")
def converted(checkpoints, tmp_path_factory):
    """Per variant: source snapshot before/after, two outputs, CLI results."""
    root = tmp_path_factory.mktemp("f2_out")
    result = {}
    for variant, model in checkpoints.items():
        before = snapshot(model)
        runs = []
        for tag in ("a", "b"):
            out = root / f"{variant}-{tag}.ninfer"
            runs.append((run_cli(model, RECIPE[variant], out), out))
        result[variant] = {"before": before, "after": snapshot(model), "runs": runs}
    return result


def test_recipes_registered():
    assert "spark_x2_6_17b_bf16" in RECIPES
    assert "spark_x2_6_17b_fp8" in RECIPES


@pytest.mark.parametrize("variant", ["bf16", "fp8"])
def test_cli_exits_zero_and_writes_single_artifact(converted, variant):
    for proc, out in converted[variant]["runs"]:
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert out.is_file() and out.stat().st_size > 0
        assert out.stat().st_size <= 4 * (1 << 30)


@pytest.mark.parametrize("variant", ["bf16", "fp8"])
def test_same_tree_same_checkpoint_gives_identical_sha256(converted, variant):
    (_, first), (_, second) = converted[variant]["runs"]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(first) == digest(second)


@pytest.mark.parametrize("variant", ["bf16", "fp8"])
def test_source_directory_unchanged(converted, variant):
    assert converted[variant]["after"] == converted[variant]["before"]


@pytest.mark.parametrize("variant", ["bf16", "fp8"])
def test_artifact_config_and_tied_head(converted, variant):
    _, out = converted[variant]["runs"][0]
    with Artifact(out) as artifact:
        config = artifact.directory.components["text"]["config"]
        assert config["architectures"] == ["Spark2_5ForCausalLM"]
        assert config["model_type"] == "spark2_5_text"
        assert config["layer_types"] == LAYERS
        assert config["sliding_window"] == 4
        assert config["rope"]["full_attention"]["rotary_dim"] == D // 4
        assert config["rope"]["sliding_attention"]["rotary_dim"] == D
        bindings = artifact.directory.bindings
        assert bindings["text/output_head"] == bindings["text/token_embedding"]
        # The embedding shape appears once among stored tensors: no duplicate head.
        stored = [o for o in artifact.objects if hasattr(o, "shape") and tuple(o.shape) == (V, H)]
        assert len(stored) == 1
