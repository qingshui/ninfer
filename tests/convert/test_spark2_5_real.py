"""Spark2.5 converter on the real checkpoints: sampled layers against raw safetensors."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from safetensors import safe_open
import torch

from tools.artifact.codecs.direct import decode_direct
from tools.artifact.codecs.fp8_row import decode_fp8_row_fp32_words
from tools.artifact.reader import Artifact
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.official_recipes import RECIPES
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource
from tools.convert.spark2_5 import build_model

ROOT = Path(os.environ.get("NINFER_SPARK_ROOT", "/home/users/humingqing/qwen"))
CHECKPOINTS = {
    "bf16": ROOT / "Spark-X2.6-1.7B",
    "fp8": ROOT / "Spark-X2.6-1.7B-FP8",
}
LAYERS = (0, 3)  # a sliding-attention and a full-attention layer
Q, KV, HEADS = 2048, 512, 8


def _source_tensor(root: Path, name: str) -> torch.Tensor:
    index = root / "model.safetensors.index.json"
    shard = (
        json.loads(index.read_text())["weight_map"][name]
        if index.is_file()
        else "model.safetensors"
    )
    with safe_open(root / shard, framework="pt") as handle:
        return handle.get_tensor(name)


@pytest.fixture(scope="module", params=sorted(CHECKPOINTS))
def sampled(request, tmp_path_factory):
    variant = request.param
    root = CHECKPOINTS[variant]
    if not (root / "config.json").is_file():
        pytest.skip(f"{variant} checkpoint is not available")
    path = tmp_path_factory.mktemp(variant) / "layers.ninfer"
    with SafetensorsSource(root) as store:
        model = build_model(store)
        recipe = Recipe(model)
        RECIPES[f"spark_x2_6_17b_{variant}"](model, recipe, {"base": store})
        prepared = recipe.prepare(device="cpu")
        prefixes = tuple(f"text/layers/{i}/" for i in LAYERS)
        jobs = [
            job
            for job in prepared.weights
            if all(name.startswith(prefixes) for name in job.parameters)
        ]
        with ArtifactWriter(
            path,
            [job.spec for job in jobs],
            components={"text": {"config": {}}},
            bindings={},
        ) as writer:
            for job in jobs:
                job.prepared.produce(TensorOutput(writer, job.spec.id))
        config = model.config
        shapes = {name: item.shape for name, item in model.parameters.items()}
        placement = {
            name: (job.spec.id, begin)
            for job in jobs
            for name, begin in zip(job.parameters, _offsets(shapes, job.parameters))
        }
    with Artifact(path) as artifact:
        yield variant, root, config, shapes, artifact, placement


def _offsets(shapes, names):
    cursor, result = 0, []
    for name in names:
        result.append(cursor)
        cursor += int(torch.Size(shapes[name]).numel())
    return result


def _read(artifact, placement, shapes, name):
    """Logical rows of one parameter: BF16 values, or (codes, scales) for row FP8."""
    object_id, begin = placement[name]
    obj = artifact.by_id[object_id]
    count = int(torch.Size(shapes[name]).numel())
    payload = artifact.read_object(object_id)
    if obj.format == "bf16":
        flat = decode_direct(payload, "bf16", obj.shape).reshape(-1)
        return flat[begin : begin + count].reshape(shapes[name])
    assert (obj.format, obj.layout) == ("fp8_e4m3fn_row_fp32", "row_scale_v1")
    codes, scales = decode_fp8_row_fp32_words(payload, obj.shape)
    k = obj.shape[1]
    rows = slice(begin // k, (begin + count) // k)
    return codes[rows], scales[rows]


def _source_rows(root, tensor, rows):
    weight = _source_tensor(root, tensor + ".weight")
    if weight.dtype == torch.bfloat16:
        return weight[rows]
    scale = _source_tensor(root, tensor + ".weight_scale")
    assert scale.dtype == torch.float32 and scale.shape[0] == weight.shape[0]
    return weight[rows].view(torch.uint8), scale.reshape(-1)[rows]


def _reference_row_fp8_fp32(bf16_rows):
    """Independent row quantization: scale = amax / 448, code = RNE_e4m3fn(x / scale).

    Both steps are single binary32 divisions (a tensor divisor, so no reciprocal
    multiplication). All-zero rows store scale 1 and zero codes.
    """
    rows = bf16_rows.float()
    amax = rows.abs().amax(dim=1)
    zero = amax == 0
    scale = amax / torch.full_like(amax, 448.0)
    scale = torch.where(zero, torch.ones_like(scale), scale)
    codes = (rows / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    codes = codes.view(torch.uint8)
    codes[zero] = 0
    return codes, scale


def test_config_matches_the_checkpoint(sampled):
    _, root, config, _, _, _ = sampled
    source = json.loads((root / "config.json").read_text())
    assert config["num_hidden_layers"] == source["num_hidden_layers"] == 28
    assert config["layer_types"] == source["layer_types"]
    assert config["rope"] == {
        "full_attention": {"rotary_dim": 64, "theta": 5e6},
        "sliding_attention": {"rotary_dim": 256, "theta": 1e4},
    }
    assert config["tie_word_embeddings"] is True


@pytest.mark.parametrize("layer", LAYERS)
def test_sampled_layer_matches_raw_source_words(sampled, layer):
    variant, root, _, shapes, artifact, placement = sampled
    p, sp = f"text/layers/{layer}/", f"model.layers.{layer}."
    qkv, mlp = sp + "self_attn.q_k_v_proj", sp + "mlp."
    projections = {
        "attention/query": (qkv, slice(0, Q)),
        "attention/key": (qkv, slice(Q, Q + KV)),
        "attention/value": (qkv, slice(Q + KV, Q + 2 * KV)),
        "attention/output": (sp + "self_attn.out_proj", slice(None)),
        "mlp/gate": (mlp + "gate_proj", slice(None)),
        "mlp/up": (mlp + "up_proj", slice(None)),
        "mlp/down": (mlp + "down_proj", slice(None)),
    }
    for role, (tensor, rows) in projections.items():
        actual = _read(artifact, placement, shapes, p + role)
        expected = _source_rows(root, tensor, rows)
        if variant == "bf16":
            # The BF16 checkpoint is quantized by the recipe: compare against an
            # independent host reference (FP32 division, E4M3FN RNE), not source words.
            expected = _reference_row_fp8_fp32(expected)
        assert torch.equal(actual[0], expected[0]), role
        # FP32 scales must match bit for bit.
        assert torch.equal(actual[1].view(torch.int32), expected[1].view(torch.int32)), role
    direct = {
        "attention/gate": sp + "self_attn.g_proj.weight",
        "input_norm": sp + "input_layernorm.weight",
        "post_attention_norm": sp + "post_attention_layernorm.weight",
    }
    for role, tensor in direct.items():
        actual = _read(artifact, placement, shapes, p + role)
        assert torch.equal(actual.reshape(-1), _source_tensor(root, tensor).reshape(-1))
        assert actual.shape == shapes[p + role]
    assert shapes[p + "attention/gate"] == (HEADS, 2048)
