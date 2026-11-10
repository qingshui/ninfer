"""Spark2.5 converter: config validation, tensor mapping, recipes and artifacts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.codecs.fp8_row import decode_fp8_row_fp32_words
from tools.artifact.reader import Artifact
from tools.convert.__main__ import main as convert_main
from tools.convert.official_recipes import RECIPES
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource
from tools.convert.spark2_5 import build_model, layer_rope, text_config

H, HEADS, KV, D, FF, V = 16, 2, 1, 8, 24, 8
Q, KVW = HEADS * D, KV * D
LAYERS = ["sliding_attention", "full_attention"]


def real_config() -> dict:
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
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
            "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
        },
        "sliding_window": 512,
        "tie_word_embeddings": True,
        "vocab_size": 131072,
    }


def fp8_config() -> dict:
    config = real_config()
    config["rope_parameters"].update(rope_theta=10000.0, rope_type="default")
    return config


def test_layer_rope_reads_only_the_per_type_tables():
    for config in (real_config(), fp8_config()):
        assert layer_rope(config, "full_attention") == (64, 5e6)
        assert layer_rope(config, "sliding_attention") == (256, 1e4)
    config = real_config()
    config["rope_theta"] = 123.0
    config["rope_type"] = "yarn"
    config["rope_parameters"]["rope_theta"] = "ignored"
    config["rope_parameters"]["rope_type"] = "ignored"
    config["rope_parameters"]["full_attention"]["rope_theta"] = 5000000.0
    assert layer_rope(config, "full_attention") == (64, 5e6)
    assert isinstance(layer_rope(config, "full_attention")[1], float)


@pytest.mark.parametrize("kind", ["full_attention", "sliding_attention"])
def test_layer_rope_requires_the_referenced_table(kind):
    config = real_config()
    del config["rope_parameters"][kind]
    with pytest.raises(ValueError, match="spark2_5:.*" + kind):
        layer_rope(config, kind)
    with pytest.raises(ValueError, match="spark2_5:"):
        layer_rope(real_config(), "linear_attention")


def test_text_config_matches_both_checkpoint_shapes():
    first, second = text_config(real_config()), text_config(fp8_config())
    assert first == second
    assert first["architectures"] == ["Spark2_5ForCausalLM"]
    assert first["model_type"] == "spark2_5_text"
    assert first["hidden_act"] == "gelu"
    assert first["attention_gate"] == "headwise_sigmoid"
    assert first["sliding_window"] == 512
    assert first["tie_word_embeddings"] is True
    assert first["rope"] == {
        "full_attention": {"rotary_dim": 64, "theta": 5e6},
        "sliding_attention": {"rotary_dim": 256, "theta": 1e4},
    }
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        assert first[key] == real_config()[key]
    assert first["layer_types"] == real_config()["layer_types"]


def test_only_present_layer_types_need_rope_tables():
    config = real_config()
    config["layer_types"] = ["sliding_attention"] * 28
    del config["rope_parameters"]["full_attention"]
    assert set(text_config(config)["rope"]) == {"sliding_attention"}
    config["layer_types"][0] = "full_attention"
    with pytest.raises(ValueError, match="spark2_5:"):
        text_config(config)


@pytest.mark.parametrize(
    "change",
    [
        {"layer_types": ["full_attention"] * 27},
        {"layer_types": ["full_attention"] * 29},
        {"layer_types": ["sliding_attention"] * 27 + ["linear_attention"]},
        {"gate_attn_act_mode": "softmax"},
        {"headwise_attn_output_gate": False},
        {"hidden_act": "silu"},
        {"attention_bias": True},
        {"mlp_bias": True},
        {"architectures": ["Qwen3_5ForCausalLM"]},
        {"num_key_value_heads": 3},
        {"sliding_window": 0},
    ],
)
def test_unsupported_configs_raise_prefixed_errors(change):
    config = deepcopy(real_config())
    config.update(change)
    with pytest.raises(ValueError, match="^spark2_5: "):
        text_config(config)


def tiny_config() -> dict:
    config = real_config()
    config.update(
        hidden_size=H,
        intermediate_size=FF,
        head_dim=D,
        layer_types=list(LAYERS),
        max_position_embeddings=4096,
        num_attention_heads=HEADS,
        num_key_value_heads=KV,
        num_hidden_layers=2,
        sliding_window=4,
        vocab_size=V,
    )
    return config


def tiny_tensors(fp8: bool) -> dict:
    gen = torch.Generator().manual_seed(7)

    def bf16(*shape):
        return torch.randn(*shape, generator=gen).to(torch.bfloat16)

    def matrix(name, n, k):
        if not fp8:
            return {name + ".weight": bf16(n, k)}
        codes = torch.randint(0, 0x7E, (n, k), generator=gen).to(torch.uint8)
        signs = torch.randint(0, 2, (n, k), generator=gen).to(torch.uint8) << 7
        return {
            name + ".weight": (codes | signs).view(torch.float8_e4m3fn),
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
    config = tiny_config()
    if fp8:
        config["rope_parameters"].update(rope_theta=10000.0, rope_type="default")
    (path / "config.json").write_text(json.dumps(config))
    save_file(tiny_tensors(fp8), path / "model.safetensors")
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
    root = tmp_path_factory.mktemp("spark")
    return {
        "bf16": make_checkpoint(root / "bf16", False),
        "fp8": make_checkpoint(root / "fp8", True),
    }


def convert(model: Path, recipe: str, out: Path) -> None:
    convert_main(
        ["--model", str(model), "--recipe", recipe, "--out", str(out), "--device", "cpu"]
    )


def snapshot(directory: Path) -> dict:
    return {
        p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(directory.iterdir())
    }


def test_build_model_maps_names_shapes_and_packing_groups(checkpoints):
    with SafetensorsSource(checkpoints["bf16"]) as store:
        model = build_model(store)
        assert model.config == text_config(tiny_config())
        params = model.parameters
        expected = {
            "text/token_embedding": (V, H),
            "text/output_head": (V, H),
            "text/final_norm": (H,),
        }
        for i in range(2):
            p = f"text/layers/{i}/"
            expected.update(
                {
                    p + "input_norm": (H,),
                    p + "post_attention_norm": (H,),
                    p + "attention/query": (Q, H),
                    p + "attention/key": (KVW, H),
                    p + "attention/value": (KVW, H),
                    p + "attention/gate": (HEADS, H),
                    p + "attention/output": (H, Q),
                    p + "mlp/gate": (FF, H),
                    p + "mlp/up": (FF, H),
                    p + "mlp/down": (H, FF),
                }
            )
        assert {k: v.shape for k, v in params.items()} == expected
        assert set(model.resources) == {
            f"resource/text/{n}"
            for n in (
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "generation_config.json",
            )
        }
        assert ("text/layers/0/mlp/gate", "text/layers/0/mlp/up") in {
            tuple(g) for g in model.packing_groups
        }
        source = tiny_tensors(False)
        qkv = source["model.layers.1.self_attn.q_k_v_proj.weight"]
        base = "text/layers/1/attention/"
        for role, rows in (
            ("query", qkv[:Q]),
            ("key", qkv[Q : Q + KVW]),
            ("value", qkv[Q + KVW :]),
        ):
            assert torch.equal(params[base + role].source.rows(0, rows.shape[0]), rows)
        head = params["text/output_head"].source.rows(0, V)
        assert torch.equal(head, source["model.embedding.weight"])


@pytest.mark.parametrize(
    "kwargs",
    [{"components": ("text", "vision")}, {"companions": {"dflash": object()}}],
)
def test_build_model_rejects_unsupported_components(checkpoints, kwargs):
    with SafetensorsSource(checkpoints["bf16"]) as store:
        with pytest.raises(ValueError, match="^spark2_5: "):
            build_model(store, **kwargs)


def locate(artifact, name):
    binding = artifact.directory.bindings[name]
    if "object" in binding:
        obj = artifact.by_id[binding["object"]]
        return obj, 0, int(torch.Size(obj.shape).numel())
    (part,) = binding["parts"]
    return artifact.by_id[part["object"]], *part["range"]


def read_bf16(artifact, name):
    obj, first, last = locate(artifact, name)
    assert obj.format == "bf16"
    raw = torch.frombuffer(bytearray(artifact.read_object(obj.id)), dtype=torch.bfloat16)
    return raw[first:last].reshape(-1)


def read_rows(artifact, name):
    obj, first, last = locate(artifact, name)
    assert obj.format == "fp8_e4m3fn_row_fp32" and obj.layout == "row_scale_v1"
    codes, scales = decode_fp8_row_fp32_words(artifact.read_object(obj.id), obj.shape)
    k = obj.shape[1]
    return codes[first // k : last // k], scales[first // k : last // k]


def logical_slices(source: dict, i: int) -> dict:
    p = f"model.layers.{i}."
    qkv = p + "self_attn.q_k_v_proj"
    rows = {
        "attention/query": (qkv, 0, Q),
        "attention/key": (qkv, Q, Q + KVW),
        "attention/value": (qkv, Q + KVW, Q + 2 * KVW),
        "attention/output": (p + "self_attn.out_proj", 0, H),
        "mlp/gate": (p + "mlp.gate_proj", 0, FF),
        "mlp/up": (p + "mlp.up_proj", 0, FF),
        "mlp/down": (p + "mlp.down_proj", 0, H),
    }
    return {f"text/layers/{i}/{role}": value for role, value in rows.items()}


def test_bf16_recipe_keeps_every_tensor_bf16_and_shares_the_head(checkpoints, tmp_path):
    source_dir = checkpoints["bf16"]
    before = snapshot(source_dir)
    out = tmp_path / "bf16.ninfer"
    convert(source_dir, "spark_x2_6_17b_bf16", out)
    assert snapshot(source_dir) == before
    source = tiny_tensors(False)
    with Artifact(out) as artifact:
        tensors = [o for o in artifact.objects if hasattr(o, "format")]
        assert {o.format for o in tensors} == {"bf16"}
        bindings = artifact.directory.bindings
        assert bindings["text/output_head"] == bindings["text/token_embedding"]
        embedding = [o for o in tensors if o.shape == (V, H)]
        assert len(embedding) == 1
        assert {u["activation_policy"] for u in artifact.directory.uses} == {"A16Only"}
        config = artifact.directory.components["text"]["config"]
        assert config == text_config(tiny_config())
        assert torch.equal(
            read_bf16(artifact, "text/token_embedding"),
            source["model.embedding.weight"].reshape(-1),
        )
        for i in range(2):
            for name, (tensor, begin, end) in logical_slices(source, i).items():
                expected = source[tensor + ".weight"][begin:end].reshape(-1)
                assert torch.equal(read_bf16(artifact, name), expected), name
            gate = source[f"model.layers.{i}.self_attn.g_proj.weight"].reshape(-1)
            assert torch.equal(read_bf16(artifact, f"text/layers/{i}/attention/gate"), gate)


def test_fp8_recipe_imports_row_words_exactly_and_keeps_gate_and_head_bf16(
    checkpoints, tmp_path
):
    source_dir = checkpoints["fp8"]
    before = snapshot(source_dir)
    out = tmp_path / "fp8.ninfer"
    convert(source_dir, "spark_x2_6_17b_fp8", out)
    assert snapshot(source_dir) == before
    source = tiny_tensors(True)
    with Artifact(out) as artifact:
        formats = [o.format for o in artifact.objects if hasattr(o, "format")]
        assert formats.count("fp8_e4m3fn_row_fp32") == 8  # qkv, out, gate_up, down
        assert set(formats) == {"bf16", "fp8_e4m3fn_row_fp32"}
        bindings = artifact.directory.bindings
        assert bindings["text/output_head"] == bindings["text/token_embedding"]
        policies = {
            (u["parameter"], u["activation_policy"]) for u in artifact.directory.uses
        }
        for i in range(2):
            prefix = f"text/layers/{i}/"
            for role in ("attention/query", "attention/output", "mlp/up", "mlp/down"):
                assert (prefix + role, "AllowA8") in policies
            assert (prefix + "attention/gate", "A16Only") in policies
        assert ("text/output_head", "A16Only") in policies
        for i in range(2):
            for name, (tensor, begin, end) in logical_slices(source, i).items():
                codes, scales = read_rows(artifact, name)
                expected_codes = source[tensor + ".weight"].view(torch.uint8)[begin:end]
                expected_scales = source[tensor + ".weight_scale"].reshape(-1)[begin:end]
                assert torch.equal(codes, expected_codes), name
                assert torch.equal(
                    scales.view(torch.int32), expected_scales.view(torch.int32)
                ), name
            gate = source[f"model.layers.{i}.self_attn.g_proj.weight"].reshape(-1)
            assert torch.equal(read_bf16(artifact, f"text/layers/{i}/attention/gate"), gate)


def test_conversion_is_byte_reproducible(checkpoints, tmp_path):
    for variant in ("bf16", "fp8"):
        digests = []
        for name in ("first", "second"):
            out = tmp_path / f"{variant}-{name}.ninfer"
            convert(checkpoints[variant], f"spark_x2_6_17b_{variant}", out)
            digests.append(hashlib.sha256(out.read_bytes()).hexdigest())
        assert digests[0] == digests[1]


def test_fp8_recipe_rejects_a_bf16_checkpoint_naming_the_tensor(checkpoints, tmp_path):
    with pytest.raises(ValueError, match=r"q_k_v_proj\.weight"):
        convert(checkpoints["bf16"], "spark_x2_6_17b_fp8", tmp_path / "bad.ninfer")
    assert not (tmp_path / "bad.ninfer").exists()


def test_recipes_are_registered_and_require_spark_mathematics():
    assert {"spark_x2_6_17b_bf16", "spark_x2_6_17b_fp8"} <= set(RECIPES)
    model = type("M", (), {"config": {"architectures": ["Other"]}, "parameters": {}})
    for name in ("spark_x2_6_17b_bf16", "spark_x2_6_17b_fp8"):
        with pytest.raises(ValueError, match="Spark2.5"):
            RECIPES[name](model, Recipe.__new__(Recipe), {"base": None})
