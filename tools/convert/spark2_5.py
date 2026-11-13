"""Spark2.5 architecture adapter: mathematical config, source axes, and packing groups.

The BF16 and compressed-tensors FP8 checkpoints share this mapping. Official
checkpoint names and representation assignments live in official_recipes.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path
import struct
from typing import Mapping

from .model import Model, Parameter
from .resources import TEXT_RESOURCES, load_resources
from .sources.logical import select_rows
from .sources.safetensors import SafetensorsSource, tensor_source
from .sources.compressed_tensors import matrix_source

_ARCHITECTURE = "Spark2_5ForCausalLM"
_LAYER_TYPES = ("full_attention", "sliding_attention")


def _error(message: str) -> ValueError:
    return ValueError("spark2_5: " + message)


def _positive(raw: dict, key: str) -> int:
    value = raw.get(key)
    if type(value) is not int or value <= 0:
        raise _error(f"{key}: expected a positive integer")
    return value


def _token_id(raw: dict, key: str) -> int:
    value = raw.get(key)
    if type(value) is not int or value < 0:
        raise _error(f"{key}: expected a nonnegative integer")
    return value


def _f32(value, name: str) -> float:
    if type(value) not in (float, int):
        raise _error(f"{name}: expected a real value")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise _error(f"{name}: not representable as F32") from error
    if not isfinite(result) or result <= 0:
        raise _error(f"{name}: expected positive finite F32")
    return result


def layer_rope(source: dict, layer_type: str) -> tuple[int, float]:
    """Return (rotary_dim, theta) of one layer type.

    Only rope_parameters[layer_type] is read. Scalar rope_theta/rope_type keys
    beside the per-type tables, and top-level keys of the same names, belong to
    other consumers of the checkpoint and are ignored.
    """
    if layer_type not in _LAYER_TYPES:
        raise _error(f"unknown layer type {layer_type!r}")
    head_dim = _positive(source, "head_dim")
    parameters = source.get("rope_parameters")
    if not isinstance(parameters, dict):
        raise _error("rope_parameters: expected an object")
    table = parameters.get(layer_type)
    if not isinstance(table, dict):
        raise _error(f"rope_parameters.{layer_type}: missing table")
    label = f"rope_parameters.{layer_type}"
    if table.get("rope_type", "default") != "default":
        raise _error(f"{label}.rope_type: scaled RoPE is not implemented")
    factor = table.get("partial_rotary_factor")
    if type(factor) not in (float, int) or not 0 < factor <= 1:
        raise _error(f"{label}.partial_rotary_factor: expected a value in (0, 1]")
    theta = _f32(table.get("rope_theta"), f"{label}.rope_theta")
    rotary_dim = int(head_dim * factor)
    if rotary_dim <= 0 or rotary_dim % 2:
        raise _error(f"{label}: rotary width {rotary_dim} must be positive and even")
    return rotary_dim, theta


def text_config(source: dict) -> dict:
    """Validate the source config and return the artifact's mathematical text config."""
    architectures = source.get("architectures")
    if architectures != [_ARCHITECTURE]:
        raise _error(f"unsupported architectures {architectures!r}")
    if source.get("hidden_act") != "gelu":
        raise _error(f"hidden_act: expected 'gelu', got {source.get('hidden_act')!r}")
    if source.get("gate_attn_act_mode") != "sigmoid":
        raise _error("gate_attn_act_mode: expected 'sigmoid'")
    if source.get("headwise_attn_output_gate") is not True:
        raise _error("headwise_attn_output_gate: expected true")
    for key in ("attention_bias", "mlp_bias"):
        if source.get(key, False):
            raise _error(f"{key}: biased projections are not supported")
    result = {"architectures": ["Spark2_5ForCausalLM"], "model_type": "spark2_5_text"}
    for key in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
    ):
        result[key] = _positive(source, key)
    if result["num_attention_heads"] % result["num_key_value_heads"]:
        raise _error("attention heads must be divisible by KV heads")
    result["rms_norm_eps"] = _f32(source.get("rms_norm_eps", 1e-6), "rms_norm_eps")
    result["hidden_act"] = "gelu"
    result["attention_gate"] = "headwise_sigmoid"
    result["sliding_window"] = _positive(source, "sliding_window")
    layers = source.get("layer_types")
    if not isinstance(layers, list) or len(layers) != result["num_hidden_layers"]:
        raise _error("layer_types: must describe every layer")
    for kind in layers:
        if kind not in _LAYER_TYPES:
            raise _error(f"layer_types: unknown layer type {kind!r}")
    result["layer_types"] = list(layers)
    result["rope"] = {}
    for kind in _LAYER_TYPES:
        if kind in layers:
            dim, theta = layer_rope(source, kind)
            result["rope"][kind] = {"rotary_dim": dim, "theta": theta}
    tied = source.get("tie_word_embeddings", False)
    if type(tied) is not bool:
        raise _error("tie_word_embeddings: expected a boolean")
    result["tie_word_embeddings"] = tied
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        result[key] = _token_id(source, key)
    return result


class _Builder:
    def __init__(self, model: Model, store: SafetensorsSource):
        self.model = model
        self.store = store

    def add(
        self,
        name,
        source_name,
        shape,
        *,
        inputs=(),
        source_shape=None,
        rows=None,
    ):
        shape = tuple(shape)
        original = shape if source_shape is None else tuple(source_shape)

        def factory(selected, format=None):
            if len(original) == 2:
                ref = matrix_source(selected, source_name, original, format)
                return select_rows(ref, rows) if rows is not None else ref
            if format is not None:
                raise ValueError(f"{name}: provide an explicit encoded source mapping")
            return tensor_source(selected, source_name, shape)

        self.model.add(
            Parameter(
                name,
                shape,
                factory(self.store),
                factory,
                tuple(inputs),
                "bf16",
                residency="text",
            )
        )

    def group(self, *names):
        self.model.packing_groups.append(tuple(names))

    def layer(self, index, config):
        h, d = config["hidden_size"], config["head_dim"]
        q = config["num_attention_heads"] * d
        kv = config["num_key_value_heads"] * d
        p, sp = f"text/layers/{index}/", f"model.layers.{index}."
        self.add(p + "input_norm", sp + "input_layernorm.weight", (h,))
        mixer = (p + "mixer_input",)
        for role, begin, count in (
            ("query", 0, q),
            ("key", q, kv),
            ("value", q + kv, kv),
        ):
            self.add(
                p + "attention/" + role,
                sp + "self_attn.q_k_v_proj.weight",
                (count, h),
                source_shape=(q + 2 * kv, h),
                rows=((begin, begin + count),),
                inputs=mixer,
            )
        self.add(
            p + "attention/gate",
            sp + "self_attn.g_proj.weight",
            (config["num_attention_heads"], h),
            inputs=mixer,
        )
        self.add(
            p + "attention/output",
            sp + "self_attn.out_proj.weight",
            (h, q),
            inputs=(p + "attention/gated_output",),
        )
        self.add(
            p + "post_attention_norm", sp + "post_attention_layernorm.weight", (h,)
        )
        intermediate = config["intermediate_size"]
        for role in ("gate", "up"):
            self.add(
                p + "mlp/" + role,
                sp + "mlp." + role + "_proj.weight",
                (intermediate, h),
                inputs=(p + "ffn_input",),
            )
        self.add(
            p + "mlp/down",
            sp + "mlp.down_proj.weight",
            (h, intermediate),
            inputs=(p + "mlp/product",),
        )
        self.group(*(p + "attention/" + role for role in ("query", "key", "value")))
        self.group(p + "mlp/gate", p + "mlp/up")


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    companions: Mapping[str, SafetensorsSource] | None = None,
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    if tuple(components) != ("text",):
        raise _error("only the text component is supported")
    if companions:
        raise _error("companion sources are not supported")
    config = text_config(base.config)
    references, payloads, count, special = load_resources(
        base.root,
        vocab_size=config["vocab_size"],
        roles={"text": TEXT_RESOURCES},
        overrides=resource_overrides,
    )
    model = Model(
        {"text": {"config": config, "resources": references["text"]}},
        resources=payloads,
        token_count=count,
        special_token_ids=special,
    )
    builder = _Builder(model, base)
    h, vocabulary = config["hidden_size"], config["vocab_size"]
    builder.add("text/token_embedding", "model.embedding.weight", (vocabulary, h))
    builder.add(
        "text/output_head",
        (
            "model.embedding.weight"
            if config["tie_word_embeddings"]
            else "lm_head.weight"
        ),
        (vocabulary, h),
        inputs=("text/final_hidden",),
    )
    builder.add("text/final_norm", "model.norm.weight", (h,))
    for index in range(config["num_hidden_layers"]):
        builder.layer(index, config)
    return model
