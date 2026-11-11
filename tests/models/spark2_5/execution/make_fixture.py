"""Builds the 2-layer Spark2_5 fixture: source checkpoints and their BF16/FP8 artifacts.

The fixture keeps layer 0 (sliding attention) and layer 3 (full attention) of the real BF16 and
FP8 checkpoints as layers 0 and 1. The source also carries an lm_head.weight that differs from the
embedding (rows reversed). The checkpoints tie the head to the embedding, so Hugging Face never
reads it, and the test uses it to prove that the native head is the embedding.

Usage: make_fixture.py OUT_DIR   (checkpoint root: $NINFER_SPARK_ROOT, shared with tests/convert)
Exit code 77 when a checkpoint is absent. OUT_DIR/FIXTURE_OK marks a complete build.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[4]
CHECKPOINTS = Path(os.environ.get("NINFER_SPARK_ROOT", "/home/users/humingqing/qwen"))
VARIANTS = {"bf16": "Spark-X2.6-1.7B", "fp8": "Spark-X2.6-1.7B-FP8"}
SOURCE_LAYERS = (0, 3)
# The artifact carries text resources the definition parses. Execution does not tokenize, so the
# fixture takes the small tokenizer of the frontend tests instead of the checkpoint's own.
FRONTEND_FIXTURES = ROOT / "tests" / "models" / "spark2_5" / "frontend" / "fixtures"


class Shards:
    """Random access to the tensors of a sharded safetensors checkpoint."""

    def __init__(self, root: Path):
        self.root = root
        index = root / "model.safetensors.index.json"
        if index.is_file():
            self.files = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(root / "model.safetensors", framework="pt") as handle:
                self.files = {key: "model.safetensors" for key in handle.keys()}

    def get(self, name: str):
        with safe_open(self.root / self.files[name], framework="pt") as handle:
            return handle.get_tensor(name)

    def names(self, prefix: str):
        return [key for key in self.files if key.startswith(prefix)]


def build_source(variant: str, destination: Path) -> None:
    root = CHECKPOINTS / VARIANTS[variant]
    shards = Shards(root)
    tensors = {}
    embedding = shards.get("model.embedding.weight")
    tensors["model.embedding.weight"] = embedding
    # Not the embedding: the reversed table makes every head logit belong to another token.
    tensors["lm_head.weight"] = embedding.flip(0).contiguous()
    tensors["model.norm.weight"] = shards.get("model.norm.weight")
    for new, old in enumerate(SOURCE_LAYERS):
        for name in shards.names(f"model.layers.{old}."):
            tensors[name.replace(f"model.layers.{old}.", f"model.layers.{new}.", 1)] = (
                shards.get(name)
            )
    config = json.loads((root / "config.json").read_text())
    kinds = config["layer_types"]
    config["layer_types"] = [kinds[i] for i in SOURCE_LAYERS]
    config["num_hidden_layers"] = len(SOURCE_LAYERS)
    config["tie_word_embeddings"] = True
    quantization = config.get("quantization_config")
    if quantization is not None:
        quantization["ignore"] = [
            f"model.layers.{i}.self_attn.g_proj" for i in range(len(SOURCE_LAYERS))
        ] + ["lm_head"]
    destination.mkdir(parents=True)
    save_file(tensors, destination / "model.safetensors", metadata={"format": "pt"})
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    write_resources(destination)


def write_resources(destination: Path) -> None:
    tokenizer = json.loads((FRONTEND_FIXTURES / "tokenizer.json").read_text())
    end = "<｜end▁of▁sentence｜>"
    eos = next(item["id"] for item in tokenizer["added_tokens"] if item["content"] == end)
    shutil.copyfile(FRONTEND_FIXTURES / "tokenizer.json", destination / "tokenizer.json")
    shutil.copyfile(FRONTEND_FIXTURES / "chat_template.jinja", destination / "chat_template.jinja")
    (destination / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<｜start▁of▁sentence｜>",
                "eos_token": end,
                "pad_token": "<｜▁pad▁｜>",
            }
        )
    )
    (destination / "generation_config.json").write_text(
        json.dumps({"eos_token_id": eos, "temperature": 1.0, "top_p": 0.95, "top_k": -1})
    )


def convert(variant: str, source: Path, artifact: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.convert",
            "--model",
            str(source),
            "--recipe",
            f"spark_x2_6_17b_{variant}",
            "--out",
            str(artifact),
            "--device",
            "cpu",
            "--cpu-threads",
            "8",
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> int:
    out = Path(sys.argv[1])
    if not all((CHECKPOINTS / name / "config.json").is_file() for name in VARIANTS.values()):
        print("Spark2_5 checkpoints are not available", file=sys.stderr)
        return 77
    if (out / "FIXTURE_OK").is_file():
        return 0
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for variant in VARIANTS:
        build_source(variant, out / f"{variant}-source")
        convert(variant, out / f"{variant}-source", out / f"{variant}.ninfer")
        # The converted artifact carries every weight; only the head oracle needs the source.
        with safe_open(out / f"{variant}-source" / "model.safetensors", framework="pt") as handle:
            head = handle.get_tensor("lm_head.weight")
        (out / f"{variant}-lm_head.bin").write_bytes(
            head.contiguous().view(torch.uint16).numpy().tobytes()
        )
        shutil.rmtree(out / f"{variant}-source")
    (out / "FIXTURE_OK").write_text("ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
