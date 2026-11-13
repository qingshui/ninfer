"""Reference logits of the Spark2_5 checkpoints from the checkpoint's own modeling_spark (F7).

    python3 -I make_hf_reference.py --model CKPT_DIR --variant bf16|fp8 --out DIR

The checkpoint's modeling_spark / configuration_spark are loaded by path (nothing is written to
CKPT_DIR) and run in FP32 with eager attention. The BF16 variant upcasts the source weights. The
FP8 variant dequantizes each fp8 weight with its per-channel FP32 scale; activations are never
quantized. The prompt is a fixed mixed Chinese / English text rendered with the chat template, with
thinking off.

DIR receives hf-<variant>.json (schema spark2_5-hf-reference/1) and hf-<variant>.logits.f32
([sequence][position][vocabulary] float32, little endian, positions as listed in the JSON).
Exit code: 0 written, 77 environment unavailable (checkpoint, torch, transformers or CUDA).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # CKPT_DIR is read-only input; importing from it must not cache.

SKIP = 77
SCHEMA = "spark2_5-hf-reference/1"
VOCABULARY = 131072
MIN_TOKENS = 1100
PRODUCTION_CHUNK = 512
# Compared positions: the start, the neighbourhood of the 512 chunk / window boundary, the end.
HEAD, MIDDLE, TAIL = range(0, 64), range(480, 640), 64
WEIGHTS = {"bf16": "source_bf16_upcast_fp32", "fp8": "fp8_dequantized_per_channel_fp32"}
SYSTEM = "You are a careful assistant. 你是一个严谨的助手，请先阅读材料再回答。"
PARAGRAPHS = (
    "Paged attention stores the keys and values of a sequence in fixed-size blocks, so a long "
    "request does not need one contiguous allocation. 分页注意力把每条序列的键值缓存切成固定大小的"
    "页，长请求不再需要一整块连续显存，页表负责把逻辑位置映射到物理页。",
    "A sliding-window layer lets position i read only the 512 most recent positions, itself "
    "included. 滑动窗口层里，位置 i 只读取包括自身在内的最近 512 个位置，更早的键值可以被环形缓冲"
    "区覆盖，因此这类层的缓存大小不随上下文长度增长。",
    "Every fourth layer attends to the whole history, and only a quarter of its head channels are "
    "rotated by the position embedding. 每隔三层出现一层全局注意力，它只旋转头维度的前四分之一，其余"
    "通道保持原样，这样长距离的位置信息衰减得更慢。",
    "Each attention head multiplies its output by a sigmoid gate computed from the hidden state "
    "before the output projection. 每个注意力头的输出在进入输出投影之前，会乘上由隐藏状态算出的"
    "sigmoid 门控，门控是逐头的标量，而不是逐通道的向量。",
    "RMSNorm divides by the root mean square of the activations and scales by a learned gain; the "
    "epsilon is one millionth. 均方根归一化用激活的均方根做除数，再乘以可学习的增益，数值稳定项取"
    "百万分之一，残差连接保持在较高精度上累加。",
    "Weight-only and weight-and-activation quantization trade accuracy for memory traffic. "
    "权重量化降低显存读取量，权重与激活同时量化还能使用低精度张量核心，代价是激活的动态范围必须"
    "逐个词元重新估计，误差会随层数累积。",
    "Continuous batching admits a new request as soon as a running one finishes, instead of "
    "waiting for the whole batch. 连续批处理在某个请求结束的那一刻就放入新请求，而不是等整批都"
    "完成，这样调度器可以让显卡一直处于忙碌状态，同时受限于剩余的缓存页数量。",
    "Chunked prefill splits a long prompt into pieces of at most 512 tokens, so each step has a "
    "bounded cost and decoding requests are not starved. 分块预填充把很长的提示切成最多 512 个词元的"
    "小块，每一步的开销有上限，正在生成的请求也不会被长提示饿死。",
    "The embedding table is shared with the output head: the logits are the dot products between "
    "the final hidden state and every row of the table. 输出头与词嵌入共用同一张表，最终隐藏状态与"
    "词表中每一行做点积就得到逐词元的 logits，所以词表越大，最后一次矩阵乘越贵。",
    "Greedy decoding picks the token with the largest logit at every step, which makes a run "
    "reproducible for a fixed model and prompt. 贪心解码每一步都选 logit 最大的词元，因此对于固定的"
    "模型和提示，结果可以完全复现，也便于比较两个实现是否一致。",
    "A numerical comparison should say what it compares against, how many positions it covers, "
    "and which tolerance it applies. 数值对拍必须说明参照物是什么、覆盖了多少个位置、采用了怎样的"
    "容差，余弦相似度反映整体方向，最大值一致率反映最终选择是否改变。",
    "Rotary embeddings rotate pairs of channels by an angle that grows linearly with the "
    "position, using a different frequency for each pair. 旋转位置编码把成对的通道按位置线性增长的"
    "角度旋转，每一对通道使用不同的频率，查询与键的点积因此只依赖相对距离。",
    "The feed-forward block expands the hidden size to 6656, applies a GELU to one projection and "
    "multiplies it with another. 前馈块先把隐藏维度扩展到 6656，对其中一个投影施加高斯误差线性单元，"
    "再与另一个投影逐元素相乘，最后投影回 2048 维，并加到残差上。",
    "When a request is cancelled, every page it owns goes back to the pool, and the pool must "
    "report the same free bytes as before the request. 请求被取消时，它占用的每一页都要归还缓存池，"
    "池中的空闲字节数必须回到请求开始之前的数值，否则长时间运行后容量会悄悄缩水。",
    "Mixed-language text stresses the tokenizer: 中文 characters, English words, numbers like "
    "1048576, and punctuation such as “引号”、括号（全角）and (ASCII). 混合语言的文本会考验分词器："
    "汉字、英文单词、像 1048576 这样的数字，以及全角与半角标点，都应该被稳定地切分。",
    "Throughput is measured in generated tokens per second across all concurrent requests, while "
    "latency is the time until the first token and between tokens. 吞吐量按所有并发请求每秒生成的"
    "词元数计算，延迟则看首个词元出现的时间和相邻词元之间的间隔，两者需要分开报告。",
)
QUESTION = (
    "Summarize the notes above in two sentences, then list the three design choices that keep "
    "the cache small. 请用中文概括以上要点，并说明为什么这样设计可以减小显存占用。"
)


def load_checkpoint_modules(model_dir: Path):
    """Imports configuration_spark and modeling_spark of model_dir as a synthetic package."""
    name = "spark2_5_hf_checkpoint"
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = [str(model_dir)]
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    configuration = importlib.import_module(name + ".configuration_spark")
    modeling = importlib.import_module(name + ".modeling_spark")
    return configuration, modeling


def adapt_mask_api(modeling) -> list[str]:
    """Maps the mask builders' call of the checkpoint (transformers 5.12) onto the installed one.

    modeling_spark passes input_embeds= and cache_position= to create_*_mask. Later transformers
    renamed the first and dropped the second. Only the call is translated; the masks are the
    installed transformers' own, and every other line of the model runs unchanged.
    """
    import inspect

    notes = []
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        original = getattr(modeling, name)
        if "input_embeds" in inspect.signature(original).parameters:
            continue

        def wrapped(*, config, input_embeds, attention_mask, cache_position, past_key_values,
                    position_ids, _original=original):
            del cache_position
            return _original(
                config=config, inputs_embeds=input_embeds, attention_mask=attention_mask,
                past_key_values=past_key_values, position_ids=position_ids,
            )

        setattr(modeling, name, wrapped)
        notes.append(name)
    return notes


def reference_ids(model_dir: Path) -> list[int]:
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    tokenizer.chat_template = (model_dir / "chat_template.jinja").read_text()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n\n".join(PARAGRAPHS) + "\n\n" + QUESTION},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < MIN_TOKENS:
        raise SystemExit(f"the reference text has {len(ids)} tokens, need >= {MIN_TOKENS}")
    return ids


def compared_positions(count: int) -> list[int]:
    return sorted({*HEAD, *MIDDLE, *range(count - TAIL, count)})


def read_tensors(model_dir: Path) -> dict:
    from safetensors import safe_open

    index = model_dir / "model.safetensors.index.json"
    files = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.is_file()
        else ["model.safetensors"]
    )
    tensors = {}
    for file in files:
        with safe_open(model_dir / file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
    return tensors


def build_state(tensors: dict, variant: str, device) -> dict:
    """FP32 parameters on device. FP8 weights are code * per-channel scale."""
    import torch

    state, scaled = {}, 0
    for key, value in tensors.items():
        if key.endswith(".weight_scale"):
            continue
        if value.dtype == torch.float8_e4m3fn:
            scale = tensors[key + "_scale"].to(device=device, dtype=torch.float32)
            if scale.shape != (value.shape[0], 1):
                raise SystemExit(f"{key}: scale shape {tuple(scale.shape)} is not per channel")
            state[key] = value.to(device=device, dtype=torch.float32) * scale
            scaled += 1
        else:
            state[key] = value.to(device=device, dtype=torch.float32)
    if (variant == "fp8") != (scaled > 0):
        raise SystemExit(f"variant {variant} does not match the checkpoint ({scaled} fp8 weights)")
    return state


def build_model(model_dir: Path, configuration, modeling, state: dict):
    import torch

    raw = json.loads((model_dir / "config.json").read_text())
    for key in ("quantization_config", "auto_map"):
        raw.pop(key, None)
    raw["dtype"] = "float32"
    config = configuration.Spark2_5Config(**raw)
    config._attn_implementation = "eager"  # the model calls eager attention; masks must be float
    # The checkpoint declares its tie as a list, newer transformers expect a mapping. forward()
    # reads the embedding directly when tie_word_embeddings is set, so no tie is registered.
    modeling.Spark2_5ForCausalLM._tied_weights_keys = None
    with torch.device("meta"):
        model = modeling.Spark2_5ForCausalLM(config)
    result = model.load_state_dict(state, strict=False, assign=True)
    if result.unexpected_keys or set(result.missing_keys) != {"lm_head.weight"}:
        raise SystemExit(f"state mismatch: {result}")
    if not config.tie_word_embeddings:
        raise SystemExit("the checkpoint is expected to tie the head to the embedding")
    return model.eval()


def run(model, ids: list[int], positions: list[int], device):
    import torch

    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits
    selected = logits[0, positions].float().cpu()
    if tuple(selected.shape) != (len(positions), VOCABULARY) or not torch.isfinite(selected).all():
        raise SystemExit("the reference logits have the wrong shape or are not finite")
    return selected


def write_outputs(out: Path, args, ids, positions, logits, versions) -> None:
    out.mkdir(parents=True, exist_ok=True)
    name = f"hf-{args.variant}.logits.f32"
    scratch = out / (name + ".tmp")
    scratch.write_bytes(logits.numpy().astype("<f4", copy=False).tobytes())
    os.replace(scratch, out / name)
    count = len(ids)
    manifest = {
        "schema": SCHEMA,
        "variant": args.variant,
        "model_dir": str(args.model),
        "reference": "hf_modeling_spark_fp32",
        "weights": WEIGHTS[args.variant],
        "activation_quantization": "none",
        "vocabulary": VOCABULARY,
        "sequences": [
            {
                "name": "mixed_zh_en_chat",
                "ids": ids,
                "chunks": [PRODUCTION_CHUNK, PRODUCTION_CHUNK, count - 2 * PRODUCTION_CHUNK],
                "positions": positions,
            }
        ],
        "logits_file": name,
        "dtype": "float32",
        "layout": "[sequence][position][vocabulary] row-major",
        "torch_version": versions[0],
        "transformers_version": versions[1],
    }
    (out / f"hf-{args.variant}.json").write_text(json.dumps(manifest))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(WEIGHTS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    files = ("config.json", "configuration_spark.py", "modeling_spark.py", "tokenizer.json")
    if not all((args.model / name).is_file() for name in files):
        print("the Spark2_5 checkpoint is not available", file=sys.stderr)
        return SKIP
    try:
        import numpy  # noqa: F401  (Tensor.numpy for the logits file)
        import safetensors  # noqa: F401
        import torch
        import transformers
    except ImportError as error:
        print(f"torch / transformers are not available: {error}", file=sys.stderr)
        return SKIP
    if not torch.cuda.is_available():
        print("CUDA is not available for the fp32 reference", file=sys.stderr)
        return SKIP
    if sys.byteorder != "little":
        raise SystemExit("the logits file is little endian")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    configuration, modeling = load_checkpoint_modules(args.model)
    shimmed = adapt_mask_api(modeling)
    ids = reference_ids(args.model)
    positions = compared_positions(len(ids))
    model = build_model(
        args.model, configuration, modeling,
        build_state(read_tensors(args.model), args.variant, device),
    )
    logits = run(model, ids, positions, device)
    write_outputs(args.out, args, ids, positions, logits, (torch.__version__, transformers.__version__))
    print(
        f"hf reference {args.variant}: tokens={len(ids)} positions={len(positions)} "
        f"mask_api_adapted={shimmed or 'no'} torch={torch.__version__} "
        f"transformers={transformers.__version__}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
