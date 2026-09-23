# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a reproducible linear-attention QAT trial on pinned local model and data files."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import pyarrow.parquet as pq
import torch
from fla.layers.kda import KimiDeltaAttention
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.linear_attention import linear_attention_training_phase


def _token_blocks(path, tokenizer, length, count):
    text = "\n\n".join(pq.read_table(path, columns=["text"])["text"].to_pylist())
    ids = tokenizer.encode(text, add_special_tokens=False)
    needed = count * (length + 1)
    if len(ids) < needed:
        raise ValueError(f"{path} has {len(ids)} tokens; {needed} are required")
    tensor = torch.tensor(ids[:needed], dtype=torch.int64).reshape(count, length + 1)
    digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    return tensor, digest


def _phase(model, prefill_tokens):
    return (
        nullcontext()
        if prefill_tokens is None
        else linear_attention_training_phase(model, [prefill_tokens])
    )


def _labels(ids, prefill_tokens):
    labels = ids.clone()
    if prefill_tokens is not None:
        labels[:, :prefill_tokens] = -100
    return labels


def _evaluate(model, blocks, prefill_tokens=None):
    model.eval()
    losses = []
    with (
        _phase(model, prefill_tokens),
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        for block in blocks:
            ids = block.unsqueeze(0).cuda()
            loss = model(input_ids=ids, labels=_labels(ids, prefill_tokens), use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite evaluation loss")
            losses.append(float(loss.detach()))
    mean = sum(losses) / len(losses)
    return {
        "mean_nll": mean,
        "perplexity": math.exp(mean),
        "block_nll": losses,
        "predicted_tokens": len(blocks) * (blocks.shape[1] - max(1, prefill_tokens or 0)),
    }


def main():
    """Compare pre/post-training held-out NLL using an explicit numerical policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", default="arcee-ai/AFM-4.5B-Base-KDA-Only")
    parser.add_argument("--model-revision", default="01ad2e06ee4f1214193c17b69e09105a9b257e80")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--eval-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--dataset-revision", default="b08601e04326c79dfdd32d625aee71d232d685c3")
    parser.add_argument("--quant-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--eval-blocks", type=int, default=4)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument(
        "--prefill-tokens", type=int, help="Explicit prefix; score/train suffix labels only"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    options = parser.parse_args()
    if options.length < 64 or options.train_steps < 0 or options.eval_blocks < 1:
        parser.error("length must be >=64, train-steps nonnegative, and eval-blocks positive")
    if options.train_data.resolve() == options.eval_data.resolve():
        parser.error("Training and evaluation data must be distinct splits")
    if options.prefill_tokens is not None and not 0 <= options.prefill_tokens <= options.length:
        parser.error("prefill-tokens must be between zero and length")
    torch.manual_seed(options.seed)
    random.seed(options.seed)
    torch.set_float32_matmul_precision("highest")
    tokenizer = AutoTokenizer.from_pretrained(
        options.model, local_files_only=True, trust_remote_code=options.trust_remote_code
    )
    train, train_hash = _token_blocks(
        options.train_data, tokenizer, options.length, options.train_steps
    )
    evaluation, eval_hash = _token_blocks(
        options.eval_data, tokenizer, options.length, options.eval_blocks
    )
    model, loading = AutoModelForCausalLM.from_pretrained(
        options.model,
        local_files_only=True,
        trust_remote_code=options.trust_remote_code,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    if loading["missing_keys"] or loading["unexpected_keys"] or loading.get("mismatched_keys"):
        raise RuntimeError(f"Checkpoint/model mismatch: {loading}")
    model.requires_grad_(False)
    layers = [m for m in model.modules() if isinstance(m, KimiDeltaAttention)]
    if not layers:
        raise RuntimeError("The selected model contains no FLA KimiDeltaAttention layers")
    # Keep trainable attention parameters and optimizer moments in FP32 under BF16 autocast.
    for layer in layers:
        layer.requires_grad_(True)
    for parameter in model.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    model.cuda()
    quant_cfg = None
    if options.quant_config:
        quant_cfg = json.loads(options.quant_config.read_text())
        mtq.quantize(model, quant_cfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    before = _evaluate(model, evaluation, options.prefill_tokens)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=options.learning_rate, weight_decay=0)
    model.train()
    losses = []
    grad_norms = []
    timings = []
    torch.cuda.reset_peak_memory_stats()
    probe = layers[0].q_proj.weight
    original = probe.detach().clone()
    order = list(range(len(train)))
    random.Random(options.seed).shuffle(order)
    for index in order:
        optimizer.zero_grad(set_to_none=True)
        ids = train[index].unsqueeze(0).cuda()
        torch.cuda.synchronize()
        start = time.perf_counter()
        # Activation checkpoint recomputation needs the same explicit phase metadata.
        with _phase(model, options.prefill_tokens):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(
                    input_ids=ids, labels=_labels(ids, options.prefill_tokens), use_cache=False
                ).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
        gradients = [p.grad for p in trainable if p.grad is not None]
        if (
            len(gradients) != len(trainable)
            or not torch.stack([torch.isfinite(g).all() for g in gradients]).all()
        ):
            raise RuntimeError("Missing or non-finite training gradients")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        grad_norms.append(float(norm))
        timings.append(time.perf_counter() - start)
        print(
            json.dumps(
                {
                    "step": len(losses),
                    "loss": losses[-1],
                    "gradient_norm": grad_norms[-1],
                    "seconds": timings[-1],
                }
            ),
            flush=True,
        )
    query_changed = not torch.equal(original, probe)
    if options.train_steps and not query_changed:
        raise RuntimeError("Training did not update the query projection")
    peak = torch.cuda.max_memory_allocated()
    optimizer.zero_grad(set_to_none=True)
    after = _evaluate(model, evaluation, options.prefill_tokens) if options.train_steps else before
    root = Path(__file__).resolve().parents[3]
    sources = [
        root / "modelopt/torch/quantization/linear_attention" / name
        for name in [
            "config.py",
            "kda.py",
            "prefill.py",
            "decode.py",
            "decode_prefill.py",
            "reference.py",
        ]
    ]
    sources.extend(
        [
            root / "modelopt/torch/kernels/quantization/linear_attention/decode.py",
            root / "modelopt/torch/kernels/quantization/linear_attention/int8.py",
            root / "modelopt/torch/quantization/plugins/kda.py",
            root / "modelopt/torch/quantization/plugins/linear_attention.py",
            Path(__file__).resolve(),
        ]
    )
    result = {
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "model": options.model_id,
        "model_revision": options.model_revision,
        "dataset": "Salesforce/wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "dataset_revision": options.dataset_revision,
        "eval_split": options.eval_split,
        "train_file_sha256": hashlib.sha256(options.train_data.read_bytes()).hexdigest(),
        "eval_file_sha256": hashlib.sha256(options.eval_data.read_bytes()).hexdigest(),
        "train_tokens_sha256": train_hash,
        "eval_tokens_sha256": eval_hash,
        "sequence_length": options.length,
        "prefill_tokens": options.prefill_tokens,
        "loss_scope": "full" if options.prefill_tokens is None else "suffix",
        "seed": options.seed,
        "train_order": order,
        "training": "KDA attention parameters only; FP32 AdamW, BF16 autocast, gradient checkpointing",
        "trainable_parameters": sum(p.numel() for p in trainable),
        "kda_layers": len(layers),
        "learning_rate": options.learning_rate,
        "weight_decay": 0,
        "gradient_clip": 1.0,
        "quant_config": quant_cfg,
        "before": before,
        "after": after,
        "training_losses": losses,
        "gradient_norms": grad_norms,
        "step_seconds": timings,
        "train_predicted_tokens": options.train_steps
        * (options.length + 1 - max(1, options.prefill_tokens or 0)),
        "peak_train_bytes": peak,
        "query_weight_changed": query_changed,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["transformers", "flash-linear-attention", "fla-core", "triton"]
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "before": before["perplexity"],
                "after": after["perplexity"],
                "output": str(options.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
