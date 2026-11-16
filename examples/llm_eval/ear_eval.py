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

"""Measure probability overlap on immutable, tokenized reference trajectories."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from ear_metrics import aggregate_ear, logical_continuation, per_position_ear

__all__ = ["qualify_batch", "score_frozen"]


def _histories(rows, eos_token_ids, max_tokens):
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Require nonempty trajectories with unique IDs.")
    histories = []
    for row in rows:
        prompt = row["prompt_ids"]
        if not prompt or any(type(token) is not int or token < 0 for token in prompt):
            raise ValueError("Require nonempty, nonnegative integer prompt token IDs.")
        response = logical_continuation(row["generated_ids"], eos_token_ids, max_tokens)
        histories.append({**row, "prompt_ids": list(prompt), "generated_ids": response})
    return histories


def _hidden_states(model, rows, pad_token_id):
    decoder = model.base_model
    head = model.get_output_embeddings()
    if decoder is model or head is None:
        raise ValueError("Require a separate causal decoder and output embedding head.")
    if head.weight.device.type == "meta":
        raise ValueError("Offloaded output heads require an execution-device adapter.")
    config = getattr(model.config, "text_config", model.config)
    for name in ("logit_scale", "logits_scaling", "final_logit_softcapping", "logit_softcap"):
        value = getattr(config, name, None)
        if value not in ((None, 1) if name in ("logit_scale", "logits_scaling") else (None, 0)):
            raise ValueError(f"Post-head {name} requires an explicit scoring adapter.")
    lengths = [len(row["prompt_ids"]) + len(row["generated_ids"]) - 1 for row in rows]
    context = getattr(config, "max_position_embeddings", None)
    if context is not None and max(lengths) + 1 > context:
        raise ValueError("A full prompt plus continuation exceeds the model context.")
    device = model.get_input_embeddings().weight.device
    if device.type == "meta":
        raise ValueError("Offloaded input embeddings require an execution-device adapter.")
    inputs = torch.full((len(rows), max(lengths)), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(inputs)
    for i, (row, length) in enumerate(zip(rows, lengths)):
        inputs[i, -length:] = torch.tensor(
            (row["prompt_ids"] + row["generated_ids"])[:-1], device=device
        )
        mask[i, -length:] = 1
    outputs = decoder(
        input_ids=inputs,
        attention_mask=mask,
        position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    return [hidden[i, -len(row["generated_ids"]) :].clone() for i, row in enumerate(rows)]


@torch.inference_mode()
def score_frozen(
    reference,
    candidate,
    rows,
    eos_token_ids,
    pad_token_id=0,
    top_ks=(),
    batch_size=1,
    chunk_size=32,
    max_tokens=None,
):
    """Score every logical response position once, preserving input order and metadata.

    Decoder states are projected in bounded head chunks. Models with transformed
    output logits need an adapter; compare this route with ordinary forwards when
    qualifying an architecture. Batch size must be qualified for the actual models.
    """
    if (
        type(batch_size) is not int
        or batch_size < 1
        or type(chunk_size) is not int
        or chunk_size < 1
    ):
        raise ValueError("Require positive integer batch and chunk sizes.")
    eos_token_ids = tuple(eos_token_ids)
    top_ks = tuple(top_ks)
    histories = _histories(rows, eos_token_ids, max_tokens)
    reference.eval()
    candidate.eval()
    results = []
    for start in range(0, len(histories), batch_size):
        batch = histories[start : start + batch_size]
        p_hidden = _hidden_states(reference, batch, pad_token_id)
        q_hidden = _hidden_states(candidate, batch, pad_token_id)
        p_head, q_head = reference.get_output_embeddings(), candidate.get_output_embeddings()
        for row, p_state, q_state in zip(batch, p_hidden, q_hidden):
            chunks = {}
            for offset in range(0, len(p_state), chunk_size):
                p = p_head(p_state[offset : offset + chunk_size].to(p_head.weight.device))
                q = q_head(q_state[offset : offset + chunk_size].to(q_head.weight.device))
                values = per_position_ear(p, q.to(p.device), top_ks, chunk_size)
                for key, vector in values.items():
                    chunks.setdefault(key, []).append(vector)
            results.append({**row, "values": {k: torch.cat(v).tolist() for k, v in chunks.items()}})
    return results


@torch.inference_mode()
def qualify_batch(reference, candidate, rows, eos_token_ids, batch_sizes, atol=1e-6, **kwargs):
    """Compare every requested batch against singleton scores; report failures without fallback."""
    if (
        not rows
        or not batch_sizes
        or not math.isfinite(atol)
        or atol < 0
        or any(type(size) is not int or size < 1 for size in batch_sizes)
    ):
        raise ValueError("Require qualification rows, batch sizes and nonnegative tolerance.")
    eos_token_ids = tuple(eos_token_ids)
    if "top_ks" in kwargs:
        kwargs["top_ks"] = tuple(kwargs["top_ks"])
    singles = score_frozen(reference, candidate, rows, eos_token_ids, batch_size=1, **kwargs)
    reports = []
    for size in batch_sizes:
        if size > len(rows):
            raise ValueError("Qualification needs enough distinct rows for each batch size.")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            scored = score_frozen(
                reference, candidate, rows, eos_token_ids, batch_size=size, **kwargs
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            error = max(
                abs(a - b)
                for actual, expected in zip(scored, singles, strict=True)
                for key in expected["values"]
                for a, b in zip(actual["values"][key], expected["values"][key], strict=True)
            )
            reports.append(
                {
                    "batch_size": size,
                    "passed": error <= atol,
                    "max_absolute_error": error,
                    "seconds": elapsed,
                    "positions_per_second": sum(len(r["generated_ids"]) for r in scored) / elapsed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available()
                    else None,
                }
            )
        except torch.OutOfMemoryError:
            reports.append({"batch_size": size, "passed": False, "error": "out_of_memory"})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {
        "absolute_tolerance": atol,
        "gpu_measurement": "current_device_only",
        "prompt_ids": [r["id"] for r in rows],
        "batches": reports,
    }


def _load_checkpoint(loader, path, **kwargs):
    model, info = loader.from_pretrained(path, output_loading_info=True, **kwargs)
    failures = {
        key: info[key]
        for key in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
            "conversion_errors",
        )
        if info.get(key)
    }
    if failures:
        raise ValueError(f"Checkpoint did not load completely: {failures}")
    if any(tensor.is_meta for tensor in (*model.parameters(), *model.buffers())):
        raise ValueError("Checkpoint contains uninitialized or offloaded tensors.")
    # Successful loader diagnostics contain empty sets in some Transformers versions.
    diagnostics = {
        key: sorted(value) if isinstance(value, (set, frozenset)) else value
        for key, value in info.items()
    }
    return model.eval(), diagnostics


def main():
    """Read frozen JSON trajectories and write position scores plus explicitly weighted aggregates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eos_token_ids", type=int, nargs="+", required=True)
    parser.add_argument("--top_ks", type=int, nargs="*", default=[])
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run this example in one process; multiple visible GPUs are supported.")
    if args.output.exists():
        parser.error("Output already exists; preserve completed artifacts.")
    if args.batch_size != 1:
        parser.error("Use score_frozen with an independently qualified batch for these models.")
    # Model loading is optional when importing scoring helpers or requesting help.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import modelopt.torch.opt as mto

    mto.enable_huggingface_checkpointing()

    data = args.trajectories.read_bytes()
    rows = json.loads(data)
    if not isinstance(rows, list):
        raise ValueError("Trajectories must be an ordered JSON array.")
    tokenizers = [
        AutoTokenizer.from_pretrained(p, trust_remote_code=args.trust_remote_code)
        for p in (args.reference, args.candidate)
    ]
    if (
        tokenizers[0].get_vocab() != tokenizers[1].get_vocab()
        or tokenizers[0].all_special_ids != tokenizers[1].all_special_ids
    ):
        raise ValueError("Reference and candidate must share token-ID semantics.")
    loaded = [
        _load_checkpoint(
            AutoModelForCausalLM,
            p,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            trust_remote_code=args.trust_remote_code,
        )
        for p in (args.reference, args.candidate)
    ]
    reference, candidate = [model for model, _ in loaded]
    started = time.perf_counter()
    scored = score_frozen(
        reference,
        candidate,
        rows,
        args.eos_token_ids,
        pad_token_id=tokenizers[0].pad_token_id or 0,
        top_ks=args.top_ks,
        chunk_size=args.chunk_size,
        max_tokens=args.max_tokens,
    )
    report = {
        "settings": {
            **vars(args),
            "backend": "pytorch",
            "probability_dtype": "float64",
            "temperature": 1,
            "torch_version": torch.__version__,
            "trajectories_sha256": hashlib.sha256(data).hexdigest(),
            "checkpoint_loading_info": [info for _, info in loaded],
            "seconds": time.perf_counter() - started,
        },
        "summary": aggregate_ear(scored),
        "rows": scored,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, default=str, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
