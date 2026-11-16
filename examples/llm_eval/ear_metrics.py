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

"""Full-vocabulary and reference-top-k probability overlap on frozen histories."""

import math

import torch

__all__ = ["aggregate_ear", "logical_continuation", "per_position_ear", "selection_indices"]


@torch.inference_mode()
def per_position_ear(reference_logits, candidate_logits, top_ks=(), chunk_size=32):
    """Return CPU FP64 overlap vectors; top-k uses full-vocabulary probabilities without renormalization."""
    if reference_logits.ndim != 2 or reference_logits.shape != candidate_logits.shape:
        raise ValueError("Require matching [positions, vocabulary] logits.")
    length, vocabulary = reference_logits.shape
    if length < 1 or vocabulary < 1 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Require nonempty logits and a positive integer chunk size.")
    if not reference_logits.is_floating_point() or not candidate_logits.is_floating_point():
        raise ValueError("Logits must be floating point.")
    top_ks = tuple(top_ks)
    if any(type(k) is not int or not 1 <= k <= vocabulary for k in top_ks):
        raise ValueError("Require integer 1 <= k <= vocabulary.")
    if len(set(top_ks)) != len(top_ks):
        raise ValueError("Top-k values must be unique.")
    result = {name: [] for name in ["full", *map(str, top_ks)]}
    result.update({f"reference_mass_{k}": [] for k in top_ks})
    for start in range(0, length, chunk_size):
        p = reference_logits[start : start + chunk_size].to(dtype=torch.float64)
        q = candidate_logits[start : start + chunk_size].to(device=p.device, dtype=torch.float64)
        if not (torch.isfinite(p).all() & torch.isfinite(q).all()):
            raise ValueError("Non-finite logits cannot be scored.")
        p = p.softmax(-1)
        q = q.softmax(-1)
        overlap = torch.minimum(p, q)
        result["full"].append(overlap.sum(-1).cpu())
        for k in top_ks:
            # Select separately for each k, including when finite-precision probabilities tie.
            ids = p.topk(k, dim=-1).indices
            result[str(k)].append(overlap.gather(-1, ids).sum(-1).cpu())
            result[f"reference_mass_{k}"].append(p.gather(-1, ids).sum(-1).cpu())
    return {name: torch.cat(chunks) for name, chunks in result.items()}


def logical_continuation(generated_ids, eos_token_ids, max_tokens=None):
    """Trim a raw continuation at the first EOS/EOT inclusive, optionally applying a token cap."""
    generated_ids = list(generated_ids)
    eos_token_ids = set(eos_token_ids)
    if not generated_ids or any(type(token) is not int or token < 0 for token in generated_ids):
        raise ValueError("Require a nonempty sequence of nonnegative integer token IDs.")
    if any(type(token) is not int or token < 0 for token in eos_token_ids):
        raise ValueError("EOS IDs must be nonnegative integers.")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
        raise ValueError("Token cap must be a positive integer.")
    end = next(
        (i + 1 for i, token in enumerate(generated_ids) if token in eos_token_ids),
        len(generated_ids),
    )
    return generated_ids[: min(end, max_tokens) if max_tokens is not None else end]


def selection_indices(length, count=None, mode="prefix"):
    """Select prefix or endpoint-inclusive spaced positions, recomputing the grid for each count."""
    if type(length) is not int or length < 1:
        raise ValueError("Position length must be a positive integer.")
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError("Position count must be a positive integer.")
    if mode not in ("prefix", "spaced"):
        raise ValueError("Selection must be prefix or spaced.")
    selected = length if count is None else min(count, length)
    if mode == "prefix" or selected == 1:
        return list(range(selected))
    return [i * (length - 1) // (selected - 1) for i in range(selected)]


def aggregate_ear(
    rows, prompt_count=None, max_tokens=None, position_count=None, selection="prefix"
):
    """Aggregate ordered {'id', 'values': {metric: position scores}} rows with equal prompt weights."""
    if not rows:
        raise ValueError("Require scored prompts.")
    if prompt_count is not None and (
        type(prompt_count) is not int or not 1 <= prompt_count <= len(rows)
    ):
        raise ValueError("Prompt count must fit the frozen bank; prompts are never replicated.")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
        raise ValueError("Token cap must be a positive integer.")
    metrics = tuple(rows[0]["values"])
    if not metrics or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Require metrics and unique prompt IDs.")
    prompt_means = {name: [] for name in metrics}
    position_sums = {name: [] for name in metrics}
    selected_positions = 0
    selected_rows = rows if prompt_count is None else rows[:prompt_count]
    for row in selected_rows:
        if set(row["values"]) != set(metrics):
            raise ValueError("Every prompt must contain the same metrics.")
        lengths = {len(values) for values in row["values"].values()}
        if len(lengths) != 1 or next(iter(lengths)) < 1:
            raise ValueError("Every metric must contain the same nonempty valid positions.")
        length = next(iter(lengths))
        indices = selection_indices(
            min(length, max_tokens) if max_tokens is not None else length, position_count, selection
        )
        selected_positions += len(indices)
        for name, values in row["values"].items():
            values = values.tolist() if isinstance(values, torch.Tensor) else values
            if any(
                not math.isfinite(value) or not -1e-12 <= value <= 1 + 1e-12 for value in values
            ):
                raise ValueError("Every position score must be finite and within [0, 1].")
            total = math.fsum(values[index] for index in indices)
            prompt_means[name].append(total / len(indices))
            position_sums[name].append(total)
    return {
        "prompt_count": len(selected_rows),
        "max_tokens": max_tokens,
        "position_count": position_count,
        "selection": selection,
        "selected_positions": selected_positions,
        "equal_prompt": {
            name: math.fsum(values) / len(selected_rows) for name, values in prompt_means.items()
        },
        "native_flat": {
            name: math.fsum(values) / selected_positions for name, values in position_sums.items()
        },
    }
