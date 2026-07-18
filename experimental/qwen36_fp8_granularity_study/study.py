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
"""Qwen3.6 fake-quant FP8 granularity study.

The program intentionally evaluates exactly one model and one quantization
candidate per process.  Reference logits are captured before ModelOpt mutates
the language model, and can be shared by later jobs through an exact-match
cache.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

# Prefer the checkout containing this experiment over an unrelated editable
# ModelOpt installation when this file is invoked directly by path.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

SCHEMA_VERSION = "qwen36-fp8-granularity-study-v1"
REFERENCE_SCHEMA_VERSION = "qwen36-reference-logits-v2"
SUPPORTED_MODELS = (
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.6-27B",
)
RECIPE_IDS = (
    "per_tensor_fp8",
    "per_tensor_fp8_weight_only_control",
    "block128_static_weight_only",
    "block128_dynamic_w8a8_research",
    "block128_dynamic_weight_only_control",
    "mxfp8",
    "mxfp8_weight_only_control",
)
TOP_LEVEL_DYNAMIC_RECIPES = frozenset(
    {
        "block128_dynamic_w8a8_research",
        "block128_dynamic_weight_only_control",
    }
)
EXTRA_EXCLUSIONS = (
    "*mtp*",
    "*shared_expert_gate*",
)
QUANTILE_POINTS = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260717
_DYNAMIC_BLOCK_SHAPE_CACHE_ATTRIBUTES = (
    "_original_shape",
    "_block_reshape_size",
    "_padding",
    "_slices",
    "_amax_shape_for_export",
)
_INDEXED_QUANTIZER_RE = re.compile(
    r"(?:^|\.)(?P<stem>[^.]*?(?P<role>weight|input)_quantizers?)\.\d+$"
)


@dataclasses.dataclass(frozen=True)
class Recipe:
    """Resolved ModelOpt candidate and study metadata."""

    recipe_id: str
    config: dict[str, Any]
    activation_quantized: bool
    weight_granularity: str
    activation_granularity: str | None
    weight_scale_bits: int
    activation_scale_bits: int | None
    backend_semantics: str
    deployable: bool
    notes: tuple[str, ...]

    def plan_metadata(self) -> dict[str, Any]:
        """Return JSON-safe recipe metadata."""
        return {
            "recipe_id": self.recipe_id,
            "activation_quantized": self.activation_quantized,
            "weight_granularity": self.weight_granularity,
            "activation_granularity": self.activation_granularity,
            "weight_scale_bits": self.weight_scale_bits,
            "activation_scale_bits": self.activation_scale_bits,
            "backend_semantics": self.backend_semantics,
            "fake_quant": True,
            "deployable": self.deployable,
            "notes": list(self.notes),
            "resolved_modelopt_config": json_safe(self.config),
        }


def _quantizer_rule(config: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [rule for rule in config["quant_cfg"] if rule.get("quantizer_name") == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name!r} rule, found {len(matches)}")
    return matches[0]


def classify_block_semantics(attribute_config: Mapping[str, Any]) -> str:
    """Describe the TensorQuantizer branch selected by an attribute config.

    ModelOpt uses two independent dynamic switches.  Top-level ``type`` controls
    whether amax is collected or computed on each forward.  Nested
    ``block_sizes['type']`` selects the dynamic-block kernel (MX formats here).
    """
    block_sizes = attribute_config.get("block_sizes")
    if not isinstance(block_sizes, Mapping):
        return "not_block_quantized"
    if block_sizes.get("type") == "dynamic":
        return "nested_dynamic_block_kernel"
    if attribute_config.get("type") == "dynamic":
        return "static_block_reshape_with_dynamic_full_precision_amax"
    return "static_block_calibrated_amax"


def validate_research_dynamic_attribute(attribute_config: Mapping[str, Any]) -> None:
    """Reject the easy-to-miss unsupported variant of the research branch."""
    semantics = classify_block_semantics(attribute_config)
    expected = "static_block_reshape_with_dynamic_full_precision_amax"
    if semantics != expected:
        raise ValueError(
            "The block128 research path requires top-level type='dynamic' and must not use "
            "nested block_sizes['type']; without the top-level switch algorithm=None leaves the "
            f"static block path without calibrated amax (observed {semantics!r})."
        )


def resolve_recipe(recipe_id: str) -> Recipe:
    """Resolve a candidate from the shipped ModelOpt presets."""
    if recipe_id not in RECIPE_IDS:
        raise ValueError(f"Unsupported recipe {recipe_id!r}; choose one of {RECIPE_IDS}")

    # ModelOpt is deliberately imported here so --help and pure metric helpers
    # stay usable without initializing the quantization plugin stack.
    import modelopt.torch.quantization as mtq

    is_weight_only = recipe_id.endswith("weight_only_control")
    common_notes = (
        "Language-model target only; default preset exclusions plus explicit MTP exclusions.",
        "Research measurements use ModelOpt fake quantization and are not exported.",
    )

    if recipe_id.startswith("per_tensor_fp8"):
        config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
        if is_weight_only:
            config["quant_cfg"].append({"quantizer_name": "*input_quantizer", "enable": False})
        recipe = Recipe(
            recipe_id=recipe_id,
            config=config,
            activation_quantized=not is_weight_only,
            weight_granularity="per_tensor",
            activation_granularity=None if is_weight_only else "per_tensor",
            weight_scale_bits=32,
            activation_scale_bits=None if is_weight_only else 32,
            backend_semantics="modelopt_scaled_e4m3_static_amax",
            deployable=not is_weight_only,
            notes=common_notes,
        )
    elif recipe_id == "block128_static_weight_only":
        config = copy.deepcopy(mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG)
        recipe = Recipe(
            recipe_id=recipe_id,
            config=config,
            activation_quantized=False,
            weight_granularity="block_128x128",
            activation_granularity=None,
            weight_scale_bits=32,
            activation_scale_bits=None,
            backend_semantics="modelopt_scaled_e4m3_static_calibrated_amax",
            deployable=False,
            notes=(*common_notes, "Diagnostic static-weight, weight-only granularity ablation."),
        )
    elif recipe_id.startswith("block128_dynamic"):
        config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
        weight_cfg = {
            "num_bits": (4, 3),
            "type": "dynamic",
            "block_sizes": {-2: 128, -1: 128},
            "fake_quant": True,
        }
        input_cfg = {
            "num_bits": (4, 3),
            "type": "dynamic",
            "block_sizes": {-1: 128},
            "fake_quant": True,
        }
        validate_research_dynamic_attribute(weight_cfg)
        validate_research_dynamic_attribute(input_cfg)
        _quantizer_rule(config, "*weight_quantizer")["cfg"] = weight_cfg
        _quantizer_rule(config, "*input_quantizer")["cfg"] = input_cfg
        config["algorithm"] = None
        if is_weight_only:
            config["quant_cfg"].append({"quantizer_name": "*input_quantizer", "enable": False})
        recipe = Recipe(
            recipe_id=recipe_id,
            config=config,
            activation_quantized=not is_weight_only,
            weight_granularity="block_128x128",
            activation_granularity=None if is_weight_only else "last_axis_block_128",
            weight_scale_bits=32,
            activation_scale_bits=None if is_weight_only else 32,
            backend_semantics="modelopt_top_level_dynamic_full_precision_amax",
            deployable=False,
            notes=(
                *common_notes,
                "Top-level type=dynamic selects reshape/scaled-E4M3 with dynamically computed "
                "full-precision amax; it is not the nested dynamic-block/MX path.",
                "A study-only forward pre-hook refreshes TensorQuantizer block-shape caches on "
                "every invocation so shared fused-MoE input quantizers can accept varying routed-"
                "token counts. This makes the candidate explicitly non-exportable.",
            ),
        )
    else:
        config = copy.deepcopy(mtq.MXFP8_DEFAULT_CFG)
        if is_weight_only:
            config["quant_cfg"].append({"quantizer_name": "*input_quantizer", "enable": False})
        recipe = Recipe(
            recipe_id=recipe_id,
            config=config,
            activation_quantized=not is_weight_only,
            weight_granularity="last_axis_block_32",
            activation_granularity=None if is_weight_only else "last_axis_block_32",
            weight_scale_bits=8,
            activation_scale_bits=None if is_weight_only else 8,
            backend_semantics="modelopt_nested_dynamic_block_mxfp8_e8m0",
            deployable=not is_weight_only,
            notes=common_notes,
        )

    for pattern in EXTRA_EXCLUSIONS:
        recipe.config["quant_cfg"].append({"quantizer_name": pattern, "enable": False})
    return recipe


def json_safe(value: Any) -> Any:
    """Recursively convert common Python/Torch objects to strict JSON values."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value cannot be serialized: {value}")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [json_safe(item) for item in value]
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return json_safe(value.detach().cpu().item())
        return json_safe(value.detach().cpu().tolist())
    return str(value)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace a JSON file without exposing a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(json_safe(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically save one reference-logit batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(dict(payload), temporary_name)
        os.replace(temporary_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)


def canonical_hash(payload: Mapping[str, Any]) -> str:
    """Hash a mapping through its canonical strict-JSON representation."""
    encoded = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile_name(point: float) -> str:
    return f"p{round(point * 100):02d}"


def summarize_values(values: torch.Tensor) -> dict[str, Any]:
    """Summarize a one-dimensional finite tensor."""
    values = values.detach().to(dtype=torch.float64, device="cpu").flatten()
    if values.numel() == 0:
        return {"count": 0, "mean": None, "std": None, "quantiles": {}}
    if not torch.isfinite(values).all():
        raise ValueError("Metric aggregation received a non-finite value")
    quantiles = torch.quantile(values, torch.tensor(QUANTILE_POINTS, dtype=torch.float64))
    return {
        "count": values.numel(),
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "max": values.max().item(),
        "quantiles": {
            _quantile_name(point): quantiles[index].item()
            for index, point in enumerate(QUANTILE_POINTS)
        },
    }


def paired_document_bootstrap(
    values: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap an equal-document mean of paired quantized/reference measurements."""
    observations = torch.as_tensor(values, dtype=torch.float64)
    if observations.ndim != 1 or observations.numel() == 0:
        raise ValueError("Document bootstrap requires a non-empty one-dimensional sample")
    if resamples <= 0:
        raise ValueError("Document bootstrap resample count must be positive")
    if not torch.isfinite(observations).all():
        raise ValueError("Document bootstrap received a non-finite value")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        observations.numel(),
        (resamples, observations.numel()),
        generator=generator,
    )
    means = observations[indices].mean(dim=1)
    bounds = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "document_count": observations.numel(),
        "resamples": resamples,
        "seed": seed,
        "confidence_level": 0.95,
        "point_estimate_equal_document_mean": observations.mean().item(),
        "percentile_interval": {
            "lower": bounds[0].item(),
            "upper": bounds[1].item(),
        },
    }


class OutputMetricAccumulator:
    """Batchwise causal-LM output-distance accumulator."""

    def __init__(self, epsilon: float = 1.0e-8) -> None:
        """Initialize an empty accumulator with a positive numerical epsilon."""
        self.epsilon = epsilon
        self._token_values: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._sample_values: list[dict[str, Any]] = []
        self._next_sample = 0

    @staticmethod
    def _masked_rows(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return values[mask]

    def add_batch(
        self,
        reference_logits: torch.Tensor,
        quantized_logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_indices: Sequence[int] | None = None,
    ) -> None:
        """Add one fixed-shape batch using next-token positions only."""
        if reference_logits.shape != quantized_logits.shape:
            raise ValueError(
                f"Reference and quantized logits differ: {reference_logits.shape} vs "
                f"{quantized_logits.shape}"
            )
        if reference_logits.ndim != 3:
            raise ValueError("Expected logits with shape [batch, sequence, vocabulary]")
        if input_ids.shape != attention_mask.shape or input_ids.shape != reference_logits.shape[:2]:
            raise ValueError("Token tensors must match the first two logit dimensions")

        reference = reference_logits[:, :-1].float()
        quantized = quantized_logits[:, :-1].float()
        labels = input_ids[:, 1:].long()
        # A causal next-token position is valid only when both the source logit
        # position and its target token are real. With left padding, checking the
        # target alone would incorrectly score the first real token from a padded
        # source position.
        valid = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
        if not valid.any():
            raise ValueError("Evaluation batch contains no valid next-token positions")

        difference = quantized - reference
        logit_mse = difference.square().mean(dim=-1)
        token_metrics: dict[str, torch.Tensor] = {
            "logit_mse": logit_mse,
            "logit_rmse_per_token": logit_mse.sqrt(),
            "logit_mae": difference.abs().mean(dim=-1),
        }

        reference_centered = reference - reference.mean(dim=-1, keepdim=True)
        quantized_centered = quantized - quantized.mean(dim=-1, keepdim=True)
        token_metrics["centered_logit_mse"] = (
            (quantized_centered - reference_centered).square().mean(dim=-1)
        )

        reference_variance = reference_centered.square().mean(dim=-1)
        token_metrics["variance_normalized_logit_mse"] = (
            quantized_centered - reference_centered
        ).square().mean(dim=-1) / (reference_variance + self.epsilon)

        reference_log_prob = reference.log_softmax(dim=-1)
        quantized_log_prob = quantized.log_softmax(dim=-1)
        reference_prob = reference_log_prob.exp()
        quantized_prob = quantized_log_prob.exp()
        token_metrics["forward_kl_ref_to_quant"] = (
            reference_prob * (reference_log_prob - quantized_log_prob)
        ).sum(dim=-1)
        token_metrics["reverse_kl_quant_to_ref"] = (
            quantized_prob * (quantized_log_prob - reference_log_prob)
        ).sum(dim=-1)
        log_mixture = torch.logaddexp(reference_log_prob, quantized_log_prob) - math.log(2.0)
        token_metrics["jensen_shannon"] = 0.5 * (
            (reference_prob * (reference_log_prob - log_mixture)).sum(dim=-1)
            + (quantized_prob * (quantized_log_prob - log_mixture)).sum(dim=-1)
        )

        target_reference = reference_log_prob.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        target_quantized = quantized_log_prob.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        target_error = target_quantized - target_reference
        token_metrics.update(
            {
                "target_logprob_error": target_error,
                "target_logprob_absolute_error": target_error.abs(),
                "target_logprob_squared_error": target_error.square(),
                "reference_nll": -target_reference,
                "quantized_nll": -target_quantized,
                "nll_delta_quant_minus_ref": target_reference - target_quantized,
                "top1_agreement": (reference.argmax(dim=-1) == quantized.argmax(dim=-1)).float(),
            }
        )

        top_k = min(5, reference.shape[-1])
        reference_top = reference.topk(top_k, dim=-1).indices
        quantized_top = quantized.topk(top_k, dim=-1).indices
        overlap = (reference_top.unsqueeze(-1) == quantized_top.unsqueeze(-2)).any(
            dim=-1
        ).float().sum(dim=-1) / top_k
        token_metrics["top5_set_overlap"] = overlap

        for name, values in token_metrics.items():
            self._token_values[name].append(self._masked_rows(values, valid).detach().cpu())

        if sample_indices is None:
            sample_indices = range(self._next_sample, self._next_sample + reference.shape[0])
        if len(sample_indices) != reference.shape[0]:
            raise ValueError("sample_indices length must equal batch size")
        for row, sample_index in enumerate(sample_indices):
            row_mask = valid[row]
            if not row_mask.any():
                continue
            metrics = {
                name: values[row][row_mask].mean().item() for name, values in token_metrics.items()
            }
            metrics["logit_rmse"] = math.sqrt(metrics["logit_mse"])
            metrics["target_logprob_rmse"] = math.sqrt(metrics["target_logprob_squared_error"])
            self._sample_values.append(
                {
                    "sample_index": int(sample_index),
                    "token_count": int(row_mask.sum().item()),
                    "metrics": metrics,
                }
            )
        self._next_sample += reference.shape[0]

    def finalize(self) -> dict[str, Any]:
        """Return per-token aggregates and per-sample distributions."""
        if not self._token_values:
            raise ValueError("No output metrics were accumulated")
        flattened = {name: torch.cat(chunks) for name, chunks in self._token_values.items()}
        aggregate = {name: values.double().mean().item() for name, values in flattened.items()}
        aggregate["logit_rmse"] = math.sqrt(aggregate["logit_mse"])
        aggregate["target_logprob_rmse"] = math.sqrt(aggregate["target_logprob_squared_error"])
        per_sample_metric_names = sorted(self._sample_values[0]["metrics"])
        per_sample_distributions = {
            name: summarize_values(
                torch.tensor(
                    [sample["metrics"][name] for sample in self._sample_values],
                    dtype=torch.float64,
                )
            )
            for name in per_sample_metric_names
        }
        document_bootstrap = {
            name: paired_document_bootstrap(
                [sample["metrics"][name] for sample in self._sample_values]
            )
            for name in per_sample_metric_names
        }
        return {
            "orientation": {
                "logit_difference": "quantized_minus_reference",
                "forward_kl": "KL(reference || quantized)",
                "reverse_kl": "KL(quantized || reference)",
                "nll_delta": "quantized_nll_minus_reference_nll",
                "top5": "symmetric top-5 set intersection divided by min(5, vocab_size)",
                "variance_normalization": (
                    "centered logit MSE divided by per-token reference-logit variance plus epsilon"
                ),
            },
            "token_count": int(next(iter(flattened.values())).numel()),
            "sample_count": len(self._sample_values),
            "aggregate_per_token": aggregate,
            "per_token_distributions": {
                name: summarize_values(values) for name, values in sorted(flattened.items())
            },
            "per_sample": {
                "values": self._sample_values,
                "distributions": per_sample_distributions,
            },
            "paired_document_bootstrap": {
                "estimator": (
                    "Percentile bootstrap over evaluation documents. Each document contributes "
                    "its mean of paired quantized-versus-reference token metrics with equal "
                    "document weight; this is distinct from aggregate_per_token."
                ),
                "metrics": document_bootstrap,
            },
        }


def estimate_tensor_cost(shape: Sequence[int], recipe_id: str) -> dict[str, Any]:
    """Estimate logical FP8 payload plus scale storage for one weight tensor."""
    if recipe_id not in RECIPE_IDS:
        raise ValueError(f"Unknown recipe {recipe_id!r}")
    if not shape or any(int(dimension) <= 0 for dimension in shape):
        raise ValueError(f"Invalid tensor shape {tuple(shape)}")
    dimensions = [int(dimension) for dimension in shape]
    elements = math.prod(dimensions)
    payload_bits = elements * 8

    if recipe_id.startswith("per_tensor"):
        scale_count, scale_bits = 1, 32
        scale_layout = "one FP32 scale per tensor"
    elif recipe_id.startswith("block128"):
        if len(dimensions) < 2:
            raise ValueError("128x128 weight blocks require a tensor with at least two dimensions")
        leading = math.prod(dimensions[:-2]) if len(dimensions) > 2 else 1
        scale_count = leading * math.ceil(dimensions[-2] / 128) * math.ceil(dimensions[-1] / 128)
        scale_bits = 32
        scale_layout = "one FP32 scale per logical 128x128 block"
    else:
        leading = math.prod(dimensions[:-1]) if len(dimensions) > 1 else 1
        scale_count = leading * math.ceil(dimensions[-1] / 32)
        scale_bits = 8
        scale_layout = "one E8M0 scale per last-axis block of 32"

    overhead_bits = scale_count * scale_bits
    total_bits = payload_bits + overhead_bits
    return {
        "shape": dimensions,
        "element_count": elements,
        "payload_bits": payload_bits,
        "scale_count": scale_count,
        "scale_bits_each": scale_bits,
        "scale_overhead_bits": overhead_bits,
        "total_bits": total_bits,
        "effective_bits_per_weight": total_bits / elements,
        "scale_layout": scale_layout,
        "assumptions": "Logical fake-quant cost; excludes allocator padding, metadata, and kernel workspace.",
    }


def module_family(name: str) -> str:
    """Map a quantizer/module name to a stable reporting family."""
    lowered = name.lower()
    if "linear_attn" in lowered or "linear_attention" in lowered:
        return "linear_attention"
    if "shared_expert" in lowered:
        return "shared_expert"
    if "expert" in lowered or ".moe" in lowered:
        return "experts"
    if "router" in lowered or "gate" in lowered:
        return "router_or_gate"
    if "self_attn" in lowered or ".attn" in lowered or "attention" in lowered:
        return "attention"
    if "mlp" in lowered or "feed_forward" in lowered:
        return "mlp"
    if "lm_head" in lowered:
        return "lm_head"
    if "mtp" in lowered:
        return "mtp"
    if "vision" in lowered or "visual" in lowered:
        return "vision"
    return "other"


def summarize_named_mse(values: Mapping[str, float], eligible: Iterable[str]) -> dict[str, Any]:
    """Preserve raw quantizer MSE and add family/coverage summaries."""
    eligible_set = set(eligible)
    raw: dict[str, float] = {}
    by_family: dict[str, list[float]] = defaultdict(list)
    for name, value in sorted(values.items()):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Non-finite quantization MSE for {name}: {numeric}")
        raw[name] = numeric
        by_family[module_family(name)].append(numeric)
    return {
        "by_quantizer": raw,
        "families": {
            family: summarize_values(torch.tensor(family_values, dtype=torch.float64))
            for family, family_values in sorted(by_family.items())
        },
        "coverage": {
            "eligible_count": len(eligible_set),
            "executed_count": len(raw),
            "missing_quantizers": sorted(eligible_set - set(raw)),
            "note": (
                "MSE averages one hook invocation at a time. Missing routed-expert quantizers "
                "were not executed by these calibration batches and are not zero-error results."
            ),
        },
    }


def normalize_batch_shape(
    batch: Mapping[str, torch.Tensor], batch_size: int, sequence_length: int, pad_token_id: int
) -> dict[str, torch.Tensor]:
    """Return CPU input IDs and masks with an exact, left-padded shape."""
    input_ids = batch["input_ids"].detach().cpu()
    attention_mask = batch["attention_mask"].detach().cpu()
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("Expected input_ids and attention_mask with shape [batch, sequence]")
    if input_ids.shape[0] != batch_size:
        raise ValueError(
            f"Short batch {input_ids.shape[0]} encountered; sample count must be divisible by "
            f"batch size {batch_size}"
        )
    if input_ids.shape[1] > sequence_length:
        input_ids = input_ids[:, -sequence_length:]
        attention_mask = attention_mask[:, -sequence_length:]
    padding = sequence_length - input_ids.shape[1]
    if padding:
        input_ids = torch.nn.functional.pad(input_ids, (padding, 0), value=pad_token_id)
        attention_mask = torch.nn.functional.pad(attention_mask, (padding, 0), value=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def sample_ids_from_batches(
    batches: Sequence[Mapping[str, torch.Tensor]], ordinal_offset: int = 0
) -> list[str]:
    """Create deterministic sample IDs from the exact non-padding token sequence."""
    sample_ids: list[str] = []
    ordinal = ordinal_offset
    for batch in batches:
        for input_ids, attention_mask in zip(batch["input_ids"], batch["attention_mask"]):
            tokens = input_ids[attention_mask.bool()].to(dtype=torch.int64).contiguous()
            digest = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
            sample_ids.append(f"{ordinal:08d}:{digest}")
            ordinal += 1
    return sample_ids


def build_reference_signature(
    args: argparse.Namespace,
    tokenizer: Any,
    eval_batches: Sequence[Mapping[str, torch.Tensor]],
    model_dtype: torch.dtype,
    full_model: torch.nn.Module,
    hf_config: Any,
    eval_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact provenance contract used as the reference-cache key."""
    batch_hashes = [
        canonical_hash(
            {
                "input_ids": batch["input_ids"].tolist(),
                "attention_mask": batch["attention_mask"].tolist(),
            }
        )
        for batch in eval_batches
    ]
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "model": args.model,
        "model_revision": args.revision,
        "model_dtype": str(model_dtype).removeprefix("torch."),
        "tokenizer": args.tokenizer or args.model,
        "tokenizer_revision": args.tokenizer_revision or args.revision,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_length": len(tokenizer),
        "tokenizer_special_ids": {
            "bos": tokenizer.bos_token_id,
            "eos": tokenizer.eos_token_id,
            "pad": tokenizer.pad_token_id,
        },
        "eval_dataset": args.eval_dataset,
        "eval_dataset_source": eval_source,
        "eval_offset": args.eval_offset,
        "eval_offset_derivation": args.eval_offset_derivation,
        "eval_size": args.eval_size,
        "eval_sequence_length": args.eval_seq_len,
        "eval_batch_size": args.eval_batch_size,
        "sample_ids": sample_ids_from_batches(eval_batches, args.eval_offset),
        "batch_hashes": batch_hashes,
        "reference_logits_dtype": "bfloat16",
        "seed": args.seed,
        "trust_remote_code": args.trust_remote_code,
        "runtime": reference_runtime_provenance(args, full_model, hf_config),
    }


def _load_reference_batch(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older supported PyTorch
        return torch.load(path, map_location="cpu")


def validate_reference_manifest(manifest_path: Path, expected_hash: str) -> dict[str, Any]:
    """Validate signature and every immutable batch object in a cache entry."""
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported reference manifest {manifest_path}")
    if manifest.get("signature_hash") != expected_hash:
        raise ValueError(f"Reference signature mismatch in {manifest_path}")
    logits_dir = manifest_path.parent / "reference_logits"
    for record in manifest.get("files", []):
        path = logits_dir / record["name"]
        if not path.is_file() or file_hash(path) != record["sha256"]:
            raise ValueError(f"Missing or corrupt reference batch {path}")
    if len(manifest.get("files", [])) != manifest.get("batch_count"):
        raise ValueError(f"Incomplete reference manifest {manifest_path}")
    return manifest


def _model_logits(model: torch.nn.Module, batch: Mapping[str, torch.Tensor], device: torch.device):
    inputs = {name: tensor.to(device) for name, tensor in batch.items()}
    with torch.no_grad():
        try:
            output = model(**inputs, use_cache=False)
        except TypeError as error:
            if "use_cache" not in str(error):
                raise
            output = model(**inputs)
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple | list) and output:
        return output[0]
    raise TypeError(f"Could not extract logits from {type(output).__name__}")


def capture_references(
    model: torch.nn.Module,
    eval_batches: Sequence[Mapping[str, torch.Tensor]],
    input_device: torch.device,
    signature: Mapping[str, Any],
    output_dir: Path,
    cache_root: Path | None,
    recompute: bool,
) -> tuple[dict[str, Any], Path, bool]:
    """Reuse an exact cache entry or capture BF16 reference logits."""
    signature_hash = canonical_hash(signature)
    if cache_root is None:
        entry_dir = output_dir
        manifest_path = output_dir / "reference_manifest.json"
    else:
        cache_root.mkdir(parents=True, exist_ok=True)
        entry_dir = cache_root / signature_hash
        manifest_path = entry_dir / "manifest.json"

    reused = False
    if manifest_path.is_file():
        try:
            manifest = validate_reference_manifest(manifest_path, signature_hash)
            reused = True
        except (OSError, ValueError, json.JSONDecodeError):
            if not recompute:
                raise
    elif cache_root is not None:
        other_manifests = list(cache_root.glob("*/manifest.json"))
        if other_manifests and not recompute:
            other_hashes = sorted(path.parent.name for path in other_manifests)
            raise ValueError(
                "Reference cache contains entries but none exactly match this run. Pass "
                f"--recompute-reference-cache to create {signature_hash}; existing entries: "
                f"{other_hashes}"
            )

    if not reused:
        logits_dir = entry_dir / "reference_logits"
        files: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(eval_batches):
            started = time.perf_counter()
            logits = (
                _model_logits(model, batch, input_device)
                .detach()
                .to(dtype=torch.bfloat16, device="cpu")
            )
            path = logits_dir / f"batch_{batch_index:06d}.pt"
            atomic_torch_save(
                path,
                {
                    "logits": logits,
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "sample_indices": list(
                        range(
                            batch_index * batch["input_ids"].shape[0],
                            (batch_index + 1) * batch["input_ids"].shape[0],
                        )
                    ),
                },
            )
            files.append(
                {
                    "name": path.name,
                    "sha256": file_hash(path),
                    "bytes": path.stat().st_size,
                    "logits_shape": list(logits.shape),
                    "logits_dtype": "bfloat16",
                    "walltime_seconds": time.perf_counter() - started,
                }
            )
            del logits
        manifest = {
            "schema_version": REFERENCE_SCHEMA_VERSION,
            "signature_hash": signature_hash,
            "signature": signature,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "batch_count": len(files),
            "files": files,
        }
        atomic_write_json(manifest_path, manifest)

    logits_dir = entry_dir / "reference_logits"
    if cache_root is not None:
        pointer = dict(manifest)
        pointer["shared_cache_entry"] = str(entry_dir.resolve())
        atomic_write_json(output_dir / "reference_manifest.json", pointer)
        output_link = output_dir / "reference_logits"
        if os.path.lexists(output_link):
            if not output_link.is_symlink() or output_link.resolve() != logits_dir.resolve():
                raise FileExistsError(
                    f"{output_link} exists and does not point at the exact reference cache entry"
                )
        else:
            os.symlink(logits_dir.resolve(), output_link, target_is_directory=True)
    return manifest, logits_dir, reused


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    """Collect package, platform, CUDA, GPU, and selected scheduler provenance."""
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    selected_environment = {
        key: os.environ[key]
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "SLURM_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "SLURM_JOB_ACCOUNT",
            "SLURM_JOB_PARTITION",
            "HF_HOME",
            "TRANSFORMERS_CACHE",
            "STUDY_CONTAINER_IMAGE",
            "QWEN36_PYDEPS",
        )
        if key in os.environ
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "modelopt": _package_version("nvidia-modelopt"),
            "datasets": _package_version("datasets"),
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "torch_runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": cuda_devices,
        },
        "selected_environment": selected_environment,
    }


def reference_runtime_provenance(
    args: argparse.Namespace, full_model: torch.nn.Module, hf_config: Any
) -> dict[str, Any]:
    """Return stable execution identity fields that can change BF16 reference logits."""
    import transformers

    gpu_types = []
    if torch.cuda.is_available():
        observed = set()
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            identity = (
                properties.name,
                f"{properties.major}.{properties.minor}",
                int(properties.total_memory),
            )
            if identity in observed:
                continue
            observed.add(identity)
            gpu_types.append(
                {
                    "name": identity[0],
                    "compute_capability": identity[1],
                    "total_memory_bytes": identity[2],
                }
            )
    git = collect_git()
    config_payload = hf_config.to_dict() if hasattr(hf_config, "to_dict") else json_safe(hf_config)
    return {
        "study_source_sha256": file_hash(Path(__file__).resolve()),
        "repository_commit": git.get("commit"),
        "packages": {
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "transformers": _package_version("transformers"),
            "modelopt": _package_version("nvidia-modelopt"),
            "datasets": _package_version("datasets"),
        },
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "gpu_types": sorted(gpu_types, key=lambda item: json.dumps(item, sort_keys=True)),
        "device_request": args.device,
        "device_map_request": args.device_map,
        "resolved_hf_device_map": json_safe(getattr(full_model, "hf_device_map", None)),
        "model_config_sha256": canonical_hash(config_payload),
        "container_image": os.environ.get("STUDY_CONTAINER_IMAGE"),
        "dependency_overlay": os.environ.get("QWEN36_PYDEPS"),
        "transformers_module": str(Path(transformers.__file__).resolve()),
        "backend_flags": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": (
                torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None
            ),
            "cudnn_allow_tf32": (
                torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None
            ),
            "cudnn_deterministic": (
                torch.backends.cudnn.deterministic if torch.cuda.is_available() else None
            ),
        },
    }


def collect_git() -> dict[str, Any]:
    """Collect the current git revision and dirty-worktree flag."""

    def command(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(command("status", "--porcelain")),
    }


def _dtype_from_name(name: str) -> torch.dtype | str:
    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model_and_tokenizer(args: argparse.Namespace):
    """Load the checkpoint's declared architecture and identify its language target."""
    # These imports are intentionally deferred: dry-run planning never loads a
    # Transformers model, contacts a hub, or initializes a CUDA context.
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    import modelopt.torch.opt as mto
    from modelopt.torch.export.model_utils import get_language_model_from_vl

    mto.enable_huggingface_checkpointing()
    common = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    config = AutoConfig.from_pretrained(args.model, **common)
    architectures = list(getattr(config, "architectures", []) or [])
    architecture = architectures[0] if architectures else None
    if architecture and hasattr(transformers, architecture):
        model_class = getattr(transformers, architecture)
    else:
        if not args.trust_remote_code and architecture:
            raise ValueError(
                f"Checkpoint declares {architecture}, which transformers {transformers.__version__} "
                "does not expose. Upgrade transformers or explicitly pass --trust-remote-code."
            )
        model_class = AutoModelForCausalLM

    device_map: str | None = args.device_map
    if args.device == "cpu":
        device_map = "cpu"
    elif args.device_map == "none":
        device_map = None
    load_kwargs = {
        **common,
        "device_map": device_map,
        "dtype": _dtype_from_name(args.dtype),
        "low_cpu_mem_usage": True,
    }
    try:
        full_model = model_class.from_pretrained(args.model, **load_kwargs)
    except TypeError as error:
        if "dtype" not in str(error):
            raise
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        full_model = model_class.from_pretrained(args.model, **load_kwargs)
    if device_map is None:
        full_model.to(args.device)
    full_model.eval()

    hf_device_map = getattr(full_model, "hf_device_map", {})
    if args.device != "cpu":
        offloaded = {
            name: device for name, device in hf_device_map.items() if str(device) in {"cpu", "disk"}
        }
        if offloaded:
            raise RuntimeError(
                "This study requires the quantized language model on GPUs; auto mapping offloaded "
                f"modules to CPU/disk: {offloaded}"
            )

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=args.tokenizer_revision or args.revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    lineage = get_language_model_from_vl(full_model)
    language_model = lineage[-1] if lineage else full_model
    embedding_owner = full_model if hasattr(full_model, "get_input_embeddings") else language_model
    embedding = embedding_owner.get_input_embeddings()
    input_device = embedding.weight.device
    if input_device.type == "meta":
        raise RuntimeError("Input embeddings remain on meta device after model loading")
    return full_model, language_model, tokenizer, input_device, config, architecture


def materialize_batches(
    dataset_name: str,
    tokenizer: Any,
    sample_count: int,
    sequence_length: int,
    batch_size: int,
    pack: bool,
    sample_offset: int = 0,
) -> list[dict[str, torch.Tensor]]:
    """Tokenize a deterministic row prefix, select an offset, and fix every shape."""
    from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

    requested_rows = sample_offset + sample_count
    dataloader = get_dataset_dataloader(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=requested_rows,
        max_sample_length=sequence_length,
        device=None,
        include_labels=False,
        pack=pack,
    )
    all_batches = []
    for batch in dataloader:
        observed_batch_size = int(batch["input_ids"].shape[0])
        all_batches.append(
            normalize_batch_shape(
                batch, observed_batch_size, sequence_length, tokenizer.pad_token_id
            )
        )
    observed = sum(batch["input_ids"].shape[0] for batch in all_batches)
    if observed != requested_rows:
        raise ValueError(f"Dataset returned {observed} rows; expected exactly {requested_rows}")
    return select_row_range(all_batches, sample_offset, sample_count, batch_size)


def dataset_source_provenance(dataset_name: str) -> dict[str, Any]:
    """Describe a local dataset snapshot by content, or preserve its remote identifier."""
    path = Path(dataset_name)
    if not path.is_file():
        return {"identifier": dataset_name, "local_snapshot": False}
    digest = file_hash(path)
    staged_digest = os.environ.get("DATASET_SHA256")
    if staged_digest is not None and staged_digest != digest:
        raise ValueError(
            f"Local dataset digest {digest} does not match staged DATASET_SHA256 {staged_digest}"
        )
    with path.open("rb") as stream:
        line_count = sum(1 for _ in stream)
    return {
        "identifier": dataset_name,
        "local_snapshot": True,
        "resolved_path": str(path.resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "source_row_count": line_count,
    }


def select_row_range(
    batches: Sequence[Mapping[str, torch.Tensor]],
    sample_offset: int,
    sample_count: int,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    """Select a deterministic row interval and re-batch it without reordering."""
    if sample_offset < 0 or sample_count <= 0 or batch_size <= 0:
        raise ValueError("Row offset must be non-negative and counts must be positive")
    if sample_count % batch_size:
        raise ValueError("Selected sample count must be divisible by batch size")
    input_ids = torch.cat([batch["input_ids"] for batch in batches])
    attention_mask = torch.cat([batch["attention_mask"] for batch in batches])
    stop = sample_offset + sample_count
    if stop > input_ids.shape[0]:
        raise ValueError(
            f"Requested rows [{sample_offset}, {stop}) from only {input_ids.shape[0]} rows"
        )
    input_ids = input_ids[sample_offset:stop]
    attention_mask = attention_mask[sample_offset:stop]
    return [
        {
            "input_ids": input_ids[start : start + batch_size],
            "attention_mask": attention_mask[start : start + batch_size],
        }
        for start in range(0, sample_count, batch_size)
    ]


def make_forward_loop(
    full_model: torch.nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    input_device: torch.device,
):
    """Create the full-wrapper forward loop expected by ModelOpt APIs."""

    def forward_loop(_quantized_target: torch.nn.Module) -> None:
        for batch in batches:
            _model_logits(full_model, batch, input_device)

    return forward_loop


def quantizer_role(name: str) -> str:
    """Classify singular, custom-weight, and indexed fused-expert quantizer names."""
    match = _INDEXED_QUANTIZER_RE.search(name)
    if match is not None:
        return match.group("role")
    final_component = name.rsplit(".", 1)[-1]
    if final_component == "weight_quantizer" or final_component.endswith("_weight_quantizer"):
        return "weight"
    if final_component == "input_quantizer" or final_component.endswith("_input_quantizer"):
        return "input"
    return "other"


def _weight_binding(
    quantizer_name: str, modules: Mapping[str, torch.nn.Module]
) -> dict[str, Any] | None:
    """Resolve a weight quantizer to its full parameter and exact expert slice."""
    components = quantizer_name.split(".")
    expert_index = None
    if components[-1].isdigit() and len(components) >= 2:
        descriptor = components[-2]
        if not descriptor.endswith(("_weight_quantizers", "_weight_quantizer")):
            return None
        expert_index = int(components[-1])
        owner_name = ".".join(components[:-2])
        suffix = (
            "_weight_quantizers"
            if descriptor.endswith("_weight_quantizers")
            else "_weight_quantizer"
        )
        weight_attribute = descriptor.removesuffix(suffix) or "weight"
    else:
        descriptor = components[-1]
        if descriptor == "weight_quantizer":
            owner_name = ".".join(components[:-1])
            weight_attribute = "weight"
        elif descriptor.endswith("_weight_quantizer"):
            owner_name = ".".join(components[:-1])
            weight_attribute = descriptor.removesuffix("_weight_quantizer")
        else:
            return None

    owner = modules.get(owner_name)
    full_weight = getattr(owner, weight_attribute, None)
    if not isinstance(full_weight, torch.Tensor) or full_weight.ndim < 2:
        return None
    if expert_index is None:
        weight_slice = full_weight
    else:
        if full_weight.ndim < 3 or expert_index >= full_weight.shape[0]:
            return None
        weight_slice = full_weight[expert_index]
    return {
        "owner_name": owner_name,
        "weight_attribute": weight_attribute,
        "expert_index": expert_index,
        "full_weight": full_weight,
        "weight_slice": weight_slice,
    }


def install_dynamic_shape_refresh_hooks(
    target: torch.nn.Module,
) -> tuple[list[Any], list[str]]:
    """Refresh research static-block reshape state before every dynamic invocation."""
    from modelopt.torch.quantization.nn import TensorQuantizer

    handles = []
    names = []

    def refresh_shape_cache(module: TensorQuantizer, _inputs: tuple[Any, ...]) -> None:
        for attribute in _DYNAMIC_BLOCK_SHAPE_CACHE_ATTRIBUTES:
            if hasattr(module, attribute):
                delattr(module, attribute)

    for name, module in target.named_modules():
        block_sizes = getattr(module, "block_sizes", None)
        if not (
            isinstance(module, TensorQuantizer)
            and module.is_enabled
            and module.fake_quant
            and bool(getattr(module, "_dynamic", False))
            and isinstance(block_sizes, Mapping)
            and block_sizes.get("type") != "dynamic"
        ):
            continue
        handles.append(module.register_forward_pre_hook(refresh_shape_cache))
        names.append(name)
    if not names:
        raise RuntimeError("Research dynamic-block recipe installed no shape-refresh hooks")
    return handles, names


def quantizer_inventory(
    target: torch.nn.Module,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Return observed TensorQuantizer metadata and enabled fake-quant names."""
    from modelopt.torch.quantization.nn import TensorQuantizer

    records = []
    eligible: dict[str, list[str]] = {"weight": [], "input": [], "other": []}
    for name, module in target.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        role = quantizer_role(name)
        enabled = bool(module.is_enabled)
        fake_quant = bool(module.fake_quant)
        if enabled and fake_quant:
            eligible[role].append(name)
        amax = getattr(module, "_amax", None)
        record = {
            "name": name,
            "role": role,
            "family": module_family(name),
            "enabled": enabled,
            "fake_quant": fake_quant,
            "num_bits": json_safe(module.num_bits),
            "axis": json_safe(module.axis),
            "block_sizes": json_safe(module.block_sizes),
            "top_level_dynamic": bool(getattr(module, "_dynamic", False)),
            "nested_block_type": (
                module.block_sizes.get("type") if module.block_sizes is not None else None
            ),
            "amax": None,
        }
        if isinstance(amax, torch.Tensor):
            values = amax.detach().float().cpu()
            record["amax"] = {
                "shape": list(values.shape),
                "dtype": str(amax.dtype).removeprefix("torch."),
                "element_count": values.numel(),
                "min": values.min().item() if values.numel() else None,
                "max": values.max().item() if values.numel() else None,
            }
        records.append(record)
    return records, eligible


def estimate_model_weight_cost(target: torch.nn.Module, recipe_id: str) -> dict[str, Any]:
    """Estimate logical payload and scale costs for every enabled weight quantizer."""
    modules = dict(target.named_modules())
    logical_records = []
    unmapped_quantizers = []
    logical_totals = defaultdict(int)
    unique_totals = defaultdict(int)
    seen_parameter_slices: set[tuple[int, int | None]] = set()
    for quantizer_name, quantizer in target.named_modules():
        if quantizer_role(quantizer_name) != "weight" or not getattr(
            quantizer, "is_enabled", False
        ):
            continue
        binding = _weight_binding(quantizer_name, modules)
        if binding is None:
            unmapped_quantizers.append(quantizer_name)
            continue
        weight_slice = binding["weight_slice"]
        cost = estimate_tensor_cost(weight_slice.shape, recipe_id)
        record = {
            "quantizer": quantizer_name,
            "module": binding["owner_name"],
            "weight_attribute": binding["weight_attribute"],
            "expert_index": binding["expert_index"],
            **cost,
        }
        logical_records.append(record)
        for key in (
            "element_count",
            "payload_bits",
            "scale_count",
            "scale_overhead_bits",
            "total_bits",
        ):
            logical_totals[key] += int(cost[key])
        slice_key = (id(binding["full_weight"]), binding["expert_index"])
        if slice_key not in seen_parameter_slices:
            seen_parameter_slices.add(slice_key)
            for key in (
                "element_count",
                "payload_bits",
                "scale_count",
                "scale_overhead_bits",
                "total_bits",
            ):
                unique_totals[key] += int(cost[key])
    for totals in (logical_totals, unique_totals):
        totals["effective_bits_per_weight"] = (
            totals["total_bits"] / totals["element_count"] if totals["element_count"] else None
        )
    return {
        "scope": "logical enabled quantizer call sites; not measured checkpoint or process memory",
        "logical_quantized_modules": logical_records,
        "logical_totals": dict(logical_totals),
        "unique_parameter_slice_totals": dict(unique_totals),
        "unique_parameter_slice_count": len(seen_parameter_slices),
        "unmapped_weight_quantizers": sorted(unmapped_quantizers),
        "assumptions": (
            "FP8 payload plus declared scale storage only. Tied singular parameter aliases are "
            "deduplicated in unique_parameter_slice_totals, while distinct fused-expert slices "
            "retain their per-expert scales. Fake-quant tensors remain in their original dtype; "
            "these values are neither checkpoint-size nor process-memory measurements."
        ),
    }


@torch.no_grad()
def compute_weight_quantization_mse(target: torch.nn.Module) -> dict[str, float]:
    """Measure every weight quantizer directly, including unrouted fused experts."""
    modules = dict(target.named_modules())
    values: dict[str, float] = {}
    for quantizer_name, quantizer in target.named_modules():
        if quantizer_role(quantizer_name) != "weight" or not getattr(
            quantizer, "is_enabled", False
        ):
            continue
        if not getattr(quantizer, "fake_quant", False):
            continue
        binding = _weight_binding(quantizer_name, modules)
        if binding is None:
            raise RuntimeError(f"Could not map enabled weight quantizer {quantizer_name!r}")
        original = binding["weight_slice"].detach()
        quantized = quantizer(original)
        values[quantizer_name] = (original.float() - quantized.float()).square().mean().item()
        del quantized
    return values


def validate_quantization_coverage(
    target: torch.nn.Module,
    model_id: str,
    recipe: Recipe,
    eligible: Mapping[str, Sequence[str]],
    weight_cost: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed when a wrapper/pattern miss would invalidate a format comparison."""
    modules = dict(target.named_modules())
    weights = set(eligible["weight"])
    inputs = set(eligible["input"])
    if not weights:
        raise RuntimeError("Coverage gate: no enabled fake-quant weight quantizers")
    if recipe.activation_quantized and not inputs:
        raise RuntimeError("Coverage gate: W8A8 candidate has no enabled input quantizers")
    if not recipe.activation_quantized and inputs:
        raise RuntimeError(
            "Coverage gate: weight-only control still has enabled input quantizers: "
            f"{sorted(inputs)[:8]}"
        )
    unmapped = list(weight_cost.get("unmapped_weight_quantizers", []))
    if unmapped:
        raise RuntimeError(f"Coverage gate: unmapped weight quantizers: {unmapped}")

    weight_families = {module_family(name) for name in weights}
    input_families = {module_family(name) for name in inputs}
    if not weight_families.intersection({"attention", "linear_attention"}):
        raise RuntimeError(
            f"Coverage gate: no attention-family weight quantizers; saw {sorted(weight_families)}"
        )
    if not weight_families.intersection({"mlp", "experts", "shared_expert"}):
        raise RuntimeError(
            f"Coverage gate: no feed-forward weight quantizers; saw {sorted(weight_families)}"
        )
    model_required_family = "experts" if model_id.endswith("35B-A3B") else "mlp"
    if model_required_family not in weight_families:
        raise RuntimeError(
            f"Coverage gate: {model_id} requires {model_required_family!r} weight coverage; "
            f"saw {sorted(weight_families)}"
        )

    semantic_errors = []
    for role, names in (("weight", weights), ("input", inputs)):
        for name in sorted(names):
            quantizer = modules[name]
            bits = (
                tuple(quantizer.num_bits)
                if isinstance(quantizer.num_bits, tuple)
                else quantizer.num_bits
            )
            blocks = quantizer.block_sizes
            if bits != (4, 3):
                semantic_errors.append(f"{name}: num_bits={bits!r}")
                continue
            if recipe.recipe_id.startswith("per_tensor"):
                valid = blocks is None and not bool(getattr(quantizer, "_dynamic", False))
            elif recipe.recipe_id.startswith("mxfp8"):
                valid = (
                    isinstance(blocks, Mapping)
                    and blocks.get(-1) == 32
                    and blocks.get("type") == "dynamic"
                    and blocks.get("scale_bits") == (8, 0)
                )
            elif role == "weight":
                numeric_blocks = [
                    value for key, value in (blocks or {}).items() if isinstance(key, int)
                ]
                valid = (
                    sorted(numeric_blocks) == [128, 128]
                    and (
                        bool(getattr(quantizer, "_dynamic", False))
                        == recipe.recipe_id.startswith("block128_dynamic")
                    )
                    and (blocks or {}).get("type") != "dynamic"
                )
            else:
                valid = (
                    isinstance(blocks, Mapping)
                    and blocks.get(-1) == 128
                    and bool(getattr(quantizer, "_dynamic", False))
                    and blocks.get("type") != "dynamic"
                )
            if not valid:
                semantic_errors.append(
                    f"{name}: role={role}, dynamic={getattr(quantizer, '_dynamic', None)!r}, "
                    f"block_sizes={blocks!r}"
                )
    if semantic_errors:
        raise RuntimeError(
            "Coverage gate: quantizer semantics do not match the declared recipe: "
            + "; ".join(semantic_errors[:16])
        )

    return {
        "status": "passed",
        "weight_quantizer_names": sorted(weights),
        "input_quantizer_names": sorted(inputs),
        "weight_families": sorted(weight_families),
        "input_families": sorted(input_families),
        "model_required_weight_family": model_required_family,
        "cross_candidate_contract": (
            "The report must require identical enabled weight-quantizer names across all seven "
            "candidates and identical input-quantizer names across the three W8A8 candidates."
        ),
    }


def require_complete_mse_coverage(summary: Mapping[str, Any], role: str) -> None:
    """Reject missing MSE observations instead of treating unexecuted modules as zero error."""
    coverage = summary["coverage"]
    if coverage["eligible_count"] <= 0 or coverage["executed_count"] != coverage["eligible_count"]:
        raise RuntimeError(
            f"{role} MSE coverage incomplete: {coverage['executed_count']}/"
            f"{coverage['eligible_count']}; missing={coverage['missing_quantizers'][:16]}"
        )


def evaluate_quantized_outputs(
    full_model: torch.nn.Module,
    eval_batches: Sequence[Mapping[str, torch.Tensor]],
    input_device: torch.device,
    manifest: Mapping[str, Any],
    logits_dir: Path,
    epsilon: float,
) -> dict[str, Any]:
    """Stream cached references against quantized wrapper logits."""
    accumulator = OutputMetricAccumulator(epsilon=epsilon)
    files = manifest["files"]
    if len(files) != len(eval_batches):
        raise ValueError("Reference batch count does not match materialized evaluation batches")
    for batch, record in zip(eval_batches, files):
        path = logits_dir / record["name"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"Reference batch changed after manifest validation: {path}")
        reference = _load_reference_batch(path)
        if not torch.equal(reference["input_ids"], batch["input_ids"]) or not torch.equal(
            reference["attention_mask"], batch["attention_mask"]
        ):
            raise ValueError(f"Reference inputs do not match current evaluation batch: {path}")
        quantized_logits = _model_logits(full_model, batch, input_device).detach().cpu()
        accumulator.add_batch(
            reference["logits"],
            quantized_logits,
            batch["input_ids"],
            batch["attention_mask"],
            reference["sample_indices"],
        )
        del quantized_logits, reference
    return accumulator.finalize()


def build_parser() -> argparse.ArgumentParser:
    """Build the one-model, one-candidate command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--recipe", required=True, choices=RECIPE_IDS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--calib-dataset", default="cnn_dailymail")
    parser.add_argument("--eval-dataset", default="cnn_dailymail")
    parser.add_argument(
        "--eval-offset",
        type=int,
        help=(
            "Dataset-row offset for evaluation. Defaults to calib-size when calibration and "
            "evaluation use the same dataset, otherwise zero."
        ),
    )
    parser.add_argument("--calib-size", type=int, default=16)
    parser.add_argument("--eval-size", type=int, default=8)
    parser.add_argument(
        "--activation-mse-size",
        type=int,
        default=8,
        help="Packed calibration rows reused for activation-quantizer MSE (W8A8 only).",
    )
    parser.add_argument("--calib-seq-len", type=int, default=128)
    parser.add_argument("--eval-seq-len", type=int, default=128)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--pack-calibration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto", choices=("auto", "sequential", "none"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--recompute-reference-cache", action="store_true")
    parser.add_argument("--metric-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--dry-run-plan", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate sizes and the fixed-shape constraint before any heavyweight work."""
    for name in (
        "calib_size",
        "eval_size",
        "activation_mse_size",
        "calib_seq_len",
        "eval_seq_len",
        "calib_batch_size",
        "eval_batch_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.calib_size % args.calib_batch_size:
        raise ValueError("--calib-size must be divisible by --calib-batch-size")
    if args.eval_size % args.eval_batch_size:
        raise ValueError("--eval-size must be divisible by --eval-batch-size")
    if args.activation_mse_size > args.calib_size:
        raise ValueError("--activation-mse-size cannot exceed --calib-size")
    if args.activation_mse_size % args.calib_batch_size:
        raise ValueError("--activation-mse-size must be divisible by --calib-batch-size")
    if args.eval_offset < 0:
        raise ValueError("--eval-offset must be non-negative")
    if args.recipe in TOP_LEVEL_DYNAMIC_RECIPES and (
        args.calib_batch_size != args.eval_batch_size or args.calib_seq_len != args.eval_seq_len
    ):
        raise ValueError(
            "The research comparison contract requires equal outer calibration/evaluation batch "
            "and padded sequence shapes; routed inner MoE shapes are refreshed per invocation."
        )
    if args.metric_epsilon <= 0:
        raise ValueError("--metric-epsilon must be positive")


def resolve_eval_offset(args: argparse.Namespace) -> tuple[int, str]:
    """Resolve an evaluation row offset that clears calibration's raw-source prefix."""
    if args.eval_offset is not None:
        return args.eval_offset, "explicit_cli_value"
    if args.eval_dataset != args.calib_dataset:
        return 0, "different_dataset_default_zero"
    if args.pack_calibration:
        return args.calib_size * 8, "same_dataset_packed_calibration_8x_raw_sample_multiplier"
    return args.calib_size, "same_dataset_unpacked_calibration_prefix"


def resolved_plan(args: argparse.Namespace, recipe: Recipe) -> dict[str, Any]:
    """Build the complete network/model/GPU-free execution plan."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "plan_only" if args.dry_run_plan else "resolved",
        "one_candidate_per_process": True,
        "model": {
            "id": args.model,
            "revision": args.revision,
            "tokenizer": args.tokenizer or args.model,
            "tokenizer_revision": args.tokenizer_revision or args.revision,
            "dtype": args.dtype,
            "device": args.device,
            "device_map": args.device_map,
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
        },
        "recipe": recipe.plan_metadata(),
        "data": {
            "seed": args.seed,
            "calibration": {
                "dataset": args.calib_dataset,
                "sample_count": args.calib_size,
                "sequence_length": args.calib_seq_len,
                "batch_size": args.calib_batch_size,
                "pack": args.pack_calibration,
            },
            "evaluation": {
                "dataset": args.eval_dataset,
                "row_offset": args.eval_offset,
                "row_offset_derivation": args.eval_offset_derivation,
                "sample_count": args.eval_size,
                "sequence_length": args.eval_seq_len,
                "batch_size": args.eval_batch_size,
                "pack": False,
            },
            "activation_mse": {
                "sample_count": args.activation_mse_size,
                "source": "packed calibration prefix",
                "applies_to": "W8A8 candidates only",
            },
            "fixed_shape_enforced": args.recipe in TOP_LEVEL_DYNAMIC_RECIPES,
        },
        "reference_cache": {
            "root": str(args.reference_cache) if args.reference_cache else None,
            "mismatch_policy": (
                "create_new_exact_entry" if args.recompute_reference_cache else "fail"
            ),
            "storage_dtype": "bfloat16",
            "runtime_key_fields": [
                "model/revision/dtype",
                "tokenizer/revision/class/special-token IDs",
                "eval dataset/offset/size/batch/sequence",
                "exact sample IDs and token batch hashes",
                "seed and trust_remote_code",
                "study/repository source and package versions",
                "model config, container, GPU type, backend flags, and resolved device map",
            ],
        },
        "metrics": {
            "reference_captured_before_in_place_quantization": True,
            "output_similarity": [
                "logit MSE/RMSE/MAE",
                "centered and variance-normalized logit MSE",
                "forward/reverse KL and Jensen-Shannon divergence",
                "target-logprob error and NLL delta",
                "top-1 agreement and top-5 set overlap",
                "per-token and per-sample distributions/quantiles",
            ],
            "quantizer_mse": {
                "weight": "direct ModelOpt TensorQuantizer pass over every mapped weight/expert slice",
                "input": (
                    "ModelOpt compute_quantization_mse over the packed calibration prefix, using "
                    "a role callable that includes indexed fused quantizers"
                ),
            },
            "epsilon": args.metric_epsilon,
        },
        "outputs": {
            "directory": str(args.output_dir),
            "plan": str(args.output_dir / "plan.json"),
            "results": str(args.output_dir / "results.json"),
            "reference_logits": str(args.output_dir / "reference_logits"),
        },
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_study(args: argparse.Namespace, recipe: Recipe, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Execute reference capture, quantization, measurements, and atomic reporting."""
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    started = time.perf_counter()
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "model": args.model,
        "recipe": args.recipe,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan": plan,
        "git": collect_git(),
        "environment": collect_environment(),
        "phase_walltime_seconds": {},
    }
    atomic_write_json(results_path, results)
    current_phase = "initialization"
    phase_started = time.perf_counter()

    def finish_phase(name: str) -> None:
        nonlocal phase_started
        results["phase_walltime_seconds"][name] = time.perf_counter() - phase_started
        atomic_write_json(results_path, results)
        phase_started = time.perf_counter()

    try:
        _set_seed(args.seed)
        full_model, target, tokenizer, input_device, hf_config, architecture = (
            load_model_and_tokenizer(args)
        )
        results["model_provenance"] = {
            "architecture": architecture,
            "config_class": type(hf_config).__name__,
            "model_class": type(full_model).__name__,
            "language_target_class": type(target).__name__,
            "language_target_is_full_model": target is full_model,
            "input_device": str(input_device),
            "model_dtype": str(next(target.parameters()).dtype).removeprefix("torch."),
            "hf_device_map": json_safe(getattr(full_model, "hf_device_map", None)),
        }
        finish_phase(current_phase)

        current_phase = "dataset_materialization"
        calib_batches = materialize_batches(
            args.calib_dataset,
            tokenizer,
            args.calib_size,
            args.calib_seq_len,
            args.calib_batch_size,
            args.pack_calibration,
        )
        eval_batches = materialize_batches(
            args.eval_dataset,
            tokenizer,
            args.eval_size,
            args.eval_seq_len,
            args.eval_batch_size,
            False,
            args.eval_offset,
        )
        calibration_source = dataset_source_provenance(args.calib_dataset)
        evaluation_source = (
            calibration_source
            if args.eval_dataset == args.calib_dataset
            else dataset_source_provenance(args.eval_dataset)
        )
        results["dataset_provenance"] = {
            "calibration": {
                "dataset": args.calib_dataset,
                "source": calibration_source,
                "sample_count": args.calib_size,
                "sample_ids": sample_ids_from_batches(calib_batches),
                "sequence_length": args.calib_seq_len,
                "batch_size": args.calib_batch_size,
                "pack": args.pack_calibration,
            },
            "evaluation": {
                "dataset": args.eval_dataset,
                "source": evaluation_source,
                "row_offset": args.eval_offset,
                "row_offset_derivation": args.eval_offset_derivation,
                "sample_count": args.eval_size,
                "sample_ids": sample_ids_from_batches(eval_batches, args.eval_offset),
                "sequence_length": args.eval_seq_len,
                "batch_size": args.eval_batch_size,
                "pack": False,
            },
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_length": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
        }
        finish_phase(current_phase)

        current_phase = "reference_logits"
        model_dtype = next(target.parameters()).dtype
        signature = build_reference_signature(
            args,
            tokenizer,
            eval_batches,
            model_dtype,
            full_model,
            hf_config,
            evaluation_source,
        )
        manifest, logits_dir, reused = capture_references(
            full_model,
            eval_batches,
            input_device,
            signature,
            output_dir,
            args.reference_cache.resolve() if args.reference_cache else None,
            args.recompute_reference_cache,
        )
        results["reference"] = {
            "signature_hash": manifest["signature_hash"],
            "cache_reused": reused,
            "manifest": str(output_dir / "reference_manifest.json"),
            "logits_directory": str(logits_dir),
        }
        finish_phase(current_phase)

        current_phase = "quantization"
        import modelopt.torch.quantization as mtq

        calibration_loop = make_forward_loop(full_model, calib_batches, input_device)
        mtq.quantize(
            target,
            copy.deepcopy(recipe.config),
            forward_loop=calibration_loop if recipe.activation_quantized else None,
        )
        shape_refresh_handles: list[Any] = []
        shape_refresh_names: list[str] = []
        if args.recipe in TOP_LEVEL_DYNAMIC_RECIPES:
            shape_refresh_handles, shape_refresh_names = install_dynamic_shape_refresh_hooks(target)
        quantizers, eligible = quantizer_inventory(target)
        weight_cost = estimate_model_weight_cost(target, args.recipe)
        coverage_contract = validate_quantization_coverage(
            target,
            args.model,
            recipe,
            eligible,
            weight_cost,
        )
        results["quantization"] = {
            "recipe": recipe.plan_metadata(),
            "quantizer_inventory": quantizers,
            "eligible_counts": {name: len(values) for name, values in eligible.items()},
            "coverage_contract": coverage_contract,
            "dynamic_shape_refresh": {
                "enabled": bool(shape_refresh_names),
                "quantizer_count": len(shape_refresh_names),
                "quantizer_names": shape_refresh_names,
                "cleared_attributes_per_invocation": list(_DYNAMIC_BLOCK_SHAPE_CACHE_ATTRIBUTES),
                "hook_handles_kept_alive": len(shape_refresh_handles),
            },
            "weight_cost_estimate": weight_cost,
        }
        finish_phase(current_phase)

        current_phase = "quantizer_mse"
        weight_mse = compute_weight_quantization_mse(target)
        input_mse: dict[str, float] = {}
        if recipe.activation_quantized:
            activation_mse_batches = select_row_range(
                calib_batches,
                sample_offset=0,
                sample_count=args.activation_mse_size,
                batch_size=args.calib_batch_size,
            )
            activation_mse_loop = make_forward_loop(
                full_model,
                activation_mse_batches,
                input_device,
            )
            input_mse = mtq.compute_quantization_mse(
                target,
                activation_mse_loop,
                wildcards=lambda name: quantizer_role(name) == "input",
            )
        weight_mse_summary = summarize_named_mse(weight_mse, eligible["weight"])
        input_mse_summary = summarize_named_mse(input_mse, eligible["input"])
        require_complete_mse_coverage(weight_mse_summary, "weight")
        if recipe.activation_quantized:
            require_complete_mse_coverage(input_mse_summary, "input")
        results["quantization_mse"] = {
            "method": {
                "weight": "direct one-pass fake quantization of every mapped weight/expert slice",
                "input": (
                    "forward-hook MSE over the packed calibration prefix"
                    if recipe.activation_quantized
                    else "not applicable to weight-only control"
                ),
                "activation_sample_count": (
                    args.activation_mse_size if recipe.activation_quantized else 0
                ),
            },
            "weight": weight_mse_summary,
            "input": input_mse_summary,
        }
        finish_phase(current_phase)

        current_phase = "output_similarity"
        results["output_similarity"] = evaluate_quantized_outputs(
            full_model,
            eval_batches,
            input_device,
            manifest,
            logits_dir,
            args.metric_epsilon,
        )
        finish_phase(current_phase)

        results["status"] = "complete"
        results["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        results["total_walltime_seconds"] = time.perf_counter() - started
        atomic_write_json(results_path, results)
        return results
    except BaseException as error:
        results["status"] = "failed"
        results["failed_phase"] = current_phase
        results["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        results["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        results["total_walltime_seconds"] = time.perf_counter() - started
        atomic_write_json(results_path, results)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.eval_offset, args.eval_offset_derivation = resolve_eval_offset(args)
    try:
        validate_args(args)
        recipe = resolve_recipe(args.recipe)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = resolved_plan(args, recipe)
    atomic_write_json(args.output_dir / "plan.json", plan)
    if args.dry_run_plan:
        print(json.dumps(json_safe(plan), indent=2, sort_keys=True, allow_nan=False))
        return 0
    run_study(args, recipe, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
