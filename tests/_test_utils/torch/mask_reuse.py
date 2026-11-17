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

"""Independent exhaustive checks for the PR-stack review's wrong-result cases."""

import itertools
import json
import math
import random
from hashlib import sha256

from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    create_checkpoint_manifest,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_contracts import (
    required_skipped_tiles,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_solver import (
    MaskReuseSolverConfig,
)
from modelopt.torch.sparsity.attention_sparsity.calibration.mask_reuse_topology import (
    calibrate_mask_reuse_topology,
)

FIT = {
    "formula": "a * exp(b * target_sparsity)",
    "prefill": {"a": 1.0, "b": 1.0, "min_observed_sparsity": 0.4, "max_observed_sparsity": 0.8},
}


def captures(
    checkpoint,
    *,
    seed=0,
    scale=1.0,
    layers=3,
    heads=2,
    buckets=((129, 512, 256),),
    dense=False,
    span=None,
):
    rng = random.Random(seed)
    span = layers - 1 if span is None else span
    rows = []
    for low, high, length in buckets:
        eligible = (length // 128) * (length // 128 + 1) // 2
        retained = [
            [eligible if dense else rng.randrange(1, eligible + 1) for _ in range(heads)]
            for _ in range(layers)
        ]
        anchor_mass = [
            [0.0 if dense else scale * rng.randrange(1, 30) / 1000 for _ in range(heads)]
            for _ in range(layers)
        ]
        risks = {
            str(c): {
                str(a): {
                    "dropped_mass": [
                        [
                            0.0 if dense else scale * rng.randrange(1, 40) / 1000
                            for _ in range(heads)
                        ]
                        for _ in range(heads)
                    ]
                }
                for a in range(max(0, c - span), c)
            }
            for c in range(1, layers)
        }
        for split in ("calibration", "heldout"):
            prompt = f"{split}-{length}"
            log2 = 0.5 * math.log2(math.e) - math.log2(length)
            geometry = {"q_tokens": length, "kv_tokens": length, "q_start_tokens": 0}
            rows.append(
                {
                    "topology_discovery_capture_schema_version": 1,
                    "invocation": {
                        "capture_schema_version": 2,
                        "model": "test-model",
                        "checkpoint_manifest_sha256": checkpoint.sha256,
                        "split": split,
                        "partition": "development" if split == "calibration" else "outer_test",
                        "inner_fold": 0 if split == "calibration" else None,
                        "prompt_id": prompt,
                        "source": "synthetic",
                        "source_group_sha256": sha256(prompt.encode()).hexdigest(),
                        "source_capture_sha256": sha256(prompt.encode()).hexdigest(),
                        "min_kv_tokens": low,
                        "max_kv_tokens": high,
                        "sample_length": length,
                        "target_sparsity_hex": (0.5).hex(),
                        "threshold_log2_hex": log2.hex(),
                        "threshold_lambda_hex": (2.0**log2).hex(),
                        "expected_geometry": geometry,
                    },
                    "geometry": geometry,
                    "global_num_heads": heads,
                    "eligible_tiles": eligible,
                    "attention_layers": list(range(layers)),
                    "max_reuse_span": span,
                    "anchor_stats_by_layer": {
                        str(a): {"retained_tiles": retained[a], "dropped_mass": anchor_mass[a]}
                        for a in range(layers)
                    },
                    "consumer_candidates_by_layer": risks,
                }
            )
    return rows


def checkpoint(tmp_path, layers=3, heads=2):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"num_hidden_layers": layers, "num_attention_heads": heads})
    )
    (root / "model.safetensors").write_bytes(b"synthetic")
    return create_checkpoint_manifest(root, model="test-model")


def solve(tmp_path, cp, rows, rho, backend):
    path = tmp_path / "captures.jsonl"
    ordered = sorted(
        rows,
        key=lambda r: (
            r["invocation"]["min_kv_tokens"],
            r["invocation"]["split"],
            r["invocation"]["prompt_id"],
            float.fromhex(r["invocation"]["target_sparsity_hex"]),
        ),
    )
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered))
    return calibrate_mask_reuse_topology(
        path,
        vanilla_calibration=FIT,
        checkpoint_manifest=cp,
        evidence={
            "topology_discovery_capture_sha256": sha256(path.read_bytes()).hexdigest(),
            "vanilla_fit_sha256": "a" * 64,
            "prompt_plan_sha256": "b" * 64,
        },
        target_bmm1_skip_ratio=rho,
        reuse_dropped_mass_report_threshold=0.02,
        solver_config=MaskReuseSolverConfig(backend=backend, time_limit_seconds=90),
    )


def brute(observations, rho):
    """Enumerate every topology/map, using all supplied calibration prompts."""
    rows = observations if isinstance(observations, list) else [observations]
    layers, heads = len(rows[0]["attention_layers"]), rows[0]["global_num_heads"]
    denominator = layers * heads
    eligible_sum = sum(r["eligible_tiles"] for r in rows)
    best = None
    for flags in itertools.product((False, True), repeat=layers - 1):
        anchors = (0, *(i + 1 for i, flag in enumerate(flags) if flag))
        consumers = [
            (layer, h) for layer in range(layers) if layer not in anchors for h in range(heads)
        ]
        nearest = {layer: max(a for a in anchors if a <= layer) for layer in range(layers)}
        for donors in itertools.product(range(-1, heads), repeat=len(consumers)):
            skipped, cost = 0, 0
            prompt_risks = []
            for row in rows:
                eligible = row["eligible_tiles"]
                risk = []
                for a in anchors:
                    stats = row["anchor_stats_by_layer"][str(a)]
                    cost += heads * eligible + sum(stats["retained_tiles"])
                    risk.extend(stats["dropped_mass"])
                for (layer, head), donor in zip(consumers, donors):
                    if donor == -1:
                        cost += 2 * eligible
                    else:
                        anchor = nearest[layer]
                        kept = row["anchor_stats_by_layer"][str(anchor)]["retained_tiles"][donor]
                        skipped += eligible - kept
                        cost += 2 * kept
                        risk.append(
                            row["consumer_candidates_by_layer"][str(layer)][str(anchor)][
                                "dropped_mass"
                            ][head][donor]
                        )
                prompt_risks.append(math.fsum(risk) / denominator)
            if skipped < required_skipped_tiles(rho, denominator * eligible_sum):
                continue
            key = (
                round(max(prompt_risks), 14),
                cost,
                sum(d >= 0 for d in donors),
                len(anchors),
                anchors,
                tuple(d + 1 for d in donors),
            )
            if best is None or key < best[0]:
                best = key, consumers, donors
    return best
