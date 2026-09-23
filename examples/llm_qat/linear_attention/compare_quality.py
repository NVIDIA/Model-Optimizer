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

"""Compare matched numerical-policy studies using paired block NLL differences."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path

_MATCHED_FIELDS = (
    "model",
    "model_revision",
    "dataset",
    "dataset_config",
    "dataset_revision",
    "eval_split",
    "train_file_sha256",
    "eval_file_sha256",
    "train_tokens_sha256",
    "eval_tokens_sha256",
    "sequence_length",
    "prefill_tokens",
    "loss_scope",
    "seed",
    "train_order",
    "training",
    "trainable_parameters",
    "kda_layers",
    "learning_rate",
    "weight_decay",
    "gradient_clip",
    "train_predicted_tokens",
    "source_sha256",
    "packages",
    "torch",
    "gpu",
)


def compare(control, candidate, *, margin=0.02, samples=5000, seed=2026):
    """Reject unmatched inputs and report descriptive paired block-bootstrap bounds."""
    if not math.isfinite(margin) or samples < 2:
        raise ValueError("margin must be finite and samples must be at least two")
    mismatches = [key for key in _MATCHED_FIELDS if control[key] != candidate[key]]
    if mismatches:
        raise ValueError(f"Unmatched study fields: {mismatches}")
    result = {}
    for phase in ("before", "after"):
        left, right = control[phase], candidate[phase]
        a, b = left["block_nll"], right["block_nll"]
        if len(a) != len(b) or len(a) < 2 or left["predicted_tokens"] != right["predicted_tokens"]:
            raise ValueError("At least two matched evaluation blocks are required")
        if not all(math.isfinite(x) for x in (*a, *b)):
            raise ValueError("NLL values must be finite")
        delta = [y - x for x, y in zip(a, b)]
        rng = random.Random(seed)
        boot = sorted(statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(samples))
        lower, upper = boot[int(0.025 * (samples - 1))], boot[int(0.975 * (samples - 1))]
        mean = statistics.mean(delta)
        result[phase] = {
            "mean_nll_delta": mean,
            "perplexity_ratio": math.exp(mean),
            "block_nll_delta": delta,
            "bootstrap_interval_95": [lower, upper],
            "margin": margin,
            "upper_bound_within_margin": upper <= margin,
            "blocks": len(delta),
            "predicted_tokens": left["predicted_tokens"],
        }
    return {
        "method": "paired block percentile bootstrap; descriptive pilot interval",
        "samples": samples,
        "seed": seed,
        "comparison": result,
    }


def main():
    """Write a reproducible matched comparison receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.02)
    options = parser.parse_args()
    result = compare(
        json.loads(options.control.read_text()),
        json.loads(options.candidate.read_text()),
        margin=options.margin,
    )
    result["control"] = str(options.control)
    result["candidate"] = str(options.candidate)
    options.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
