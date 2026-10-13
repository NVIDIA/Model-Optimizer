# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Regression check for training jobs.

Reads trainer_state.json from a HuggingFace Trainer checkpoint and validates
that final metrics meet specified thresholds. Used by training scripts to
catch regressions in CI.

Environment variables (all optional — no check if unset):
    MAX_FINAL_LOSS:  Final loss must be below this value
    MIN_FINAL_ACC:   Final accuracy must be above this value (any key containing 'acc')
    MAX_FINAL_PERPLEXITY: Final perplexity must be below this value

Usage:
    python check_regression.py /path/to/output_dir

    Or from a shell script:
    python common/check_regression.py ${OUTPUT_DIR}

Exit codes:
    0 — all checks pass (or no thresholds set)
    1 — regression detected
"""

import json
import os
import sys
from glob import glob


def find_trainer_state(output_dir):
    """Find the latest trainer_state.json in the output directory."""
    # Check checkpoint subdirs first (sorted by step number)
    checkpoint_states = sorted(glob(os.path.join(output_dir, "checkpoint-*", "trainer_state.json")))
    if checkpoint_states:
        return checkpoint_states[-1]
    # Fall back to output_dir itself
    direct = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(direct):
        return direct
    return None


def get_final_metrics(trainer_state_path):
    """Extract final loss and accuracy from trainer_state.json."""
    with open(trainer_state_path) as f:
        state = json.load(f)

    logs = [h for h in state.get("log_history", []) if "loss" in h]
    if not logs:
        return {}

    last = logs[-1]
    metrics = {"loss": float(last["loss"])}

    # Find any accuracy key (train_acc/parallel_0_step_0, eval_accuracy, etc.)
    for key, value in last.items():
        if "acc" in key.lower():
            metrics["accuracy"] = float(value)
            break

    # Perplexity if available
    if "perplexity" in last:
        metrics["perplexity"] = float(last["perplexity"])

    return metrics


def check_regression(metrics):
    """Check metrics against environment variable thresholds. Returns (passed, messages)."""
    checks = [
        (
            "MAX_FINAL_LOSS",
            "loss",
            lambda val, thresh: val <= thresh,
            "loss {val:.3f} > threshold {thresh}",
        ),
        (
            "MIN_FINAL_ACC",
            "accuracy",
            lambda val, thresh: val >= thresh,
            "acc {val:.3f} < threshold {thresh}",
        ),
        (
            "MAX_FINAL_PERPLEXITY",
            "perplexity",
            lambda val, thresh: val <= thresh,
            "perplexity {val:.3f} > threshold {thresh}",
        ),
    ]

    passed = True
    messages = []

    for env_var, metric_key, check_fn, fail_msg in checks:
        thresh_str = os.environ.get(env_var)
        if thresh_str is None:
            continue
        thresh = float(thresh_str)
        val = metrics.get(metric_key)
        if val is None:
            messages.append(f"WARNING: {env_var} set but '{metric_key}' not found in metrics")
            continue
        if check_fn(val, thresh):
            messages.append(f"PASS: {metric_key}={val:.3f} (threshold: {env_var}={thresh})")
        else:
            messages.append(f"REGRESSION: {fail_msg.format(val=val, thresh=thresh)}")
            passed = False

    return passed, messages


def main():
    """Entry point for regression check CLI."""
    if len(sys.argv) < 2:
        print("Usage: python check_regression.py <output_dir>")
        sys.exit(0)

    output_dir = sys.argv[1]

    # Skip if no thresholds set
    if not any(
        os.environ.get(v) for v in ["MAX_FINAL_LOSS", "MIN_FINAL_ACC", "MAX_FINAL_PERPLEXITY"]
    ):
        return

    trainer_state = find_trainer_state(output_dir)
    if not trainer_state:
        print(f"WARNING: No trainer_state.json found in {output_dir}, skipping regression check")
        return

    print(f"=== Regression Check ({trainer_state}) ===")
    metrics = get_final_metrics(trainer_state)
    if not metrics:
        print("No training logs found in trainer_state.json")
        return

    print(f"Final metrics: {metrics}")
    passed, messages = check_regression(metrics)
    for msg in messages:
        print(f"  {msg}")

    if not passed:
        sys.exit(1)
    print("Regression check PASSED")


if __name__ == "__main__":
    main()
