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
    MIN_FINAL_ACC:   Final accuracy must be above this value (any key containing 'acc').
                     A threshold whose metric is absent fails the check rather than
                     passing unevaluated.
    MAX_FINAL_PERPLEXITY: Final perplexity must be below this value

An empty variable counts as unset, and one that isn't a finite number fails the check.
Once any threshold is set, a missing or unreadable trainer_state.json, or one without
training logs, also fails the check, and so does a NaN or infinite value for a metric
that has a threshold.

Usage:
    python check_regression.py /path/to/output_dir [--json verdict.json]

    Or from a shell script:
    python common/check_regression.py ${OUTPUT_DIR}

    --json also writes the verdict as {"pass", "failure_class", "trainer_state",
    "metrics", "thresholds", "messages"}, so a pipeline gate can read it without
    parsing logs. failure_class is null, "TRAIN_REGRESSION" or "METRICS_MISSING". It is
    strict JSON: a non-finite metric is written as null.

Exit codes:
    0 — all checks pass (or no thresholds set)
    1 — regression detected, or thresholds set but no metrics to check
"""

import argparse
import json
import math
import os
import sys
from glob import glob
from typing import Any

THRESHOLD_VARS = ("MAX_FINAL_LOSS", "MIN_FINAL_ACC", "MAX_FINAL_PERPLEXITY")


def _checkpoint_step(trainer_state_path):
    """Step number of <output_dir>/checkpoint-<step>/trainer_state.json, or -1."""
    name = os.path.basename(os.path.dirname(trainer_state_path))
    step = name.removeprefix("checkpoint-")
    return int(step) if step.isdigit() else -1


def find_trainer_state(output_dir):
    """Find the latest trainer_state.json in the output directory."""
    checkpoint_states = sorted(
        glob(os.path.join(output_dir, "checkpoint-*", "trainer_state.json")),
        key=_checkpoint_step,
    )
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


def check_regression(metrics, thresholds):
    """Check metrics against {env_var: threshold}. Returns (passed, messages)."""
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
        if env_var not in thresholds:
            continue
        thresh = thresholds[env_var]
        val = metrics.get(metric_key)
        if val is None:
            # Setting a threshold asserts the run must be checked against it. If the
            # metric never reaches trainer_state.json the assertion cannot be satisfied,
            # so fail rather than stamping PASS on an unchecked run.
            messages.append(
                f"REGRESSION: {env_var}={thresh} was set but '{metric_key}' is absent from "
                "trainer_state.json, so the threshold was never evaluated"
            )
            passed = False
            continue
        if not math.isfinite(val):
            messages.append(f"REGRESSION: {metric_key} is {val}, not a finite number")
            passed = False
            continue
        if check_fn(val, thresh):
            messages.append(f"PASS: {metric_key}={val:.3f} (threshold: {env_var}={thresh})")
        else:
            messages.append(f"REGRESSION: {fail_msg.format(val=val, thresh=thresh)}")
            passed = False

    return passed, messages


def parse_thresholds(environ):
    """Returns ({env_var: value}, [error]) for the set threshold variables."""
    thresholds, errors = {}, []
    for v in THRESHOLD_VARS:
        raw = environ.get(v)
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            value = math.nan
        if math.isfinite(value):
            thresholds[v] = value
        else:
            errors.append(f"REGRESSION: {v}={raw!r} is not a finite number")
    return thresholds, errors


def evaluate(output_dir):
    """Run the check on output_dir. Returns the verdict that --json writes."""
    thresholds, errors = parse_thresholds(os.environ)
    verdict: dict[str, Any] = {
        "pass": True,
        "failure_class": None,
        "trainer_state": None,
        "metrics": {},
        "thresholds": thresholds,
        "messages": [],
    }
    if errors:
        # The recipe's thresholds are unusable, so the run cannot be checked against them.
        verdict.update({"pass": False, "failure_class": "METRICS_MISSING", "messages": errors})
        return verdict
    if not thresholds:
        return verdict

    trainer_state = find_trainer_state(output_dir)
    verdict["trainer_state"] = trainer_state
    where = trainer_state or f"{output_dir} (no trainer_state.json)"
    try:
        metrics = get_final_metrics(trainer_state) if trainer_state else {}
        missing = f"no training logs to check in {where}"
    # TypeError/AttributeError: valid JSON of the wrong shape, e.g. "loss": null or a list.
    except (OSError, ValueError, TypeError, AttributeError) as e:
        metrics, missing = {}, f"cannot read {where}: {e}"
    if not metrics:
        # Thresholds were set, so an unreadable run must not pass as unchecked.
        verdict.update(
            {
                "pass": False,
                "failure_class": "METRICS_MISSING",
                "messages": [f"REGRESSION: {missing}"],
            }
        )
        return verdict

    passed, messages = check_regression(metrics, thresholds)
    verdict.update({"pass": passed, "metrics": metrics, "messages": messages})
    if not passed:
        verdict["failure_class"] = "TRAIN_REGRESSION"
    return verdict


def main(argv=None):
    """Entry point for regression check CLI."""
    parser = argparse.ArgumentParser(description="Regression check for training jobs.")
    parser.add_argument("output_dir", nargs="?", help="Training output directory")
    parser.add_argument("--json", dest="json_path", help="Create a JSON file storing the verdict")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        parser.print_usage()
        return 0

    verdict = evaluate(args.output_dir)
    if verdict["thresholds"] or verdict["messages"]:
        print(f"=== Regression Check ({verdict['trainer_state'] or args.output_dir}) ===")
        if verdict["metrics"]:
            print(f"Final metrics: {verdict['metrics']}")
        for msg in verdict["messages"]:
            print(f"  {msg}")
        print(f"Regression check {'PASSED' if verdict['pass'] else 'FAILED'}")

    if args.json_path:
        # Strict JSON, serialized before the file is opened so a failure cannot truncate it.
        metrics = {k: v if math.isfinite(v) else None for k, v in verdict["metrics"].items()}
        text = json.dumps({**verdict, "metrics": metrics}, indent=2, allow_nan=False)
        with open(args.json_path, "w") as f:
            f.write(text)
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
