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

"""Spec-dec serving canary gate: target + drafter load, generate, and speculate.

Reads the SMOKE_JSON written by tools/launcher/common/specdec/vllm_smoke_test.sh and
classifies the failure from its raw values: the script's own ``pass`` cannot tell a serving
failure from an acceptance-length regression. A failing ``pass`` the values do not explain
still fails the gate. Pure decision logic in ``evaluate_smoke``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

# Mean acceptance length counts the bonus token, so exactly 1.0 means no draft token was
# ever accepted. Anything at or below this is a drafter that is wired up but useless, or
# not wired up at all; generic, not a performance bar.
SPEC_ON_MIN_AL = 1.05

_RANKING = (
    "CHECKPOINT_NOT_SERVABLE",
    "SPEC_CONFIG_MISMATCH",
    "SPEC_NOT_ACTIVE",
    "ACCEPTANCE_REGRESSION",
)


def evaluate_smoke(smoke, draft=None, target=None):
    """Validate a smoke-test result.

    Args:
        smoke: dict from SMOKE_JSON.
        draft: expected drafter path or HF id, as the smoke job saw it; optional.
        target: expected target model path or HF id, as the smoke job saw it; optional.

    Returns:
        dict ``{pass, failure_class, detail, notes, mean_acceptance_length}``.
    """
    if "stage" not in smoke:
        # Written by a vllm_smoke_test.sh that predates "stage": it cannot say how far the
        # run got, so re-run the smoke test rather than guess.
        return {
            "pass": False,
            "failure_class": "USER_CONFIG_ERROR",
            "detail": "SMOKE_JSON has no 'stage'; re-run the smoke test with the current script",
            "notes": [],
            "mean_acceptance_length": smoke.get("mean_acceptance_length"),
        }
    stage = smoke["stage"]
    if stage == "invalid_config":
        # The script rejected a recipe value before serving anything: nothing about the
        # drafter is known, and re-running unchanged fails the same way.
        return {
            "pass": False,
            "failure_class": "USER_CONFIG_ERROR",
            "detail": f"smoke task config: {smoke.get('reason')}; fix it in the recipe and "
            "re-run the smoke task",
            "notes": [],
            "mean_acceptance_length": smoke.get("mean_acceptance_length"),
        }
    problems = []
    passed, failed = smoke.get("prompts_passed"), smoke.get("prompts_failed")
    if stage != "completed":
        problems.append(("CHECKPOINT_NOT_SERVABLE", f"stopped at {stage}: {smoke.get('reason')}"))
    elif failed or not passed:
        problems.append(("CHECKPOINT_NOT_SERVABLE", f"{passed} prompts passed, {failed} failed"))
    for label, expected in (("draft", draft), ("target", target)):
        if expected is not None and smoke.get(label) != expected:
            problems.append(
                ("SPEC_CONFIG_MISMATCH", f"{label} {smoke.get(label)!r} != expected {expected!r}")
            )
    al = smoke.get("mean_acceptance_length")
    valid = isinstance(al, (int, float)) and not isinstance(al, bool) and math.isfinite(al)
    if not valid:
        problems.append(("SPEC_NOT_ACTIVE", f"no acceptance length reported ({al!r})"))
    elif al <= SPEC_ON_MIN_AL:
        problems.append(("SPEC_NOT_ACTIVE", f"acceptance length {al} <= {SPEC_ON_MIN_AL}"))
    min_al = smoke.get("min_acceptance_length")
    if valid and min_al is not None and al < min_al:
        problems.append(
            ("ACCEPTANCE_REGRESSION", f"acceptance length {al} < recipe floor {min_al}")
        )
    if not problems and smoke.get("pass") is not True:
        # Every failure the script knows is re-derived above; one this gate does not
        # recognize must still not pass a run whose launcher task failed.
        problems.append(
            ("CHECKPOINT_NOT_SERVABLE", f"smoke test failed: {smoke.get('reason') or 'no reason'}")
        )

    notes = []
    if min_al is None:
        notes.append("recipe set no MIN_ACCEPTANCE_LENGTH; only the generic floor applied")
    result = {"mean_acceptance_length": al, "notes": notes}
    if not problems:
        return {"pass": True, "failure_class": None, "detail": "servable, speculating", **result}
    fc = min((c for c, _ in problems), key=_RANKING.index)
    return {
        "pass": False,
        "failure_class": fc,
        "detail": "; ".join(f"{c}: {m}" for c, m in problems),
        **result,
    }


def main(argv=None):
    """CLI entry point: read SMOKE_JSON and print the verdict."""
    p = argparse.ArgumentParser(description="Spec-dec serving canary gate")
    p.add_argument("--smoke", required=True, help="SMOKE_JSON from vllm_smoke_test.sh")
    p.add_argument("--draft", help="expected drafter path or HF id")
    p.add_argument("--target", help="expected target model path or HF id")
    args = p.parse_args(argv)

    try:
        with open(args.smoke) as f:
            smoke = json.load(f)
        if not isinstance(smoke, dict):
            raise ValueError(f"{args.smoke} does not hold a JSON object")
    except (OSError, ValueError) as e:
        error = {"pass": False, "failure_class": "USER_CONFIG_ERROR", "detail": str(e)}
        print(json.dumps({**error, "notes": [], "inputs": {"smoke": args.smoke}}, indent=2))
        return 2

    result = evaluate_smoke(smoke, args.draft, args.target)
    result["inputs"] = {"smoke": args.smoke}
    print(json.dumps(result, indent=2))
    if result["failure_class"] == "USER_CONFIG_ERROR":
        return 2
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
