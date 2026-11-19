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

"""Replay discrete fixed-topology assignments against the original calibration measurements."""

import itertools
import json
import math
from fractions import Fraction
from hashlib import sha256

from .mask_reuse import _Selection, _sha256
from .mask_reuse_contracts import required_skipped_tiles

__all__ = ["assess_mask_reuse_benchmark_gates"]


def _selected_binary_keys(variables):
    keys = []
    for key, variable in variables.items():
        value = variable.varValue
        if value is None or not math.isfinite(value) or min(abs(value), abs(value - 1)) > 1e-6:
            return None
        if value > 0.5:
            keys.append(key)
    return keys


class _BenchmarkIncumbent:
    """Keep the best work-feasible assignment seen for one range and target."""

    def __init__(self, menus, anchors, target, eligible, required, heads, include_anchor):
        self.keys = tuple(menus)
        self.menus = tuple(menus.values())
        self.anchor_mass = tuple(sum(map(Fraction, values)) for values in anchors)
        self.anchor_max = max(map(max, anchors))
        self.target = target
        self.eligible = eligible
        self.required = required
        self.heads = heads
        self.include_anchor = include_anchor
        self.best = None
        self.rank = None
        self.proved = False

    def consider(self, assignment):
        skipped = sum(option.bmm1_skipped_tiles for option in assignment)
        if skipped < self.required:
            return
        retained = sum(option.choice.retained_tiles for option in assignment)
        donors = tuple(0 if o.choice.fallback else o.choice.donor_head + 1 for o in assignment)
        if self.rank is not None and donors == self.rank[-1]:
            return
        reuse = tuple(
            sum(Fraction(option.risk_by_prompt[i]) for option in assignment)
            for i in range(len(self.anchor_mass))
        )
        total = tuple(x + y for x, y in zip(self.anchor_mass, reuse))
        risk = max(total if self.include_anchor else reuse)
        rank = (risk, retained, sum(bool(donor) for donor in donors), donors)
        if self.rank is not None and rank >= self.rank:
            return
        self.rank = rank
        individual = max((max(option.risk_by_prompt) for option in assignment), default=0.0)
        self.best = _Selection(
            self.target,
            dict(zip(self.keys, (o.choice for o in assignment), strict=True)),
            (),
            bmm1_eligible_tiles=self.eligible,
            bmm1_skipped_tiles=skipped,
            target_bmm1_skip_ratio_met=True,
            risk_objective=risk,
            worst_prompt_reuse_dropped_mass=float(max(reuse) / self.heads),
            mean_prompt_reuse_dropped_mass=float(sum(reuse) / (len(reuse) * self.heads)),
            worst_individual_reuse_dropped_mass=individual,
            worst_prompt_total_dropped_mass=float(max(total) / self.heads),
            mean_prompt_total_dropped_mass=float(sum(total) / (len(total) * self.heads)),
            worst_individual_dropped_mass=max(self.anchor_max, individual),
        )

    def initialize(self):
        # Small menus can be certified directly, including every canonical tie.
        if math.prod(len(menu) for menu in self.menus) <= 2048:
            for combination in itertools.product(*self.menus):
                self.consider(combination)
            self.proved = True
            return
        # Head choices are independent at fixed topology. This assignment attains
        # the largest possible integer work saving and witnesses feasibility.
        assignment = [
            min(
                menu,
                key=lambda o: (
                    -o.bmm1_skipped_tiles,
                    max(o.risk_by_prompt),
                    0 if o.choice.fallback else o.choice.donor_head + 1,
                ),
            )
            for menu in self.menus
        ]
        spare = sum(o.bmm1_skipped_tiles for o in assignment) - self.required
        if spare < 0:
            self.proved = True  # Maximum possible work still misses the target.
            return
        self.consider(assignment)
        for i in sorted(range(len(assignment)), key=lambda i: -max(assignment[i].risk_by_prompt)):
            old = assignment[i]
            affordable = (
                o for o in self.menus[i] if old.bmm1_skipped_tiles - o.bmm1_skipped_tiles <= spare
            )
            new = min(
                affordable,
                key=lambda o: (
                    max(o.risk_by_prompt),
                    sum(o.risk_by_prompt),
                    -o.bmm1_skipped_tiles,
                    not o.choice.fallback,
                    o.choice.donor_head,
                ),
            )
            spare -= old.bmm1_skipped_tiles - new.bmm1_skipped_tiles
            assignment[i] = new
        self.consider(assignment)


def assess_mask_reuse_benchmark_gates(
    policy_bytes: bytes,
    *,
    expected_policy_sha256: str,
    search_scope: str,
    required_benchmarks,
    matched_scores,
):
    """Check the selection proof and paired accuracy evidence of one pinned candidate.

    Scores are percentages. The caller supplies the preregistered benchmark names and
    trusted paired evaluation receipts. Other deployment gates remain independent.
    """
    _sha256(expected_policy_sha256, "expected_policy_sha256")
    if sha256(policy_bytes).hexdigest() != expected_policy_sha256:
        raise ValueError("candidate policy SHA-256 differs from the pinned policy")
    if search_scope not in {
        "joint_topology_and_donors",
        "fixed_topology_all_targets_and_permitted_donors",
    }:
        raise ValueError("unknown proof scope")
    if isinstance(required_benchmarks, str):
        raise ValueError("required_benchmarks must contain individual benchmark names")
    benchmarks = tuple(required_benchmarks)
    if (
        not benchmarks
        or len(set(benchmarks)) != len(benchmarks)
        or any(not isinstance(name, str) or not name.strip() for name in benchmarks)
    ):
        raise ValueError("specify each required benchmark once")
    policy = json.loads(policy_bytes)
    report = policy["calibration_report"]
    reasons = []
    source = policy["provenance"]
    if (
        source.get("selection_mode") != "verified_benchmark_candidate"
        or source.get("source", {}).get("capture_scope") != "whole_prefill"
    ):
        reasons.append("whole-prefill benchmark replay is missing")
    policies = [(p["min_kv_tokens"], p["max_kv_tokens"]) for p in policy["context_policies"]]
    reports = [(p["min_kv_tokens"], p["max_kv_tokens"]) for p in report["by_bucket"]]
    if not policies or policies != reports or len(policies) != len(set(policies)):
        raise ValueError("incomplete per-range replay report")
    if not report["overall"]["bmm1_skip_objective"]["all_context_buckets_met"]:
        reasons.append("work target is unmet")
    for row in report["by_bucket"]:
        work = row["bmm1_skip_objective"]
        eligible, skipped = work["eligible_tiles"], work["skipped_tiles"]
        if (
            type(skipped) is not int
            or skipped > eligible
            or work["target_met"] is not True
            or skipped < required_skipped_tiles(policy["target_bmm1_skip_ratio"], eligible)
        ):
            reasons.append("work target is unmet in a calibrated range")
        proof = row.get("benchmark_proof", {})
        if (
            proof.get("scope") != search_scope
            or proof.get("status") != "proved_optimal"
            or proof.get("absolute_gap_exact") != "0"
            or not proof.get("worst_prompt_risk_exact")
            or proof.get("lower_bound_exact") != proof["worst_prompt_risk_exact"]
        ):
            reasons.append("missing exact selection proof for the declared search space")
            break
    regressions = {}
    for name in benchmarks:
        if name not in matched_scores:
            reasons.append(f"missing matched accuracy for {name}")
            continue
        candidate, dense = (matched_scores[name][arm] for arm in ("candidate", "dense"))
        if (
            candidate.get("policy_sha256") != expected_policy_sha256
            or dense.get("attention_mode") != "dense"
            or any(
                row.get("checkpoint_manifest_sha256") != policy["checkpoint_manifest_sha256"]
                for row in (candidate, dense)
            )
            or _sha256(candidate.get("evaluation_sha256"), "evaluation_sha256")
            != dense.get("evaluation_sha256")
        ):
            raise ValueError(f"unmatched policy, checkpoint or evaluation for {name}")
        values = [row["accuracy_percent"] for row in (dense, candidate)]
        if any(type(v) not in (int, float) or not 0 <= v <= 100 for v in values):
            raise ValueError(f"invalid percentage for {name}")
        regression = Fraction(str(values[0])) - Fraction(str(values[1]))
        regressions[name] = float(regression)
        if regression > 1:
            reasons.append(f"{name} exceeds one percentage point of regression")
    return {
        "policy_sha256": expected_policy_sha256,
        "search_scope": search_scope,
        "selection_and_accuracy_pass": not reasons,
        "regression_pp": regressions,
        "reasons": reasons,
    }
