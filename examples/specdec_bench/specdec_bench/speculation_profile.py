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

"""Build a portable ``speculation_profile.json`` from measured acceptance statistics.

The profile is the deployment-facing summary of *how good a draft checkpoint is*:
per-position acceptance rates plus enough provenance to know what they describe. It
is intended to travel with an exported draft checkpoint so downstream consumers stop
guessing.

Two known consumers want the same information in two different conventions:

===================  ===================================================  ==================
Consumer             Wants                                                Field
===================  ===================================================  ==================
Dynamo mocker/AIC    *conditional* -- P(draft i+1 accepted | first i ok)   conditional_accept_rates
vLLM synthetic       *marginal* -- P(first i+1 drafts all accepted)        marginal_accept_rates
===================  ===================================================  ==================

Publishing only one of the two invites a silent misread by the other, so both are
emitted, explicitly named, and cross-checked against the measured mean.

This module is deliberately dependency-free (stdlib only) so it can also be imported
from ``examples/speculative_decoding`` -- ``ar_validate.py`` is a second producer of
the same schema and must not have to pull in the benchmark harness. If a third
producer appears, move this file to a shared location; nothing here binds it to
specdec_bench.
"""

# Not re-exported from specdec_bench/__init__.py: that module deliberately exposes
# only __version__ (and must stay importable without modelopt), so widening it here
# would break its own convention.
__all__ = ["SCHEMA_VERSION", "build_profile", "checkpoint_id", "stub_profile"]

SCHEMA_VERSION = "1.0"

# Methods whose K=n draft is a strict prefix of their K=n+1 draft. For those, the
# marginal vector determines accept_length at every K <= num_speculative_tokens, so a
# single measurement extrapolates. Block-parallel methods (dflash, dspark) and tree
# drafting re-plan the whole block when K changes, so each K must be measured.
_CHAIN_DRAFTING_METHODS = frozenset({"eagle", "eagle1", "eagle2", "eagle3", "draft_model"})


def checkpoint_id(path):
    """Reduce a checkpoint path to its ``org/model`` identifier.

    Unlike ``configuration.json``, which stays with the benchmark run, this profile is
    meant to be *published* next to a checkpoint. Absolute paths would then carry
    internal cluster layout (``/lustre/fsw/portfolios/...``) into a public artifact,
    and they are not portable for a reader anyway. The trailing two components are
    both the useful part and the HuggingFace-style id.

    The full path remains in ``configuration.json`` for local debugging.
    """
    if not path:
        return None
    parts = [p for p in str(path).replace("\\", "/").split("/") if p]
    if not parts:
        return None
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _as_int_keyed(mapping):
    """Normalize a {length: value} map whose keys may be int or str (post-JSON)."""
    if not mapping:
        return {}
    return {int(k): float(v) for k, v in mapping.items()}


def _dense_from_length_keyed(length_keyed, num_speculative_tokens):
    """Project an acceptance-length-keyed map onto a dense per-draft-position vector.

    ``AcceptanceRate`` keys its maps by *acceptance length* -- the number of tokens
    emitted in a decode step, which counts the target model's own bonus token. So
    length 1 means "no draft token was accepted" and the entry for length 1 is
    always 1.0 by construction.

    Consumers index by *draft position*: entry i concerns the (i+1)-th drafted
    token. The two are therefore offset by two, not one::

        position i  <->  length i + 2

    The map is also sparse -- lengths never observed simply do not appear -- while
    consumers require a dense vector of exactly ``num_speculative_tokens`` entries.
    Missing entries mean "never accepted this far", i.e. 0.0.

    Getting either the offset or the densification wrong yields a plausible-looking
    but wrong profile, which is why this lives in one place with one test.
    """
    return [length_keyed.get(i + 2, 0.0) for i in range(num_speculative_tokens)]


def _consistency_check(mean_accept_length, marginal_accept_rates, tolerance=0.02):
    """Cross-check the reported mean against the one implied by the marginals.

    For longest-prefix verification, mean accept length is the sum of the survival
    function: ``AL = 1 + sum_i P(first i+1 drafts all accepted)``. That identity ties
    two independently-derived numbers together, so a mismatch means the histogram,
    the offset, or the densification is wrong -- exactly the failure that would
    otherwise ship silently.

    Returns a dict rather than raising: a profile that fails the check is still worth
    emitting (with the failure recorded) so the discrepancy can be inspected.
    """
    implied = 1.0 + sum(marginal_accept_rates)
    delta = abs(implied - mean_accept_length)
    return {
        "implied_mean_accept_length": round(implied, 6),
        "reported_mean_accept_length": round(mean_accept_length, 6),
        "abs_delta": round(delta, 6),
        "tolerance": tolerance,
        "passed": delta <= tolerance,
    }


def _monotonicity_check(marginal_accept_rates):
    """vLLM's synthetic sampler requires marginals to be non-increasing.

    A survival function cannot increase, so a violation indicates a malformed
    histogram rather than an unusual draft model.
    """
    violations = [
        {"position": i, "value": marginal_accept_rates[i], "previous": marginal_accept_rates[i - 1]}
        for i in range(1, len(marginal_accept_rates))
        if marginal_accept_rates[i] > marginal_accept_rates[i - 1] + 1e-9
    ]
    return {"passed": not violations, "violations": violations}


def build_profile(
    acceptance_out,
    num_speculative_tokens,
    method=None,
    draft_checkpoint=None,
    target_model=None,
    block_size=None,
    max_supported_k=None,
    verification_method="longest_prefix",
    accept_length_model=None,
    per_category=None,
    measurement_conditions=None,
):
    """Assemble a ``speculation_profile.json`` payload.

    Args:
        acceptance_out: the ``AcceptanceRate.out`` dict, after ``process_final``.
            Requires ``Conditional_Acceptance_Rate``, ``Joint_Acceptance_Rate`` and
            ``Average_AL``.
        num_speculative_tokens: K the measurement ran at. Determines vector length.
        method: speculation method (``eagle3``, ``dflash``, ``dspark``, ...). Used to
            pick a default ``accept_length_model``.
        draft_checkpoint / target_model: dicts describing what was measured.
        block_size: trained block size for block-parallel methods.
        max_supported_k: hard ceiling on K. For block-parallel methods, exceeding it
            is invalid rather than merely degraded, so consumers generating a draft
            length schedule must respect it.
        verification_method: ``longest_prefix`` (standard) or ``block``. Block
            verification does not produce a longest-correct-prefix distribution, so
            these vectors would not describe it -- recorded rather than assumed.
        accept_length_model: ``chain_analytic`` (safe to extrapolate over K) or
            ``measured_per_k``. Defaults from ``method``.
        per_category: optional {category: {mean_accept_length, ...}}.
        measurement_conditions: dataset, concurrency, engine, GPU, etc. specdec_bench
            already writes the full record to ``configuration.json``; this carries the
            subset needed to interpret the numbers standalone.

    Returns:
        A JSON-serializable dict.
    """
    conditional_by_length = _as_int_keyed(acceptance_out.get("Conditional_Acceptance_Rate"))
    marginal_by_length = _as_int_keyed(acceptance_out.get("Joint_Acceptance_Rate"))
    mean_accept_length = float(acceptance_out.get("Average_AL", 0.0))

    conditional = _dense_from_length_keyed(conditional_by_length, num_speculative_tokens)
    marginal = _dense_from_length_keyed(marginal_by_length, num_speculative_tokens)

    if accept_length_model is None:
        accept_length_model = (
            "chain_analytic"
            if method and method.lower() in _CHAIN_DRAFTING_METHODS
            else "measured_per_k"
        )

    profile = {
        "schema_version": SCHEMA_VERSION,
        "measured": True,
        "method": method,
        "draft_checkpoint": draft_checkpoint,
        "target_model": target_model,
        "num_speculative_tokens": num_speculative_tokens,
        "block_size": block_size,
        "max_supported_k": max_supported_k
        if max_supported_k is not None
        else num_speculative_tokens,
        "verification_method": verification_method,
        "conditional_accept_rates": [round(x, 6) for x in conditional],
        "marginal_accept_rates": [round(x, 6) for x in marginal],
        "mean_accept_length": round(mean_accept_length, 6),
        "accept_length_model": accept_length_model,
        # Only meaningful once measured at more than one K; populated by the
        # AR-vs-K sweep for block-parallel methods.
        "accept_length_by_k": {str(num_speculative_tokens): round(mean_accept_length, 6)},
        "acceptance_length_histogram": acceptance_out.get("Acceptance_Length_Histogram"),
        "per_category": per_category,
        "measurement_conditions": measurement_conditions,
        "validation": {
            "mean_consistency": _consistency_check(mean_accept_length, marginal),
            "marginal_monotonicity": _monotonicity_check(marginal),
        },
    }
    return profile


def stub_profile(num_speculative_tokens, method=None, **kwargs):
    """An unmeasured placeholder, so ``measured: false`` is distinguishable from absent.

    Consumers can then treat a missing profile as an error rather than having to
    guess whether the checkpoint predates the schema.
    """
    profile = build_profile(
        {"Conditional_Acceptance_Rate": {}, "Joint_Acceptance_Rate": {}, "Average_AL": 0.0},
        num_speculative_tokens,
        method=method,
        **kwargs,
    )
    profile["measured"] = False
    profile["mean_accept_length"] = None
    profile["accept_length_by_k"] = {}
    profile["validation"] = None
    return profile
