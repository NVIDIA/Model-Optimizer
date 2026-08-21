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

"""Aumann-Shapley sensitivity scoring for AutoQuantize.

The ``"aumann_shapley"`` method scores each (runtime group, candidate format) pair by how much
damage it causes, measured in nats of KL divergence against the model's own outputs. Because
the model supplies its own reference, scoring needs no labels. The reference keeps any fixed or
forced-single-format groups quantized, so scores are incremental KL relative to the baseline
recorded in ``damage_model["damage_reference"]`` (``{"type": "unquantized"}`` when nothing is
pinned).

Scoring measures each candidate at a few points partway between the unquantized and the
quantized model, rather than only at the unquantized one. At step
``t = (k + 1/2) / num_path_nodes`` every scored module outputs ``y + t * (Q(y) - y)`` -- a blend
of its real and quantized output, produced by re-running the module with the candidate's
quantizers active -- and one backward pass accumulates ``<dL/dy, Q(y) - y>`` per (group,
format). Spreading the measurement over several steps is what makes a KL objective usable: KL
against the model's own outputs is exactly zero at the unquantized point, and flat there, so
measuring only at that point would give no signal.

Cost per batch is one reference forward, one forward with every group at its most aggressive
format, and one forward+backward per (format, path node) -- independent of how many
configurations the solver later considers. Each scored module additionally re-runs its own
forward once per candidate within those passes. Replay assumes module forwards are
side-effect-free: they are re-run with no state reset between calls, as in the gradient method.

Those raw attributions become solver scores by fitting them to a directly measured calibration
point: the damage of running every group at its most aggressive format. The fit reproduces that
measurement through ``damage = c * (1 - exp(-sum(b)))``, and the resulting per-group values
``b`` are written to ``candidate_stats["scores"]``, so the standard solve gives the best
allocation under this model and the chosen recipe carries a ``predicted_damage`` estimate in
the same measured units. Scores are additionally adjusted so a more aggressive format never
scores better than a less aggressive one, which keeps estimates conservative
(``b_unprojected`` retains the unadjusted values). ``predicted_damage`` is an estimate from
this model, not a bound on realized deployment KL; ``damage_model["valid"]``,
``approximation_flags`` and ``completeness`` -- the fraction of the measured damage the
summed contributions reproduce, 1.0 being exact -- record how far to trust it.

An efficient implementation of the estimator in https://arxiv.org/abs/2607.12266, validated
empirically against it.

Method-specific ``method_options``:

- ``num_path_nodes`` (default 2): how many steps between the unquantized and quantized
  model to measure at.
- ``damage_link`` (default ``"coverage"``): how per-group scores combine into a damage
  estimate. ``"coverage"`` uses the coverage form ``damage = c * (1 - exp(-sum(b)))`` of
  https://arxiv.org/abs/2607.12266; ``"additive"`` sums the raw scores.
- ``solver`` (default ``"lp"``): budget-mode solver -- ``"lp"`` (default, exact) or ``"dp"``
  (deterministic, grid-approximate).
- ``max_predicted_damage`` (default None): minimize weight cost subject to predicted damage <=
  this bound (mutually exclusive with an ``effective_bits`` constraint). Always solved on
  the grid-approximate DP path with conservative rounding, regardless of ``solver``.
"""

import gc
import math
import types
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
import torch

from modelopt.torch.opt.searcher import SearchConfig, SearchStateDict
from modelopt.torch.opt.utils import named_hparams
from modelopt.torch.utils import (
    create_param_grad_clear_hook,
    print_rank_0,
    report_memory,
    warn_rank_0,
)
from modelopt.torch.utils.distributed import DistributedProcessGroup

from .algorithms import (
    AUTO_QUANTIZE_SEARCHERS,
    QuantRecipe,
    QuantRecipeHparam,
    _AutoQuantizeBackwardScoringSearcher,
    _get_kl_div_loss,
    _get_lm_head,
    _get_log_prob,
)

__all__ = ["AutoQuantizeAumannShapleySearcher"]

_DP_GRID = 4096


@dataclass(frozen=True)
class _AttributionData:
    """Normalized attribution tables used by the damage fit."""

    names: tuple[str, ...]
    keys: tuple[str, ...]
    labels: dict[str, str]
    signed_by_key: dict[str, np.ndarray]
    nonnegative_by_key: dict[str, np.ndarray]
    corner_mask_by_key: dict[str, np.ndarray]
    heterogeneous_ladders: bool
    negative_mass: float
    corner_mass: float


@dataclass(frozen=True)
class _DamageFit:
    """Result of mapping attributions into the configured damage model."""

    scores_by_key: dict[str, np.ndarray]
    unprojected_b_by_key: dict[str, np.ndarray]
    is_coverage: bool
    valid: bool
    flags: tuple[str, ...] = ()
    ceiling: float = 0.0
    kappa: float = 1.0
    ceiling_inflation: float = 1.0
    converged: bool = False


@dataclass(frozen=True)
class _SolverTables:
    """Dense per-group tables consumed by the local solvers."""

    names: tuple[str, ...]
    scores: np.ndarray
    costs: np.ndarray
    lengths: np.ndarray


class _ScoreEstimationSession:
    """Own temporary model state and execute Aumann-Shapley scoring batches."""

    def __init__(
        self,
        model,
        hparams,
        score_modules,
        recipes,
        num_path_nodes,
        forward_step,
        lm_head,
        is_param_grad_enabled,
    ):
        self.model = model
        self.hparams = hparams
        self.score_modules = score_modules
        self.recipes = recipes
        self.num_path_nodes = num_path_nodes
        self.forward_step = forward_step
        self.lm_head = lm_head
        self.is_param_grad_enabled = is_param_grad_enabled
        self.no_quant = QuantRecipe(quant_cfg=None)
        self.current_recipe: QuantRecipe | None = None
        self.path_position = 0.0
        self.corner_kl_sum: torch.Tensor | None = None
        self.score_tokens = 0
        self._callbacks = ExitStack()
        self._original_forwards = {}
        self._invocation_diffs = {module: [] for module in score_modules}
        self._grad_accumulators = []

    def __enter__(self):
        try:
            for hparam in self.hparams:
                original_recipe = hparam.active
                self._callbacks.callback(setattr, hparam, "active", original_recipe)

            for module in self.score_modules:
                original_forward = module.forward
                self._original_forwards[module] = original_forward
                module.forward = types.MethodType(self._score_forward, module)
                self._callbacks.callback(setattr, module, "forward", original_forward)
                handle = module.register_full_backward_hook(self._backward_hook)
                self._callbacks.callback(handle.remove)

            for name, param in self.model.named_parameters():
                original_requires_grad = param.requires_grad
                self._callbacks.callback(setattr, param, "requires_grad", original_requires_grad)
                enable_grad = self.is_param_grad_enabled(name, self.model)
                param.requires_grad = enable_grad
                if enable_grad:
                    accumulator, handle = create_param_grad_clear_hook(param)
                    self._grad_accumulators.append(accumulator)
                    self._callbacks.callback(handle.remove)
        except BaseException:
            self._callbacks.close()
            raise
        return self

    def __exit__(self, *_exc_info):
        self._callbacks.close()
        # Release replay state here, not when the session is garbage collected: the caller
        # reports memory right after this block.
        self._clear_invocations()
        self._grad_accumulators.clear()
        self._original_forwards.clear()

    def _set_all_hparams(self, recipe_of) -> None:
        """Set every configurable recipe hparam."""
        for hparam in self.hparams:
            hparam.active = recipe_of(hparam)

    def _clear_invocations(self) -> None:
        """Discard cached calls left unused by the preceding backward pass."""
        for invocation_stack in self._invocation_diffs.values():
            invocation_stack.clear()

    def _score_forward(self, module, input, *args, **kwargs):
        """Emit a path-shifted output and cache each candidate output difference."""
        recipe = self.current_recipe
        original_forward = self._original_forwards[module]
        if recipe is None:
            return original_forward(input, *args, **kwargs)

        grad_pass = torch.is_grad_enabled()
        for hparam in module._hparams_for_scoring:
            if hparam.is_configurable:
                hparam.active = self.no_quant
        output = original_forward(input, *args, **kwargs)
        base = output[0] if isinstance(output, tuple) else output

        diffs: dict[QuantRecipeHparam, torch.Tensor] = {}
        diff_total = None
        with torch.no_grad():
            for hparam in module._hparams_for_scoring:
                if not hparam.is_configurable or recipe not in hparam.choices:
                    continue
                hparam.active = recipe
                try:
                    quant_output = original_forward(input, *args, **kwargs)
                finally:
                    hparam.active = self.no_quant
                quant_output = quant_output[0] if isinstance(quant_output, tuple) else quant_output
                diff = (quant_output - base).detach()
                diffs[hparam] = diff
                diff_total = diff if diff_total is None else diff_total + diff

        if diff_total is None:
            return output
        if grad_pass and base.requires_grad:
            self._invocation_diffs[module].append(diffs)
        shifted = base + self.path_position * diff_total
        return (shifted, *output[1:]) if isinstance(output, tuple) else shifted

    def _backward_hook(self, module, _grad_input, grad_output) -> None:
        """Consume the matching replay data and accumulate its attribution."""
        invocation_stack = self._invocation_diffs[module]
        diffs = invocation_stack.pop() if invocation_stack else None
        recipe = self.current_recipe
        if not diffs or recipe is None or grad_output[0] is None:
            return
        with torch.no_grad():
            grad = grad_output[0].float()
            for hparam, diff in diffs.items():
                contribution = (grad * diff.float()).sum() / self.num_path_nodes
                current = hparam._importance_dict[recipe][module]
                hparam._importance_dict[recipe][module] = (
                    contribution if current is None else current + contribution
                )

    def score_step(self, model, data) -> None:
        """Score every format and path node for one calibration batch."""
        self.current_recipe = None
        self._set_all_hparams(lambda _hparam: self.no_quant)
        with torch.no_grad():
            ref_logits = self.forward_step(model, data)
            ref_logprob = _get_log_prob(ref_logits, lm_head=self.lm_head).detach()
            self.score_tokens += int(ref_logprob.numel() // ref_logprob.shape[-1])
            del ref_logits

            self._set_all_hparams(lambda hparam: hparam.choices[0])
            try:
                corner_logits = self.forward_step(model, data)
                corner_loss = _get_kl_div_loss(ref_logprob, corner_logits, self.lm_head).detach()
                self.corner_kl_sum = (
                    corner_loss if self.corner_kl_sum is None else self.corner_kl_sum + corner_loss
                )
                del corner_logits
            finally:
                self._set_all_hparams(lambda _hparam: self.no_quant)

        try:
            for recipe in self.recipes:
                self.current_recipe = recipe
                for node in range(self.num_path_nodes):
                    self.path_position = (node + 0.5) / self.num_path_nodes
                    self._clear_invocations()
                    logits = self.forward_step(model, data)
                    loss = _get_kl_div_loss(ref_logprob, logits, self.lm_head)
                    loss.backward()
                    del logits, loss
        finally:
            self.current_recipe = None


def _as_seed_coverage(attributions, c, *, p=0.5, iters=200, tol=1e-10, damping=0.5):
    """Invert ``AS_i = c a_i prod_{j != i} (1 - p a_j)`` for per-group damage fractions.

    Each ``a_i`` is the fraction of the ceiling ``c`` that group ``i`` accounts for.

    Returns ``(a, b = -log(1 - a), converged)``. Zero attributions map to exactly zero
    fractions; arbitrarily small positive attributions map to proportionally small ones. The
    system is infeasible when the attribution mass is too large for the ceiling; the
    iteration then diverges to the clip and the caller should inflate ``c`` and retry (see
    :func:`_anchor_ceiling`).
    """
    attributions = np.maximum(np.asarray(attributions, dtype=float), 0.0)
    if not (attributions > 0).any():
        zeros = np.zeros_like(attributions)
        return zeros, zeros.copy(), True
    a = np.clip(attributions / max(c, 1e-12), 0.0, 0.999)
    converged = False
    for _ in range(iters):
        log_prod = np.log(np.clip(1.0 - p * a, 1e-9, None)).sum()
        discount = np.exp(log_prod - np.log(np.clip(1.0 - p * a, 1e-9, None)))
        new = np.clip(attributions / (max(c, 1e-12) * np.clip(discount, 1e-6, None)), 0.0, 0.999)
        nxt = damping * a + (1.0 - damping) * new
        delta = float(np.abs(nxt - a).max())
        a = nxt
        if delta < tol:
            converged = bool(a.max() < 0.995)
            break
    b = -np.log(np.clip(1.0 - a, 1e-9, None))
    return a, b, converged


def _anchor_ceiling(as_by_key, f_corner, corner_mask_by_key, max_inflation=10.0):
    """Invert per-format attributions into per-group damage fractions, sharing one ceiling.

    The ceiling starts at the measured corner damage and is inflated minimally until the
    inversion converges for every format; ``b`` is then rescaled by ``kappa`` so the link stays
    exact at the corner. Returns ``(c, b_by_key, kappa, inflation, converged)``. Raises
    ``ValueError`` on non-finite inputs (callers must screen measurements first).
    """
    if not math.isfinite(f_corner) or not all(
        bool(np.isfinite(v).all()) for v in as_by_key.values()
    ):
        raise ValueError("corner damage and attributions must be finite")
    f_corner = max(float(f_corner), 1e-12)
    c = f_corner
    # The exit conditions bound this well below the cap; the cap is a backstop only.
    for _ in range(1 + math.ceil(math.log(max_inflation) / math.log(1.3))):
        inversions = {k: _as_seed_coverage(v, c=c) for k, v in as_by_key.items()}
        if all(conv for _a, _b, conv in inversions.values()) or c > max_inflation * f_corner:
            break
        c *= 1.3
    converged = all(conv for _a, _b, conv in inversions.values())
    b_by_key = {k: inv[1] for k, inv in inversions.items()}

    def corner_b_sum():
        """Total log-headroom over the corner candidates."""
        return sum(float(b_by_key[k][mask].sum()) for k, mask in corner_mask_by_key.items())

    kappa = 1.0
    tolerance = 0.01 * f_corner
    if abs(_predict_damage(c, corner_b_sum()) - f_corner) > tolerance:
        # Exact anchoring needs strict headroom above the corner (kappa solves
        # c * (1 - exp(-kappa * sum(b_corner))) == f_corner, impossible at c == f_corner).
        if c < 1.01 * f_corner:
            c = 1.01 * f_corner
            inversions = {k: _as_seed_coverage(v, c=c) for k, v in as_by_key.items()}
            converged = all(conv for _a, _b, conv in inversions.values())
            b_by_key = {k: inv[1] for k, inv in inversions.items()}
        total = corner_b_sum()
        if total > 0:
            kappa = -np.log(1.0 - f_corner / c) / total
            b_by_key = {k: b * kappa for k, b in b_by_key.items()}
    anchored = abs(_predict_damage(c, corner_b_sum()) - f_corner) <= tolerance
    return float(c), b_by_key, float(kappa), float(c / f_corner), converged and anchored


def _predict_damage(c, b_sum):
    """Damage implied by a total score under the coverage link."""
    return float(c * (1.0 - np.exp(-max(float(b_sum), 0.0))))


def _mckp_max_value(values, costs, budget):
    """Exact multiple-choice knapsack: max ``sum(values)`` s.t. ``sum(costs) <= budget``.

    Choice 0 of each row must have zero cost so every row has a feasible pick.
    """
    values = np.asarray(values, dtype=float)
    costs = np.asarray(costs, dtype=np.int64)
    n, num_choices = values.shape
    if (costs[:, 0] != 0).any():
        raise ValueError("choice 0 must have zero cost")
    grid = int(budget)
    dp = np.zeros(grid + 1)
    choice = np.zeros((n, grid + 1), dtype=np.int32)
    candidates = np.empty((num_choices, grid + 1))
    for i in range(n):
        for k in range(num_choices):
            cost = int(costs[i, k])
            if cost > grid:
                candidates[k] = -np.inf
                continue
            candidates[k, :cost] = -np.inf
            candidates[k, cost:] = dp[: grid + 1 - cost] + values[i, k]
        best = candidates.argmax(axis=0)
        choice[i] = best
        dp = candidates[best, np.arange(grid + 1)]
    selection = np.zeros(n, dtype=int)
    g = grid
    for i in range(n - 1, -1, -1):
        k = int(choice[i, g])
        selection[i] = k
        g -= int(costs[i, k])
    return selection, float(dp[grid])


class AutoQuantizeAumannShapleySearcher(_AutoQuantizeBackwardScoringSearcher):
    """AutoQuantize searcher scoring with Aumann-Shapley damage attributions (see module doc)."""

    method_name = "aumann_shapley"
    method_options_keys = frozenset(
        {"num_path_nodes", "damage_link", "solver", "max_predicted_damage"}
    )

    @property
    def default_search_config(self) -> SearchConfig:
        """Get the default config for the searcher."""
        config = super().default_search_config
        config.update(
            {
                "forward_step": None,
                "num_path_nodes": 2,
                "damage_link": "coverage",
                "solver": "lp",
                "max_predicted_damage": None,
            }
        )
        return config

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Get the default state dict for AutoQuantize."""
        state = super().default_state_dict
        state["damage_model"] = None
        state["scoring_signature"] = None
        return state

    def _damage_reference(self, no_quant) -> dict:
        """The baseline every score, corner, and quote is measured against.

        Groups pinned to one quantized format -- via ``fixed_quantization_config`` or a
        single-candidate ``module_search_spaces`` entry with ``allow_no_quant=False`` -- stay
        active during the reference passes, so all damage values are INCREMENTAL KL relative
        to this resolved baseline, not total degradation from the unquantized model.
        """
        forced_groups = {
            name: str(stat["formats"][0])
            for name, stat in self.candidate_stats.items()
            if len(stat["formats"]) == 1 and stat["formats"][0] != no_quant
        }
        if getattr(self, "fixed_quantization_config", None) is None and not forced_groups:
            return {"type": "unquantized"}
        return {
            "type": "quantized_baseline",
            "fixed_quantization_config_signature": getattr(
                self, "fixed_quantization_config_signature", None
            ),
            "forced_groups": forced_groups,
        }

    def _current_scoring_signature(self) -> dict:
        """Settings that determine what the stored scores mean."""
        # Settings that change what the stored scores MEAN; solver and max_predicted_damage only
        # change how they are solved and may differ across a checkpoint resume.
        return {
            "version": 1,
            "path_variant": "module_output_replay_v1",
            "num_path_nodes": int(self.config["num_path_nodes"]),
            "damage_link": self.config["damage_link"],
        }

    def sanitize_search_config(self, config: SearchConfig | None) -> SearchConfig:
        """Sanitize the search config dict."""
        config = config or {}
        for ignored_key in ["score_func", "loss_func", "forward_backward_step"]:
            if ignored_key in config:
                if config[ignored_key] is not None:
                    warn_rank_0(
                        f"`{ignored_key}` is ignored for Aumann-Shapley `auto_quantize`: the loss "
                        "is fixed to KL divergence against the model's own reference outputs."
                    )
                config.pop(ignored_key)
        config = super().sanitize_search_config(config)
        assert config["forward_step"] is not None, (
            "`forward_step` must be provided for Aumann-Shapley `auto_quantize`. "
            "`forward_step(model, data)` should return model logits."
        )
        nodes = config["num_path_nodes"]
        if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes < 1:
            raise ValueError(f"num_path_nodes must be an integer >= 1, got {nodes!r}")
        if config["damage_link"] not in ("coverage", "additive"):
            raise ValueError(
                f"damage_link must be 'coverage' or 'additive', got {config['damage_link']!r}"
            )
        if config["solver"] not in ("lp", "dp"):
            raise ValueError(f"solver must be 'lp' or 'dp', got {config['solver']!r}")
        bound = config["max_predicted_damage"]
        if bound is not None and (
            not isinstance(bound, (int, float))
            or isinstance(bound, bool)
            or not math.isfinite(bound)
            or bound <= 0
        ):
            raise ValueError(
                f"max_predicted_damage must be a finite positive number, got {bound!r}"
            )
        return config

    def validate_search_input(self, constraints, config) -> None:
        """Reject ambiguous target combinations (runs before any model mutation)."""
        if (
            config.get("max_predicted_damage") is not None
            and (constraints or {}).get("effective_bits") is not None
        ):
            raise ValueError(
                "Provide either constraints['effective_bits'] or "
                "method_options['max_predicted_damage'], not both: the damage-bound mode "
                "solves for the minimum effective bits itself."
            )

    def before_search(self) -> None:
        """Prepare the model for search; damage-bound mode supplies the bit budget itself."""
        # Reject before ``super().before_search()`` calibrates every search recipe.
        self._raise_if_vocab_sharded()
        if self.config["max_predicted_damage"] is not None:
            self.validate_search_input(self.constraints, self.config)
            self.constraints = {"effective_bits": 16.0, **self.constraints}
        # Stored scores are only reusable when their meaning is unchanged (see
        # _current_scoring_signature); solver/SLA re-solves are allowed on resume.
        current_signature = self._current_scoring_signature()
        restored_signature = getattr(self, "scoring_signature", None)
        if self.candidate_stats and restored_signature not in (None, current_signature):
            raise ValueError(
                f"Checkpoint scoring signature {restored_signature} does not match the "
                f"current search config {current_signature}. Use a different checkpoint path."
            )
        self.scoring_signature = current_signature
        super().before_search()

    def _configurable_hparams(self) -> list[QuantRecipeHparam]:
        """Every configurable quant-recipe hparam in the model."""
        return [
            hparam
            for _name, hparam in named_hparams(self.model, unique=True)
            if isinstance(hparam, QuantRecipeHparam) and hparam.is_configurable
        ]

    def _configurable_score_modules(self) -> list[torch.nn.Module]:
        """Unique modules that contribute to configurable score groups."""
        score_modules = []
        seen: set[int] = set()
        for module in self.model.modules():
            scoring_hparams = getattr(module, "_hparams_for_scoring", ())
            if any(hparam.is_configurable for hparam in scoring_hparams) and id(module) not in seen:
                score_modules.append(module)
                seen.add(id(module))
        return score_modules

    @torch.enable_grad()
    def _estimate_auto_quantize_scores(self, is_param_grad_enabled):
        """Accumulate path-integral damage attributions for each candidate."""
        model = self.model
        no_quant = QuantRecipe(quant_cfg=None)
        self._raise_if_vocab_sharded()
        hparams = self._configurable_hparams()
        recipes = sorted({r for h in hparams for r in h.choices if r != no_quant})
        scoring_session = _ScoreEstimationSession(
            model,
            hparams,
            self._configurable_score_modules(),
            recipes,
            int(self.config["num_path_nodes"]),
            self.config["forward_step"],
            _get_lm_head(model),
            is_param_grad_enabled,
        )
        with scoring_session:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                report_memory("AutoQuantize(aumann_shapley): starting score estimation, ")
            self._run_func(
                scoring_session.score_step,
                num_iters=self.config["num_score_steps"],
                desc="Estimating aumann_shapley scores",
            )

        self._corner_kl_sum = scoring_session.corner_kl_sum
        self._score_tokens = scoring_session.score_tokens
        gc.collect()
        if torch.cuda.is_available():
            report_memory("AutoQuantize(aumann_shapley): after score estimation")

    def _loss_is_vocab_sharded(self) -> bool:
        """Whether the loss is computed over a vocab-sharded lm_head."""
        lm_head = _get_lm_head(self.model)
        parallel_state = getattr(lm_head, "parallel_state", None) if lm_head is not None else None
        return parallel_state is not None and parallel_state.tensor_parallel_group.is_initialized()

    def _raise_if_vocab_sharded(self) -> None:
        """Reject vocab-sharded losses, which the score pass cannot backprop."""
        # The score passes backprop through the KL loss; the vocab-sharded log-softmax
        # uses in-place collectives that autograd cannot differentiate through.
        if self._loss_is_vocab_sharded():
            raise NotImplementedError(
                "aumann_shapley scoring does not support vocab-sharded (Megatron "
                "tensor-parallel) losses yet. Use method='gradient' with a Megatron loss_func."
            )

    def _reduce_loss_scalar(self, value: float) -> float:
        """Reduce a loss scalar over the data-parallel group."""
        # A loss scalar is sharded over DP (disjoint batches) and, only when the loss itself
        # is vocab-sharded, over TP; it is REPLICATED across EP ranks (unlike per-module
        # importances, which get_score sums over all three groups).
        module = self._any_score_parallel_module()
        if module is None:
            return value
        parallel_state = module.parallel_state
        sum_groups = [parallel_state.data_parallel_group]
        if self._loss_is_vocab_sharded():
            sum_groups.append(parallel_state.tensor_parallel_group)
        value = DistributedProcessGroup.get_dist_syncd_obj(value, sum_groups, sum)
        return DistributedProcessGroup.get_dist_syncd_obj(
            value, [parallel_state.expert_model_parallel_group], lambda a: a[0]
        )

    def _reduce_token_count(self, count: int) -> int:
        """Reduce a token count over the data-parallel group."""
        module = self._any_score_parallel_module()
        if module is None:
            return count
        return DistributedProcessGroup.get_dist_syncd_obj(
            count, [module.parallel_state.data_parallel_group], sum
        )

    def _any_score_parallel_module(self):
        """A module carrying a parallel state, if any.

        Falls back to quant modules so the loss and token reductions stay consistent with
        get_score, which uses the same fallback when a score module is a plain container.
        """
        hparams = self._configurable_hparams()
        for collection in ("score_modules", "quant_modules"):
            for hparam in hparams:
                for module in getattr(hparam, collection, ()):
                    if getattr(module, "parallel_state", None) is not None:
                        return module
        return None

    def _exclude_non_finite_candidates(
        self, no_quant
    ) -> tuple[dict[str, list[str]], dict[str, str], bool]:
        """Drop candidates whose measured attribution is non-finite.

        A non-finite score is a measurement of a format that destroys the reference output
        (or of a numerically broken pass); zeroing or clamping it would make that candidate
        look cheap to the solver, so it is removed from its group's ladder instead. When a
        constraint is only reachable through removed candidates, the solve reports
        ``is_satisfied=False``. Returns ``(excluded, forced, corner_removed)``:

        - ``excluded``: the removed format labels per group.
        - ``forced``: groups with neither a finite candidate nor a no-quant fallback, mapped
          to their retained format -- the least aggressive entry, kept as the forced choice
          with a neutral solver score (the non-finite raw measurement is preserved). The
          caller reports such searches unsatisfied and invalidates the damage model.
        - ``corner_removed``: True when some group lost its most aggressive format. The
          measured corner ran with that format active, so the anchor no longer corresponds
          to the candidate corner and the caller invalidates the coverage fit.
        """
        excluded: dict[str, list[str]] = {}
        forced: dict[str, str] = {}
        corner_removed = False
        for name, stat in self.candidate_stats.items():
            if stat.get("is_fixed", False) or len(stat["formats"]) <= 1:
                continue
            keep = [
                index
                for index, (recipe, raw) in enumerate(
                    zip(stat["formats"], stat["raw_scores"], strict=True)
                )
                if recipe == no_quant or math.isfinite(raw)
            ]
            if len(keep) == len(stat["formats"]):
                continue
            if not keep:
                keep = [len(stat["formats"]) - 1]
                stat["scores"][keep[0]] = 0.0
                forced[name] = str(stat["formats"][keep[0]])
            if keep[0] != 0:
                corner_removed = True
            excluded[name] = [
                str(stat["formats"][index])
                for index in range(len(stat["formats"]))
                if index not in keep
            ]
            for field in ("formats", "scores", "raw_scores", "costs"):
                stat[field] = [stat[field][index] for index in keep]
            # The base's running-min chain propagates an excluded candidate's -inf into every
            # less aggressive entry; groups pruned to no_quant are never rewritten downstream.
            stat["scores"] = [score if math.isfinite(score) else 0.0 for score in stat["scores"]]
        if excluded:
            warn_rank_0(
                "aumann_shapley: excluding candidates with non-finite damage measurements "
                f"from the search space: {excluded}"
            )
        return excluded, forced, corner_removed

    def _collect_attributions(
        self, names: list[str], no_quant: QuantRecipe, tokens: int
    ) -> _AttributionData:
        """Build normalized, format-keyed attribution tables from candidate stats."""
        recipe_by_key: dict[str, QuantRecipe] = {}
        for name in names:
            for recipe in self.candidate_stats[name]["formats"]:
                if recipe != no_quant:
                    recipe_by_key.setdefault(recipe.checkpoint_signature, recipe)

        keys = tuple(sorted(recipe_by_key))
        labels = {key: str(recipe_by_key[key]) for key in keys}
        signed_by_key = {key: np.zeros(len(names)) for key in keys}
        ladders = set()
        for index, name in enumerate(names):
            stat = self.candidate_stats[name]
            ladder = []
            for recipe, raw_score in zip(stat["formats"], stat["raw_scores"], strict=True):
                if recipe == no_quant:
                    continue
                key = recipe.checkpoint_signature
                signed_by_key[key][index] = raw_score / tokens
                ladder.append(key)
            ladders.add(tuple(ladder))

        corner_mask_by_key = {key: np.zeros(len(names), dtype=bool) for key in keys}
        for index, name in enumerate(names):
            corner = self.candidate_stats[name]["formats"][0]
            if corner.is_no_quant:
                raise ValueError(
                    f"no_quant sorted first in the candidate ladder for {name}; "
                    "QuantRecipe ordering must keep it last."
                )
            corner_mask_by_key[corner.checkpoint_signature][index] = True

        signed_total = sum(float(np.abs(vector).sum()) for vector in signed_by_key.values())
        negative_total = sum(
            float(np.abs(np.minimum(vector, 0.0)).sum()) for vector in signed_by_key.values()
        )
        corner_mass = sum(
            float(signed_by_key[key][mask].sum()) for key, mask in corner_mask_by_key.items()
        )
        return _AttributionData(
            names=tuple(names),
            keys=keys,
            labels=labels,
            signed_by_key=signed_by_key,
            nonnegative_by_key={
                key: np.maximum(vector, 0.0) for key, vector in signed_by_key.items()
            },
            corner_mask_by_key=corner_mask_by_key,
            heterogeneous_ladders=len(ladders) > 1,
            negative_mass=negative_total / max(signed_total, 1e-12),
            corner_mass=corner_mass,
        )

    def _fit_damage_link(
        self,
        attributions: _AttributionData,
        f_corner: float,
        *,
        corner_removed: bool,
        valid: bool,
    ) -> _DamageFit:
        """Map normalized attributions to additive or coverage-link solver scores."""
        is_coverage = (
            self.config["damage_link"] == "coverage"
            and bool(attributions.names)
            and bool(attributions.keys)
        )
        if not is_coverage:
            return _DamageFit(
                scores_by_key=attributions.nonnegative_by_key,
                unprojected_b_by_key={},
                is_coverage=False,
                valid=valid,
            )

        flags: list[str] = []
        zeros_by_key = {key: np.zeros(len(attributions.names)) for key in attributions.keys}
        positive_mass = sum(
            float(vector.sum()) for vector in attributions.nonnegative_by_key.values()
        )
        if not math.isfinite(f_corner) or corner_removed:
            return _DamageFit(
                scores_by_key=attributions.nonnegative_by_key,
                unprojected_b_by_key=zeros_by_key,
                is_coverage=True,
                valid=False,
            )

        if f_corner <= 1e-12:
            if positive_mass <= 1e-12:
                flags.append("zero_damage")
                return _DamageFit(
                    scores_by_key=zeros_by_key,
                    unprojected_b_by_key=zeros_by_key,
                    is_coverage=True,
                    valid=valid,
                    flags=tuple(flags),
                    converged=True,
                )
            flags.append("zero_corner_with_attribution_mass")
            return _DamageFit(
                scores_by_key=attributions.nonnegative_by_key,
                unprojected_b_by_key=zeros_by_key,
                is_coverage=True,
                valid=False,
                flags=tuple(flags),
            )

        ceiling, b_by_key, kappa, inflation, converged = _anchor_ceiling(
            attributions.nonnegative_by_key,
            f_corner,
            attributions.corner_mask_by_key,
        )
        return _DamageFit(
            scores_by_key=b_by_key,
            unprojected_b_by_key=b_by_key,
            is_coverage=True,
            valid=valid and converged and math.isfinite(ceiling),
            ceiling=ceiling,
            kappa=kappa,
            ceiling_inflation=inflation,
            converged=converged,
        )

    def _project_solver_scores(
        self,
        attributions: _AttributionData,
        no_quant: QuantRecipe,
        scores_by_key: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], float]:
        """Write monotone per-group scores and return their format-keyed projection."""
        projected_by_key = {key: np.zeros(len(attributions.names)) for key in attributions.keys}
        for index, name in enumerate(attributions.names):
            stat = self.candidate_stats[name]
            scores = [
                0.0
                if recipe == no_quant
                else float(scores_by_key[recipe.checkpoint_signature][index])
                for recipe in stat["formats"]
            ]
            for choice in range(len(scores) - 2, -1, -1):
                scores[choice] = max(scores[choice], scores[choice + 1])
            stat["scores"] = scores
            for recipe, score in zip(stat["formats"], scores, strict=True):
                if recipe != no_quant:
                    projected_by_key[recipe.checkpoint_signature][index] = score

        unprojected_total = sum(float(vector.sum()) for vector in scores_by_key.values())
        projected_total = sum(float(vector.sum()) for vector in projected_by_key.values())
        adjustment = (projected_total - unprojected_total) / max(unprojected_total, 1e-12)
        return projected_by_key, adjustment

    @staticmethod
    def _measurement_diagnostics(
        attributions: _AttributionData,
        f_corner: float,
        excluded: dict[str, list[str]],
        forced_candidates: dict[str, str],
        corner_removed: bool,
    ) -> tuple[bool, list[str]]:
        """Return ordered diagnostic flags and whether the fit can be certified."""
        diagnostics = (
            ("non_finite_measurements", not math.isfinite(f_corner), True),
            ("non_finite_scores_excluded", bool(excluded), False),
            ("corner_format_excluded", corner_removed, True),
            ("non_finite_candidate_forced", bool(forced_candidates), True),
            ("heterogeneous_ladders", attributions.heterogeneous_ladders, False),
            ("negative_attribution_mass", attributions.negative_mass > 1e-3, False),
        )
        flags = [name for name, present, _invalidates in diagnostics if present]
        valid = not any(present and invalidates for _name, present, invalidates in diagnostics)
        return valid, flags

    def _finalize_damage_model(
        self,
        attributions: _AttributionData,
        fit: _DamageFit,
        no_quant: QuantRecipe,
        *,
        f_corner: float,
        tokens: int,
        damage_reference: dict,
        excluded: dict[str, list[str]],
        forced_candidates: dict[str, str],
        flags: list[str],
    ) -> dict:
        """Project solver scores and assemble the persisted damage-model record."""
        damage_model = {
            "link": self.config["damage_link"],
            "f_corner": f_corner,
            "n_score_tokens": tokens,
            "damage_reference": damage_reference,
            "negative_attribution_mass": attributions.negative_mass,
            "as_scores": {
                attributions.labels[key]: dict(zip(attributions.names, vector.tolist()))
                for key, vector in attributions.signed_by_key.items()
            },
        }
        if excluded:
            damage_model["excluded_candidates"] = excluded
        if forced_candidates:
            damage_model["forced_candidates"] = forced_candidates

        projected_by_key, adjustment = self._project_solver_scores(
            attributions, no_quant, fit.scores_by_key
        )
        if adjustment > 1e-9:
            flags.append("monotonicity_projection")
            damage_model["monotonicity_adjustment"] = adjustment

        if fit.is_coverage:
            projected_corner_b = sum(
                float(projected_by_key[key][mask].sum())
                for key, mask in attributions.corner_mask_by_key.items()
            )
            damage_model.update(
                {
                    "c": fit.ceiling,
                    "kappa": fit.kappa,
                    "ceiling_inflation": fit.ceiling_inflation,
                    "inversion_converged": fit.converged,
                    "b": {
                        attributions.labels[key]: dict(zip(attributions.names, vector.tolist()))
                        for key, vector in projected_by_key.items()
                    },
                    "b_unprojected": {
                        attributions.labels[key]: dict(zip(attributions.names, vector.tolist()))
                        for key, vector in fit.unprojected_b_by_key.items()
                    },
                    "projected_corner_damage": _predict_damage(fit.ceiling, projected_corner_b),
                }
            )

        damage_model["valid"] = fit.valid
        damage_model["approximation_flags"] = flags
        damage_model["completeness"] = attributions.corner_mass / max(f_corner, 1e-12)
        return damage_model

    def initialize_candidate_stats(self):
        """Initialize candidate stats, then convert raw attributions through the damage link.

        The base implementation performs the distributed score reduction; the nonlinear
        coverage inversion runs after it, on rank-identical values, and overwrites the
        per-choice scores with log-headroom ``b`` so that minimizing their sum under the
        standard solve is the coverage-optimal allocation.
        """
        super().initialize_candidate_stats()

        no_quant = QuantRecipe(quant_cfg=None)
        # Scoring-time semantics must be captured before pruning rewrites the ladders: the
        # reference baseline, and which groups were configurable when scores were measured.
        damage_reference = self._damage_reference(no_quant)
        eligible = [
            name
            for name, stat in self.candidate_stats.items()
            if not stat.get("is_fixed", False)
            and len(stat["formats"]) > 1
            and any(r != no_quant for r in stat["formats"])
        ]
        excluded, forced_candidates, corner_removed = self._exclude_non_finite_candidates(no_quant)
        tokens = self._reduce_token_count(int(getattr(self, "_score_tokens", 0)))
        if tokens <= 0:
            warn_rank_0("aumann_shapley: no scored tokens; leaving raw scores in place.")
            # Record an invalidated model rather than leaving it unset, with the same key
            # shape as the normal path: the quote is keyed off damage_model.
            self.damage_model = {
                "link": self.config["damage_link"],
                "valid": False,
                "approximation_flags": ["no_scored_tokens"],
                "completeness": float("nan"),
                "f_corner": float("nan"),
                "n_score_tokens": 0,
                "damage_reference": None,
                "as_scores": {},
            }
            return
        corner_kl_sum = getattr(self, "_corner_kl_sum", None)
        corner_kl_sum = 0.0 if corner_kl_sum is None else float(corner_kl_sum.item())
        f_corner = self._reduce_loss_scalar(corner_kl_sum) / tokens

        names = [
            name
            for name in eligible
            if name not in forced_candidates
            and any(r != no_quant for r in self.candidate_stats[name]["formats"])
        ]
        attributions = self._collect_attributions(names, no_quant, tokens)
        if attributions.heterogeneous_ladders:
            warn_rank_0(
                "aumann_shapley: groups have differing candidate ladders; the joint coverage "
                "interpretation is approximate for the formats not shared by all groups."
            )

        valid, flags = self._measurement_diagnostics(
            attributions,
            f_corner,
            excluded,
            forced_candidates,
            corner_removed,
        )
        fit = self._fit_damage_link(
            attributions,
            f_corner,
            corner_removed=corner_removed,
            valid=valid,
        )
        valid = fit.valid
        flags.extend(fit.flags)
        if fit.is_coverage and not valid:
            warn_rank_0(
                "aumann_shapley: the coverage damage fit is not valid "
                f"(flags={flags or ['inversion_not_converged']}); damage quotes are "
                "unreliable and damage-bound searches will report is_satisfied=False."
            )

        self.damage_model = self._finalize_damage_model(
            attributions,
            fit,
            no_quant,
            f_corner=f_corner,
            tokens=tokens,
            damage_reference=damage_reference,
            excluded=excluded,
            forced_candidates=forced_candidates,
            flags=flags,
        )

    def run_search_with_stats(self, max_weight_size, verbose=False):
        """Dispatch to the LP (default), the exact DP, or the SLA search."""
        max_predicted_damage = self.config.get("max_predicted_damage")
        if max_predicted_damage is not None:
            recipes, is_satisfied = self._run_damage_bound_search(
                float(max_predicted_damage), verbose
            )
        elif self.config.get("solver", "lp") == "dp":
            recipes, is_satisfied = self._run_dp_budget_search(max_weight_size, verbose)
        else:
            recipes, is_satisfied = self._run_linear_program_search(max_weight_size, verbose)
        flags = (getattr(self, "damage_model", None) or {}).get("approximation_flags", [])
        if is_satisfied and "non_finite_candidate_forced" in flags:
            warn_rank_0(
                "AutoQuantize FAILED to find a valid solution! The selection includes a "
                "forced candidate whose damage measurement was non-finite. "
            )
            is_satisfied = False
        return recipes, is_satisfied

    def run_search(self):
        """Run the inherited search and attach the predicted-damage quote."""
        super().run_search()
        self._attach_predicted_damage()
        # The base flow only saves before solving; re-save so the checkpoint file carries the
        # chosen recipe and can be re-solved offline.
        self.save_search_checkpoint(verbose=self.config.get("verbose", False))

    def _attach_predicted_damage(self) -> None:
        """Record the damage quote for the selected recipe."""
        damage_model = getattr(self, "damage_model", None)
        if not damage_model or not self.best.get("recipe"):
            return
        no_quant = QuantRecipe(quant_cfg=None)
        total_score = 0.0
        for name, recipe in self.best["recipe"].items():
            stat = self.candidate_stats[name]
            if stat.get("is_fixed", False) or recipe == no_quant:
                continue
            total_score += stat["scores"][stat["formats"].index(recipe)]
        if damage_model["link"] == "coverage" and "c" in damage_model:
            predicted = _predict_damage(damage_model["c"], total_score)
        else:
            predicted = float(total_score)
        valid = bool(damage_model.get("valid", True))
        if not valid:
            # An invalidated fit leaves c = 0.0, which would otherwise quote exactly 0.0.
            predicted = float("nan")
        self.best["predicted_damage"] = predicted
        self.best["predicted_damage_valid"] = valid
        if self.config.get("verbose"):
            print_rank_0(
                f"AutoQuantize(aumann_shapley) predicted damage: {predicted:.4e} "
                "(mean per-token KL, calibration units)"
            )

    def _stat_tables(self) -> _SolverTables:
        """Per-group score, cost and length tables for the solvers."""
        stats = self.candidate_stats
        names = tuple(stats)
        n = len(names)
        num_choices = max(len(stats[name]["formats"]) for name in names)
        scores = np.full((n, num_choices), np.inf)
        costs = np.full((n, num_choices), np.inf)
        lengths = np.zeros(n, dtype=int)
        for i, name in enumerate(names):
            stat = stats[name]
            k = len(stat["formats"])
            lengths[i] = k
            scores[i, :k] = stat["scores"]
            costs[i, :k] = stat["costs"]
        return _SolverTables(names, scores, costs, lengths)

    def _best_recipes_from_choice(self, names, choice_idx):
        """Map per-group choice indices back to recipes."""
        best_recipes = {}
        for name, k in zip(names, choice_idx, strict=True):
            stat = self.candidate_stats[name]
            best_recipes[name] = {
                "format": stat["formats"][int(k)],
                "costs": stat["costs"][int(k)],
                "scores": stat["scores"][int(k)],
            }
        return best_recipes

    @staticmethod
    def _min_score_choices(scores, lengths):
        """Per-group choice with the lowest score."""
        # Ties resolve toward the least aggressive choice (scores are non-increasing along
        # ascending compression, so scan from the end).
        return np.array(
            [
                int(lengths[i]) - 1 - int(np.argmin(scores[i, : lengths[i]][::-1]))
                for i in range(len(lengths))
            ]
        )

    @staticmethod
    def _minimize_within_budget(objective, constraint, budget):
        """Per-row choices minimizing ``sum(objective)`` s.t. ``sum(constraint) <= budget``.

        Each row's true minimum constraint is subtracted before discretizing, so only the
        increments above the mandatory baseline consume grid resolution (rows never overflow
        the grid regardless of their count) and ceil rounding keeps the returned selection
        feasible on the true budget. Returns None when even the per-row minima exceed it.
        """
        n = len(objective)
        row_min = np.where(np.isfinite(constraint), constraint, np.inf).min(axis=1)
        remaining = float(budget - row_min.sum())
        if remaining < -1e-9 * max(abs(budget), 1.0):
            return None
        if remaining <= 0:
            # No slack above the mandatory baseline: take each row's minimum-constraint
            # column, breaking ties toward the lower objective.
            at_minimum = constraint <= row_min[:, None]
            return np.where(at_minimum, objective, np.inf).argmin(axis=1)

        quantum = remaining / _DP_GRID
        increments = np.where(
            np.isfinite(constraint),
            np.ceil((constraint - row_min[:, None]) / quantum),
            _DP_GRID + 1,
        ).astype(np.int64)
        # Reorder each row so its zero-increment column is first (DP precondition).
        order = np.argsort(increments, axis=1, kind="stable")
        rows = np.arange(n)[:, None]
        values = np.where(np.isfinite(objective[rows, order]), -objective[rows, order], -np.inf)
        selection, _ = _mckp_max_value(values, increments[rows, order], _DP_GRID)
        return order[np.arange(n), selection]

    def _run_dp_budget_search(self, max_weight_size, verbose=False):
        """Deterministic grid-approximate knapsack solve for budget mode (LP alternative)."""
        tables = self._stat_tables()
        n = len(tables.names)

        choice_idx = self._minimize_within_budget(
            tables.scores, tables.costs, float(max_weight_size)
        )
        if choice_idx is None:
            warn_rank_0(
                "AutoQuantize FAILED to find a solution! The searched model might not meet "
                "all constraints. "
            )
            # Best effort: each row's minimum true cost, ties toward the lower score.
            at_minimum = tables.costs <= np.where(
                np.isfinite(tables.costs), tables.costs, np.inf
            ).min(axis=1, keepdims=True)
            choice_idx = np.where(at_minimum, tables.scores, np.inf).argmin(axis=1)
        realized = float(tables.costs[np.arange(n), choice_idx].sum())
        is_satisfied = realized <= max_weight_size * (1 + 1e-12) + 1e-9
        if verbose:
            print_rank_0(
                f"AutoQuantize(dp): realized weight size {realized:.2f} "
                f"(target {max_weight_size:.2f}), satisfied={is_satisfied}"
            )
        return self._best_recipes_from_choice(tables.names, choice_idx), is_satisfied

    def _run_damage_bound_search(self, max_predicted_damage, verbose=False):
        """Minimize total weight cost subject to predicted damage <= ``max_predicted_damage``.

        Under the coverage link the bound maps to a score budget ``-log(1 - eps/c)``; score
        costs are rounded UP on the constraint axis so the quote is certified.
        """
        damage_model = getattr(self, "damage_model", None) or {}
        link = damage_model.get("link", self.config.get("damage_link", "additive"))
        tables = self._stat_tables()
        n = len(tables.names)

        # Not gated on the link: an invalidated fit cannot certify a bound under either one.
        if not damage_model.get("valid", False):
            warn_rank_0(
                "AutoQuantize FAILED to find a solution! The damage fit is invalid "
                f"(flags={damage_model.get('approximation_flags')}), so the damage bound "
                "cannot be certified. Returning the minimum-damage configuration. "
            )
            choice_idx = self._min_score_choices(tables.scores, tables.lengths)
            return self._best_recipes_from_choice(tables.names, choice_idx), False

        if link == "coverage" and "c" in damage_model:
            c = float(damage_model["c"])
            budget = (
                np.inf if max_predicted_damage >= c else -math.log(1.0 - max_predicted_damage / c)
            )
        else:
            budget = float(max_predicted_damage)

        if not np.isfinite(budget):
            choice_idx = np.zeros(n, dtype=int)
        else:
            choice_idx = (
                self._minimize_within_budget(tables.costs, tables.scores, budget)
                if budget > 0
                else None
            )
            if choice_idx is None:
                warn_rank_0(
                    "AutoQuantize FAILED to find a solution! Even the least aggressive "
                    "choices exceed the damage budget. "
                )
                choice_idx = self._min_score_choices(tables.scores, tables.lengths)

        total_score = float(tables.scores[np.arange(n), choice_idx].sum())
        is_satisfied = (not np.isfinite(budget)) or bool(total_score <= budget + 1e-12)
        if verbose:
            realized = float(tables.costs[np.arange(n), choice_idx].sum())
            print_rank_0(
                f"AutoQuantize(sla): score total {total_score:.4e} (budget {budget:.4e}), "
                f"weight size {realized:.2f}, satisfied={is_satisfied}"
            )
        return self._best_recipes_from_choice(tables.names, choice_idx), is_satisfied


AUTO_QUANTIZE_SEARCHERS[AutoQuantizeAumannShapleySearcher.method_name] = (
    AutoQuantizeAumannShapleySearcher
)
