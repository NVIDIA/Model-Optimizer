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

"""Whole-model Riemannian allocation of frozen AutoQuant candidates."""

import math
import random
import types
from contextlib import ExitStack, contextmanager
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from torch.utils.checkpoint import checkpoint

from modelopt.torch.opt.searcher import LPS
from modelopt.torch.opt.utils import named_hparams
from modelopt.torch.utils import safe_load

from .algorithms import (
    AUTO_QUANTIZE_SEARCHERS,
    QuantRecipe,
    QuantRecipeHparam,
    _AutoQuantizeBaseSearcher,
)
from .nn import TensorQuantizer

__all__ = ["AutoQuantizeRCOSearcher"]


def _logits(output):
    return output.logits if hasattr(output, "logits") else output


def _rng_state():
    name, keys, *rest = np.random.get_state()
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "python": random.getstate(),
        "numpy": (name, np.asarray(keys).tolist(), *rest),
    }


def _set_rng(state):
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    name, keys, *rest = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), *rest))


class AutoQuantizeRCOSearcher(_AutoQuantizeBaseSearcher):
    """Optimize constrained hard allocations through native differentiable linear interpolation.

    ``steps``, ``lr``, ``temperature`` and ``final_temperature`` configure allocation Adam.
    ``eval_every`` controls full-set hard evaluation over ``num_score_steps`` batches.
    Without callbacks the objective is valid-token full-vocabulary teacher KL; task callbacks
    own their normalization. A supplied ``forward_backward_step`` performs backward itself
    and returns a detached scalar. Checkpoint reuse requires ``data_fingerprint`` and, for
    task callbacks, ``objective_fingerprint``. ``candidate_checkpoint`` reuses an independently
    fingerprinted complete calibration snapshot while keeping allocation checkpoints small.
    """

    method_name = "rco"
    rco_state: dict[str, Any]
    rco_signature: dict[str, Any] | None
    _preserve_candidate_weights = True
    method_config_keys = _AutoQuantizeBaseSearcher.method_config_keys | frozenset(
        {
            "steps",
            "lr",
            "temperature",
            "final_temperature",
            "seed",
            "checkpoint_every",
            "checkpoint_linear",
            "candidate_checkpoint",
            "objective_fingerprint",
            "eval_every",
        }
    )

    @property
    def default_search_config(self):
        """Return the native RCO schedule and callback options."""
        return {
            **super().default_search_config,
            "forward_step": None,
            "loss_func": None,
            "forward_backward_step": None,
            "steps": 100,
            "lr": 0.05,
            "temperature": 1.0,
            "final_temperature": 0.1,
            "seed": 0,
            "checkpoint_every": 1,
            "checkpoint_linear": True,
            "candidate_checkpoint": None,
            "objective_fingerprint": None,
            "eval_every": 10,
        }

    @property
    def default_state_dict(self):
        """Extend ordinary candidate checkpoints with allocation state."""
        return {**super().default_state_dict, "rco_state": {}, "rco_signature": None}

    def sanitize_search_config(self, config):
        """Validate the declared schedule and replayable evaluation data."""
        config = super().sanitize_search_config(config)
        for key in ("steps", "seed", "checkpoint_every", "eval_every"):
            if (
                isinstance(config[key], bool)
                or not isinstance(config[key], int)
                or config[key] < (1 if key in {"checkpoint_every", "eval_every"} else 0)
            ):
                raise ValueError(f"Invalid RCO {key}.")
        for key in ("lr", "temperature", "final_temperature"):
            if (
                not isinstance(config[key], (float, int))
                or not math.isfinite(config[key])
                or config[key] <= 0
            ):
                raise ValueError(f"RCO {key} must be finite and positive.")
        if not isinstance(config["checkpoint_linear"], bool) or config["num_score_steps"] < 1:
            raise ValueError("RCO needs positive evaluation steps and a boolean checkpoint_linear.")
        if iter(config["data_loader"]) is config["data_loader"]:
            raise ValueError("RCO requires a replayable data_loader.")
        return config

    def validate_search_input(self, constraints, config):
        """Require stable objective/data identity before creating resumable allocation state."""
        super().validate_search_input(constraints, config)
        if config["checkpoint"] or config["candidate_checkpoint"]:
            if not config["data_fingerprint"]:
                raise ValueError("RCO checkpoint reuse requires data_fingerprint.")
            if (config["loss_func"] or config["forward_backward_step"]) and not config[
                "objective_fingerprint"
            ]:
                raise ValueError(
                    "RCO task callbacks require objective_fingerprint for checkpoint reuse."
                )
        if (
            config["checkpoint"]
            and config["candidate_checkpoint"]
            and Path(config["checkpoint"]).resolve()
            == Path(config["candidate_checkpoint"]).resolve()
        ):
            raise ValueError("Candidate and allocation checkpoint paths must differ.")

    def before_search(self):
        """Validate the complete optimization signature before candidate installation."""
        self._entry_flags = {module: module.training for module in self.model.modules()}
        signature = {
            key: self.config[key]
            for key in (
                "steps",
                "lr",
                "temperature",
                "final_temperature",
                "seed",
                "eval_every",
                "data_fingerprint",
                "objective_fingerprint",
                "num_score_steps",
                "num_calib_steps",
            )
        }
        signature.update(
            version=1,
            constraints=self.constraints,
            callback="backward"
            if self.config["forward_backward_step"]
            else "loss"
            if self.config["loss_func"]
            else "kl",
        )
        if (self.rco_state and self.rco_signature is None) or (
            self.rco_signature is not None and self.rco_signature != signature
        ):
            raise ValueError("RCO checkpoint budget, objective, data, or schedule does not match.")
        self.rco_signature = signature
        configs = [getattr(module, "config", None) for module in self.model.modules()]
        configs += [getattr(config, "text_config", None) for config in configs]
        self._entry_caches = [
            (config, config.use_cache)
            for config in configs
            if config is not None and hasattr(config, "use_cache")
        ]
        for config, _ in self._entry_caches:
            config.use_cache = False
        super().before_search()

    def _restore_execution_state(self):
        for module, training in getattr(self, "_entry_flags", {}).items():
            module.training = training
        for config, use_cache in getattr(self, "_entry_caches", []):
            config.use_cache = use_cache

    def _verify_resolved_constraint(self, hparams):
        super()._verify_resolved_constraint(hparams)
        denominator = math.fsum(h.get_cost(QuantRecipe(None)) for h in hparams)
        minimum = math.fsum(min(h.get_cost(r) for r in h.solver_choices) for h in hparams)
        if minimum > denominator * self.constraints["effective_bits"] / 16:
            raise ValueError("RCO budget is below the exact minimum candidate cost.")

    def _get_formatted_weight_compression_constraint(self):
        target = self.constraints["effective_bits"]
        if (
            isinstance(target, bool)
            or not isinstance(target, (float, int))
            or not math.isfinite(target)
            or target <= 0
        ):
            raise ValueError("RCO effective_bits must be finite and positive.")
        return target / 16

    def load_search_checkpoint(self):
        """Load secure optimizer state and an independently fingerprinted candidate snapshot."""
        path = self._get_checkpoint_path()
        restored = bool(path and Path(path).is_file())
        if path and restored:
            saved = safe_load(path, weights_only=True, map_location="cpu")
            for key, default in self.default_state_dict.items():
                setattr(self, key, saved.get(key, default))
        source = self.config["candidate_checkpoint"]
        if source:
            saved = safe_load(source, weights_only=True, map_location="cpu")
            if not saved.get("candidate_stats"):
                raise ValueError("Candidate checkpoint must record complete candidate preparation.")
            if (
                saved.get("candidate_weights_version") != 1
                or not saved.get("candidate_model_fingerprint")
                or saved.get("candidate_data_fingerprint") != self.config["data_fingerprint"]
            ):
                raise ValueError("Candidate checkpoint lacks matching model/data fingerprints.")
            expected = set(saved.get("quantization_formats_signature") or ())
            for space in saved.get("module_search_space_signature") or ():
                expected.update(space[1])
            if saved.get("fixed_quantization_config_signature"):
                expected.add(saved["fixed_quantization_config_signature"])
            expected.discard(QuantRecipe(None).checkpoint_signature)
            if expected != {recipe.checkpoint_signature for recipe in saved["quantizer_states"]}:
                raise ValueError("Candidate checkpoint must contain every configured calibration.")
            for key in super().default_state_dict:
                if key not in {"method", "best", "candidate_stats"}:
                    setattr(self, key, saved.get(key))
        return restored

    def save_search_checkpoint(self, verbose=False):
        """Keep allocation checkpoints small when candidates have an independent snapshot."""
        weights = self.quantizer_states
        if self.config["candidate_checkpoint"]:
            self.quantizer_states = {}
        try:
            super().save_search_checkpoint(verbose)
        finally:
            self.quantizer_states = weights

    def _snapshot(self, optimizer, step, assignment, score):
        self.rco_state.update(
            step=step,
            alpha=self._alpha.detach().to("cpu", copy=True),
            optimizer=tree_map(
                lambda value: (
                    value.detach().to("cpu", copy=True)
                    if isinstance(value, torch.Tensor)
                    else value
                ),
                optimizer.state_dict(),
            ),
            best_assignment=assignment,
            best_score=score,
            rng=_rng_state(),
            sample_cursor=step % len(self._batches),
        )

    def _install_assignment(self, assignment):
        cost = math.fsum(
            h.get_cost(h.solver_choices[index]) for h, index in zip(self._hparams, assignment)
        )
        if cost > self._ceiling:
            raise RuntimeError("Selected RCO assignment exceeds the exact declared ceiling.")
        for hparam, index in zip(self._hparams, assignment):
            hparam.active = hparam.solver_choices[index]
        QuantRecipe.fold_pqs_to_weights(self.model)
        return cost / self._denominator * 16

    def estimate_sensitivity_scores(self):
        """Populate ordinary cost metadata; RCO never uses additive sensitivity scores."""
        for _, hparam in named_hparams(self.model, unique=True):
            if isinstance(hparam, QuantRecipeHparam):
                for values in hparam._importance_dict.values():
                    values.update(dict.fromkeys(values, torch.tensor(0.0)))

    def run_search_with_stats(self, max_weight_size, verbose=False):
        """Reject offline additive-score solving for a whole-model objective."""
        raise ValueError("RCO allocations require the original model and frozen candidates.")

    @contextmanager
    def _interpolation(self):
        with ExitStack() as stack:
            self.model.eval()
            for hparam in self._hparams:
                hparam._apply_quantizer_choice(
                    hparam.solver_choices[0]
                    if len(hparam.solver_choices) == 1
                    else QuantRecipe(None)
                )
            for parameter in self.model.parameters():
                stack.callback(parameter.requires_grad_, parameter.requires_grad)
                parameter.requires_grad_(False)
            for row, hparam in enumerate(self._hparams):
                for module in hparam.quant_modules:
                    for choices in hparam._all_quantizer_choices.values():
                        for attr, quantizer in choices[module].items():
                            if "weight" not in attr and any(
                                isinstance(q, TensorQuantizer) and q.is_enabled
                                for q in quantizer.modules()
                            ):
                                raise ValueError("RCO supports weight-only candidates.")
                    if len(hparam.solver_choices) == 1:
                        continue
                    weights = [
                        hparam.reference_weights[module]
                        if recipe.is_no_quant
                        else hparam.weight_candidates[recipe][module]
                        for recipe in hparam.solver_choices
                    ]
                    had_forward, forward = (
                        "forward" in module.__dict__,
                        module.__dict__.get("forward"),
                    )
                    stack.callback(
                        setattr, module, "forward", forward
                    ) if had_forward else stack.callback(module.__dict__.pop, "forward", None)

                    def mixed_forward(layer, inputs, weights=weights, row=row):
                        def linear(inputs, coefficients):
                            mixed = torch.zeros_like(layer.weight)
                            for coefficient, weight in zip(coefficients, weights):
                                mixed = mixed + coefficient.to(inputs.dtype) * weight.to(
                                    inputs.device
                                )
                            return F.linear(inputs, mixed, layer.bias)

                        coefficients = self._coefficients[row, : len(weights)]
                        if self.config["checkpoint_linear"] and torch.is_grad_enabled():
                            return checkpoint(linear, inputs, coefficients, use_reentrant=False)
                        return linear(inputs, coefficients)

                    module.forward = types.MethodType(mixed_forward, module)
            yield

    def _loss(self, batch, index, backward=False):
        callback = self.config["forward_backward_step"]
        if callback is not None:
            metric = callback(self.model, batch)
            if isinstance(metric, torch.Tensor) and (metric.ndim or metric.requires_grad):
                raise ValueError("forward_backward_step must return a detached scalar metric.")
            return float(metric), 1
        output = self.config["forward_step"](self.model, batch)
        if self.config["loss_func"] is not None:
            loss, count = self.config["loss_func"](output, batch), 1
        else:
            student = _logits(output).float().log_softmax(-1)
            teacher = self._teachers[index].to(student.device).float().log_softmax(-1)
            token_loss = (teacher.exp() * (teacher - student)).sum(-1)
            valid = torch.ones_like(token_loss)
            for name in ("attention_mask", "loss_mask"):
                mask = batch.get(name) if isinstance(batch, dict) else None
                if mask is not None:
                    if mask.shape != token_loss.shape:
                        raise ValueError("RCO token masks must align with returned logits.")
                    valid.mul_(mask.to(valid))
            count = float(valid.sum())
            loss = (token_loss * valid).sum() / valid.sum().clamp_min(1)
        if not isinstance(loss, torch.Tensor) or loss.ndim:
            raise ValueError("RCO loss must be a scalar Tensor.")
        if backward:
            loss.backward()
        return float(loss.detach()), count

    def _evaluate(self, assignment):
        hard = F.one_hot(
            torch.tensor(assignment, device=self._alpha.device), self._alpha.shape[1]
        ).double()
        total, count = 0.0, 0.0
        for index, batch in enumerate(self._batches):
            self._coefficients = hard.detach().requires_grad_(
                self.config["forward_backward_step"] is not None
            )
            with torch.set_grad_enabled(self.config["forward_backward_step"] is not None):
                metric, weight = self._loss(batch, index)
            total, count = total + metric * weight, count + weight
        if count <= 0 or not math.isfinite(total):
            raise ValueError("RCO evaluation has no valid finite objective.")
        return total / count

    def _solve(self, scores):
        allowed = [mask.nonzero().flatten().tolist() for mask in self._mask]
        selections, status = LPS(
            "rco",
            {"weight": self._ceiling / self._denominator * 16},
            {"weight": [row[indices].tolist() for row, indices in zip(self._costs, allowed)]},
            [row[indices].detach().tolist() for row, indices in zip(scores, allowed)],
            objective_type="maximize",
        )()
        selections = [indices[index] for indices, index in zip(allowed, selections)]
        cost = math.fsum(
            item["costs"][index] for item, index in zip(self.candidate_stats.values(), selections)
        )
        if status != "Optimal" or cost > self._ceiling:
            raise RuntimeError(
                f"Uncertified RCO allocation: status={status}, cost={cost}, ceiling={self._ceiling}."
            )
        self.rco_state["solver_calls"] += 1
        return [int(index) for index in selections], cost / self._denominator * 16

    def _probabilities(self):
        return self._alpha.masked_fill(~self._mask, -1e100).softmax(-1)

    def _project(self, vector):
        probabilities = self._probabilities()
        normal = probabilities * (self._costs - (probabilities * self._costs).sum(-1, keepdim=True))
        if self._active_constraint:
            vector.sub_(normal * (vector * normal).sum() / normal.square().sum().clamp_min(1e-100))

    @torch.no_grad()
    def _retract(self):
        if not self._active_constraint:
            return
        base = self._alpha.masked_fill(~self._mask, -1e100)

        def expected(shift):
            return ((base + shift * self._costs).softmax(-1) * self._costs).sum()

        low, high = self._alpha.new_tensor(-1), self._alpha.new_tensor(1)
        for _ in range(32):
            low = torch.where(expected(low) > self._target, low * 2, low)
            high = torch.where(expected(high) < self._target, high * 2, high)
        for _ in range(64):
            middle = (low + high) / 2
            above = expected(middle) > self._target
            low, high = torch.where(above, low, middle), torch.where(above, middle, high)
        self._alpha.add_((low + high) / 2 * self._costs)
        if abs(float((self._probabilities() * self._costs).sum()) - self._target) > 1e-8:
            raise RuntimeError("RCO budget retraction did not converge.")

    def run_search(self):
        """Optimize and materialize the best feasible full-evaluation allocation."""
        state = _rng_state()
        try:
            self._run_search()
        finally:
            _set_rng(state)
            self._restore_execution_state()

    def _run_search(self):
        self._hparams = [
            h for _, h in named_hparams(self.model, unique=True) if isinstance(h, QuantRecipeHparam)
        ]
        self._batches = list(islice(self.config["data_loader"], self.config["num_score_steps"]))
        if not self._batches:
            raise ValueError("RCO evaluation set is empty.")
        device = next(self.model.parameters()).device
        stats = list(self.candidate_stats.values())
        self._denominator = math.fsum(item["uncompressed_cost"] for item in stats)
        self._target = self.constraints["effective_bits"]
        self._ceiling = self._denominator * self._target / 16
        if self.rco_state.get("completed"):
            self._install_assignment(self.best["assignment"])
            return
        width = max(len(h.solver_choices) for h in self._hparams)
        optimization_steps = self.config["steps"] if width > 1 else 0
        self._costs = torch.zeros(len(stats), width, dtype=torch.float64, device=device)
        self._mask = torch.zeros_like(self._costs, dtype=torch.bool)
        for row, item in enumerate(stats):
            self._costs[row, : len(item["costs"])] = (
                torch.tensor(item["costs"], device=device, dtype=torch.float64)
                / self._denominator
                * 16
            )
            self._mask[row, : len(item["costs"])] = True
        minimum = sum(min(item["costs"]) for item in stats) / self._denominator * 16
        maximum = sum(max(item["costs"]) for item in stats) / self._denominator * 16
        self._active_constraint = minimum < self._target < maximum
        if self._target == minimum:
            self._mask &= (
                self._costs
                == self._costs.masked_fill(~self._mask, math.inf).min(-1, keepdim=True).values
            )
        self._alpha = nn.Parameter(torch.zeros_like(self._costs))
        self._retract()
        optimizer = torch.optim.Adam([self._alpha], lr=self.config["lr"])
        restored = "alpha" in self.rco_state
        if restored:
            with torch.no_grad():
                self._alpha.copy_(self.rco_state["alpha"].to(device))
            optimizer.load_state_dict(self.rco_state["optimizer"])
        else:
            self.rco_state = {"history": [], "solver_calls": 0, "step": 0}
        self.rco_state["effective_optimization_steps"] = optimization_steps
        torch.manual_seed(self.config["seed"])
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"] % 2**32)
        for hparam in self._hparams:
            hparam._apply_quantizer_choice(QuantRecipe(None))
        self.model.eval()
        if self.config["loss_func"] is None and self.config["forward_backward_step"] is None:
            with torch.no_grad():
                self._teachers = [
                    _logits(self.config["forward_step"](self.model, batch)).detach().cpu()
                    for batch in self._batches
                ]
        best_assignment = [
            min(range(len(item["costs"])), key=item["costs"].__getitem__) for item in stats
        ]
        if restored:
            _set_rng(self.rco_state["rng"])
            best_assignment = self.rco_state["best_assignment"]
        with self._interpolation():
            best_score = (
                self.rco_state["best_score"] if restored else self._evaluate(best_assignment)
            )
            for step in range(self.rco_state["step"], optimization_steps):
                optimizer.zero_grad()
                temperature = self.config["temperature"] * (
                    self.config["final_temperature"] / self.config["temperature"]
                ) ** (step / max(1, self.config["steps"] - 1))
                noise = -torch.rand_like(self._alpha).clamp_(1e-15, 1 - 1e-15).log().neg().log()
                noisy = (self._alpha + noise).masked_fill(~self._mask, -1e100) / temperature
                assignment, sampled = self._solve(noisy)
                soft = noisy.softmax(-1)
                hard = F.one_hot(torch.tensor(assignment, device=device), width).double()
                self._coefficients = hard + (soft - soft.detach())
                index = step % len(self._batches)
                self._loss(self._batches[index], index, backward=True)
                if self._alpha.grad is None or not torch.isfinite(self._alpha.grad).all():
                    raise RuntimeError("RCO backward did not produce finite assignment gradients.")
                gradient_norm = float(self._alpha.grad.norm())
                with torch.no_grad():
                    self._project(self._alpha.grad)
                projected = float(self._alpha.grad.norm())
                optimizer.step()
                if not torch.isfinite(self._alpha).all():
                    raise RuntimeError("RCO optimizer produced non-finite assignment logits.")
                self._retract()
                with torch.no_grad():
                    self._project(optimizer.state[self._alpha]["exp_avg"])
                metric = (
                    self._evaluate(assignment)
                    if (step + 1) % self.config["eval_every"] == 0
                    or step + 1 == self.config["steps"]
                    else None
                )
                if metric is not None and metric < best_score:
                    best_assignment, best_score = assignment, metric
                expected = float((self._probabilities() * self._costs).sum().detach())
                self.rco_state["history"].append(
                    {
                        "sampled_effective_bits": sampled,
                        "expected_effective_bits": expected,
                        "gradient_norm": gradient_norm,
                        "projected_gradient_norm": projected,
                        "score": metric,
                    }
                )
                self._snapshot(optimizer, step + 1, best_assignment, best_score)
                if (step + 1) % self.config["checkpoint_every"] == 0 or step + 1 == self.config[
                    "steps"
                ]:
                    self.save_search_checkpoint()
            final_scores = self._alpha.masked_fill(~self._mask, -1e100).log_softmax(-1)
            final_assignment, _ = self._solve(final_scores)
            final_score = self._evaluate(final_assignment)
            if final_score < best_score:
                best_assignment, best_score = final_assignment, final_score
        self._snapshot(optimizer, optimization_steps, best_assignment, best_score)
        self.rco_state.update(
            final_scores=[
                row[: len(h.solver_choices)].detach().tolist()
                for row, h in zip(final_scores, self._hparams)
            ],
            final_assignment=final_assignment,
        )
        achieved = self._install_assignment(best_assignment)
        self.best = {
            "recipe": {
                name: item["formats"][index]
                for (name, item), index in zip(self.candidate_stats.items(), best_assignment)
            },
            "constraints": {"effective_bits": achieved},
            "score": best_score,
            "is_satisfied": True,
            "requested_effective_bits": self._target,
            "continuous_expected_effective_bits": float(
                (self._probabilities() * self._costs).sum().detach()
            ),
            "assignment": best_assignment,
        }
        self.rco_state["completed"] = True
        self.save_search_checkpoint()


AUTO_QUANTIZE_SEARCHERS[AutoQuantizeRCOSearcher.method_name] = AutoQuantizeRCOSearcher
