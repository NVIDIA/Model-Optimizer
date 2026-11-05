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

"""Riemannian precision selection over calibrated, frozen AutoQuant candidates.

Algorithmic reference: https://github.com/IST-DASLab/RCO/tree/9a1e09c07d468109cbe60a1b87d5036034a79d10.
ModelOpt supplies the candidate quantizers, grouping, cost model and hard assignment solver.
Only allocation logits are trained; finalized models use the ordinary quantizers and export.
"""

import math
import types
from contextlib import ExitStack, contextmanager
from fnmatch import fnmatch
from itertools import islice
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modelopt.torch.opt.searcher import LPS
from modelopt.torch.opt.utils import named_hparams
from modelopt.torch.utils import print_rank_0

from .algorithms import (
    AUTO_QUANTIZE_SEARCHERS,
    QuantRecipe,
    QuantRecipeHparam,
    _AutoQuantizeBaseSearcher,
    _match_quantizer_cfg,
)
from .nn import TensorQuantizer

__all__ = ["AutoQuantizeRCOSearcher"]


class AutoQuantizeRCOSearcher(_AutoQuantizeBaseSearcher):
    """Select weight-only precisions using whole-model Riemannian constrained optimization.

    The default objective is causal teacher KL. ``loss_func`` can supply a task objective.
    ``checkpoint`` caches calibration, not allocation optimizer state. Searching a new budget
    requires the model and calibration data, rather than only additive sensitivity scores.
    """

    method_name = "rco"
    rco: dict[str, Any]
    method_config_keys = frozenset(
        {"steps", "num_samples", "lr", "temperature", "final_temperature", "seed", "kl_chunk_size"}
    )

    @property
    def default_search_config(self):
        """Return the reference shell's optimizer settings."""
        return {
            **super().default_search_config,
            "forward_step": None,
            "loss_func": None,
            "forward_backward_step": None,
            "steps": 200,
            "num_samples": 4,
            "lr": 0.1,
            "temperature": 1.0,
            "final_temperature": 0.01,
            "seed": 42,
            "kl_chunk_size": 128,
        }

    @property
    def default_state_dict(self):
        """Include joint-loss diagnostics separately from initialization scores."""
        return {**super().default_state_dict, "rco": {"history": []}}

    def sanitize_search_config(self, config):
        """Check optimizer settings before model conversion."""
        config = super().sanitize_search_config(config)
        for key in ("steps", "num_samples", "kl_chunk_size", "seed"):
            if isinstance(config[key], bool) or not isinstance(config[key], int):
                raise ValueError(f"RCO {key} must be an integer.")
            if config[key] < (0 if key == "seed" else 1):
                raise ValueError(f"RCO {key} is outside its valid range.")
        for key in ("lr", "temperature", "final_temperature"):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"RCO {key} must be a positive finite number.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"RCO {key} must be a positive finite number.")
        if config["num_score_steps"] < 1:
            raise ValueError("RCO requires positive num_score_steps.")
        if iter(config["data_loader"]) is config["data_loader"]:
            raise ValueError("RCO requires a replayable data_loader.")
        if config["forward_backward_step"] is not None:
            raise ValueError("RCO requires forward_step and optionally loss_func.")
        return config

    def validate_search_input(self, constraints, config):
        """Reject activation candidates before converting or calibrating the model."""
        formats = list(config["quantization_formats"])
        for space in config["module_search_spaces"]:
            formats.extend(space["quantization_formats"])
        if config["fixed_quantization_config"] is not None:
            formats.append(config["fixed_quantization_config"])
        for candidate in formats:
            recipe = QuantRecipe(candidate[0], name=candidate[1])
            if any(
                entry.get("parent_class") and entry["enable"] for entry in recipe.config.quant_cfg
            ):
                raise ValueError("RCO does not support enabling class-scoped quantizers.")
            for attr in ("input_quantizer", "output_quantizer"):
                _, enabled = _match_quantizer_cfg(recipe.config.quant_cfg, attr)
                if enabled:
                    raise ValueError("RCO supports weight-only quantization candidates.")

    def validate_model_input(self, model, config):
        """Refuse unsupported searched layouts while preserving protected model components."""
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise ValueError("RCO does not yet support distributed process groups.")
        disabled = config["disabled_layers"] or []
        disabled = [disabled] if isinstance(disabled, str) else disabled
        for name, module in model.named_modules():
            if any(fnmatch(name, pattern) for pattern in disabled):
                continue
            if (
                isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
                or self._is_auto_quantize_module(module)
            ) and (
                not isinstance(module, nn.Linear)
                or type(module.weight) is not nn.Parameter
                or module.weight.is_meta
            ):
                raise ValueError("RCO requires materialized dense Linear parameters.")

    def before_search(self):
        """Check converted layouts before calibrating candidate quantizers."""
        self.validate_model_input(self.model, self.config)
        super().before_search()

    def estimate_sensitivity_scores(self):
        """Initialize allocation logits from the reference normalized weight-delta MSE."""
        for _, hparam in named_hparams(self.model, unique=True):
            if not isinstance(hparam, QuantRecipeHparam):
                continue
            for recipe in hparam.choices:
                total = torch.zeros(())
                for module in hparam.quant_modules:
                    quantizers = hparam._all_quantizer_choices[recipe][module]
                    for attr, quantizer in quantizers.items():
                        for child in quantizer.modules():
                            if isinstance(child, TensorQuantizer) and (
                                ("weight" not in attr and child.is_enabled)
                                or child.pre_quant_scale is not None
                            ):
                                raise ValueError(
                                    "RCO requires weight-only candidates without pre-scales."
                                )
                    with torch.no_grad():
                        weight = cast("torch.Tensor", module.weight)
                        candidate = quantizers["weight_quantizer"](weight)
                        distortion = torch.zeros((), device=weight.device)
                        for start in range(0, weight.shape[0], 128):
                            delta = (
                                (
                                    candidate[start : start + 128].float()
                                    - weight[start : start + 128].float()
                                )
                                .to(weight.dtype)
                                .float()
                            )
                            distortion += delta.square().sum()
                        total += (distortion / weight.numel()).cpu()
                for score_module in hparam.score_modules:
                    hparam._importance_dict[recipe][score_module] = total / len(
                        hparam.score_modules
                    )

    def run_search(self):
        """Run allocation search and let AutoQuant apply the selected quantizer states."""
        super().run_search()
        self.best["score"] = self.rco["final_loss"]

    @torch.enable_grad()
    def run_search_with_stats(self, max_weight_size, verbose=False):
        """Optimize the joint objective; cached additive scores alone are insufficient."""
        if not hasattr(self, "model"):
            raise ValueError("RCO needs the model and calibration data to search a new budget.")
        self._hparams = {
            name: hparam
            for name, hparam in named_hparams(self.model, unique=True)
            if isinstance(hparam, QuantRecipeHparam)
        }
        batches = list(islice(self.config["data_loader"], self.config["num_score_steps"]))
        if not batches:
            raise ValueError("RCO calibration data is empty.")
        stats = list(self.candidate_stats.values())
        device = next(self.model.parameters()).device
        width = max(len(stat["formats"]) for stat in stats)
        self._costs = torch.zeros((len(stats), width), dtype=torch.float64, device=device)
        self._mask = torch.zeros_like(self._costs, dtype=torch.bool)
        distortion = torch.zeros_like(self._costs)
        denominator = self._get_total_weight_size_from_candidate_stats(self.candidate_stats)
        self._weights = torch.tensor(
            [stat["uncompressed_cost"] / denominator for stat in stats],
            device=device,
            dtype=torch.float64,
        )
        self._target = max_weight_size / denominator * 16
        for row, (name, stat) in enumerate(self.candidate_stats.items()):
            hparam = self._hparams[name]
            unweighted = hparam.get_cost(QuantRecipe(None), cost_weight=1.0)
            count = len(stat["formats"])
            self._costs[row, :count] = torch.tensor(
                [
                    hparam.get_cost(recipe, cost_weight=1.0) / unweighted * 16
                    for recipe in stat["formats"]
                ],
                device=device,
                dtype=self._costs.dtype,
            )
            distortion[row, :count] = torch.tensor(stat["raw_scores"], device=device)
            self._mask[row, :count] = True
        weighted_costs = self._costs * self._weights[:, None]
        minimum = weighted_costs.masked_fill(~self._mask, math.inf).min(-1, keepdim=True).values
        maximum = weighted_costs.masked_fill(~self._mask, -math.inf).max(-1, keepdim=True).values
        if self._target <= float(minimum.sum()) + 1e-10:
            self._mask &= weighted_costs == minimum
        elif self._target >= float(maximum.sum()) - 1e-10:
            self._mask &= weighted_costs == maximum
        self._alpha = nn.Parameter(
            (-torch.log(distortion + 1e-8)).masked_fill(~self._mask, -math.inf)
        )
        self._retract()
        optimizer = torch.optim.Adam([self._alpha], lr=self.config["lr"])
        devices = sorted({p.device.index for p in self.model.parameters() if p.is_cuda})
        self.rco = {
            "history": [],
            "objective": "task_loss" if self.config["loss_func"] else "teacher_kl",
        }
        with torch.random.fork_rng(devices=devices), self._interpolation():
            torch.manual_seed(self.config["seed"])
            steps = self.config["steps"] if (self._mask.sum(-1) > 1).any() else 0
            for step in range(steps):
                index = int(torch.randperm(len(batches))[0])
                batch = batches[index]
                teacher = self._teacher(batch)
                optimizer.zero_grad()
                temperature = self.config["temperature"] * (
                    self.config["final_temperature"] / self.config["temperature"]
                ) ** (step / max(steps - 1, 1))
                total_loss = 0.0
                for _ in range(self.config["num_samples"]):
                    uniform = torch.rand_like(self._alpha).clamp_(1e-20, 1 - 1e-15)
                    noisy = (self._alpha - (-uniform.log()).log()) / temperature
                    assignment = self._solve(noisy, max_weight_size)
                    soft = noisy.softmax(-1)
                    hard = F.one_hot(torch.tensor(assignment, device=device), width).to(soft)
                    self._coefficients = hard + (soft - soft.detach())
                    with torch.enable_grad():
                        loss = self._loss(batch, teacher)
                        (loss / self.config["num_samples"]).backward()
                    total_loss += float(loss.detach()) / self.config["num_samples"]
                if self._alpha.grad is None or not torch.isfinite(self._alpha.grad).all():
                    raise RuntimeError("RCO did not produce finite allocation gradients.")
                with torch.no_grad():
                    self._project(self._alpha.grad)
                optimizer.step()
                self._retract()
                with torch.no_grad():
                    self._project(optimizer.state[self._alpha]["exp_avg"])
                self.rco["history"].append(
                    {
                        "step": step + 1,
                        "loss": total_loss,
                        "expected_bits": float(self._expected_cost().detach()),
                    }
                )
                if verbose:
                    print_rank_0(f"RCO step {step + 1}/{steps}: loss={total_loss:.6g}")
            assignment = self._solve(self._alpha, max_weight_size)
            self._coefficients = F.one_hot(torch.tensor(assignment, device=device), width).to(
                self._alpha
            )
            with torch.no_grad():
                self.rco["final_loss"] = sum(
                    float(self._loss(batch, self._teacher(batch))) for batch in batches
                ) / len(batches)
        self.rco["logits"] = self._alpha.detach().cpu()
        self.rco["steps"] = steps
        return {
            name: {"format": stat["formats"][index], "costs": stat["costs"][index], "scores": 0.0}
            for (name, stat), index in zip(self.candidate_stats.items(), assignment, strict=True)
        }, True

    def _solve(self, logits, budget):
        values = logits.detach().cpu().tolist()
        masks = self._mask.cpu().tolist()
        choices = [
            [index for index in range(len(stat["formats"])) if mask[index]]
            for stat, mask in zip(self.candidate_stats.values(), masks, strict=True)
        ]
        solver = LPS(
            name="RCO",
            constraints={"cost": budget},
            constraints_to_candidate_costs={
                "cost": [
                    [stat["costs"][index] for index in row]
                    for stat, row in zip(self.candidate_stats.values(), choices, strict=True)
                ]
            },
            candidate_scores=[
                [values[row][index] for index in indices] for row, indices in enumerate(choices)
            ],
            objective_type="maximize",
        )
        # CBC's integer preprocessing can discard the optimum for signed allocation logits.
        solver.solver.options.append("preprocess off")
        selected, status = solver()
        assignment = [row[index] for row, index in zip(choices, selected, strict=True)]
        actual = math.fsum(
            stat["costs"][index]
            for stat, index in zip(self.candidate_stats.values(), assignment, strict=True)
        )
        if status != "Optimal" or actual > budget:
            raise RuntimeError("RCO hard assignment did not satisfy the declared budget.")
        return assignment

    def _expected_cost(self, alpha=None):
        probabilities = (self._alpha if alpha is None else alpha).softmax(-1)
        return (probabilities * self._costs * self._weights[:, None]).sum()

    @torch.no_grad()
    def _retract(self):
        lo, hi = -1.0, 1.0
        for _ in range(40):
            if float(self._expected_cost(self._alpha + lo * self._costs)) > self._target:
                lo *= 2
            elif float(self._expected_cost(self._alpha + hi * self._costs)) < self._target:
                hi *= 2
            else:
                break
        for _ in range(60):
            mid = (lo + hi) / 2
            if float(self._expected_cost(self._alpha + mid * self._costs)) > self._target:
                hi = mid
            else:
                lo = mid
        self._alpha.add_(((lo + hi) / 2) * self._costs)
        if not torch.isfinite(self._alpha[self._mask]).all():
            raise RuntimeError("RCO allocation logits became non-finite.")

    def _project(self, vector):
        probabilities = self._alpha.softmax(-1)
        normal = (
            probabilities
            * (self._costs - (probabilities * self._costs).sum(-1, keepdim=True))
            * self._weights[:, None]
        )
        vector.sub_((vector * normal).sum() / (normal.square().sum() + 1e-12) * normal)

    @contextmanager
    def _interpolation(self):
        with ExitStack() as stack:
            for module in self.model.modules():
                stack.callback(setattr, module, "training", module.training)
                module.training = False
                config = getattr(module, "config", None)
                if config is not None and hasattr(config, "use_cache"):
                    stack.callback(setattr, config, "use_cache", config.use_cache)
                    config.use_cache = False
            for parameter in self.model.parameters():
                stack.callback(parameter.requires_grad_, parameter.requires_grad)
                parameter.requires_grad_(False)
            self._reference_pass = False
            for row, name in enumerate(self.candidate_stats):
                hparam = self._hparams[name]
                stack.callback(setattr, hparam, "active", hparam.active)
                hparam.active = (
                    hparam.solver_choices[0]
                    if len(hparam.solver_choices) == 1
                    else QuantRecipe(None)
                )
                for module in hparam.quant_modules:
                    had_forward = "forward" in module.__dict__
                    original = module.__dict__.get("forward")
                    if had_forward:
                        stack.callback(setattr, module, "forward", original)
                    else:
                        stack.callback(module.__dict__.pop, "forward", None)

                    def forward(layer, input, hparam=hparam, row=row):
                        if self._reference_pass:
                            return F.linear(input, layer.weight, layer.bias)

                        def linear(input, coefficients):
                            reference = layer.weight.detach()
                            mixed = reference
                            for index, recipe in enumerate(hparam.solver_choices):
                                if recipe == QuantRecipe(None):
                                    continue
                                with torch.no_grad():
                                    candidate = hparam._all_quantizer_choices[recipe][layer][
                                        "weight_quantizer"
                                    ](reference)
                                    delta = candidate - reference
                                coefficient = coefficients[index].to(
                                    device=reference.device, dtype=reference.dtype
                                )
                                mixed = mixed + coefficient * delta
                            return F.linear(input, mixed, layer.bias)

                        coefficients = self._coefficients[row, : len(hparam.solver_choices)]
                        if torch.is_grad_enabled():
                            return checkpoint(linear, input, coefficients, use_reentrant=False)
                        return linear(input, coefficients)

                    module.forward = types.MethodType(forward, module)
            yield

    def _teacher(self, batch):
        if self.config["loss_func"] is not None:
            return None
        self._reference_pass = True
        try:
            with torch.no_grad():
                output = self.config["forward_step"](self.model, batch)
                return (output.logits if hasattr(output, "logits") else output).detach().cpu()
        finally:
            self._reference_pass = False

    def _loss(self, batch, teacher):
        output = self.config["forward_step"](self.model, batch)
        if self.config["loss_func"] is not None:
            return self.config["loss_func"](output, batch)
        logits = output.logits if hasattr(output, "logits") else output
        if logits.ndim != 3 or logits.shape[1] < 2:
            raise ValueError(
                "Default RCO KL requires [batch, sequence, vocabulary] logits with at least two positions."
            )
        mask = torch.ones(logits.shape[:2], device=logits.device, dtype=torch.bool)
        if isinstance(batch, dict):
            for key in ("attention_mask", "loss_mask"):
                if batch.get(key) is not None:
                    mask &= batch[key].to(mask.device).bool()
            if batch.get("labels") is not None:
                mask &= batch["labels"].to(mask.device) != -100
        mask = mask[:, 1:].reshape(-1)
        student = logits[:, :-1].reshape(-1, logits.shape[-1])
        reference = teacher[:, :-1].reshape_as(student)
        count = mask.sum()
        if not count:
            raise ValueError("RCO has no valid next-token targets.")

        def chunk_loss(student, reference, valid):
            reference = reference.to(student.device).float().log_softmax(-1)
            student = student.float().log_softmax(-1)
            return ((reference.exp() * (reference - student)).sum(-1) * valid).sum()

        loss = logits.new_zeros((), dtype=torch.float32)
        for start in range(0, student.shape[0], self.config["kl_chunk_size"]):
            end = start + self.config["kl_chunk_size"]
            inputs = (student[start:end], reference[start:end], mask[start:end])
            loss = loss + (
                checkpoint(chunk_loss, *inputs, use_reentrant=False)
                if torch.is_grad_enabled()
                else chunk_loss(*inputs)
            )
        return loss / count


AUTO_QUANTIZE_SEARCHERS[AutoQuantizeRCOSearcher.method_name] = AutoQuantizeRCOSearcher
