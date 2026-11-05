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

"""Sequence reconstruction stages for dense Llama NVFP4 GSQ calibration."""

from math import ceil, cos, pi

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F
from torch.utils._pytree import tree_map

from modelopt.torch.utils import print_rank_0

from .conversion import set_quantizer_by_cfg_context
from .qtensor import NVFP4QTensor
from .utils.calib_utils import update_hessian

__all__ = []


class StagedReplay:
    @staticmethod
    def validate(model, modules):
        # Structural lookalikes may differ in residuals, normalization, or RoPE.
        if (type(model).__name__, type(model).__module__) != (
            "LlamaDecoderLayer",
            "transformers.models.llama.modeling_llama",
        ):
            raise ValueError(
                "Staged GSQ requires a Llama decoder layer; enable layerwise calibration"
            )
        expected = {
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        }
        if {name for name, _ in modules} != expected:
            raise ValueError("Staged GSQ requires all seven dense Llama projections")
        if any(m.input_quantizer.pre_quant_scale is not None for _, m in modules):
            raise ValueError("Staged GSQ requires unscaled inputs")

    def __init__(self, model, forward_loop, config, originals=None):
        self.device = next(model.parameters()).device
        self.originals = {
            n: originals[id(p)] if originals is not None else p.detach().to("cpu", copy=True)
            for n, p in model.named_parameters()
        }
        self.samples = []
        cached_inputs = getattr(forward_loop, "_layerwise_cpu_inputs", None)

        def capture(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            if hidden.ndim != 3 or len(args) > 1:
                raise ValueError(
                    "Staged GSQ requires sequence inputs and keyword attention metadata"
                )
            if any(kwargs.get(k) is not None for k in ("past_key_values", "past_key_value")):
                raise ValueError("Staged GSQ requires use_cache=False")
            batch = hidden.shape[0]
            for i in range(batch):

                def copy(t):
                    if not isinstance(t, torch.Tensor):
                        return t
                    if t.ndim >= 2 and t.shape[0] == batch:
                        t = t[i : i + 1]
                    return t.detach().to("cpu", copy=cached_inputs is None)

                self.samples.append(tree_map(copy, (args, kwargs)))

        if cached_inputs is not None:
            for args, kwargs in cached_inputs:
                capture(model, args, kwargs)
        else:
            handle = model.register_forward_pre_hook(capture, with_kwargs=True)
            try:
                with (
                    torch.no_grad(),
                    set_quantizer_by_cfg_context(model, [{"quantizer_name": "*", "enable": False}]),
                ):
                    forward_loop(model)
            finally:
                handle.remove()
        initializer_count = config.initializer_samples
        required = initializer_count + config.train_samples
        if len(self.samples) != required:
            raise ValueError(
                f"Staged GSQ needs exactly {required} sequences, initializer first, then training"
            )
        self.initializer_samples = self.samples[:initializer_count]
        self.training_samples = self.samples[initializer_count:]

    def batch(self, samples):
        # Batches may have different masks or positions; concatenate only matching metadata.
        def merge(*values):
            first = values[0]
            if isinstance(first, torch.Tensor):
                if first.ndim >= 2 and first.shape[0] == 1:
                    return torch.cat(values).to(self.device)
                if not all(torch.equal(first, v) for v in values[1:]):
                    raise ValueError("Staged GSQ microbatch metadata must be compatible")
                return first.to(self.device)
            if not all(v == first for v in values[1:]):
                raise ValueError("Staged GSQ microbatch metadata must be compatible")
            return first

        return tree_map(merge, *samples)

    def initialize(self, model):
        for sample in self.initializer_samples:
            args, kwargs = self.batch([sample])
            model(*args, **kwargs)


class _AttentionRegion(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        self.self_attn = layer.self_attn

    def forward(self, hidden_states, **kwargs):
        return hidden_states + self.self_attn(self.input_layernorm(hidden_states), **kwargs)[0]


def refine_staged(model, modules, replay, config):
    # Local import avoids the calibration registry's gsq -> stages -> gsq cycle.
    from .gsq import _lion_step, _LocalShift

    named = dict(modules)
    for _, m in modules:
        m.weight_quantizer.disable_quant()

    def assignments_for(names):
        globals = {}
        members = [named[n] for n in names]
        for m in members:
            q = m.weight_quantizer
            key = id(q.global_amax)
            if key not in globals:
                scale = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(q)
                # A shared global scale cannot move while a sibling is fixed.
                trainable = all(
                    other in members
                    for _, other in modules
                    if id(other.weight_quantizer.global_amax) == key
                )
                globals[key] = nn.Parameter(scale.log(), requires_grad=trainable)
        # Initialization needs the quantizer's hard NVFP4 rounding once.
        for m in members:
            m.weight_quantizer.enable_quant()
        try:
            assignments = nn.ModuleList(
                [
                    _LocalShift(m, globals[id(m.weight_quantizer.global_amax)], config)
                    for m in members
                ]
            )
        finally:
            for m in members:
                m.weight_quantizer.disable_quant()
        return assignments

    def groups_for(assignments):
        return [
            ([a.logits for a in assignments], config.assignment_lr, config.weight_decay),
            (
                [
                    p
                    for n, p in assignments.named_parameters()
                    if not n.endswith("logits") and p.requires_grad
                ],
                config.scale_lr,
                0.0,
            ),
        ]

    @torch.no_grad()
    def install(names, assignments):
        for name, a in zip(names, assignments):
            weight, amax, global_amax = a.snapshot()
            m = named[name]
            if not torch.isfinite(weight).all():
                raise FloatingPointError("Nonfinite GSQ hard weights")
            m.weight.copy_(weight)
            m.weight_quantizer.amax = amax.to(replay.device).reshape_as(m.weight_quantizer.amax)
            m.weight_quantizer.global_amax = global_amax.to(replay.device).reshape_as(
                m.weight_quantizer.global_amax
            )

    def step(assignments, groups, momentum, multiplier):
        if any(
            not torch.isfinite(p.grad).all() for p in assignments.parameters() if p.grad is not None
        ):
            raise FloatingPointError("Nonfinite GSQ gradients")
        _lion_step(groups, momentum, multiplier)

    def schedule(index, total):
        progress = index / max(total - 1, 1)
        return (
            config.temperature + progress * (config.final_temperature - config.temperature),
            config.kappa + progress * (config.final_kappa - config.kappa),
            progress,
        )

    with torch.enable_grad():
        for name in ("self_attn.q_proj", "self_attn.k_proj"):
            module = named[name]
            hessian = torch.zeros(module.in_features, module.in_features, device=replay.device)
            count = 0
            with torch.no_grad():
                for sample in replay.initializer_samples:
                    args, kwargs = replay.batch([sample])
                    x = model.input_layernorm(args[0] if args else kwargs["hidden_states"])
                    hessian, count = update_hessian(x, hessian, count)
                # IST normalizes by sequences rather than tokens.
                hessian.mul_(count)
                hessian.div_(config.initializer_samples)
                dead = hessian.diagonal() == 0
                hessian.diagonal()[dead] = 1
                hessian.diagonal().add_(config.hessian_damping * hessian.diagonal().mean())
                factor = torch.linalg.cholesky(hessian).to(module.weight.dtype).float()
                target = replay.originals[name + ".weight"].to(replay.device, copy=True)
                target[:, dead] = 0
            del hessian
            assignments = assignments_for([name])
            groups, momentum = groups_for(assignments), {}
            for index in range(config.qk_steps):
                assignments.zero_grad(set_to_none=True)
                tau, kappa, _ = schedule(index, config.qk_steps)
                loss = (
                    ((target.float() - assignments[0](tau, kappa).float()) @ factor).square().sum()
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite GSQ Hessian reconstruction")
                loss.backward()
                step(assignments, groups, momentum, 1.0)
                if (index + 1) % 250 == 0 or index + 1 == config.qk_steps:
                    print_rank_0(
                        f"GSQ {name}: step {index + 1}/{config.qk_steps}, loss {loss.detach().item():.6g}"
                    )
            install([name], assignments)
            del assignments, groups, momentum, factor, target

        attention = _AttentionRegion(model)
        for region, names in [
            (attention, ["self_attn.v_proj", "self_attn.o_proj"]),
            (model, ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]),
        ]:
            assignments = assignments_for(names)
            groups, momentum = groups_for(assignments), {}
            originals = {
                n: p.to(replay.device)
                for n, p in replay.originals.items()
                if n in dict(region.named_parameters())
            }
            total = config.epochs * ceil(config.train_samples / config.batch_size)
            index = 0
            for _epoch in range(config.epochs):
                epoch_loss = torch.zeros((), device=replay.device)
                # Match upstream's shuffle of contiguous logical batch blocks.
                batches = torch.randperm(ceil(config.train_samples / config.batch_size)).tolist()
                for block in batches:
                    start = block * config.batch_size
                    end = min(start + config.batch_size, config.train_samples)
                    samples = replay.training_samples[start:end]
                    assignments.zero_grad(set_to_none=True)
                    tau, kappa, progress = schedule(index, total)
                    for micro in range(0, len(samples), config.microbatch_size):
                        subset = samples[micro : micro + config.microbatch_size]
                        args, kwargs = replay.batch(subset)
                        with (
                            torch.no_grad(),
                            set_quantizer_by_cfg_context(
                                region, [{"quantizer_name": "*", "enable": False}]
                            ),
                        ):
                            teacher = functional_call(region, originals, args, kwargs)
                            teacher = teacher[0] if isinstance(teacher, tuple) else teacher
                        weights = {n + ".weight": a(tau, kappa) for n, a in zip(names, assignments)}
                        student = functional_call(region, weights, args, kwargs)
                        student = student[0] if isinstance(student, tuple) else student
                        loss = F.mse_loss(student.float(), teacher.float())
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite GSQ sequence reconstruction")
                        (loss * len(subset) / (end - start)).backward()
                        epoch_loss += loss.detach() * len(subset) / config.train_samples
                    step(
                        assignments,
                        groups,
                        momentum,
                        0.1 + 0.45 * (1 + cos(pi * (progress if total > 1 else 1))),
                    )
                    index += 1
                print_rank_0(
                    f"GSQ {names}: epoch {_epoch + 1}/{config.epochs}, loss {epoch_loss.item():.6g}"
                )
            install(names, assignments)
            del assignments, groups, momentum, originals
