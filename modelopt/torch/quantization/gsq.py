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

"""NVFP4 adaptation of local-shift GSQ (https://arxiv.org/abs/2604.18556)."""

from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from modelopt.torch.utils import distributed as dist_utils

from .config import GSQCalibConfig, QuantizeAlgorithmConfig
from .nn import SequentialQuantizer, StaticBlockScaleQuantizer, TensorQuantizer
from .qtensor import NVFP4QTensor
from .utils import SHARED_PATTERNS, find_shared_input_groups, is_quantized_linear

if TYPE_CHECKING:
    from collections.abc import Callable

    from modelopt.torch.opt.mode import ConvertReturnType

__all__ = ["gsq"]


def gsq(model, forward_loop, **kwargs):
    """Refine supported dense NVFP4 decoders with staged local-shift reconstruction.

    The replay loop supplies initializer sequences followed by training sequences.
    Final hard weights and scales use ordinary ModelOpt save/restore and export.
    """
    # Deferred imports break the calibration registry/export dependency cycle.
    from modelopt.torch.export.quant_utils import preprocess_linear_fusion

    from .gsq_staged import StagedReplay, refine_staged
    from .mode import CalibrateModeRegistry, get_modelike_from_algo_cfg
    from .model_calib import max_calibrate

    config = GSQCalibConfig(**kwargs)
    if forward_loop is None:
        raise ValueError("GSQ requires a replayable forward_loop")
    modules = _validate_model(model)
    if not modules:
        return
    StagedReplay.validate(model, modules)
    mode, options = get_modelike_from_algo_cfg(config.initializer)[0]
    descriptor = CalibrateModeRegistry[mode]
    initializer = cast("QuantizeAlgorithmConfig", descriptor.config_class(**options))
    if initializer.layerwise.enable:
        raise ValueError("Set layerwise on GSQ, rather than on its initializer")
    if initializer.moe_calib_experts_ratio is not None:
        raise ValueError("GSQ does not support MoE calibration")
    device = modules[0][1].weight.device
    cuda_devices = [device.index] if device.type == "cuda" else []
    with _calibration_state(model) as originals, torch.random.fork_rng(devices=cuda_devices):
        torch.default_generator.manual_seed(config.seed)
        if cuda_devices:
            torch.cuda.default_generators[device.index].manual_seed(config.seed)
        replay = StagedReplay(model, forward_loop, config, originals)
        if any(not isinstance(m.weight_quantizer, StaticBlockScaleQuantizer) for _, m in modules):
            max_calibrate(model, forward_loop=None)
        initialize: Callable[..., ConvertReturnType] = descriptor.convert
        initialized_model, _ = initialize(model, initializer, forward_loop=replay.initialize)
        if initialized_model is not model:
            raise ValueError("GSQ initializer must calibrate the original model in place")
        for _, siblings in find_shared_input_groups(model, SHARED_PATTERNS):
            preprocess_linear_fusion(siblings)
            if len({id(m.weight_quantizer.global_amax) for m in siblings}) != 1:
                raise ValueError(
                    "GSQ initializer must share global scales for fusible NVFP4 siblings"
                )
        for _, module in modules:
            q = module.weight_quantizer
            if not isinstance(q, StaticBlockScaleQuantizer) or q._lsq:
                raise ValueError("GSQ initializer did not produce static NVFP4 scales")
        refine_staged(model, modules, replay, config)


def _validate_model(model):
    if dist_utils.is_distributed_launch() or (
        torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
    ):
        raise ValueError("GSQ currently requires single-process calibration")
    modules = []
    for name, module in model.named_modules():
        if not is_quantized_linear(module):
            continue
        q = module.weight_quantizer
        input_quantizer = module.input_quantizer
        if not q.is_enabled:
            continue
        if (
            not isinstance(module, nn.Linear)
            or isinstance(q, SequentialQuantizer)
            or not q.is_nvfp4_static
            or q.block_sizes[-1] != 16
            or q.block_sizes.get("four_over_six", False)
            or q.backend is not None
            or q.pre_quant_scale is not None
            or not isinstance(input_quantizer, TensorQuantizer)
            or input_quantizer.pre_quant_scale is not None
            or q.rotate_is_enabled
            or getattr(q, "_lsq", False)
            or not q.fake_quant
        ):
            raise ValueError("GSQ requires dense static block-16 NVFP4 weight-only quantizers")
        modules.append((name, module))
    allowed = {id(m.weight_quantizer) for _, m in modules}
    if any(
        isinstance(q, TensorQuantizer) and q.is_enabled and id(q) not in allowed
        for q in model.modules()
    ):
        raise ValueError("GSQ requires disabled activation, KV-cache, and fused expert quantizers")
    weights = [m.weight for _, m in modules]
    if any(isinstance(p, DTensor) or p.is_meta for p in weights):
        raise ValueError("GSQ requires resident, unsharded weights")
    if len({p.device for p in weights}) > 1 or any(hasattr(m, "_hf_hook") for m in model.modules()):
        raise ValueError("GSQ requires resident weights on one device; use layerwise calibration")
    parameter_uses = defaultdict(int)
    for _, parameter in model.named_parameters(remove_duplicate=False):
        parameter_uses[id(parameter)] += 1
    if any(parameter_uses[id(p)] > 1 for p in weights):
        raise ValueError("GSQ does not support tied quantized weights")
    if any(p.shape[-1] % 16 for p in weights):
        raise ValueError("GSQ weights must have input dimensions divisible by 16")
    return modules


@contextmanager
def _calibration_state(model):
    parameters = list(model.parameters())
    tensors = list(dict.fromkeys([*parameters, *model.buffers()]))
    originals = [(t, t.detach().to("cpu", copy=True)) for t in tensors]
    module_states = {
        m: (
            dict(m.__dict__),
            {
                key: dict(m.__dict__[key])
                for key in (
                    "_modules",
                    "_parameters",
                    "_buffers",
                    "_forward_hooks",
                    "_forward_pre_hooks",
                )
            },
        )
        for m in model.modules()
    }
    memo = {id(t): t for t in tensors}
    memo.update({id(m): m for m in model.modules()})
    quantizers = [
        (q, type(q), deepcopy(q.__dict__, memo))
        for q in model.modules()
        if isinstance(q, TensorQuantizer)
    ]
    training = {m: m.training for m in model.modules()}
    gradients = {name: p.requires_grad for name, p in model.named_parameters()}
    try:
        model.eval()
        for p in parameters:
            p.requires_grad_(False)
        yield {id(tensor): original for tensor, original in originals}
    except BaseException:
        with torch.no_grad():
            for tensor, original in originals:
                tensor.copy_(original)
        for module, (saved, containers) in module_states.items():
            # Hook handles and shared state retain references to these dictionaries.
            for key, contents in containers.items():
                saved[key].clear()
                saved[key].update(contents)
            module.__dict__.clear()
            module.__dict__.update(saved)
        for q, cls, state in quantizers:
            state.update({key: module_states[q][0][key] for key in module_states[q][1]})
            q.__class__ = cls
            q.__dict__.clear()
            q.__dict__.update(state)
        raise
    finally:
        for module, flag in training.items():
            module.training = flag
        for name, parameter in model.named_parameters():
            if name in gradients:
                parameter.requires_grad_(gradients[name])


class _LocalShift(nn.Module):
    grid: torch.Tensor
    center: torch.Tensor

    def __init__(self, module, global_scale, config):
        super().__init__()
        self.dtype = module.weight.dtype
        q = module.weight_quantizer
        block, _ = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(q, module.weight)
        self.log_block = nn.Parameter(block.float().clamp_min(2**-9).log())
        self.log_global = global_scale
        self.register_buffer(
            "grid",
            torch.tensor(
                [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6], device=block.device
            ),
        )
        scale = block.float() * global_scale.exp()
        hard = q(module.weight).detach().float().reshape(*scale.shape, 16)
        normalized = (hard / scale.unsqueeze(-1)).reshape_as(module.weight)
        upper = torch.searchsorted(self.grid, normalized.contiguous()).clamp_max(14)
        lower = (upper - 1).clamp_min(0)
        center = torch.where(
            (normalized - self.grid[lower]).abs() <= (normalized - self.grid[upper]).abs(),
            lower,
            upper,
        )
        self.register_buffer("center", center.to(torch.uint8))
        shifts = torch.arange(-2, 3, device=block.device, dtype=torch.float32)
        indices = torch.arange(len(self.grid), device=block.device).unsqueeze(-1) + shifts
        valid = (indices >= 0) & (indices < len(self.grid))
        prior = (-shifts.square() / 2).expand_as(indices)
        prior = prior - (prior * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True)
        logits = torch.randn((*center.shape, 5), device=block.device)
        for chunk, centers in zip(
            logits.reshape(-1, 5).split(262144), center.flatten().split(262144)
        ):
            chunk.add_(config.prior_strength * prior[centers.long()]).mul_(config.init_std)
        self.logits = nn.Parameter(logits)

    def scales(self):
        block = self.log_block.exp().clamp(2**-9, 448)
        projected = block.to(torch.float8_e4m3fn).float()
        return block + (projected - block).detach(), self.log_global.exp()

    def _codes(self, logits, center, temperature, kappa, hard):
        shifts = torch.arange(-2, 3, device=center.device, dtype=torch.int32)
        indices = center.to(torch.int32).unsqueeze(-1) + shifts
        valid = (indices >= 0) & (indices < len(self.grid))
        logits = logits.masked_fill(~valid, -1e9)
        if hard:
            return self.grid[indices.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)]
        values = self.grid[indices.clamp(0, len(self.grid) - 1)]
        probabilities = F.gumbel_softmax(kappa * logits, tau=temperature, dim=-1)
        return (values * probabilities).sum(-1)

    def forward(self, temperature=1.0, kappa=1.0, hard=False, initial=False):
        if initial:
            codes = self.grid[self.center.long()]
        else:
            # Bound candidate/gumbel temporaries independently of the linear's width.
            codes = torch.cat(
                [
                    checkpoint(
                        self._codes, logits, center, temperature, kappa, hard, use_reentrant=False
                    )
                    for logits, center in zip(
                        self.logits.reshape(-1, 5).split(262144),
                        self.center.flatten().split(262144),
                    )
                ]
            ).reshape_as(self.center)
        block, global_scale = self.scales()
        return (
            (codes.reshape(*block.shape, 16) * (block * global_scale).unsqueeze(-1))
            .reshape_as(self.center)
            .to(self.dtype)
        )

    @torch.no_grad()
    def snapshot(self, initial=False):
        block, global_scale = self.scales()
        return (
            self(hard=True, initial=initial).cpu(),
            (block * global_scale * 6).cpu(),
            (global_scale * (6 * 448)).cpu(),
        )


@torch.no_grad()
def _lion_step(groups, momentum, lr_multiplier):
    # Lion's sign update avoids Adam's second-moment normalization of saturated logits.
    for parameters, base_lr, weight_decay in groups:
        lr = base_lr * lr_multiplier
        for p in parameters:
            if p.grad is None:
                continue
            if p not in momentum:
                momentum[p] = torch.zeros_like(p)
            m = momentum[p]
            p.mul_(1 - lr * weight_decay)
            p.add_(m.mul(0.9).add_(p.grad, alpha=0.1).sign_(), alpha=-lr)
            m.lerp_(p.grad, 0.05)
