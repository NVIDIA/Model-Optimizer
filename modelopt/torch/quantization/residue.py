# Adapted from ResInfer (algo.py / algo_helper.py):
# https://github.com/ShawRong/resinfer/tree/51b906836a69fa43915040314df57bc9f23030ad/quant/modelopt_integration
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-FileCopyrightText: Copyright (c) 2023 MIT HAN Lab (get_act_scale)
# SPDX-License-Identifier: Apache-2.0 AND MIT

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

"""Calibration for paired activation-residual and copied-weight QDQ."""

from collections import Counter
from contextlib import ExitStack, contextmanager
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from modelopt.torch.utils.network import bind_forward_method, unpatch_forward_method

from .config import ResidueMaxCalibConfig
from .conversion import create_and_replace_residue_linear_on_the_fly, set_quantizer_by_cfg_context
from .model_calib import max_calibrate
from .nn import ResidueTensorQuantizer, TensorQuantizer

__all__ = ["residue_max_calibrate"]


def get_salient_indices(act_scale_mean: torch.Tensor, block_size: int, ratio: float):
    """Select the largest scores in each block, excluding padded channels."""
    width = act_scale_mean.shape[-1]
    act_scale_mean = F.pad(act_scale_mean, (0, -width % block_size), value=-float("inf"))
    act_scale_mean = act_scale_mean.contiguous().view(-1, block_size)
    channels_per_block = int(ratio * block_size)
    _, local_indices = torch.topk(act_scale_mean, k=channels_per_block, dim=1, largest=True)
    # Convert local indices to global indices by adding each block's offset.
    block_indices = torch.arange(act_scale_mean.shape[0], device=act_scale_mean.device).unsqueeze(1)
    salient_indices = (local_indices + block_indices * block_size).flatten().sort().values
    return salient_indices[salient_indices < width]


def get_nvfp4_error_weighted_score(accumulated_error_sq, weight):
    """Return activation-error norm times weight-column norm for each input channel."""
    weight_col_norm = weight.detach().float().square().sum(dim=0).sqrt()
    return accumulated_error_sq.float().clamp_min(0).sqrt() * weight_col_norm


class ResidueHelper:
    """Collect the donor's raw-input salience statistics without quantized Linear output."""

    def __init__(self, module, salience_metric):
        """Initialize per-layer statistics."""
        self.module = module
        self.salience_metric = salience_metric
        self.act_scale = 0.0
        self.num_cache_steps = 0
        self.is_input_quantized = module.input_quantizer.is_enabled
        self._saved_awq_amax = module.input_quantizer.amax

    def setup(self):
        """Temporarily collect raw-input statistics with the weight quantizer disabled."""
        module = self.module
        self._had_instance_forward = "forward" in module.__dict__
        bind_forward_method(module, customized_forward, "_forward_no_residuemax")
        module.input_quantizer.disable()
        # Preserve an existing AWQ amax; rank channels using raw activations.
        module.input_quantizer.reset_amax()
        module.input_quantizer._calibrator.reset()
        module.input_quantizer.axis = -1

    def cleanup(self):
        """Restore the original forward method."""
        unpatch_forward_method(self.module, "_forward_no_residuemax")
        if not self._had_instance_forward:
            del self.module.forward


def customized_forward(self, input, *args, **kwargs):
    """Collect salience while forwarding the unquantized layer output."""
    helper = self.residue_max
    weight_enabled = self.weight_quantizer.is_enabled
    salient_enabled = getattr(self.input_quantizer, "_enable_salient_indices", True)
    self.weight_quantizer.disable()
    if isinstance(self.input_quantizer, ResidueTensorQuantizer):
        self.input_quantizer._enable_salient_indices = False
    try:
        out_actual = self._forward_no_residuemax(input, *args, **kwargs)
    finally:
        self.weight_quantizer.enable() if weight_enabled else self.weight_quantizer.disable()
        if isinstance(self.input_quantizer, ResidueTensorQuantizer):
            self.input_quantizer._enable_salient_indices = salient_enabled

    if input.numel() == 0:
        return out_actual

    # Salient channel ranking reflects raw activations, not the AWQ-smoothed path.
    if helper.salience_metric == "nvfp4_error_weighted":
        with ExitStack() as stack:
            if isinstance(self.input_quantizer, ResidueTensorQuantizer):
                stack.enter_context(self.input_quantizer.allow_residue_linear_fake_quant())
            # The public quantizer is in pass-through mode during statistics collection.
            quantized_input = self.input_quantizer._fake_quantize(input)
        flat_error = (input.float() - quantized_input.float()).reshape(-1, input.shape[-1])
        helper.act_scale += flat_error.square().sum(dim=0)
    else:
        # Average each batch first, matching the donor and AWQ activation statistics.
        helper.act_scale += input.abs().contiguous().view(-1, input.shape[-1]).mean(0).float()
    helper.num_cache_steps += 1
    if helper.is_input_quantized and helper._saved_awq_amax is None:
        with set_quantizer_by_cfg_context(
            self.input_quantizer, [{"quantizer_name": "*", "enable": True}]
        ):
            max_calibrate(
                self.input_quantizer, lambda quantizer: quantizer(input), distributed_sync=False
            )
    return out_actual


@contextmanager
def _calibration_state(model):
    quantizers = [q for q in model.modules() if isinstance(q, TensorQuantizer)]
    states = [
        (
            q._disabled,
            q._if_quant,
            q._if_calib,
            q.axis,
            q.amax.detach().clone() if q.amax is not None else None,
            deepcopy(q._calibrator),
        )
        for q in quantizers
    ]
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        for q, (disabled, quant, calib, axis, amax, calibrator) in zip(quantizers, states):
            q._disabled, q._if_quant, q._if_calib = disabled, quant, calib
            q.axis = axis
            q._calibrator = calibrator
            if not succeeded:
                q.reset_amax()
                if amax is not None:
                    q.amax = amax


def _validate_modules(model, salience_metric):
    if dist.is_initialized() and dist.get_world_size() > 1:
        raise NotImplementedError("Residue calibration currently supports single-process models.")
    if any(getattr(getattr(m, "_hf_hook", None), "offload", False) for m in model.modules()):
        raise NotImplementedError("Residue calibration requires resident weights.")
    names = []
    seen_quantizers = set()
    parameter_ids = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))
    for name, module in model.named_modules():
        if not hasattr(module, "weight_quantizer") or not module.weight_quantizer.is_enabled:
            continue
        if not isinstance(module, nn.Linear) or hasattr(module, "_original_in_features"):
            raise NotImplementedError(
                "Residue calibration requires unextended dense Linear modules."
            )
        if module.weight.is_meta or hasattr(module.weight, "to_local"):
            raise NotImplementedError("Residue calibration requires resident, unsharded weights.")
        if parameter_ids[id(module.weight)] > 1:
            raise NotImplementedError("Residue calibration does not yet support tied weights.")
        formats = []
        for q in (module.input_quantizer, module.weight_quantizer):
            if type(q) not in (TensorQuantizer, ResidueTensorQuantizer) or id(q) in seen_quantizers:
                raise NotImplementedError(
                    "Residue requires independent, non-sequential quantizers."
                )
            seen_quantizers.add(id(q))
            if getattr(q, "_shared_quant_tied_attrs", None):
                raise NotImplementedError(
                    "Residue calibration does not yet support tied quantizer state."
                )
            blocks = q.block_sizes or {}
            scale_bits = blocks.get("scale_bits")
            if (
                not q.is_enabled
                or not q.fake_quant
                or q.num_bits != (2, 1)
                or scale_bits not in ((4, 3), (8, 0))
                or blocks.get("type") != "dynamic"
                or blocks.get(-1) != (16 if scale_bits == (4, 3) else 32)
                or q.axis is not None
                or q.rotate_is_enabled
                or q.bias is not None
                or q.backend is not None
            ):
                raise ValueError("Residue requires paired dynamic-block NVFP4 or MXFP4 quantizers.")
            formats.append(scale_bits)
        if formats[0] != formats[1]:
            raise ValueError("Residue activation and weight formats must match.")
        if salience_metric == "nvfp4_error_weighted" and formats[0] != (4, 3):
            raise ValueError("nvfp4_error_weighted requires NVFP4 quantizers.")
        names.append(name)
    return names


@torch.no_grad()
def residue_max_calibrate(
    model, forward_loop, block_size=8, residue_ratio=0.25, salience_metric="activation_max"
):
    """Calibrate salient channels without materializing extended weights."""
    ResidueMaxCalibConfig(
        block_size=block_size, residue_ratio=residue_ratio, salience_metric=salience_metric
    )
    if forward_loop is None:
        raise ValueError("forward_loop is required for residue_max calibration.")
    names = _validate_modules(model, salience_metric)
    selections = {}
    with _calibration_state(model), ExitStack() as stack:
        for name in names:
            module = model.get_submodule(name)
            helper = ResidueHelper(module, salience_metric)
            module.residue_max = helper
            stack.callback(delattr, module, "residue_max")
            helper.setup()
            stack.callback(helper.cleanup)
        max_calibrate(
            model,
            forward_loop,
            distributed_sync=False,
            shared_states={"weight_global_amax": {"patterns": []}},
        )
        for name in names:
            module = model.get_submodule(name)
            helper = module.residue_max
            if helper.num_cache_steps == 0:
                raise ValueError(f"No calibration samples for residue layer {name!r}.")
            score = helper.act_scale
            if salience_metric == "activation_max":
                score = score / helper.num_cache_steps
            else:
                score = get_nvfp4_error_weighted_score(score, module.weight)
            if not torch.isfinite(score).all():
                raise ValueError(f"Nonfinite residue salience for layer {name!r}.")
            selections[name] = get_salient_indices(score, block_size, residue_ratio)
            calibrated_amax = module.input_quantizer.amax
            module.input_quantizer.reset_amax()
            module.input_quantizer.axis = None
            if helper._saved_awq_amax is not None:
                module.input_quantizer.amax = helper._saved_awq_amax
            elif calibrated_amax is not None:
                module.input_quantizer.amax = calibrated_amax.amax()
    create_and_replace_residue_linear_on_the_fly(model, names)
    for name, indices in selections.items():
        quantizer = model.get_submodule(name).input_quantizer
        if hasattr(quantizer, "_salient_indices"):
            delattr(quantizer, "_salient_indices")
        quantizer.salient_indices = indices
