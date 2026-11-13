# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2023 MIT HAN Lab (get_act_scale)
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

from .config import ResidueMaxCalibConfig
from .conversion import create_and_replace_residue_linear_on_the_fly, set_quantizer_by_cfg_context
from .model_calib import _register_weight_input_hooks, max_calibrate
from .nn import TensorQuantizer
from .utils import module_name_maps
from .utils.shared_input import SHARED_PATTERNS, find_shared_input_groups

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
    weight_col_norm = torch.linalg.vector_norm(weight.detach(), ord=2, dim=0, dtype=torch.float32)
    return accumulated_error_sq.float().clamp_min(0).sqrt() * weight_col_norm


def _set_residue_indices(quantizer: TensorQuantizer, indices: torch.Tensor):
    """Store candidate-specific channels in native quantizer checkpoint state."""
    if indices.ndim != 1 or indices.dtype != torch.long:
        raise ValueError("Residue indices must be a one-dimensional int64 tensor.")
    # Recalibration may select a different number of channels.
    quantizer.register_buffer("_salient_indices", indices.detach().clone())
    quantizer._enable_salient_indices = True


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
        """Collect raw-input statistics while the native forward stays unquantized."""
        quantizer = self.module.input_quantizer
        quantizer.disable()
        quantizer.reset_amax()
        quantizer.axis = -1
        if hasattr(quantizer, "_enable_salient_indices"):
            quantizer._enable_salient_indices = False

    def collect(self, inputs):
        """Score raw activations and separately collect native quantizer scales."""
        if inputs.numel() == 0:
            return
        quantizer = self.module.input_quantizer
        if self.salience_metric == "nvfp4_error_weighted":
            with (
                set_quantizer_by_cfg_context(quantizer, [{"quantizer_name": "*", "enable": True}]),
                quantizer.disable_pre_quant_scale(),
            ):
                quantizer.enable_quant()
                quantizer.disable_calib()
                quantized_input = quantizer(inputs)
            flat_error = (inputs.float() - quantized_input.float()).reshape(-1, inputs.shape[-1])
            self.act_scale += flat_error.square().sum(dim=0)
        else:
            # Average each batch first, preserving the donor's equal batch weighting.
            self.act_scale += inputs.abs().reshape(-1, inputs.shape[-1]).mean(0).float()
        self.num_cache_steps += 1
        if self.is_input_quantized and self._saved_awq_amax is None:
            with set_quantizer_by_cfg_context(quantizer, [{"quantizer_name": "*", "enable": True}]):
                max_calibrate(quantizer, lambda q: q(inputs), distributed_sync=False)


@contextmanager
def _calibration_state(model):
    quantizers = [q for q in model.modules() if isinstance(q, TensorQuantizer)]
    states = [
        (
            q._disabled,
            q._if_quant,
            q._if_calib,
            q.axis,
            {
                name: q._buffers[name].clone()
                for name in ("_amax", "_bias_value")
                if name in q._buffers
            },
            deepcopy(q._calibrator),
            deepcopy(q._bias_calibrator),
            getattr(q, "_enable_salient_indices", None),
        )
        for q in quantizers
    ]
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        for q, (disabled, quant, calib, axis, buffers, calibrator, bias_calibrator, salient) in zip(
            quantizers, states
        ):
            q._disabled, q._if_quant, q._if_calib = disabled, quant, calib
            q.axis = axis
            q._calibrator = calibrator
            q._bias_calibrator = bias_calibrator
            if salient is not None:
                q._enable_salient_indices = salient
            if not succeeded:
                # Resetting amax here would also clear the restored observer statistics.
                for name in ("_amax", "_bias_value"):
                    if hasattr(q, name):
                        delattr(q, name)
                    if name in buffers:
                        q.register_buffer(name, buffers[name])


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
            if type(q) is not TensorQuantizer or id(q) in seen_quantizers:
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
    model,
    forward_loop,
    block_size=8,
    residue_ratio=0.25,
    salience_metric="activation_max",
    shared_input_patterns=None,
):
    """Calibrate salient channels without materializing extended weights."""
    ResidueMaxCalibConfig(
        block_size=block_size,
        residue_ratio=residue_ratio,
        salience_metric=salience_metric,
        shared_input_patterns=shared_input_patterns,
    )
    if forward_loop is None:
        raise ValueError("forward_loop is required for residue_max calibration.")
    names = _validate_modules(model, salience_metric)
    patterns = SHARED_PATTERNS if shared_input_patterns is None else shared_input_patterns
    module_names = {id(model.get_submodule(name)): name for name in names}
    groups = []
    for _, members in find_shared_input_groups(model, patterns, target_quantizer_kind="input"):
        if any(id(m) not in module_names for m in members):
            raise ValueError("Residue groups must contain only residue-calibrated Linear layers.")
        first = members[0]
        if any(
            m.in_features != first.in_features
            or m.weight.device != first.weight.device
            or m.input_quantizer.block_sizes != first.input_quantizer.block_sizes
            or m.input_quantizer.pre_quant_scale is not None
            or m.weight_quantizer.pre_quant_scale is not None
            for m in members
        ):
            raise ValueError(
                "Residue groups require matching input widths, devices and formats without pre-quant scales."
            )
        groups.append(tuple(module_names[id(m)] for m in members))
    selections = {}
    scores = {}
    helpers = {name: ResidueHelper(model.get_submodule(name), salience_metric) for name in names}
    by_quantizer = {id(h.module.weight_quantizer): h for h in helpers.values()}
    with _calibration_state(model), ExitStack() as stack:

        def capture(weight_quantizer, weight, inputs):
            by_quantizer[id(weight_quantizer)].collect(inputs)

        # Register before disabling quantizers: the shared helper filters enabled weights.
        handles = _register_weight_input_hooks(model, module_name_maps(model), capture)
        for handle in handles:
            stack.callback(handle.remove)
        for helper in helpers.values():
            helper.setup()

        def collect_forward(calibration_model):
            # Only disable weights for the data pass; max_calibrate still calibrates weights.
            with ExitStack() as contexts:
                for helper in helpers.values():
                    contexts.enter_context(
                        set_quantizer_by_cfg_context(
                            helper.module.weight_quantizer,
                            [{"quantizer_name": "*", "enable": False}],
                        )
                    )
                forward_loop(calibration_model)

        max_calibrate(
            model,
            collect_forward,
            distributed_sync=False,
            shared_states={"weight_global_amax": {"patterns": []}},
        )
        for name in names:
            module = model.get_submodule(name)
            helper = helpers[name]
            if helper.num_cache_steps == 0:
                raise ValueError(f"No calibration samples for residue layer {name!r}.")
            score = helper.act_scale
            if salience_metric == "activation_max":
                score = score / helper.num_cache_steps
            else:
                score = get_nvfp4_error_weighted_score(score, module.weight)
            if not torch.isfinite(score).all():
                raise ValueError(f"Nonfinite residue salience for layer {name!r}.")
            scores[name] = score
            calibrated_amax = module.input_quantizer.amax
            module.input_quantizer.reset_amax()
            module.input_quantizer.axis = None
            if helper._saved_awq_amax is not None:
                module.input_quantizer.amax = helper._saved_awq_amax
            elif calibrated_amax is not None:
                module.input_quantizer.amax = calibrated_amax.amax()
        for group in groups:
            values = torch.stack([scores[name] for name in group])
            # Squared weighted scores add the error energy across all output projections.
            score = (
                values.square().sum(0).sqrt()
                if salience_metric == "nvfp4_error_weighted"
                else values.mean(0)
            )
            if not torch.isfinite(score).all():
                raise ValueError("Nonfinite grouped residue salience.")
            indices = get_salient_indices(score, block_size, residue_ratio)
            selections.update(dict.fromkeys(group, indices))
        for name in names:
            if name not in selections:
                selections[name] = get_salient_indices(scores[name], block_size, residue_ratio)
    create_and_replace_residue_linear_on_the_fly(model, names)
    for name, indices in selections.items():
        quantizer = model.get_submodule(name).input_quantizer
        _set_residue_indices(quantizer, indices)
        model.get_submodule(name).weight_quantizer._enable_salient_indices = True
    membership = {name: group for group in groups for name in group}
    for name in names:
        model.get_submodule(name)._residue_group = membership.get(name, ())
