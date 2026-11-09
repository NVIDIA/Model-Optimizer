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


"""Scoped output-head capture and bounded native projection for HF tasks."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.modules import module as module_hooks

from modelopt.torch.quantization.model_calib import _WeightInputHook
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.repair_partition import requires_full_context_activations
from modelopt.torch.quantization.utils.calib_utils import GPTQHelper
from modelopt.torch.quantization.utils.core_utils import _get_enclosing_fsdp_module

__all__ = []

_FULL_REPAIR_LOGITS = ContextVar("modelopt_repair_full_logits", default=False)


@contextmanager
def capture_head_inputs(head):
    """Capture the actual projection input without allocating vocabulary logits."""
    captured = []
    had_override, previous = "forward" in head.__dict__, head.__dict__.get("forward")
    copy_inputs = bool(head._forward_pre_hooks)

    def capture(module, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states", kwargs.get("input"))
        if not isinstance(hidden, torch.Tensor):
            raise ValueError("The output head requires a tensor projection input.")
        captured.append(hidden.detach().clone() if copy_inputs else hidden.detach())

    def identity(*args, **kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states", kwargs.get("input"))
        assert isinstance(hidden_states, torch.Tensor)  # Validated by the capture pre-hook.
        # Task-owned in-place softcapping must not modify the captured hidden states.
        return hidden_states.new_zeros((*hidden_states.shape[:-1], 1))

    handle = head.register_forward_pre_hook(capture, with_kwargs=True, prepend=True)
    head.forward = identity
    try:
        yield captured
    finally:
        handle.remove()
        if had_override:
            head.forward = previous
        else:
            delattr(head, "forward")


@contextmanager
def repair_head_context(model, units, head):
    """Skip unused head math, retaining the original module, aliases and hooks."""
    if head is None:
        yield
        return
    if not units:
        calibrating = any(
            bool(getattr(module, "_weight_input_capture", False))
            or hasattr(module, GPTQHelper.CACHE_NAME)
            or hasattr(module, "_forward_no_awq")
            or any(
                isinstance(hook, _WeightInputHook) for hook in module._forward_pre_hooks.values()
            )
            or (isinstance(module, TensorQuantizer) and module.is_enabled and module._if_calib)
            for module in head.modules()
        )
        token = _FULL_REPAIR_LOGITS.set(calibrating)
        try:
            yield
        finally:
            _FULL_REPAIR_LOGITS.reset(token)
        return
    paths = [name for name, child in model._modules.items() if child is head]
    selected = (
        any(
            not unit.teacher_path
            or path == unit.teacher_path
            or path.startswith(unit.teacher_path + ".")
            for unit in units
            for path in paths
        )
        if paths
        else any(
            head is child
            for unit in units
            for child in model.get_submodule(unit.teacher_path).modules()
        )
    )
    token = _FULL_REPAIR_LOGITS.set(selected)
    try:
        if selected:
            yield
        else:
            with capture_head_inputs(head):
                yield
    finally:
        _FULL_REPAIR_LOGITS.reset(token)


class OutputHeadProjection:
    """Project final features using model-owned logit transformations."""

    def __init__(self, model, family):
        self.model, self.family = model, family
        self.head = family.output_head(model)
        if self.head is not None and (
            module_hooks._global_forward_pre_hooks or module_hooks._global_forward_hooks
        ):
            raise ValueError(
                "Chunked head diagnostics cannot replay global module hooks safely; "
                "use module-local hooks or a specialized diagnostic path."
            )
        self.owner = _get_enclosing_fsdp_module(self.head, model) if self.head is not None else None
        if self.head is not None and requires_full_context_activations(self.head):
            raise ValueError(
                "Chunked output-head diagnostics require token-independent activation statistics; "
                "use a specialized diagnostic path for context-coupled head quantization."
            )

    @contextmanager
    def weight_window(self):
        # Open only the head's own FSDP group, never all nested model groups.
        owner = self.owner
        group = fully_shard.state(owner)._fsdp_param_group if owner is not None else None
        sharded = group is not None and group.is_sharded
        if sharded and owner is not None:
            owner.unshard()
        try:
            yield
        finally:
            if sharded and owner is not None:
                owner.reshard()

    @property
    def sliceable(self):
        """Whether vocabulary slicing preserves this module's actual forward."""
        return (
            isinstance(self.head, nn.Linear)
            and type(self.head).forward is nn.Linear.forward
            and "forward" not in self.head.__dict__
            and not self.head._forward_pre_hooks
            and not self.head._forward_hooks
            and not hasattr(self.head, "_hf_hook")
            and not isinstance(self.head.weight, DTensor)
            and not isinstance(self.head.bias, DTensor)
            and not self.head.weight.is_meta
        )

    def project(self, hidden, columns=None):
        """Keep quantized/custom/TP heads on their own module-forward path."""
        if columns is None:
            logits = self.head(hidden)
            if isinstance(logits, DTensor):
                logits = logits.full_tensor()
        else:
            bias = self.head.bias[columns] if self.head.bias is not None else None
            logits = torch.nn.functional.linear(hidden, self.head.weight[columns], bias)
        return self.family.transform_logits(self.model, logits)
