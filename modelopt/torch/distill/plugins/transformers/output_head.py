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

"""Skip an unselected task head while preserving its module and outer hooks."""

import importlib
from contextlib import contextmanager

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.nn.modules import module as module_hooks


@contextmanager
def bypass_head_context(model, units, head, *, head_paths=None):
    """Avoid allocating unused vocabulary logits; restore any existing forward override."""
    task = getattr(model, "thinker", model)
    getter = getattr(task, "get_input_embeddings", None)
    try:
        input_embeddings = getter() if callable(getter) else None
    except NotImplementedError:
        input_embeddings = None
    # Some tasks expose the input embedding module itself as their output head.
    # Replacing its forward would corrupt tokens before the selected boundaries.
    if head is None or head is input_embeddings:
        yield
        return
    if not units:
        from modelopt.torch.quantization.model_calib import _WeightInputHook
        from modelopt.torch.quantization.nn import TensorQuantizer
        from modelopt.torch.quantization.utils.calib_utils import GPTQHelper

        calibrating = any(
            hasattr(module, GPTQHelper.CACHE_NAME)
            or getattr(module, "_weight_input_capture", False)
            or any(
                isinstance(hook, _WeightInputHook) for hook in module._forward_pre_hooks.values()
            )
            or (isinstance(module, TensorQuantizer) and module.is_enabled and module._if_calib)
            for module in head.modules()
        )
        if calibrating:
            yield
            return
    paths = (
        [path for path, child in model.named_modules() if child is head]
        if head_paths is None
        else head_paths
    )
    selected = any(
        not unit.teacher_path
        or path == unit.teacher_path
        or path.startswith(unit.teacher_path + ".")
        for unit in units
        for path in paths
    )
    if selected or not paths:
        yield
        return
    had_override, previous = "forward" in head.__dict__, head.__dict__.get("forward")

    def unused_logits(*args, **kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states", kwargs.get("input"))
        if not isinstance(hidden, torch.Tensor):
            raise ValueError("A Bypass output head requires tensor hidden states")
        return hidden.new_zeros((*hidden.shape[:-1], 1))

    head.forward = unused_logits
    try:
        yield
    finally:
        if had_override:
            head.forward = previous
        else:
            delattr(head, "forward")


def softcapped_logits(model, logits):
    """Apply the declared native task's scale and tanh softcap in its original dtype."""
    target = getattr(model, "thinker", model)
    config = target.config.get_text_config()
    multiplier = getattr(config, "output_multiplier", None)
    if multiplier is not None:
        logits = logits * multiplier
    cap = getattr(target, "final_logit_softcapping", None)
    if cap is None:
        cap = getattr(config, "final_logit_softcapping", None)
    if cap in (None, 0):
        return logits
    return (logits / cap).tanh() * cap


def _resolve_logits_transform(model):
    """Resolve post-head mathematics through the same ModelSpec as Bypass replay."""
    from .bypass import _model_spec

    spec = _model_spec(model)
    if spec.logits_transform_path is not None:
        module, _, name = spec.logits_transform_path.rpartition(".")
        transform = getattr(importlib.import_module(module), name)
        if not callable(transform):
            raise ValueError("The declared logits transform must be callable.")
        return transform, spec.logits_transform_slices
    target = getattr(model, "thinker", model)
    config = getattr(target, "config", None)
    if config is not None and callable(getattr(config, "get_text_config", None)):
        config = config.get_text_config()
    if (
        getattr(config, "final_logit_softcapping", None) not in (None, 0)
        or getattr(config, "output_multiplier", 1) != 1
    ):
        raise ValueError("Declare this task's logits_transform_path on its ModelSpec.")
    return None, True


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


class OutputHeadProjection:
    """Project final features using model-owned logit transformations."""

    def __init__(self, model):
        """Resolve a real native head and reject context-dependent chunking."""
        from modelopt.torch.quantization.repair_partition import requires_full_context_activations
        from modelopt.torch.quantization.utils.core_utils import _get_enclosing_fsdp_module

        self.model = model
        target = getattr(model, "thinker", model)
        getter = getattr(target, "get_output_embeddings", None)
        try:
            self.head = getter() if callable(getter) and hasattr(target, "config") else None
        except NotImplementedError:
            self.head = None
        if self.head is not None and (
            module_hooks._global_forward_pre_hooks or module_hooks._global_forward_hooks
        ):
            raise ValueError("Chunked diagnostics cannot safely replay global module hooks.")
        self.owner = _get_enclosing_fsdp_module(self.head, model) if self.head is not None else None
        if self.head is not None and requires_full_context_activations(self.head):
            raise ValueError(
                "Chunked output heads require token-independent activation statistics."
            )
        self.transform, self._transform_sliceable = (
            _resolve_logits_transform(model) if self.head is not None else (None, True)
        )

    @contextmanager
    def weight_window(self):
        """Materialize only the head's actual FSDP owner during projection."""
        # Open only the head's own FSDP group, never all nested model groups.
        from torch.distributed.fsdp import fully_shard

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
            self._transform_sliceable
            and isinstance(self.head, nn.Linear)
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
        assert self.head is not None
        if columns is None:
            logits = self.head(hidden)
            if isinstance(logits, DTensor):
                logits = logits.full_tensor()
        else:
            bias = self.head.bias[columns] if self.head.bias is not None else None
            logits = torch.nn.functional.linear(hidden, self.head.weight[columns], bias)
        return self.transform(self.model, logits) if self.transform is not None else logits
