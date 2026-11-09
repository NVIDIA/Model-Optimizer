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


"""Native FSDP lifecycles for children whose parameters are read by their parent."""

from functools import wraps
from types import MethodType

from torch import nn
from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method

from .families import transformers_family

__all__ = ["extend_family_fsdp_plan", "install_functional_fsdp_calls"]

_CALL_METHOD = "_modelopt_functional_fsdp_call"
_BINDINGS = "_modelopt_functional_fsdp_children"


def extend_family_fsdp_plan(model, plan):
    """Separate declared called children from an already selected precision owner."""
    family = transformers_family(model)
    if not family.additional_fsdp_children:
        return plan
    result = dict(plan)
    names = {id(module): name for name, module in model.named_modules()}
    layers = family.decoder_layers(model)
    for block in layers.values() if isinstance(layers, nn.ModuleDict) else layers:
        for path in family.additional_fsdp_children:
            name = names[id(block)] + "." + path
            if name.rpartition(".")[0] in plan:
                model.get_submodule(name)
                result.setdefault(name, "free_full_weight")
    return dict(sorted(result.items(), key=lambda item: item[0].count("."), reverse=True))


def _call_parent(self, forward, *args, **kwargs):
    return forward(*args, **kwargs)


def _wrap_parent(parent, children):
    signature = tuple((name, id(child)) for name, child in children)
    existing = getattr(parent, _BINDINGS, None)
    if existing is not None:
        if existing != signature:
            raise ValueError("Functional FSDP children changed after their forward was prepared.")
        return
    forward = parent.forward
    for _, child in reversed(children):
        if not hasattr(child, _CALL_METHOD):
            setattr(child, _CALL_METHOD, MethodType(_call_parent, child))
            register_fsdp_forward_method(child, _CALL_METHOD)
        inner = forward

        @wraps(inner)
        def forward(*args, _inner=inner, _child=child, **kwargs):
            return getattr(_child, _CALL_METHOD)(_inner, *args, **kwargs)

    parent.forward = forward
    setattr(parent, _BINDINGS, signature)


def install_functional_fsdp_calls(model):
    """Execute native parent math through its declared children's FSDP forward methods.

    Install after native FSDP setup. Tensor arguments reach FSDP unchanged, so its
    normal input policy, resharding and backward hooks apply during unit replay too.
    """
    family = transformers_family(model)
    if not family.functional_children:
        return
    layers = family.decoder_layers(model)
    parents = {}
    for block in layers.values() if isinstance(layers, nn.ModuleDict) else layers:
        for path in family.functional_children:
            owner, _, name = path.rpartition(".")
            try:
                parent = block.get_submodule(owner)
            except AttributeError:
                continue
            child = parent._modules.get(name)
            if isinstance(child, FSDPModule):
                parents.setdefault(parent, []).append((name, child))
    for parent, children in parents.items():
        _wrap_parent(parent, children)
