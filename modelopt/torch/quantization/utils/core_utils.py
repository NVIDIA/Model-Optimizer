# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Quantization utilities."""

import copy
import itertools
import sys
import threading
import warnings
import weakref
from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.tensor import DTensor, Replicate

from modelopt.torch.quantization.config import QuantizerCfgEntry
from modelopt.torch.utils import get_unwrapped_name, print_rank_0
from modelopt.torch.utils.network import temporarily_remove_accelerate_hook

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch.distributed.device_mesh import DeviceMesh

# FP8 dtypes do not implement reduction kernels (e.g. ``max_all_cuda``), ``abs``, or
# elementwise ``maximum``, so tensors of these dtypes must be upcast before amax reduction.
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def reduce_block_amax(input_tensor: torch.Tensor, block_sizes: dict):
    """Computes the amax of the input tensor using block-based reduction for each dimension.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        block_sizes (dict): A dictionary specifying the block size for each dimension.
                            Example: `{-1: 128, -2: 128}` reduces over 2D blocks.

    Returns:
        torch.Tensor: The reduced tensor with amax computed per block.

    Example:
        Input Shape: [256, 512]
        Block Sizes: {-1: 128, -2: 128}
        Process:
            - Block along last dim → Shape [256, 4, 128]
            - Compute block-wise amax → Shape [256, 4]
            - Block along second-to-last dim → Shape [2, 128, 4]
            - Compute block-wise amax → Shape [2, 4]
    """
    with torch.no_grad():
        amax = input_tensor.clone()

        for dim, block_size in block_sizes.items():
            # Convert negative dimensions to positive
            dim = dim if dim >= 0 else len(amax.shape) + dim
            assert amax.shape[dim] % block_size == 0, (
                f"Tensor dimension {amax.shape[dim]}, {amax.shape[dim]} is not divisible by {block_size}"
            )

            # Compute new shape for blocking
            outer_dim = amax.shape[dim] // block_size
            new_shape = [
                *list(amax.shape[:dim]),
                outer_dim,
                block_size,
                *list(amax.shape[dim + 1 :]),
            ]

            # Reshape into blocks
            amax = amax.reshape(new_shape)

            # Reduce along the newly created block dimension
            # Shift by 1 because we added an extra dimension
            amax = reduce_amax(amax, dim + 1, keepdims=False, squeeze_scalar=False)

        return amax


def reduce_block_padding(input: torch.Tensor, block_sizes: dict, pad_value: float = 0):
    """Padding the input using block-based reduction for each dimension.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        block_sizes (dict): A dictionary specifying the block size for padding each dimension.
                            Example: `{-1: 128, -2: 128}` pads the input over 2D blocks.
    """
    with torch.no_grad():
        padded_tensor = input
        num_dims = padded_tensor.dim()

        # Process each specified dimension independently
        for dim, block in block_sizes.items():
            # Convert negative dimension to positive index
            pos_dim = dim if dim >= 0 else num_dims + dim

            # Calculate how many elements are missing along that dimension
            current_size = padded_tensor.size(pos_dim)
            remainder = current_size % block
            pad_amt = 0 if remainder == 0 else block - remainder

            if pad_amt > 0:
                # F.pad expects a pad tuple of length 2*num_dims.
                pad = [0] * (2 * num_dims)
                # For dimension pos_dim, the right padding is at index: (num_dims - 1 - pos_dim)*2 + 1.
                pad_index = (num_dims - 1 - pos_dim) * 2
                pad[pad_index + 1] = (
                    pad_amt  # Set padding on the right side of the target dimension
                )

                padded_tensor = F.pad(padded_tensor, pad, value=pad_value)

        return padded_tensor


def convert_quantization_axis_to_reduce_axis(input, axis):
    """Convert the quantization axis to the reduce axis.

    Args:
        input (torch.Tensor): The input tensor.
        axis (int, tuple, list of None): The quantization axis. None means per-tensor quantization.

    Returns:
        list: The axis to reduce. None suggests all dimensions should be reduced.
    """
    if axis is None:
        return None
    axis = axis if isinstance(axis, list | tuple) else [axis]
    # Handle positive and negative axis.
    reduce_axis = [i for i in range(input.dim()) if i not in axis and (i - input.dim()) not in axis]
    return reduce_axis


@torch.no_grad()
def reduce_amax(input, axis=None, keepdims=True, squeeze_scalar=True):
    """Compute the absolute maximum value of a tensor.

    Reduces input_tensor along the dimensions given in axis. Unless keepdims is true,
    the rank of the tensor is reduced by 1 for each entry in axis. If keepdims is true,
    the reduced dimensions are retained with length 1.

    .. note::
        Gradient computation is disabled as this function is never meant learning reduces amax

    Args:
        input: Input tensor
        axis: The dimensions to reduce. None or int or tuple of ints. If None (the default),
            reduces all dimensions. Must be in the range [-rank(input_tensor), rank(input_tensor)).
        keepdims: A boolean. If true, retains reduced dimensions with length 1. Default True

    Returns:
        The reduced tensor.
    """
    # FP8 dtypes lack reduction/abs kernels (e.g. ``max_all_cuda``); upcast to the default
    # float dtype, which represents every FP8 value exactly so the amax is computed losslessly.
    if input.dtype in _FP8_DTYPES:
        input = input.to(torch.get_default_dtype())
    # A memory-efficient implementation that avoids copying input tensor
    if axis is None:
        max_val = torch.max(input)
        min_val = torch.min(input)
        output = torch.maximum(torch.abs(max_val), torch.abs(min_val))
    else:
        if isinstance(axis, int):
            axis = (axis,)
        max_val = torch.amax(input, dim=axis, keepdim=keepdims)
        min_val = torch.amin(input, dim=axis, keepdim=keepdims)
        output = torch.maximum(torch.abs(max_val), torch.abs(min_val))
        if squeeze_scalar and output.numel() == 1:
            output.squeeze_()
    return output


@torch.no_grad()
def reduce_sum(input, axis=None, keepdims=True):
    """Compute the sum of a tensor along specified axes.

    Reduces input_tensor along the dimensions given in axis. Unless keepdims is true,
    the rank of the tensor is reduced by 1 for each entry in axis. If keepdims is true,
    the reduced dimensions are retained with length 1.

    .. note::
        Gradient computation is disabled as this function is never meant for learning.

    Args:
        input: Input tensor
        axis: The dimensions to reduce. None or int or tuple of ints. If None (the default),
            reduces all dimensions. Must be in the range [-rank(input_tensor), rank(input_tensor)).
        keepdims: A boolean. If true, retains reduced dimensions with length 1. Default True

    Returns:
        The reduced tensor.
    """
    if axis is None:
        output = torch.sum(input)
    else:
        if isinstance(axis, int):
            axis = (axis,)
        output = torch.sum(input, dim=axis, keepdim=keepdims)
    return output


def _weight_quantizer_container(module, weight_name):
    """Resolve singular, grouped, plural and legacy first-projection layouts once."""
    from ..nn import GroupedQuantizer, SequentialQuantizer, TensorQuantizer

    singular = quantizer_attr_names(weight_name).weight_quantizer
    container = getattr(module, singular, None)
    if isinstance(container, TensorQuantizer | SequentialQuantizer | GroupedQuantizer):
        return container
    container = getattr(module, singular + "s", None)
    if isinstance(container, nn.ModuleList):
        return container
    if weight_name == getattr(module, "_first_proj_attr", None) and weight_name != "gate_up_proj":
        container = getattr(module, "gate_up_proj_weight_quantizers", None)
        if isinstance(container, nn.ModuleList):
            return container
    return None


def representative_weight_quantizer(module: nn.Module, weight_name: str = "weight"):
    """Return the representative weight quantizer for ``weight_name`` on ``module``.

    Handles three layouts:

    - singular ``<name>_weight_quantizer`` — standard ``nn.Linear`` / ``_QuantLinear``.
    - singular ``<name>_weight_quantizer`` that is a ``GroupedQuantizer`` — TEGroupedLinear
      fused experts (one quantizer per expert); the first is representative.
    - plural ``<name>_weight_quantizers`` (``nn.ModuleList``) — fused-experts modules
      (``_QuantFusedExperts``) hold one ``TensorQuantizer`` per expert. Per-expert
      formats are identical, so the first element is representative.

    Returns ``None`` if no matching quantizer is found.
    """
    from ..nn import GroupedQuantizer, SequentialQuantizer, TensorQuantizer

    container = _weight_quantizer_container(module, weight_name)
    if isinstance(container, TensorQuantizer | SequentialQuantizer):
        return container
    if isinstance(container, GroupedQuantizer | nn.ModuleList) and len(container):
        first = container[0]
        if isinstance(first, TensorQuantizer | SequentialQuantizer):
            return first
    return None


def _weight_quantizer_owners(module, selected):
    """Map selected quantizers to physical parameters using the layouts below."""
    from ..nn import GroupedQuantizer  # ..nn imports this utils package at module scope

    owners = {}
    for name in weight_attr_names(module):
        container = _weight_quantizer_container(module, name)
        pairs = (
            [(f"weight{i}", q) for i, q in enumerate(container)]
            if isinstance(container, GroupedQuantizer)
            else [(name, container)]
        )
        for parameter_name, quantizers in pairs:
            if quantizers is None:
                continue
            matches = selected.intersection(id(q) for q in quantizers.modules())
            if matches:
                if module._parameters.get(parameter_name) is None:
                    raise ValueError(
                        f"Selected quantizer on {type(module).__name__}.{parameter_name} "
                        "has no registered physical parameter"
                    )
                owners.update(dict.fromkeys(matches, parameter_name))
    if owners.keys() != selected:
        raise ValueError("Selected quantizers have unsupported physical weight ownership")
    return owners


def weight_attr_names(module: nn.Module) -> "Generator[str, None, None]":
    """Get the weight param attribute names in a converted module, non-recursive.

    Covers four layouts:

    - standard ``nn.Linear``: ``weight`` + ``weight_quantizer``.
    - custom per-weight quantizer (e.g. ``Llama4TextExperts`` with ``gate_up_proj`` +
      ``gate_up_proj_weight_quantizer``).
    - fused-experts ``nn.ModuleList`` quantizers (``_QuantFusedExperts`` with
      ``<first_proj>`` + ``<first_proj>_weight_quantizers`` plural list).
    - TEGroupedLinear: ``weight0..N`` behind one ``GroupedQuantizer``, reported as ``weight``.
    """
    from ..nn import GroupedQuantizer  # local: ..nn imports this utils package at module scope

    # standard: "weight" + "weight_quantizer" (singular) or "weight_quantizers" (plural)
    if getattr(module, "weight", None) is not None:
        if representative_weight_quantizer(module, "weight") is not None:
            yield "weight"
    elif isinstance(getattr(module, "weight_quantizer", None), GroupedQuantizer):
        # TEGroupedLinear: ``weight0..N`` share one GroupedQuantizer and there is no ``weight``.
        yield "weight"

    # per-parameter custom attr names
    for name, _ in module.named_parameters(recurse=False):
        if name == "weight":
            continue
        weight = getattr(module, name, None)
        if not isinstance(weight, nn.Parameter):
            continue
        if representative_weight_quantizer(module, name) is not None:
            yield name


"""The whole set of quantizer related attribute names for a given weight name."""
QuantizerAttrNames = namedtuple(
    "QuantizerAttrNames",
    (
        "weight_quantizer",
        "input_quantizer",
        "output_quantizer",
        "weight_scale",
        "weight_scale_2",
        "input_scale",
        "output_scale",
    ),
)


def quantizer_attr_names(weight_name: str = "weight") -> QuantizerAttrNames:
    """Get all the quantizer related attribute names for a given weight name."""
    prefix = f"{weight_name}_" if weight_name != "weight" else ""
    return QuantizerAttrNames(
        weight_quantizer=f"{prefix}weight_quantizer",
        input_quantizer=f"{prefix}input_quantizer",
        output_quantizer=f"{prefix}output_quantizer",
        weight_scale=f"{prefix}weight_scale",
        weight_scale_2=f"{prefix}weight_scale_2",
        input_scale=f"{prefix}input_scale",
        output_scale=f"{prefix}output_scale",
    )


def is_quantized(module):
    """Check if a module is quantized."""
    from ..nn import TensorQuantizer

    return any(isinstance(_module, TensorQuantizer) for _module in module.modules())


def is_quantized_linear(module):
    """Check if a module is a quantized linear module."""
    from ..nn import QuantModule, TensorQuantizer

    # Embedding has a 2D weight but is not a GEMM op, so calibration passes that operate
    # on linear activations (AWQ, SmoothQuant, SVDQuant) must skip it.
    if isinstance(module, nn.Embedding):
        return False

    return (
        isinstance(module, QuantModule)
        and isinstance(getattr(module, "input_quantizer", None), TensorQuantizer)
        and hasattr(module, "weight_quantizer")
        and (
            (getattr(module, "weight", None) is not None and module.weight.dim() == 2)
            # module.weight0 check is required to support TEGroupedLinear
            or (getattr(module, "weight0", None) is not None and module.weight0.dim() == 2)
        )
    )


def is_quantized_column_parallel_linear(module):
    """Check if a module is a quantized column parallel linear module."""
    return is_quantized_linear(module) and getattr(module, "_is_column_parallel", False)


def is_quantized_row_parallel_linear(module):
    """Check if a module is a quantized row parallel linear module."""
    return is_quantized_linear(module) and getattr(module, "_is_row_parallel", False)


def is_quantized_parallel_linear(module):
    """Check if a module is a quantized parallel linear module."""
    return is_quantized_column_parallel_linear(module) or is_quantized_row_parallel_linear(module)


@contextmanager
def calibrate_with_adapters(model, args):
    """Disables LoRA adapters during calibration, then re-enables them afterward."""
    is_lora = getattr(args, "lora", None)
    if is_lora:
        print_rank_0("Disabling LoRA adapters during calibration...")
        model.disable_adapters()

    yield

    if is_lora:
        print_rank_0("Enabling LoRA adapters after calibration...")
        model.enable_adapters()


def disable_lora_quantizers_in_config(config, layers):
    """Turns off input, weight, and output quantizers for LoRA weights and LoRALinear layers in config."""
    config["quant_cfg"].append({"quantizer_name": "*lora*", "enable": False})
    for layer in layers:
        config["quant_cfg"].append({"quantizer_name": f"*{layer}.input_quantizer", "enable": False})
        config["quant_cfg"].append(
            {"quantizer_name": f"*{layer}.weight_quantizer", "enable": False}
        )
        config["quant_cfg"].append(
            {"quantizer_name": f"*{layer}.output_quantizer", "enable": False}
        )
    return config


@contextmanager
def replace_function(package, name, new_func, og_func_cache_name=None):
    """Replace a function with a new one within a context."""
    if og_func_cache_name is None:
        og_func_cache_name = "_" + name
    old_func = getattr(package, name)
    setattr(package, name, new_func)
    setattr(package, og_func_cache_name, old_func)
    try:
        yield
    finally:
        # Also when the body raises: a replacement left behind would stay patched process-wide.
        setattr(package, name, old_func)
        delattr(package, og_func_cache_name)


@contextmanager
def multi_context(*cms):
    """Context manager enabling variable number of context managers."""
    with ExitStack() as stack:
        yield [stack.enter_context(cls) for cls in cms]


EXPORT_MODE: bool = False


@contextmanager
def export_torch_mode():
    """Context manager enabling the export mode."""
    global EXPORT_MODE
    original_value = EXPORT_MODE
    EXPORT_MODE = True
    try:
        yield
    finally:
        EXPORT_MODE = original_value


def is_torch_export_mode():
    """Check whether in the context of exporting model to torch."""
    return EXPORT_MODE


def is_pow2(n):
    """Check if a number is the power of 2."""
    return (n != 0) and (n & (n - 1) == 0)


def _get_fsdp2_mesh(module: nn.Module):
    """Get the mesh info of the model."""
    fsdp_state = fully_shard.state(module)
    pg = fsdp_state._fsdp_param_group
    if pg is None:
        return None
    mesh_info = pg.post_forward_mesh_info if pg.is_sharded_post_forward else pg.mesh_info
    return mesh_info.mesh if mesh_info is not None else None


class ModuleNames(NamedTuple):
    """A model's ``name -> module`` and ``id(module) -> name`` maps, built together."""

    name_to_module: dict
    module_to_name: dict


def module_name_maps(root_model: nn.Module) -> ModuleNames:
    """Build both name maps for a model, so lookups are O(1) in either direction.

    Callers that resolve names for many modules should build this once and pass it down; resolving
    a module's name without it costs a ``named_modules()`` walk plus a linear scan every time.
    """
    name_to_module = dict(root_model.named_modules())
    return ModuleNames(name_to_module, {id(m): n for n, m in name_to_module.items()})


def _get_module_name(module: nn.Module, root_model: nn.Module, names: ModuleNames | None = None):
    if names is None:
        names = module_name_maps(root_model)
    return names.module_to_name.get(id(module))


def _get_enclosing_fsdp_module(
    module: nn.Module, root_model: nn.Module, names: ModuleNames | None = None
):
    """Get the enclosing FSDP module for a given module.

    Args:
        module: The module to find the enclosing FSDP for.
        root_model: The root model containing the module.
        names: Optional pre-built name maps (see :func:`module_name_maps`) to avoid rebuilding them.
    """
    if isinstance(module, FSDPModule):
        return module

    if names is None:
        names = module_name_maps(root_model)
    target_module_name = _get_module_name(module, root_model, names)
    name_to_module = names.name_to_module

    if target_module_name is None:
        raise ValueError(f"Module {module} not found in the root model {root_model}.")

    current_name = target_module_name
    while "." in current_name:
        parent_name = ".".join(current_name.split(".")[:-1])
        parent_module = name_to_module.get(parent_name)
        if parent_module and isinstance(parent_module, FSDPModule):
            return parent_module
        current_name = parent_name

    if isinstance(root_model, FSDPModule):
        return root_model


def _set_parameter(module: nn.Module, name: str, value: nn.Parameter):
    """Set a parameter on a module by dotted name (e.g. ``self_attn.q_proj.weight``)."""
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        parent = module.get_submodule(parts[0])
        attr = parts[1]
    else:
        parent = module
        attr = name
    parent._parameters[attr] = value


@contextmanager
def _fsdp2_unshard_context(fsdp_module: FSDPModule):
    """Materialize either shard state and reshard on exit; preserve existing native full views."""
    fsdp_param_group = fully_shard.state(fsdp_module)._fsdp_param_group
    was_sharded = not fsdp_param_group.is_unsharded
    if was_sharded:
        fsdp_module.unshard()
    try:
        with _disable_fsdp_unshard_reshard(fsdp_module, groups=[fsdp_param_group]):
            yield
    finally:
        if was_sharded:
            fsdp_module.reshard()


def _fsdp2_param_groups(module: nn.Module):
    """The distinct FSDP2 param groups of the units in ``module``, its own included."""
    return {
        id(group): group  # ``fully_shard([a, b])`` gives several modules one group
        for unit in module.modules()
        if isinstance(unit, FSDPModule)
        and (group := fully_shard.state(unit)._fsdp_param_group) is not None
    }.values()


def _fsdp2_persistent_parameters(module: nn.Module) -> dict[tuple[nn.Module, str], nn.Parameter]:
    """Map logical registrations to persistent shards without retaining unsharded views."""
    parameters = {}
    for group in _fsdp2_param_groups(module):
        for parameter in group.fsdp_params:
            info = parameter._module_info
            parameters[info.module, info.param_name] = parameter.sharded_param
            for owner, name in zip(info.shared_modules, info.shared_param_names, strict=True):
                parameters[owner, name] = parameter.sharded_param
    return parameters


class _FSDP2ParameterInfo(NamedTuple):
    reference: weakref.ref
    mesh: Any
    unsharded: bool


# Per FSDP2 unit, the parameters _fsdp2_owned_params found: the weight-access window opens once per
# quantized module, and a scan walks the unit's whole parameter group.
_FSDP2_OWNED_PARAMS: "weakref.WeakKeyDictionary[nn.Module, dict[int, _FSDP2ParameterInfo]]" = (
    weakref.WeakKeyDictionary()
)
_FSDP2_UNOWNED_PLAIN_PARAMS: "weakref.WeakKeyDictionary[nn.Module, dict[int, weakref.ref]]" = (
    weakref.WeakKeyDictionary()
)
_FSDP2_OWNED_MODULES: "weakref.WeakKeyDictionary[nn.Module, weakref.WeakSet[nn.Module]]" = (
    weakref.WeakKeyDictionary()
)
_FSDP2_ACTIVE_PARAMS: list[dict[int, weakref.ref]] = []


def _fsdp2_owned_modules(fsdp_module: FSDPModule, groups=None):
    # Registration owners stay stable across shard-state changes; weak caches retain no model.
    owners = _FSDP2_OWNED_MODULES.get(fsdp_module)
    if owners is None:
        owners = weakref.WeakSet()
        for group in _fsdp2_param_groups(fsdp_module) if groups is None else groups:
            for parameter in group.fsdp_params:
                info = parameter._module_info
                owners.add(info.module)
                owners.update(info.shared_modules)
        _FSDP2_OWNED_MODULES[fsdp_module] = owners
    return owners


def _fsdp2_owned_params(
    fsdp_module: FSDPModule, refresh: bool = False
) -> dict[int, _FSDP2ParameterInfo]:
    """The parameters FSDP2 manages in ``fsdp_module``'s unit and in the units inside it, by id.

    FSDP2 registers a different parameter object in each shard state -- sharded, resharded after the
    forward, unsharded -- so all of them count, whichever the unit is in. A DTensor parameter outside
    them is not FSDP2's to gather, whatever its mesh is named: an expert-parallel weight
    ``fully_shard`` was told to ignore, say. They are kept per unit; ``refresh`` scans again, for a
    state's parameter FSDP2 has made since. Weak references, checked by :func:`_fsdp2_owns`, keep no
    parameter alive and let an id a freed parameter left behind match nothing.
    """
    owned = None if refresh else _FSDP2_OWNED_PARAMS.get(fsdp_module)
    if owned is None:
        owned = {}
        groups = tuple(_fsdp2_param_groups(fsdp_module))
        _fsdp2_owned_modules(fsdp_module, groups)
        for group in groups:
            for fsdp_param in group.fsdp_params:
                # The last two exist once their state is first entered.
                for state_param, mesh_info, unsharded in (
                    (fsdp_param.sharded_param, group.mesh_info, False),
                    (
                        getattr(fsdp_param, "_sharded_post_forward_param", None),
                        group.post_forward_mesh_info,
                        False,
                    ),
                    (getattr(fsdp_param, "_unsharded_param", None), group.mesh_info, True),
                ):
                    if state_param is not None:
                        owned[id(state_param)] = _FSDP2ParameterInfo(
                            weakref.ref(state_param), mesh_info.mesh, unsharded
                        )
        # Remember plain misses across the whole unit, not once per leaf window. DTensor misses
        # still refresh ownership each window so newly materialized FSDP2 views are discovered.
        _FSDP2_UNOWNED_PLAIN_PARAMS[fsdp_module] = {
            id(parameter): weakref.ref(parameter)
            for parameter in fsdp_module.parameters()
            if not isinstance(parameter, DTensor) and not _fsdp2_owns(owned, parameter)
        }
        _FSDP2_OWNED_PARAMS[fsdp_module] = owned
    return owned


def _fsdp2_owns(owned: dict[int, _FSDP2ParameterInfo], param: nn.Parameter) -> bool:
    """Whether ``param`` itself, not just an object at its address, is among ``owned``."""
    ref = owned.get(id(param))
    if isinstance(ref, _FSDP2ParameterInfo):
        ref = ref.reference
    return ref is not None and ref() is param


@contextmanager
def fsdp2_weight_access_and_writeback_context(
    module: nn.Module,
    root_model: nn.Module,
    writeback: bool = True,
    names: "ModuleNames | None" = None,
    *,
    rollback: bool = False,
):
    """Context manager for FSDP2 weight access and writeback.

    Gathers sharded DTensor parameters across FSDP/HSDP shards so they can be
    read or modified. Works for both leaf modules (single ``weight``) and
    composite modules like decoder layers (all ``named_parameters``). Only the
    parameters FSDP2 manages are gathered: a DTensor it does not own (an
    expert-parallel weight ``fully_shard`` was told to ignore, say) stays sharded.

    Read-only composite access keeps the enclosing group and every distinct descendant
    FSDP group materialized together until the body exits. Its peak includes all those
    full weights; readers needing one-owner residency must open separate owner scopes.

    Gathered TP weights expose their local tensors in this scope; retained native-full TP views
    remain DTensors on the TP mesh.
    Default writable windows whose subtree contains an FSDP2 registration owner coordinate entry,
    final preparation, body failures, each publication, and final registration restoration. Failed
    bodies make every rank fail and skip writeback.
    Publication failures stop later parameters on every rank; callers restore any values already
    published before failure and repair any registration whose restoration failed.
    Default windows whose subtree contains no FSDP2 registration owner skip coordination. Aliased
    values may already have changed without rollback.
    ``rollback=True`` always coordinates entry, isolates singleton/replicated aliases, and coordinates
    writable staging errors.
    Its additional consensus is per parameter: a failure after one gather stops peers before the next
    gather, and writeback preparation must finish on every rank before each scatter.
    Registration-metadata initialization, coordination-resource allocation, and backend collective
    failures require job termination.
    Default access coordinates preparation only after the final gather. An earlier rank-local
    gather/staging failure can leave peers blocked until backend timeout; terminate rather than retry.
    No undo callback is yielded; callers own restoration of values published before a later failure.

    Writable access rejects temporary post-forward shards in both modes; call ``reshard()`` then
    retry. ``rollback=True`` also rejects native unsharded views. Default access retains existing
    native full views; their materialization scope remains responsible for publishing edits on exit.
    Native-full TP registrations warn that callers must publish edits before ``reshard()``.
    Default writable access requires matching native-full or persistent-shard state on every rank;
    rank-local reshard divergence is unsupported and may stall until the backend timeout.
    ``writeback=False`` materializes either shard state with native unshard/reshard, returning to the
    ordinary persistent shards on exit. This gives up a post-forward state's smaller backward groups.
    An existing native full view is preserved. The read-only route supplies no staging coordination
    or undo callback, even with ``rollback=True``.
    Writable windows require matching sharding groups for all managed parameters;
    open separate windows for nested units with different rank memberships. This is
    checked and coordinated before gathering so consensus covers every gather participant.
    Writable rollback rejects overlap with an active writable window; disjoint nesting is supported.
    Callers must externally serialize entry/exit and use the same window order on every rank. The
    local guard protects temporary registrations and active-window metadata; it detects thread
    contention only. Rank-divergent concurrent calls can block peers until backend timeout and
    require job termination.

    Direct entry and publication expire overlapping Accelerate rollback callbacks, as the dispatcher
    does; disjoint callbacks remain valid. Outside writable consensus, a receipt-inspection error
    conservatively expires all callbacks without raising or changing the native FSDP outcome.

    Raises:
        RuntimeError: Another thread holds the process-local weight-access guard.
    """
    with (
        _weight_access_guard(),
        _expire_accelerate_rollbacks_on_exit(
            module,
            _fsdp2_weight_access_and_writeback_context(
                module, root_model, writeback, names, rollback=rollback
            ),
            fsdp_access=True,
        ),
    ):
        yield


@contextmanager
def _fsdp2_weight_access_and_writeback_context(module, root_model, writeback, names, *, rollback):
    assert not hasattr(module, "_hf_hook"), "We dont support FSDP2 with HF accelerate hooks"
    fsdp_module = _get_enclosing_fsdp_module(module, root_model, names)
    assert fsdp_module is not None, "Module is not wrapped by FSDP"
    if not writeback:
        _invalidate_pending_accelerate_rollbacks(module, strict=False)
        with ExitStack() as stack:
            seen = set()
            # A composite MoE may contain expert FSDP units on another mesh.
            for owner in (
                fsdp_module,
                *(child for child in module.modules() if isinstance(child, FSDPModule)),
            ):
                group = fully_shard.state(owner)._fsdp_param_group
                if group is not None and id(group) not in seen:
                    seen.add(id(group))
                    stack.enter_context(_fsdp2_unshard_context(owner))
            yield
        return

    fsdp_device_mesh = _get_fsdp2_mesh(fsdp_module)
    fsdp_dim = fsdp_device_mesh.ndim
    # Decide participation before fallible parameter discovery, including its cold/refresh path.
    owners = _fsdp2_owned_modules(fsdp_module)
    coordinate_window = rollback or any(child in owners for child in module.modules())

    # Collect all DTensor parameters, replacing them with local replicated copies.
    originals: dict[str, tuple] = {}
    ready = propagating = False
    error = None

    failure_flag = None

    def synchronize_staging_error(local_error, *, coordinate=rollback, mesh=fsdp_device_mesh):
        nonlocal failure_flag
        if coordinate:
            if failure_flag is None:
                failure_flag = torch.zeros(
                    (), dtype=torch.int32, device=fsdp_device_mesh.device_type
                )
            failure_flag.fill_(local_error is not None)
            for dimension in range(mesh.ndim):
                torch.distributed.all_reduce(
                    failure_flag,
                    op=torch.distributed.ReduceOp.MAX,
                    group=mesh.get_group(dimension),
                )
            if failure_flag.item() and local_error is None:
                local_error = RuntimeError("FSDP2 weight-access failed on another rank")
        if local_error is not None:
            raise local_error

    def record_error(exc):
        nonlocal error
        if error is None:
            error = exc
        elif exc is not error:
            error.add_note(f"FSDP2 weight-access cleanup also failed: {exc!r}")

    owned_params = []
    gather_params = []
    preflight_error = None
    try:
        _invalidate_pending_accelerate_rollbacks(module, strict=coordinate_window)
        if (
            rollback
            and _FSDP2_ACTIVE_PARAMS
            and any(
                _fsdp2_owns(active, parameter)
                for parameter in module.parameters()
                for active in _FSDP2_ACTIVE_PARAMS
            )
        ):
            raise ValueError("Writable FSDP2 rollback cannot overlap an active writable window")
        # Recognize new registrations, including native full and post-forward views.
        owned = _fsdp2_owned_params(fsdp_module, refresh=rollback)
        known_unowned = _FSDP2_UNOWNED_PLAIN_PARAMS.get(fsdp_module, {})
        if not rollback and any(
            not _fsdp2_owns(owned, parameter) and not _fsdp2_owns(known_unowned, parameter)
            for parameter in module.parameters()
        ):
            owned = _fsdp2_owned_params(fsdp_module, refresh=True)
        owned_params = [
            (name, parameter)
            for name, parameter in module.named_parameters()
            if _fsdp2_owns(owned, parameter)
        ]
        gather_params = [
            (name, parameter)
            for name, parameter in owned_params
            if isinstance(parameter, torch.distributed.tensor.DTensor)
        ]
        if gather_params and not rollback:
            # Native full TP registrations remain DTensors, on the TP mesh rather than FSDP's.
            native_full = {
                id(parameter.unsharded_param)
                for unit in itertools.chain((fsdp_module,), module.modules())
                if isinstance(unit, FSDPModule)
                and (group := fully_shard.state(unit)._fsdp_param_group) is not None
                and not group.is_sharded
                and not group.is_sharded_post_forward
                for parameter in group.fsdp_params
            }
            if any(id(parameter) in native_full for _, parameter in gather_params):
                warnings.warn(
                    f"FSDP2 owner {type(fsdp_module).__name__} preserves native-full TP parameters "
                    + ", ".join(
                        name for name, parameter in gather_params if id(parameter) in native_full
                    )
                    + "; this window does not publish edits to persistent shards. "
                    "The caller must publish edits before reshard().",
                    RuntimeWarning,
                    stacklevel=2,
                )
            gather_params = [
                (name, parameter)
                for name, parameter in gather_params
                if id(parameter) not in native_full
            ]
        if rollback or gather_params:
            temporary = {
                id(
                    parameter._sharded_post_forward_param
                    if group.is_sharded_post_forward
                    else parameter.unsharded_param
                )
                for unit in itertools.chain((fsdp_module,), module.modules())
                if isinstance(unit, FSDPModule)
                and (group := fully_shard.state(unit)._fsdp_param_group) is not None
                and (group.is_sharded_post_forward or (rollback and group.is_unsharded))
                for parameter in group.fsdp_params
            }
            if temporary and any(id(parameter) in temporary for parameter in module.parameters()):
                raise ValueError(
                    "Writable FSDP2 access requires reshard() before entering the window"
                )
        # A DTensor's trailing mesh dimensions may belong to HSDP rather than TP.
        for _, parameter in owned_params:
            mesh = owned[id(parameter)].mesh
            if mesh.ndim != fsdp_dim or any(
                torch.distributed.get_process_group_ranks(mesh.get_group(dimension))
                != torch.distributed.get_process_group_ranks(fsdp_device_mesh.get_group(dimension))
                for dimension in range(fsdp_dim)
            ):
                raise ValueError(
                    "Writable FSDP2 access requires the same sharding mesh; "
                    "open separate windows for differently sharded nested units"
                )
    except BaseException as exc:
        preflight_error = exc
    # Coordinate temporary-view refusals on stable groups, including rank-asymmetric states.
    entry_mesh = fully_shard.state(fsdp_module)._fsdp_param_group.mesh_info.mesh
    synchronize_staging_error(preflight_error, coordinate=coordinate_window, mesh=entry_mesh)

    active = {id(parameter): weakref.ref(parameter) for _, parameter in owned_params}
    if active:
        _FSDP2_ACTIVE_PARAMS.append(active)
    try:
        for index, (name, param) in enumerate(gather_params):
            original_placements = param.placements
            original_device_mesh = param.device_mesh
            staging_error = None
            try:
                info = owned[id(param)]
                parameter_mesh = info.mesh
                mesh_names, fsdp_names = (
                    original_device_mesh.mesh_dim_names,
                    parameter_mesh.mesh_dim_names,
                )
                if mesh_names is not None and fsdp_names is not None:
                    dimensions = [
                        index for index, axis in enumerate(mesh_names) if axis in fsdp_names
                    ]
                    if not dimensions and not info.unsharded and parameter_mesh.size() > 1:
                        raise ValueError(
                            "The parameter's FSDP axes are absent from its realized mesh"
                        )
                elif info.unsharded:
                    dimensions = []
                elif parameter_mesh.ndim == original_device_mesh.ndim:
                    dimensions = list(range(original_device_mesh.ndim))
                else:
                    raise ValueError("Combined FSDP/TP parameters require explicit mesh axis names")
                placements = list(original_placements)
                for dimension in dimensions:
                    placements[dimension] = Replicate()
                unsharded_dtensor = param.redistribute(
                    placements=placements,
                    device_mesh=original_device_mesh,
                )
                unsharded_tensor = unsharded_dtensor.to_local()
                # Singleton/replicated gathers can alias the persistent shard: isolate body edits.
                if rollback and (
                    unsharded_tensor.untyped_storage().data_ptr()
                    == param.to_local().untyped_storage().data_ptr()
                ):
                    unsharded_dtensor = unsharded_dtensor.clone()
                    unsharded_tensor = unsharded_dtensor.to_local()
                # CPU offload gathers on the host; mirror to the execution device for forward.
                needs_gpu_copy = unsharded_tensor.device.type == "cpu" and torch.cuda.is_available()
                gpu_tensor = (
                    unsharded_tensor.to(torch.cuda.current_device())
                    if needs_gpu_copy
                    else unsharded_tensor
                )
                cpu_writeback_tensor = unsharded_tensor if needs_gpu_copy else None
                originals[name] = (
                    param,
                    unsharded_dtensor,
                    original_placements,
                    original_device_mesh,
                    cpu_writeback_tensor,
                    gpu_tensor,
                )
                staged_param = nn.Parameter(gpu_tensor, requires_grad=param.requires_grad)
                active[id(staged_param)] = weakref.ref(staged_param)
                _set_parameter(module, name, staged_param)
            except BaseException as exc:
                staging_error = exc
            synchronize_staging_error(
                staging_error,
                coordinate=rollback or (index == len(gather_params) - 1 and coordinate_window),
            )
        ready = True
        yield
    except BaseException as exc:
        error, propagating = exc, True
        raise
    finally:
        # A failed last preparation must still match the peers' body-error round.
        if not rollback and coordinate_window:
            try:
                synchronize_staging_error(error, coordinate=True)
            except BaseException as exc:
                record_error(exc)
        if ready and (rollback or error is None):
            # Coordinate after publication too: the final local shard copy may fail on one rank.
            for (
                original_param,
                unsharded_dtensor,
                original_placements,
                original_device_mesh,
                cpu_writeback_tensor,
                gpu_tensor,
            ) in originals.values():
                try:
                    staging_error = error if rollback else None
                    if staging_error is None and cpu_writeback_tensor is not None:
                        try:
                            cpu_writeback_tensor.data.copy_(
                                gpu_tensor.data.to(cpu_writeback_tensor.device)
                            )
                        except BaseException as exc:
                            staging_error = exc
                    synchronize_staging_error(staging_error)
                    # Repartitioning a replicated FSDP dimension is local; peers synchronize below.
                    shard = unsharded_dtensor.redistribute(
                        placements=original_placements, device_mesh=original_device_mesh
                    ).to_local()
                    original_param.to_local().data.copy_(shard)
                except BaseException as exc:
                    record_error(exc)
                try:
                    synchronize_staging_error(error, coordinate=coordinate_window)
                except BaseException as exc:
                    record_error(exc)
                if error is not None:
                    break
        # Entry failures never publish partial temporaries. Every registration is restored
        # independently of entry, body, and repartitioning failures.
        for name, original in originals.items():
            try:
                _set_parameter(module, name, original[0])
            except BaseException as exc:
                record_error(exc)
        try:
            synchronize_staging_error(error, coordinate=coordinate_window)
        except BaseException as exc:
            record_error(exc)
        if active:
            _FSDP2_ACTIVE_PARAMS.pop()
        originals.clear()
        original = staged_param = unsharded_dtensor = unsharded_tensor = None
        cpu_writeback_tensor = gpu_tensor = None
        shard = None
        if error is not None and not propagating:
            raise error


_WEIGHT_ACCESS_LOCK = threading.RLock()


@contextmanager
def _weight_access_guard():
    if not _WEIGHT_ACCESS_LOCK.acquire(blocking=False):
        raise RuntimeError("Weight access and rollback must be serialized across threads")
    try:
        yield
    finally:
        _WEIGHT_ACCESS_LOCK.release()


def _invalidate_pending_accelerate_rollbacks(module, *, strict=True):
    # A pending callback implies the optional plugin is already loaded.
    plugin = sys.modules.get("modelopt.torch.quantization.plugins.accelerate")
    if plugin is not None:
        plugin._expire_overlapping_rollbacks(module, strict=strict)


@contextmanager
def _expire_accelerate_rollbacks_on_exit(module, context, *, fsdp_access=False):
    error = None

    def scan(strict):
        try:
            _invalidate_pending_accelerate_rollbacks(module, strict=strict)
        except Exception as exc:
            if error is None:
                raise
            error.add_note(f"Accelerate rollback receipt inspection failed: {exc!r}")

    try:
        with context as owner_rollback:
            try:
                yield owner_rollback
            except BaseException as exc:
                error = exc
                raise
            finally:
                scan(not fsdp_access)
        error = None
    except BaseException as exc:
        error = exc
        raise
    finally:
        # After FSDP consensus, bookkeeping must not introduce rank-local failure.
        # Clearing all receipts on inspection failure also covers restored shard aliases.
        scan(not fsdp_access)


@contextmanager
def enable_weight_access_and_writeback(
    module,
    root_model,
    names: "ModuleNames | None" = None,
    writeback: bool = True,
    *,
    rollback: bool = False,
):
    """Enable weight access and writeback for a module.

    Useful for modules with weight not intact such as Linear layer in FSDP wrapped model or
    HF accelerate offloaded models (CPU or disk). Under FSDP2 only the parameters FSDP2 manages
    are gathered (see :func:`fsdp2_weight_access_and_writeback_context`).

    Windows and undo callbacks must be serialized across threads in each process. Concurrent
    access raises before entry; same-thread nesting remains supported.

    Args:
        module: The module to access weights for.
        root_model: The root model containing the module.
        names: Pre-built name maps from :func:`module_name_maps`. Without this, every call
            rebuilds them, which is O(N^2) in total when called in a loop -- significant CPU
            overhead on large models, particularly Sparse MoE architectures where each expert
            is its own module.
        writeback: Whether modified weights must be written back to the owning sharded/offload
            representation when exiting the context.
        rollback: Retain aliased CPU backing snapshots for Accelerate and coordinate recoverable
            FSDP2 staging failures before starting another collective in writable windows.
            Read-only FSDP2 access uses native unshard/reshard without this coordination.
            Accelerate's callback restores backing maps, registrations and captured CPU aliases;
            callers still restore already-resident execution values from their own backups,
            including inside outer materialization scopes. Backing maps holding execution-device
            tensors instead of CPU tensors are not snapshotted; callers own their restoration.
            The callback becomes valid after exit
            and expires when overlapping access through any dispatcher route or the direct FSDP2
            context enters or publishes, including read-only or failed access and enclosing
            publication. Stale undo raises before mutation; successful undo stays idempotent.
            Custom offload mappings, including ``dict`` subclasses, are opaque and conservatively
            expire all pending callbacks.
            FSDP2, HF TP and ``nullcontext`` yield no rollback callback; callers own restoration.
    """
    with (
        _weight_access_guard(),
        _enable_weight_access_and_writeback(
            module, root_model, names, writeback, rollback=rollback
        ) as owner_rollback,
    ):
        yield owner_rollback


@contextmanager
def _enable_weight_access_and_writeback(module, root_model, names, writeback, *, rollback):
    fsdp_access = _get_enclosing_fsdp_module(module, root_model, names) is not None
    if not fsdp_access:
        _invalidate_pending_accelerate_rollbacks(module)
    accelerate_access = False
    if fsdp_access:
        context = _fsdp2_weight_access_and_writeback_context(
            module, root_model, writeback, names, rollback=rollback
        )
    elif is_quantized_parallel_linear(module) and hasattr(
        module, "enable_weight_access_and_writeback"
    ):
        # HF transformers TP sharded linear layer
        context = module.enable_weight_access_and_writeback()
    elif any(hasattr(child, "_hf_hook") for child in module.modules()):
        from ..plugins.accelerate import weight_access_and_writeback_context

        accelerate_access = True
        context = weight_access_and_writeback_context(module, writeback, rollback=rollback)
    else:
        context = nullcontext()

    if not accelerate_access:
        context = _expire_accelerate_rollbacks_on_exit(module, context, fsdp_access=fsdp_access)
    with context as owner_rollback:
        yield owner_rollback


def requires_weight_materialization(module, root_model, names: "ModuleNames | None" = None) -> bool:
    """Whether ``module``'s own weights are currently unreadable and need a window.

    Checks the module's own weight owners; the dispatcher also accepts composite modules
    with offloaded descendants. Gated export callers visit descendants separately, so checking
    descendants here would repeatedly materialize containers and their children.
    Two things must hold: the module owns tensors that are
    not directly readable right now
    (offloaded to meta, or a sharded ``DTensor``), and a context exists that can
    materialize them. Modules already materialized are excluded -- re-entering a window
    would re-run export handlers over already-packed weights.
    """
    if not any(
        t is not None and (t.is_meta or isinstance(t, DTensor))
        for t in itertools.chain(module._parameters.values(), module._buffers.values())
    ):
        return False
    if _get_enclosing_fsdp_module(module, root_model, names) is not None:
        return True
    if is_quantized_parallel_linear(module) and hasattr(
        module, "enable_weight_access_and_writeback"
    ):
        return True
    hook = getattr(module, "_hf_hook", None)
    if hook is None:
        return False
    from ..plugins.accelerate import _get_offload_hook

    return _get_offload_hook(hook) is not None


def has_accelerate_offload(module: nn.Module) -> bool:
    """Return True if any module in ``module`` has a CPU- or disk-offload accelerate hook."""
    try:
        from ..plugins.accelerate import _get_offload_hook
    except ImportError:
        return False

    return any(
        _get_offload_hook(getattr(m, "_hf_hook", None)) is not None for m in module.modules()
    )


@contextmanager
def persistent_materialization(layer, writeback: bool = True):
    """Keep all layer weights materialized on GPU for the duration.

    Suppresses per-forward weight transfers so that N calibration batches
    pay the cost of one load/unload instead of N.

    - **FSDP2**: gathers weights once via ``enable_weight_access_and_writeback``,
      then patches ``FSDPParamGroup.unshard/reshard`` to no-ops.
    - **Accelerate**: materializes weights, sets ``hook.offload = False``,
      and bypasses the layer's top-level accelerate hook while the weights are
      materialized.
    """
    with (
        enable_weight_access_and_writeback(layer, layer, writeback=writeback),
        _disable_fsdp_unshard_reshard(layer),
        temporarily_remove_accelerate_hook(layer),
    ):
        yield


def get_quantizer_state_dict(model: nn.Module):
    """Get the state dict of the quantizers in the model."""
    # We should not call model.state_dict() here.
    # With FSDP, model.state_dict() will hang if it is not called from all processes
    from ..nn import TensorQuantizer

    quantizer_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            quantizer_state_dict[get_unwrapped_name(name, model)] = module.state_dict()
    return quantizer_state_dict


def set_quantizer_state_dict(model: nn.Module, quantizer_state_dict: dict):
    """Set the state dict of the quantizers in the model."""
    from ..nn import TensorQuantizer

    for name, module in model.named_modules():
        key = get_unwrapped_name(name, model)
        if isinstance(module, TensorQuantizer) and key in quantizer_state_dict:
            module.load_state_dict(quantizer_state_dict[key])


def sync_moe_expert_amax(experts, sync_weight_amax=False):
    """Sync quantizer amax across MoE experts and fix missing weight amax.

    1. Takes the element-wise max of each ``input_quantizer`` amax across all experts
       and writes it back, so every expert shares the same input amax.
    2. If ``sync_weight_amax`` is True, also syncs ``weight_quantizer`` amax across
       experts (max across experts), so the layer ends up with one effective weight
       scale. Off by default: experts otherwise keep an independent amax each, which
       is also what ``TEGroupedLinear``'s per-expert ``GroupedQuantizer`` does.
    3. For any ``weight_quantizer`` that is enabled but has ``amax is None`` (expert
       received no tokens during calibration), runs a weight-only ``max_calibrate``
       to populate the missing amax.
    """
    from ..model_calib import max_calibrate
    from ..nn import TensorQuantizer

    amax_dict: dict[str, torch.Tensor] = {}
    for expert in experts:
        for name, module in expert.named_modules():
            if not isinstance(module, TensorQuantizer) or module.amax is None:
                continue
            if "input_quantizer" in name or (sync_weight_amax and "weight_quantizer" in name):
                stored_amax = amax_dict.get(name)
                amax_tensor = module.amax.detach().clone()
                amax_dict[name] = (
                    amax_tensor if stored_amax is None else torch.maximum(stored_amax, amax_tensor)
                )

    for expert in experts:
        for name, module in expert.named_modules():
            if isinstance(module, TensorQuantizer) and name in amax_dict:
                module.amax = amax_dict[name].detach().clone()

    for expert in experts:
        for name, module in expert.named_modules():
            if name.endswith("weight_quantizer") and module.is_enabled and module.amax is None:
                weight = expert.state_dict().get(name.replace("weight_quantizer", "weight"))
                if weight is not None:
                    # max_calibrate invokes the forward_loop synchronously, so capturing
                    # ``weight`` by closure (rather than a default arg) is safe and lets mypy
                    # infer the lambda type.
                    max_calibrate(module, lambda m: m(weight), distributed_sync=False)


@contextmanager
def patch_fsdp_mp_dtypes():
    """Patch FSDP2 to handle mixed dtypes properly during quantization.

    This patch is used to relax the requirement of uniform original parameter dtype in FSDP2 and is
    copied from the latest torch FSDP repository `torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py <https://github.com/pytorch/pytorch/blob/c40048472cc4e28f44e8e5835cae319add231bf5/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L227>`_.
    """

    def _init_mp_dtypes(self) -> None:
        """This function is directly copied from the latest version of torch FSDP."""
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_dtype_attrs(self.mp_policy)

        trainable_params: list[FSDPParam] = [
            p for p in self.fsdp_params if p.sharded_param.requires_grad
        ]
        orig_dtypes = {p.orig_dtype for p in trainable_params}
        reduce_dtypes = {p.reduce_dtype for p in trainable_params}

        if len(trainable_params) > 0 and len(orig_dtypes) != 1:
            raise AssertionError(
                f"FSDP expects uniform original parameter dtype but got {orig_dtypes}"
            )

        self._orig_dtype = next(iter(orig_dtypes)) if len(trainable_params) else None

        if len(trainable_params) > 0 and len(reduce_dtypes) != 1:
            raise AssertionError(f"FSDP expects uniform reduce dtype but got {reduce_dtypes}")

        self._reduce_dtype = next(iter(reduce_dtypes)) if len(trainable_params) else None

    # Apply the patch
    original_init_mp_dtypes = (
        torch.distributed.fsdp._fully_shard._fsdp_param_group.FSDPParamGroup._init_mp_dtypes
    )
    try:
        torch.distributed.fsdp._fully_shard._fsdp_param_group.FSDPParamGroup._init_mp_dtypes = (
            _init_mp_dtypes
        )
        yield
    finally:
        torch.distributed.fsdp._fully_shard._fsdp_param_group.FSDPParamGroup._init_mp_dtypes = (
            original_init_mp_dtypes
        )


@contextmanager
def _disable_fsdp_unshard_reshard(layer, *, groups=None):
    """Make the FSDP2 param groups inside *layer* skip unshard/reshard (already gathered).

    Patched per group instance, not on ``FSDPParamGroup``: the rest of the model still runs forward
    (e.g. while collecting the next layer's inputs) and must unshard as usual.
    """
    if not isinstance(layer, FSDPModule):  # nothing was gathered for its FSDP submodules
        yield
        return
    groups = list(_fsdp2_param_groups(layer) if groups is None else groups)
    # Contexts nest (``persistent_materialization(writeback=False)`` enters this twice), so restore
    # whatever instance attributes were there before instead of deleting them.
    saved = [
        (group, {k: group.__dict__.get(k) for k in ("unshard", "reshard")}) for group in groups
    ]
    for group in groups:
        group.unshard = lambda async_op=False: None
        group.reshard = lambda: None
    try:
        yield
    finally:
        for group, prior in saved:
            for name, value in prior.items():
                if value is None:
                    del group.__dict__[name]
                else:
                    setattr(group, name, value)


def build_param_index(model):
    """Map ``id(param)`` to its ``(position, name)`` in ``model.named_parameters()``.

    Lets callers resolve many modules against one walk of the parameters instead of one walk
    each; the position keeps "first in ``named_parameters()`` order" resolvable.
    """
    return {id(param): (i, name) for i, (name, param) in enumerate(model.named_parameters())}


def get_prefixed_param_names(parent_model, target_module, param_index=None):
    """Get parameter names for a target module prefixed with the parent model name.

    This function is used to get full parameter name from FSDPParam module_info which stores the
    unprefixed parameter name.

    Pass ``param_index`` (see :func:`build_param_index`) when resolving many target modules
    against the same parent, so the parent's parameters are walked once rather than per module.
    """
    if param_index is None:
        param_index = build_param_index(parent_model)
    target_ids = {id(p) for p in target_module.parameters()}
    # Lowest position == first in named_parameters() order, matching a linear scan's result.
    match = min((param_index[pid] for pid in target_ids if pid in param_index), default=None)
    # "" for a parameter the parent owns itself.
    return match[1].rpartition(".")[0] if match is not None else None


def _param_key(module_name: str | None, param_name: str) -> str:
    """The full name of ``module_name``'s parameter ``param_name``; ``""`` names the root module.

    A module the name map does not know (``None``) keeps the ``None.`` key no parameter has, so it is
    skipped rather than taken for the root.
    """
    return param_name if module_name == "" else f"{module_name}.{param_name}"


def create_fsdp_param_mapping(fsdp_param_list, model):
    """Builds a mapping from full parameter name to their corresponding FSDPParam.

    Args:
        fsdp_param_list (list): List of FSDPParam.
        model (nn.Module): FSDP root module.

    Returns:
        dict: Full parameter name → FSDP parameter.
    """
    # Built once per call, not once per FSDPParam: export resolves every quantized module, so the
    # per-param walk made this quadratic in (params x modules) and stalled MoE exports for hours.
    # It cannot be cached across calls -- callers swap in quantized params between them.
    param_index = build_param_index(model)
    mapping = {}
    for param in fsdp_param_list:
        # Get the module name
        module_name = get_prefixed_param_names(model, param._module_info.module, param_index)
        if module_name is not None:
            # Get the parameter name from _module_info and construct full param name
            param_name = param._module_info.param_name
            mapping[_param_key(module_name, param_name)] = param
    return mapping


@contextmanager
def no_requires_grad():
    """Context manager to temporarily set requires_grad to False.

    This is used to allow us to call init_sharded_parameter() on the compressed weights. Currently FSDP2 creates
    a new parameter with default requires_grad and then update the requires_grad attribute as needed. This
    triggers an error when torch.nn.Parameter is called on compressed weights as requires_grad cannot be set to True
    for integer tensors.
    """
    original_new = torch.nn.Parameter.__new__

    def patched_new(cls, data=None, requires_grad=True):
        return original_new(cls, data, requires_grad=False)

    torch.nn.Parameter.__new__ = patched_new
    try:
        yield
    finally:
        torch.nn.Parameter.__new__ = original_new


@contextmanager
def enable_fake_quant(module):
    """Temporarily set the fake_quant attribute of a module to True.

    This is used to prevent weight compression from being triggered during an unshard() call. A
    sequential weight quantizer (W4A8) holds the flag on each of its stages.
    """
    from ..nn import TensorQuantizer  # local: ..nn imports this utils package at module scope

    quantizers = [
        q
        for m in module.modules()
        if hasattr(m, "weight_quantizer")
        for q in m.weight_quantizer.modules()
        if isinstance(q, TensorQuantizer)
    ]
    original_fake_quant = [q._fake_quant for q in quantizers]
    for q in quantizers:
        q._fake_quant = True
    try:
        yield
    finally:
        for q, fake_quant in zip(quantizers, original_fake_quant):
            q._fake_quant = fake_quant


@contextmanager
def enable_quant(quantizer):
    """Temporarily enable quantization for a quantizer.

    Args:
        quantizer: The quantizer module to enable quantization for.
    """
    original_if_quant = quantizer._if_quant
    quantizer._if_quant = True
    try:
        yield
    finally:
        quantizer._if_quant = original_if_quant


def apply_without_qdq(quantizer, inputs: torch.Tensor) -> torch.Tensor:
    """Apply ``quantizer``'s pre-quant scale and rotation to ``inputs``, without quantizing them.

    For rows a quantized module computes in high precision, so they keep the transforms (e.g.
    an AWQ / SmoothQuant scale that the stored weight compensates for) of the quantized rows.
    """
    from ..nn import SequentialQuantizer  # local import: ..nn imports this module

    members = list(quantizer) if isinstance(quantizer, SequentialQuantizer) else [quantizer]
    for member in members:
        disabled = member._disabled
        member._disabled = True
        try:
            inputs = member(inputs)
        finally:
            member._disabled = disabled
    return inputs


@contextmanager
def disable_calib(quantizer):
    """Temporarily disable calibration for a quantizer.

    Args:
        quantizer: The quantizer module to disable calibration for.
    """
    original_if_calib = quantizer._if_calib
    quantizer._if_calib = False
    try:
        yield
    finally:
        quantizer._if_calib = original_if_calib


@contextmanager
def fsdp2_aware_weight_update(
    root_model, modules_to_update, reshard=True, names: "ModuleNames | None" = None
):
    """Context manager to update the FSDPParam list if an update is made to a submodule of an FSDPModule.

    This context manager is to be used when updating a weight of a sharded module to ensure the changes are properly
    reflected for future unsharding and resharding the FSDP root module. The context manager will unshard the FSDP root
    module, register new FSDPParam/QFSDPParam for the updated modules and updates the FSDP param group list.

    If reshard is True, the context manager will also reshard the FSDP root module after the weight update.

    Args:
        root_model (nn.Module): The root model of the FSDPModule.
        modules_to_update (list): The list of modules to update which should be a list of modules that are
            direct children of the FSDPModule. A list is read again when the context exits, so a caller
            may append modules of the same FSDP unit to it inside the context, and their weights are
            re-registered with FSDP2 too (the export's held weight windows rely on this).
        reshard (bool): Whether to reshard the FSDP root module after the weight update.

    Returns:
        None
    """
    # Bound only once setup succeeds; the finally skips FSDP bookkeeping until then so an
    # early error (e.g. an OOM from the unshard below) propagates and FSDP state stays intact.
    fsdp_param_group = None
    try:
        if isinstance(root_model, FSDPModule):
            # Get FSDP root module, if none is returned, then the update is not made to a submodule of an FSDPModule
            if not isinstance(modules_to_update, list):
                modules_to_update = [modules_to_update]

            root_modules = set()
            for module in modules_to_update:
                root_module = _get_enclosing_fsdp_module(module, root_model, names)
                root_modules.add(root_module)

            # Ensure all modules in root_modules are the same
            assert len(root_modules) == 1, "All modules must be in the same root FSDPModule"
            root_module = next(iter(root_modules))

            # Check if root module state is sharded and unshard if needed
            if fully_shard.state(root_module)._fsdp_param_group.is_sharded:
                with enable_fake_quant(root_module):
                    root_module.unshard()

            # Get FSDPParam list
            param_group = fully_shard.state(root_module)._fsdp_param_group
            fsdp_param_mapping = create_fsdp_param_mapping(param_group.fsdp_params, root_model)
            fsdp_param_group = param_group

            # Assert that all the modules in the module list are present in this fsdp_param_group
            if len(modules_to_update) > 1:
                for module in modules_to_update:
                    module_name = _get_module_name(module, root_model, names)
                    # Check if any parameter from this module is in the mapping
                    module_params_in_mapping = any(
                        _param_key(module_name, n) in fsdp_param_mapping
                        for n, _ in module.named_parameters()
                    )
                    assert module_params_in_mapping, (
                        f"Module {module} with name '{module_name}' not found in fsdp_param_mapping. "
                        f"Available keys: {list(fsdp_param_mapping.keys())}"
                    )
        # Yields for necessary weight updates/processing
        yield
    finally:
        from modelopt.torch.quantization.qtensor.base_qtensor import QFSDPParam, QTensorWrapper

        if fsdp_param_group is not None:
            # Update FSDPParam list
            for module in modules_to_update:
                for param_name, param in module.named_parameters():
                    name = _param_key(_get_module_name(module, root_model, names), param_name)
                    if name not in fsdp_param_mapping:
                        continue

                    old_fsdp_param = fsdp_param_mapping[name]

                    # Update mp policy to reflect the new dtype
                    new_mp_policy = MixedPrecisionPolicy(
                        param_dtype=param.dtype,
                        reduce_dtype=None,
                        output_dtype=None,
                        cast_forward_inputs=False,
                    )

                    with no_requires_grad(), enable_fake_quant(module):
                        # Create a new QFSDPParam or FSDPParam based on weight type
                        param_class = QFSDPParam if isinstance(param, QTensorWrapper) else FSDPParam

                        new_param = param_class(
                            param,
                            old_fsdp_param._module_info,
                            old_fsdp_param.mesh_info,
                            old_fsdp_param.post_forward_mesh_info,
                            old_fsdp_param.device,
                            None,
                            new_mp_policy,
                            None,
                        )
                        if not isinstance(new_param, QFSDPParam):
                            new_param.init_dtype_attrs(new_mp_policy)

                        # Update the FSDPParam mapping to keep track of the new FSDPParam
                        fsdp_param_mapping[name] = new_param

                        # Remove the post_load_hook_handle to allow gc to collect the old FSDPParam
                        old_fsdp_param._post_load_hook_handle.remove()

            # Update FSDPParam list with new compressed weights
            fsdp_param_group.fsdp_params = list(fsdp_param_mapping.values())

            # Reshard FSDP root module
            if reshard:
                with enable_fake_quant(root_module):
                    root_module.reshard()


def update_quant_cfg_with_kv_cache_quant(
    quant_cfg: dict[str, Any], kv_cache_quant_cfg: list[QuantizerCfgEntry]
) -> dict[str, Any]:
    """Update the quant_cfg with the kv cache quant_cfg.

    Args:
        quant_cfg: The outer quantization config dict (with ``"quant_cfg"`` and ``"algorithm"`` keys).
        kv_cache_quant_cfg: A list of :class:`QuantizerCfgEntry
            <modelopt.torch.quantization.config.QuantizerCfgEntry>` dicts for KV cache quantization,
            typically ``some_kv_cfg["quant_cfg"]``.

    Returns:
        A deep copy of ``quant_cfg`` with the KV cache entries appended to ``quant_cfg["quant_cfg"]``.
    """
    # If quant_cfg["quant_cfg"] is None, it corresponds to only kv cache quantization case
    quant_cfg = copy.deepcopy(quant_cfg)
    inner: list[QuantizerCfgEntry] = quant_cfg.get("quant_cfg") or [
        {"quantizer_name": "*", "enable": False}
    ]
    quant_cfg["quant_cfg"] = inner + list(kv_cache_quant_cfg)

    # Set default algorithm for kv cache quantization if not provided.
    if not quant_cfg.get("algorithm"):
        quant_cfg["algorithm"] = "max"
    print_rank_0(f"Updated quant_cfg with KV cache quantization: {quant_cfg}")
    return quant_cfg


def promote_static_block_weight_quantizers(model: nn.Module) -> int:
    """Convert eligible static-block weight TensorQuantizers in-place.

    After max calibration sets per-block amax values, NVFP4 static quantizers
    need to be promoted so they use the two-level scaling path (global amax +
    per-block amax) instead of the generic E4M3 path.

    If the quantizer belongs to a shared state with a populated ``global_amax`` inside
    ``model``, the promoted quantizer's ``_global_amax`` buffer is tied to that canonical
    state buffer instead of receiving an independent copy.

    Returns the number of NVFP4 quantizers converted.
    """
    from modelopt.torch.quantization.nn import (
        QuantModule,
        SequentialQuantizer,
        StaticBlockScaleQuantizer,
        TensorQuantizer,
    )
    from modelopt.torch.quantization.utils.shared_input import (
        SharedWeightGlobalAmaxState,
        iter_shared_quant_states,
    )

    # Shared states owned within THIS promotion root. This function also runs on submodules
    # / individual linears, so build the reverse lookup only from states reachable here.
    shared_by_quantizer = {
        id(quantizer): state
        for state in iter_shared_quant_states(model, SharedWeightGlobalAmaxState)
        for quantizer in state._member_quantizers()
    }
    converted = 0
    # Quantizer-only iteration: this loop reads quantizer state and never the weight, and asking
    # for weights here costs one DTensor collective per expert on a fused-MoE model under FSDP2.
    warned_dtensor_amax = False
    for _name, module in list(model.named_modules()):
        if not isinstance(module, QuantModule):
            continue
        for quantizer in module.iter_weight_quantizers_for_calibration():
            if isinstance(quantizer, SequentialQuantizer):
                if len(quantizer) == 0:
                    continue
                quantizer = quantizer[0]
            if not isinstance(quantizer, TensorQuantizer):
                continue
            quantizer_id = id(quantizer)
            if not quantizer.is_enabled or not quantizer.is_static_block_quant:
                continue
            amax = quantizer.amax
            if amax is None:
                continue
            if isinstance(amax, DTensor) and not warned_dtensor_amax:
                # _amax is a buffer, so FSDP2 leaves it replicated and reduce_amax below stays
                # local. If that ever stops holding, the reduction becomes a per-quantizer
                # collective and this loop needs a batched reduction instead.
                warned_dtensor_amax = True
                warnings.warn(
                    "promote_static_block_weight_quantizers: _amax is a DTensor, so the "
                    "per-quantizer global-amax reduction is a collective. Batch the reduction "
                    "before running this at scale."
                )
            if quantizer.is_nvfp4_static:
                # Grouped siblings share one canonical global_amax (common FP8 grid); otherwise
                # fall back to this quantizer's own per-block amax.
                already_promoted = isinstance(quantizer, StaticBlockScaleQuantizer)
                shared = shared_by_quantizer.get(quantizer_id)
                if shared is not None and shared.global_amax is not None:
                    StaticBlockScaleQuantizer.from_tensor_quantizer(quantizer)
                    shared.tie_member_quantizer(quantizer)
                else:
                    if shared is not None and not amax.is_meta:
                        raise RuntimeError(
                            f"{_name}: weight quantizer is in a shared group whose global_amax was "
                            "not populated before promotion; run populate after calibration so "
                            "siblings share one scale instead of falling back to their own."
                        )
                    global_amax = reduce_amax(amax.clone().detach(), axis=None)
                    StaticBlockScaleQuantizer.from_tensor_quantizer(
                        quantizer, global_amax=global_amax
                    )
                if not already_promoted:
                    converted += 1
            elif isinstance(quantizer._num_bits, int):
                # Integer static-block weights are promoted so LSQ can use
                # StaticBlockScaleQuantizer.
                StaticBlockScaleQuantizer.from_tensor_quantizer(quantizer)
    return converted


def promote_nvfp4_static_quantizers(model: nn.Module) -> int:
    """Compatibility wrapper for static-block weight quantizer promotion."""
    return promote_static_block_weight_quantizers(model)
