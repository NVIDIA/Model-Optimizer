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

"""Native HF plans and overlapping dense/expert mesh ownership."""

import copy
import inspect
from functools import wraps

import torch
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from modelopt.torch.quantization.nn import QuantModule, SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.qtensor import QTensorWrapper
from modelopt.torch.quantization.repair import repair_backend_supports_tp
from modelopt.torch.quantization.repair_partition import (
    NativeWeightPartition,
    prepare_packed_weights_for_tp,
    validate_activation_partition,
    validate_native_weight_partition,
    validate_tp_output_quantizer,
)

from ...local_parallel import _all_rank_values
from ...local_runtime import RealizedMeshContext, prepare_quantizer_partitions
from .functional_fsdp import extend_family_fsdp_plan, install_functional_fsdp_calls

__all__ = []


def _axis_rank_groups(mesh, axis):
    index = mesh.mesh_dim_names.index(axis)
    groups = mesh.mesh.movedim(index, -1).reshape(-1, mesh.mesh.shape[index])
    return sorted(tuple(group) for group in groups.tolist())


def native_mesh_context(manager):
    """Reuse caller HF views and combine optional CP with each field's replica axes."""
    try:
        cp = manager.get_mesh("cp")
    except KeyError:
        cp = None
    try:
        expert_cp = manager.get_mesh(("pp", "cp", "efsdp", "ep"))
    except KeyError:
        expert_cp = None
    if (cp is None) != (expert_cp is None):
        raise ValueError("Native dense and expert meshes must both expose the caller's CP axis.")
    dense = manager.get_mesh(("pp", "cp", "fsdp", "tp") if cp is not None else ("pp", "fsdp", "tp"))
    expert = expert_cp if cp is not None else manager.get_mesh(("pp", "efsdp", "ep"))
    if not torch.equal(dense.mesh.flatten().sort().values, expert.mesh.flatten().sort().values):
        raise ValueError("Native HF dense and expert meshes must contain the same model ranks.")
    if cp is not None and _axis_rank_groups(dense, "cp") != _axis_rank_groups(expert, "cp"):
        raise ValueError(
            "Native dense and expert meshes must use matching context parallel rank groups."
        )
    data = manager.get_mesh("fsdp")
    tensor = manager.get_mesh("tp")
    owners = manager.get_mesh("ep")
    replicas = manager.get_mesh("efsdp")
    if (
        owners.size() > 1 and owners.size() % tensor.size()
    ) or owners.size() * replicas.size() != data.size() * tensor.size():
        raise ValueError(
            "Native expert mesh must satisfy efsdp * ep == fsdp * tp and ep % tp == 0."
        )
    data_context = data
    if cp is not None:
        data_context = dense[("cp", "fsdp")]._flatten("local_repair_native_data_context")
        replicas = expert[("cp", "efsdp")]._flatten("local_repair_native_expert_replicas")
    return RealizedMeshContext(
        device_mesh=dense,
        moe_mesh=expert,
        process_group=None,
        data_mesh=data,
        data_context_mesh=data_context,
        tensor_mesh=tensor,
        expert_mesh=owners,
        expert_replica_mesh=replicas,
        pipeline_mesh=manager.get_mesh("pp"),
        context_mesh=cp,
        mesh_manager=manager,
    )


def apply_hf_tensor_parallelism(model, mesh, plan=None):
    """Apply the installed HF TP API, preserving the frozen two-argument interface."""
    from transformers.distributed.tensor_parallel import apply_tensor_parallelism

    if "tp_plan" not in inspect.signature(apply_tensor_parallelism).parameters:
        if plan is not None:
            raise ValueError("Explicit resolved plans require the newer native HF parallel API.")
        return apply_tensor_parallelism(model, mesh)
    if plan is None:
        if not getattr(model, "tp_plan", None):
            raise ValueError("Tensor parallelism requires a model-native TP plan.")
        plan = model._expand_parallel_plan(model.tp_plan)
    return apply_tensor_parallelism(model, mesh, plan)


def validate_hf_tensor_parallelism(model, mesh, plan=None):
    """Validate resolved HF physical ownership before packed staging or native hooks."""
    from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES

    modules = dict(model.named_modules())
    if plan is None:
        if hasattr(model, "_expand_parallel_plan"):
            plan = model._expand_parallel_plan(model.tp_plan)
        else:
            from transformers.distributed.tensor_parallel import _get_parameter_tp_plan

            plan = {}
            for owner, module in modules.items():
                for name, _ in module.named_parameters(recurse=False):
                    full = f"{owner}.{name}" if owner else name
                    style = _get_parameter_tp_plan(
                        parameter_name=full, tp_plan=model.tp_plan, is_weight=True
                    )
                    if style is not None:
                        plan[full] = style
    parameters = dict(model.named_parameters())
    for full, style in plan.items():
        if full in parameters:
            owner, _, name = full.rpartition(".")
            _validate_parameter_geometry(
                modules[owner], name, ALL_PARALLEL_STYLES[style], mesh, full
            )


def _synchronize_validation(context, error):
    errors = sorted({message for message in _all_rank_values(context, error) if message})
    if errors:
        raise ValueError("; ".join(errors))


def _expert_partitions(model, plan, context, adopted):
    modules = dict(model.named_modules())
    partitions, global_ids = [], {}
    size, rank = context.expert_mesh.size(), context.expert_mesh.get_local_rank()
    for name, style in plan.items():
        if style not in ("moe_tp_experts", "ep_dispatch_experts"):
            continue
        module = modules[name]
        if adopted and not getattr(module, "_is_expert_parallel", False):
            raise ValueError(
                "An adopted native model must already have the requested expert placement."
            )
        weights = [p for p in module.parameters(recurse=False) if p.ndim >= 3]
        if not weights:
            raise ValueError("Native HF expert plans require fused parameters with an expert axis.")
        count = weights[0].shape[0]
        if count % size:
            raise ValueError("Native expert counts must divide evenly across their owner mesh.")
        width, start = count // size, rank * (count // size)
        global_ids[name] = tuple(range(start, start + width))
        if not hasattr(module, "_first_proj_weight_quantizers_attr"):
            continue
        for attribute in (module._first_proj_weight_quantizers_attr, "down_proj_weight_quantizers"):
            quantizers = getattr(module, attribute)
            if adopted and len(quantizers) == width:
                continue
            if len(quantizers) != count:
                raise ValueError(
                    "Calibrate every native expert quantizer before partitioning its list."
                )
            partitions.append((module, attribute, quantizers[start : start + width]))
    return partitions, global_ids


def _connect_masked_inputs(module):
    # Every rank must visit native input-gradient collectives, including an unused expert owner.
    original = module.forward

    @wraps(original)
    def connected(*args, **kwargs):
        output = original(*args, **kwargs)
        for value in (*args, *kwargs.values()):
            if isinstance(value, torch.Tensor) and value.requires_grad:
                output = output + _zero_link(value)
        for parameter in module.parameters():
            if parameter.requires_grad:
                output = output + _zero_link(parameter)
                break
        return output

    module.forward = connected


def _zero_link(value):
    scalar = value.sum() if value.numel() == 0 else value[(0,) * value.ndim]
    return scalar * 0


def _validate_parameter_geometry(module, name, style, mesh, full_name):
    """Resolve the installed HF style on a storage-free logical parameter."""
    parameter = module._parameters[name]
    native = parameter.get_qtensor() if isinstance(parameter, QTensorWrapper) else None
    shape = native.metadata["shape"] if native is not None else parameter.shape
    dtype = native.metadata["dtype"] if native is not None else parameter.dtype
    probe = copy.copy(module)
    probe._parameters = dict(module._parameters)
    probe._buffers = dict(module._buffers)
    probe._modules = dict(module._modules)
    probe._parameters[name] = torch.nn.Parameter(
        torch.empty(shape, dtype=dtype, device="meta"), requires_grad=False
    )
    style.validate_param(probe, name, mesh, parameter_name=full_name)
    style.shard_param(probe, name, mesh)
    sharded = probe._parameters[name]
    if not isinstance(module, QuantModule) or not isinstance(sharded, DTensor):
        return
    placement = sharded.placements[0]
    if not isinstance(placement, Shard):
        return
    if hasattr(module, "_first_proj_weight_quantizers_attr"):
        bindings = {
            module._first_proj_attr: (
                module._first_proj_weight_quantizers_attr,
                module._first_proj_input_quantizer_attr,
            ),
            "down_proj": ("down_proj_weight_quantizers", "down_proj_input_quantizer"),
        }
    else:
        bindings = {"weight": ("weight_quantizer", "input_quantizer")}
    if name not in bindings:
        return
    fused = len(shape) == 3 and name != "weight"
    dim = placement.dim % len(shape) - int(fused)
    if dim not in (0, 1) or len(shape) != 2 + int(fused):
        # Expert-axis ownership is handled by the separately validated quantizer-list map.
        if fused and dim == -1:
            return
        raise ValueError("Native TP fields require physical matrix or fused expert views.")
    view = NativeWeightPartition(
        tuple(shape[int(fused) :]),
        dim,
        mesh.get_local_rank(),
        mesh.size(),
        placement.split_factor if isinstance(placement, _StridedShard) else 1,
    )
    validate_tp_output_quantizer(getattr(module, "output_quantizer", None), view)
    quantizers_name, input_name = bindings[name]
    container = getattr(module, quantizers_name)
    members = (
        list(container)
        if isinstance(container, (SequentialQuantizer, torch.nn.ModuleList))
        else [container]
    )
    for quantizer in members:
        if not isinstance(quantizer, TensorQuantizer):
            raise ValueError("Composite TP weights require an explicit physical-view adapter.")
        if quantizer.is_enabled:
            if not repair_backend_supports_tp(quantizer.backend):
                raise ValueError(f"Backend {quantizer.backend!r} requires a native TP partitioner.")
            validate_native_weight_partition(quantizer, view)
    activation = getattr(module, input_name, None)
    if isinstance(activation, TensorQuantizer):
        validate_activation_partition(activation, view)


def parallelize_native_transformers(model, context, *, initialize=None):
    """Apply explicit native TP/EP plans and prepare repair fields before FSDP2.

    Return per-model plans, expert IDs, packed representation and prepared field
    ownership; teacher and student never overwrite each other's metadata.
    """
    from transformers.distributed.configuration_utils import DistributedConfig
    from transformers.distributed.expert_parallel import apply_expert_parallelism
    from transformers.distributed.fsdp import apply_fully_sharded_data_parallelism
    from transformers.distributed.pipeline_parallel import apply_pipeline_parallelism
    from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES

    if id(model) in context.parallel_metadata:
        raise ValueError("This model has already been prepared on the native HF meshes.")
    adopted = getattr(model, "_mesh_manager", None) is not None
    config = copy.copy(getattr(model.config, "distributed_config", None))
    if config is None:
        config = DistributedConfig(
            tp_size=context.tensor_mesh.size(),
            fsdp_size=context.data_mesh.size(),
            pp_size=context.pipeline_mesh.size(),
            ep_size=context.expert_mesh.size(),
        )
    original_plans = {
        name: (name in model.__dict__, getattr(model, name, None))
        for name in ("_tp_plan", "_ep_plan", "_fsdp_plan")
    }
    error, plans, partitions, global_ids = None, None, None, None
    try:
        if initialize is not None and (not callable(initialize) or adopted):
            raise ValueError("Initialization requires a callable and an undistributed model.")
        if adopted:
            current = model._mesh_manager
            for axes in (context.device_mesh.mesh_dim_names, context.moe_mesh.mesh_dim_names):
                if not torch.equal(
                    current.get_mesh(axes).mesh, context.mesh_manager.get_mesh(axes).mesh
                ):
                    raise ValueError(
                        "Adopted native model meshes must match the caller's realized rank ordering."
                    )
        for name, mesh in (
            ("tp", context.tensor_mesh),
            ("fsdp", context.data_mesh),
            ("pp", context.pipeline_mesh),
            ("ep", context.expert_mesh),
        ):
            if getattr(config, name + "_size") != mesh.size():
                raise ValueError("Native HF config sizes must match the caller's realized meshes.")
        plans = model._resolve_parallel_plans(config)
        tp_plan, ep_plan, fsdp_plan = plans
        fsdp_plan = extend_family_fsdp_plan(model, fsdp_plan)
        plans = tp_plan, ep_plan, fsdp_plan
        for axis, plan, mesh in (
            ("TP", tp_plan, context.tensor_mesh),
            ("EP", ep_plan, context.expert_mesh),
            ("FSDP", fsdp_plan, context.data_mesh),
        ):
            if mesh.size() > 1 and not plan:
                raise ValueError(
                    f"Native HF resolved {axis} plan is empty for requested size {mesh.size()}."
                )
        if "megamoe_experts" in ep_plan.values():
            raise ValueError(
                "Native Mega MoE inference kernels lack local-repair autograd; select eager or grouped_mm."
            )
        dispatch_overrides = {}
        for name, style in ep_plan.items():
            if style != "ep_dispatch_experts":
                continue
            module = model.get_submodule(name)
            strategy = getattr(module, "_modelopt_ep_dispatch", None)
            if strategy is None:
                continue
            if not callable(getattr(strategy, "install_forward", None)):
                raise ValueError("A model-owned EP dispatcher must implement install_forward.")
            if adopted and not getattr(module, "_modelopt_custom_ep_dispatch", False):
                raise ValueError("Adopting this model requires its model-owned EP dispatcher.")
            dispatch_overrides[name] = strategy
        if adopted:
            modules = dict(model.named_modules())
            if any(
                style == "moe_tp_experts"
                and not getattr(modules[name], "_modelopt_safe_masked_ep", False)
                for name, style in ep_plan.items()
            ):
                raise ValueError(
                    "Adopting masked expert parallelism requires safe masked wrappers installed by ModelOpt; "
                    "prepare the undistributed model with this adapter first."
                )
        partitions, global_ids = _expert_partitions(model, ep_plan, context, adopted)
        if not adopted:
            modules, parameters = dict(model.named_modules()), dict(model.named_parameters())
            for plan, mesh in ((tp_plan, context.tensor_mesh), (ep_plan, context.expert_mesh)):
                for name, style in plan.items():
                    if name in parameters:
                        owner, _, parameter = name.rpartition(".")
                        _validate_parameter_geometry(
                            modules[owner], parameter, ALL_PARALLEL_STYLES[style], mesh, name
                        )
    except (ValueError, AttributeError, KeyError, RuntimeError, TypeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    try:
        _synchronize_validation(context, error)
    except ValueError:
        for name, (owned, plan) in original_plans.items():
            if owned:
                setattr(model, name, plan)
            else:
                model.__dict__.pop(name, None)
        raise
    tp_plan, ep_plan, fsdp_plan = plans
    route = (
        "dispatch" if "ep_dispatch_experts" in ep_plan.values() else "masked" if ep_plan else None
    )
    metadata = {
        "expert_global_ids": global_ids,
        "expert_mode": route,
        "plans": {"tp": tp_plan, "ep": ep_plan, "fsdp": fsdp_plan},
    }
    model.config.distributed_config = config
    if not adopted:
        prepare_packed_weights_for_tp(model)
        if context.pipeline_mesh.size() > 1:
            apply_pipeline_parallelism(model, context.pipeline_mesh)
        if context.tensor_mesh.size() > 1:
            apply_hf_tensor_parallelism(model, context.tensor_mesh, tp_plan)
    for module, name, quantizers in partitions:
        setattr(module, name, quantizers)
    if not adopted and ep_plan:
        if route == "dispatch":
            native_plan = {
                name: style for name, style in ep_plan.items() if name not in dispatch_overrides
            }
            apply_expert_parallelism(model, context.expert_mesh, context.tensor_mesh, native_plan)
            for name, strategy in dispatch_overrides.items():
                module = model.get_submodule(name)
                strategy.install_forward(
                    module, ep_mesh=context.expert_mesh, tp_mesh=context.tensor_mesh
                )
                module._is_hooked = True
                module._modelopt_custom_ep_dispatch = True
        else:
            wrappers = {name: style for name, style in ep_plan.items() if style == "moe_tp_experts"}
            modules = dict(model.named_modules())
            for name in wrappers:
                _connect_masked_inputs(modules[name])
            apply_hf_tensor_parallelism(
                model,
                context.expert_mesh,
                {name: style for name, style in ep_plan.items() if name not in wrappers},
            )
            for name in wrappers:
                ALL_PARALLEL_STYLES["moe_tp_experts"].install_forward(
                    modules[name],
                    context.expert_mesh,
                    is_expert_parallel=True,
                )
                modules[name]._modelopt_safe_masked_ep = True
    if initialize is not None:
        error = None
        try:
            initialize(model)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _synchronize_validation(context, error)
    context.tp_quantizer_layouts[id(model)] = prepare_quantizer_partitions(model, context)
    if not adopted and context.data_mesh.size() > 1:
        apply_fully_sharded_data_parallelism(model, context.mesh_manager, fsdp_plan, ep_plan)
    install_functional_fsdp_calls(model)
    model._mesh_manager = context.mesh_manager
    model._device_mesh = context.device_mesh
    model._tp_size, model._fsdp_size = config.tp_size, config.fsdp_size
    model._is_distributed_loading_by_transformers = True
    context.parallel_metadata[id(model)] = metadata
    return metadata
