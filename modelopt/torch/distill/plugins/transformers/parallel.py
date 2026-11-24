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
import re

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard

from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.qtensor import QTensorWrapper
from modelopt.torch.quantization.repair_partition import (
    native_tp_weight_view,
    prepare_packed_weights_for_tp,
    quantized_weight_bindings,
    validate_native_weight_partition,
    validate_tp_output_quantizer,
)

from ..._bypass_parallel import (
    BypassMeshContext,
    _all_rank_values,
    _size,
    plan_tp_input_quantizer_fields,
    prepare_quantizer_partitions,
    validate_tp_quantizer_aliases,
    validate_tp_replicated_inputs,
)
from ..._bypass_parallel import torch_mesh_context as torch_mesh_context

__all__ = []


def _axis_rank_groups(mesh, axis):
    index = mesh.mesh_dim_names.index(axis)
    groups = mesh.mesh.movedim(index, -1).reshape(-1, mesh.mesh.shape[index])
    return sorted(tuple(group) for group in groups.tolist())


def _optional(manager, dims):
    try:
        return manager.get_mesh(dims)
    except (KeyError, RuntimeError, ValueError):
        return None


def _combine(mesh, axes, name):
    active = [axis for axis in axes if axis in mesh.mesh_dim_names and mesh[axis].size() > 1]
    active = active or [next(axis for axis in axes if axis in mesh.mesh_dim_names)]
    return mesh[active[0]] if len(active) == 1 else mesh[tuple(active)]._flatten(name)


def native_mesh_context(manager):
    """Use HF's overlapping meshes; DP replicas and FSDP both consume unique data."""
    cp, expert_cp = _optional(manager, "cp"), _optional(manager, ("cp", "efsdp", "ep"))
    if (cp is None) != (expert_cp is None):
        raise ValueError("Native dense and expert meshes must expose the same CP axis")
    replicate = _optional(manager, "dp_replicate")
    dense_axes = (
        ("pp",)
        + (("dp_replicate",) if replicate is not None else ())
        + (("cp",) if cp is not None else ())
        + ("fsdp", "tp")
    )
    expert_axes = (
        ("pp",)
        + (("dp_replicate",) if replicate is not None else ())
        + (("cp",) if cp is not None else ())
        + ("efsdp", "ep")
    )
    dense, expert = manager.get_mesh(dense_axes), manager.get_mesh(expert_axes)
    if not torch.equal(dense.mesh.flatten().sort().values, expert.mesh.flatten().sort().values):
        raise ValueError("Dense and expert meshes must contain the same model ranks")
    if cp is not None and _axis_rank_groups(dense, "cp") != _axis_rank_groups(expert, "cp"):
        raise ValueError("Dense and expert CP rank groups must match")
    fsdp, tp, ep, efsdp = (manager.get_mesh(axis) for axis in ("fsdp", "tp", "ep", "efsdp"))
    if ep.size() * efsdp.size() != fsdp.size() * tp.size() or (
        ep.size() > 1 and ep.size() % tp.size()
    ):
        raise ValueError("Native meshes require efsdp * ep == fsdp * tp and ep divisible by tp")
    data = _combine(dense, ("dp_replicate", "fsdp"), "modelopt_bypass_data")
    data_context = _combine(dense, ("cp", "dp_replicate", "fsdp"), "modelopt_bypass_data_context")
    replicas = _combine(expert, ("cp", "dp_replicate", "efsdp"), "modelopt_bypass_expert_replicas")
    return BypassMeshContext(
        device_mesh=dense,
        moe_mesh=expert,
        data_mesh=data,
        data_context_mesh=data_context,
        tensor_mesh=tp,
        expert_mesh=ep,
        expert_replica_mesh=replicas,
        pipeline_mesh=manager.get_mesh("pp"),
        context_mesh=cp,
        fsdp_mesh=fsdp,
        replica_mesh=replicate,
        mesh_manager=manager,
    )


def apply_hf_tensor_parallelism(model, mesh, plan=None):
    """Apply the exact stable-5.23 native TP interface."""
    from transformers.distributed.tensor_parallel import apply_tensor_parallelism

    plan = model._expand_parallel_plan(model.tp_plan) if plan is None else plan
    return apply_tensor_parallelism(model, mesh, plan)


def _tp_output_layouts(model, plan):
    """Normalize exact native module output declarations, never weight ownership."""
    from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES

    modules, outputs = dict(model.named_modules()), {}
    for path, name in plan.items():
        if path not in modules:
            continue
        style = ALL_PARALLEL_STYLES[name]
        if name in (
            "colwise",
            "colwise_rep",
            "colwise_gather_output",
            "rowwise",
            "rowwise_rep",
            "rowwise_split_input",
        ):
            layout = style.output_layouts
        elif name == "packed_colwise":
            layout = Shard(-1)
        elif name == "sequence_parallel":
            layout = Shard(style.sequence_dim)
        elif name in (
            "all_reduce",
            "moe_tp_experts",
            "replicate",
            "replicated_with_grad_allreduce",
        ):
            layout = Replicate()
        else:
            layout = f"Unsupported native TP boundary output style {name!r}"
        outputs[path] = layout
    return outputs


def validate_hf_tensor_parallelism(model, mesh, plan=None):
    """Validate resolved HF physical ownership before packed staging or native hooks."""
    from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES

    modules = dict(model.named_modules(remove_duplicate=False))
    expanded = plan is None
    if expanded:
        plan = model._expand_parallel_plan(model.tp_plan)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    aliases, placed_inputs = {}, set()
    for full, style in plan.items():
        if full in parameters:
            owner, _, name = full.rpartition(".")
            _validate_parameter_geometry(
                modules[owner],
                name,
                ALL_PARALLEL_STYLES[style],
                mesh,
                full,
                aliases,
                placed_inputs=placed_inputs,
            )
    validate_tp_replicated_inputs(model, placed_inputs, aliases, metadata_only=True)
    _validate_hf_plan_aliases(
        model, plan, modules, parameters, rules=model.tp_plan if expanded else None
    )
    for name in plan:
        if name in parameters:
            continue
        # Expanded plans include forward hooks after their physical parameter entries.
        if name in modules:
            owned_parameters = [
                f"{name}.{parameter}" if name else parameter
                for parameter, _ in modules[name].named_parameters(recurse=False)
            ]
            if (expanded or owned_parameters) and all(key in plan for key in owned_parameters):
                continue
        raise ValueError(
            f"Native HF TP requires expanded parameter-key plans; unsupported key {name!r}."
        )


def _validate_hf_plan_aliases(model, plan, modules, parameters, *, rules=None):
    """Refuse planned owner aliases that HF's deduplicated placement cannot resolve."""
    from transformers.distributed.parallel_utils import _get_parameter_plan

    canonical = {name for name, _ in model.named_modules()}
    canonical.update(name for name, _ in model.named_parameters())
    for owners, is_weight in ((modules, False), (parameters, True)):
        for name in owners:
            if name not in canonical and (
                name in plan
                or (
                    rules is not None
                    and _get_parameter_plan(name, rules, is_weight=is_weight) is not None
                )
            ):
                raise ValueError(
                    f"Native HF TP does not support placing tied/shared owner alias {name!r}."
                )


def _synchronize_placement_state(context, *state):
    if any(value != state for value in _all_rank_values(context, state)):
        raise ValueError("Bypass placement state must agree on all model ranks.")


def _synchronize_validation(context, error):
    errors = sorted({message for message in _all_rank_values(context, error) if message})
    if errors:
        raise ValueError("; ".join(errors))


def _expert_partitions(model, plan, context, adopted):
    modules = dict(model.named_modules(remove_duplicate=False))
    partitions, global_ids = [], {}
    size, rank = context.expert_mesh.size(), context.expert_mesh.get_local_rank()
    for name, style in plan.items():
        if style != "ep_dispatch_experts":
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


def _validate_parameter_geometry(
    module,
    name,
    style,
    mesh,
    full_name,
    aliases=None,
    *,
    allow_expert_axis=False,
    placed_inputs=None,
):
    """Resolve the installed HF style on a storage-free logical parameter."""
    parameter = module._parameters[name]
    native = parameter.get_qtensor() if isinstance(parameter, QTensorWrapper) else None
    if native is not None:
        from modelopt.torch.quantization.repair_partition import _NATIVE_TP_CODECS

        if type(native) not in _NATIVE_TP_CODECS:
            raise ValueError("This packed codec needs its own native TP partitioner")
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
    if not isinstance(sharded, DTensor):
        return
    bindings = {
        weight: (quantizers, inputs)
        for weight, quantizers, inputs in quantized_weight_bindings(module)
    }
    if name not in bindings:
        return
    view = native_tp_weight_view(
        name, shape, sharded.placements[0], mesh, allow_expert_axis=allow_expert_axis
    )
    if view is None:
        return
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
            if quantizer.backend not in (None, "ggml"):
                raise ValueError(f"Backend {quantizer.backend!r} requires a native TP partitioner.")
            validate_native_weight_partition(quantizer, view)
            if aliases is not None:
                validate_tp_quantizer_aliases(quantizer, view, aliases)
    activation = getattr(module, input_name, None)
    plan_tp_input_quantizer_fields(
        activation, view, {} if aliases is None else aliases, metadata_only=True
    )
    if placed_inputs is not None:
        placed_inputs.add((id(module), input_name))


def _restore_parallel_plans(model, original_plans):
    for name, (owned, plan) in original_plans.items():
        if owned:
            setattr(model, name, plan)
        else:
            model.__dict__.pop(name, None)


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

    saved = getattr(model, "_modelopt_bypass_placement", None)
    adopted = getattr(model, "_mesh_manager", None) is not None
    _synchronize_placement_state(context, saved is not None, adopted)
    if saved is not None:
        error, layouts = None, {}
        try:
            if (
                saved["dense_ranks"] != context.device_mesh.mesh.tolist()
                or saved["expert_ranks"] != context.moe_mesh.mesh.tolist()
            ):
                raise ValueError(
                    "Prepared native Bypass meshes differ from the supplied rank ordering"
                )
            layouts = {
                id(model.get_submodule(name)): layout
                for name, layout in saved["layouts_by_name"].items()
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _synchronize_validation(context, error)
        context.parallel_metadata[id(model)] = saved
        context.tp_quantizer_layouts[id(model)] = layouts
        return saved
    config = copy.copy(getattr(model.config, "distributed_config", None))
    if config is None:
        config = DistributedConfig(
            tp_size=context.tensor_mesh.size(),
            fsdp_size=context.fsdp_mesh.size(),
            dp_replicate_size=_size(context.replica_mesh),
            pp_size=context.pipeline_mesh.size(),
            ep_size=context.expert_mesh.size(),
        )
    if _size(context.pipeline_mesh) > 1:
        raise ValueError("Native PP Bypass is not enabled without a validated family transport")
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
            ("fsdp", context.fsdp_mesh),
            ("pp", context.pipeline_mesh),
            ("ep", context.expert_mesh),
        ):
            if getattr(config, name + "_size") != mesh.size():
                raise ValueError("Native HF config sizes must match the caller's realized meshes.")
        plans = model._resolve_parallel_plans(config)
        tp_plan, ep_plan, fsdp_plan = plans
        from .functional_fsdp import extend_family_fsdp_plan

        plans = tp_plan, ep_plan, extend_family_fsdp_plan(model, fsdp_plan)
        for axis, plan, mesh in (
            ("TP", tp_plan, context.tensor_mesh),
            ("EP", ep_plan, context.expert_mesh),
            ("FSDP", fsdp_plan, context.data_mesh),
        ):
            if mesh.size() > 1 and not plan:
                raise ValueError(
                    f"Native HF resolved {axis} plan is empty for requested size {mesh.size()}."
                )
        if ep_plan and config.ep_implementation not in ("auto", "all_to_all"):
            raise ValueError("Native Bypass requires ep_implementation='auto' or 'all_to_all'.")
        dispatch_overrides = {}
        for name, style in ep_plan.items():
            if style != "ep_dispatch_experts":
                continue
            module = model.get_submodule(name)
            if getattr(getattr(module, "config", None), "_experts_implementation", None) == (
                "deepgemm_megamoe"
            ):
                raise ValueError(
                    "Native Bypass requires differentiable eager or grouped_mm experts."
                )
            strategy = getattr(module, "_modelopt_ep_dispatch", None)
            if strategy is None:
                continue
            if not callable(getattr(strategy, "install_forward", None)):
                raise ValueError("A model-owned EP dispatcher must implement install_forward.")
            supported = getattr(strategy, "supported_implementations", ("auto",))
            if config.ep_implementation not in supported:
                raise ValueError(
                    "This model-owned dispatcher does not support the requested EP implementation."
                )
            if adopted and not getattr(module, "_modelopt_custom_ep_dispatch", False):
                raise ValueError("Adopting this model requires its model-owned EP dispatcher.")
            dispatch_overrides[name] = strategy
        partitions, global_ids = _expert_partitions(model, ep_plan, context, adopted)
        if not adopted:
            modules = dict(model.named_modules(remove_duplicate=False))
            parameters = dict(model.named_parameters(remove_duplicate=False))
            aliases, placed_inputs = {}, set()
            for plan, mesh, allow_expert_axis in (
                (tp_plan, context.tensor_mesh, False),
                (ep_plan, context.expert_mesh, True),
            ):
                for name, style in plan.items():
                    if name in parameters:
                        owner, _, parameter = name.rpartition(".")
                        _validate_parameter_geometry(
                            modules[owner],
                            parameter,
                            ALL_PARALLEL_STYLES[style],
                            mesh,
                            name,
                            aliases,
                            allow_expert_axis=allow_expert_axis,
                            placed_inputs=placed_inputs,
                        )
            validate_tp_replicated_inputs(model, placed_inputs, aliases, metadata_only=True)
            # HF retains merged original rules, while expansion drops secondary module owners.
            tp_rules = model.tp_plan if config.tp_size > 1 else {}
            if config.ep_size > 1:
                is_expert = re.compile(
                    rf"(?:{'|'.join(map(re.escape, model.ep_plan))})(?:\..+)?"
                ).fullmatch
                tp_rules = {name: style for name, style in tp_rules.items() if not is_expert(name)}
            _validate_hf_plan_aliases(model, tp_plan, modules, parameters, rules=tp_rules)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    try:
        _synchronize_validation(context, error)
    except ValueError:
        _restore_parallel_plans(model, original_plans)
        raise
    assert plans is not None and partitions is not None and global_ids is not None
    tp_plan, ep_plan, fsdp_plan = plans
    metadata = {
        "expert_global_ids": global_ids,
        "expert_mode": "dispatch" if ep_plan else None,
        "ep_implementation": config.ep_implementation,
        "plans": {"tp": tp_plan, "ep": ep_plan, "fsdp": fsdp_plan},
    }
    if not adopted:
        model.config.distributed_config = config
        if context.device_mesh.size() > 1:
            metadata["packed"] = prepare_packed_weights_for_tp(model)
        if context.pipeline_mesh.size() > 1:
            apply_pipeline_parallelism(model, context.pipeline_mesh)
        if context.tensor_mesh.size() > 1:
            apply_hf_tensor_parallelism(model, context.tensor_mesh, tp_plan)
    original_partitions = (
        [(module, name, getattr(module, name)) for module, name, _ in partitions] if adopted else []
    )
    for module, name, quantizers in partitions:
        setattr(module, name, quantizers)
    if not adopted and ep_plan:
        native_plan = {
            name: style for name, style in ep_plan.items() if name not in dispatch_overrides
        }
        apply_expert_parallelism(
            model,
            context.expert_mesh,
            context.tensor_mesh,
            native_plan,
            ep_implementation=config.ep_implementation,
        )
        for name, strategy in dispatch_overrides.items():
            module = model.get_submodule(name)
            strategy.install_forward(
                module,
                ep_mesh=context.expert_mesh,
                tp_mesh=context.tensor_mesh,
                ep_implementation=config.ep_implementation,
            )
            module._is_hooked = True
            module._modelopt_custom_ep_dispatch = True
    if initialize is not None:
        error = None
        try:
            initialize(model)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _synchronize_validation(context, error)
    try:
        context.tp_quantizer_layouts[id(model)] = prepare_quantizer_partitions(model, context)
    except ValueError:
        if adopted:
            for module, name, quantizers in original_partitions:
                setattr(module, name, quantizers)
            _restore_parallel_plans(model, original_plans)
        raise
    if adopted:
        model.config.distributed_config = config
    if not adopted and context.data_mesh.size() > 1:
        apply_fully_sharded_data_parallelism(model, context.mesh_manager, fsdp_plan, ep_plan)
    from .functional_fsdp import install_functional_fsdp_calls

    install_functional_fsdp_calls(model)
    names = {id(module): name for name, module in model.named_modules()}
    metadata.update(
        dense_ranks=context.device_mesh.mesh.tolist(),
        expert_ranks=context.moe_mesh.mesh.tolist(),
        layouts_by_name={
            names[key]: layout for key, layout in context.tp_quantizer_layouts[id(model)].items()
        },
    )
    model._modelopt_bypass_placement = metadata
    model._mesh_manager = context.mesh_manager
    model._device_mesh = context.device_mesh
    model._tp_size, model._fsdp_size = config.tp_size, config.fsdp_size
    model._is_distributed_loading_by_transformers = True
    # Component extraction must retain its realized expert routing semantics.
    for name in global_ids:
        model.get_submodule(name)._modelopt_expert_parallel_mode = metadata["expert_mode"]
    context.parallel_metadata[id(model)] = metadata
    return metadata
