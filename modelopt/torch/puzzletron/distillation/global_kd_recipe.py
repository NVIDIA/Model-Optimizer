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

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Small Puzzletron extensions to NeMo AutoModel's native KD recipes.

AutoModel owns model construction, FSDP2/TP/EP/CP/SP/PP, dataloading,
checkpointing, and the train loop. This module only adds independent weighted
main/MTP CE and KD terms, TVD selection, and the VLM PP teacher pass missing
from the upstream VLM KD recipe.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import types
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
from nemo_automodel.components.distributed.config import DistributedSetup
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.components.distributed.pipelining.functional import reset_pp_stage_shapes
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loss.utils import (
    _get_final_hidden_states,
    _get_lm_head_module,
    calculate_loss,
)
from nemo_automodel.components.models.common.mtp import roll_tensor
from nemo_automodel.components.training.rng import ScopedRNG
from nemo_automodel.components.training.utils import ScopedModuleOffloading
from nemo_automodel.components.utils.model_utils import VLM_INPUT_KEYS, filter_forward_kwargs
from nemo_automodel.recipes.llm.kd import (
    KnowledgeDistillationRecipeForNextTokenPrediction as _AutoModelLLMKD,
)
from nemo_automodel.recipes.llm.kd import _build_teacher_model as _build_llm_teacher
from nemo_automodel.recipes.llm.kd import _verify_tokenizer_compatibility as _verify_llm_tokenizers
from nemo_automodel.recipes.llm.train_ft import (
    TrainFinetuneRecipeForNextTokenPrediction,
    _get_num_thd_chunks,
    _uses_te_dot_product_attention,
    _uses_thd_collater,
)
from nemo_automodel.recipes.llm.train_ft import build_model as build_llm_model
from nemo_automodel.recipes.vlm.finetune import (
    FinetuneRecipeForVLM,
    _move_to_device,
    stage_vlm_media_for_pp,
)
from nemo_automodel.recipes.vlm.finetune import build_model as build_vlm_model
from nemo_automodel.recipes.vlm.kd import KnowledgeDistillationRecipeForVLM as _AutoModelVLMKD
from nemo_automodel.recipes.vlm.kd import _build_teacher_model as _build_vlm_teacher
from nemo_automodel.recipes.vlm.kd import _validate_cp_pre_embed_teacher_compatibility
from nemo_automodel.recipes.vlm.kd import _verify_tokenizer_compatibility as _verify_vlm_tokenizers
from torch.distributed.tensor import DTensor, Replicate
from torch.utils.checkpoint import checkpoint

from ..plugins.automodel.batch_adapter import VisionForwardMonitor
from ..plugins.automodel.local_kd_recipe import _copy_hf_auxiliary_assets
from ..plugins.automodel.pp_utils import set_pp_vlm_chunk_specs
from ..security_policy import require_boolean_policy
from .flash_kld import TrainingFlashKLD


def _config_value(config: Any, name: str) -> Any:
    """Read one field from an AutoModel config node or plain mapping."""

    if isinstance(config, dict):
        return config.get(name)
    getter = getattr(config, "get", None)
    if callable(getter):
        value = getter(name, None)
        if value is not None:
            return value
    return getattr(config, name, None)


def _global_kd_checkpoint_adapter_context(model_parts, descriptor_name: str | None = None):
    """Keep heterogeneous AnyModel geometry active during checkpoint conversion.

    AutoModel creates a temporary per-layer adapter context while loading an
    AnyModel checkpoint.  Its state-dict adapters are invoked again when a KD
    checkpoint is saved, after that construction context has exited.  Recover
    the serialized block configs from the active student and re-enter the same
    descriptor-owned context so physical MoE pruning is exported with each
    layer's actual expert count.
    """

    from ..anymodel.automodel import AutoModelDescriptorFactory
    from ..block_config import maybe_cast_block_configs

    for part in model_parts:
        candidates = (
            part,
            getattr(part, "module", None),
            getattr(part, "_fsdp_wrapped_module", None),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            config = getattr(candidate, "config", None)
            if config is None:
                continue
            block_configs = _config_value(config, "block_configs")
            if not block_configs:
                text_config = _config_value(config, "text_config")
                block_configs = _config_value(text_config, "block_configs")
            if not block_configs:
                continue
            active_descriptor = descriptor_name or _config_value(config, "anymodel_descriptor")
            if not active_descriptor:
                continue
            descriptor = AutoModelDescriptorFactory.get(str(active_descriptor))
            if descriptor is not None:
                return descriptor.native_state_dict_adapter_context(
                    maybe_cast_block_configs(block_configs)
                )
    return nullcontext()


def install_pp_checkpoint_state_dict_support() -> None:
    """Allow DCP to save and restore disjoint pipeline-stage state dictionaries.

    PyTorch DCP's strict model-state verification assumes every distributed rank
    describes the same module tree. Pipeline ranks intentionally own disjoint
    parameter FQNs, so strict verification can reduce a valid local state dict
    to an empty mapping. NeMo AutoModel's checkpoint wrappers already describe
    multi-stage PP as non-strict on load; apply the same rule consistently to
    model and optimizer save/load calls for this PP training process.
    """
    from nemo_automodel.components.checkpoint import stateful_wrappers
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    def relaxed(options):
        if options is None:
            return StateDictOptions(strict=False)
        if getattr(options, "strict", False):
            return dataclasses.replace(options, strict=False)
        return options

    for name in (
        "get_model_state_dict",
        "set_model_state_dict",
        "get_optimizer_state_dict",
        "set_optimizer_state_dict",
    ):
        original = getattr(stateful_wrappers, name)
        if getattr(original, "_puzzletron_pp_non_strict", False):
            continue

        def non_strict(*args, _original=original, _name=name, options=None, **kwargs):
            if (
                _name == "get_optimizer_state_dict"
                and os.environ.get("PUZZLETRON_TRACE_GLOBAL_KD") == "1"
                and len(args) >= 2
            ):
                model, optimizer = args[:2]
                model_parameters = {
                    id(parameter): fqn for fqn, parameter in model.named_parameters()
                }
                unmatched = []
                for group_index, group in enumerate(optimizer.param_groups):
                    for parameter_index, parameter in enumerate(group["params"]):
                        if id(parameter) in model_parameters:
                            continue
                        local = (
                            parameter.to_local() if isinstance(parameter, DTensor) else parameter
                        )
                        unmatched.append(
                            {
                                "group": group_index,
                                "index": parameter_index,
                                "shape": tuple(parameter.shape),
                                "local_shape": tuple(local.shape),
                                "requires_grad": bool(parameter.requires_grad),
                                "has_grad": parameter.grad is not None,
                                "state": sorted(
                                    str(key) for key in optimizer.state.get(parameter, {})
                                ),
                            }
                        )
                if unmatched:
                    print(
                        "PUZZLETRON_GLOBAL_KD_CHECKPOINT "
                        f"rank={torch.distributed.get_rank()} unmatched_optimizer_params={unmatched}",
                        flush=True,
                    )
            return _original(*args, options=relaxed(options), **kwargs)

        setattr(non_strict, "_puzzletron_pp_non_strict", True)
        setattr(stateful_wrappers, name, non_strict)


def install_unsharded_checkpoint_state_dict_support() -> None:
    """Fall back to named tensors for a one-rank dynamic model checkpoint.

    PyTorch DCP can reject an otherwise valid unsharded DynamicModule after its
    model-state verification removes every entry. The module's regular state
    dict is empty for the same reason, so collect registered parameters and
    persistent buffers directly. A one-rank checkpoint needs no distributed
    state-dict transformation.
    """
    from nemo_automodel.components.checkpoint import stateful_wrappers

    def is_empty_dcp_state(error: RuntimeError) -> bool:
        return "model state_dict is required to save or load, but model state_dict is empty" in str(
            error
        )

    original_get = stateful_wrappers.get_model_state_dict
    if not getattr(original_get, "_puzzletron_unsharded_fallback", False):

        def get_model_state_dict(model, *args, **kwargs):
            try:
                return original_get(model, *args, **kwargs)
            except RuntimeError as error:
                if not is_empty_dcp_state(error):
                    raise
                state_dict = {
                    name: parameter.detach()
                    for name, parameter in model.named_parameters(remove_duplicate=False)
                }
                for name, buffer in model.named_buffers(remove_duplicate=False):
                    module_name, _, buffer_name = name.rpartition(".")
                    owner = model.get_submodule(module_name)
                    if buffer_name not in owner._non_persistent_buffers_set:
                        state_dict[name] = buffer.detach()
                if not state_dict:
                    raise
                return state_dict

        setattr(get_model_state_dict, "_puzzletron_unsharded_fallback", True)
        stateful_wrappers.get_model_state_dict = get_model_state_dict

    original_set = stateful_wrappers.set_model_state_dict
    if not getattr(original_set, "_puzzletron_unsharded_fallback", False):

        def set_model_state_dict(model, state_dict, *args, options=None, **kwargs):
            try:
                return original_set(model, state_dict, *args, options=options, **kwargs)
            except RuntimeError as error:
                if not is_empty_dcp_state(error):
                    raise
                return model.load_state_dict(
                    state_dict,
                    strict=True if options is None else bool(options.strict),
                )

        setattr(set_model_state_dict, "_puzzletron_unsharded_fallback", True)
        stateful_wrappers.set_model_state_dict = set_model_state_dict


def _trace_global_kd_phase(phase: str) -> None:
    if os.environ.get("PUZZLETRON_TRACE_GLOBAL_KD") != "1":
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print(f"PUZZLETRON_GLOBAL_KD rank={rank} phase={phase}", flush=True)


def _attach_global_kd_gdn_traces(parts, *, prefix: str, trace_backward: bool = False) -> None:
    if os.environ.get("PUZZLETRON_TRACE_GLOBAL_KD") != "1":
        return
    for part in parts:
        for module in part.modules():
            if "GatedDeltaNet" not in type(module).__name__:
                continue
            layer_idx = getattr(module, "layer_idx", "unknown")
            module.register_forward_pre_hook(
                lambda _module, _args, layer_idx=layer_idx: _trace_global_kd_phase(
                    f"{prefix}_gdn_{layer_idx}_forward_begin"
                )
            )

            def _forward_end(_module, _args, output, *, layer_idx=layer_idx):
                _trace_global_kd_phase(f"{prefix}_gdn_{layer_idx}_forward_end")
                if trace_backward and isinstance(output, torch.Tensor) and output.requires_grad:

                    def trace_backward_begin(grad, *, layer_idx=layer_idx):
                        _trace_global_kd_phase(f"{prefix}_gdn_{layer_idx}_backward_begin")
                        return grad

                    output.register_hook(trace_backward_begin)

            module.register_forward_hook(_forward_end)
            if trace_backward:
                parameter = next(module.parameters(), None)
                if parameter is not None and parameter.requires_grad:

                    def trace_parameter_grad(grad, *, layer_idx=layer_idx):
                        _trace_global_kd_phase(f"{prefix}_gdn_{layer_idx}_parameter_grad")
                        return grad

                    parameter.register_hook(trace_parameter_grad)


def _instantiate(node):
    if node is None:
        return None
    if hasattr(node, "instantiate"):
        return node.instantiate()
    if hasattr(node, "build"):
        return node.build()
    return node


def _find_dtensor_weight_owner(module):
    """Find the actual TP-linear owner beneath LoRA/module wrappers."""
    owners = [module]
    seen = set()
    while owners:
        owner = owners.pop(0)
        if owner is None or id(owner) in seen:
            continue
        seen.add(id(owner))
        candidate = getattr(owner, "weight", None)
        if isinstance(candidate, DTensor):
            return owner, candidate
        for attribute in ("base_layer", "original_module", "module"):
            child = getattr(owner, attribute, None)
            if child is not None:
                owners.append(child)
        get_base_layer = getattr(owner, "get_base_layer", None)
        if callable(get_base_layer):
            owners.append(get_base_layer())
    return None, None


def _align_dtensor_to_module_mesh(value, module):
    """Rewrap an equivalent-mesh DTensor on the module weight's mesh identity.

    Student and teacher models build separate ``DeviceMesh`` objects over the
    same TP ranks.  Pipeline capture can preserve the first object's identity
    while a later LM-head projection owns the second.  PyTorch rejects that as
    cross-mesh communication even though no communication or redistribution is
    required.  Rewrapping the unchanged local shard is safe only when rank
    layout and mesh dimension names are exactly identical.
    """
    _, weight = _find_dtensor_weight_owner(module)
    if not isinstance(value, DTensor) or not isinstance(weight, DTensor):
        return value
    source_mesh = value.device_mesh
    target_mesh = weight.device_mesh
    if source_mesh is target_mesh:
        return value
    source_names = tuple(getattr(source_mesh, "mesh_dim_names", ()) or ())
    target_names = tuple(getattr(target_mesh, "mesh_dim_names", ()) or ())
    source_ranks = getattr(source_mesh, "mesh", None)
    target_ranks = getattr(target_mesh, "mesh", None)
    equivalent = (
        source_names == target_names
        and source_ranks is not None
        and target_ranks is not None
        and torch.equal(source_ranks.detach().cpu(), target_ranks.detach().cpu())
    )
    if not equivalent:
        raise RuntimeError(
            "Cannot project a DTensor through an LM head on a different device mesh: "
            f"source_names={source_names}, target_names={target_names}"
        )
    return DTensor.from_local(
        value.to_local(),
        device_mesh=target_mesh,
        placements=value.placements,
        run_check=False,
    )


def _project_teacher_hidden_on_reference_mesh(hidden, teacher_head, reference_logits):
    """Project frozen teacher hidden shards without cross-mesh DTensor dispatch."""
    owner, weight = _find_dtensor_weight_owner(teacher_head)
    if not isinstance(weight, DTensor):
        projection = getattr(teacher_head, "_puzzletron_projection_forward", None)
        return projection(hidden) if projection is not None else teacher_head(hidden)
    local_hidden = hidden.to_local() if isinstance(hidden, DTensor) else hidden
    reference_local = (
        reference_logits.to_local() if isinstance(reference_logits, DTensor) else reference_logits
    )
    expected_vocab = int(reference_local.shape[-1])

    def _local_tp_parameter(value):
        if not isinstance(value, DTensor):
            return value
        local = value.to_local()
        if int(local.shape[0]) == expected_vocab:
            return local
        mesh_names = tuple(getattr(value.device_mesh, "mesh_dim_names", ()) or ())
        if len(mesh_names) != len(value.placements):
            raise RuntimeError(
                "Cannot isolate the TP vocabulary shard from an unnamed teacher "
                f"parameter mesh: names={mesh_names}, placements={value.placements}"
            )
        placements = tuple(
            placement if name == "tp" else Replicate()
            for name, placement in zip(mesh_names, value.placements)
        )
        local = value.redistribute(
            device_mesh=value.device_mesh,
            placements=placements,
        ).to_local()
        if int(local.shape[0]) != expected_vocab:
            raise RuntimeError(
                "Teacher LM-head shard does not match the student vocabulary shard "
                f"after non-TP unsharding: expected={expected_vocab}, got={local.shape[0]}, "
                f"mesh_names={mesh_names}, placements={value.placements}"
            )
        return local

    cache_key = (
        id(weight),
        expected_vocab,
        str(local_hidden.device),
        local_hidden.dtype,
    )
    cache = owner.__dict__.setdefault("_puzzletron_frozen_tp_projection_cache", {})
    cached = cache.get(cache_key)
    if cached is None:
        _trace_global_kd_phase("teacher_mtp_head_unshard_begin")
        local_weight = _local_tp_parameter(weight).detach()
        bias = getattr(owner, "bias", None)
        bias = _local_tp_parameter(bias).detach() if bias is not None else None
        cache[cache_key] = (local_weight, bias)
        _trace_global_kd_phase("teacher_mtp_head_unshard_end")
    else:
        local_weight, bias = cached
    local_logits = torch.nn.functional.linear(local_hidden, local_weight, bias)
    if not isinstance(reference_logits, DTensor):
        return local_logits
    return DTensor.from_local(
        local_logits,
        device_mesh=reference_logits.device_mesh,
        placements=reference_logits.placements,
        run_check=False,
    )


def _detach_output(output):
    if isinstance(output, torch.Tensor):
        return output.detach().clone()
    if isinstance(output, tuple):
        return tuple(_detach_output(item) for item in output)
    if isinstance(output, list):
        return [_detach_output(item) for item in output]
    return output


def _distillation_hidden_forward(_head, hidden_states, *args, **kwargs):
    """Run the native final-head branch while returning pre-projection hidden states."""
    del args, kwargs
    return hidden_states


def _install_distillation_head_passthrough(parts) -> int:
    """Make model outputs hidden-state-sized without removing LM-head parameters."""
    installed = 0
    for part in parts:
        head = getattr(part, "lm_head", None)
        if head is None or hasattr(head, "_puzzletron_projection_forward"):
            continue
        head._puzzletron_projection_forward = head.forward
        head.forward = types.MethodType(_distillation_hidden_forward, head)
        part._puzzletron_distillation_hidden_output = True
        installed += 1
    return installed


def _distillation_lm_head(model):
    """Return the real projection retained by a distillation passthrough."""
    head = getattr(model, "lm_head", None)
    return head if hasattr(head, "_puzzletron_projection_forward") else _get_lm_head_module(model)


def _reset_global_kd_pp_stage_shapes(pp, *, seq_len: int) -> None:
    """Refresh PP metadata with its activation dtype, not parameter storage dtype."""
    reset_pp_stage_shapes(
        pp.info.schedule,
        pp.info.stages,
        pp.parts[0].config,
        pp.pp_microbatch_size,
        seq_len,
        tensor_dtype=pp.dtype,
    )


def _refresh_pp_hidden_output_meta(pp) -> int:
    """Change precomputed last-stage PP output metadata from logits to hidden states."""
    stages = getattr(getattr(pp, "info", None), "stages", None) or []
    refreshed = 0
    for stage in stages:
        part = getattr(stage, "submod", None)
        if (
            part is None
            or not getattr(stage, "is_last", False)
            or not getattr(part, "_puzzletron_distillation_hidden_output", False)
        ):
            continue
        inputs_meta = getattr(stage, "inputs_meta", None)
        if not inputs_meta or not torch.is_tensor(inputs_meta[0]):
            continue
        input_meta = inputs_meta[0]
        if input_meta.ndim < 3:
            continue
        dtype = getattr(pp, "dtype", None)
        if dtype is None:
            try:
                dtype = next(part.parameters()).dtype
            except StopIteration:
                dtype = input_meta.dtype
        existing_outputs = tuple(getattr(stage, "_outputs_meta", ()) or ())
        hidden_meta = torch.empty(
            *input_meta.shape[:-1],
            int(input_meta.shape[-1]),
            device="meta",
            dtype=dtype,
        )
        stage._outputs_meta = (hidden_meta, *existing_outputs[1:])
        refreshed += 1
    return refreshed


def _first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return getattr(output, "logits", None)


def _pipeline_config_with_capture(base, overrides, capture):
    values = {field.name: getattr(base, field.name) for field in dataclasses.fields(PipelineConfig)}
    if overrides is not None:
        override_values = overrides.to_dict() if hasattr(overrides, "to_dict") else dict(overrides)
        for key, value in override_values.items():
            if key in values:
                values[key] = value

    def capture_loss(output, _target, **_kwargs):
        capture.append(_detach_output(output))
        tensor = _first_tensor(output)
        if tensor is None:
            raise TypeError("Teacher PP output did not contain logits")
        return tensor.new_zeros(())

    values["loss_fn"] = capture_loss
    return PipelineConfig(**values)


def _build_pp_teacher(recipe, *, domain: str):
    # Upstream LLM KD inspects capture[0] on every PP rank even though only the
    # last stage invokes the capture loss.  Keep a None sentinel on non-output
    # ranks; the weighted loss wrapper filters it and queues real last-stage
    # outputs in microbatch order.
    capture: list[Any] = [None]
    pipeline_config = _pipeline_config_with_capture(
        recipe.pipeline_config,
        recipe.cfg.get("teacher_pipeline", None),
        capture,
    )
    distributed_setup = DistributedSetup(
        mesh_context=recipe.distributed_setup.mesh_context,
        strategy_config=recipe.distributed_setup.strategy_config,
        pipeline_config=pipeline_config,
        moe_parallel_config=recipe.distributed_setup.moe_parallel_config,
        activation_checkpointing=recipe.activation_checkpointing,
    )
    if domain == "llm":
        teacher = build_llm_model(
            recipe.cfg.teacher_model,
            cfg_peft=None,
            has_packed_sequence=recipe.cfg.get("packed_sequence.packed_sequence_size", 0) > 0,
            seed=recipe.cfg.get("seed", 42),
            cfg_fp8=None,
            cfg_compile=None,
            cfg_quantization=None,
            distributed_setup=distributed_setup,
            cfg_qat=None,
        )
    else:
        teacher = build_vlm_model(
            recipe.cfg.teacher_model,
            recipe.cfg.get("teacher_freeze_config", None),
            None,
            seed=recipe.cfg.get("seed", 42),
            distributed_setup=distributed_setup,
        )
    teacher_parts = getattr(teacher, "parts", [teacher])
    _install_distillation_head_passthrough(teacher_parts)
    for part in teacher_parts:
        part.eval()
        for parameter in part.parameters():
            parameter.requires_grad_(False)
    _attach_global_kd_gdn_traces(teacher_parts, prefix="teacher")
    teacher._teacher_logits_capture = capture
    return teacher


def _set_teacher_mtp_enabled(teacher):
    """Enable MTP output without recursively enabling dropout in the backbone."""
    for part in getattr(teacher, "parts", [teacher]):
        part.training = True
        for module in part.modules():
            mtp = getattr(module, "mtp", None)
            if mtp is not None:
                module.training = True
                mtp.training = True


def _split_output(output, model):
    seq_idx = None
    main_is_hidden = bool(getattr(model, "_puzzletron_distillation_hidden_output", False))
    if isinstance(output, tuple):
        values = list(output)
        if values and isinstance(values[-1], torch.Tensor) and values[-1].dtype == torch.int32:
            seq_idx = values.pop()
        logits = values[0]
        mtp_values = values[1:]
        mtp_are_logits = bool(getattr(model, "mtp_outputs_are_logits", False))
        return (
            logits,
            main_is_hidden,
            None if mtp_are_logits else mtp_values,
            mtp_values if mtp_are_logits else None,
            seq_idx,
        )
    return (
        getattr(output, "logits", output),
        main_is_hidden,
        getattr(output, "mtp_per_depth_h", None),
        getattr(output, "mtp_per_depth_logits", None),
        getattr(output, "seq_idx", None),
    )


class _WeightedObjectiveMixin:
    """Objective behavior shared by the LLM and VLM AutoModel recipe bases."""

    cfg: Any
    checkpointer: Any
    device_mesh: Any
    dist_env: Any
    loss_fn: Any
    metric_logger_train: Any
    model_parts: Any
    optimizer: Any
    pp: Any
    pp_enabled: bool
    teacher_model: Any
    _ce_loss_buffer: list[torch.Tensor]
    _kd_loss_buffer: list[torch.Tensor]
    _dp_allreduce: Callable[..., torch.Tensor]

    def _configure_objective(self):
        objective = self.cfg.get("objective", {})
        self.objective = {
            name: float(objective.get(name, default))
            for name, default in (
                ("main_ce", 1.0),
                ("mtp_ce", 0.0),
                ("main_kd", 1.0),
                ("mtp_kd", 0.0),
            )
        }
        self.needs_teacher = self.objective["main_kd"] > 0 or self.objective["mtp_kd"] > 0
        self.main_kd_loss_fn = _instantiate(self.cfg.get("main_kd_loss_fn", None))
        self.mtp_kd_loss_fn = _instantiate(self.cfg.get("mtp_kd_loss_fn", None))
        self._objective_buffers = {name: [] for name in self.objective}
        self._objective_step_cursor = {name: 0 for name in self.objective}
        self._loss_topology_logged = False
        self._gradient_squared = {
            name: torch.tensor(0.0) for name in ("vision", "projector", "language", "mtp")
        }
        self._gradient_hook_handles = []
        self._vision_monitors = []
        self._media_input_checksums = []

    def _configure_incremental_metric_logging(self):
        """Make canonical JSONL records visible after every optimizer step."""

        logger = self.metric_logger_train
        logger.buffer_size = 1
        logger.flush = True

    def save_checkpoint(
        self,
        epoch,
        step,
        train_loss,
        val_loss,
        best_metric_key="default",
    ):
        """Publish a completion marker only after model and optimizer DCP succeed."""

        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
            install_unsharded_checkpoint_state_dict_support()
        result = None
        publication_error: Exception | None = None
        publication_error_text: str | None = None
        try:
            result = super().save_checkpoint(  # type: ignore[misc]
                epoch,
                step,
                train_loss,
                val_loss,
                best_metric_key=best_metric_key,
            )
        except Exception as error:  # noqa: BLE001 - all ranks must reach the collective
            publication_error = error
            publication_error_text = f"{type(error).__name__}: {error}"
        distributed = torch.distributed.is_initialized()
        if distributed:
            parent_save_errors: list[str | None] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(parent_save_errors, publication_error_text)
            parent_save_failure = next(
                (
                    (rank, error)
                    for rank, error in enumerate(parent_save_errors)
                    if error is not None
                ),
                None,
            )
            if parent_save_failure is not None:
                failing_rank, error_text = parent_save_failure
                publication_error_text = f"parent save failed on rank {failing_rank}: {error_text}"
        checkpoint_path = os.path.join(
            str(self.checkpointer.config.checkpoint_dir),
            f"epoch_{epoch}_step_{step}",
        )
        if publication_error_text is None and self.dist_env.is_main:
            try:
                consolidated = Path(checkpoint_path, "model", "consolidated")
                config_path = consolidated / "config.json"
                config = json.loads(config_path.read_text()) if config_path.is_file() else {}
                text_config = config.get("text_config")
                if config.get("block_configs") or (
                    isinstance(text_config, dict) and text_config.get("block_configs")
                ):
                    from ..utils.vllm_adapter import refresh_realized_checkpoint_config

                    model_config = _config_value(getattr(self, "cfg", None), "model")
                    configured_trust = _config_value(model_config, "trust_remote_code")
                    refresh_realized_checkpoint_config(
                        consolidated,
                        trust_remote_code=require_boolean_policy(
                            configured_trust,
                            path="model.trust_remote_code",
                            default=False,
                        ),
                    )
                model_config = _config_value(getattr(self, "cfg", None), "model")
                source_dir = _config_value(model_config, "pretrained_model_name_or_path")
                if source_dir:
                    _copy_hf_auxiliary_assets(Path(source_dir), consolidated)
                Path(checkpoint_path, "saving_completed").touch()
            except Exception as error:  # noqa: BLE001 - all ranks must reach the collective
                publication_error = error
                publication_error_text = (
                    f"publication failed on rank 0: {type(error).__name__}: {error}"
                )
        if distributed:
            publication_status = [publication_error_text]
            torch.distributed.broadcast_object_list(publication_status, src=0)
            publication_error_text = publication_status[0]
        if publication_error is not None:
            raise publication_error
        if publication_error_text is not None:
            raise RuntimeError(f"global KD checkpoint {publication_error_text}")
        return result

    def _install_vision_observers(self, parts, *, role: str):
        for part in parts:
            candidates = [
                (name.count("."), name, module)
                for name, module in part.named_modules()
                if name.rsplit(".", 1)[-1] in {"visual", "vision_tower", "vision_model"}
            ]
            if not candidates:
                continue
            _, name, module = min(candidates, key=lambda item: (item[0], item[1]))
            monitor = VisionForwardMonitor(module)
            monitor.__enter__()
            self._vision_monitors.append((role, name, monitor))

    def _record_media_batch(self, batch):
        digest = hashlib.sha256()
        found = False
        for key in ("input_ids", "image_grid_thw", "mm_token_type_ids"):
            value = batch.get(key) if isinstance(batch, dict) else None
            if not isinstance(value, torch.Tensor):
                continue
            tensor = value.detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
            found = True
        if found:
            self._media_input_checksums.append(digest.hexdigest())

    def observability_metadata(self):
        local = {
            "vision_forward_count": sum(
                monitor.forward_count for _, _, monitor in self._vision_monitors
            ),
            "vision_by_role": {
                role: sum(
                    monitor.forward_count
                    for monitor_role, _, monitor in self._vision_monitors
                    if monitor_role == role
                )
                for role in {role for role, _, _ in self._vision_monitors}
            },
            "vision_output_checksums": [
                checksum
                for _, _, monitor in self._vision_monitors
                for checksum in monitor.output_checksums
            ],
            "media_input_checksums": list(dict.fromkeys(self._media_input_checksums)),
        }
        if not torch.distributed.is_initialized():
            return local
        gathered: list[dict[str, Any] | None] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local)
        observations = [item for item in gathered if item is not None]
        if len(observations) != len(gathered):
            raise RuntimeError("Missing global KD observability metadata from a distributed rank")
        roles = set().union(*(item["vision_by_role"] for item in observations))
        return {
            "vision_forward_count": sum(item["vision_forward_count"] for item in observations),
            "vision_by_role": {
                role: sum(item["vision_by_role"].get(role, 0) for item in observations)
                for role in sorted(roles)
            },
            "vision_output_checksums": sorted(
                checksum for item in observations for checksum in item["vision_output_checksums"]
            ),
            "media_input_checksums": sorted(
                set(checksum for item in observations for checksum in item["media_input_checksums"])
            ),
        }

    def close_observability(self):
        for _, _, monitor in reversed(self._vision_monitors):
            monitor.__exit__(None, None, None)
        self._vision_monitors.clear()

    @staticmethod
    def _gradient_group(parameter_name: str) -> str:
        components = set(parameter_name.lower().split("."))
        if "mtp" in components or parameter_name.startswith("mtp."):
            return "mtp"
        projector_components = {
            "merger",
            "mm_projector",
            "multimodal_projector",
            "projector",
            "vision_projector",
        }
        if components & projector_components:
            return "projector"
        vision_components = {
            "image_encoder",
            "visual",
            "vision",
            "vision_encoder",
            "vision_model",
            "vision_tower",
        }
        if components & vision_components:
            return "vision"
        return "language"

    def _remove_text_inactive_optimizer_parameters(self) -> None:
        """Exclude modality branches that a text-only forward cannot exercise.

        Adam creates state lazily on the first gradient.  Keeping vision or
        multimodal-projector parameters in a text-only optimizer therefore
        produces a checkpoint whose parameter groups mention state entries
        that can never exist, and DCP rejects that checkpoint on load.
        """

        inactive_ids = set()
        for part in self.model_parts:
            for name, parameter in part.named_parameters():
                if self._gradient_group(name) not in {"vision", "projector"}:
                    continue
                parameter.requires_grad_(False)
                inactive_ids.add(id(parameter))

        if not inactive_ids:
            return
        optimizers = (
            self.optimizer if isinstance(self.optimizer, (list, tuple)) else [self.optimizer]
        )
        removed = 0
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                retained = []
                for parameter in group["params"]:
                    if id(parameter) in inactive_ids:
                        optimizer.state.pop(parameter, None)
                        removed += 1
                    else:
                        retained.append(parameter)
                group["params"] = retained
        print(
            "PUZZLETRON_GLOBAL_KD_CHECKPOINT "
            f"rank={torch.distributed.get_rank() if torch.distributed.is_initialized() else 0} "
            f"removed_text_inactive_optimizer_params={removed}",
            flush=True,
        )

    def load_checkpoint(self, restore_from=None):
        # Base recipe setup creates the optimizer immediately before invoking
        # this virtual method.  This is the last common point at which both a
        # fresh run and a resumed run can establish the same parameter groups.
        if not bool(self.cfg.get("puzzletron_resume", True)):
            return None
        if getattr(self, "_puzzletron_global_kd_domain", None) == "llm":
            self._remove_text_inactive_optimizer_parameters()
        return super().load_checkpoint(restore_from or "LATEST")  # type: ignore[misc]

    def _install_gradient_norm_observers(self):
        self._gradient_squared = {
            name: torch.tensor(0.0, device=self.dist_env.device)
            for name in ("vision", "projector", "language", "mtp")
        }
        parameter_groups = {}
        for part in self.model_parts:
            for parameter_name, parameter in part.named_parameters():
                if not parameter.requires_grad:
                    continue
                parameter_groups[id(parameter)] = self._gradient_group(parameter_name)

        def observe_optimizer_step(optimizer, _args, _kwargs):
            seen = set()
            for parameter_group in optimizer.param_groups:
                for parameter in parameter_group["params"]:
                    if id(parameter) in seen or parameter.grad is None:
                        continue
                    seen.add(id(parameter))
                    group = parameter_groups.get(id(parameter))
                    if group is None:
                        continue
                    gradient = parameter.grad
                    value = gradient.to_local() if isinstance(gradient, DTensor) else gradient
                    self._gradient_squared[group].add_(value.detach().float().square().sum())

        optimizers = (
            self.optimizer if isinstance(self.optimizer, (list, tuple)) else [self.optimizer]
        )
        for optimizer in optimizers:
            self._gradient_hook_handles.append(
                optimizer.register_step_pre_hook(observe_optimizer_step)
            )

    def _rebind_optimizer_to_current_model_parameters(self, model=None) -> None:
        """Repair PP parameter identities invalidated while building the teacher.

        Some custom-model PP builders replace the last stage's student modules
        while constructing a second pipeline model.  The optimizer then retains
        the pre-replacement Parameters: they receive no gradients and PyTorch DCP
        cannot map their integer IDs back to student FQNs.  Rebind only when the
        complete ordered shape signature proves that the replacement is
        one-for-one, preserving any restored Adam state along the way.
        """

        model_parts = self.model_parts if model is None else model
        if not isinstance(model_parts, (list, tuple)):
            model_parts = [model_parts]

        current_parameters = []
        seen = set()
        for part in model_parts:
            for parameter in part.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                current_parameters.append(parameter)

        optimizers = (
            self.optimizer if isinstance(self.optimizer, (list, tuple)) else [self.optimizer]
        )
        for optimizer in optimizers:
            optimizer_parameters = [
                parameter for group in optimizer.param_groups for parameter in group["params"]
            ]
            current_ids = {id(parameter) for parameter in current_parameters}
            if all(id(parameter) in current_ids for parameter in optimizer_parameters):
                continue
            if len(optimizer_parameters) != len(current_parameters):
                raise RuntimeError(
                    "cannot safely rebind stale PP optimizer parameters: "
                    f"optimizer={len(optimizer_parameters)} model={len(current_parameters)}"
                )
            mismatched_shapes = [
                (index, tuple(old.shape), tuple(new.shape))
                for index, (old, new) in enumerate(zip(optimizer_parameters, current_parameters))
                if tuple(old.shape) != tuple(new.shape)
            ]
            if mismatched_shapes:
                raise RuntimeError(
                    "cannot safely rebind stale PP optimizer parameters because ordered shapes differ: "
                    f"{mismatched_shapes[:8]}"
                )

            offset = 0
            rebound = 0
            for group in optimizer.param_groups:
                replacements = current_parameters[offset : offset + len(group["params"])]
                for old, new in zip(group["params"], replacements):
                    if old is new:
                        continue
                    if old in optimizer.state:
                        optimizer.state[new] = optimizer.state.pop(old)
                    rebound += 1
                group["params"] = replacements
                offset += len(replacements)
            print(
                "PUZZLETRON_GLOBAL_KD_CHECKPOINT "
                f"rank={torch.distributed.get_rank() if torch.distributed.is_initialized() else 0} "
                f"rebound_optimizer_params={rebound}",
                flush=True,
            )

    def _install_pre_optimizer_save_rebind(self) -> None:
        """Recheck identities after DCP model extraction and before Adam save."""

        if getattr(self.checkpointer, "_puzzletron_optimizer_rebind_installed", False):
            return
        save_model = self.checkpointer.save_model
        save_optimizer = self.checkpointer.save_optimizer

        def save_model_with_current_parts(*args, **kwargs):
            descriptor_name = _config_value(
                _config_value(getattr(self, "cfg", None), "model"), "anymodel_descriptor"
            )
            if args:
                args = (self.model_parts, *args[1:])
            else:
                kwargs["model"] = self.model_parts
            with _global_kd_checkpoint_adapter_context(
                self.model_parts, descriptor_name=descriptor_name
            ):
                return save_model(*args, **kwargs)

        def save_optimizer_with_current_parameters(optimizer, model, path, scheduler):
            del model
            optimizer_model = self.model_parts
            self._rebind_optimizer_to_current_model_parameters(optimizer_model)
            return save_optimizer(optimizer, optimizer_model, path, scheduler)

        self.checkpointer.save_model = save_model_with_current_parts
        self.checkpointer.save_optimizer = save_optimizer_with_current_parameters
        self.checkpointer._puzzletron_optimizer_rebind_installed = True

    def _consume_gradient_norms(self) -> dict[str, float]:
        metrics = {}
        for group, squared in self._gradient_squared.items():
            value = squared.clone()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(value)
            metrics[f"gradient_norm_{group}"] = float(value.clamp_min(0).sqrt().cpu())
            squared.zero_()
        return metrics

    def _pp_objective_source_rank(self) -> int:
        """Return the last PP stage in the first replica of every other mesh axis."""
        if self.device_mesh is not None and "pp" in self.device_mesh.mesh_dim_names:
            dim_names = list(self.device_mesh.mesh_dim_names)
            index = tuple(-1 if name == "pp" else 0 for name in dim_names)
            return int(self.device_mesh.mesh[index].item())
        return int(self.device_mesh.mesh.reshape(-1)[-1].item())

    def _publish_objective_metrics(self, log_data) -> dict[str, float]:
        """Publish all weighted-objective terms on rank zero for LLM and VLM recipes.

        Under PP only the last stage owns loss terms.  First reduce that stage's
        replicas over DP+CP, normalize its microbatch sums by the optimizer-step
        label count, and forward the result to rank zero.  Non-PP objectives are
        already normalized by their loss functions and only need the replica
        reduction.
        """
        source_rank = self._pp_objective_source_rank() if self.pp_enabled else None
        num_label_tokens = log_data.metrics["num_label_tokens"]
        objective_metrics = {}
        for name, values in self._objective_buffers.items():
            start = self._objective_step_cursor.get(name, 0)
            current = values[start:]
            value = (
                torch.stack(current).sum()
                if current
                else torch.tensor(0.0, device=self.dist_env.device)
            )
            value = self._dp_allreduce(value, include_cp=True)
            if self.pp_enabled:
                value = value / num_label_tokens if num_label_tokens > 0 else value * 0.0
                value = value.float().to(self.dist_env.device)
                if source_rank != 0:
                    if self.dist_env.rank == source_rank:
                        torch.distributed.send(value, dst=0)
                    elif self.dist_env.is_main:
                        torch.distributed.recv(value, src=source_rank)
            objective_metrics[name] = value.item()
            values.clear()
            self._objective_step_cursor[name] = 0
        log_data.metrics.update(objective_metrics)
        return objective_metrics

    def _post_teacher_setup(self):
        if self.objective["mtp_kd"] > 0:
            _set_teacher_mtp_enabled(self.teacher_model)

    def _ensure_student_mtp_outputs(self):
        if self.objective["mtp_ce"] > 0 or self.objective["mtp_kd"] > 0:
            _set_teacher_mtp_enabled(self.pp if self.pp_enabled else self.model_parts[0])

    def _teacher_loss_model(self):
        teacher_pp = getattr(self, "teacher_pp", None)
        if teacher_pp is None:
            return self.teacher_model
        return next(
            part for part, stage in zip(teacher_pp.parts, teacher_pp.info.stages) if stage.is_last
        )

    @staticmethod
    def _local_zero(tensor):
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
        return tensor.new_zeros(())

    @staticmethod
    def _flatten_tokens(tensor):
        return tensor.reshape(-1, tensor.shape[-1])

    def _ce_without_inner_checkpoint(self, logits, labels, model, num_label_tokens):
        forward = getattr(self.loss_fn, "forward_no_checkpoint", None)
        if forward is not None:
            return forward(logits, labels, num_label_tokens=num_label_tokens)
        return calculate_loss(
            self.loss_fn,
            logits=logits,
            labels=labels,
            model=model,
            num_label_tokens=num_label_tokens,
        )

    def _kd_without_inner_checkpoint(
        self, student_logits, teacher_logits, labels, num_label_tokens
    ):
        forward = getattr(self.mtp_kd_loss_fn, "forward_no_checkpoint", None)
        if forward is None:
            forward = self.mtp_kd_loss_fn
        return forward(
            student_logits,
            teacher_logits,
            labels,
            num_batch_labels=num_label_tokens,
        )

    def _flash_kld_engine(self, domain: str) -> TrainingFlashKLD:
        attribute = f"{domain}_flash_kld"
        engine = getattr(self, attribute, None)
        if engine is not None:
            return engine
        kd_loss = self.main_kd_loss_fn if domain == "main" else self.mtp_kd_loss_fn
        chunk_sizes = [
            int(getattr(loss, "chunk_size", 0))
            for loss in (getattr(self, "loss_fn", None), kd_loss)
            if loss is not None and int(getattr(loss, "chunk_size", 0)) > 0
        ]
        engine = TrainingFlashKLD(
            token_chunk_size=min(chunk_sizes) if chunk_sizes else 128,
            temperature=float(getattr(kd_loss, "temperature", 1.0)),
            checkpoint_chunks=True,
        )
        setattr(self, attribute, engine)
        return engine

    def _hidden_objective_losses(
        self,
        *,
        domain: str,
        student_hidden,
        teacher_hidden,
        labels,
        student_model,
        teacher_model,
        num_label_tokens,
    ):
        ce_name = f"{domain}_ce"
        kd_name = f"{domain}_kd"
        compute_ce = self.objective[ce_name] > 0
        compute_kd = self.objective[kd_name] > 0
        student_head = _distillation_lm_head(student_model)
        if student_head is None:
            raise ValueError(f"{domain} FlashKLD requires an accessible student lm_head")
        student_hidden = _align_dtensor_to_module_mesh(student_hidden, student_head)
        teacher_head = None
        if compute_kd:
            if teacher_model is None or teacher_hidden is None:
                raise ValueError(f"{domain} FlashKLD requires teacher hidden states")
            teacher_head = _distillation_lm_head(teacher_model)
            if teacher_head is None:
                raise ValueError(f"{domain} FlashKLD requires an accessible teacher lm_head")

        def teacher_project(hidden, reference_logits):
            return _project_teacher_hidden_on_reference_mesh(
                hidden,
                teacher_head,
                reference_logits,
            )

        return self._flash_kld_engine(domain)(
            student_hidden,
            student_head,
            labels,
            teacher_hidden=teacher_hidden if compute_kd else None,
            teacher_project=teacher_project if compute_kd else None,
            compute_ce=compute_ce,
            compute_kd=compute_kd,
            num_label_tokens=num_label_tokens,
        )

    def _mtp_objective_losses(
        self,
        *,
        student_h,
        student_logits,
        teacher_h,
        teacher_logits,
        labels,
        student_model,
        teacher_model,
        num_label_tokens,
        seq_idx=None,
    ):
        """Compute MTP CE and KD without retaining full token-by-vocab logits.

        Each checkpointed token chunk owns the hidden-to-LM-head projection and
        both distribution losses. Backward therefore saves only hidden-state
        chunks and recomputes the vocabulary projection/softmax, while TP-sharded
        LM heads and logits remain sharded end to end.
        """
        student_values = list(student_h if student_h is not None else student_logits or [])
        student_is_hidden = student_h is not None
        if not student_values:
            raise ValueError("MTP has non-zero weight but the student emitted no predictions")

        needs_mtp_kd = self.objective["mtp_kd"] > 0
        teacher_values = list(teacher_h if teacher_h is not None else teacher_logits or [])
        teacher_is_hidden = teacher_h is not None
        if needs_mtp_kd and len(student_values) != len(teacher_values):
            raise ValueError(
                "MTP KD requires identical student/teacher prediction depths; "
                f"got {len(student_values)} and {len(teacher_values)}"
            )

        if student_is_hidden:
            ce_total = self._local_zero(student_values[0])
            kd_total = self._local_zero(student_values[0])
            cur_labels = labels
            for depth, student_value in enumerate(student_values):
                cur_labels = roll_tensor(cur_labels, shifts=-1, dim=-1)
                depth_labels = cur_labels.clone()
                depth_labels[..., -(depth + 1) :] = -100
                if seq_idx is not None:
                    rolled = roll_tensor(seq_idx, shifts=-(depth + 1), dim=-1)
                    depth_labels = torch.where(rolled == seq_idx, depth_labels, -100)
                teacher_value = teacher_values[depth] if needs_mtp_kd else None
                ce_depth, kd_depth = self._hidden_objective_losses(
                    domain="mtp",
                    student_hidden=student_value,
                    teacher_hidden=teacher_value,
                    labels=depth_labels,
                    student_model=student_model,
                    teacher_model=teacher_model,
                    num_label_tokens=num_label_tokens,
                )
                ce_total = ce_total + ce_depth
                kd_total = kd_total + kd_depth
            return ce_total / len(student_values), kd_total / len(student_values)

        student_head = _get_lm_head_module(student_model) if student_is_hidden else None
        teacher_head = (
            _get_lm_head_module(teacher_model) if needs_mtp_kd and teacher_is_hidden else None
        )
        if student_is_hidden and student_head is None:
            raise ValueError("MTP losses require an accessible student lm_head")
        if needs_mtp_kd and teacher_is_hidden and teacher_head is None:
            raise ValueError("MTP KD requires an accessible teacher lm_head")

        configured_chunks = [
            int(getattr(loss, "chunk_size", 0))
            for loss in (self.loss_fn, self.mtp_kd_loss_fn)
            if loss is not None and int(getattr(loss, "chunk_size", 0)) > 0
        ]
        chunk_size = min(configured_chunks) if configured_chunks else 128
        ce_total = self._local_zero(student_values[0])
        kd_total = self._local_zero(student_values[0])
        cur_labels = labels

        for depth, student_value in enumerate(student_values):
            cur_labels = roll_tensor(cur_labels, shifts=-1, dim=-1)
            depth_labels = cur_labels.clone()
            depth_labels[..., -(depth + 1) :] = -100
            if seq_idx is not None:
                rolled = roll_tensor(seq_idx, shifts=-(depth + 1), dim=-1)
                depth_labels = torch.where(rolled == seq_idx, depth_labels, -100)

            flat_student = self._flatten_tokens(student_value)
            flat_teacher = self._flatten_tokens(teacher_values[depth]) if needs_mtp_kd else None
            flat_labels = depth_labels.reshape(-1)
            for start in range(0, flat_student.shape[0], chunk_size):
                stop = min(start + chunk_size, flat_student.shape[0])
                student_chunk = flat_student[start:stop]
                labels_chunk = flat_labels[start:stop]
                teacher_chunk = (
                    flat_teacher[start:stop]
                    if flat_teacher is not None
                    else student_chunk.new_empty((0, student_chunk.shape[-1]))
                )

                def _chunk_objectives(s_chunk, t_chunk, chunk_labels):
                    phase = f"mtp_depth_{depth}_chunk_{start}_{stop}"
                    _trace_global_kd_phase(f"{phase}_student_head_begin")
                    if student_is_hidden:
                        if student_head is None:
                            raise RuntimeError("MTP hidden-state projection is missing its lm_head")
                        s_chunk = _align_dtensor_to_module_mesh(s_chunk, student_head)
                        s_logits = student_head(s_chunk)
                    else:
                        s_logits = s_chunk
                    _trace_global_kd_phase(f"{phase}_student_head_end")
                    zero = self._local_zero(s_logits)
                    _trace_global_kd_phase(f"{phase}_ce_begin")
                    ce = (
                        self._ce_without_inner_checkpoint(
                            s_logits,
                            chunk_labels,
                            student_model,
                            num_label_tokens,
                        )
                        if self.objective["mtp_ce"] > 0
                        else zero
                    )
                    _trace_global_kd_phase(f"{phase}_ce_end")
                    kd = zero
                    if needs_mtp_kd:
                        _trace_global_kd_phase(f"{phase}_teacher_head_begin")
                        with torch.no_grad():
                            t_logits = (
                                _project_teacher_hidden_on_reference_mesh(
                                    t_chunk, teacher_head, s_logits
                                )
                                if teacher_is_hidden
                                else t_chunk
                            )
                        _trace_global_kd_phase(f"{phase}_teacher_head_end")
                        _trace_global_kd_phase(f"{phase}_kd_begin")
                        kd = self._kd_without_inner_checkpoint(
                            s_logits,
                            t_logits,
                            chunk_labels,
                            num_label_tokens,
                        )
                        _trace_global_kd_phase(f"{phase}_kd_end")
                    return ce, kd

                if torch.is_grad_enabled() and student_chunk.requires_grad:
                    ce_chunk, kd_chunk = checkpoint(
                        _chunk_objectives,
                        student_chunk,
                        teacher_chunk,
                        labels_chunk,
                        use_reentrant=False,
                    )
                else:
                    ce_chunk, kd_chunk = _chunk_objectives(
                        student_chunk, teacher_chunk, labels_chunk
                    )
                ce_total = ce_total + ce_chunk
                kd_total = kd_total + kd_chunk

        depth_count = len(student_values)
        return ce_total / depth_count, kd_total / depth_count

    def _objective_loss(self, student_out, teacher_out, labels, model, num_label_tokens):
        (
            student_logits,
            student_main_is_hidden,
            student_h,
            student_mtp_logits,
            seq_idx,
        ) = _split_output(student_out, model)
        if not self._loss_topology_logged:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                placements = (
                    tuple(
                        type(item).__name__ + f"({getattr(item, 'dim', '')})"
                        for item in student_logits.placements
                    )
                    if isinstance(student_logits, DTensor)
                    else ("replicated_tensor",)
                )
                print(
                    "PUZZLETRON_GLOBAL_KD_LOSS_TOPOLOGY "
                    f"main_logits={type(student_logits).__name__} placements={placements} "
                    f"ce={type(self.loss_fn).__name__} "
                    f"main_kd={type(self.main_kd_loss_fn).__name__} "
                    f"mtp_kd={type(self.mtp_kd_loss_fn).__name__} "
                    f"mtp_hidden={student_h is not None}",
                    flush=True,
                )
            self._loss_topology_logged = True
        zero = self._local_zero(student_logits)
        terms = {name: zero for name in self.objective}

        teacher_model = None
        teacher_h = None
        teacher_mtp_logits = None
        teacher_seq_idx = None
        teacher_logits = None
        teacher_main_is_hidden = False
        if self.needs_teacher:
            teacher_model = self._teacher_loss_model()
            (
                teacher_logits,
                teacher_main_is_hidden,
                teacher_h,
                teacher_mtp_logits,
                teacher_seq_idx,
            ) = _split_output(teacher_out, teacher_model)

        if student_main_is_hidden:
            if self.objective["main_kd"] > 0 and not teacher_main_is_hidden:
                raise ValueError("Main FlashKLD requires teacher hidden-state output")
            main_ce, main_kd = self._hidden_objective_losses(
                domain="main",
                student_hidden=student_logits,
                teacher_hidden=teacher_logits,
                labels=labels,
                student_model=model,
                teacher_model=teacher_model,
                num_label_tokens=num_label_tokens,
            )
            terms["main_ce"] = main_ce
            terms["main_kd"] = main_kd
        else:
            if self.objective["main_ce"] > 0:
                terms["main_ce"] = calculate_loss(
                    self.loss_fn,
                    logits=student_logits,
                    labels=labels,
                    model=model,
                    hidden_states=(
                        None
                        if isinstance(student_out, tuple)
                        else _get_final_hidden_states(student_out)
                    ),
                    num_label_tokens=num_label_tokens,
                )
            if self.objective["main_kd"] > 0:
                terms["main_kd"] = self.main_kd_loss_fn(
                    student_logits,
                    teacher_logits,
                    labels,
                    num_batch_labels=num_label_tokens,
                )

        if self.objective["mtp_ce"] > 0 or self.objective["mtp_kd"] > 0:
            mtp_ce, mtp_kd = self._mtp_objective_losses(
                student_h=student_h,
                student_logits=student_mtp_logits,
                teacher_h=teacher_h,
                teacher_logits=teacher_mtp_logits,
                labels=labels,
                student_model=model,
                teacher_model=teacher_model,
                num_label_tokens=num_label_tokens,
                seq_idx=seq_idx if seq_idx is not None else teacher_seq_idx,
            )
            terms["mtp_ce"] = mtp_ce
            terms["mtp_kd"] = mtp_kd

        total = sum(self.objective[name] * value for name, value in terms.items())
        for name, value in terms.items():
            self._objective_buffers[name].append(value.detach().clone())
        return total, terms

    def _make_pp_kd_loss_wrapper(self):
        def loss_wrapper(student_out, target, **_kwargs):
            teacher_out = None
            if self.needs_teacher:
                queue: deque = getattr(self, "_teacher_logits_queue", deque())
                if queue:
                    teacher_out = queue.popleft()
                else:
                    teacher_out = getattr(self, "_current_teacher_logits", None)
                    self._current_teacher_logits = None
                if teacher_out is None:
                    capture = getattr(
                        getattr(self, "teacher_model", None),
                        "_teacher_logits_capture",
                        None,
                    )
                    if capture:
                        queue = deque(item for item in capture if item is not None)
                        capture.clear()
                        if queue:
                            teacher_out = queue.popleft()
                            self._teacher_logits_queue = queue
                if teacher_out is None:
                    raise RuntimeError("Teacher PP output queue is empty")
            model = next(
                part for part, stage in zip(self.model_parts, self.pp.info.stages) if stage.is_last
            )
            # PP schedules rescale by the optimizer-step label count after
            # backward, so every microbatch loss must remain an unnormalized sum.
            total, terms = self._objective_loss(student_out, teacher_out, target, model, 1)
            self._ce_loss_buffer.append(terms["main_ce"].detach().clone())
            self._kd_loss_buffer.append(terms["main_kd"].detach().clone())
            return total

        return loss_wrapper


class KnowledgeDistillationRecipeForNextTokenPrediction(_WeightedObjectiveMixin, _AutoModelLLMKD):
    """AutoModel LLM KD with independently weighted main/MTP objectives."""

    def setup(self):
        self._configure_objective()
        self._puzzletron_global_kd_domain = "llm"
        self._ce_loss_buffer = []
        self._kd_loss_buffer = []
        self.kd_loss_fn = self.main_kd_loss_fn
        self.kd_ratio = 0.5
        if not self.needs_teacher:
            TrainFinetuneRecipeForNextTokenPrediction.setup(self)
        else:
            _verify_llm_tokenizers(self.cfg.model, self.cfg.teacher_model)
            TrainFinetuneRecipeForNextTokenPrediction.setup(self)
            if self.pp_enabled:
                self.teacher_model = _build_pp_teacher(self, domain="llm")
                self.teacher_pp = self.teacher_model
            else:
                self.teacher_model = _build_llm_teacher(
                    self.cfg.teacher_model,
                    self.cfg.get("seed", 42),
                    self.cfg.get("packed_sequence.packed_sequence_size", 0) > 0,
                    distributed_setup=self.distributed_setup,
                    device=self.dist_env.device,
                )
                _install_distillation_head_passthrough([self.teacher_model])
                self.teacher_pp = None
            self._post_teacher_setup()
        self._configure_incremental_metric_logging()
        _install_distillation_head_passthrough(self.model_parts)
        if self.pp_enabled:
            self._rebind_optimizer_to_current_model_parameters()
            self._install_pre_optimizer_save_rebind()
        _attach_global_kd_gdn_traces(
            self.model_parts,
            prefix="student",
            trace_backward=True,
        )
        self._install_gradient_norm_observers()
        if self.pp_enabled:
            self.pp.info.schedule._loss_fn = self._make_pp_kd_loss_wrapper()

    def _forward_backward_step(
        self,
        idx,
        batch,
        *,
        num_label_tokens,
        num_batches,
        is_train=True,
        loss_buffer=None,
    ):
        self._ensure_student_mtp_outputs()
        if self.pp_enabled:
            if self.needs_teacher:
                return self._forward_backward_step_pp(
                    idx,
                    batch,
                    loss_buffer=loss_buffer,
                    num_label_tokens=num_label_tokens,
                    num_batches=num_batches,
                    is_train=is_train,
                )
            return TrainFinetuneRecipeForNextTokenPrediction._forward_backward_step(
                self,
                idx,
                batch,
                num_label_tokens=num_label_tokens,
                num_batches=num_batches,
                is_train=is_train,
                loss_buffer=loss_buffer,
            )
        batch = {
            key: value.to(self.dist_env.device, non_blocking=True) for key, value in batch.items()
        }
        # Current AutoModel CP owns label sharding through the batch mapping.
        # Keep labels present until CP has padded/sharded every sequence tensor,
        # then remove the CP-local labels for the weighted objective.
        train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch)
        labels = batch.pop("labels")
        model = self.model_parts[0]
        sync_ctx = (
            get_sync_ctx(
                model,
                idx == num_batches - 1,
                defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
            )
            if is_train
            else nullcontext()
        )
        with train_ctx(), sync_ctx:
            teacher_out = None
            if self.needs_teacher:
                with ScopedModuleOffloading(self.teacher_model, enabled=False), torch.no_grad():
                    teacher_out = self.teacher_model(
                        **filter_forward_kwargs(self.teacher_model, batch)
                    )
            student_out = model(**filter_forward_kwargs(model, batch))
            total, terms = self._objective_loss(
                student_out, teacher_out, labels, model, num_label_tokens
            )
            if is_train:
                (total * self._get_dp_group_size(include_cp=True)).backward()
        detached = total.detach().clone()
        if loss_buffer is not None:
            loss_buffer.append(detached)
        return detached, terms["main_kd"].detach().clone(), terms["main_ce"].detach().clone()

    def _forward_backward_step_pp(
        self,
        idx,
        batch,
        *,
        loss_buffer,
        num_label_tokens,
        num_batches,
        is_train=True,
    ):
        """Run PP KD with metadata and targets based on the live CP-local batch."""
        del idx, num_batches
        batch = {
            key: (
                {
                    nested_key: nested_value.to(self.dist_env.device, non_blocking=True)
                    for nested_key, nested_value in value.items()
                    if nested_value is not None
                }
                if isinstance(value, dict)
                else (
                    value.to(self.dist_env.device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                )
            )
            for key, value in batch.items()
        }
        train_ctx, batch = make_cp_batch_and_ctx(
            self.device_mesh,
            batch,
            use_te=_uses_te_dot_product_attention(self.cfg.model)
            and _uses_thd_collater(self.cfg.dataloader),
            padding_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
            num_chunks=_get_num_thd_chunks(True, self.cfg),
        )
        labels = batch.pop("labels")
        input_ids = batch.pop("input_ids")
        batch_filtered = {
            key: value
            for key, value in batch.items()
            if value is not None and not (isinstance(value, dict) and not value)
        }
        fp8_ctx = self.te_fp8.maybe_te_autocast() if self.te_fp8 is not None else nullcontext()

        with train_ctx(), fp8_ctx:
            # Entering CP resizes its no-restore buffers in place. PyTorch PP
            # metadata and loss targets must reflect that post-shard shape.
            local_seq_len = input_ids.shape[1]
            for pp_model in (self.teacher_pp, self.pp):
                _reset_global_kd_pp_stage_shapes(pp_model, seq_len=local_seq_len)
                _refresh_pp_hidden_output_meta(pp_model)
                set_pp_vlm_chunk_specs(
                    pp_model.info.schedule,
                    batch_filtered,
                    batch_size=int(input_ids.shape[0]),
                )
            targets = labels.clone() if self.pp.info.has_last_stage else None

            if os.environ.get("PUZZLETRON_TRACE_GLOBAL_KD") == "1":
                rank = torch.distributed.get_rank()
                print(
                    "PUZZLETRON_GLOBAL_KD_PP_INPUT "
                    f"rank={rank} first={self.pp.info.has_first_stage} "
                    f"input_shape={tuple(input_ids.shape)} labels_shape={tuple(labels.shape)} "
                    f"microbatches={getattr(self.pp.info.schedule, '_n_microbatches', None)} "
                    f"args_spec={getattr(self.pp.info.schedule, '_args_chunk_spec', None)} "
                    f"kwargs_shapes={{{', '.join(f'{key}: {tuple(value.shape)}' for key, value in batch_filtered.items() if isinstance(value, torch.Tensor))}}}",
                    flush=True,
                )

            _trace_global_kd_phase("llm_teacher_schedule_begin")
            # KD losses (especially checkpointed MTP terms) save teacher outputs
            # as backward inputs. ``inference_mode`` creates tensors that autograd
            # is forbidden to save; ``no_grad`` keeps the teacher frozen while
            # producing ordinary tensors suitable for the student loss graph.
            with torch.no_grad():
                teacher_losses = [] if self.teacher_pp.info.has_last_stage else None
                if self.teacher_pp.info.has_first_stage:
                    self.teacher_pp.info.schedule.eval(
                        input_ids, target=targets, losses=teacher_losses, **batch_filtered
                    )
                else:
                    self.teacher_pp.info.schedule.eval(
                        target=targets, losses=teacher_losses, **batch_filtered
                    )
                capture = getattr(self.teacher_model, "_teacher_logits_capture", None)
                if capture is not None and capture[0] is not None:
                    self._current_teacher_logits = capture[0]
                    capture[0] = None
                else:
                    self._current_teacher_logits = None
                self._current_num_label_tokens = num_label_tokens
            _trace_global_kd_phase("llm_teacher_schedule_end")

            student_losses = [] if self.pp.info.has_last_stage else None
            schedule_method = self.pp.info.schedule.step if is_train else self.pp.info.schedule.eval
            _trace_global_kd_phase("llm_student_schedule_begin")
            if self.pp.info.has_first_stage:
                schedule_method(input_ids, target=targets, losses=student_losses, **batch_filtered)
            else:
                schedule_method(target=targets, losses=student_losses, **batch_filtered)
            _trace_global_kd_phase("llm_student_schedule_end")

            if self.pp.info.has_last_stage:
                loss_buffer.append(torch.sum(torch.stack(student_losses)).detach().clone())
            else:
                loss_buffer.append(torch.tensor(0.0, device=self.dist_env.device))

    def _run_train_optim_step(self, batches, max_grad_norm=None):
        if self.needs_teacher:
            log_data = _AutoModelLLMKD._run_train_optim_step(self, batches, max_grad_norm)
        else:
            log_data = TrainFinetuneRecipeForNextTokenPrediction._run_train_optim_step(
                self, batches, max_grad_norm
            )

        # Publish every weighted objective independently.  The shared helper
        # handles the last-stage-to-rank-zero route under PP for both LLM and
        # VLM recipes.
        self._publish_objective_metrics(log_data)
        if self.dist_env.is_main:
            print(
                "PUZZLETRON_GLOBAL_KD_OBJECTIVES "
                f"step={log_data.step} "
                + " ".join(
                    f"{name}={log_data.metrics[name]:.8f}"
                    for name in ("main_ce", "main_kd", "mtp_ce", "mtp_kd")
                ),
                flush=True,
            )
        log_data.metrics.update(self._consume_gradient_norms())
        return log_data

    def run_train_validation_loop(self):
        # The upstream KD loop deliberately skips validation under PP.  Calling
        # the base fine-tuning loop here makes it invoke KD's PP validation
        # stub, then dereference the returned None as metric data after an
        # otherwise successful optimizer step.
        if self.needs_teacher:
            return _AutoModelLLMKD.run_train_validation_loop(self)
        return TrainFinetuneRecipeForNextTokenPrediction.run_train_validation_loop(self)


def _clone_tensor_tree(value: Any) -> Any:
    """Separate teacher CP buffers from student buffers before in-place sharding."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item) for item in value)
    return value


class KnowledgeDistillationRecipeForVLM(_WeightedObjectiveMixin, _AutoModelVLMKD):
    """AutoModel VLM KD, including a teacher-first PP pass."""

    def setup(self):
        self._configure_objective()
        self._puzzletron_global_kd_domain = "vlm"
        self._ce_loss_buffer = []
        self._kd_loss_buffer = []
        self.kd_loss_fn = self.main_kd_loss_fn
        self.kd_ratio = 0.5
        if self.needs_teacher:
            _verify_vlm_tokenizers(self.cfg.model, self.cfg.teacher_model)
        FinetuneRecipeForVLM.setup(self)
        self._configure_incremental_metric_logging()
        _install_distillation_head_passthrough(self.model_parts)
        freeze_policy = str(self.cfg.get("freeze_policy", "vision_frozen"))
        if freeze_policy == "projector_and_language":
            for part in self.model_parts:
                for name, parameter in part.named_parameters():
                    components = set(name.split("."))
                    if "visual" in components and "merger" not in components:
                        parameter.requires_grad_(False)
        elif freeze_policy not in ("vision_frozen", "train_all"):
            raise ValueError(f"unknown global KD freeze_policy={freeze_policy!r}")
        if self.needs_teacher:
            if self.pp_enabled:
                self.teacher_model = _build_pp_teacher(self, domain="vlm")
                self.teacher_pp = self.teacher_model
            else:
                self.teacher_model = _build_vlm_teacher(
                    self.cfg.teacher_model,
                    self.cfg.get("teacher_freeze_config", None),
                    self.cfg.get("seed", 42),
                    distributed_setup=self.distributed_setup,
                    device=self.dist_env.device,
                )
                _install_distillation_head_passthrough([self.teacher_model])
                self.teacher_pp = None
            self._post_teacher_setup()
        if self.pp_enabled:
            self._rebind_optimizer_to_current_model_parameters()
            self._install_pre_optimizer_save_rebind()
        self._install_vision_observers(self.model_parts, role="student")
        if self.needs_teacher:
            teacher_parts = getattr(self.teacher_model, "parts", [self.teacher_model])
            self._install_vision_observers(teacher_parts, role="teacher")
        self._install_gradient_norm_observers()
        if self.pp_enabled:
            self.pp.info.schedule._loss_fn = self._make_pp_kd_loss_wrapper()

    def _forward_backward_step(
        self, idx, batch, *, loss_buffer, num_label_tokens, num_batches, is_train=True
    ):
        self._record_media_batch(batch)
        self._ensure_student_mtp_outputs()
        if self.pp_enabled:
            return self._forward_backward_step_pp(
                batch,
                loss_buffer=loss_buffer,
                num_label_tokens=num_label_tokens,
                is_train=is_train,
            )
        batch = {key: _move_to_device(value, self.dist_env.device) for key, value in batch.items()}
        model = self.model_parts[0]
        cp_active = (
            self.device_mesh is not None
            and "cp" in getattr(self.device_mesh, "mesh_dim_names", ())
            and self.device_mesh["cp"].size() > 1
        )
        if cp_active and hasattr(model, "prepare_model_inputs_for_cp"):
            media = {key: batch[key] for key in VLM_INPUT_KEYS if batch.get(key) is not None}
            with torch.no_grad():
                prepared = model(_pre_embed_only=True, **media)
            if self.needs_teacher and "inputs_embeds" in prepared:
                _validate_cp_pre_embed_teacher_compatibility(
                    prepared["inputs_embeds"], self.teacher_model
                )
            for key in VLM_INPUT_KEYS:
                batch.pop(key, None)
            batch.update(prepared)
        train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch)
        labels = batch.pop("labels")
        sync_ctx = (
            get_sync_ctx(
                model,
                idx == num_batches - 1,
                defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
            )
            if is_train
            else nullcontext()
        )
        with train_ctx(), sync_ctx:
            teacher_out = None
            if self.needs_teacher:
                with torch.no_grad():
                    teacher_out = self.teacher_model(
                        **filter_forward_kwargs(self.teacher_model, batch)
                    )
            student_out = model(**filter_forward_kwargs(model, batch))
            total, _ = self._objective_loss(
                student_out, teacher_out, labels, model, num_label_tokens
            )
            loss_buffer.append(total.detach().clone())
            if is_train:
                (total * self._get_dp_group_size(include_cp=True)).backward()

    def _forward_backward_step_pp(self, batch, *, loss_buffer, num_label_tokens, is_train):
        self._ensure_student_mtp_outputs()
        batch = {key: _move_to_device(value, self.dist_env.device) for key, value in batch.items()}
        teacher_batch = _clone_tensor_tree(batch)

        def prepare_cp_inputs(pp, parts, values):
            cp_active = (
                self.device_mesh is not None
                and "cp" in getattr(self.device_mesh, "mesh_dim_names", ())
                and self.device_mesh["cp"].size() > 1
            )
            if not cp_active:
                return values
            if pp.info.has_first_stage and hasattr(parts[0], "prepare_model_inputs_for_cp"):
                media = {key: values[key] for key in VLM_INPUT_KEYS if values.get(key) is not None}
                prepared = parts[0](_pre_embed_only=True, **media)
                for key in VLM_INPUT_KEYS:
                    values.pop(key, None)
                values.update(prepared)
            elif not pp.info.has_first_stage:
                for key in VLM_INPUT_KEYS:
                    if key != "input_ids":
                        values.pop(key, None)
            return values

        batch = prepare_cp_inputs(self.pp, self.model_parts, batch)
        if self.needs_teacher:
            teacher_batch = prepare_cp_inputs(self.teacher_pp, self.teacher_pp.parts, teacher_batch)
        train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch)
        labels = batch.pop("labels")
        model_input_key = "inputs_embeds" if "inputs_embeds" in batch else "input_ids"
        model_input = batch.pop(model_input_key)
        # VLM+CP prepares one differentiable multimodal embedding graph for the
        # full local batch before PP splits it into microbatches.  Backwarding
        # each PP microbatch directly through that shared graph frees it on the
        # first microbatch and fails on the second.  Let PP accumulate gradients
        # into a leaf boundary, then apply the exact accumulated gradient to the
        # original CP-sharded embedding graph once after the schedule completes.
        # This preserves gradients for both the vision tower and token embeddings
        # without retain_graph=True or freezing either module.
        embedding_graph = None
        schedule_input = model_input
        if (
            is_train
            and self.pp.info.has_first_stage
            and model_input_key == "inputs_embeds"
            and model_input.requires_grad
        ):
            embedding_graph = model_input
            schedule_input = model_input.detach().requires_grad_(True)
        if self.needs_teacher:
            teacher_ctx, teacher_batch = make_cp_batch_and_ctx(self.device_mesh, teacher_batch)
            teacher_labels = teacher_batch.pop("labels")
            teacher_input_key = "inputs_embeds" if "inputs_embeds" in teacher_batch else "input_ids"
            teacher_input = teacher_batch.pop(teacher_input_key)
            with teacher_ctx():
                teacher_targets = (
                    teacher_labels.clone() if self.teacher_pp.info.has_last_stage else None
                )
                self.teacher_pp.update_seq_len(teacher_input.shape[1])
                _refresh_pp_hidden_output_meta(self.teacher_pp)
                set_pp_vlm_chunk_specs(self.teacher_pp.info.schedule, teacher_batch)
                capture = self.teacher_model._teacher_logits_capture
                capture.clear()
                with (
                    torch.no_grad(),
                    stage_vlm_media_for_pp(self.teacher_pp, self.teacher_pp.parts, teacher_batch),
                ):
                    teacher_losses = [] if self.teacher_pp.info.has_last_stage else None
                    if self.teacher_pp.info.has_first_stage:
                        self.teacher_pp.info.schedule.eval(
                            teacher_input,
                            target=teacher_targets,
                            losses=teacher_losses,
                            **teacher_batch,
                        )
                    else:
                        self.teacher_pp.info.schedule.eval(
                            target=teacher_targets, losses=teacher_losses, **teacher_batch
                        )
                self._teacher_logits_queue = deque(capture)
                capture.clear()
        with train_ctx():
            targets = labels.clone() if self.pp.info.has_last_stage else None
            self.pp.update_seq_len(schedule_input.shape[1])
            _refresh_pp_hidden_output_meta(self.pp)
            self._maybe_set_pp_first_stage_embed_input_meta(schedule_input)
            losses = [] if self.pp.info.has_last_stage else None
            schedule_method = self.pp.info.schedule.step if is_train else self.pp.info.schedule.eval
            set_pp_vlm_chunk_specs(self.pp.info.schedule, batch)
            with stage_vlm_media_for_pp(self.pp, self.model_parts, batch):
                if self.pp.info.has_first_stage:
                    schedule_method(schedule_input, target=targets, losses=losses, **batch)
                else:
                    schedule_method(target=targets, losses=losses, **batch)
            if embedding_graph is not None:
                if schedule_input.grad is None:
                    raise RuntimeError(
                        "VLM PP schedule produced no gradient for the multimodal embedding boundary"
                    )
                embedding_graph.backward(schedule_input.grad)
        loss_buffer.append(
            torch.stack(losses).sum().detach().clone()
            if self.pp.info.has_last_stage
            else torch.zeros((), device=self.dist_env.device)
        )

    def _run_train_optim_step(self, batches, max_grad_norm=None):
        log_data = FinetuneRecipeForVLM._run_train_optim_step(self, batches, max_grad_norm)

        # The shared publisher normalizes last-stage PP microbatch sums and
        # forwards every objective term to rank zero.
        objective_metrics = self._publish_objective_metrics(log_data)
        # Compatibility aliases for AutoModel's VLM-KD logger, plus its display
        # metadata.  The JSONL record retains all four objective components.
        log_data.metrics.update(
            {
                "ce_loss": objective_metrics["main_ce"],
                "kd_loss": objective_metrics["main_kd"],
                "kd_ratio": self.kd_ratio,
                "temperature": getattr(self.kd_loss_fn, "temperature", float("nan")),
            }
        )
        log_data.metrics.update(self._consume_gradient_norms())
        return log_data
