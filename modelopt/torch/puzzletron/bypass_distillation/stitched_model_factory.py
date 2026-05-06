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

"""Factory for creating stitched teacher/student models for bypass distillation."""

import copy
import dataclasses
import re
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW, Optimizer
from transformers import PretrainedConfig, PreTrainedModel

import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptor
from modelopt.torch.puzzletron.pruning.pruning_utils import GQAInitMode, LinearInitMode, MlpInitMode
from modelopt.torch.puzzletron.sewing_kit import (
    ExternalTarget,
    FunctionTarget,
    InputArgs,
    ModuleTarget,
    Needle,
    RemoteTarget,
    StitchedModule,
    always_true_predicate,
)
from modelopt.torch.puzzletron.sewing_kit.core import InputReducer
from modelopt.torch.puzzletron.sewing_kit.utils import (
    batched_normalized_mse_loss,
    normalized_mse_loss,
    vectorwise_normalized_mse_loss,
)
from modelopt.torch.puzzletron.tools.bypassed_training.child_init import (
    create_child_state_dict,
    update_model_config,
)
from modelopt.torch.puzzletron.tools.logger import mprint
from modelopt.torch.puzzletron.tools.sharded_checkpoint_utils import create_sharded_model
from modelopt.torch.puzzletron.utils.parsing import format_block_configs, parse_dtype

StitchedModulesProcessOwnership = list[int]
SyncDistributedModelWeightsFn = Callable[[], None]
Config = Mapping[str, Any]
Args = Namespace


@dataclasses.dataclass
class StitchedModuleDescriptor:
    stitched_module: StitchedModule
    owned_parameters: dict[str, torch.nn.Parameter]
    owned_buffers: dict[str, torch.Tensor]
    optimizer: Optional[Optimizer] = None
    grad_scaler: Optional[GradScaler] = None


_SUBBLOCK_KEYS_TO_LEARN = frozenset(
    {"subblock_ffn", "subblock_attention", "subblock_mamba", "entire_block"}
)


def _set_keys_to_learn(
    model: PreTrainedModel,
    descriptor: ModelDescriptor,
    keys_to_learn: str | Sequence[str],
) -> None:
    """Set ``requires_grad=True`` on parameters selected by ``keys_to_learn``.

    * A **sequence of strings** (not a bare ``str``): each string is a full parameter
      name; gradients are enabled only where ``named_parameters()`` names match exactly.
    * A **single string**: if it is ``"subblock_ffn"``, ``"subblock_attention"``, or
      ``"entire_block"``, enables gradients for the corresponding descriptor weight
      groups; otherwise ``re.search`` is applied to each parameter name.
    """
    # If keys_to_learn is a sequence of strings.
    if isinstance(keys_to_learn, Sequence) and not isinstance(keys_to_learn, str):
        param_names = set(keys_to_learn)
    # If keys_to_learn is a single string.
    # If keys_to_learn is a single string that is a subblock key.
    elif keys_to_learn in _SUBBLOCK_KEYS_TO_LEARN:
        lm_config = descriptor.get_language_model_config(model.config)
        weight_groups = descriptor.get_weight_groups(
            model.state_dict().keys(), lm_config.num_hidden_layers
        )

        attn_group_names = [
            group_name for group_name in weight_groups.keys() if group_name.endswith("_attention")
        ]
        ffn_group_names = [
            group_name for group_name in weight_groups.keys() if group_name.endswith("_ffn")
        ]
        if keys_to_learn == "subblock_attention":
            group_names = attn_group_names
        elif keys_to_learn == "subblock_ffn":
            group_names = ffn_group_names
        elif keys_to_learn == "subblock_mamba":
            group_names = attn_group_names  # Mamba params live in _attention groups
        else:  # entire_block
            group_names = attn_group_names + ffn_group_names

        block_configs = getattr(lm_config, "block_configs", None)

        collected: list[str] = []
        for group_name in group_names:
            # For hybrid models (e.g. NemotronH), a single "_attention" group
            # name can contain either Mamba SSM params *or* GQA params depending
            # on the block.  Use the block config — not the keys_to_learn string
            # — to decide whether each block belongs to the current subblock type.
            if block_configs is not None:
                m = re.match(r"block_(\d+)_attention", group_name)
                if m:
                    block_idx = int(m.group(1))
                    if block_idx < len(block_configs):
                        is_mamba = (
                            getattr(block_configs[block_idx].attention, "mamba", None) is not None
                        )
                        # subblock_attention → GQA blocks only (not Mamba)
                        # subblock_mamba     → Mamba blocks only (not GQA)
                        # entire_block       → all blocks (no filtering)
                        if keys_to_learn == "subblock_attention" and is_mamba:
                            continue
                        if keys_to_learn == "subblock_mamba" and not is_mamba:
                            continue
            collected.extend(weight_groups[group_name])
        param_names = set(collected)
    # If keys_to_learn is a single string that is not a subblock key, treat as regex.
    else:
        param_names = {
            param_name
            for param_name, _ in model.named_parameters()
            if re.search(keys_to_learn, param_name)
        }
    # In pipeline-parallel training a rank may own only blocks that don't match
    # keys_to_learn (e.g. a rank with only Mamba blocks during subblock_attention
    # bypass has no GQA params after the _mamba rename).  That is a valid state:
    # all its blocks will produce NaN loss and be excluded from statistics.
    if not param_names:
        return

    # Set requires_grad to True for the selected parameters.
    for param_name, param in model.named_parameters():
        if param_name in param_names and torch.is_floating_point(param):
            param.requires_grad_(True)


def _get_all_non_persistent_buffers_set(module: torch.nn.Module) -> set[str]:
    all_non_persistent = set()
    for module_name, submodule in module.named_modules():
        for buffer_name in submodule._non_persistent_buffers_set:
            full_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
            all_non_persistent.add(full_name)
    return all_non_persistent


def bypass_factory_fn(
    teacher_model: PreTrainedModel,
    descriptor: ModelDescriptor,
    cfg: DictConfig,
    model_blocks_process_ownership: Sequence[int],
    student_model: Optional[PreTrainedModel] = None,
) -> tuple[
    PreTrainedModel,
    StitchedModule,
    StitchedModule,
    StitchedModule,
    OrderedDict[str, StitchedModuleDescriptor],
    PretrainedConfig,
]:
    """Unified factory function for bypass (blockwise local) distillation.

    Handles all layer types — FFN, attention (GQA/MHA), MoE experts, Mamba, and whole blocks —
    through a single pipeline. Behavior is driven entirely by ``model_factory`` config fields:

    - ``mlp_init_mode``: how student FFN / MoE weights are initialised
        - ``"ExpertRemoval"``: select top-N experts from teacher (MoE models)
        - ``"Truncate"`` / ``"PruneByActivationsLog"``: prune FFN channels (dense models)
        - ``"CopyAsIs"``: copy weights unchanged (attention-only or Mamba-only runs)
    - ``gqa_init_mode``: how attention KV heads are initialised (optional, default ``AverageKV``).
      Irrelevant when the student has the same number of KV heads as the teacher.
    - ``keys_to_learn``: which parameters to train.
      Accepts ``"subblock_ffn"``, ``"subblock_attention"``, ``"entire_block"``, or a regex string.

    The stitching logic (pipeline-parallel per-block KD) is architecture-agnostic and unchanged
    regardless of which layer type is being distilled.

    Args:
        teacher_model: The teacher model to use for stitching.
        descriptor: Model descriptor for layer naming and pruning mixin lookup.
        cfg: The bypass config section.
        model_blocks_process_ownership: Ownership mapping of model blocks to process ranks.
        student_model: Optionally provided pre-built student model (skips initialisation).

    Returns:
        Tuple of (student_model, teacher_stitched, teacher_val_stitched,
                  student_val_stitched, stitched_module_descriptors, student_config)
    """
    device = torch.device(f"cuda:{dist.local_rank()}")
    model_config_overrides = cfg.model.model_config_overrides

    _block_loss_funcs: dict[str, Callable[..., Any]] = {
        "normalized_mse_loss": normalized_mse_loss,
        "vectorwise_normalized_mse_loss": vectorwise_normalized_mse_loss,
        "batched_normalized_mse_loss": batched_normalized_mse_loss,
    }
    block_loss_func = _block_loss_funcs[cfg.model_factory.block_loss_func]
    mprint(f"{block_loss_func.__name__=}")

    owned_block_indexes = set(
        block_index
        for block_index, owner_rank in enumerate(model_blocks_process_ownership)
        if owner_rank == dist.rank()
    )

    # Initialize student_model
    if student_model is None:
        mprint("Creating student model from teacher model")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if isinstance(model_config_overrides, DictConfig):
                config_to_override = OmegaConf.to_container(model_config_overrides, resolve=True)
            else:
                config_to_override = model_config_overrides
            mprint(f"{config_to_override=}")
            student_model_config = update_model_config(
                model_config=teacher_model.config,
                model_config_overrides=config_to_override,
            )
            student_model_config.use_cache = False

            mprint(f"Student model config:\n {format_block_configs(student_model_config)}")

            from modelopt.torch.puzzletron.anymodel.puzzformer import deci_x_patcher

            runtime = Namespace(
                device=device,
                dtype=torch.bfloat16,
                global_rank=dist.rank(),
                world_size=dist.size(),
                is_main_process=dist.is_master(),
                is_last_process=dist.is_last_process(),
            )

            with deci_x_patcher(
                model_descriptor=descriptor,
                block_configs=getattr(student_model_config, "block_configs", None),
            ):
                student_model = create_sharded_model(
                    runtime=runtime,
                    descriptor=descriptor,
                    model_config=student_model_config,
                    owned_block_indexes=owned_block_indexes,
                    device=device,
                )
                # `_init_weights` is HF's per-module initializer; apply it across the
                # whole model rather than passing the model itself as a single module.
                student_model.apply(student_model._init_weights)

        student_weights_dtype = parse_dtype(cfg.model.student_weights_dtype)
        descriptor.init_rotary_embedding(student_model, runtime)
        student_model.type(student_weights_dtype)

        mlp_init_mode = MlpInitMode(cfg.model_factory.mlp_init_mode or MlpInitMode.CopyAsIs)

        # For expert removal, use the model-specific pruning mixin so that model-specific
        # key paths (e.g. backbone.layers.{i}.mixer for Nemotron-H vs model.layers.{i}.mlp
        # for GPT-OSS) are handled correctly. For all other init modes the legacy inline
        # key logic in create_child_state_dict is sufficient.
        _mixins = []
        if mlp_init_mode == MlpInitMode.ExpertRemoval:
            _expert_mixin = descriptor.pruning_mixins().get("experts_removal")
            if _expert_mixin is not None:
                _mixins.append(_expert_mixin)

        # If any attention layer has fewer KV heads in the student than the teacher, use the
        # model-specific KV heads mixin so that k_proj/v_proj weights are correctly sliced
        # rather than copied verbatim from the (larger) teacher state dict.
        _kv_mixin = descriptor.pruning_mixins().get("kv_heads")
        if _kv_mixin is not None:
            _student_kv = [
                b.attention.num_key_value_heads
                for b in student_model_config.block_configs
                if b.attention is not None and b.attention.num_key_value_heads is not None
            ]
            _teacher_kv = [
                b.attention.num_key_value_heads
                for b in teacher_model.config.block_configs
                if b.attention is not None and b.attention.num_key_value_heads is not None
            ]
            if _student_kv != _teacher_kv:
                _mixins.append(_kv_mixin)

        # If any FFN layer has a smaller intermediate_size in the student than the teacher,
        # use the model-specific FFN-intermediate mixin. The generic create_child_state_dict
        # path is hardcoded to `model.layers.{i}.mlp.*` (Llama-style), so for families that
        # place FFN under a different prefix (e.g. `backbone.layers.{i}.mixer.*` for
        # Nemotron-H/H_v2) the mixin is required to slice up_proj/down_proj correctly.
        # Filter out no_op FFN blocks (their intermediate_size is None) — relevant for
        # hybrid families where each layer is exactly one of {attention, ffn, mamba}.
        _ffn_mixin = descriptor.pruning_mixins().get("ffn_intermediate")
        if _ffn_mixin is not None and mlp_init_mode in (
            MlpInitMode.Truncate,
            MlpInitMode.PruneByActivationsLog,
        ):
            _student_ffn = [
                b.ffn.intermediate_size
                for b in student_model_config.block_configs
                if b.ffn is not None and b.ffn.intermediate_size is not None
            ]
            _teacher_ffn = [
                b.ffn.intermediate_size
                for b in teacher_model.config.block_configs
                if b.ffn is not None and b.ffn.intermediate_size is not None
            ]
            if _student_ffn != _teacher_ffn:
                _mixins.append(_ffn_mixin)

        if len(_mixins) == 0:
            pruning_mixin = None
        elif len(_mixins) == 1:
            pruning_mixin = _mixins[0]
        else:
            pruning_mixin = _mixins

        # GQA init mode is optional: only relevant when the student has fewer KV heads than
        # the teacher. Defaults to AverageKV and is a no-op when head counts are equal.
        gqa_init_mode = GQAInitMode(cfg.model_factory.get("gqa_init_mode", GQAInitMode.AverageKV))

        student_state_dict = create_child_state_dict(
            pruning_mixin=pruning_mixin,
            descriptor=descriptor,
            original_state_dict=teacher_model.state_dict(),
            new_state_dict=student_model.state_dict(),
            original_config=teacher_model.config,
            new_config=student_model_config,
            gqa_init_mode=gqa_init_mode,
            mlp_init_mode=mlp_init_mode,
            mlp_init_config=cfg.model_factory.mlp_init_config,
            owned_block_indexes=owned_block_indexes,
            linear_init_mode=LinearInitMode(
                cfg.model_factory.linear_init_mode or LinearInitMode.Random
            ),
        )

        # Load student state dict
        missing_keys, unexpected_keys = student_model.load_state_dict(
            student_state_dict, strict=False
        )
        assert len(unexpected_keys) == 0, f"{unexpected_keys=}"
        # GQA models have learnable logit parameters not present in the teacher state dict;
        # allow those to be absent and assert nothing else is missing.
        non_gqa_missing = [k for k in missing_keys if not re.search(r"gqa_\w+_logits", k)]
        assert len(non_gqa_missing) == 0, f"Unexpected missing keys: {non_gqa_missing}"

    else:
        mprint("Student model provided explicitly, not using teacher model to instantiate")
        student_model_config = student_model.config

    # Set up training parameters
    lm_config = descriptor.get_language_model_config(student_model_config)
    all_block_indices = list(range(lm_config.num_hidden_layers))

    student_model.requires_grad_(False)
    keys_to_learn = cfg.model_factory.keys_to_learn
    mprint(f"Keys to learn: {keys_to_learn}")

    _set_keys_to_learn(model=student_model, descriptor=descriptor, keys_to_learn=keys_to_learn)

    dist.barrier()
    mprint(f"Global rank: {dist.rank()}, {owned_block_indexes=}")
    dist.barrier()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    dist.barrier()

    min_owned_index = min(owned_block_indexes)
    max_owned_index = max(owned_block_indexes)
    prev_rank: Optional[int] = (
        None
        if min_owned_index == min(all_block_indices)
        else model_blocks_process_ownership[min_owned_index - 1]
    )
    next_rank: Optional[int] = (
        None
        if max_owned_index == max(all_block_indices)
        else model_blocks_process_ownership[max_owned_index + 1]
    )

    teacher_parameters = set(teacher_model.parameters())
    teacher_buffers = set(teacher_model.buffers())

    # Setup the student model's submodules for knowledge distillation training
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.device(device):
        stitched_module_descriptors = OrderedDict[str, StitchedModuleDescriptor]()
        submodule_for_loss_calculation = cfg.model_factory.submodule_for_loss_calculation

        teacher_target = ModuleTarget("teacher", teacher_model)
        teacher_stitcher = Needle()
        teacher_val_stitcher = Needle()

        student_target = ModuleTarget("student", student_model)
        student_val_stitcher = Needle()

        for local_block_index, global_block_index in enumerate(sorted(owned_block_indexes)):
            module_name = descriptor.layer_block_name(global_block_index)
            module = student_model.get_submodule(module_name)

            submodule_name = ""
            submodule_input_descriptor = submodule_name
            submodule_output_descriptor = submodule_name

            if submodule_for_loss_calculation is not None:
                assert hasattr(module, submodule_for_loss_calculation)
                submodule_output_descriptor = submodule_for_loss_calculation

            input_descriptor = f"{module_name}.{submodule_input_descriptor}".rstrip(".")
            output_descriptor = f"{module_name}.{submodule_output_descriptor}".rstrip(".")

            # Receive activations from previous rank
            if global_block_index > 0 and local_block_index == 0 and prev_rank is not None:
                teacher_stitcher.stitch(
                    RemoteTarget(peer_rank=prev_rank).value(
                        name="teacher_activations", adapter=lambda x: InputArgs(x)
                    ),
                    teacher_target.input(
                        name=module_name,
                        reducer=InputReducer(
                            lambda acc, override, orig, *args: override + orig.drop_args(0)
                        ),
                    ),
                )
                teacher_val_stitcher.stitch(
                    RemoteTarget(peer_rank=prev_rank).value(
                        name="teacher_activations", adapter=lambda x: InputArgs(x)
                    ),
                    teacher_target.input(
                        name=module_name,
                        reducer=InputReducer(
                            lambda acc, override, orig, *args: override + orig.drop_args(0)
                        ),
                    ),
                )
                student_val_stitcher.stitch(
                    RemoteTarget(peer_rank=prev_rank).value(
                        name="student_activations", adapter=lambda x: InputArgs(x)
                    ),
                    student_target.input(
                        name=module_name,
                        reducer=InputReducer(
                            lambda acc, override, orig, *args: override + orig.drop_args(0)
                        ),
                    ),
                )

            # Send activations to next rank or register model output
            if local_block_index + 1 == len(owned_block_indexes):
                if next_rank is None:
                    student_val_stitcher.stitch(
                        student_target.output(name=""),
                        ExternalTarget().output("model_output"),
                    )
                    teacher_val_stitcher.stitch(
                        teacher_target.output(name=""),
                        ExternalTarget().output("model_output"),
                    )
                else:
                    teacher_stitcher.stitch(
                        teacher_target.output(name=module_name),
                        RemoteTarget(peer_rank=next_rank).value(name="teacher_activations"),
                    )
                    teacher_val_stitcher.stitch(
                        teacher_target.output(name=module_name),
                        RemoteTarget(peer_rank=next_rank).value(name="teacher_activations"),
                    )
                    student_val_stitcher.stitch(
                        student_target.output(name=module_name),
                        RemoteTarget(peer_rank=next_rank).value(name="student_activations"),
                    )

            # Bypass training stitches
            teacher_stitcher.stitch(
                teacher_target.input(name=input_descriptor),
                ExternalTarget().input(name=input_descriptor),
            ).stitch(
                teacher_target.output(name=output_descriptor),
                ExternalTarget().output(name=output_descriptor),
            )

            # Create the student block stitched module
            student_stitched_module_loss_target = FunctionTarget(
                "module_loss_func", block_loss_func
            )
            student_stitched_module_name = f"block_{global_block_index}"
            student_submodule_target = ModuleTarget("student_submodule", module)
            student_stitched_module = (
                Needle()
                .stitch(
                    ExternalTarget().input(name=input_descriptor),
                    student_submodule_target.input(name=submodule_input_descriptor),
                )
                .stitch(
                    ExternalTarget().output(
                        name=output_descriptor,
                        adapter=lambda v: InputArgs(target=v)
                        if not isinstance(v, tuple)
                        else InputArgs(target=v[0]),
                    ),
                    student_stitched_module_loss_target.input(),
                )
                .stitch(
                    student_submodule_target.output(
                        name=submodule_output_descriptor,
                        adapter=lambda v: InputArgs(input=v)
                        if not isinstance(v, tuple)
                        else InputArgs(input=v[0]),
                    ),
                    student_stitched_module_loss_target.input(),
                )
                .stitch(
                    student_stitched_module_loss_target.output(),
                    ExternalTarget().output(name="loss"),
                )
                .knot(
                    ignore_extra_overrides=True,
                    capture_cache_outputs_predicate=always_true_predicate,
                )
            )

            assert "learning_rate" in cfg.training
            num_trainable_params = sum(
                p.requires_grad and submodule_name in p_name
                for p_name, p in student_stitched_module.named_parameters()
                if "dummy_param" not in p_name  # exclude placeholder params
            )
            # Do NOT enable dummy params: blocks with no real trainable parameters
            # (e.g. Mamba blocks during an attention-only bypass run) should produce
            # NaN loss so they are excluded from statistics — identical to the
            # optimizer=None path in the training loop.

            student_module_parameters = {
                p_name: p
                for p_name, p in student_stitched_module.named_parameters()
                if p not in teacher_parameters and "dummy_param" not in p_name
            }
            student_module_buffers = {
                p_name: p
                for p_name, p in student_stitched_module.named_buffers()
                if p not in teacher_buffers
                and p_name not in _get_all_non_persistent_buffers_set(student_stitched_module)
            }

            trainable_params = {
                p_name: p for p_name, p in student_module_parameters.items() if p.requires_grad
            }

            optimizer = (
                AdamW(
                    list(trainable_params.values()),
                    lr=cfg.training.learning_rate,
                    weight_decay=cfg.training.weight_decay,
                    betas=(cfg.training.beta1, cfg.training.beta2),
                    fused=True,
                )
                if len(trainable_params) > 0
                else None
            )

            grad_scaler = (
                None
                if optimizer is None
                else GradScaler(device=device.type, enabled=cfg.training.use_grad_scaling)
            )

            stitched_module_descriptors[student_stitched_module_name] = StitchedModuleDescriptor(
                stitched_module=student_stitched_module,
                owned_parameters=student_module_parameters,
                owned_buffers=student_module_buffers,
                optimizer=optimizer,
                grad_scaler=grad_scaler,
            )

        teacher_stitched_module = teacher_stitcher.knot(ignore_extra_overrides=True)
        teacher_val_stitched_module = teacher_val_stitcher.knot(ignore_extra_overrides=True)
        student_val_stitched_module = student_val_stitcher.knot(ignore_extra_overrides=True)

    return (
        student_model,
        teacher_stitched_module,
        teacher_val_stitched_module,
        student_val_stitched_module,
        stitched_module_descriptors,
        student_model_config,
    )


# Backward-compatible name aliases
gqa_factory_fn = bypass_factory_fn
moe_factory_fn = bypass_factory_fn
