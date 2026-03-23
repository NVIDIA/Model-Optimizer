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

"""ModelOpt plugin for enabling automatic save/restore of ModelOpt state for HuggingFace models."""

import fnmatch
import types
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, Trainer, TrainerCallback
from transformers import modeling_utils as tf_modeling_utils

from modelopt.torch.utils import print_rank_0, report_memory

from ..conversion import ModeloptStateManager
from .huggingface import (
    _new_save_pretrained,
    _patch_model_init_for_modelopt,
    enable_huggingface_checkpointing,
    register_for_patching,
)

__all__ = ["ModelOptHFTrainer", "ModelOptTrainerArguments"]


@contextmanager
def _undo_torch_init_override_by_transformers():
    if not hasattr(tf_modeling_utils, "TORCH_INIT_FUNCTIONS"):
        yield
        return
    # transformers override weight initialization during model instantiation for faster loading;
    # this leads to a secondary bug causing fx symbolic tracing to fail (torch does not allow
    # overriding torch.nn.init functions - fx tracing asserts that this does not happen and fails)
    # correct fx symbolic tracing is needed for NAS/Pruned model restoration
    # lets restore the original init functions before modelopt restore so that tracing works during nas restore
    # weight initialization is anyways done, so this wont affect performance
    modelopt_reverted_torch_init_funcs = {}
    for name, init_func in tf_modeling_utils.TORCH_INIT_FUNCTIONS.items():
        torch_init_func = getattr(torch.nn.init, name)
        # Check if the init function has been overridden by transformers
        if id(torch_init_func) != id(init_func):
            modelopt_reverted_torch_init_funcs[name] = torch_init_func
            setattr(torch.nn.init, name, init_func)

    yield

    for name, init_func in modelopt_reverted_torch_init_funcs.items():
        setattr(torch.nn.init, name, init_func)


def _new_from_pretrained(cls, /, pretrained_model_name_or_path, *args, **kwargs):
    """Patch for `cls.from_pretrained` method to restore ModelOpt state."""
    with _patch_model_init_for_modelopt(
        cls, pretrained_model_name_or_path, extra_context=_undo_torch_init_override_by_transformers
    ):
        model = types.MethodType(cls._modelopt_cache["from_pretrained"].__func__, cls)(
            pretrained_model_name_or_path, *args, **kwargs
        )

    return model


def _new_from_config(cls, /, config, **kwargs):
    """Patch for `cls.from_config` method to restore ModelOpt state."""
    with _patch_model_init_for_modelopt(
        cls, config._name_or_path, extra_context=_undo_torch_init_override_by_transformers
    ):
        model = types.MethodType(cls._modelopt_cache["_from_config"].__func__, cls)(
            config, **kwargs
        )
    return model


def _save_pretrained_with_checks(self, save_directory, *args, **kwargs):
    if getattr(self, "_tp_size", None) is not None and ModeloptStateManager.is_converted(self):
        raise NotImplementedError(
            "ModelOpt does not support saving tensor parallel sharded Huggingface transformer models yet. "
        )
    return _new_save_pretrained(self, save_directory, *args, **kwargs)


# [Fix for huggingface bug] deepspeed zero3 training backend only loads params into the model from
# state_dict, but not buffers. So lets explicitly load the buffers into the model from state_dict.
def _load_params_and_buffers_into_zero3_model(model_to_load, state_dict):
    buffer_names = [name for name, _ in model_to_load.named_buffers()]
    buffer_state_dict = {k: v for k, v in state_dict.items() if k in buffer_names}
    model_to_load.load_state_dict(buffer_state_dict, strict=False)
    return tf_modeling_utils._modelopt_cache["_load_state_dict_into_zero3_model"](
        model_to_load, state_dict
    )


pretrained_model_patch_methods = [
    ("from_pretrained", classmethod(_new_from_pretrained)),
    # We need to patch _from_config of PreTrainedModel; from_config is a private method in _BaseAutoModelClass and
    # patching it is more complex
    ("_from_config", classmethod(_new_from_config)),
    ("save_pretrained", _save_pretrained_with_checks),
]

register_for_patching("transformers", PreTrainedModel, pretrained_model_patch_methods)
register_for_patching(
    "transformers",
    tf_modeling_utils,
    [("_load_state_dict_into_zero3_model", _load_params_and_buffers_into_zero3_model)],
)


def _report_memory(msg):
    if not torch.cuda.is_available():
        return
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        report_memory(msg + ":", device=torch.cuda.current_device())
    else:
        for device in range(torch.cuda.device_count()):
            report_memory(f"{msg}, device={device}:", device=device)


class _MemoryReportCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            _report_memory("Memory usage at training step 1")

    def on_evaluate(self, args, state, control, **kwargs):
        if state.global_step <= 1:
            _report_memory("Memory usage at evaluation")


@dataclass
class ModelOptTrainerArguments:
    """Arguments for ModelOptHFTrainer controlling param freezing and loss fusion.

    This class can be used with HuggingFace's ``HfArgumentParser`` for CLI parsing.
    """

    trainable_params: list[str] | None = field(
        default=None,
        metadata={
            "nargs": "+",
            "help": (
                "Glob patterns (fnmatch) for parameters that should be trainable. "
                "All other parameters will be frozen. Mutually exclusive with frozen_params."
            ),
        },
    )
    frozen_params: list[str] | None = field(
        default=None,
        metadata={
            "nargs": "+",
            "help": (
                "Glob patterns (fnmatch) for parameters that should be frozen. "
                "Mutually exclusive with trainable_params."
            ),
        },
    )


class ModelOptHFTrainer(Trainer):
    """A drop-in replacement of HuggingFace's Trainer for ModelOpt.

    This class adds extra utilities for ModelOpt checkpointing and memory reporting.
    """

    def __init__(self, *args, trainer_args: ModelOptTrainerArguments | None = None, **kwargs):
        """Initialize."""
        enable_huggingface_checkpointing()
        super().__init__(*args, **kwargs)
        self.trainer_args = trainer_args or ModelOptTrainerArguments()
        self._apply_gradient_checkpointing_defaults()
        self.add_callback(_MemoryReportCallback())
        self.use_liger_kernel = self.args.use_liger_kernel
        if self.use_liger_kernel:
            if self.is_fsdp_enabled and not self.accelerator.is_fsdp2:
                raise ValueError("Liger fused loss is not supported with FSDP1. Use FSDP2 instead.")
            self._setup_liger_fused_loss()
        self._configure_trainable_params()

    def _apply_gradient_checkpointing_defaults(self):
        """Ensure non-reentrant gradient checkpointing when no explicit kwargs are set."""
        args = self.args
        if not getattr(args, "gradient_checkpointing", False):
            return
        if args.gradient_checkpointing_kwargs is None:
            args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            # use_reentrant=True requires embedding weights to have requires_grad=True,
            # which conflicts when training is disabled for those layers.
            if args.gradient_checkpointing_kwargs.get("use_reentrant", False):
                warnings.warn(
                    "ModelOpt overriding `use_reentrant=True` to `use_reentrant=False` "
                    "for gradient checkpointing compatibility.",
                    UserWarning,
                    stacklevel=2,
                )
            args.gradient_checkpointing_kwargs["use_reentrant"] = False

    def _configure_trainable_params(self):
        """Freeze/unfreeze parameters based on trainer_args.trainable_params or frozen_params."""
        trainable = self.trainer_args.trainable_params
        frozen = self.trainer_args.frozen_params
        if not trainable and not frozen:
            return
        if trainable and frozen:
            raise ValueError("trainable_params and frozen_params are mutually exclusive.")

        def _matches(name, patterns):
            return any(fnmatch.fnmatch(name, p) for p in patterns)

        model = self.model
        if trainable:
            for name, param in model.named_parameters():
                param.requires_grad_(_matches(name, trainable))
        else:
            for name, param in model.named_parameters():
                if _matches(name, frozen):
                    param.requires_grad_(False)

        trainable_count = sum(p.requires_grad for p in model.parameters())
        total_count = sum(1 for _ in model.parameters())
        print_rank_0(
            f"Trainable params: {trainable_count}/{total_count} "
            f"({100 * trainable_count / max(total_count, 1):.1f}%)"
        )

    def _get_lm_head(self, model):
        """Resolve lm_head from model at call time (no cached pointer to FSDP-managed params)."""
        return model.lm_head

    def _setup_liger_fused_loss(self):
        """Set compute_loss_func for fused CE."""
        model = self.accelerator.unwrap_model(self.model)
        if not hasattr(model, "lm_head"):
            self.use_liger_kernel = False
            return
        self.compute_loss_func = self._liger_loss_func

    @contextmanager
    def _liger_identity_lm_head(self):
        """Temporarily patch lm_head to identity for fused loss computation."""
        model = self.accelerator.unwrap_model(self.model)
        lm_head = self._get_lm_head(model)
        original_forward = lm_head.forward
        lm_head.forward = lambda x: x
        try:
            yield
        finally:
            lm_head.forward = original_forward

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss, patching lm_head to identity when using liger fused loss."""
        if self.use_liger_kernel:
            with self._liger_identity_lm_head():
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def _liger_loss_func(self, outputs, labels, num_items_in_batch=None, **kwargs):
        """Fused lm_head + CE loss via liger kernel."""
        from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss

        model = self.accelerator.unwrap_model(self.model)
        hidden_states = outputs.logits
        lm_head = self._get_lm_head(model)

        def _compute():
            return LigerForCausalLMLoss(
                hidden_states=hidden_states,
                lm_head_weight=lm_head.weight,
                labels=labels,
                hidden_size=hidden_states.size(-1),
                num_items_in_batch=num_items_in_batch,
            )

        if self.is_fsdp_enabled:
            return _fsdp_forward_redirect(self.model, _compute)
        return _compute()


def _fsdp_forward_redirect(fsdp_module, fn):
    """Run ``fn`` inside ``fsdp_module``'s forward to unshard child module params.

    In FSDP2 child modules (e.g. lm_head) are not their own FSDPModule.
    To unshard their weights we redirect ``fn`` through the FSDP parent's forward.
    """
    original_forward = fsdp_module.forward

    def wrapped_forward(*a, **kw):
        fsdp_module.forward = original_forward
        return fn()

    fsdp_module.forward = wrapped_forward
    # Dummy arg required since FSDP2 forward expects at least one argument.
    return fsdp_module("_fsdp_redirect")
