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

import dataclasses
import fnmatch
import gc
import json
import os
import sys
import types
import warnings
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import transformers
from transformers import HfArgumentParser, PreTrainedModel, Trainer, TrainerCallback
from transformers import modeling_utils as tf_modeling_utils

from modelopt.torch.utils import print_rank_0, report_memory

from ..conversion import ModeloptStateManager, load_modelopt_state
from .huggingface import (
    _get_modelopt_state_path,
    _new_save_pretrained,
    _patch_model_init_for_modelopt,
    enable_huggingface_checkpointing,
    register_for_patching,
)

IGNORE_INDEX = nn.CrossEntropyLoss().ignore_index

__all__ = [
    "ModelOptArgParser",
    "ModelOptHFArguments",
    "ModelOptHFTrainer",
    "ModelOptTrainerArguments",
]


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


def _restore_qtensor_wrappers(model, model_path):
    """Re-wrap QTensorWrapper weights that were replaced during HF weight loading.

    Transformers>=5.0 uses ``setattr`` to load weights, which replaces ``QTensorWrapper``
    objects with plain ``Parameter`` tensors.  The compressed data is loaded correctly but
    the wrapper metadata (original shape, dtype, qtensor class) is lost.  This function
    reads the saved ``q_tensor_state`` from ``modelopt_state.pth`` and re-wraps the affected
    weights.
    """
    modelopt_state_path = _get_modelopt_state_path(model_path)
    if not os.path.isfile(modelopt_state_path):
        return

    from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear
    from modelopt.torch.quantization.qtensor import QTensorWrapper

    state = load_modelopt_state(modelopt_state_path)
    for _, mode_config in state["modelopt_state_dict"]:
        q_tensor_state = mode_config.get("metadata", {}).get("q_tensor_state", {})
        if not q_tensor_state:
            continue
        for name, module in model.named_modules():
            if (
                isinstance(module, RealQuantLinear)
                and name in q_tensor_state
                and not isinstance(module.weight, QTensorWrapper)
            ):
                module._parameters["weight"] = QTensorWrapper(
                    qtensor=module.weight.data,
                    metadata=q_tensor_state[name]["metadata"],
                )


def _new_from_pretrained(cls, /, pretrained_model_name_or_path, *args, **kwargs):
    """Patch for `cls.from_pretrained` method to restore ModelOpt state."""
    with _patch_model_init_for_modelopt(
        cls, pretrained_model_name_or_path, extra_context=_undo_torch_init_override_by_transformers
    ):
        model = types.MethodType(cls._modelopt_cache["from_pretrained"].__func__, cls)(
            pretrained_model_name_or_path, *args, **kwargs
        )

    _restore_qtensor_wrappers(model, pretrained_model_name_or_path)

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
def _load_params_and_buffers_into_zero3_model(model_to_load, state_dict, load_config=None):
    buffer_names = [name for name, _ in model_to_load.named_buffers()]
    buffer_state_dict = {k: v for k, v in state_dict.items() if k in buffer_names}
    model_to_load.load_state_dict(buffer_state_dict, strict=False)
    return tf_modeling_utils._modelopt_cache["_load_state_dict_into_zero3_model"](
        model_to_load, state_dict, load_config
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


@dataclasses.dataclass
class ModelOptHFArguments:
    """Base for all ModelOpt argument dataclasses used with :class:`ModelOptArgParser`.

    Subclasses are automatically treated as dataclasses (no ``@dataclass`` decorator needed).
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(cls)


class ModelOptTrainerArguments(ModelOptHFArguments):
    """Arguments for ModelOptHFTrainer controlling param freezing, LR config, and save dtype.

    This class can be used with HuggingFace's ``HfArgumentParser`` for CLI parsing.
    """

    trainable_params: list[str] | None = dataclasses.field(
        default=None,
        metadata={
            "nargs": "+",
            "help": (
                "Glob patterns (fnmatch) for parameters that should be trainable. "
                "All other parameters will be frozen. Mutually exclusive with frozen_params."
            ),
        },
    )
    frozen_params: list[str] | None = dataclasses.field(
        default=None,
        metadata={
            "nargs": "+",
            "help": (
                "Glob patterns (fnmatch) for parameters that should be frozen. "
                "Mutually exclusive with trainable_params."
            ),
        },
    )
    lr_config: str | None = dataclasses.field(
        default=None,
        metadata={
            "help": (
                "Path to a YAML file mapping fnmatch patterns to optimizer kwargs "
                "(e.g. lr, weight_decay). First matching pattern wins per parameter. "
                "See examples/llm_qat/configs/train/lr_config_example.yaml."
            ),
        },
    )
    save_dtype: str | None = dataclasses.field(
        default="bfloat16",
        metadata={
            "help": (
                "Dtype string to write into the saved model's config.json "
                "(e.g. 'bfloat16', 'float16'). Set to None to preserve the original dtype."
            ),
        },
    )
    manual_gc: bool = dataclasses.field(
        default=False,
        metadata={
            "help": (
                "Run `gc.collect()` before each training/prediction step to work around "
                "GPU memory leaks during QAT/distillation."
            ),
        },
    )
    liger_ce_label_smoothing: float = dataclasses.field(
        default=0.0,
        metadata={
            "help": (
                "Label smoothing for Liger fused CE loss. "
                "Only used when --use_liger_kernel is enabled."
            ),
        },
    )


class ModelOptArgParser(HfArgumentParser):
    """HfArgumentParser with ``--config`` YAML support and ``--generate_docs`` for ARGUMENTS.md."""

    def parse_args_into_dataclasses(self, args=None, **kwargs):
        """Parse args with optional YAML config defaults and doc generation."""
        if args is None:
            args = list(sys.argv[1:])

        # --generate_docs [output_path]: generate markdown and exit
        if "--generate_docs" in args:
            idx = args.index("--generate_docs")
            output = (
                args[idx + 1]
                if idx + 1 < len(args) and not args[idx + 1].startswith("--")
                else "ARGUMENTS.md"
            )
            self._generate_docs(output)
            sys.exit(0)

        # --config <yaml_file>: load YAML as defaults, CLI args override
        if "--config" in args:
            idx = args.index("--config")
            if idx + 1 >= len(args):
                raise ValueError("--config requires a path argument")
            config_path = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]  # strip --config <path> from argv
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)
            if config:
                known_by_parser = {a.dest for a in self._actions}
                all_modelopt_fields = self._all_modelopt_fields()
                applicable = {}
                for k, v in config.items():
                    if k in known_by_parser:
                        applicable[k] = v
                    elif k not in all_modelopt_fields:
                        raise ValueError(
                            f"Unknown config key '{k}' in {config_path}. "
                            f"Not recognized by any ModelOptHFArguments subclass."
                        )
                self.set_defaults(**applicable)

        return super().parse_args_into_dataclasses(args=args, **kwargs)

    @staticmethod
    def _all_modelopt_fields():
        """Collect all field names from every ModelOptHFArguments subclass."""
        fields = set()
        queue = list(ModelOptHFArguments.__subclasses__())
        while queue:
            cls = queue.pop()
            if dataclasses.is_dataclass(cls):
                fields.update(f.name for f in dataclasses.fields(cls))
            queue.extend(cls.__subclasses__())
        return fields

    def _generate_docs(self, output_path: str) -> None:
        """Generate a markdown argument reference from registered dataclass types."""
        lines = ["# Argument Reference", "", "_Auto-generated — do not edit by hand._", ""]

        # Sort: modelopt library classes first, then example-specific
        def _sort_key(dc):
            mod = dc.__module__ or ""
            return (0 if mod.startswith("modelopt.") else 1, mod, dc.__name__)

        sorted_types = sorted(self.dataclass_types, key=_sort_key)

        # Fields belonging to HF TrainingArguments (used to detect "own" fields)
        hf_training_fields: set[str] = set()
        if hasattr(transformers, "TrainingArguments"):
            hf_training_fields = {
                f.name for f in dataclasses.fields(transformers.TrainingArguments)
            }

        for dc in sorted_types:
            group_name = dc.__name__
            lines.append(f"## {group_name}")
            lines.append("")

            is_hf_subclass = (
                hasattr(transformers, "TrainingArguments")
                and issubclass(dc, transformers.TrainingArguments)
                and dc is not transformers.TrainingArguments
            )

            if is_hf_subclass:
                own_fields = [f for f in dataclasses.fields(dc) if f.name not in hf_training_fields]
                lines.append(
                    "Extends [HuggingFace TrainingArguments]"
                    "(https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)."
                    " Only additional arguments are shown below."
                )
                lines.append("")
            else:
                own_fields = list(dataclasses.fields(dc))

            if not own_fields:
                lines.append("_No additional arguments._")
                lines.append("")
                continue

            lines.append("| Argument | Type | Default | Description |")
            lines.append("|----------|------|---------|-------------|")

            for f in own_fields:
                name = f"`--{f.name}`"
                type_str = self._format_type(f.type)
                default_str = self._format_default(f.default, f.default_factory)
                help_text = dict(f.metadata).get("help", "")
                # Collapse multi-line help into single line
                help_text = " ".join(help_text.split())
                lines.append(f"| {name} | {type_str} | {default_str} | {help_text} |")

            lines.append("")

        # Remove trailing blank lines so markdownlint won't modify the file
        while lines and lines[-1] == "":
            lines.pop()
        Path(output_path).write_text("\n".join(lines) + "\n")
        print(f"Generated {output_path}")

    @staticmethod
    def _format_type(type_hint) -> str:
        """Format a type hint for display in markdown."""
        s = str(type_hint)
        # Clean up common type representations
        for old, new in [
            ("typing.", ""),
            ("typing_extensions.", ""),
            ("<class '", ""),
            ("'>", ""),
        ]:
            s = s.replace(old, new)
        return f"`{s}`"

    @staticmethod
    def _format_default(default, default_factory) -> str:
        """Format a default value for display in markdown."""
        if default is not dataclasses.MISSING:
            if default is None:
                return "`None`"
            if isinstance(default, str):
                return f'`"{default}"`'
            return f"`{default}`"
        if default_factory is not dataclasses.MISSING:
            return "_factory_"
        return "_required_"


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


def _forward_redirect(module, fn):
    """Run ``fn`` inside ``module``'s forward to trigger distributed param gathering.

    Works for both FSDP2 (unshards DTensor params) and DeepSpeed ZeRO-3
    (gathers partitioned params via per-module forward hooks).
    """
    original_forward = module.forward

    def wrapped_forward(*a, **kw):
        module.forward = original_forward
        return fn()

    module.forward = wrapped_forward
    try:
        dummy = torch.empty(1, device=next(module.parameters()).device)
        return module(dummy)
    except Exception:
        module.forward = original_forward
        raise


class ModelOptHFTrainer(Trainer):
    """A drop-in replacement of HuggingFace's Trainer for ModelOpt.

    This class adds extra utilities for ModelOpt checkpointing, memory reporting,
    parameter freezing, per-layer learning rates, Liger fused loss, and save dtype.

    **Liger kernel support:** When ``--use_liger_kernel`` is set, this trainer provides
    model-agnostic fused loss computation that extends HuggingFace's built-in Liger
    integration in three ways:

    1. **Model-agnostic**: Works with any causal LM that has an ``lm_head``, unlike
       HF's Liger which only supports `a fixed set of model architectures
       <https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py>`_.
    2. **DeepSpeed ZeRO-3 support**: HF's Liger integration only works with FSDP.
       ModelOpt adds distributed param gathering for DeepSpeed ZeRO-3 and DDP as well.
    3. **KD loss support**: ``KDTrainer`` extends fused loss to knowledge distillation
       via ``LigerFusedLinearJSD`` for fused lm_head + Jensen-Shannon divergence.
    """

    def __init__(
        self,
        *args,
        trainer_args: ModelOptTrainerArguments | None = None,
        lr_config: dict[str, dict[str, Any]] | None = None,
        **kwargs,
    ):
        """Initialize.

        Args:
            trainer_args: Optional arguments for param freeze, lr config, and save dtype.
            lr_config: Optional dict for per-pattern optimizer param groups
                (overrides trainer_args.lr_config).
        """
        enable_huggingface_checkpointing()
        super().__init__(*args, **kwargs)
        if trainer_args is None and isinstance(self.args, ModelOptTrainerArguments):
            trainer_args = self.args
        self.trainer_args = trainer_args or ModelOptTrainerArguments()
        self._lr_config = self._resolve_lr_config(lr_config, self.trainer_args)
        self._apply_gradient_checkpointing_defaults()
        self.add_callback(_MemoryReportCallback())
        self.use_liger_kernel = getattr(self.args, "use_liger_kernel", False)
        if self.use_liger_kernel:
            if self.is_fsdp_enabled and not self.accelerator.is_fsdp2:
                raise ValueError("Liger fused loss is not supported with FSDP1. Use FSDP2 instead.")
            self._setup_liger_fused_loss()
        self._configure_trainable_params()

    def _prepare_model(self, model):
        """Prepare a model via accelerator (materializes meta-device params, applies sharding).

        Uses a dummy optimizer because ``accelerator.prepare`` requires one for FSDP2.
        Works generically for FSDP2, DDP, and DeepSpeed backends. For fully-frozen models
        under DS ZeRO-3, falls back to inference-mode prep since ZeRO-3 asserts on empty
        trainable_param_groups; in that case the caller is responsible for gathering
        ``zero.Init``-partitioned params around forward passes.
        """
        if self.is_deepspeed_enabled and not any(p.requires_grad for p in model.parameters()):
            return self.accelerator.prepare_model(model, evaluation_mode=True)
        dummy_optimizer = torch.optim.SGD([next(model.parameters())], lr=0.0)
        model, _ = self.accelerator.prepare(model, dummy_optimizer)
        return model

    def training_step(self, *args, **kwargs):
        """Run gc.collect() before the training step if manual_gc is enabled."""
        if self.trainer_args.manual_gc:
            gc.collect()
        return super().training_step(*args, **kwargs)

    def prediction_step(self, *args, **kwargs):
        """Run gc.collect() before the prediction step if manual_gc is enabled."""
        if self.trainer_args.manual_gc:
            gc.collect()
        return super().prediction_step(*args, **kwargs)

    def _load_best_model(self, *args, **kwargs):
        """Run gc.collect() before loading the best model if manual_gc is enabled."""
        if self.trainer_args.manual_gc:
            gc.collect()
        return super()._load_best_model(*args, **kwargs)

    def _apply_gradient_checkpointing_defaults(self):
        """Ensure non-reentrant gradient checkpointing when no explicit kwargs are set."""
        args = self.args
        if not getattr(args, "gradient_checkpointing", False):
            return
        if args.gradient_checkpointing_kwargs is None:
            args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
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

    @staticmethod
    def _resolve_lr_config(
        lr_config: dict[str, dict[str, Any]] | None,
        trainer_args: ModelOptTrainerArguments,
    ) -> dict[str, dict[str, Any]] | None:
        if lr_config is not None:
            return lr_config
        path = getattr(trainer_args, "lr_config", None)
        if path is not None:
            return ModelOptHFTrainer.load_lr_config(path)
        return None

    @staticmethod
    def load_lr_config(path: str) -> dict[str, dict[str, Any]]:
        """Load an lr_config YAML file mapping fnmatch patterns to optimizer kwargs.

        Example YAML::

            "*lm_head*":
              lr: 1e-5
            "*mlp*":
              lr: 5e-5

        Returns:
            Ordered dict of ``{pattern: {kwarg: value, ...}}``.
        """
        import yaml

        with open(path) as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"lr_config must be a YAML mapping, got {type(cfg).__name__}")
        for pattern, kwargs in cfg.items():
            if not isinstance(pattern, str) or not isinstance(kwargs, dict):
                raise ValueError(
                    f"lr_config entry must be str -> dict, got {pattern!r} -> {kwargs!r}"
                )
            for key, val in kwargs.items():
                if isinstance(val, str):
                    with suppress(ValueError):
                        kwargs[key] = float(val)
        return cfg

    def _match_lr_config_pattern(self, name: str) -> str | None:
        """Return the first lr_config pattern matching ``name``, or None."""
        for pattern in self._lr_config:  # type: ignore[union-attr]
            if fnmatch.fnmatch(name, pattern):
                return pattern
        return None

    def create_optimizer(self):
        """Build per-pattern param groups from lr_config, then delegate to HF Trainer."""
        if self._lr_config is None:
            return super().create_optimizer()

        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model

        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
        else:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )

        decay_parameters = self.get_decay_parameter_names(opt_model)
        groups: dict[tuple[str | None, bool], list] = {}
        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            pattern = self._match_lr_config_pattern(name)
            is_decay = name in decay_parameters
            groups.setdefault((pattern, is_decay), []).append(param)

        param_groups = []
        for (pattern, is_decay), params in groups.items():
            group: dict[str, Any] = {
                "params": params,
                "weight_decay": self.args.weight_decay if is_decay else 0.0,
            }
            if pattern is not None:
                group.update(self._lr_config[pattern])
            param_groups.append(group)

        optimizer_kwargs["params"] = param_groups
        self.optimizer_cls_and_kwargs = (optimizer_cls, optimizer_kwargs)

        result = super().create_optimizer()

        self._log_lr_config_summary()
        return result

    def _log_lr_config_summary(self):
        if self.optimizer is None:
            return
        lines = ["lr_config optimizer param groups:"]
        for i, group in enumerate(self.optimizer.param_groups):
            lr = group.get("lr", "default")
            wd = group.get("weight_decay", "default")
            n_params = len(group["params"])
            lines.append(f"  group {i}: {n_params} params, lr={lr}, weight_decay={wd}")
        print_rank_0("\n".join(lines))

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

    def _sharded_liger_compute(self, fn):
        """Route fn through sharded DP to ensure lm_head params are gathered. No-op for DDP."""
        if self.is_fsdp_enabled:
            return _forward_redirect(self.model, fn)
        if self.is_deepspeed_enabled:
            lm_head = self._get_lm_head(self.accelerator.unwrap_model(self.model))
            return _forward_redirect(lm_head, fn)
        return fn()

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
        lm_head = self._get_lm_head(model)
        hidden_states = outputs.logits.to(lm_head.weight.dtype)  # RMSNorm may upcast to fp32

        def _compute():
            return LigerForCausalLMLoss(
                hidden_states=hidden_states,
                lm_head_weight=lm_head.weight,
                labels=labels,
                hidden_size=hidden_states.size(-1),
                num_items_in_batch=num_items_in_batch,
                ignore_index=IGNORE_INDEX,
                label_smoothing=self.trainer_args.liger_ce_label_smoothing,
            )

        return self._sharded_liger_compute(_compute)

    def save_model(self, *args, **kwargs):
        """Save the model and rewrite config.json dtype to ``save_dtype``."""
        outputs = super().save_model(*args, **kwargs)
        if (not self.is_in_train) and self.args.should_save:
            out_dir = args[0] if args else self.args.output_dir
            save_dtype = getattr(self.trainer_args, "save_dtype", None)
            self._update_config_json_dtype(out_dir, save_dtype)
        return outputs

    def _update_config_json_dtype(self, output_dir: str, dtype_str: str | None) -> None:
        """Rewrite <output_dir>/config.json 'dtype' (preferred) or 'torch_dtype' to dtype_str."""
        if dtype_str is None:
            return
        cfg_path = os.path.join(output_dir, "config.json")
        if not os.path.isfile(cfg_path):
            print_rank_0(f"[warn] config.json not found under {output_dir}; skip dtype rewrite.")
            return
        try:
            with open(cfg_path, encoding="utf-8") as f:
                data = json.load(f)
            key_to_update = (
                "dtype" if "dtype" in data else ("torch_dtype" if "torch_dtype" in data else None)
            )
            if key_to_update is None:
                print_rank_0(
                    "[warn] Neither 'dtype' nor 'torch_dtype' present in config.json; "
                    "skip dtype rewrite."
                )
                return
            if data.get(key_to_update) != dtype_str:
                data[key_to_update] = dtype_str
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print_rank_0(f'Updated config.json: {key_to_update} -> "{dtype_str}"')
        except Exception as e:
            print_rank_0(f"[warn] Failed to update dtype in config.json: {e}")
