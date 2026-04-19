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
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import torch
import transformers
from transformers import HfArgumentParser, PreTrainedModel, Trainer, TrainerCallback
from transformers import modeling_utils as tf_modeling_utils

from modelopt.torch.utils import report_memory

from ..conversion import ModeloptStateManager, load_modelopt_state
from .huggingface import (
    _get_modelopt_state_path,
    _new_save_pretrained,
    _patch_model_init_for_modelopt,
    enable_huggingface_checkpointing,
    register_for_patching,
)

__all__ = ["ModelOptArgParser", "ModelOptHFArguments", "ModelOptHFTrainer"]


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


class ModelOptHFTrainer(Trainer):
    """A drop-in replacement of HuggingFace's Trainer for ModelOpt.

    This class adds extra utilities for ModelOpt checkpointing and memory reporting.
    """

    def __init__(self, *args, **kwargs):
        """Initialize."""
        enable_huggingface_checkpointing()
        super().__init__(*args, **kwargs)
        self.add_callback(_MemoryReportCallback())
