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

import types
import warnings
from contextlib import contextmanager

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
    reload_quant_linear_weights_from_safetensors_checkpoint,
)

__all__ = ["ModelOptHFTrainer"]


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

    try:
        n = reload_quant_linear_weights_from_safetensors_checkpoint(
            model, pretrained_model_name_or_path
        )
        if n > 0:
            print_rank_0(
                f"Reloaded {n} QuantLinear weight tensor(s) from safetensors with original dtypes "
                "(avoids dtype=/torch_dtype= upcast corrupting uint8 / int8 / FP8 weights)."
            )
    except Exception as e:
        warnings.warn(
            f"ModelOpt safetensors QuantLinear weight reload failed ({e!r}); "
            "if generation is garbled, try loading without forcing dtype= on quantized checkpoints.",
            stacklevel=1,
        )

    try:
        from modelopt.torch.quantization.nn.modules.quant_module import (
            sync_hf_export_pre_quant_scale_buffers,
        )

        sync_hf_export_pre_quant_scale_buffers(model)
    except Exception as e:
        warnings.warn(
            f"ModelOpt pre_quant_scale sync after HF load failed ({e!r}); "
            "smoothquant exports may leave root buffers unused.",
            stacklevel=1,
        )

    return model


def _new_from_config(cls, /, config, **kwargs):
    """Patch for `cls._from_config`; skip ModelOpt restore on nested submodule builds."""
    with _patch_model_init_for_modelopt(
        cls,
        config._name_or_path,
        extra_context=_undo_torch_init_override_by_transformers,
        restore_after_init=False,
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


def _quant_linear_weight_requires_grad_false_before_hf_assign(model: torch.nn.Module) -> None:
    """Let INT8 / packed-integer safetensors load into :class:`QuantLinear`.

    Meta / float ``weight`` Parameters are often created with ``requires_grad=True``. Hugging Face
    then assigns ``int8`` checkpoint tensors with ``assign=True``; PyTorch forbids
    ``requires_grad=True`` on integer dtypes. Turn off grad on quantized linear weights before
    assignment (inference loads do not need it).
    """
    try:
        from modelopt.torch.quantization.utils import is_quantized_linear
    except ImportError:
        return

    for module in model.modules():
        if not is_quantized_linear(module):
            continue
        for name in ("weight", "weight0"):
            p = module._parameters.get(name)
            if p is None or not isinstance(p, torch.nn.Parameter):
                continue
            if not p.requires_grad:
                continue
            module._parameters[name] = torch.nn.Parameter(p.data, requires_grad=False)


def _load_state_dict_into_meta_model_with_nvfp4_pack(model, state_dict, *args, **kwargs):
    """Ensure NVFP4 packed ``weight`` shapes before HF assigns checkpoint tensors.

    Some HF paths reset or materialize logical ``(out, in)`` weights after ModelOpt
    ``restore_from_modelopt_state``; re-applying resize immediately before meta load fixes
    ``[out, in//2]`` safetensors vs ``[out, in]`` module mismatches.

    Register export scale buffers again here: after restore the weights are often already
    packed ``uint8``, so the registration in ``restore_quantizer_state`` may have skipped
    every layer, leaving ``weight_scale`` / ``input_scale`` keys unused in safetensors.
    The same applies to **FP8** exports: NVFP4-only placeholder registration would skip those
    linears unless we also register FP8 scale buffers before ``load_state_dict``.

    The meta ``state_dict`` for this shard is the authoritative source for scale **shapes**
    (mixed auto-quant, missing ``_amax``, DynamicModule ``weight`` access).

    Scale tensors may be omitted from that per-shard dict when Hugging Face filters keys via
    ``key_renaming_mapping``. We then register buffers from ``model.safetensors.index.json`` (once
    per checkpoint directory) so ``weight_scale`` / ``input_scale`` / ``pre_quant_scale`` still load.
    """
    from pathlib import Path

    from modelopt.torch.quantization.conversion import (
        _resize_int4_awq_packed_linear_weights_for_hf_load,
        _resize_nvfp4_packed_linear_weights_for_hf_load,
    )
    from modelopt.torch.quantization.nn.modules.quant_module import (
        register_hf_export_scale_buffers_from_checkpoint_dir_once,
        register_hf_export_scale_buffers_from_meta_state_dict,
        register_hf_fp8_export_scale_buffer_placeholders,
        register_hf_nvfp4_export_scale_buffer_placeholders,
    )

    shard_file = args[0] if len(args) > 0 else ""
    if isinstance(shard_file, str) and shard_file.endswith(".safetensors"):
        register_hf_export_scale_buffers_from_checkpoint_dir_once(model, Path(shard_file).parent)

    if isinstance(state_dict, dict):
        register_hf_export_scale_buffers_from_meta_state_dict(model, state_dict)
    register_hf_nvfp4_export_scale_buffer_placeholders(model)
    register_hf_fp8_export_scale_buffer_placeholders(model)
    _resize_nvfp4_packed_linear_weights_for_hf_load(model)
    _resize_int4_awq_packed_linear_weights_for_hf_load(model)
    _quant_linear_weight_requires_grad_false_before_hf_assign(model)
    return tf_modeling_utils._modelopt_cache["_load_state_dict_into_meta_model"](
        model, state_dict, *args, **kwargs
    )


def _get_parameter_or_buffer_with_quant_linear_fallback(self, target: str):
    """Compatibility with Hugging Face ``transformers`` 4.57+ :meth:`_initialize_missing_keys`.

    That path calls :meth:`get_parameter_or_buffer` for every ``state_dict`` key that was loaded.
    :meth:`torch.nn.Module.get_parameter` requires ``getattr(module, leaf)`` to be an
    ``nn.Parameter``. ModelOpt :class:`~modelopt.torch.quantization.nn.modules.quant_module.QuantLinearConvBase`
    registers ``weight`` as a DynamicModule *dynamic attribute* whose getter returns a
    **dequantized tensor** (NVFP4 / INT8 / FP8 packed checkpoints), so the standard HF path
    raises. When it does, resolve the leaf from ``_parameters`` / ``_buffers`` directly (same
    tensors ``state_dict`` names refer to).
    """
    try:
        return self._modelopt_cache["get_parameter_or_buffer"](self, target)
    except AttributeError:
        module_path, _, leaf = target.rpartition(".")
        try:
            mod = self.get_submodule(module_path) if module_path else self
        except AttributeError:
            raise
        if leaf in mod._parameters:
            p = mod._parameters[leaf]
            if p is not None:
                return p
        if leaf in mod._buffers:
            b = mod._buffers[leaf]
            if b is not None:
                return b
        try:
            from transformers.quantizers.quantizers_utils import get_module_from_name
        except ImportError:
            get_module_from_name = None  # type: ignore[assignment]
        if get_module_from_name is not None:
            module, param_name = get_module_from_name(self, target)
            if (
                param_name == "_extra_state"
                and getattr(module.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
                is not torch.nn.Module.get_extra_state
            ):
                return module.get_extra_state()
        raise AttributeError(f"`{target}` is neither a parameter, buffer, nor extra state.") from None


def _modelopt_init_weights(self, module):
    """Skip HF ``normal_`` init on packed integer weights (e.g. NVFP4 ``uint8``).

    ``_initialize_missing_keys`` calls ``initialize_weights``; the default Linear path uses
    ``weight.data.normal_``, which is invalid for packed quant dtypes.

    Also skip init on ModelOpt :class:`TensorQuantizer` submodules: their ``_amax`` (and related)
    buffers are **not** in HF safetensors (they live in ``modelopt_state.pth``). Hugging Face may
    still list them as "newly initialized" because they are absent from weight shards; we must not
    let generic init overwrite values restored from ModelOpt state.
    """
    try:
        from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
            SequentialQuantizer,
            TensorQuantizer,
        )

        if isinstance(module, (TensorQuantizer, SequentialQuantizer)):
            return
    except ImportError:
        pass

    w = getattr(module, "weight", None)
    if isinstance(w, torch.nn.Parameter):
        dt = w.dtype
        if not dt.is_floating_point and not dt.is_complex:
            b = getattr(module, "bias", None)
            if b is not None and isinstance(b, torch.nn.Parameter) and b.dtype.is_floating_point:
                b.data.zero_()
            return
    return PreTrainedModel._modelopt_cache["_init_weights"](self, module)


pretrained_model_patch_methods = [
    ("from_pretrained", classmethod(_new_from_pretrained)),
    # We need to patch _from_config of PreTrainedModel; from_config is a private method in _BaseAutoModelClass and
    # patching it is more complex
    ("_from_config", classmethod(_new_from_config)),
    ("save_pretrained", _save_pretrained_with_checks),
]
if hasattr(PreTrainedModel, "_init_weights"):
    pretrained_model_patch_methods.append(("_init_weights", _modelopt_init_weights))
if hasattr(PreTrainedModel, "get_parameter_or_buffer"):
    pretrained_model_patch_methods.append(
        ("get_parameter_or_buffer", _get_parameter_or_buffer_with_quant_linear_fallback)
    )

register_for_patching("transformers", PreTrainedModel, pretrained_model_patch_methods)
_modeling_utils_patch_methods = [("_load_state_dict_into_zero3_model", _load_params_and_buffers_into_zero3_model)]
if hasattr(tf_modeling_utils, "_load_state_dict_into_meta_model"):
    _modeling_utils_patch_methods.append(
        ("_load_state_dict_into_meta_model", _load_state_dict_into_meta_model_with_nvfp4_pack)
    )
register_for_patching("transformers", tf_modeling_utils, _modeling_utils_patch_methods)


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
