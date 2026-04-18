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

import functools
import json
import os
import threading
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn.parameter import Parameter

from modelopt.torch.utils import print_rank_0

from ..conversion import ModeloptStateManager, modelopt_state, restore_from_modelopt_state
from ..dynamic import unwrap_shadowing_slots_before_register_parameter

__all__ = ["enable_huggingface_checkpointing"]


_MODELOPT_STATE_SAVE_NAME = "modelopt_state.pth"


def _quant_linear_ckpt_weight_must_reload(ckpt_dtype: torch.dtype, param_dtype: torch.dtype) -> bool:
    """Whether a safetensors tensor should replace the in-memory Parameter (HF dtype upcast)."""
    if ckpt_dtype in (torch.uint8, torch.int8) and param_dtype not in (torch.uint8, torch.int8):
        return True
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is not None and ckpt_dtype == fp8 and param_dtype != fp8:
        return True
    return False


def reload_quant_linear_weights_from_safetensors_checkpoint(
    model: torch.nn.Module,
    pretrained_model_name_or_path: str | os.PathLike[str],
) -> int:
    """Re-load :class:`~modelopt.torch.quantization.nn.QuantLinear` ``weight`` / ``weight0`` tensors from disk.

    ``PreTrainedModel.from_pretrained(..., dtype=torch.bfloat16)`` (and ``torch_dtype=``) often
    upcasts ``uint8`` / ``int8`` / ``float8_e4m3fn`` checkpoint weights to a floating dtype. That
    destroys packed / FP8 bit patterns even when ``weight_scale`` loads correctly, which yields
    plausible-looking runs with garbage logits.

    This pass reads matching keys from local ``*.safetensors`` shards (using
    ``model.safetensors.index.json`` when present) and replaces affected Parameters in-place.
    Remote Hub IDs (non-directory) are skipped.
    """
    root = Path(pretrained_model_name_or_path).expanduser()
    if not root.is_dir():
        return 0

    try:
        from safetensors.torch import load_file
    except ImportError:
        return 0

    try:
        from modelopt.torch.quantization.utils import is_quantized_linear
    except ImportError:
        return 0

    index = root / "model.safetensors.index.json"
    weight_map: dict[str, str]
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
    else:
        shards = sorted(root.glob("*.safetensors"))
        if not shards:
            return 0
        weight_map = {}
        try:
            from safetensors import safe_open

            for sp in shards:
                with safe_open(str(sp), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        weight_map[k] = sp.name
        except Exception:
            preferred = [p for p in shards if p.name == "model.safetensors"]
            pick = preferred[0] if preferred else max(shards, key=lambda p: p.stat().st_size)
            weight_map = {k: pick.name for k in load_file(str(pick), device="cpu").keys()}

    if not weight_map:
        return 0

    shard_cache: dict[str, dict[str, torch.Tensor]] = {}

    def _get_tensor(key: str, shard_name: str) -> torch.Tensor | None:
        spath = root / shard_name
        if not spath.is_file():
            return None
        if shard_name not in shard_cache:
            shard_cache[shard_name] = load_file(str(spath), device="cpu")
        return shard_cache[shard_name].get(key)

    n = 0
    for key, shard_name in weight_map.items():
        if key.endswith(".weight"):
            param_name = "weight"
            mod_path = key[: -len(".weight")]
        elif key.endswith(".weight0"):
            param_name = "weight0"
            mod_path = key[: -len(".weight0")]
        else:
            continue
        try:
            mod = model.get_submodule(mod_path)
        except AttributeError:
            continue
        if not is_quantized_linear(mod):
            continue
        wparam = mod._parameters.get(param_name)
        if wparam is None:
            continue
        t = _get_tensor(key, shard_name)
        if t is None or t.shape != wparam.shape:
            continue
        if not _quant_linear_ckpt_weight_must_reload(t.dtype, wparam.dtype):
            continue
        device = wparam.device
        mod._parameters[param_name] = torch.nn.Parameter(
            t.to(device=device, dtype=t.dtype, non_blocking=device.type == "cuda"),
            requires_grad=False,
        )
        n += 1
    return n
_LIBRARY_CLASSES_FOR_PATCHING: dict[str, tuple[list[type], list[list[tuple[str, Any]]]]] = {}
_PATCHED_CLASSES = set()


def register_for_patching(name: str, cls: type, patch_methods: list[tuple[str, Any]]):
    """Register a HuggingFace class for patching with ModelOpt functionality.

    This function registers a class from a HuggingFace library to be patched with ModelOpt's
    save/restore functionality. This allows ModelOpt state to be automatically preserved
    during saving and loading of models created from this class.

    Args:
        name: The name of the HuggingFace library to patch (e.g., 'transformers', 'diffusers').
        cls: The class within the library to patch (e.g., PreTrainedModel).
        patch_methods: List of tuples containing method names and their patch methods.
    """
    if name not in _LIBRARY_CLASSES_FOR_PATCHING:
        _LIBRARY_CLASSES_FOR_PATCHING[name] = ([], [])

    classes, methods_list = _LIBRARY_CLASSES_FOR_PATCHING[name]
    classes.append(cls)
    methods_list.append(patch_methods)


def _get_modelopt_state_path(model_name_or_path: str) -> str:
    return os.path.join(model_name_or_path, _MODELOPT_STATE_SAVE_NAME)


@contextmanager
def _patch_model_init_for_modelopt(
    cls, model_path, extra_context=None, *, restore_after_init: bool = True
):
    """Patch for `cls.init` method to restore ModelOpt state after ``__init__``.

    Args:
        restore_after_init: If False, run the original ``__init__`` only (no ``modelopt_state.pth``
            restore). Used for ``_from_config`` builds of **nested** submodules (e.g. VL language
            tower): restoring the full checkpoint state onto a subgraph raises unmatched quantizer
            keys. The outer ``from_pretrained`` patch still restores once on the root model.
    """
    # Note: Keeping original config in local as the package will be shared among threads
    added_original_init = False
    if hasattr(cls, "original_init"):
        _original__init__ = cls.original_init
    else:
        _original__init__ = cls.__init__
        cls.original_init = _original__init__
        # Avoid patching the init method twice, which can happen if one model is wrapped in another
        # e.g. in the case of distillation
        added_original_init = True

    @functools.wraps(_original__init__)
    def new_init_fn(self, *args, **kwargs):
        modelopt_state_path = _get_modelopt_state_path(model_path)
        _original__init__(self, *args, **kwargs)
        if restore_after_init and os.path.isfile(modelopt_state_path):
            with extra_context() if extra_context else nullcontext():
                restore_from_modelopt_state(self, modelopt_state_path=modelopt_state_path)

            print_rank_0(f"Restored ModelOpt state from {modelopt_state_path}")

            # Ensure export scale buffers exist before HF loads safetensors. Some ``from_pretrained``
            # paths never call ``modeling_utils._load_state_dict_into_meta_model`` (our other hook);
            # without placeholders, ``weight_scale`` / ``input_scale`` shards stay unused and FP8
            # reload uses wrong scales.
            try:
                from modelopt.torch.quantization.nn.modules.quant_module import (
                    register_hf_fp8_export_scale_buffer_placeholders,
                    register_hf_nvfp4_export_scale_buffer_placeholders,
                )

                register_hf_nvfp4_export_scale_buffer_placeholders(self)
                register_hf_fp8_export_scale_buffer_placeholders(self)
            except Exception as e:
                warnings.warn(
                    f"ModelOpt HF scale-buffer registration after restore failed ({e!r}); "
                    "quantized reload may leave safetensors scales unused.",
                    stacklevel=1,
                )

    cls.__init__ = new_init_fn
    try:
        yield
    finally:
        if added_original_init:
            delattr(cls, "original_init")
        cls.__init__ = _original__init__


def _new_save_pretrained(self, save_directory, *args, **kwargs):
    """Patch for `cls.save_pretrained` method to save ModelOpt state."""
    save_modelopt_state = kwargs.pop("save_modelopt_state", True)
    if save_modelopt_state:
        from modelopt.torch.opt.conversion import collapse_modelopt_state_to_hf_root

        # One holder before HF weight write; VLMs / fused exports can leave multiple otherwise.
        collapse_modelopt_state_to_hf_root(self)
    outputs = self._modelopt_cache["save_pretrained"](self, save_directory, *args, **kwargs)
    if save_modelopt_state:
        from modelopt.torch.opt.conversion import (
            _module_has_instance_modelopt_state,
            collapse_modelopt_state_to_hf_root,
            modelopt_state as modelopt_state_fn,
        )

        key = ModeloptStateManager._state_key
        n_holders = 0
        for _ in range(16):
            collapse_modelopt_state_to_hf_root(self)
            n_holders = sum(
                1 for m in self.modules() if _module_has_instance_modelopt_state(m, key)
            )
            if n_holders <= 1:
                break
        if n_holders == 1:
            path = _get_modelopt_state_path(save_directory)
            torch.save(modelopt_state_fn(self), path)
            print_rank_0(f"Saved ModelOpt state to {path}")
        elif n_holders > 1:
            warnings.warn(
                f"Could not reduce ModelOpt state to a single holder (found {n_holders}); "
                "skipping modelopt_state.pth. Fake-quant reload may require re-export or manual collapse.",
                stacklevel=1,
            )

    return outputs


_patch_lock = threading.Lock()

_MODELOPT_MODULE_REGISTER_PARAMETER_PATCHED = False
_MODELOPT_ORIG_MODULE_REGISTER_PARAMETER = None


def _modelopt_register_parameter_with_shadow_unwrap(self, name: str, param):
    """Delegate to PyTorch after clearing stale ``__dict__`` / buffer / submodule name clashes."""
    global _MODELOPT_ORIG_MODULE_REGISTER_PARAMETER
    if param is not None and isinstance(param, Parameter):
        unwrap_shadowing_slots_before_register_parameter(self, name)
    assert _MODELOPT_ORIG_MODULE_REGISTER_PARAMETER is not None
    return _MODELOPT_ORIG_MODULE_REGISTER_PARAMETER(self, name, param)


def _patch_torch_module_register_parameter_for_hf_load() -> None:
    """Avoid ``KeyError: attribute 'bias' already exists`` on meta / HF weight assignment.

    Plain :class:`torch.nn.Module` subclasses (not :class:`~modelopt.torch.opt.dynamic.DynamicModule`)
    can still hit PyTorch's ``register_parameter`` guard; unwrapping at registration time covers them.
    """
    global _MODELOPT_MODULE_REGISTER_PARAMETER_PATCHED, _MODELOPT_ORIG_MODULE_REGISTER_PARAMETER
    if _MODELOPT_MODULE_REGISTER_PARAMETER_PATCHED:
        return
    _MODELOPT_ORIG_MODULE_REGISTER_PARAMETER = torch.nn.Module.register_parameter
    torch.nn.Module.register_parameter = _modelopt_register_parameter_with_shadow_unwrap
    _MODELOPT_MODULE_REGISTER_PARAMETER_PATCHED = True


def patch_pretrained_methods(cls: type, patch_methods: list[tuple[str, Any]]):
    """Patch the pretrained methods of a library."""
    with _patch_lock:
        # in case multiple threads patch the same library
        if hasattr(cls, "_modelopt_cache"):
            return
        cls._modelopt_cache = {}  # type: ignore[attr-defined]
        for method_name, patch_method in patch_methods:
            if not hasattr(cls, method_name):
                warnings.warn(f"Method {method_name} not found in {cls.__name__}")
                continue
            cls._modelopt_cache[method_name] = getattr(cls, method_name)  # type: ignore[attr-defined]
            setattr(cls, method_name, patch_method)


def enable_huggingface_checkpointing():
    """Enables automatic save/restore of ModelOpt state with HuggingFace checkpointing APIs.

    ModelOpt automatically saves `modelopt_state` to `save_directory/modelopt_state.pth` when
    a Huggingface model is saved using
    `model.save_pretrained(save_directory) <https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.save_pretrained>`_.

    Conversely, ModelOpt restores the saved state from `pretrained_model_name_or_path/modelopt_state.pth` if it exists
    when a Huggingface model is loaded using
    `cls.from_pretrained(pretrained_model_name_or_path) <https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained>`_.


    This function should be called once in the program before loading/saving any HuggingFace models.

    Here is an example usage:

    .. code-block:: python

        from transformers import AutoModelForCausalLM
        import modelopt.torch.opt as mto

        # Enable ModelOpt save/restore for HuggingFace models
        # This only needs to be called once in the program.
        mto.enable_huggingface_checkpointing()

        # Instantiate a HuggingFace model, modelopt_state will be automatically loaded if it exists.
        model = AutoModelForCausalLM.from_pretrained(model_path).cuda()

    """
    _patch_torch_module_register_parameter_for_hf_load()
    for name, (classes, methods_list) in _LIBRARY_CLASSES_FOR_PATCHING.items():
        for cls, patch_methods in zip(classes, methods_list):
            if cls in _PATCHED_CLASSES:
                continue
            patch_pretrained_methods(cls, patch_methods)
            _PATCHED_CLASSES.add(cls)
        print_rank_0(f"ModelOpt save/restore enabled for `{name}` library.")
