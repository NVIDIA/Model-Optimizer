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
import os
import threading
import warnings
from contextlib import contextmanager, nullcontext
from typing import Any

import torch

from modelopt.torch.utils import print_rank_0

from ..conversion import ModeloptStateManager, modelopt_state, restore_from_modelopt_state

__all__ = ["enable_huggingface_checkpointing"]


_MODELOPT_STATE_SAVE_NAME = "modelopt_state.pth"
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
def _patch_model_init_for_modelopt(cls, model_path, extra_context=None):
    """Patch for `cls.init` method to restore ModelOpt state after `init`."""
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
        if os.path.isfile(modelopt_state_path):
            # Security NOTE: weights_only=False is used on ModelOpt-generated state_dict, not on untrusted user input
            modelopt_state = torch.load(modelopt_state_path, map_location="cpu", weights_only=False)
            with extra_context() if extra_context else nullcontext():
                restore_from_modelopt_state(self, modelopt_state)

            print_rank_0(f"Restored ModelOpt state from {modelopt_state_path}")

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
    outputs = self._modelopt_cache["save_pretrained"](self, save_directory, *args, **kwargs)
    if save_modelopt_state and ModeloptStateManager.is_converted(self):
        path = _get_modelopt_state_path(save_directory)
        torch.save(modelopt_state(self), path)
        print_rank_0(f"Saved ModelOpt state to {path}")

    return outputs


_patch_lock = threading.Lock()


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
    for name, (classes, methods_list) in _LIBRARY_CLASSES_FOR_PATCHING.items():
        for cls, patch_methods in zip(classes, methods_list):
            if cls in _PATCHED_CLASSES:
                continue
            patch_pretrained_methods(cls, patch_methods)
            _PATCHED_CLASSES.add(cls)
        print_rank_0(f"ModelOpt save/restore enabled for `{name}` library.")
