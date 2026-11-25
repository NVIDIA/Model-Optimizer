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
"""Patch manager for NAS."""

import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch.nn as nn

from modelopt.torch.opt.searcher import ForwardLoop

from .utils import is_modelopt_patches_enabled

PatchData = dict[str, Any]
PatchMethods = dict[str, Callable]


class PatchManager(ABC):
    """A standard interface to handle the monkey patching of a model.

    The class provides two main interfaces, ``patch`` and ``unpatch``, which can be used to properly
    overwrite methods and add new attributes to the model. They will all be removed upon calling
    ``unpatch()``.
    """

    _patch_data_key = "_modelopt_patch_data"
    _patch_cls_key = "_modelopt_patch_cls"

    def __init__(self, model: nn.Module) -> None:
        """Constructor."""
        self._model = model

        # find submodule with patch (if it exists)
        self._modelopt_name, self._modelopt_module = self._get_named_patched_module(model)

    @classmethod
    def _get_named_patched_module(cls, model: nn.Module) -> tuple[str | None, nn.Module | None]:
        """Return the name of the patched module and the patched module itself."""
        for name, module in model.named_modules():
            if hasattr(module, cls._patch_data_key):
                return name, module
        return None, None

    @classmethod
    def is_patched(cls, model: nn.Module) -> bool:
        """Return whether the model is patched."""
        return cls._get_named_patched_module(model)[1] is not None

    @abstractmethod
    def _get_default_patch_data(self) -> PatchData:
        """Return the default patch data of the model."""

    @classmethod
    def get_manager(cls, model: nn.Module) -> "PatchManager":
        """Return the patch manager of the model using the "correct" class."""
        _, modelopt_module = cls._get_named_patched_module(model)
        assert modelopt_module is not None, "Model is not patched!"
        return getattr(modelopt_module, cls._patch_cls_key)(model)

    @property
    def patch_data(self) -> PatchData:
        """Return the patch data of the model."""
        assert self._modelopt_module is not None, "Model is not patched!"
        return self.patch_data_or_empty

    @property
    def patch_data_or_empty(self) -> PatchData:
        """Return the patch data of the model or an empty dictionary."""
        if self._modelopt_module is None:
            return {}
        return getattr(self._modelopt_module, self._patch_data_key)

    def patch(self) -> None:
        """Patch model in-place to be compatible with subsequent Model Optimizer tasks."""
        # sanity check that there is no patch data in another submodule
        assert self._modelopt_name is None, (
            f"Model Optimizer patch detected in {self._modelopt_name}."
        )

        # initialize patch data
        setattr(self._model, self._patch_data_key, self._get_default_patch_data())
        setattr(self._model, self._patch_cls_key, type(self))

        # monkey-patch methods
        self._set_patched_methods()

        # correctly set _modelopt_name, _modelopt_module
        self._modelopt_name, self._modelopt_module = "", self._model

    def unpatch(self) -> None:
        """Remove and delete patching from the model.

        For example:

        .. code-block:: python

            PatchManager(model).unpatch()
            model.forward(x)  # no patched (auto-) operations will be executed anymore.
        """
        # sanity check that there is patch data
        assert self._modelopt_name is not None, (
            f"No Model Optimizer patch detected in {self._model}."
        )

        # remove patch-related artifacts
        delattr(self._modelopt_module, self._patch_data_key)
        delattr(self._modelopt_module, self._patch_cls_key)
        self._unset_patched_methods()
        self._modelopt_name, self._modelopt_module = None, None

    def reset_before_sample(self) -> None:
        """Call reset hook before sample-related operations (sample & select)."""
        self._hook_pre_sample()

    def call_post_eval(self, forward_loop: ForwardLoop | None = None) -> None:
        """Call post-eval hook explicitly.

        Args:
            forward_loop: A ``Callable`` that takes a model as input and runs a forward loop on it.
        """
        self._hook_post_eval(forward_loop)

    def _hook_pre_sample(self) -> None:
        """Optional hook to be called before sample-related operations (sample & select)."""

    def _hook_post_eval(self, forward_loop: ForwardLoop | None = None) -> None:
        """Optional hook that is called after eval() (or train(False)) to calibrate the model."""

    def _hook_pre_forward(self, *args, **kwargs) -> None:
        """Optional hook that is called after the original forward function is called."""

    def _hook_post__replicate_for_data_parallel(self) -> None:
        """Optional hook to be called after _replicate_for_data_parallel."""
        self._set_patched_methods()

    @staticmethod
    def hooked_forward(mod: nn.Module, *args, **kwargs):
        """The forward method with hooks."""
        if is_modelopt_patches_enabled():
            PatchManager.get_manager(mod)._hook_pre_forward(*args, **kwargs)
        return getattr(mod, "_modelopt_unhooked_forward")(
            mod, *args, **kwargs
        )  # call original forward

    @staticmethod
    def hooked_train(mod: nn.Module, mode: bool = True) -> nn.Module:
        """Sets the model into train or eval mode according to flag."""
        ret = getattr(mod, "_modelopt_unhooked_train")(mod, mode)  # call original train

        # call model calibration directly here if we go into eval mode
        # Note that train might have already been called from prep_for_eval in which case we
        # don't want to call model calibration again!!!
        with _modelopt_eval_recursion_guard(mod) as guard:
            if not guard and not mode and is_modelopt_patches_enabled():
                PatchManager.get_manager(mod)._hook_post_eval()

        return ret

    @staticmethod
    def hooked__replicate_for_data_parallel(mod: nn.Module) -> nn.Module:
        """The _replicate_for_data_parallel method with hooks."""
        mod = getattr(mod, "_modelopt_unhooked__replicate_for_data_parallel")(mod)
        if is_modelopt_patches_enabled():
            PatchManager.get_manager(mod)._hook_post__replicate_for_data_parallel()

    def _get_methods_for_patching(self) -> PatchMethods:
        """Returns a lookup for methods to be patched together with the getters for the patch."""
        return {
            "forward": self.hooked_forward,
            "train": self.hooked_train,
            "_replicate_for_data_parallel": self.hooked__replicate_for_data_parallel,
        }

    def _set_patched_methods(self):
        for name, func_hooked in self._get_methods_for_patching().items():
            func_unhooked = getattr(self._model, name).__func__
            if func_unhooked is func_hooked:
                continue
            setattr(self._model, f"_modelopt_unhooked_{name}", func_unhooked)
            setattr(self._model, name, types.MethodType(func_hooked, self._model))

    def _unset_patched_methods(self):
        for name, func_hooked in self._get_methods_for_patching().items():
            func_current = getattr(self._model, name).__func__
            if func_current is not func_hooked:
                continue
            func_unhooked = getattr(self._model, f"_modelopt_unhooked_{name}")
            delattr(self._model, f"_modelopt_unhooked_{name}")
            if func_unhooked is getattr(type(self._model), name):
                delattr(self._model, name)
            else:
                setattr(self._model, name, types.MethodType(func_unhooked, self._model))


@contextmanager
def _modelopt_eval_recursion_guard(model: nn.Module) -> Iterator[bool]:
    """Context manager to guard against infinite recursion when calling eval/prep_for_eval.

    Args:
        model: The model to guard.

    For example:

    .. code-block:: python

        with _modelopt_eval_recursion_guard(model):
            model.eval()
    """
    guard_key = "_modelopt_eval_recursion_guard"
    has_key = hasattr(model, guard_key)
    original_guard: bool = getattr(model, guard_key, False)
    setattr(model, guard_key, True)
    try:
        yield original_guard
    finally:
        # depending on whether key already existed we either delete it or reset it to original val
        if has_key:
            setattr(model, guard_key, original_guard)
        else:
            delattr(model, guard_key)


def prep_for_eval(model: nn.Module, forward_loop: ForwardLoop | None = None) -> nn.Module:
    """Calibrate model for evaluation and enable eval().

    Args:
        model: A nn.Module that might be dynamic.
        forward_loop: A ``Callable`` that takes a model as input and runs a pre-defined forward
            loop on it using real data.

    .. note::

        This function should only be explicitly called once after conversion when the model is
        immediately used for evaluation without prior training!
    """
    # regular eval mode with recursion guard
    # this is to ensure that model calibration won't be called again from within eval()
    with _modelopt_eval_recursion_guard(model):
        ret = model.eval()

    # call post-eval hook explicitly
    if PatchManager.is_patched(model):
        PatchManager.get_manager(model).call_post_eval(forward_loop)

    # keep return value consistent with native eval()
    return ret
