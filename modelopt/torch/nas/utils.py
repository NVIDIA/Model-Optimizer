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

"""Utility functions for prune-related and search-space related tasks.

.. note::

    Generally, methods in the :meth:`modelopt.torch.nas<modelopt.torch.nas>` module should use these utility
    functions directly instead of accessing the ``SearchSpace`` class. This is to ensure that
    potentially required pre- and post-processing operations are performed correctly.
"""

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from torch.autograd.grad_mode import _DecoratorContextManager

from modelopt.torch.utils import (
    get_module_device,
    is_parallel,
    num2hrb,
    random,
    standardize_model_args,
    unwrap_model,
)

from .search_space import SampleFunc, SearchSpace

# only certain categories of utils via __all__
__all__ = [  # noqa: RUF022
    # search/profile related utils
    "inference_flops",
    "print_search_space_summary",
    # sample/select-related utils
    "get_subnet_config",
    "sample",
    "select",
    # auto-mode related utils
    "enable_modelopt_patches",
    "is_modelopt_patches_enabled",
    "no_modelopt_patches",
    "set_modelopt_patches_enabled",
    # other utils that are used frequently
    "replace_forward",
]


@contextmanager
def batch_norm_ignored_flops():
    import torchprofile.profile as profile

    handlers_bck = profile.handlers
    handlers_new = []
    for op_names, op in profile.handlers:
        if "aten::batch_norm" in op_names:
            op_names_new = [op_name for op_name in op_names if op_name != "aten::batch_norm"]
            handlers_new.append((tuple(op_names_new), op))
        else:
            handlers_new.append((op_names, op))
    profile.handlers = tuple(handlers_new)
    yield
    profile.handlers = handlers_bck


def inference_flops(
    network: nn.Module,
    dummy_input: Any | tuple[Any, ...] | None = None,
    data_shape: tuple[Any, ...] | None = None,
    unit: float = 1e6,
    return_str: bool = False,
) -> float | str:
    """Get the inference FLOPs of a PyTorch model.

    Args:
        network: The PyTorch model.
        args: The dummy input as defined in
            :meth:`mtn.convert() <modelopt.torch.nas.conversion.convert>`.
        data_shape: The shape of the dummy input if the dummy input is a single tensor. If provided,
            ``args`` must be ``None``.
        unit: The unit to return the number of parameters in. Default is 1e6 (million).
        return_str: Whether to return the number of FLOPs as a string.

    Returns:
        The number of inference FLOPs in the given unit as either string or float.
    """
    try:
        import torchprofile.profile as profile
    except ImportError as e:
        raise ImportError("Please run `pip install torchprofile` to use this function.") from e

    if is_parallel(network):
        network = network.module
    if data_shape is not None and dummy_input is not None:
        raise ValueError("Please provide either data_shape or args tuple.")
    if data_shape is not None:
        dummy_input = (torch.zeros(data_shape, device=get_module_device(network)),)
    dummy_input = standardize_model_args(network, dummy_input)
    is_training = network.training
    network.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with batch_norm_ignored_flops():
            flops = 2 * profile.profile_macs(network, args=dummy_input)
    network.train(is_training)
    if return_str:
        return num2hrb(flops)
    return flops / unit


class _SearchSpaceUnwrapped(SearchSpace):
    """A wrapper for the SearchSpace class to handle unwrapping of model wrappers like DDP.

    This is useful to ensure that configurations are valid for both vanilla models and wrapped
    models, see :meth:`unwrap_models<modelopt.torch.utils.network.unwrap_model>` for supported wrappers.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__(unwrap_model(model))


def sample(model: nn.Module, sample_func: SampleFunc = random.choice) -> dict[str, Any]:
    """Sample searchable hparams using the provided sample_func and return resulting config.

    Args:
        model: A searchable model that contains one or more DynamicModule(s).
        sample_func: A sampling function for hyperparameters. Default: random sampling.

    Returns:
        A dict of ``(parameter_name, choice)`` that specifies an active subnet.
    """
    _reset_before_sample(model)
    return _SearchSpaceUnwrapped(model).sample(sample_func)


def select(model: nn.Module, config: dict[str, Any], strict: bool = True) -> None:
    """Select the sub-net according to the provided config dict.

    Args:
        model: A model that contains DynamicModule(s).
        config: Config of the target subnet as returned by
            :meth:`mtn.config()<modelopt.torch.nas.utils.config>`
            and :meth:`mtn.search()<modelopt.torch.nas.algorithms.search>`.
        strict: Raise an error when the config does not contain all necessary keys.
    """
    # select new sub-net configuration
    _reset_before_sample(model)
    _SearchSpaceUnwrapped(model).select(config, strict)


def get_subnet_config(model: nn.Module, configurable: bool | None = None) -> dict[str, Any]:
    """Return the config dict of all hyperparameters.

    Args:
        model: A model that contains DynamicModule(s).
        configurable: None -> all hps, True -> configurable hps without duplicates

    Returns:
        A dict of ``(parameter_name, choice)`` that specifies an active subnet.
    """
    return _SearchSpaceUnwrapped(model).config(configurable)


def _reset_before_sample(model: nn.Module):
    """Call pre-sample reset hook from patch manager."""
    from .patch import PatchManager

    if PatchManager.is_patched(model):
        PatchManager.get_manager(model).reset_before_sample()


def sort_parameters(
    model: nn.Module, hps_to_sort: set[str] | None = None, verbose: bool = False
) -> None:
    """Sort the parameters of the model according to the stored importances.

    Args:
        model: A model that contains DynamicModule(s).
        hps_to_sort: A set of hparam names to sort. If not provided or empty, all hparams will be sorted.
        verbose: Whether to print the search space and hparam importances.
    """
    _SearchSpaceUnwrapped(model).sort_parameters(hps_to_sort, verbose)


def print_search_space_summary(
    model: nn.Module, skipped_hparams: list[str] = ["kernel_size"]
) -> None:
    """Print the search space summary.

    Args:
        model: A model that contains DynamicModule(s).
    """
    _SearchSpaceUnwrapped(model).print_summary(skipped_hparams)


class _ModeloptOpsState:
    """Global flag to enable/disable modelopt patches - defaults to True."""

    _instance: "_ModeloptOpsState | None" = None
    _auto_enabled = True

    def __new__(cls) -> "_ModeloptOpsState":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_modelopt_mode_enabled(self, enabled: bool):
        self._auto_enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._auto_enabled


def is_modelopt_patches_enabled() -> bool:
    """Check if modelopt patches for model are enabled."""
    return _ModeloptOpsState().enabled


class no_modelopt_patches(_DecoratorContextManager):  # noqa: N801
    """Context manager to disable ``modelopt`` patches to the model.

    Disabling ``modelopt`` patches is useful when you want to use the model's original behavior
    For example, you can use this to perform a forward pass without NAS operations.

    It can also be used as a decorator (make sure to instantiate with parenthesis).

    For example:

    .. code-block:: python

        modelopt_model.train()
        modelopt_model(inputs)  # architecture changes

        with mtn.no_modelopt():
            modelopt_model(inputs)  # architecture does not change


        @mtn.no_modelopt()
        def forward(model, inputs):
            return model(inputs)


        forward(modelopt_model, inputs)  # architecture does not change
    """

    def __init__(self):
        """Constructor."""
        self.prev = False

    def __enter__(self):
        self.prev = is_modelopt_patches_enabled()
        set_modelopt_patches_enabled(False)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        set_modelopt_patches_enabled(self.prev)


class enable_modelopt_patches(_DecoratorContextManager):  # noqa: N801
    """Context manager to enable ``modelopt`` patches such as those for autonas/fastnas.

    It can also be used as a decorator (make sure to instantiate with parenthesis).

    For example:

    .. code-block:: python

        modelopt_model.train()
        modelopt_model(inputs)  # architecture changes

        with mtn.no_modelopt():
            with mtn.enable_modelopt():
                modelopt_model(inputs)  # architecture changes


        @mtn.enable_modelopt()
        def forward(model, inputs):
            return model(inputs)


        with mtn.no_modelopt():
            forward(modelopt_model, inputs)  # architecture changes because of decorator on forward
    """

    def __init__(self):
        """Constructor."""
        self.prev = True

    def __enter__(self):
        self.prev = is_modelopt_patches_enabled()
        set_modelopt_patches_enabled(True)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        set_modelopt_patches_enabled(self.prev)


class set_modelopt_patches_enabled(_DecoratorContextManager):  # noqa: N801
    """Context manager that sets patches to on or off.

    It can be used as context manager or as a function. If used as function,  operations
    are disabled globally (thread local).

    Args:
        enabled: whether to enable (``True``) or disable (``False``) patched methods.

    For example:

    .. code-block:: python

        modelopt_model.train()
        modelopt_model(inputs)  # architecture changes

        mtn.set_modelopt_enabled(False)
        modelopt_model(inputs)  # architecture does not change

        with mtn.set_modelopt_enabled(True):
            modelopt_model(inputs)  # architecture changes

        modelopt_model(inputs)  # architecture does not change
    """

    def __init__(self, enabled: bool):
        """Constructor."""
        self.prev = is_modelopt_patches_enabled()
        _ModeloptOpsState().set_modelopt_mode_enabled(enabled)
        self.enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        _ModeloptOpsState().set_modelopt_mode_enabled(self.prev)

    def clone(self):
        """Clone the context manager."""
        return self.__class__(self.enabled)


@contextmanager
def replace_forward(model: nn.Module, new_forward: Callable) -> Iterator[None]:
    """Context manager to temporarily replace the forward method of the underlying type of a model.

    The original forward function is temporarily accessible via ``model.forward_original``.

    Args:
        model: The model whose type's forward method is to be temporarily replaced.
        new_forward: The new forward method. The forward method should either be a bound method to
            the model instance or take the model (``self``) as the first argument.

    For example:

    .. code-block:: python

        fake_forward = lambda _: None

        with replace_forward(model, fake_forward):
            out = model(inputs)  # this output is None

        out_original = model(inputs)  # this output is the original output
    """
    # This is functional context manager using the @contextmanager decorator:
    # https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager
    type(model).forward_original = type(model).forward
    type(model).forward = getattr(new_forward, "__func__", new_forward)
    try:
        yield
    finally:
        type(model).forward = type(model).forward_original
        del type(model).forward_original


@contextmanager
def sample_and_reset(model: nn.Module, sample_func: SampleFunc = random.choice) -> Iterator[None]:
    """Context manager to temporarily sample a subnet based on a given sample_func.

    Args:
        model: A searchable model that contains one or more DynamicModule(s).
        sample_func: A sampling function for hyperparameters. Default: random sampling.

    For example:

    .. code-block:: python

        with sample_and_reset(model, random.original):
            out = model(inputs)  # uses the original subnet

    """
    config = get_subnet_config(model)
    try:
        sample(model, sample_func)
        yield
    finally:
        select(model, config)
