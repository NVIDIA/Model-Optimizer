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

"""Utility functions for PyTorch models."""

import inspect
import types
import warnings
from collections import abc, deque
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any, Union

import torch
import torch.distributed.fsdp
import torch.nn as nn

try:
    from torch.distributed.fsdp._state_dict_utils import _convert_to_wrapped_module_name
except ImportError:
    warnings.warn(
        "`_convert_to_wrapped_module_name` could not be imported from torch.distributed.fsdp._state_dict_utils"
    )

    def _convert_to_wrapped_module_name(name: str) -> str:
        return name


from torch.nn.modules.batchnorm import _BatchNorm
from tqdm import tqdm

from .tensor import torch_to

__all__ = [
    "ModelLike",
    "compare_dict",
    "create_param_grad_clear_hook",
    "get_model_attributes",
    "get_module_device",
    "get_same_padding",
    "get_unwrapped_name",
    "init_model_from_model_like",
    "is_channels_last",
    "is_parallel",
    "make_divisible",
    "model_to",
    "param_num",
    "param_num_from_forward",
    "remove_bn",
    "run_forward_loop",
    "set_submodule",
    "standardize_constructor_args",
    "standardize_model_args",
    "standardize_model_like_tuple",
    "standardize_named_model_args",
    "unwrap_model",
    "zero_grad",
]

# NOTE: can be extended dynamically in appropriate plugin files if available (e.g. megatron core)
SUPPORTED_WRAPPERS: dict[type[nn.Module], str] = {
    nn.parallel.DataParallel: "module",  # indicating attribute key to unwrap
    nn.parallel.DistributedDataParallel: "module",
    torch.distributed.fsdp.FullyShardedDataParallel: "module",
}

try:
    from deepspeed.runtime.engine import DeepSpeedEngine
except:  # noqa: E722
    DeepSpeedEngine = None

if DeepSpeedEngine is not None:
    SUPPORTED_WRAPPERS[DeepSpeedEngine] = "module"
ModelLike = Union[nn.Module, type[nn.Module], tuple, Callable]  # noqa: UP007
ConstructorLike = Callable | tuple


def is_parallel(model: nn.Module) -> bool:
    """Check if a PyTorch model is parallelized."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def get_module_device(module: nn.Module) -> torch.device:
    """Get the device of a PyTorch module."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        # For modules without parameters
        return torch.device("cpu")


def param_num(network: nn.Module, trainable_only: bool = False, unit=1e6) -> float:
    """Get the number of parameters of a PyTorch model.

    Args:
        network: The PyTorch model.
        trainable_only: Whether to only count trainable parameters. Default is False.
        unit: The unit to return the number of parameters in. Default is 1e6 (million).

    Returns:
        The number of parameters in the model in the given unit.
    """
    return (
        sum(
            p.numel() if not trainable_only or p.requires_grad else 0
            for mod in network.modules()
            for p in mod.parameters(recurse=False)
            if not isinstance(mod, _BatchNorm)
        )
        / unit
    )


# TODO: we could also use the same approach as in inference_flops to get the number of params,
# which might be more accurate. Another approach could be to run a backwards pass and use a hook
# on the tensor directly.
def param_num_from_forward(
    model: nn.Module,
    trainable_only: bool = False,
    args: torch.Tensor | tuple | None = None,
    unit: float = 1e6,
):
    """Get the number of parameters of a PyTorch model from a forward pass.

    Args:
        network: The PyTorch model.
        trainable_only: Whether to only count trainable parameters. Default is False.
        unit: The unit to return the number of parameters in. Default is 1e6 (million).

    Returns:
        The number of parameters from the model's forward pass in the given unit.

    This can helpful for dynamic modules, where the state dict might contain extra parameters that
    is not actively used in the model, e.g., because of a DynamicModule that is deactivated for the
    forward pass. We circumvent this issue by just counting parameters of modules that appear in a
    forward pass.
    """
    params = {}

    def count_hook(m: nn.Module, *_):
        if m not in params:  # don't double-count parameters
            params[m] = sum(
                getattr(m, n).numel()  # use getattr to retrieve param since it might be dynamic
                for n, p in m.named_parameters(recurse=False)  # don't recurse!
                if not trainable_only or p.requires_grad
            )

    # add hook to count parameters to all modules except _BatchNorm
    hooks = [
        m.register_forward_hook(count_hook)
        for m in model.modules()
        if not isinstance(m, _BatchNorm)
    ]

    # run forward pass
    args = standardize_model_args(model, args, use_kwargs=True)
    args = torch_to(args, get_module_device(model))
    model(*args[:-1], **args[-1])

    # remove hooks
    for h in hooks:
        h.remove()

    # count parameters and return
    return sum(params.values()) / unit


def get_same_padding(kernel_size: int | tuple[int, int]) -> int | tuple:
    """Get the same padding for a given kernel size."""
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, f"invalid kernel size: {kernel_size}"
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    else:
        assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
        assert kernel_size % 2 == 1, "kernel size should be odd number"
        return kernel_size // 2


def make_divisible(v: int | float, divisor: int | None, min_val: int | None = None) -> int | float:
    """Function taken from the original tf repo.

    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if divisor is None:
        return v

    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


def is_channels_last(model: nn.Module):
    """Check if the model is using channels last memory format."""
    # Infer target_model's memory_format
    # from https://github.com/pytorch/tutorials/blob/444fbd16f2ddf9967baf8b06e83867a141b071c2/
    # intermediate_source/memory_format_tutorial.py#L283
    has_channels_last = any(
        p.is_contiguous(memory_format=torch.channels_last) and not p.is_contiguous()
        for p in model.parameters()
    )
    return has_channels_last


def model_to(model: nn.Module, target_model: nn.Module):
    """Convert model to the same device, dtype and memory layout as the target_model."""
    has_channels_last = is_channels_last(target_model)
    # return model with same device, dtype, memory_format as self
    return model.to(
        tensor=next(target_model.parameters()),
        memory_format=torch.channels_last if has_channels_last else torch.contiguous_format,
    )


def set_submodule(model: nn.Module, target: str, target_submodule: nn.Module):
    """The set function that complements nn.Module.get_submodule()."""
    assert target != "", "Cannot set root module"

    # Verify the original submodule exists
    model.get_submodule(target)
    parent_module = model.get_submodule(target.rpartition(".")[0])
    child_name = target.split(".")[-1]
    parent_module.add_module(child_name, target_submodule)


def remove_bn(model: nn.Module):
    """Remove all batch normalization layers in the network."""
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.weight = m.bias = None
            m.forward = lambda x: x


def _preprocess_args(args: Any | tuple) -> tuple:
    """Return args in standardized format as tuple with last entry as kwargs."""
    if not isinstance(args, tuple):
        args = (args,)
    if not isinstance(args[-1], abc.Mapping):
        args = (*args, {})

    return args


def standardize_named_model_args(
    model_or_fw_or_sig: nn.Module | Callable | inspect.Signature, args: Any | tuple
) -> tuple[dict[str, Any], set[str]]:
    """Standardize model arguments according to torch.onnx.export and give them a name.

    Args:
        model_or_fw_or_sig: A nn.Module, its forward method, or its forward method's signature.
        args: A tuple of args/kwargs or torch.Tensor feed into the model's ``forward()`` method.

    Returns: A tuple (args_normalized, args_with_default) where
        args_normalized is a dictionary of ordered model args where the key represents a unique
            serialized string based on the the argument's name in the function signature and the
            value contains the actual argument,
        args_with_default is a set indicating whether the argument was retrieved from the default
            value in the function signature of the model's ``forward()`` method or whether the
            argument exactly corresponds to the default value.

    .. note::

        See :meth:`standardize_model_args() <modelopt.torch.utils.network.standardize_model_args>` for
        more info as well.
    """
    # pre-process args
    args = _preprocess_args(args)

    # extract parameters from model signature
    if isinstance(model_or_fw_or_sig, nn.Module):
        model_or_fw_or_sig = inspect.signature(model_or_fw_or_sig.forward)
    elif callable(model_or_fw_or_sig):
        model_or_fw_or_sig = inspect.signature(model_or_fw_or_sig)
    params = model_or_fw_or_sig.parameters

    # we now continue to process the parameters in the function signature and classify them according
    # to their kind (see https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind for
    # an overview of the different kinds of parameters in a function signature)

    # sanity-check: kw-only must have default value and cannot be provided by user
    kw_only = [
        n
        for n, p in params.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY and (p.default == p.empty or n in args[-1])
    ]
    if kw_only:
        raise AssertionError(f"Keyword-only args ({kw_only}) can only be used w/ default values.")

    # sanity-check: kwargs in signature are okay but cannot be used by user!
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs_unexpected = any(kw not in params for kw in args[-1])
    if has_kwargs and kwargs_unexpected:
        raise AssertionError("Variable kwargs (**kwargs) are not supported.")

    # sanity-check: no unexpected kwargs provided by user
    assert not kwargs_unexpected, "Cannot provide unexpected keyword args!"

    # now sort in args_dict and default values
    args_queue = deque(args[:-1])
    args_dict = args[-1]
    args_normalized = {}
    args_with_default = set()
    for pname, param in params.items():
        # we peel off all positional/keyword arguments and fill them accordingly
        if param.kind in [param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD]:
            if args_queue:
                args_normalized[pname] = args_queue.popleft()
            elif pname in args_dict:
                args_normalized[pname] = args_dict[pname]
            elif param.default != param.empty:
                args_normalized[pname] = param.default
            else:
                # sanity check: any args without default value must be provided by the user
                raise AssertionError(f"Argument {pname} must be provided by the user.")
            # check if provided arg is exactly (``is``) the default value
            if args_normalized[pname] is param.default:
                args_with_default.add(pname)
        # when we have a var-positional arg (*args) we fill in the rest of the args
        elif param.kind == param.VAR_POSITIONAL:
            idx = 0
            while args_queue:
                args_normalized[f"{pname}.{idx}"] = args_queue.popleft()
                idx += 1
            # we also do not need to process further since everything following a var-positional
            # argument is keyword-only, which we don't allow!
            break

    # sanity-check: no positional arguments left
    assert not args_queue, "Positional arguments left unprocessed; too many provided!"

    # return the args (without kw-only args and kwargs) and set to indicate which args were
    # retrieved from default value in the function signature and which args exactly correspond to
    # the default value
    return args_normalized, args_with_default


def standardize_model_args(
    model_or_fw_or_sig: nn.Module | Callable | inspect.Signature,
    args: Any | tuple,
    use_kwargs: bool = False,
) -> tuple:
    """Standardize model arguments according to torch.onnx.export.

    Args:
        model_or_fw_or_sig: A nn.Module, its forward method, or its forward method's signature.
        args: Refer to the ``dummy_input`` parameter in
            :meth:`mtn.profile() <modelopt.torch.nas.algorithms.profile>`.
        use_kwargs: Affects the return value, see below. For ``use_kwargs==False``, the returned
            args are also compatible with ``torch.onnx.export``.

    Returns:
        Standardized model args that can be used in ``model.forward()`` in the same standardized
        way no matter how they were provided, see below for more info.

    * If ``use_kwargs == False``, the returned args can be used as

      .. code-block:: python

            args = standardize_model_args(model, args, use_kwargs=False)
            model(*args)

    * If ``use_kwargs == True``, the returned args can be used as

      .. code-block:: python

            args = standardize_model_args(model, args, use_kwargs=True)
            model.forward(*args[:-1], **args[-1])

    .. warning::

        If ``use_kwargs == False`` the model's ``forward()`` method **cannot** contain keyword-only
        arguments (e.g. ``forward(..., *, kw_only_args)``) without default values and you must not
        provide them in ``args``.

    .. warning::

        If ``use_kwargs == False`` you must not provide variable keyword arguments in ``args`` that
        are processed via variable keyword arguments in the model's ``forward()`` method
        (e.g. ``forward(..., **kwargs)``).

    """
    # preprocess args
    args = _preprocess_args(args)

    # simply return as args/kwargs in this case
    if use_kwargs:
        return args

    # return sorted args without names in this case
    return tuple(standardize_named_model_args(model_or_fw_or_sig, args)[0].values())


def get_model_attributes(model: nn.Module) -> dict[str, Any]:
    """Get the key attributes of a PyTorch model."""
    attrs = {}
    attrs["type(model)"] = type(model).__name__
    attrs["model.forward"] = getattr(model.forward, "__name__", None)
    keys = ["training"]
    for key in keys:
        attrs[key] = getattr(model, key)
    return attrs


def compare_dict(dict1: dict[str, Any], dict2: dict[str, Any]) -> tuple[str, ...]:
    """Compare two dictionaries and return keys with unmatched values."""
    keys_unmatched = tuple(k for k in dict1.keys() & dict2.keys() if dict1[k] != dict2[k])
    keys_unmatched += tuple(dict1.keys() ^ dict2.keys())
    return keys_unmatched


def unwrap_model(
    model: nn.Module,
    warn: bool = False,
    raise_error: bool = False,
    msg: str = "",
    force_unwrap: bool = False,
) -> nn.Module:
    """Unwrap a model that is wrapped by supported wrapper module or return original model."""
    if force_unwrap:
        try:
            if type(model) in SUPPORTED_WRAPPERS:
                return getattr(model, SUPPORTED_WRAPPERS[type(model)])
        except AttributeError:
            raise ValueError(
                f"Model of type {type(model)} could not be forcefully unwrapped! Please manually"
                " unwrap the model before passing it in."
            )

    if type(model) in SUPPORTED_WRAPPERS:
        if raise_error:
            raise ValueError(msg or f"Model {model} is wrapped by {type(model)}!")
        elif warn:
            warnings.warn(msg or f"Model {model} is wrapped by {type(model)}; unwrapping...")
        return getattr(model, SUPPORTED_WRAPPERS[type(model)])
    return model


def zero_grad(model: nn.Module) -> None:
    """Set any gradients in the model's parameters to None."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


def standardize_model_like_tuple(model: ModelLike) -> tuple[type[nn.Module], tuple, dict]:
    """Standardize a model-like tuple."""
    if not (isinstance(model, (type, tuple)) or callable(model)):
        raise ValueError(f"Expected type or tuple or callable but got {model}")
    return standardize_constructor_args(model)  # type: ignore[return-value]


def standardize_constructor_args(constructor_args: ConstructorLike) -> tuple[Callable, tuple, dict]:
    """Standardize a constructor-like tuple."""
    if not isinstance(constructor_args, tuple):
        constructor_args = (constructor_args,)

    if len(constructor_args) == 1:
        constructor_args = (*constructor_args, (), {})
    elif len(constructor_args) == 2:
        constructor_args = (*constructor_args, {})

    cls_or_callable, args, kwargs = constructor_args
    assert isinstance(args, (tuple, list)), f"Invalid model args: {args}"
    assert isinstance(kwargs, dict), f"Invalid model kwargs: {kwargs}"
    return cls_or_callable, tuple(args), kwargs


def init_model_from_model_like(model: ModelLike) -> nn.Module:
    """Initialize a model from a model-like object.

    Args:
        model: A model-like object. Can be a nn.Module (returned as it is), a model class or callable, or a tuple.
            If a tuple, it must be of the form (model_cls_or_callable,) or (model_cls_or_callable, args) or
            (model_cls_or_callable, args, kwargs).
            Model will be initialized as ``model_cls_or_callable(*args, **kwargs)``.
    """
    if isinstance(model, nn.Module):
        return model

    model_cls, args, kwargs = standardize_model_like_tuple(model)
    return model_cls(*args, **kwargs)


def run_forward_loop(
    model,
    data_loader: Iterable,
    max_iters: int | None = None,
    collect_func: Callable[[Any], Any | tuple] | None = None,
    progress_bar: str | None = None,
    post_process: Callable | None = None,
):
    """Run multiple forward passes with a model according to the provided data loader.

    Args:
        model: The model with which we run forward.
        data_loader: An iterator with data samples.
        max_iters: Number of batches to run; by default it is infiinite or until ``data_loader``
            is exhausted.
        collect_func: A ``Callable`` that takes a batch of data from the ``data_loader``
              as input and returns the input to ``model.forward()`` such that the return value
              (``input``) is either:

                #. a single argument (``type(input) != tuple``) corresponding to

                   .. code-block:: python

                        model.forward(input)

                #. a tuple of arguments corresponding to

                   .. code-block:: python

                        model.forward(*input)

                #. a tuple of arguments such that ``type(input[-1]) == dict`` corresponding to

                   .. code-block:: python

                        model.forward(*input[:-1], **input[-1])

                .. note::

                    In order to pass a dict as last non-keyword argument, you need to use a tuple as
                    ``input`` and add an *empty* dict as the last element, e.g.,

                    .. code-block:: python

                        input = (x, {"y": y, "z": z}, {})

                    The empty dict at the end will then be interpreted as the keyword args.

                See the ``args`` argument of
                `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_
                for more info on the format of the return value of ``collect_func`` (``input``).

                The default ``collect_func`` assumes that the data loader returns a tuple, e.g.,
                ``(images, labels, ...)``, and returns the first element of the tuple.

        progress_bar: Set to a description string to see the progress bar.
        post_process: A callable that takes the model outputs and the data as input and can be used to
            run any post-processing or operations such as backward pass.
    """
    device = get_module_device(model)
    collect_fn = collect_func or (lambda x: x[0])
    with tqdm(total=max_iters, desc=progress_bar, disable=(not progress_bar)) as t:
        for idx, data in enumerate(data_loader):
            if isinstance(max_iters, int) and idx >= max_iters:
                break
            args = standardize_model_args(model, collect_fn(data), use_kwargs=True)
            args = torch_to(args, device)
            outputs = model(*args[:-1], **args[-1])
            if post_process:
                post_process(outputs, data)
            t.update()


@torch.enable_grad()
def create_param_grad_clear_hook(param):
    """Create a hook to clear gradients for a parameter.

    The hook will be fired after the gradient is accumulated for the parameter.
    Important: For this to work, ``accum_grad`` should be kept alive as longs as this utility is needed.
    """
    # For methods such as AutoQuantize, gradnas involving backward -
    # We want to clear parameter gradients as soon as they are computed to save memory
    # This can be done by register_post_accumulate_grad_hook
    # However torch <= 2.0 does not have register_post_accumulate_grad_hook
    # This is a workaround for that
    # Pytorch FSDP uses a similar workaround - https://github.com/pytorch/pytorch/blob/00cb184512f3a636d87793f46d3f9c7fea406b25/torch/distributed/fsdp/fully_sharded_data_parallel.py#L2825-L2835

    def delete_grad_hook(*_unused):
        param.grad = None

    # Gets param's AccumulateGrad object & register a hook on it.
    accum_grad = param.view_as(param).grad_fn.next_functions[0][0]
    handle = accum_grad.register_hook(delete_grad_hook)
    return accum_grad, handle


def get_unwrapped_name(name: str, model: nn.Module | None = None) -> str:
    """Get the cleaned module name (i.e, the name before wrapping with sharded modules)."""
    # The distributed sharded wrappers such as FSDP wraps the child modules as well
    # So unwrapping just the parent module is not enough
    # Instead of unwrapping the child modules and changing the model, we can just clean the name
    # _convert_to_wrapped_module_name is a Pytorch utility function to do this
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)) or (
        DeepSpeedEngine is not None and isinstance(model, DeepSpeedEngine)
    ):
        name = name.removeprefix("module.")

    name = _convert_to_wrapped_module_name(name)
    name = name.removesuffix(".")
    return name


@contextmanager
def temporarily_remove_accelerate_hook(module):
    """Context manager to temporarily remove accelerate hook from a module."""
    accelerate_hook = None
    if hasattr(module, "_hf_hook"):
        # A module with forward method patched by accelerate
        from accelerate.hooks import add_hook_to_module, remove_hook_from_module

        accelerate_hook = module._hf_hook
        remove_hook_from_module(module)
    try:
        yield
    finally:
        if accelerate_hook is not None:
            from accelerate.hooks import add_hook_to_module

            add_hook_to_module(module, accelerate_hook)


def bind_forward_method(
    module: nn.Module, forward_fn: Callable, orig_forward_cache_name: str | None = None
):
    """Correctly bind the forward method of a module with specified `forward_fn`.

    If this module's forward is already patched by accelerate, we temporarily remove the patch,
    bind the forward method, and then reapply the patch.

    If the specified `forward_fn` is not bound to the module, it will be bound to the module.
    Optionally, a name can be specified for the caching of the original forward method.
    """
    with temporarily_remove_accelerate_hook(module):
        if orig_forward_cache_name is not None:
            setattr(module, orig_forward_cache_name, getattr(module, "forward"))

        if not hasattr(forward_fn, "__self__") or forward_fn.__self__ is not module:
            # The forward function is not bound to the module, so we need to bind it
            module.forward = types.MethodType(forward_fn, module)
        else:
            module.forward = forward_fn


def unpatch_forward_method(module: nn.Module, orig_forward_cache_name: str):
    """Unpatch the forward method of a module."""
    with temporarily_remove_accelerate_hook(module):
        setattr(module, "forward", getattr(module, orig_forward_cache_name))
        delattr(module, orig_forward_cache_name)
