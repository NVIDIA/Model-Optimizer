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
# mypy: ignore-errors
from __future__ import annotations

import inspect
from collections.abc import Sequence
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    ContextManager,
    Generic,
    Iterable,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    cast,
    overload,
)

import torch
import torch._C
import torch._dynamo
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch import Tensor
from torch._subclasses import FakeTensor, FakeTensorMode
from typing_extensions import override

Fn = TypeVar("Fn", bound=Callable)


class DynamoSkip(Protocol):
    @overload
    def __call__(self, fn: None = None) -> Callable[[Fn], Fn]: ...
    @overload
    def __call__(self, fn: Fn) -> Fn: ...


class DynamoDisable(Protocol):
    @overload
    def __call__(self, fn: None = None, disable: bool = False) -> Callable[[Fn], Fn]: ...
    @overload
    def __call__(self, fn: Fn, disable: bool = False) -> Fn: ...


try:
    dynamo_skip: DynamoSkip = cast(Any, torch._dynamo.decorators).skip
    dynamo_disable: DynamoDisable = cast(Any, torch._dynamo.decorators).disable
except:
    dynamo_skip: DynamoSkip = cast(Any, torch._dynamo.eval_frame).skip
    dynamo_disable: DynamoDisable = cast(Any, torch._dynamo.eval_frame).disable


TModule = TypeVar("TModule", bound=nn.Module)


class ModuleRef(Generic[TModule]):
    def __init__(self, module: TModule):
        self.module = module


class ActivityContextMaxDepthException(Exception):
    pass


class ActivityContextDuplicateException(Exception):
    pass


T = TypeVar("T")


class ActivityContext(Generic[T]):
    def __init__(self, max_depth: Optional[int] = None, no_duplicates=False, reversed=False):
        self.activity_stack: list[T] = []
        self.max_depth = max_depth
        self.no_duplicates = no_duplicates
        self.reversed = reversed
        self.head_index = 0 if self.reversed else -1

    def __contains__(self, value: T) -> bool:
        result = value in self.activity_stack
        return result

    def __call__(self, value: T) -> ContextManager:
        @contextmanager
        def fn():
            try:
                if self.no_duplicates and value in self.activity_stack:
                    raise ActivityContextDuplicateException(
                        f"Activity stack cannot have a duplicate of item {value}"
                    )

                self.activity_stack.insert(self.head_index, value)

                if self.max_depth is not None and len(self) > self.max_depth:
                    raise ActivityContextMaxDepthException(
                        f"Activity stack exceeds max depth of {self.max_depth}"
                    )

                yield
            finally:
                assert self.is_active()
                self.activity_stack.pop(self.head_index)

        return fn()

    def __len__(self) -> int:
        result = len(self.activity_stack)
        return result

    @overload
    def __getitem__(self, key: int) -> T: ...
    @overload
    def __getitem__(self, key: slice) -> Sequence[T]: ...
    def __getitem__(self, key: int | slice) -> T | Sequence[T]:
        result = self.activity_stack[key]
        return result

    def is_active(self) -> bool:
        result = len(self) > 0
        return result

    def get_active(self) -> Optional[T]:
        if self.is_active:
            return self.activity_stack[-1]
        else:
            return None


def is_submodule_of(module_name: str, other_module_name: str) -> bool:
    result = module_name.startswith(f"{other_module_name}.") or (
        module_name != "" and other_module_name == ""
    )
    return result


def is_submodule_or_same(module_name: str, other_module_name: str) -> bool:
    result = module_name == other_module_name or is_submodule_of(module_name, other_module_name)
    return result


fake_mode = FakeTensorMode(
    allow_non_fake_inputs=True,
    # allow_fallback_kernels=False,
)


@overload
def fake_tensor(t: Tensor, *, dtype: Optional[torch.dtype] = None, use_meta=False) -> Tensor: ...


@overload
def fake_tensor(
    size: Sequence[int] | torch.Size, *, dtype: Optional[torch.dtype] = None, use_meta=False
) -> Tensor: ...


@overload
def fake_tensor(*args: int, dtype: Optional[torch.dtype] = None, use_meta=False) -> Tensor: ...


class MyFakeTensor(Tensor):
    @dynamo_disable
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._t: FakeTensor

    @override
    @dynamo_disable
    def __repr__(self, *, tensor_contents=None):
        return f"MyFakeTensor(shape={list(self._t.shape)}, dtype={self._t.dtype}, device={self._t.device})"

    @classmethod
    @override
    @dynamo_disable
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args, kwargs = pytree.tree_map_only(MyFakeTensor, lambda t: t._t, (args, kwargs))

        types = pytree.tree_map_only(type(MyFakeTensor), lambda t: FakeTensor, types)

        out = func(*args, **kwargs)

        out = pytree.tree_map_only(Tensor, lambda t: MyFakeTensor.create(t), out)

        return out

    __torch_function__ = torch._C._disabled_torch_function_impl

    # @dynamo_disable
    # def __getattribute__(self, attr: str):
    #     if attr in {'_t', 'device', '__repr__', '__torch_function__', '__class__'}:
    #         return object.__getattribute__(self, attr)

    #     result = getattr(self._t, attr)

    #     result = pytree.tree_map_only(
    #         Tensor, lambda t: MyFakeTensor.create(t), result
    #     )
    #     print('__getattribute__', 'attr', attr, 'ret', result)

    #     return result

    @property
    @dynamo_disable
    def device(self):
        return self._t.device

    # @property
    # @dynamo_disable
    # def shape(self):
    #     return self._t.shape

    # @dynamo_disable
    # def size(self):
    #     return self._t.size()

    # @classmethod
    # @dynamo_disable
    # def __torch_function__(cls, func, types, args=(), kwargs=None):
    #     if kwargs is None:
    #         kwargs = {}

    #     args, kwargs = pytree.tree_map_only(
    #         MyFakeTensor, lambda t: t._t, (args, kwargs)
    #     )

    #     ret = func(*args, **kwargs)

    #     ret = pytree.tree_map_only(
    #         Tensor, lambda t: MyFakeTensor.create(t), ret
    #     )
    #     print('__torch_function__', 'func', func, 'ret', ret)

    #     return ret

    @staticmethod
    @dynamo_disable
    def __new__(cls, elem, device) -> MyFakeTensor:
        self = torch.Tensor._make_subclass(
            cls,
            elem,
            elem.requires_grad,
            dispatch_device=True,
            device_for_backend_keys=device,
        )
        return cast(MyFakeTensor, self)

    @classmethod
    @dynamo_disable
    def create(cls, data: Tensor) -> MyFakeTensor:
        if isinstance(data, MyFakeTensor):
            return data

        if isinstance(data, FakeTensor):
            t = data
        else:
            t = FakeTensor.from_tensor(data, fake_mode=fake_mode)

        # my_fake_tensor = MyFakeTensor(torch.empty(t.shape, dtype=t.dtype, device='meta'))
        my_fake_tensor = MyFakeTensor(
            torch.empty(t.shape, dtype=t.dtype, device="meta"),
            t.device,
        )
        my_fake_tensor._t = t

        return my_fake_tensor


@dynamo_disable
def fake_tensor(*args, **kwargs) -> Tensor:
    dtype: Optional[torch.dtype] = kwargs.get("dtype")
    use_meta = kwargs.get("use_meta", False)

    if len(args) == 1 and isinstance(args[0], Tensor):
        if use_meta:
            fake_tensor = torch.empty(args[0].size(), dtype=dtype or args[0].dtype, device="meta")
        else:
            fake_tensor = MyFakeTensor.create(args[0])
    else:
        fake_tensor = torch.empty(*args, dtype=dtype, device="meta")
        if not use_meta:
            fake_tensor = MyFakeTensor.create(fake_tensor)

    return fake_tensor


@dynamo_skip
def fake_tensor_like(t: Tensor, use_meta=False) -> Tensor:
    return fake_tensor(t, use_meta=use_meta)


T = TypeVar("T")


@dynamo_skip
def fake_tensors(value: T, use_meta=False) -> T:
    result = pytree.tree_map_only(Tensor, lambda t: fake_tensor_like(t, use_meta), value)
    return result
    # if isinstance(value, Mapping):
    #     return cast(Any, value.__class__)({k: fake_tensors(v, use_meta) for k, v in value.items()})
    # if isinstance(value, Sequence):
    #     return cast(Any, value.__class__)([fake_tensors(v, use_meta) for v in value])
    # if isinstance(value, Tensor):
    #     return fake_tensor_like(value, use_meta)
    # return value


@dynamo_skip
def real_tensors(value: Any) -> Any:
    result = pytree.tree_map_only(Tensor, lambda t: None if is_fake_tensor(t) else t, value)
    return result
    # if isinstance(value, Mapping):
    #     return cast(Any, value.__class__)({k: real_tensors(v) for k, v in value.items()})
    # if isinstance(value, Sequence):
    #     return cast(Any, value.__class__)([real_tensors(v) for v in value])
    # if is_fake_tensor(value):
    #     return None
    # return value


@dynamo_skip
def is_fake_tensor(t: Any) -> bool:
    return isinstance(t, (MyFakeTensor, FakeTensor)) or (isinstance(t, Tensor) and t.is_meta)


@dynamo_skip
def has_fake_tensor(v: Any) -> bool:
    result = pytree.tree_any(is_fake_tensor, v)
    return result


def _get_device_for_distributed(
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> str:
    """
    Determine the appropriate device for distributed communication based on the backend.
    NCCL backend requires CUDA tensors, while Gloo supports both CPU and CUDA.
    """
    if not torch.distributed.is_initialized():
        return "cpu"

    backend = torch.distributed.get_backend(group)
    if backend == "nccl":
        # NCCL requires CUDA tensors
        return torch.cuda.current_device()
    else:
        # Gloo and other backends support CPU tensors
        return "cpu"


def distributed_isend_obj(
    obj: Any,
    dst: int = 0,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> list[Optional[torch.distributed.Work]]:
    device = _get_device_for_distributed(group)
    obj_tensor, obj_size_tensor = torch.distributed.distributed_c10d._object_to_tensor(
        obj, device=device, **_get_group_kwarg_if_necessary()
    )
    works: list[Optional[torch.distributed.Work]] = [
        torch.distributed.isend(obj_size_tensor, dst, group),
        torch.distributed.isend(obj_tensor, dst, group),
    ]
    # p2p_ops = [
    #     torch.distributed.P2POp(torch.distributed.isend, obj_size_tensor, dst, group),
    #     torch.distributed.P2POp(torch.distributed.isend, obj_tensor, dst, group),
    # ]

    # works = torch.distributed.batch_isend_irecv(p2p_ops)

    return works


def distributed_send_obj(
    obj: Any,
    dst: int = 0,
    group: Optional[torch.distributed.ProcessGroup] = None,
):
    works = distributed_isend_obj(obj=obj, dst=dst, group=group)
    for work in works:
        if work is not None:
            work.wait()


def distributed_recv_obj(
    src: Optional[int] = None,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> Any:
    device = _get_device_for_distributed(group)
    obj_size_tensor = torch.LongTensor(1).to(device)
    torch.distributed.recv(obj_size_tensor, src=src, group=group)
    obj_size = int(obj_size_tensor.item())

    obj_tensor = torch.ByteTensor(obj_size).to(device)
    torch.distributed.recv(obj_tensor, src=src, group=group)

    obj = torch.distributed.distributed_c10d._tensor_to_object(
        obj_tensor, obj_size, **_get_group_kwarg_if_necessary()
    )

    return obj


def _get_group_kwarg_if_necessary() -> dict:
    """For newer versions of torch"""
    arg_names = inspect.signature(
        torch.distributed.distributed_c10d._object_to_tensor
    ).parameters.keys()
    return dict(group=None) if "group" in arg_names else dict()
