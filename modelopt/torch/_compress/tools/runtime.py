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

"""Classes for torch distributed runtime management"""

import os
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, suppress
from datetime import timedelta
from pathlib import Path
from typing import Literal, TypeVar, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import override

PrepareModelsT = TypeVar("PrepareModelsT", bound=Sequence[nn.Module])
PrepareDataLoaderT = TypeVar("PrepareDataLoaderT", bound=DataLoader)
CompileT = TypeVar("CompileT", bound=nn.Module)
Filter = (
    Literal["main_process", "last", "local_main_process", "local_last", "all"]
    | list[int]
    | set[int]
    | Callable[[int], bool]
)


class IRuntime(ABC):
    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def cleanup(self) -> None: ...

    @abstractmethod
    def autocast(self) -> AbstractContextManager: ...

    @abstractmethod
    def wait_for_everyone(self) -> None: ...

    @abstractmethod
    def set_seed(self, seed: int, device_specific: bool = False) -> int: ...

    @abstractmethod
    def prepare_models(self, models: PrepareModelsT) -> PrepareModelsT: ...

    @abstractmethod
    def prepare_train_dataloader(
        self, train_dataloader: PrepareDataLoaderT
    ) -> PrepareDataLoaderT: ...

    @abstractmethod
    def prepare_val_dataloader(self, val_dataloader: PrepareDataLoaderT) -> PrepareDataLoaderT: ...

    @abstractmethod
    def compile(self, model: CompileT) -> CompileT: ...

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None: ...

    @abstractmethod
    def clip_grad_norm_(
        self,
        parameters: Iterable[torch.Tensor] | torch.Tensor,
        max_norm: float,
        norm_type: float = 2,
    ) -> torch.Tensor: ...

    @abstractmethod
    def clip_grad_value_(
        self, parameters: Iterable[torch.Tensor] | torch.Tensor, clip_value: float
    ) -> None: ...

    @abstractmethod
    def save_state(self, path: str | Path) -> None: ...

    @abstractmethod
    def load_state(self, path: str | Path) -> None: ...

    @abstractmethod
    def skip_first_batches(self, dataloader_iterator: Iterator, num_batches: int) -> None: ...

    @property
    @abstractmethod
    def sync_gradients(self) -> bool: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def is_main_process(self) -> bool: ...

    @property
    @abstractmethod
    def is_local_main_process(self) -> bool: ...

    @property
    @abstractmethod
    def is_last_process(self) -> bool: ...

    @property
    @abstractmethod
    def is_local_last_process(self) -> bool: ...

    @property
    @abstractmethod
    def local_rank(self) -> int: ...

    @property
    @abstractmethod
    def global_rank(self) -> int: ...

    @property
    @abstractmethod
    def local_world_size(self) -> int: ...

    @property
    @abstractmethod
    def world_size(self) -> int: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # avoid barrier if exceution errored
        if exc_type is None:
            self.cleanup()

        # if exc_type is not None:
        #     raise exc_value
        # Handle exceptions if necessary
        # pass

    # def __del__(self):
    #     torch.distributed.barrier()
    #     torch.distributed.destroy_process_group()

    def check_filter(self, filter_: Filter):
        return (
            filter_ == "all"
            or (filter_ == "main_process" and self.is_main_process)
            or (filter_ == "local_main_process" and self.is_local_main_process)
            or (filter_ == "last" and self.is_last_process)
            or (filter_ == "local_last" and self.is_local_last_process)
            or (isinstance(filter_, (list, set)) and self.global_rank in filter_)
            or (callable(filter_) and filter_(self.global_rank))
        )

    def print(
        self, *args, filter_: Filter = "main_process", rank_prefix=False, flush=True, **kwargs
    ) -> None:
        if not self.check_filter(filter_):
            return

        if rank_prefix:
            print(f"[global_rank={self.global_rank}]", *args, flush=flush, **kwargs)
        else:
            print(*args, flush=flush, **kwargs)

    def process_print(
        self, *args, filter_: Filter = "all", rank_prefix=True, flush=True, **kwargs
    ) -> None:
        if not self.check_filter(filter_):
            return

        if rank_prefix:
            prefix = f"[global_rank={self.global_rank}]"
            if len(args) == 1:  # avoid out-of-order printing if possible
                out = f"{prefix} {args[0]}"
                args = (out,)
            else:
                args = (prefix, *args)
            print(*args, flush=flush, **kwargs)
        else:
            print(*args, flush=flush, **kwargs)


class NativeDdpRuntime(IRuntime):
    def __init__(
        self,
        dtype: torch.dtype = torch.float,
        torch_distributed_timeout: timedelta | None = None,
    ):
        self._master_addr = os.environ["MASTER_ADDR"]
        self._master_port = int(os.environ["MASTER_PORT"])
        self._local_rank = int(os.environ["LOCAL_RANK"])
        self._global_rank = int(os.environ["RANK"])
        self._local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        self._world_size = int(os.environ["WORLD_SIZE"])
        self._device = torch.device(self.local_rank)
        self._dtype = dtype
        self._torch_distributed_timeout = torch_distributed_timeout

    @override
    def setup(self):
        torch.cuda.set_device(self._device)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "cpu:gloo,cuda:nccl", timeout=self._torch_distributed_timeout
            )
        input_tensors = [
            torch.tensor([0], dtype=torch.float32, device=self._device)
            for _ in range(self.world_size)
        ]
        output_tensors = [
            torch.tensor([0], dtype=torch.float32, device=self._device)
            for _ in range(self.world_size)
        ]
        torch.distributed.all_to_all(input_tensors, output_tensors)

    @override
    def cleanup(self):
        with suppress(Exception):
            torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    @override
    def autocast(self) -> AbstractContextManager:
        result = torch.autocast(device_type="cuda", dtype=self._dtype, enabled=True)
        return result

    @override
    def wait_for_everyone(self):
        torch.distributed.barrier()

    @override
    def set_seed(self, seed: int, device_specific: bool = False) -> int:
        """
        Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

        Args:
            seed (`int`):
                The seed to set.
            device_specific (`bool`, *optional*, defaults to `False`):
                Whether to differ the seed on each device slightly with `self.process_index`.
        """
        if device_specific:
            seed += self.global_rank

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        return seed

    @override
    def prepare_models(self, models: PrepareModelsT) -> PrepareModelsT:
        assert all(isinstance(x, nn.Module) for x in models)
        new_models = [nn.parallel.DistributedDataParallel(m) for m in models]
        new_models = cast("PrepareModelsT", new_models)
        return new_models  # type: ignore[return-value]

    @override
    def prepare_train_dataloader(self, train_dataloader: PrepareDataLoaderT) -> PrepareDataLoaderT:
        return train_dataloader

    @override
    def prepare_val_dataloader(self, val_dataloader: PrepareDataLoaderT) -> PrepareDataLoaderT:
        return val_dataloader

    @override
    def compile(self, model: CompileT) -> CompileT:
        result = torch.compile(model)
        result = cast("CompileT", result)
        return result

    @override
    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    @override
    def clip_grad_norm_(
        self,
        parameters: Iterable[torch.Tensor] | torch.Tensor,
        max_norm: float,
        norm_type: float = 2,
    ) -> torch.Tensor:
        result = torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)
        return result

    @override
    def clip_grad_value_(
        self, parameters: Iterable[torch.Tensor] | torch.Tensor, clip_value: float
    ) -> None:
        torch.nn.utils.clip_grad_value_(parameters, clip_value)

    @override
    def save_state(self, path: str | Path) -> None:
        pass

    @override
    def load_state(self, path: str | Path) -> None:
        pass

    @override
    def skip_first_batches(self, dataloader_iterator: Iterator, num_batches: int) -> None:
        for _ in tqdm(
            range(num_batches), desc=f"rank {self._global_rank}: skip_first_batches({num_batches=})"
        ):
            next(dataloader_iterator)

    @property
    @override
    def sync_gradients(self) -> bool:
        return True

    @property
    @override
    def is_main_process(self) -> bool:
        result = self.global_rank == 0
        return result

    @property
    @override
    def is_local_main_process(self) -> bool:
        result = self.local_rank == 0
        return result

    @property
    @override
    def is_last_process(self) -> bool:
        result = self.global_rank == self.world_size - 1
        return result

    @property
    @override
    def is_local_last_process(self) -> bool:
        result = self.local_rank == self.local_world_size - 1
        return result

    @property
    @override
    def local_rank(self) -> int:
        return self._local_rank

    @property
    @override
    def global_rank(self) -> int:
        return self._global_rank

    @property
    @override
    def local_world_size(self) -> int:
        return self._local_world_size

    @property
    @override
    def world_size(self) -> int:
        return self._world_size

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @property
    @override
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def master_addr(self) -> str:
        return self._master_addr

    @property
    def master_port(self) -> int:
        return self._master_port


class BaseRuntime(IRuntime):
    def __init__(self, dtype: torch.dtype = torch.float):
        self._device = torch.device(self.local_rank)
        self._dtype = dtype

    @override
    def setup(self):
        torch.cuda.set_device(self._device)

    @override
    def cleanup(self): ...

    @override
    def autocast(self) -> AbstractContextManager:
        result = torch.autocast(device_type="cuda", dtype=self._dtype, enabled=True)
        return result

    @override
    def wait_for_everyone(self): ...

    @override
    def set_seed(self, seed: int, device_specific: bool = False) -> int:
        """
        Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

        Args:
            seed (`int`):
                The seed to set.
            device_specific (`bool`, *optional*, defaults to `False`):
                Whether to differ the seed on each device slightly with `self.process_index`.
        """
        if device_specific:
            seed += self.global_rank

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        return seed

    @override
    def prepare_models(self, models: PrepareModelsT) -> PrepareModelsT:
        assert all(isinstance(x, nn.Module) for x in models)
        return models

    @override
    def prepare_train_dataloader(self, train_dataloader: PrepareDataLoaderT) -> PrepareDataLoaderT:
        return train_dataloader

    @override
    def prepare_val_dataloader(self, val_dataloader: PrepareDataLoaderT) -> PrepareDataLoaderT:
        return val_dataloader

    @override
    def compile(self, model: CompileT) -> CompileT:
        result = torch.compile(model)
        result = cast("CompileT", result)
        return result

    @override
    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    @override
    def clip_grad_norm_(
        self,
        parameters: Iterable[torch.Tensor] | torch.Tensor,
        max_norm: float,
        norm_type: float = 2,
    ) -> torch.Tensor:
        result = torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)
        return result

    @override
    def clip_grad_value_(
        self, parameters: Iterable[torch.Tensor] | torch.Tensor, clip_value: float
    ) -> None:
        torch.nn.utils.clip_grad_value_(parameters, clip_value)

    @override
    def save_state(self, path: str | Path) -> None:
        pass

    @override
    def load_state(self, path: str | Path) -> None:
        pass

    @override
    def skip_first_batches(self, dataloader_iterator: Iterator, num_batches: int) -> None:
        for _ in tqdm(
            range(num_batches), desc=f"rank {self.global_rank}: skip_first_batches({num_batches=})"
        ):
            next(dataloader_iterator)

    @property
    @override
    def sync_gradients(self) -> bool:
        return True

    @property
    @override
    def is_main_process(self) -> bool:
        result = self.global_rank == 0
        return result

    @property
    @override
    def is_local_main_process(self) -> bool:
        result = self.local_rank == 0
        return result

    @property
    @override
    def is_last_process(self) -> bool:
        result = self.global_rank == self.world_size - 1
        return result

    @property
    @override
    def is_local_last_process(self) -> bool:
        result = self.local_rank == self.local_world_size - 1
        return result

    @property
    @override
    def local_rank(self) -> int:
        return 0

    @property
    @override
    def global_rank(self) -> int:
        return 0

    @property
    @override
    def local_world_size(self) -> int:
        return 1

    @property
    @override
    def world_size(self) -> int:
        return 1

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @property
    @override
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def master_addr(self) -> str | None:
        return None

    @property
    def master_port(self) -> int | None:
        return None


def get_runtime(
    dtype: torch.dtype = torch.bfloat16, torch_distributed_timeout: timedelta | None = None
) -> IRuntime:
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        return NativeDdpRuntime(dtype=dtype, torch_distributed_timeout=torch_distributed_timeout)
    return BaseRuntime(dtype=dtype)
