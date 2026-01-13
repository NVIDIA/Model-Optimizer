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

"""DataLoader utilities for language model training and validation."""

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Protocol, TypeVar

import datasets
import torch
import torch.distributed
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from modelopt.torch.puzzletron.tools.logger import mprint
from modelopt.torch.puzzletron.utils.data.dataset import ConstantLengthDataset


def collate_none_fn(
    batch, *, collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None
):
    return None


collate_fn_map_with_none_support = {**default_collate_fn_map, type(None): collate_none_fn}
collate_fn_with_none_support = partial(collate, collate_fn_map=collate_fn_map_with_none_support)


class LoadDatasetFn(Protocol):
    def __call__(
        self, dataset_path: str, content_field: str, keep_in_memory: bool = False
    ) -> Mapping[str, Dataset]: ...


def load_from_disk_fn(
    dataset_path: str, content_field: str, keep_in_memory: bool = False
) -> Mapping[str, Dataset]:
    return datasets.load_from_disk(dataset_path, keep_in_memory=keep_in_memory)


def load_streaming_fn(
    dataset_path: str, content_field: str, keep_in_memory: bool = False
) -> Mapping[str, Dataset]:
    dataset = datasets.load_dataset(
        dataset_path,
        streaming=True,
        features=datasets.Features(
            {
                content_field: datasets.Value(dtype="string"),
            }
        ),
        keep_in_memory=keep_in_memory,
    )

    return dataset


def create_validation_dataloader(
    accelerator: Accelerator | None,
    seed: int,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    dataset: str | Mapping[str, Dataset],
    content_field: str,
    fim_rate: float,
    fim_spm_rate: float,
    micro_batch_size: int,
    eval_samples: int | None = None,
    load_dataset_fn: LoadDatasetFn = load_from_disk_fn,
    dataset_name: str = "__auto__",
    keep_in_memory: bool = False,
    source_datasets_to_discard: Sequence[str] = (),
    bos_rate: float = 1.0,
    varlen: bool = True,
    shuffle_seed: int | None = None,
):
    if accelerator is None:
        accelerator = Printer()

    if accelerator.is_main_process:
        if isinstance(dataset, str):
            dataset = load_dataset_fn(dataset, content_field, keep_in_memory)

        if isinstance(dataset, datasets.Dataset | torch.utils.data.Dataset):
            valid_data = dataset
            mprint(
                "#### Path to specific dataset was given (not DatasetDict), taking it as-is ####"
            )
        else:
            assert isinstance(dataset, datasets.DatasetDict)
            if dataset_name == "__auto__":
                val_split_options = []
                for val_key_prefix in ("val", "test"):
                    if len(val_split_options) == 0:
                        val_split_options = [
                            split
                            for split in dataset  # DatasetDict is dict-like and supports direct iteration
                            if split.lower().startswith(val_key_prefix)
                        ]
                assert len(val_split_options) == 1, (
                    f"Expected exactly one validation split, got {val_split_options=} ({dataset.keys()=})"
                )
                val_split = val_split_options[0]
                mprint(f"Inferred validation split automatically: '{val_split}'")
            else:
                val_split = dataset_name
                mprint(f"Validation split explicitly chosen: '{val_split}'")
            valid_data = dataset[val_split]

        if shuffle_seed is not None:
            mprint(f"Shuffling with {shuffle_seed=}")
            valid_data = valid_data.shuffle(seed=shuffle_seed)

        valid_dataset = ConstantLengthDataset(
            tokenizer,
            valid_data,
            infinite=False,
            seq_length=block_size * micro_batch_size if varlen else block_size,
            content_field=content_field,
            fim_rate=fim_rate,
            fim_spm_rate=fim_spm_rate,
            seed=seed,
            source_datasets_to_discard=source_datasets_to_discard,
            bos_rate=bos_rate,
            # return_cu_seqlens=varlen,
            # seqlen_cap=block_size if varlen else None
        )
        if varlen and eval_samples is not None:
            eval_samples = eval_samples // micro_batch_size
        val_offloaded_dataset = realize_dataset_in_memory(valid_dataset, eval_samples)

        valid_data_len = len(val_offloaded_dataset)
        mprint(f"num validation examples = {valid_data_len}")
    else:
        val_offloaded_dataset = None

    if not isinstance(accelerator, Printer):
        obj_list = [val_offloaded_dataset]
        torch.distributed.broadcast_object_list(obj_list)
        val_offloaded_dataset = obj_list[0]

    # let accelerate prepare to handle distributed sampling
    val_dataloader = DataLoader(
        val_offloaded_dataset,
        batch_size=1 if varlen else micro_batch_size,
        pin_memory=True,
        collate_fn=collate_fn_with_none_support,
    )

    return val_dataloader


def realize_dataset_in_memory(dataset: IterableDataset, eval_samples: int | None) -> list[dict]:
    tqdm_desc = f"realize_dataset_in_memory({eval_samples=})"
    if eval_samples is None:
        offloaded_dataset = list(tqdm(dataset, desc=tqdm_desc))
    else:
        val_iter = iter(dataset)
        offloaded_dataset = [next(val_iter) for _ in tqdm(range(eval_samples), desc=tqdm_desc)]
    return offloaded_dataset


TensorT = TypeVar("TensorT", bound=torch.Tensor)


@torch.no_grad()
def create_padded_tensor(
    tensor: TensorT, desired_shape: Sequence[int], padding_value: float = 0
) -> TensorT:
    if tensor.shape == torch.Size(desired_shape):
        return tensor

    padded_tensor = torch.full(
        desired_shape, fill_value=padding_value, dtype=tensor.dtype, device=tensor.device
    )
    indices = torch.where(torch.ones_like(tensor, dtype=torch.bool))
    padded_tensor[indices] = tensor.view(-1)
    return padded_tensor


class Printer:
    is_main_process = True
    process_index = None

    @staticmethod
    def print(*args, **kwargs) -> None:
        print(*args, **kwargs)
