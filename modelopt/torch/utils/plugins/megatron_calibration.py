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

"""Shared calibration forward-loop builder for Megatron-Core models."""

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from megatron.core import parallel_state as mpu
from tqdm import tqdm

from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

from .megatron_generate import cp_split_sequence, megatron_prefill

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = ["get_megatron_calibration_forward_loop"]


def get_megatron_calibration_forward_loop(
    tokenizer: "PreTrainedTokenizerBase",
    *,
    dataset_name: str | list[str] = "cnn_dailymail",
    batch_size: int = 1,
    num_samples: int | list[int] = 512,
    seq_length: int = 512,
    device: torch.device | str | None = "cuda",
    apply_chat_template: bool = True,
    pack: bool = False,
) -> Callable[[torch.nn.Module], None]:
    """Build a Megatron-Core calibration ``forward_loop(model)``.

    Iterates a packed dataloader built via ``get_dataset_dataloader(pack=True)``
    and drives a logits-free prefill pass through the model so activation hooks
    fire on every layer. All kwargs except ``seq_length`` are forwarded
    1:1 — see :func:`get_dataset_dataloader` for their semantics. ``seq_length``
    maps to that function's ``max_sample_length``.

    Returns:
        A ``forward_loop(model)`` callable to pass into ``mtq.quantize``,
        ``mtp.prune``, or other such APIs.
    """
    # Deepcopy before mutating pad_token so the caller's tokenizer isn't silently changed.
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer = copy.deepcopy(tokenizer)
        tokenizer.pad_token = tokenizer.eos_token

    # Shard calibration data across DP ranks; amax is max-reduced across DP inside ``mtq``.
    dp_size = mpu.get_data_parallel_world_size()
    dataloader = get_dataset_dataloader(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=num_samples,
        max_sample_length=seq_length,
        device=device,
        apply_chat_template=apply_chat_template,
        pack=pack,
        distributed=dp_size > 1,
        sampler_kwargs={
            "num_replicas": dp_size,
            "rank": mpu.get_data_parallel_rank(),
            "shuffle": False,
        },
    )

    def _forward_loop(model: torch.nn.Module) -> None:
        cp_size = mpu.get_context_parallel_world_size()
        cp_group = mpu.get_context_parallel_group()
        for sample in tqdm(dataloader, disable=not dist.is_master()):
            if cp_size > 1:
                input_ids, position_ids = cp_split_sequence(sample["input_ids"], cp_group)
                megatron_prefill(
                    model, input_ids, position_ids=position_ids, skip_return_logits=True
                )
            else:
                megatron_prefill(model, sample["input_ids"], skip_return_logits=True)

    return _forward_loop
