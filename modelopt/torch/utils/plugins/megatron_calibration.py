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
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from megatron.core import parallel_state as mpu
from tqdm import tqdm

from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader
from modelopt.torch.utils.vlm_dataset_utils import get_vlm_dataset_dataloader

from .megatron_generate import cp_split_sequence, megatron_prefill

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, ProcessorMixin

__all__ = [
    "get_megatron_calibration_dataloader",
    "get_megatron_calibration_forward_loop",
    "get_megatron_vlm_calibration_forward_loop",
]


def get_megatron_calibration_dataloader(
    tokenizer: "PreTrainedTokenizerBase",
    *,
    dataset_name: str | list[str] = "cnn_dailymail",
    batch_size: int = 1,
    num_samples: int | list[int] = 512,
    seq_length: int = 512,
    device: torch.device | str | None = "cuda",
    apply_chat_template: bool = True,
    pack: bool = False,
) -> torch.utils.data.DataLoader:
    """Build a DP-sharded calibration dataloader for Megatron-Core models.

    Each batch is a dict with at least ``input_ids`` and ``attention_mask`` tensors
    on ``device``. The dataloader is suitable as the ``data_loader`` argument to
    ``mtq.auto_quantize`` or any other API that iterates batches directly.

    All kwargs are forwarded to :func:`get_dataset_dataloader`; ``seq_length``
    maps to that function's ``max_sample_length``.
    """
    # Deepcopy before mutating pad_token so the caller's tokenizer isn't silently changed.
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer = copy.deepcopy(tokenizer)
        tokenizer.pad_token = tokenizer.eos_token

    # Shard calibration data across DP ranks; amax is max-reduced across DP inside ``mtq``.
    dp_size = mpu.get_data_parallel_world_size()
    return get_dataset_dataloader(
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

    Iterates a dataloader built via :func:`get_megatron_calibration_dataloader`
    and drives a logits-free prefill pass through the model so activation hooks
    fire on every layer. All kwargs are forwarded 1:1 — see
    :func:`get_megatron_calibration_dataloader` for their semantics.

    Returns:
        A ``forward_loop(model)`` callable to pass into ``mtq.quantize``,
        ``mtp.prune``, or other such APIs.
    """
    dataloader = get_megatron_calibration_dataloader(
        tokenizer,
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_samples=num_samples,
        seq_length=seq_length,
        device=device,
        apply_chat_template=apply_chat_template,
        pack=pack,
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


def get_megatron_vlm_calibration_forward_loop(
    model: torch.nn.Module,
    processor: "ProcessorMixin",
    *,
    dataset_name: str = "scienceqa",
    batch_size: int = 1,
    num_samples: int = 512,
    device: torch.device | str | None = "cuda",
    subsets: list[str] | None = None,
    max_shards: int | None = None,
) -> Callable[[torch.nn.Module], None]:
    """Build a Megatron-Core **multimodal** calibration ``forward_loop`` for a VLM.

    This iterates image-text pairs and drives the **full VLM** forward so the language model's activation
    statistics are conditioned on the merged vision tokens. The returned loop ignores the model
    argument passed to it and always runs ``model`` (the full VLM captured here) -- this lets the
    caller quantize only the inner ``language_model`` while still calibrating it on real multimodal activations.

    Args:
        model: The full VLM (e.g. the ``Qwen3VLModel`` wrapper) to run forward for calibration.
        processor: The HF processor (e.g. ``AutoProcessor``) used to encode the image-text pairs.
        dataset_name: VLM calibration dataset name (see ``vlm_dataset_utils``).
        batch_size: Calibration batch size.
        num_samples: Number of calibration samples.
        device: Device to move the encoded tensors to.
        subsets: Subsets to use (only for ``nemotron_vlm_dataset_v2``; ignored otherwise).
        max_shards: Max media tar shards to download per subset (only for ``nemotron_vlm_dataset_v2``;
            ignored otherwise). Caps the download for large multi-shard subsets.

    Returns:
        A ``forward_loop(model)`` callable to pass into ``mtq.quantize``, ``mtp.prune``, or other such APIs.
    """
    # Shard the image-text data across DP ranks
    dp_size = mpu.get_data_parallel_world_size()
    dataloader = get_vlm_dataset_dataloader(
        dataset_name=dataset_name,
        processor=processor,
        subsets=subsets,
        max_shards=max_shards,
        batch_size=batch_size,
        num_samples=num_samples,
        device=device,
        require_image=True,
        dp_size=dp_size,
        dp_rank=mpu.get_data_parallel_rank(),
    )
    # tqdm total: the streaming/sharded dataloader has no __len__.
    total_batches = math.ceil(math.ceil(num_samples / dp_size) / batch_size)

    def _forward_loop(_model: torch.nn.Module | None = None) -> None:
        # CP would have to split each sequence across ranks, but the multimodal forward merges vision
        # embeddings into the sequence (the vision tower is not CP-split), so the text-style sequence
        # split would misalign them. Use DP/TP/PP, or run text-only calibration (a text
        # --calib_dataset_name uses get_megatron_calibration_forward_loop, which supports CP).
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size != 1:
            raise RuntimeError(
                f"get_megatron_vlm_calibration_forward_loop requires CP=1, got "
                f"context_parallel_world_size={cp_size}. Run calibration without CP."
            )
        for batch in tqdm(dataloader, total=total_batches, disable=not dist.is_master()):
            megatron_prefill(
                model,
                batch["input_ids"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                image_sizes=batch.get("image_sizes"),
                skip_return_logits=True,
            )

    return _forward_loop
