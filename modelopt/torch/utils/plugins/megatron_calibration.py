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

"""Shared calibration forward-loop builder for Megatron-Core models.

Drives a prefill pass through the model over a calibration dataset, producing the
``forward_loop`` callable that ``mtq.quantize`` / ``mtp.prune`` / ``mtq.calibrate``
expect.

Picks the best primitives from each existing path:
- ``get_dataset_dataloader`` for dataset surface (HF registry + JSONL auto-detection,
  multi-source blending, one-sample-per-row with batch-padding)
- Per-row trim + EOS-at-row-end before forward, matching MBridge's
  ``GPTSFTDataset(add_eos=True)`` semantics.
- ``megatron_prefill(skip_return_logits=True)`` for the forward primitive — skips
  returning logits / loss compute compared to the legacy training-step path; the LM
  head still runs and activation hooks still fire on every layer.

Context parallelism: this loop targets CP=1. Splitting a calibration sequence across
CP ranks doesn't help (calibration sequences are short and we want the same activations
on every rank), and ``megatron_prefill`` builds its causal mask / position_ids over the
local tensor length, which would silently produce wrong activations under CP>1.
"""

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from megatron.core import parallel_state as mpu
from tqdm import tqdm

from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

from .megatron_generate import megatron_prefill

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = ["get_megatron_calibration_forward_loop"]


def get_megatron_calibration_forward_loop(
    tokenizer: "PreTrainedTokenizerBase",
    *,
    dataset_name: str | list[str] = "cnn_dailymail",
    num_samples: int | list[int] = 512,
    seq_length: int = 512,
    batch_size: int = 1,
    apply_chat_template: bool = True,
    device: torch.device | str | None = "cuda",
) -> Callable[[torch.nn.Module], None]:
    """Build a Megatron-Core calibration ``forward_loop(model)``.

    The returned callable iterates a one-sample-per-row dataloader, gathers the real
    tokens of each row via boolean indexing on the dataloader's ``attention_mask`` (so
    left- and right-padded tokenizers both work), forces EOS at the trimmed row's last
    position, and drives a logits-free prefill pass through the model so activation
    hooks fire. Padding positions are kept out of the forward entirely — they would
    otherwise be hooked into calibration statistics regardless of attention masking.

    Behavior mirrors M-Bridge's ``GPTSFTDataset(add_eos=True)`` for the
    truncated-row case: each calibration row is one document, capped at ``seq_length``,
    with the last position overwritten by EOS as an explicit end-of-document marker.
    Under-cap rows lose their natural last content token in exchange for the marker —
    a deliberate trade-off so hooks see a consistent EOS signal at row end.

    Args:
        tokenizer: HuggingFace tokenizer.
        dataset_name: Dataset key (see :func:`get_supported_datasets`), a ``.jsonl``
            path, or a list mixing the two.
        num_samples: Number of raw samples to draw.
        seq_length: Truncation / padding target per row.
        batch_size: Calibration micro-batch size. With variable-length samples and a
            mix of short and long, the forward loop iterates per-row when the batch
            contains any padding — so true batched throughput requires uniform-length
            samples (or all-long samples where every row is truncated to ``seq_length``).
        apply_chat_template: Forwarded to :func:`get_dataset_dataloader`.
        device: Forwarded to :func:`get_dataset_dataloader`.

    Returns:
        A ``forward_loop(model)`` callable to pass into ``mtq.quantize``,
        ``mtp.prune``, or other such APIs.
    """
    # Deepcopy before mutating pad_token so the caller's tokenizer isn't silently changed.
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer = copy.deepcopy(tokenizer)
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = get_dataset_dataloader(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=num_samples,
        max_sample_length=seq_length,
        device=device,
        apply_chat_template=apply_chat_template,
    )

    eos_id = getattr(tokenizer, "eos_token_id", None)

    # Sort samples by real length descending so front batches are mostly full-length
    # (no padding → batched forward). Back batches end up all-short and fall to the
    # per-row path. Calibration statistics are order-invariant aggregates (amax /
    # channel importance), so this re-ordering doesn't affect quality, just throughput.
    all_ids: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    for sample in dataloader:
        all_ids.append(sample["input_ids"])
        all_masks.append(sample.get("attention_mask", torch.ones_like(sample["input_ids"])))
    cat_ids = torch.cat(all_ids, dim=0)
    cat_masks = torch.cat(all_masks, dim=0)
    # Pre-compute per-row real lengths once on CPU; sort + per-batch padding check both
    # read from this CPU tensor, avoiding a CPU-GPU sync inside the forward hot loop.
    lengths_cpu = cat_masks.sum(dim=-1).cpu()
    sort_idx = torch.argsort(lengths_cpu, descending=True)
    sorted_ids = cat_ids[sort_idx]
    sorted_masks = cat_masks[sort_idx]
    sorted_lengths = lengths_cpu[sort_idx]
    seq_len = sorted_ids.shape[-1]

    def _forward_loop(model: torch.nn.Module) -> None:
        # ``megatron_prefill`` builds its causal mask + position_ids over the local input
        # tensor length, so splitting a calibration sequence across CP ranks would silently
        # produce wrong activations. Calibration sequences are short enough that CP doesn't
        # help anyway — fail loud rather than ship broken statistics.
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size != 1:
            raise RuntimeError(
                f"get_megatron_calibration_forward_loop requires CP=1, got "
                f"context_parallel_world_size={cp_size}. Run calibration without CP."
            )
        n = sorted_ids.shape[0]
        for start in tqdm(range(0, n, batch_size), disable=not dist.is_master()):
            ids = sorted_ids[start : start + batch_size]
            mask = sorted_masks[start : start + batch_size]
            lens = sorted_lengths[start : start + batch_size]
            # If any row in this batch has padding, forward each row at its real length so
            # calibration hooks don't fire on padding positions — padding tokens contribute
            # their (constant-ish) hidden states to the activation statistics, causing a
            # substantial MMLU regression on prune calibration.
            if bool((lens < seq_len).any()):
                for b in range(ids.shape[0]):
                    # Boolean-mask gather works for both left- and right-padded sequences:
                    # we extract exactly the real tokens regardless of which side the
                    # padding sits on.
                    row = ids[b][mask[b].bool()].unsqueeze(0).clone()
                    if row.shape[1] < 1:
                        continue
                    if eos_id is not None:
                        # Overwrites the row's last real token with EOS — matches the
                        # truncated-row case of M-Bridge's ``GPTSFTDataset(add_eos=True)``.
                        # For under-cap rows, this loses one content token in exchange for
                        # an explicit end-of-document marker that hooks see during prune
                        # importance scoring.
                        row[0, -1] = eos_id
                    megatron_prefill(model, row, skip_return_logits=True)
            else:
                if eos_id is not None:
                    ids = ids.clone()
                    ids[:, -1] = eos_id
                megatron_prefill(model, ids, skip_return_logits=True)

    return _forward_loop
