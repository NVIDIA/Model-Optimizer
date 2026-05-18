# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Streaming dataset that fetches per-sample hidden states from a running ``vllm serve`` over HTTP.

Design notes:
- One epoch only. Spinning a fresh producer thread per __iter__ would re-issue
  every request, which is wasteful; if you want multi-epoch, pre-dump instead.
- Pre-tokenizes the entire (sliced) conversation list at __init__ so we can
  expose __len__ to HF Trainer. For very large jsonls, slice beforehand.
- IterableDataset, not Map: random __getitem__(i) would defeat the point of
  letting vLLM batch many in-flight requests.
- Strict requirement: dataloader_num_workers=0. Multiple workers would each
  spawn their own asyncio thread and hammer the server with duplicate work.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import threading
from typing import Any

import httpx
import torch
from safetensors import safe_open
from torch.utils.data import IterableDataset
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

_SENTINEL = object()


def _tokenize_with_loss_mask(
    tokenizer,
    conversations: list,
    answer_only_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize one conversation and derive its loss mask in the same call.

    Single ``apply_chat_template`` invocation so ``input_ids`` and ``loss_mask`` cannot
    drift. When ``answer_only_loss=True`` the chat template must carry ``{% generation %}``
    tags so the tokenizer can return ``assistant_masks``.
    """
    out = tokenizer.apply_chat_template(
        conversations,
        return_tensors="pt",
        return_dict=True,
        return_assistant_tokens_mask=answer_only_loss,
        add_generation_prompt=False,
    )
    input_ids = out["input_ids"]
    seq_len = input_ids.shape[-1]
    if answer_only_loss:
        mask = out["assistant_masks"]
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long)
        loss_mask = mask.squeeze(0).to(torch.long)
        if loss_mask.shape[0] != seq_len:
            raise RuntimeError(
                f"assistant_masks length {loss_mask.shape[0]} does not match "
                f"input_ids length {seq_len}"
            )
    else:
        loss_mask = torch.ones(seq_len, dtype=torch.long)
    return input_ids, loss_mask


class StreamingHiddenStatesDataset(IterableDataset):
    """IterableDataset that pulls per-sample hidden states from a running ``vllm serve``."""

    def __init__(
        self,
        entries: list[dict],
        tokenizer,
        server_url: str,
        model: str,
        max_seq_len: int = 2048,
        min_seq_len: int = 10,
        answer_only_loss: bool = False,
        prefetch: int = 64,
        request_timeout: float = 600.0,
    ):
        """Eagerly tokenize ``entries`` and filter by length; HTTP fetches happen lazily during iteration."""
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.prefetch = prefetch
        self.request_timeout = request_timeout

        # Eagerly tokenize + filter so we can give HF Trainer a __len__.
        self.samples: list[dict[str, Any]] = []
        skipped, invalid = 0, 0
        for entry in entries:
            cid = entry.get("conversation_id") or entry.get("uuid")
            if cid is None:
                invalid += 1
                continue
            convs = entry.get("messages") or entry.get("conversations")
            if not convs or not isinstance(convs, list):
                invalid += 1
                continue
            input_ids, loss_mask = _tokenize_with_loss_mask(tokenizer, convs, answer_only_loss)
            n = input_ids.shape[-1]
            if not (min_seq_len <= n <= max_seq_len):
                skipped += 1
                continue
            self.samples.append(
                {
                    "cid": str(cid),
                    "token_ids": input_ids.squeeze(0).tolist(),
                    "loss_mask": loss_mask,
                }
            )
        print(
            f"[StreamingHiddenStatesDataset] kept {len(self.samples)} "
            f"(skipped {skipped} by length, {invalid} invalid)"
        )
        if not self.samples:
            raise ValueError("No samples remain after filtering")

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        # Fresh producer per __iter__ call so re-iteration (which shouldn't
        # happen in 1-epoch streaming) at least doesn't deadlock.
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def run():
            try:
                asyncio.run(self._produce(q, stop))
            except Exception as e:
                # Surface to consumer
                q.put(e)
                q.put(_SENTINEL)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        try:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                try:
                    yield item["data"]
                finally:
                    with contextlib.suppress(OSError):
                        os.unlink(item["path"])
        finally:
            stop.set()
            # Drain any leftover items so producer can exit
            with contextlib.suppress(queue.Empty):
                while True:
                    q.get_nowait()

    async def _produce(self, q: queue.Queue, stop: threading.Event):
        sem = asyncio.Semaphore(self.prefetch)
        timeout = httpx.Timeout(self.request_timeout, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:

            async def one(sample):
                async with sem:
                    if stop.is_set():
                        return
                    try:
                        result = await self._fetch(client, sample)
                    except Exception as e:
                        print(f"[streaming] error for {sample['cid']}: {e!r}")
                        return
                    if result is None:
                        return
                    # Blocking put → backpressure when trainer is slow
                    await asyncio.to_thread(q.put, result)

            try:
                await asyncio.gather(*[one(s) for s in self.samples], return_exceptions=False)
            finally:
                q.put(_SENTINEL)

    async def _fetch(self, client: httpx.AsyncClient, sample: dict) -> dict | None:
        r = await client.post(
            f"{self.server_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": sample["token_ids"],
                "max_tokens": 1,
                "temperature": 0,
            },
        )
        r.raise_for_status()
        body = r.json()
        kvt = body.get("kv_transfer_params") or {}
        path = kvt.get("hidden_states_path")
        if path is None:
            print(f"[streaming] no hidden_states_path for {sample['cid']}")
            return None
        return await asyncio.to_thread(self._load_and_format, path, sample)

    def _load_and_format(self, path: str, sample: dict) -> dict:
        with safe_open(path, framework="pt") as f:
            token_ids_tensor = f.get_tensor("token_ids")
            hidden_states_tensor = f.get_tensor("hidden_states")  # [seq, n_layers, hidden]

        output_hidden_states = hidden_states_tensor[:, -1, :]
        aux_hidden_states = hidden_states_tensor[:, :-1, :].reshape(
            hidden_states_tensor.shape[0], -1
        )

        n = token_ids_tensor.shape[0]
        loss_mask = sample["loss_mask"]
        if loss_mask.shape[0] > n:
            loss_mask = loss_mask[:n]
        elif loss_mask.shape[0] < n:
            pad = torch.ones(n - loss_mask.shape[0], dtype=loss_mask.dtype)
            loss_mask = torch.cat([loss_mask, pad], dim=0)

        input_ids = token_ids_tensor.to(torch.int64)
        labels = torch.full_like(input_ids, IGNORE_TOKEN_ID)
        labels[..., :-1] = input_ids[..., 1:]

        return {
            "path": path,
            "data": {
                "input_ids": input_ids,
                "base_model_hidden_states": output_hidden_states,
                "aux_hidden_states": aux_hidden_states,
                "attention_mask": torch.ones_like(input_ids),
                "loss_mask": loss_mask,
                "labels": labels,
            },
        }
