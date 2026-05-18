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
- Lazy: tokenization and HTTP fetch happen during iteration. Memory and startup
  time stay O(prefetch), so this scales to large jsonls without pre-slicing.
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

import httpx
import torch
from safetensors import safe_open
from torch.utils.data import IterableDataset
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

_SENTINEL = object()


def estimate_conversation_chars(entry: dict) -> int:
    """Sum the character count across all messages in an entry.

    Cheap proxy for tokenized length when paired with a chars-per-token ratio. Returns
    0 for entries with no recognized messages list, so callers can filter them out too.
    """
    convs = entry.get("messages") or entry.get("conversations") or []
    if not isinstance(convs, list):
        return 0
    return sum(len(m.get("content", "") or "") for m in convs if isinstance(m, dict))


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
        answer_only_loss: bool = False,
        prefetch: int = 64,
        request_timeout: float = 600.0,
    ):
        """Hold a reference to ``entries``; tokenize and fetch lazily during iteration.

        Length filtering is the caller's responsibility — apply a cheap char-count
        pre-filter (see :func:`estimate_conversation_chars`) on ``entries`` before
        constructing the dataset. Anything that slips through and exceeds vllm's
        ``max_model_len`` will be rejected by the server and skipped silently.
        """
        if not entries:
            raise ValueError("entries is empty")
        self.entries = entries
        self.tokenizer = tokenizer
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.answer_only_loss = answer_only_loss
        self.prefetch = prefetch
        self.request_timeout = request_timeout
        print(f"[StreamingHiddenStatesDataset] {len(entries)} entries")

    def __len__(self) -> int:
        return len(self.entries)

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
        """Stream entries through a sliding window of at most ``prefetch`` in-flight tasks."""
        timeout = httpx.Timeout(self.request_timeout, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            pending: set[asyncio.Task] = set()
            entries_iter = iter(self.entries)
            exhausted = False
            try:
                while not stop.is_set():
                    while len(pending) < self.prefetch and not exhausted:
                        try:
                            entry = next(entries_iter)
                        except StopIteration:
                            exhausted = True
                            break
                        pending.add(asyncio.create_task(self._process(client, entry, q, stop)))
                    if not pending:
                        break
                    _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                q.put(_SENTINEL)

    async def _process(
        self,
        client: httpx.AsyncClient,
        entry: dict,
        q: queue.Queue,
        stop: threading.Event,
    ) -> None:
        """Tokenize -> POST -> load safetensors -> enqueue. Malformed or vllm-rejected samples are skipped."""
        if stop.is_set():
            return
        sample = await asyncio.to_thread(self._tokenize_entry, entry)
        if sample is None:
            return
        try:
            result = await self._fetch(client, sample)
        except Exception as e:
            print(f"[streaming] error for {sample['cid']}: {e!r}")
            return
        if result is None:
            return
        # Blocking put -> backpressure when trainer is slow.
        await asyncio.to_thread(q.put, result)

    def _tokenize_entry(self, entry: dict) -> dict | None:
        """Tokenize a single entry. Returns None for entries missing ``cid`` or ``messages``."""
        cid = entry.get("conversation_id") or entry.get("uuid")
        convs = entry.get("messages") or entry.get("conversations")
        if cid is None or not convs or not isinstance(convs, list):
            return None
        input_ids, loss_mask = _tokenize_with_loss_mask(
            self.tokenizer, convs, self.answer_only_loss
        )
        return {
            "cid": str(cid),
            "token_ids": input_ids.squeeze(0).tolist(),
            "loss_mask": loss_mask,
        }

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
        return await asyncio.to_thread(self._load_and_format, path, sample["loss_mask"])

    def _load_and_format(self, path: str, loss_mask: torch.Tensor) -> dict:
        with safe_open(path, framework="pt") as f:
            token_ids_tensor = f.get_tensor("token_ids")
            hidden_states_tensor = f.get_tensor("hidden_states")  # [seq, n_layers, hidden]

        output_hidden_states = hidden_states_tensor[:, -1, :]
        aux_hidden_states = hidden_states_tensor[:, :-1, :].reshape(
            hidden_states_tensor.shape[0], -1
        )

        n = token_ids_tensor.shape[0]
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
