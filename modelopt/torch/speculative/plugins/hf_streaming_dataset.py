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

"""Streaming datasets that fetch per-sample hidden states from a running inference server.

The base class :class:`StreamingHiddenStatesDataset` owns all the backend-/algorithm-
agnostic plumbing: threading, queue, tokenization, the bounded sliding-window
producer, loss_mask alignment, and HTTP-client lifecycle. Concrete subclasses
specialize along two axes:

- **Backend** (how to talk to the server, how to decode the response): override
  :meth:`_fetch`.
- **Algorithm** (how to shape the per-sample dict for the trainer): override
  :meth:`_format`.

:class:`EagleVllmStreamingHiddenStatesDataset` is currently the only concrete
combination (Eagle algorithm × vLLM backend); future combinations (e.g.
Eagle × TRT-LLM, distillation × vLLM) live as sibling subclasses.

Design notes:
- One epoch only. Spinning a fresh producer thread per ``__iter__`` would re-issue
  every request, which is wasteful; if you want multi-epoch, pre-dump instead.
- Lazy: tokenization and HTTP fetch happen during iteration. Memory and startup
  time stay O(prefetch), so this scales to large jsonls without pre-slicing.
- IterableDataset, not Map: random ``__getitem__(i)`` would defeat the point of
  letting the backend batch many in-flight requests.
- Strict requirement: ``dataloader_num_workers=0``. Multiple workers would each
  spawn their own asyncio thread and hammer the server with duplicate work.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import random
import threading
from typing import TYPE_CHECKING

import httpx
import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict, field_validator
from safetensors import safe_open
from torch.utils.data import IterableDataset, get_worker_info
from transformers.trainer_pt_utils import LabelSmoother

if TYPE_CHECKING:
    from collections.abc import Callable

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

_SENTINEL = object()


def _get_rank_world() -> tuple[int, int]:
    """Return ``(rank, world_size)``, falling back to ``(0, 1)`` outside of ``torch.distributed``.

    Single-process runs (tests, single-GPU smoke) hit the fallback and the dataset
    behaves as if it owns the whole corpus. Under ``accelerate launch`` /
    ``torchrun``, ``init_distributed_env`` is called before dataset construction,
    so ``is_initialized()`` is True here.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


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


class StreamingHiddenStatesConfig(BaseModel):
    """Static tuning knobs for :class:`StreamingHiddenStatesDataset`.

    Bundles the rarely-changing settings (loss masking, concurrency, HTTP timeout)
    so the dataset ctor takes only ``entries`` + ``tokenizer`` + this config.
    """

    model_config = ConfigDict(extra="forbid")

    answer_only_loss: bool = False
    prefetch: int = 64
    request_timeout: float = 600.0
    # Same value on every rank — the dataset shuffles with this seed, then stripes
    # by rank, so identical seeds across ranks are required for the partition to
    # be a true partition (no overlap, no gaps).
    seed: int = 0


class StreamingHiddenStatesDataset(IterableDataset):
    """Base class: stream per-sample hidden states from a running inference server.

    Backend- and algorithm-agnostic; subclasses implement :meth:`_fetch` (backend) and
    :meth:`_format` (algorithm).
    """

    config_cls: type[StreamingHiddenStatesConfig] = StreamingHiddenStatesConfig

    def __init__(
        self,
        entries: list[dict],
        tokenizer,
        config: StreamingHiddenStatesConfig | None = None,
    ):
        """Hold a per-rank slice of ``entries``; tokenize and fetch lazily during iteration.

        Length filtering is the caller's responsibility — apply a cheap char-count
        pre-filter (see :func:`estimate_conversation_chars`) on ``entries`` before
        constructing the dataset. Anything that slips through and gets rejected by
        the server is skipped silently.

        DDP sharding: ``entries`` is the *full* corpus on every rank. We shuffle with
        ``config.seed`` (identical on all ranks) and then take ``[rank::world_size]``,
        so the union across ranks is the shuffled corpus, no overlap. ``__len__``
        returns the per-rank length, matching PyTorch's ``IterableDataset`` convention
        (HF Trainer / DataLoader does not multiply by world_size).
        """
        if not entries:
            raise ValueError("entries is empty")
        self.tokenizer = tokenizer
        self.config = config if config is not None else self.config_cls()

        rank, world = _get_rank_world()
        indices = list(range(len(entries)))
        random.Random(self.config.seed).shuffle(indices)
        indices = indices[rank::world]
        if not indices:
            raise ValueError(
                f"rank {rank}/{world} got 0 entries after sharding "
                f"(corpus has {len(entries)}); need len(entries) >= world_size"
            )
        self.entries = [entries[i] for i in indices]
        print(
            f"[{type(self).__name__}] rank {rank}/{world}: "
            f"{len(self.entries)}/{len(entries)} entries"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        # IterableDataset with DataLoader workers > 0 would spawn one asyncio loop
        # per worker, each issuing the full request set — silent Nx duplication
        # against the server. Fail loud instead.
        if get_worker_info() is not None:
            raise RuntimeError(
                f"{type(self).__name__} requires dataloader_num_workers=0; "
                "multiple workers would each spawn an asyncio loop and duplicate requests."
            )
        # Fresh producer per __iter__ call so re-iteration (which shouldn't
        # happen in 1-epoch streaming) at least doesn't deadlock.
        q: queue.Queue = queue.Queue(maxsize=self.config.prefetch)
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
                    cleanup = item.get("cleanup")
                    if cleanup is not None:
                        with contextlib.suppress(Exception):
                            cleanup()
        finally:
            stop.set()
            # Drain any leftover items so producer can exit
            with contextlib.suppress(queue.Empty):
                while True:
                    q.get_nowait()

    async def _produce(self, q: queue.Queue, stop: threading.Event):
        """Stream entries through a sliding window of at most ``prefetch`` in-flight tasks."""
        timeout = httpx.Timeout(self.config.request_timeout, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            pending: set[asyncio.Task] = set()
            entries_iter = iter(self.entries)
            exhausted = False
            try:
                while not stop.is_set():
                    while len(pending) < self.config.prefetch and not exhausted:
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
        """Tokenize -> backend fetch -> align -> format -> enqueue. Errors are logged and skipped."""
        if stop.is_set():
            return
        sample = await asyncio.to_thread(self._tokenize_entry, entry)
        if sample is None:
            return
        try:
            fetched = await self._fetch(client, sample)
        except Exception as e:
            print(f"[streaming] error for {sample['cid']}: {e!r}")
            return
        if fetched is None:
            return
        token_ids, hidden_states, loss_mask, cleanup = fetched
        data = self._format(token_ids, hidden_states, loss_mask)
        # Blocking put -> backpressure when trainer is slow.
        await asyncio.to_thread(q.put, {"data": data, "cleanup": cleanup})

    def _tokenize_entry(self, entry: dict) -> dict | None:
        """Tokenize a single entry. Returns None for entries missing ``cid`` or ``messages``."""
        cid = entry.get("conversation_id") or entry.get("uuid")
        convs = entry.get("messages") or entry.get("conversations")
        if cid is None or not convs or not isinstance(convs, list):
            return None
        input_ids, loss_mask = _tokenize_with_loss_mask(
            self.tokenizer, convs, self.config.answer_only_loss
        )
        return {
            "cid": str(cid),
            "token_ids": input_ids.squeeze(0).tolist(),
            "loss_mask": loss_mask,
        }

    async def _fetch(
        self,
        client: httpx.AsyncClient,
        sample: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable[[], None] | None] | None:
        """Backend hook: send the request and decode hidden states.

        Override in subclass.

        Returns:
            ``(token_ids, hidden_states, loss_mask, cleanup)`` where ``token_ids`` is
            the tokens the server actually saw, ``hidden_states`` is shaped
            ``(seq, n_captured_layers, hidden)``, ``loss_mask`` is aligned to
            ``token_ids`` (the backend is responsible for reconciling any mismatch
            between ``sample["loss_mask"]`` and the server's token count), and
            ``cleanup`` is an optional callable invoked after the consumer is done
            with this sample (e.g. to unlink a scratch file). Return ``None`` to
            skip the sample.
        """
        raise NotImplementedError("Subclasses must implement _fetch")

    def _format(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Algorithm hook: shape one sample's tensors into the dict the trainer expects.

        Override in subclass.

        Args:
            token_ids: ``LongTensor`` of shape ``(seq,)`` — what the server actually tokenized.
            hidden_states: tensor of shape ``(seq, n_captured_layers, hidden)``.
            loss_mask: ``LongTensor`` of shape ``(seq,)``, aligned to ``token_ids``.
        """
        raise NotImplementedError("Subclasses must implement _format")


class EagleVllmStreamingHiddenStatesConfig(StreamingHiddenStatesConfig):
    """Adds vLLM endpoint info on top of :class:`StreamingHiddenStatesConfig`."""

    server_url: str
    model: str

    @field_validator("server_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


class EagleVllmStreamingHiddenStatesDataset(StreamingHiddenStatesDataset):
    """Eagle (algorithm) × vLLM (backend).

    Talks to a ``vllm serve`` instance configured with the
    ``ExampleHiddenStatesConnector`` KV-transfer connector (the server dumps captured
    layers to a per-request safetensors file under ``shared_storage_path`` and
    returns the path via ``kv_transfer_params.hidden_states_path``). Expects vLLM
    to capture ``aux_layers + [final_layer]`` along ``hidden_states.shape[1]``.
    """

    config_cls = EagleVllmStreamingHiddenStatesConfig

    def __init__(
        self,
        entries: list[dict],
        tokenizer,
        config: EagleVllmStreamingHiddenStatesConfig,
    ):
        """Same as the base; ``config`` must include ``server_url`` and ``model``."""
        super().__init__(entries=entries, tokenizer=tokenizer, config=config)
        self.config: EagleVllmStreamingHiddenStatesConfig = config

    async def _fetch(
        self,
        client: httpx.AsyncClient,
        sample: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable[[], None] | None] | None:
        r = await client.post(
            f"{self.config.server_url}/v1/completions",
            json={
                "model": self.config.model,
                "prompt": sample["token_ids"],
                "max_tokens": 1,
                "temperature": 0,
            },
        )
        r.raise_for_status()
        body = r.json()
        path = (body.get("kv_transfer_params") or {}).get("hidden_states_path")
        if path is None:
            print(f"[streaming] no hidden_states_path for {sample['cid']}")
            return None
        token_ids, hidden_states, cleanup = await asyncio.to_thread(self._load_safetensors, path)
        # vLLM may capture a different number of positions than we tokenized (e.g. the
        # decode step from ``max_tokens=1``, or BOS handling edge cases). Realign here.
        loss_mask = self._align_loss_mask(sample["loss_mask"], token_ids.shape[0])
        return token_ids, hidden_states, loss_mask, cleanup

    @staticmethod
    def _load_safetensors(
        path: str,
    ) -> tuple[torch.Tensor, torch.Tensor, Callable[[], None]]:
        with safe_open(path, framework="pt") as f:
            token_ids = f.get_tensor("token_ids")
            hidden_states = f.get_tensor("hidden_states")  # [seq, n_layers, hidden]

        def cleanup():
            with contextlib.suppress(OSError):
                os.unlink(path)

        return token_ids, hidden_states, cleanup

    @staticmethod
    def _align_loss_mask(loss_mask: torch.Tensor, n: int) -> torch.Tensor:
        """Trim or right-pad ``loss_mask`` to length ``n`` to match what vLLM tokenized to."""
        if loss_mask.shape[0] > n:
            return loss_mask[:n]
        if loss_mask.shape[0] < n:
            pad = torch.ones(n - loss_mask.shape[0], dtype=loss_mask.dtype)
            return torch.cat([loss_mask, pad], dim=0)
        return loss_mask

    def _format(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base_model_hidden_states = hidden_states[:, -1, :]
        aux_hidden_states = hidden_states[:, :-1, :].reshape(hidden_states.shape[0], -1)

        input_ids = token_ids.to(torch.int64)
        labels = torch.full_like(input_ids, IGNORE_TOKEN_ID)
        labels[..., :-1] = input_ids[..., 1:]

        return {
            "input_ids": input_ids,
            "base_model_hidden_states": base_model_hidden_states,
            "aux_hidden_states": aux_hidden_states,
            "attention_mask": torch.ones_like(input_ids),
            "loss_mask": loss_mask,
            "labels": labels,
        }
