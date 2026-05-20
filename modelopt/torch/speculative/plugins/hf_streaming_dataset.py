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

The base class :class:`StreamingDataset` owns all the backend-/algorithm-
agnostic plumbing: threading, queue, tokenization, the bounded sliding-window
producer, loss_mask alignment, and HTTP-client lifecycle. Concrete subclasses
specialize along two axes:

- **Backend** (how to talk to the server, how to decode the response): override
  :meth:`_fetch`.
- **Algorithm** (how to shape the per-sample dict for the trainer): override
  :meth:`_format`.

:class:`EagleVllmStreamingDataset` is currently the only concrete
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
from typing import TypedDict

import httpx
import torch
from pydantic import BaseModel, ConfigDict, field_validator
from safetensors import safe_open
from torch.utils.data import IterableDataset, get_worker_info
from transformers import TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother

from modelopt.torch.utils import distributed as dist_utils

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


class StreamingConfig(BaseModel):
    """Static tuning knobs for :class:`StreamingDataset`.

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
    # Circuit breaker: raise after this many consecutive _fetch failures (network
    # errors, 5xx, malformed responses). High enough to absorb transient blips,
    # low enough that a dead server doesn't silently drain the corpus.
    fail_after_consecutive_skips: int = 16


class StreamingDataset(IterableDataset):
    """Base class: stream per-sample hidden states from a running inference server.

    Backend- and algorithm-agnostic; subclasses implement :meth:`_fetch` (backend) and
    :meth:`_format` (algorithm). The dict shape exchanged between them is the
    algorithm-level contract, declared as a ``TypedDict`` in :attr:`fetch_payload_cls`
    and validated against the actual ``_fetch`` output on every sample.
    """

    config_cls: type[StreamingConfig] = StreamingConfig
    # Algorithm subclasses set this to a TypedDict declaring the keys their _format
    # reads from the _fetch output. When set, base class validates _fetch's return
    # value carries all required keys (fail-loud on the first sample).
    fetch_payload_cls: type | None = None
    # One-shot offset honored at the start of the next ``__iter__``. Set via
    # :meth:`set_resume_position` (typically by :class:`StreamingResumeCallback`).
    _resume_skip: int = 0

    def __init__(
        self,
        entries: list[dict],
        tokenizer,
        config: StreamingConfig | None = None,
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

        rank, world = dist_utils.rank(), dist_utils.size()
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

    def set_resume_position(self, skip: int) -> None:
        """Drop the first ``skip`` entries on the next ``__iter__`` without fetching.

        One-shot; cleared once iteration starts. Used by
        :class:`StreamingResumeCallback` on HF Trainer checkpoint resume so the
        server is not re-queried for already-consumed samples.
        """
        self._resume_skip = skip

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
        self._consecutive_skips = 0  # per-iter; tripped in _process, read in _produce
        skip = self._resume_skip
        self._resume_skip = 0  # one-shot
        entries = self.entries[skip:] if skip else self.entries

        def run():
            try:
                asyncio.run(self._produce(q, stop, entries))
            except Exception as e:
                q.put(e)  # surface to consumer
            finally:
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
                yield item
        finally:
            stop.set()
            # Drain any leftover items so producer can exit
            with contextlib.suppress(queue.Empty):
                while True:
                    q.get_nowait()

    async def _produce(self, q: queue.Queue, stop: threading.Event, entries):
        """Stream ``entries`` through a sliding window of at most ``prefetch`` in-flight tasks.

        Sentinel is enqueued by the caller (``__iter__.run``) after this returns,
        so an exception raised here (e.g. circuit breaker) reaches the consumer
        before the loop's terminating sentinel.
        """
        timeout = httpx.Timeout(self.config.request_timeout, connect=10.0)
        threshold = self.config.fail_after_consecutive_skips
        async with httpx.AsyncClient(timeout=timeout) as client:
            pending: set[asyncio.Task] = set()
            entries_iter = iter(entries)
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
                    if self._consecutive_skips >= threshold:
                        raise RuntimeError(
                            f"{self._consecutive_skips} consecutive _fetch failures "
                            f"in {type(self).__name__}; server likely down."
                        )
            finally:
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

    async def _process(
        self,
        client: httpx.AsyncClient,
        entry: dict,
        q: queue.Queue,
        stop: threading.Event,
    ) -> None:
        """Tokenize -> backend fetch -> align -> format -> enqueue.

        Per-sample _fetch errors (and ``None`` returns) are logged and skipped,
        bumping ``self._consecutive_skips`` for the producer's circuit breaker.
        Successful enqueue resets the counter.
        """
        if stop.is_set():
            return
        sample = await asyncio.to_thread(self._tokenize_entry, entry)
        if sample is None:
            return
        try:
            fetched = await self._fetch(client, sample)
        except Exception as e:
            print(f"[streaming] error for {sample['cid']}: {e!r}")
            self._consecutive_skips += 1
            return
        if fetched is None:
            self._consecutive_skips += 1
            return
        if self.fetch_payload_cls is not None:
            # ``__required_keys__`` is a TypedDict runtime attribute mypy doesn't
            # track on ``type``; the assignment site guarantees it's a TypedDict.
            required: frozenset[str] = self.fetch_payload_cls.__required_keys__  # type: ignore[attr-defined]
            missing = required - set(fetched)
            if missing:
                raise RuntimeError(
                    f"{type(self).__name__}._fetch missing required keys {missing}; "
                    f"{self.fetch_payload_cls.__name__} requires "
                    f"{set(required)}, got {set(fetched)}"
                )
        data = self._format(fetched)
        # Blocking put -> backpressure when trainer is slow.
        await asyncio.to_thread(q.put, data)
        self._consecutive_skips = 0

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

    async def _fetch(self, client: httpx.AsyncClient, sample: dict) -> dict | None:
        """Backend hook: send the request and decode the server's response into a dict.

        Override in subclass.

        Returns:
            A dict whose required keys are declared by :attr:`fetch_payload_cls`
            (the algorithm-level contract). Any scratch resources (per-request
            files, mmap'd buffers) should be released before returning. Return
            ``None`` to skip the sample.

        Note:
            If a future backend needs to return zero-copy mmap views (tensor
            lifetime tied to the scratch file), reintroduce a deferred-cleanup
            hook here: extend the return value with an optional ``cleanup``
            callable, have ``_process`` enqueue ``{"data": ..., "cleanup": ...}``,
            and run ``cleanup()`` in ``__iter__``'s ``finally`` after the consumer
            yields. The current eager-cleanup contract was chosen because every
            backend so far materializes a real copy in ``_fetch``.
        """
        raise NotImplementedError("Subclasses must implement _fetch")

    def _format(self, fetched: dict) -> dict[str, torch.Tensor]:
        """Algorithm hook: shape the fetched dict into the trainer's expected dict.

        Override in subclass. The keys ``fetched`` is guaranteed to carry are
        declared in :attr:`fetch_payload_cls`.
        """
        raise NotImplementedError("Subclasses must implement _format")


class EagleFetchPayload(TypedDict):
    """The dict shape every Eagle backend must produce in :meth:`_fetch`.

    Fields:
        token_ids:     ``LongTensor`` of shape ``(seq,)`` — what the server tokenized.
        hidden_states: tensor of shape ``(seq, n_captured_layers, hidden)``.
        loss_mask:     ``LongTensor`` of shape ``(seq,)``, aligned to ``token_ids``.
    """

    token_ids: torch.Tensor
    hidden_states: torch.Tensor
    loss_mask: torch.Tensor


class EagleVllmStreamingConfig(StreamingConfig):
    """Adds vLLM endpoint info on top of :class:`StreamingConfig`."""

    server_url: str
    model: str

    @field_validator("server_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


class EagleVllmStreamingDataset(StreamingDataset):
    """Eagle (algorithm) × vLLM (backend).

    Talks to a ``vllm serve`` instance configured with the
    ``ExampleHiddenStatesConnector`` KV-transfer connector (the server dumps captured
    layers to a per-request safetensors file under ``shared_storage_path`` and
    returns the path via ``kv_transfer_params.hidden_states_path``). Expects vLLM
    to capture ``aux_layers + [final_layer]`` along ``hidden_states.shape[1]``.
    """

    config_cls = EagleVllmStreamingConfig
    fetch_payload_cls = EagleFetchPayload

    def __init__(
        self,
        entries: list[dict],
        tokenizer,
        config: EagleVllmStreamingConfig,
    ):
        """Same as the base; ``config`` must include ``server_url`` and ``model``."""
        super().__init__(entries=entries, tokenizer=tokenizer, config=config)
        self.config: EagleVllmStreamingConfig = config

    async def _fetch(self, client: httpx.AsyncClient, sample: dict) -> EagleFetchPayload | None:
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
        token_ids, hidden_states = await asyncio.to_thread(self._load_safetensors, path)
        # vLLM may capture a different number of positions than we tokenized (e.g. the
        # decode step from ``max_tokens=1``, or BOS handling edge cases). Realign here.
        loss_mask = self._align_loss_mask(sample["loss_mask"], token_ids.shape[0])
        return {
            "token_ids": token_ids,
            "hidden_states": hidden_states,
            "loss_mask": loss_mask,
        }

    @staticmethod
    def _load_safetensors(path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Read tensors into CPU memory then unlink the scratch file.

        ``safe_open(..., framework="pt").get_tensor`` materializes an independent
        torch Tensor (not a view into the mmap'd file), so it is safe to unlink
        right after the ``with`` block exits.
        """
        with safe_open(path, framework="pt") as f:
            token_ids = f.get_tensor("token_ids")
            hidden_states = f.get_tensor("hidden_states")  # [seq, n_layers, hidden]
        with contextlib.suppress(OSError):
            os.unlink(path)
        return token_ids, hidden_states

    @staticmethod
    def _align_loss_mask(loss_mask: torch.Tensor, n: int) -> torch.Tensor:
        """Trim or right-pad ``loss_mask`` to length ``n`` to match what vLLM tokenized to."""
        if loss_mask.shape[0] > n:
            return loss_mask[:n]
        if loss_mask.shape[0] < n:
            pad = torch.ones(n - loss_mask.shape[0], dtype=loss_mask.dtype)
            return torch.cat([loss_mask, pad], dim=0)
        return loss_mask

    def _format(self, fetched: EagleFetchPayload) -> dict[str, torch.Tensor]:
        token_ids = fetched["token_ids"]
        hidden_states = fetched["hidden_states"]
        loss_mask = fetched["loss_mask"]

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


class StreamingResumeCallback(TrainerCallback):
    """Wire HF Trainer's checkpoint resume into :class:`StreamingDataset`.

    On train begin, if ``state.global_step > 0`` (resuming from a checkpoint),
    tell the dataset to fast-forward past samples already consumed in the prior
    run so the server is not re-queried for them.

    Per-rank skip = ``global_step * per_device_train_batch_size * gradient_accumulation_steps``.

    Requires ``training_args.ignore_data_skip=True`` (else HF Trainer would also
    skip, doubling the offset — ``main.py`` sets this for streaming mode).
    Resume only round-trips correctly when ``world_size`` and ``config.seed``
    match the original run.
    """

    def on_train_begin(self, args, state, control, train_dataloader=None, **kwargs):
        """Push per-rank skip count into the dataset if resuming mid-training."""
        if state.global_step <= 0 or train_dataloader is None:
            return
        ds = train_dataloader.dataset
        if not hasattr(ds, "set_resume_position"):
            return
        consumed = (
            state.global_step * args.per_device_train_batch_size * args.gradient_accumulation_steps
        )
        ds.set_resume_position(consumed)
        if dist_utils.is_master():
            print(
                f"[StreamingResumeCallback] resuming at global_step={state.global_step}; "
                f"skipping {consumed} entries per rank"
            )
