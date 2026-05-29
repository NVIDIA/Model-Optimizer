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
combination (Eagle algorithm × vLLM backend); future combinations live as
sibling subclasses.

Requires ``dataloader_num_workers=0``: multiple workers would each spawn their
own asyncio loop and issue duplicate requests against the server.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import random
import threading
from pathlib import Path
from typing import TypedDict

import httpx
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from safetensors import safe_open
from torch.utils.data import IterableDataset, get_worker_info
from transformers import TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother

from modelopt.torch.utils import distributed as dist_utils
from modelopt.torch.utils import print_rank_0, warn_rank_0

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

_SENTINEL = object()


def _tokenize_with_loss_mask(
    tokenizer,
    conversations: list,
    answer_only_loss: bool,
    max_seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize one conversation and derive its loss mask in the same call.

    Single ``apply_chat_template`` invocation so ``input_ids`` and ``loss_mask`` cannot
    drift. When ``answer_only_loss=True`` the chat template must carry ``{% generation %}``
    tags so the tokenizer can return ``assistant_masks``. When ``max_seq_len`` is set,
    truncation is delegated to the tokenizer so ids and assistant_masks are truncated
    in lockstep.
    """
    out = tokenizer.apply_chat_template(
        conversations,
        return_tensors="pt",
        return_dict=True,
        return_assistant_tokens_mask=answer_only_loss,
        add_generation_prompt=False,
        truncation=max_seq_len is not None,
        max_length=max_seq_len,
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
    prefetch: int = Field(default=64, ge=1)
    request_timeout: float = Field(default=600.0, gt=0)
    # Token-level cap applied during tokenization (right-truncation). Must hold
    # ``max_seq_len <= vllm.max_model_len``. ``None`` disables truncation.
    max_seq_len: int | None = None
    # Must be identical on every rank — the dataset shuffles with this seed then
    # stripes by rank, so equal seeds are required for the partition to be disjoint.
    seed: int = 0
    # Circuit breaker: raise after this many consecutive _fetch failures so a dead
    # server doesn't silently drain the corpus.
    fail_after_consecutive_skips: int = Field(default=16, ge=1)


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

    def __init__(
        self,
        entries: list[dict],
        tokenizer,
        config: StreamingConfig | None = None,
    ):
        """Hold the *full* corpus on every rank; fetch lazily, rank 0 only.

        DDP sharding is delegated to Accelerate's ``DataLoaderDispatcher``: rank 0
        consumes the dataset and broadcasts each batch; non-zero ranks rely on
        :meth:`__iter__`'s rank guard. The corpus is held in full on every rank --
        the dispatcher reads only rank 0's stream, so sharding here would just
        shrink that view. Shuffling with ``config.seed`` runs on every rank so
        the order is reproducible regardless of which rank ends up fetching.

        Args:
            entries: Untokenized per-sample dicts from the input jsonl. Schema is
                subclass-defined (see :meth:`_tokenize_entry`); passed through to :meth:`_fetch`.
            tokenizer: HF tokenizer; used for client-side tokenization and the
                server/client loss-mask alignment in :meth:`_fetch`.
            config: Tuning knobs (prefetch, timeout, seed, ...); defaults to
                ``self.config_cls()``. See :class:`StreamingConfig`.
        """
        if not entries:
            raise ValueError("entries is empty")
        self.tokenizer = tokenizer
        self.config = config if config is not None else self.config_cls()
        # One-shot, consumed by the next __iter__.
        self._resume_skip = 0

        indices = list(range(len(entries)))
        random.Random(self.config.seed).shuffle(indices)
        self.entries = [entries[i] for i in indices]
        rank, world = dist_utils.rank(), dist_utils.size()
        print_rank_0(
            f"[{type(self).__name__}] rank {rank}/{world}: "
            f"holds {len(self.entries)} entries (full corpus; rank 0 fetches)"
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

    @staticmethod
    def _verify_accelerate_dispatcher() -> None:
        """Raise if Accelerate is initialized for DDP with ``dispatch_batches=False``.

        Best-effort: no-op when Accelerate isn't installed/initialized or in single-process.
        """
        try:
            from accelerate.state import AcceleratorState
        except ImportError:
            return
        if not AcceleratorState._shared_state:
            return
        state = AcceleratorState()
        if getattr(state, "num_processes", 1) <= 1:
            return
        # Field moved to ``dataloader_config`` in newer Accelerate; check both.
        dispatch = getattr(state, "dispatch_batches", None)
        if dispatch is None:
            dl_cfg = getattr(state, "dataloader_config", None)
            if dl_cfg is not None:
                dispatch = getattr(dl_cfg, "dispatch_batches", None)
        if dispatch is False:
            raise RuntimeError(
                "StreamingDataset requires Accelerate's DataLoaderDispatcher "
                "(dispatch_batches=True); got False — non-zero ranks would receive no data."
            )

    def __iter__(self):
        # IterableDataset with DataLoader workers > 0 would spawn one asyncio loop
        # per worker, each issuing the full request set — silent Nx duplication
        # against the server. Fail loud instead.
        if get_worker_info() is not None:
            raise RuntimeError(
                f"{type(self).__name__} requires dataloader_num_workers=0; "
                "multiple workers would each spawn an asyncio loop and duplicate requests."
            )
        # Without dispatch_batches the rank-0 guard below would silently starve
        # non-zero ranks; fail loud instead.
        self._verify_accelerate_dispatcher()
        # Only rank 0 fetches; non-zero ranks receive batches via the dispatcher's broadcast.
        if dist_utils.rank() != 0:
            return
        # Fresh producer per __iter__ call so re-iteration (which shouldn't
        # happen in 1-epoch streaming) at least doesn't deadlock.
        q: queue.Queue = queue.Queue(maxsize=self.config.prefetch)
        stop = threading.Event()
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

        Counter is local (single writer); ``_process`` tasks report outcome via return value.
        The circuit breaker has *batch-level* (not per-task) granularity: when
        ``asyncio.wait(FIRST_COMPLETED)`` returns several tasks in the same loop turn,
        ``consecutive_skips`` reflects set-iteration order over ``done`` -- sufficient
        for "detect a dead server" but not strict temporal ordering.

        Args:
            q: Bounded queue drained by :meth:`__iter__`; full queue backpressures fetching.
            stop: Set by the consumer to request shutdown; checked between samples.
            entries: Resume-adjusted slice of ``self.entries`` to fetch this iteration.
        """
        timeout = httpx.Timeout(self.config.request_timeout, connect=10.0)
        threshold = self.config.fail_after_consecutive_skips
        consecutive_skips = 0
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
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        outcome = task.result()  # re-raises unexpected errors
                        if outcome is True:
                            consecutive_skips = 0
                        elif outcome is False:
                            consecutive_skips += 1
                        # None -> entry unfit pre-fetch; server not at fault
                    if consecutive_skips >= threshold:
                        raise RuntimeError(
                            f"{consecutive_skips} consecutive _fetch failures "
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
    ) -> bool | None:
        """Tokenize -> fetch -> format -> enqueue.

        Returns True on enqueue, False on fetch failure (bumps breaker), None
        when the entry is unfit pre-fetch (no breaker effect).
        """
        if stop.is_set():
            return None
        sample = await asyncio.to_thread(self._tokenize_entry, entry)
        if sample is None:
            return None
        try:
            fetched = await self._fetch(client, sample)
        except Exception as e:
            warn_rank_0(f"[streaming] error for {sample['cid']}: {e!r}")
            return False
        if fetched is None:
            return False
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
        return True

    def _tokenize_entry(self, entry: dict) -> dict | None:
        """Tokenize a single entry.

        Returns ``None`` for entries missing ``cid`` / ``messages``, or when
        right-truncation to ``max_seq_len`` drops the entire supervised span
        (``answer_only_loss`` mode with the assistant turn at the tail).
        """
        cid = entry.get("conversation_id") or entry.get("uuid")
        convs = entry.get("messages") or entry.get("conversations")
        if cid is None or not convs or not isinstance(convs, list):
            return None
        input_ids, loss_mask = _tokenize_with_loss_mask(
            self.tokenizer,
            convs,
            self.config.answer_only_loss,
            max_seq_len=self.config.max_seq_len,
        )
        if int(loss_mask.sum()) == 0:
            return None
        return {
            "cid": str(cid),
            "token_ids": input_ids.squeeze(0).tolist(),
            "loss_mask": loss_mask,
        }

    async def _fetch(self, client: httpx.AsyncClient, sample: dict) -> dict | None:
        """Backend hook: send the request and decode the server's response.

        Override in subclass. Any scratch resources (per-request files, mmap'd
        buffers) must be released before returning.

        Args:
            client: Shared async HTTP client owned by :meth:`_produce`.
            sample: :meth:`_tokenize_entry` output:
                ``{"cid": str, "token_ids": list[int], "loss_mask": LongTensor[seq]}``.

        Returns:
            Dict carrying at least the keys declared by :attr:`fetch_payload_cls`,
            or ``None`` to skip this sample (counts toward the circuit breaker).
        """
        raise NotImplementedError("Subclasses must implement _fetch")

    def _format(self, fetched: dict) -> dict[str, torch.Tensor]:
        """Algorithm hook: shape the fetched dict into the trainer's per-sample batch.

        Override in subclass.

        Args:
            fetched: :meth:`_fetch` output; the keys declared by
                :attr:`fetch_payload_cls` are guaranteed to be present.

        Returns:
            Tensor dict consumed directly by the trainer's ``model.forward(**batch)``.
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
    # Allowlist for ``hidden_states_path`` returned by the server. Must match the
    # connector's ``shared_storage_path``; out-of-tree paths are rejected.
    shared_storage_root: str

    @field_validator("server_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("shared_storage_root")
    @classmethod
    def _resolve_root(cls, v: str) -> str:
        return str(Path(v).resolve())


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
            warn_rank_0(f"[streaming] no hidden_states_path for {sample['cid']}")
            return None
        if not self._path_under_root(path):
            warn_rank_0(
                f"[streaming] path outside shared_storage_root for {sample['cid']}: {path!r}"
            )
            return None
        token_ids, hidden_states = await asyncio.to_thread(self._load_safetensors, path)
        # Contract: the server tokenization is the client's pre-tokenized prompt
        # verbatim, plus at most one decode-step token at the tail (from
        # ``max_tokens=1``). Anything else (e.g. server-side BOS prepend, chat
        # templating, tokenizer drift) means ``loss_mask`` no longer aligns to
        # the supervised positions, so fail loudly rather than silently train
        # on misaligned tokens.
        client_ids = torch.as_tensor(sample["token_ids"], dtype=token_ids.dtype)
        n = client_ids.shape[0]
        if token_ids.shape[0] not in (n, n + 1) or not torch.equal(token_ids[:n], client_ids):
            raise RuntimeError(
                f"server token_ids drift for {sample['cid']}: "
                f"client_len={n}, server_len={token_ids.shape[0]}; "
                "the server must consume the pre-tokenized prompt verbatim "
                "(no BOS prepend or chat templating)"
            )
        loss_mask = self._align_loss_mask(sample["loss_mask"], token_ids.shape[0])
        return {
            "token_ids": token_ids,
            "hidden_states": hidden_states,
            "loss_mask": loss_mask,
        }

    def _path_under_root(self, path: str) -> bool:
        try:
            resolved = Path(path).resolve()
        except (OSError, ValueError):
            return False
        return resolved.is_relative_to(self.config.shared_storage_root)

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
        """Trim or right-pad ``loss_mask`` to length ``n``.

        Caller guarantees the server-side token sequence is a strict prefix of
        the client-side sequence (or the same length), so the only realignment
        ever needed is at the tail — pad the optional decode-step position
        with 0 (no client-side label).
        """
        if loss_mask.shape[0] > n:
            return loss_mask[:n]
        if loss_mask.shape[0] < n:
            pad = torch.zeros(n - loss_mask.shape[0], dtype=loss_mask.dtype)
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
    """Fast-forward :class:`StreamingDataset` past consumed samples on resume.

    Dispatcher pulls a *global* batch per micro-step, hence the ``world_size`` factor.
    Requires ``training_args.ignore_data_skip=True``; round-trips only when
    ``world_size`` and ``config.seed`` match the original run.
    """

    def on_train_begin(self, args, state, control, train_dataloader=None, **kwargs):
        """Push the skip count into the dataset when resuming mid-training."""
        if state.global_step <= 0 or train_dataloader is None:
            return
        ds = train_dataloader.dataset
        if not hasattr(ds, "set_resume_position"):
            return
        if not getattr(args, "ignore_data_skip", False):
            raise RuntimeError(
                "StreamingResumeCallback requires ignore_data_skip=True to avoid "
                "double-skipping on resume."
            )
        consumed = (
            state.global_step
            * args.per_device_train_batch_size
            * dist_utils.size()
            * args.gradient_accumulation_steps
        )
        ds.set_resume_position(consumed)
        print_rank_0(
            f"[StreamingResumeCallback] resuming at global_step={state.global_step}; "
            f"skipping {consumed} entries"
        )
