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

"""Map-style datasets that fetch per-sample hidden states from a running inference server.

This is the streaming sibling of :class:`OfflineSupervisedDataset`: instead of
reading a pre-dumped ``.pt`` file in ``__getitem__``, it fetches the per-sample
hidden states from a live inference server over HTTP. It is a plain
``torch.utils.data.Dataset`` (map-style), so DDP sharding is handled the standard
way -- HF Trainer wraps it in a ``DistributedSampler`` and each rank's DataLoader
calls ``__getitem__`` only for that rank's indices. Each rank therefore fetches
**only its own shard** (no rank-0 funnel, no broadcast); aggregate read bandwidth
scales with the number of trainer ranks.

Fetch concurrency comes from the DataLoader's ``num_workers`` (each worker process
issues one blocking request at a time); there is no in-process producer thread.
Keep ``num_workers`` modest and bounded so the per-server in-flight request count
(``ranks-hitting-a-server x num_workers``) stays near the server's ``max_num_seqs``
-- flooding a cold NVFP4 MoE server can stall a worker past vLLM's execute-model
timeout and kill EngineCore.

The base class :class:`StreamingDataset` owns the backend-/algorithm-agnostic
plumbing: tokenization, the resample-on-failure ``__getitem__`` loop, the
consecutive-failure circuit breaker, and loss_mask alignment. Concrete subclasses
specialize along two axes:

- **Backend** (how to talk to the server, how to decode the response): override
  :meth:`_fetch`.
- **Algorithm** (how to shape the per-sample dict for the trainer): override
  :meth:`_format`.

:class:`EagleVllmStreamingDataset` is currently the only concrete combination
(Eagle algorithm x vLLM backend); future combinations live as sibling subclasses.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import TypedDict

import httpx
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from safetensors import SafetensorError, safe_open
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother

from modelopt.torch.utils import print_rank_0, warn_rank_0

__all__ = [
    "EagleFetchPayload",
    "EagleVllmStreamingConfig",
    "EagleVllmStreamingDataset",
    "StreamingConfig",
    "StreamingDataset",
]

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# The vLLM connector writes the safetensors file asynchronously (writer thread pool)
# and returns its path before the write is durably visible, so an immediate read can
# race the writer. Retry the open with linear backoff until the file lands
# (worst case ~_READ_RETRIES * (_READ_RETRIES+1)/2 * _READ_BACKOFF s).
_READ_RETRIES = 10
_READ_BACKOFF = 0.05  # seconds

# Errors from ``_fetch`` that are genuinely transient (server overloaded / connection
# reset / timeout, or the safetensors writer race) and so count against the circuit
# breaker and trigger a resample. Anything else -- notably the ``RuntimeError`` raised
# on server token drift, or a programming/contract bug (``ValueError``/``KeyError``) --
# is a real fault and propagates instead of being silently masked as a fetch miss.
_TRANSIENT_FETCH_ERRORS = (httpx.HTTPError, OSError, SafetensorError)


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
        tokenize=True,
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

    Bundles the rarely-changing settings (loss masking, HTTP timeout) so the dataset
    ctor takes only ``entries`` + ``tokenizer`` + this config.
    """

    model_config = ConfigDict(extra="forbid")

    answer_only_loss: bool = False
    request_timeout: float = Field(default=600.0, gt=0)
    # Token-level cap applied during tokenization (right-truncation). Must hold
    # ``max_seq_len <= vllm.max_model_len``. ``None`` disables truncation.
    max_seq_len: int | None = None
    # Circuit breaker: raise after this many consecutive _fetch failures (per worker
    # process) so a dead server doesn't silently resample the whole corpus.
    fail_after_consecutive_skips: int = Field(default=16, ge=1)


class StreamingDataset(Dataset):
    """Base class: map-style dataset that streams per-sample hidden states from a server.

    Backend- and algorithm-agnostic; subclasses implement :meth:`_fetch` (backend) and
    :meth:`_format` (algorithm). The dict shape exchanged between them is the
    algorithm-level contract, declared as a ``TypedDict`` in :attr:`fetch_payload_cls`
    and validated against the actual ``_fetch`` output on every sample.

    ``__getitem__`` must always return a valid sample for the sampler's index, so it
    resamples forward through the corpus on an unfit entry or a fetch failure rather
    than skipping (a skip would shrink the batch and desync DDP).
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
        """Hold the full corpus; fetch lazily, per index, in ``__getitem__``.

        DDP sharding is handled by HF Trainer's ``DistributedSampler``: each rank's
        DataLoader requests only its own indices, so each rank fetches only its
        shard. The corpus order is left as given -- the sampler shuffles indices
        (seeded by ``training_args.seed``), so no shuffle is needed here.

        Args:
            entries: Untokenized per-sample dicts from the input jsonl. Schema is
                subclass-defined (see :meth:`_tokenize_entry`); passed to :meth:`_fetch`.
            tokenizer: HF tokenizer; used for client-side tokenization and the
                server/client loss-mask alignment in :meth:`_fetch`.
            config: Tuning knobs (timeout, answer_only_loss, ...); defaults to
                ``self.config_cls()``. See :class:`StreamingConfig`.
        """
        if not entries:
            raise ValueError("entries is empty")
        self.tokenizer = tokenizer
        self.config = config if config is not None else self.config_cls()
        # Materialize to a plain list so DataLoader worker processes fork it cheaply.
        self.entries = list(entries)
        # Per-process consecutive-failure counter for the circuit breaker. Reset to 0
        # on every successful fetch; tripped only by fetch failures (not unfit entries).
        self._consecutive_fail = 0
        print_rank_0(f"[{type(self).__name__}] map-style dataset over {len(self.entries)} entries")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Tokenize -> fetch -> format the sample at ``idx``, resampling on miss.

        Always returns a valid sample. An unfit entry (tokenization yields nothing) or
        a fetch failure causes a forward probe to the next index; fetch failures bump
        the circuit breaker, which raises once ``fail_after_consecutive_skips`` is hit.
        """
        n = len(self.entries)
        for offset in range(n):
            entry = self.entries[(idx + offset) % n]
            sample = self._tokenize_entry(entry)
            if sample is None:
                continue  # entry unfit pre-fetch; server not at fault, try the next one
            try:
                fetched = self._fetch(sample)
            except _TRANSIENT_FETCH_ERRORS as e:
                # Transport/IO miss: count against the circuit breaker and resample.
                # Contract violations and bugs are not caught here -- they propagate.
                warn_rank_0(f"[streaming] fetch error for {sample['cid']}: {e!r}")
                fetched = None
            if fetched is None:
                self._consecutive_fail += 1
                if self._consecutive_fail >= self.config.fail_after_consecutive_skips:
                    raise RuntimeError(
                        f"{self._consecutive_fail} consecutive _fetch failures in "
                        f"{type(self).__name__}; server likely down."
                    )
                continue  # resample forward
            self._consecutive_fail = 0
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
            return self._format(fetched)
        raise RuntimeError(
            f"{type(self).__name__}: no fetchable sample found in the entire corpus "
            f"({n} entries) starting at index {idx}."
        )

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

    def _fetch(self, sample: dict) -> dict | None:
        """Backend hook: send the request and decode the server's response.

        Override in subclass. Synchronous (called from a DataLoader worker). Any
        scratch resources (per-request files, mmap'd buffers) must be released before
        returning.

        Args:
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

    # One or more vLLM endpoints; fetches round-robin across them so a single fetcher
    # can spread load over several server replicas. Accepts a list or a single
    # (optionally comma-separated) string.
    server_urls: list[str]
    model: str
    # Allowlist for ``hidden_states_path`` returned by the server. Must match (or be a
    # parent of) the connector's ``shared_storage_path``; out-of-tree paths are rejected.
    shared_storage_root: str

    @field_validator("server_urls", mode="before")
    @classmethod
    def _normalize_urls(cls, v):
        if isinstance(v, str):
            v = v.split(",")
        urls = [u.strip().rstrip("/") for u in v if u and str(u).strip()]
        if not urls:
            raise ValueError("server_urls must contain at least one non-empty URL")
        return urls

    @field_validator("shared_storage_root")
    @classmethod
    def _resolve_root(cls, v: str) -> str:
        return str(Path(v).resolve())


class EagleVllmStreamingDataset(StreamingDataset):
    """Eagle (algorithm) x vLLM (backend).

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
        """Same as the base; ``config`` must include ``server_urls`` and ``model``."""
        super().__init__(entries=entries, tokenizer=tokenizer, config=config)
        self.config: EagleVllmStreamingConfig = config

    def _client(self) -> httpx.Client:
        """Lazily build a per-process HTTP client and round-robin cursor.

        DataLoader workers are forked processes; httpx connection pools must not be
        shared across a fork, so each process gets its own client (and its own
        round-robin cursor over ``server_urls``), keyed by PID.
        """
        pid = os.getpid()
        if getattr(self, "_client_pid", None) != pid:
            self._http = httpx.Client(
                timeout=httpx.Timeout(self.config.request_timeout, connect=10.0)
            )
            self._client_pid = pid
            self._rr = 0
        return self._http

    def _next_url(self) -> str:
        """Round-robin the next server URL (per-process cursor)."""
        urls = self.config.server_urls
        url = urls[self._rr % len(urls)]
        self._rr += 1
        return url

    def _fetch(self, sample: dict) -> EagleFetchPayload | None:
        client = self._client()
        url = self._next_url()
        r = client.post(
            f"{url}/v1/completions",
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
        token_ids, hidden_states = self._load_safetensors(path)
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

        Retries past the writer race (see ``_READ_RETRIES``): a missing file means
        the write hasn't started; a ``SafetensorError`` means it's mid-write. Both
        clear once the writer finishes, so back off and retry before giving up.
        """
        for attempt in range(_READ_RETRIES):
            try:
                with safe_open(path, framework="pt") as f:
                    token_ids = f.get_tensor("token_ids")
                    hidden_states = f.get_tensor("hidden_states")  # [seq, n_layers, hidden]
                with contextlib.suppress(OSError):
                    os.unlink(path)
                return token_ids, hidden_states
            except (FileNotFoundError, SafetensorError):  # noqa: PERF203 -- retry-on-race loop
                if attempt == _READ_RETRIES - 1:
                    raise
                time.sleep(_READ_BACKOFF * (attempt + 1))
        # Unreachable (the last attempt above re-raises); guards _READ_RETRIES < 1.
        raise RuntimeError(f"_load_safetensors exhausted {_READ_RETRIES} retries for {path}")

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
