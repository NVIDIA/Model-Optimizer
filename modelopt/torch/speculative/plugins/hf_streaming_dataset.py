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
hidden states from a live inference server over NIXL RDMA (a small HTTP sidecar
only carries the transfer metadata). It is a plain
``torch.utils.data.Dataset`` (map-style), so DDP sharding is handled the standard
way -- HF Trainer wraps it in a ``DistributedSampler`` and each rank's DataLoader
calls ``__getitem__`` only for that rank's indices. Each rank therefore fetches
**only its own shard** (no rank-0 funnel, no broadcast); aggregate read bandwidth
scales with the number of trainer ranks.

Fetch concurrency comes from the DataLoader's ``num_workers`` (each worker process
issues one blocking request at a time); there is no in-process producer thread.
Keep ``num_workers`` modest so the per-server in-flight request count
(``ranks-hitting-a-server x num_workers``) stays near the server's ``max_num_seqs``;
flooding a cold server can stall a worker past vLLM's execute-model timeout.

The base class :class:`StreamingDataset` owns the backend-/algorithm-agnostic
plumbing (tokenization, the resample-on-failure ``__getitem__`` loop, the
consecutive-failure circuit breaker, loss_mask alignment); subclasses override
:meth:`_fetch` (backend) and :meth:`_format` (algorithm).
"""

from __future__ import annotations

import base64
import os
import time
from typing import TypedDict

import httpx
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother

from modelopt.torch.utils import print_rank_0, warn_rank_0
from modelopt.torch.utils.loss_mask import get_loss_mask_recovery

__all__ = [
    "EagleFetchPayload",
    "EagleVllmStreamingConfig",
    "EagleVllmStreamingDataset",
    "StreamingConfig",
    "StreamingDataset",
]

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# Errors from ``_fetch`` that are genuinely transient (server overloaded / connection
# reset / timeout) and so count against the circuit breaker and trigger a resample.
# Anything else -- notably the ``RuntimeError`` raised on server token drift, or a
# programming/contract bug (``ValueError``/``KeyError``) -- is a real fault and
# propagates instead of being silently masked as a fetch miss.
_TRANSIENT_FETCH_ERRORS = (httpx.HTTPError, OSError)


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

    ``assistant_masks`` requires a fast tokenizer (it needs ``char_to_token``). For
    tokenizers without it, the mask is rebuilt from token ids via a registered
    model-specific recovery (see ``modelopt.torch.utils.loss_mask``) if one matches.
    """
    recovery = None
    if answer_only_loss and not getattr(tokenizer, "is_fast", False):
        recovery = get_loss_mask_recovery(tokenizer)
    out = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        return_assistant_tokens_mask=answer_only_loss and recovery is None,
        add_generation_prompt=False,
        truncation=max_seq_len is not None,
        max_length=max_seq_len,
    )
    input_ids = out["input_ids"]
    seq_len = input_ids.shape[-1]
    if not answer_only_loss:
        loss_mask = torch.ones(seq_len, dtype=torch.long)
    elif recovery is not None:
        loss_mask = recovery.compute(tokenizer, input_ids[0])
    else:
        mask = out["assistant_masks"]
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long)
        loss_mask = mask.squeeze(0).to(torch.long)
    if loss_mask.shape[0] != seq_len:
        raise RuntimeError(
            f"loss_mask length {loss_mask.shape[0]} does not match input_ids length {seq_len}"
        )
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
    # Required here (the base field is optional): the RDMA recv buffer is pre-sized and
    # registered once from max_seq_len, so it must be known before the first fetch.
    max_seq_len: int = Field(gt=0)

    @field_validator("server_urls", mode="before")
    @classmethod
    def _normalize_urls(cls, v):
        if isinstance(v, str):
            v = v.split(",")
        urls = [u.strip().rstrip("/") for u in v if u and str(u).strip()]
        if not urls:
            raise ValueError("server_urls must contain at least one non-empty URL")
        return urls


class EagleVllmStreamingDataset(StreamingDataset):
    """Eagle (algorithm) x vLLM (backend).

    Talks to a ``vllm serve`` instance configured with the ``RdmaHiddenStatesConnector``
    KV-transfer connector: the server captures the hidden states into a pre-registered
    pinned pool and returns a per-request id via ``kv_transfer_params.hs_req_id``; this
    dataset pulls the captured layers straight from the server's pool over NIXL RDMA (no
    disk round-trip), keyed by that id through the connector's HTTP sidecar. Expects vLLM
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

    def _next_url(self) -> str:
        """Round-robin the next server URL (per-process cursor)."""
        urls = self.config.server_urls
        url = urls[self._rr % len(urls)]
        self._rr += 1
        return url

    # ---- per-worker NIXL agent + one-time remote-agent registration ----
    def _rdma(self):
        pid = os.getpid()
        if getattr(self, "_nixl_pid", None) != pid:
            from nixl._api import nixl_agent, nixl_agent_config

            self._nixl = nixl_agent(f"hs-trainer-{pid}", nixl_agent_config(backends=["UCX"]))
            self._nixl_pid = pid
            self._remote_by_host: dict = {}
            self._recv = None
            self._recv_reg = None  # NIXL registration handle for self._recv
            _wi = torch.utils.data.get_worker_info()
            self._rr = int(os.environ.get("RANK", "0")) * (_wi.num_workers if _wi else 1) + (
                _wi.id if _wi else 0
            )
            self._http_rdma = httpx.Client(
                timeout=httpx.Timeout(self.config.request_timeout, connect=10.0)
            )
        return self._nixl

    def _remote(self, host, port):
        if host not in self._remote_by_host:
            m = self._http_rdma.get(f"http://{host}:{port}/meta").json()["agent_metadata"]
            self._remote_by_host[host] = self._nixl.add_remote_agent(base64.b64decode(m))
        return self._remote_by_host[host]

    def _fetch(self, sample: dict) -> EagleFetchPayload | None:
        """Fetch one sample's hidden states from the server over NIXL RDMA.

        POST the pre-tokenized prompt (``max_tokens=1``) to get the connector's
        ``hs_req_id``, poll the sidecar ``/desc`` for the transfer descriptor, then
        RDMA-READ the captured slot straight into a per-worker recv buffer -- no disk
        round-trip. Same server/client token-drift + loss-mask contract as the rest of
        the Eagle backends (fail loud on drift; pad the optional decode-step position).
        """
        agent = self._rdma()
        url = self._next_url()
        host = url.split("://", 1)[-1].split(":")[0]
        r = self._http_rdma.post(
            f"{url}/v1/completions",
            json={
                "model": self.config.model,
                "prompt": sample["token_ids"],
                "max_tokens": 1,
                "temperature": 0,
            },
        )
        r.raise_for_status()
        kv = r.json().get("kv_transfer_params") or {}
        rid = kv.get("hs_req_id")
        if rid is None:
            warn_rank_0(f"[streaming] no hs_req_id for {sample['cid']}")
            return None
        # The connector advertises its sidecar port per request (one source of truth);
        # fall back to the env var only if an older server omits it.
        port = int(str(kv.get("hs_sidecar_port") or os.environ.get("HS_SIDECAR_PORT", "18999")))
        remote = self._remote(host, port)
        desc = None
        deadline = time.time() + self.config.request_timeout
        while time.time() < deadline:
            rr = self._http_rdma.get(f"http://{host}:{port}/desc", params={"req_id": rid})
            if rr.status_code == 200 and rr.json().get("ready"):
                desc = rr.json()
                break
            time.sleep(0.002)
        if desc is None:
            warn_rank_0(f"[streaming] rdma desc timeout for {sample['cid']}")
            return None
        shape = tuple(desc["hs_shape"])
        dtype = getattr(torch, desc["hs_dtype"])
        feat = shape[1:]
        maxtok = self.config.max_seq_len
        if shape[0] > maxtok:
            # The server captured more tokens than the recv buffer holds (its connector
            # max_tokens > our max_seq_len). Reading would silently truncate the slice;
            # fail loud so the size mismatch is configured away, not trained on.
            raise RuntimeError(
                f"server returned {shape[0]} tokens > max_seq_len={maxtok} for "
                f"{sample['cid']}; set max_seq_len >= the serve connector's max_tokens"
            )
        if self._recv is None or self._recv.dtype != dtype or tuple(self._recv.shape[1:]) != feat:
            # plain (pageable) host tensor: NIXL/ibv_reg_mr pins the pages itself.
            # Do NOT call .pin_memory() here — dataloader workers are forked and have no
            # valid CUDA context (cudaHostAlloc -> CUDA initialization error).
            if self._recv_reg is not None:
                agent.deregister_memory(self._recv_reg)  # release the prior pin on reshape
            self._recv = torch.empty((maxtok, *feat), dtype=dtype)
            self._recv_reg = agent.register_memory([self._recv])
        view = self._recv[: shape[0]]
        ldescs = agent.get_xfer_descs([view])
        rdescs = agent.deserialize_descs(base64.b64decode(desc["hs_descs"]))
        h = agent.initialize_xfer("READ", ldescs, rdescs, remote)
        agent.transfer(h)
        # Bound the completion poll: a dead/stuck remote agent must not hang this worker
        # forever (a single hung fetch stalls its DataLoader and, in DDP, the whole world).
        xfer_deadline = time.time() + self.config.request_timeout
        while True:
            st = agent.check_xfer_state(h)
            if st == "DONE":
                break
            if st == "ERR":
                agent.release_xfer_handle(h)
                warn_rank_0(f"[streaming] rdma xfer ERR for {sample['cid']}")
                return None
            if time.time() >= xfer_deadline:
                agent.release_xfer_handle(h)
                warn_rank_0(f"[streaming] rdma xfer timeout for {sample['cid']}")
                return None
            time.sleep(0.0002)
        agent.release_xfer_handle(h)
        hidden_states = view.clone()  # copy out before /done so the gen check brackets the read
        # /done frees the slot + reports valid; valid=False -> ring lapped us mid-read, bytes
        # stale -> resample. A failed /done can't prove staleness, so default valid=True.
        try:
            valid = self._http_rdma.get(
                f"http://{host}:{port}/done", params={"req_id": rid}
            ).json()["valid"]
        except Exception:
            valid = True
        if not valid:
            warn_rank_0(f"[streaming] slot lapped mid-read for {sample['cid']}; resampling")
            return None
        token_ids = torch.tensor(desc["token_ids"], dtype=torch.long)
        client_ids = torch.as_tensor(sample["token_ids"], dtype=token_ids.dtype)
        n = client_ids.shape[0]
        if token_ids.shape[0] not in (n, n + 1) or not torch.equal(token_ids[:n], client_ids):
            raise RuntimeError(
                f"server token_ids drift for {sample['cid']}: client_len={n}, "
                f"server_len={token_ids.shape[0]}"
            )
        loss_mask = self._align_loss_mask(sample["loss_mask"], token_ids.shape[0])
        return {"token_ids": token_ids, "hidden_states": hidden_states, "loss_mask": loss_mask}

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
