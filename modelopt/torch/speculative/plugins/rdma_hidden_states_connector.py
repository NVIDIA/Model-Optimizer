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

"""RdmaHiddenStatesConnector (pooled) — no-disk hidden-states transfer over NIXL RDMA.

Pooled variant: register ONE pinned buffer pool with NIXL at startup (the only
ibv_reg_mr), assign each request a ring slot, copy its hidden states into that
slot, and hand the consumer a transfer descriptor for the slot sub-region.
No per-request memory registration; agent metadata is static (taken once after
the pool is registered).

Load out-of-tree:
  --kv-transfer-config '{"kv_connector":"RdmaHiddenStatesConnector",
     "kv_connector_module_path":"modelopt.torch.speculative.plugins.rdma_hidden_states_connector",
     "kv_role":"kv_producer",
     "kv_connector_extra_config":{"sidecar_port":"18999","pool_slots":"64","max_tokens":"512"}}'

Scope: TP>=1 (hidden states are replicated across TP ranks; only rank 0 owns the
pool + sidecar and serves them), host(pinned) memory (container UCX has no CUDA),
ring-slot reuse (fine when in-flight requests < pool_slots; no credit protocol yet).
A request may set ``kv_transfer_params={"hs_window_start": s}`` to have the whole prompt
prefilled but only positions ``s:`` captured; ``max_tokens`` then bounds the window, not
the prompt.
PP>1 is not supported (the capture layer lives on one PP rank; owner election only
covers TP). The sidecar is unauthenticated and binds all interfaces, so it assumes a
trusted cluster fabric -- any reachable peer can read descriptors / free slots.
"""

import base64
import json
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import prod
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlparse

import torch
from vllm.config import get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

from .hf_streaming_dataset import nixl_backends_from_env

logger = init_logger(__name__)


def extract_from_kv_cache(kv_cache, slot_mapping, num_tokens, heads_first=False):
    """Gather the first ``num_tokens`` rows of ``kv_cache`` addressed by ``slot_mapping``.

    ``kv_cache`` is ``[num_blocks, block_size, num_heads, head_size]``, or with ``heads_first``
    ``[num_blocks, num_heads, block_size, head_size]`` (newer vLLM's cache-only layer); either
    way the result is ``[num_tokens, num_heads, head_size]``.
    """
    if heads_first:
        kv_cache = kv_cache.transpose(1, 2)
    block_size = kv_cache.shape[1]
    return kv_cache[slot_mapping // block_size, slot_mapping % block_size][:num_tokens]


@dataclass
class ReqMeta:
    """Per-request scheduler->worker metadata: which pool slot holds this request."""

    req_id: str
    token_ids: torch.Tensor
    slot: int
    # The request's blocks in the hidden-states KV cache group: where its tokens sit.
    block_ids: list[int] = field(default_factory=list)
    # First prompt position to capture: the trainer can have the whole history prefilled but
    # fetch only a window at its end (kv_transfer_params["hs_window_start"]).
    start: int = 0

    @staticmethod
    def make(req_id, token_ids, slot, block_ids, start=0):
        """Build a :class:`ReqMeta`, tensorizing ``token_ids``."""
        return ReqMeta(
            req_id=req_id,
            token_ids=torch.tensor(token_ids),
            slot=slot,
            block_ids=list(block_ids),
            start=start,
        )


@dataclass
class RdmaConnMeta(KVConnectorMetadata):
    """Connector metadata for one scheduler step: the requests to capture this step."""

    requests: list = field(default_factory=list)

    def add(self, req_id, token_ids, slot, block_ids, start=0):
        """Append one request's capture metadata."""
        self.requests.append(ReqMeta.make(req_id, token_ids, slot, block_ids, start))


class _Sidecar(BaseHTTPRequestHandler):
    connector: ClassVar[Any] = None

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        c = type(self).connector
        u = urlparse(self.path)
        if u.path == "/meta":
            # static: pool was registered before we cached this
            self._send(200, {"agent_metadata": base64.b64encode(c._agent_meta).decode()})
            return
        if u.path == "/desc":
            rid = parse_qs(u.query).get("req_id", [None])[0]
            e = c._bufs.get(rid)
            if e is None:
                self._send(404, {"error": "unknown", "req_id": rid})
                return
            if not e["event"].query():
                self._send(202, {"ready": False})
                return
            self._send(
                200,
                {
                    "ready": True,
                    "hs_descs": base64.b64encode(e["descs"]).decode(),
                    "hs_shape": list(e["shape"]),
                    "hs_dtype": e["dtype"],
                    "token_ids": e["token_ids"],
                    "slot": e["slot"],
                },
            )
            return
        if u.path == "/done":
            # valid=False: slot's gen moved past this req -> ring lapped it mid-read, bytes stale.
            rid = parse_qs(u.query).get("req_id", [None])[0]
            with c._lock:
                e = c._bufs.pop(rid, None)
                valid = e is not None and c._slot_gen.get(e["slot"]) == e["gen"]
            self._send(200, {"freed": rid, "valid": valid})
            return
        self._send(404, {"error": "not found"})


class RdmaHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):
    """vLLM KV-transfer connector that serves captured hidden states over NIXL RDMA.

    See the module docstring for the pooled design and scope.
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        """Keep the capture layer's blocks separate from the drafter's KV cache."""
        return False

    def __init__(self, vllm_config, role, kv_cache_config):
        """Read pool/sidecar config from ``kv_connector_extra_config`` and init state."""
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self._role = role
        ex = self._kv_transfer_config.get_from_extra_config
        self._sidecar_port = int(ex("sidecar_port", "18999"))
        self._pool_slots = int(ex("pool_slots", "64"))
        self._max_tokens = int(ex("max_tokens", "512"))
        self._cache_group_id = self._find_cache_kv_group_id(kv_cache_config)
        self._block_size = 0  # the cache-only layer's block size (worker, set at registration)
        self.cache_layers: list[str] = []
        # TP: hidden states are replicated across ranks, so only rank 0 owns the
        # pool + sidecar (set for real in register_kv_caches). Default True so the
        # scheduler-side instance and TP=1 behave exactly as before.
        self._tp_rank = 0
        self._owner = True
        # worker state
        self._copy_stream = None
        self._nixl: Any = None  # nixl_agent (owner only); untyped lib -> Any
        self._agent_meta = b""
        self._pool: Any = None  # [pool_slots, slot_elems] pinned (owner only)
        self._slot_elems = 0
        self._per_token_elems = 0
        self._heads_first = False
        self._bufs: dict[str, dict] = {}
        # slot -> write counter; a req records its gen to detect a mid-read lap (see /done).
        self._slot_gen: dict[int, int] = {}
        self._accum_finished: set[str] = set()
        self._oversize_warned = False  # one-shot guard against log spam on oversized prompts
        self._lock = threading.Lock()
        # scheduler state
        self._slot_ctr = 0
        self._window_start: dict[str, int] = {}  # req id -> hs_window_start, until scheduled
        logger.info(
            "RdmaHiddenStatesConnector role=%s slots=%d max_tokens=%d port=%d",
            role,
            self._pool_slots,
            self._max_tokens,
            self._sidecar_port,
        )

    @staticmethod
    def _find_cache_kv_group_id(kv_cache_config) -> int:
        """Index of the KV cache group holding the captured hidden states.

        Same rule as vLLM's ExampleHiddenStatesConnector: found by spec type, so it resolves
        identically on the scheduler and the worker.
        """
        if kv_cache_config is None:
            return 0
        from vllm.v1.kv_cache_interface import HiddenStateCacheSpec

        groups = kv_cache_config.kv_cache_groups
        ids = [
            gid for gid, g in enumerate(groups) if isinstance(g.kv_cache_spec, HiddenStateCacheSpec)
        ]
        if len(ids) == 1:
            return ids[0]
        if not ids and len(groups) == 1:
            return 0
        raise ValueError(
            f"RdmaHiddenStatesConnector: cannot identify the hidden-states KV cache group "
            f"among {len(groups)} groups."
        )

    def _cs(self):
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
        return self._copy_stream

    # ---------- worker ----------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Elect the TP-rank-0 owner and (owner only) register the pinned NIXL pool."""
        from nixl._api import nixl_agent, nixl_agent_config
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionLayer

        # TP: the captured hidden states are REPLICATED across TP ranks
        # (CacheOnlyAttentionLayer is sized to the full hidden_size, never
        # hidden_size/tp; the residual stream is all-reduced). So only TP rank 0
        # registers the pool + runs the sidecar and serves the (identical) hidden
        # states; other ranks no-op. This avoids the sidecar TCP-port collision
        # and the TPx pinned-memory waste, and needs no trainer-side change.
        self._tp_rank = get_tensor_model_parallel_rank()
        self._owner = self._tp_rank == 0

        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys())
        )
        self.cache_layers = list(layers.keys())
        assert len(self.cache_layers) == 1
        if not self._owner:
            logger.info(
                "RdmaHiddenStatesConnector tp_rank=%d: non-owner, skipping "
                "pool/sidecar (hidden states are replicated on rank 0).",
                self._tp_rank,
            )
            return
        kv = kv_caches[self.cache_layers[0]]
        # Newer vLLM stores the cache-only layer heads-first; sizing the pool off the old
        # layout there counts a whole block per token (an ~800 GB slot -> pinned-alloc OOM).
        num_heads = layers[self.cache_layers[0]].num_heads
        if kv.shape[1] == kv.shape[2]:
            raise ValueError(
                f"RdmaHiddenStatesConnector: cannot tell block_size from num_heads in "
                f"cache-only KV shape {tuple(kv.shape)}."
            )
        self._heads_first = kv.shape[1] == num_heads
        if self._heads_first:
            kv = kv.transpose(1, 2)
        self._block_size = int(kv.shape[1])
        # per-token feature = everything past [num_blocks, block_size]
        self._per_token_elems = int(prod(kv.shape[2:]))
        self._feat_shape = tuple(kv.shape[2:])
        self._dtype = kv.dtype
        self._slot_elems = self._max_tokens * self._per_token_elems

        # Live slots = in-flight fetch concurrency, capped by max_num_seqs. If the ring
        # laps an unread slot the gen check (see /done) catches it as a miss + resample,
        # so undersizing wastes work rather than corrupting; warn to size the pool right.
        max_num_seqs = getattr(
            getattr(self._vllm_config, "scheduler_config", None), "max_num_seqs", None
        )
        if max_num_seqs is not None and self._pool_slots < max_num_seqs:
            logger.warning(
                "RdmaHiddenStatesConnector: pool_slots=%d < max_num_seqs=%d; the ring may "
                "lap unread slots under high fetch concurrency (-> wasted resamples). Raise "
                "kv_connector_extra_config.pool_slots to >= %d.",
                self._pool_slots,
                max_num_seqs,
                max_num_seqs,
            )

        _backends = nixl_backends_from_env()
        self._nixl = nixl_agent(
            f"hs-producer-{uuid.uuid4()}", nixl_agent_config(backends=_backends)
        )
        # ONE-TIME pool registration (the only ibv_reg_mr).
        t0 = time.time()
        self._pool = torch.empty(
            (self._pool_slots, self._slot_elems), dtype=self._dtype
        ).pin_memory()
        self._reg = self._nixl.register_memory([self._pool])
        reg_ms = (time.time() - t0) * 1e3
        self._agent_meta = self._nixl.get_agent_metadata()  # static after registration
        gib = self._pool.numel() * self._pool.element_size() / 2**30
        self._start_sidecar()
        logger.info(
            "RDMA pool registered: %d slots x %d elems (%.2f GiB) in %.1f ms; "
            "per_token_elems=%d feat=%s dtype=%s",
            self._pool_slots,
            self._slot_elems,
            gib,
            reg_ms,
            self._per_token_elems,
            self._feat_shape,
            self._dtype,
        )

    def _start_sidecar(self):
        h = type("H", (_Sidecar,), {"connector": self})
        # Bind all interfaces: cross-node trainers reach this sidecar over the
        # cluster fabric, not just loopback.
        self._sidecar = ThreadingHTTPServer(("0.0.0.0", self._sidecar_port), h)  # nosec B104
        threading.Thread(target=self._sidecar.serve_forever, daemon=True).start()
        logger.info("RDMA sidecar on %s:%d", socket.gethostname(), self._sidecar_port)

    def start_load_kv(self, *a, **k):
        """No-op: this is a store-only (producer) connector, nothing to load."""

    def wait_for_layer_load(self, *a, **k):
        """No-op: store-only connector, nothing to load."""

    def wait_for_save(self):
        """No-op: the per-request DtoH copy is awaited via the get_finished event."""

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        """Owner only: copy each request's hidden states into its pool slot + record a descriptor."""
        if not self._owner:
            return  # non-owner TP ranks hold an identical copy; rank 0 serves it
        if layer_name not in self.cache_layers:
            return
        from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionMetadata

        assert isinstance(attn_metadata, CacheOnlyAttentionMetadata)
        md = self._get_connector_metadata()
        assert isinstance(md, RdmaConnMeta)
        cs = self._cs()
        ready = torch.cuda.Event()
        ready.record()
        cs.wait_event(ready)
        # Each request's cache slots come from its own block ids, as in vLLM's
        # ExampleHiddenStatesConnector. The batch's slot_mapping is in the model runner's batch
        # order, which need not match the scheduler's order of new requests, so slicing it
        # with a running offset hands a request another request's hidden states whenever
        # several are in flight (token ids still match, so nothing downstream notices).
        bs = self._block_size
        offsets = torch.arange(bs, dtype=torch.long)
        for req in md.requests:
            start, end = req.start, req.token_ids.shape[0]
            n = end - start  # captured tokens: the window [start, end) of the prompt
            nelems = n * self._per_token_elems
            slot = req.slot
            if nelems > self._slot_elems:
                # Prompt longer than a pool slot (max_tokens). Copying would overrun the
                # slot into the neighbour's bytes; skip capture instead. The trainer's
                # /desc poll times out for this req and resamples. Misconfiguration, so
                # log loudly once and tell the operator how to size the pool.
                if not self._oversize_warned:
                    logger.error(
                        "RdmaHiddenStatesConnector: request with %d tokens exceeds "
                        "max_tokens=%d (pool slot capacity); skipping capture. Raise "
                        "kv_connector_extra_config.max_tokens to >= the trainer's "
                        "max_seq_len. Further occurrences suppressed.",
                        n,
                        self._max_tokens,
                    )
                    self._oversize_warned = True
                continue
            blocks = torch.tensor(req.block_ids, dtype=torch.long)
            if blocks.numel() * bs < end:
                raise RuntimeError(
                    f"RdmaHiddenStatesConnector: req {req.req_id} has {end} tokens but only "
                    f"{blocks.numel()} blocks of {bs} in the hidden-states group."
                )
            slots = (blocks[:, None] * bs + offsets[None, :]).flatten()[start:end]
            with torch.cuda.stream(cs):
                rsm = slots.to(kv_layer.device, non_blocking=True)
                hs_gpu = extract_from_kv_cache(kv_layer, rsm, n, self._heads_first)  # [n, *feat]
                # copy into the pre-registered pool slot (flattened)
                self._pool[slot, :nelems].copy_(hs_gpu.reshape(-1), non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(cs)
            # descriptor for exactly this slot's used sub-region
            sub = self._pool[slot, :nelems]
            descs = self._nixl.get_serialized_descs(self._nixl.get_xfer_descs([sub]))
            with self._lock:
                gen = self._slot_gen.get(slot, 0) + 1  # this write's gen
                self._slot_gen[slot] = gen
                self._bufs[req.req_id] = {
                    "slot": slot,
                    "descs": descs,
                    "shape": (n, *self._feat_shape),
                    "dtype": str(self._dtype).split(".")[-1],
                    "token_ids": req.token_ids[start:].tolist(),
                    "event": ev,
                    "gen": gen,
                }

    # ---------- scheduler ----------
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """Store-only connector: nothing is loaded from an external cache."""
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        """Store-only connector: no external tokens are ever allocated.

        Records the request's ``hs_window_start`` (from the completion request's
        ``kv_transfer_params``) for :meth:`build_connector_meta`.
        """
        assert num_external_tokens == 0
        start = int((request.kv_transfer_params or {}).get("hs_window_start", 0))
        if start:
            self._window_start[request.request_id] = start

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Assign each newly scheduled request a ring-pool slot for this step.

        ``save_kv_layer`` reads each prompt's hidden states from its own blocks in one go, so
        it needs each prompt computed whole in one step; assert that here (chunked prefill
        would split it and capture only part).
        """
        meta = RdmaConnMeta()
        for nr in scheduler_output.scheduled_new_reqs:
            prompt = nr.prompt_token_ids or []
            n_sched = scheduler_output.num_scheduled_tokens[nr.req_id]
            start = self._window_start.pop(nr.req_id, 0)
            assert n_sched >= len(prompt), (
                f"RdmaHiddenStatesConnector requires the full prompt in one step "
                f"(disable chunked prefill): req {nr.req_id} scheduled {n_sched} of "
                f"{len(prompt)} prompt tokens."
            )
            if n_sched > len(prompt):
                # A preempted request resumes as new with its output tokens re-prefilled. Its
                # prompt was already captured before preemption (or it is a generation request
                # nobody fetches), so skip it rather than kill the engine.
                logger.warning(
                    "RdmaHiddenStatesConnector: req %s resumed after preemption (%d tokens "
                    "scheduled, %d-token prompt); not capturing it again.",
                    nr.req_id,
                    n_sched,
                    len(prompt),
                )
                continue
            if not 0 <= start < len(prompt):
                # A bad client value must not kill the engine: capture the whole prompt, and
                # the trainer's token-drift check rejects the mismatched window loudly.
                logger.error(
                    "RdmaHiddenStatesConnector: hs_window_start=%d outside the %d-token "
                    "prompt of req %s; capturing the whole prompt.",
                    start,
                    len(prompt),
                    nr.req_id,
                )
                start = 0
            slot = self._slot_ctr % self._pool_slots
            self._slot_ctr += 1
            meta.add(
                nr.req_id,
                token_ids=prompt,
                slot=slot,
                block_ids=nr.block_ids[self._cache_group_id],
                start=start,
            )
        return meta

    def request_finished(self, request, block_ids):
        """Return the request id + sidecar port for the trainer to fetch over RDMA.

        ``hs_max_tokens`` lets the trainer fail loud if its ``max_seq_len`` exceeds our pool
        slot capacity (which would otherwise silently drop long prompts and hang the fetch).
        """
        return True, {
            "hs_req_id": request.request_id,
            "hs_sidecar_port": self._sidecar_port,
            "hs_max_tokens": self._max_tokens,
        }

    def request_finished_all_groups(self, request, block_ids):
        """Multi-group variant of :meth:`request_finished` (first group's blocks)."""
        return self.request_finished(request, block_ids[0])

    def get_finished_count(self):
        """Expect a single completion (TP rank 0) per request, not world_size."""
        # Override KVOutputAggregator's default (world_size): only TP rank 0 (the
        # owner) reports finished_sending, so a single completion frees a request.
        # Without this the aggregator would wait for all TP workers and never free.
        return 1

    def get_finished(self, finished_req_ids):
        """Owner: report requests whose DtoH copy event has completed (non-owner: none)."""
        # Only the owner gates completion on its copy event. Non-owners have no
        # _bufs and MUST report nothing — else (with get_finished_count==1) they
        # would mark a request done before rank 0's DtoH copy lands, and the
        # trainer would RDMA-read a stale/empty slot.
        if not self._owner:
            return None, None
        self._accum_finished.update(finished_req_ids)
        done = set()
        for rid in list(self._accum_finished):
            e = self._bufs.get(rid)
            if e is None or e["event"].query():
                done.add(rid)
                self._accum_finished.discard(rid)
        return (done or None), None

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config):
        """Require NHD layout so each token's hidden state stays contiguous."""
        if cls is KVConnectorBase_V1:
            raise TypeError("not on base class")
        return "NHD"
