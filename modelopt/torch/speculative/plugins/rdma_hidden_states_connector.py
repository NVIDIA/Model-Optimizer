# SPDX-License-Identifier: Apache-2.0
"""RdmaHiddenStatesConnector (pooled) — no-disk hidden-states transfer over NIXL RDMA.

Pooled variant: register ONE pinned buffer pool with NIXL at startup (the only
ibv_reg_mr), assign each request a ring slot, copy its hidden states into that
slot, and hand the consumer a transfer descriptor for the slot sub-region.
No per-request memory registration; agent metadata is static (taken once after
the pool is registered).

Load out-of-tree:
  --kv-transfer-config '{"kv_connector":"RdmaHiddenStatesConnector",
     "kv_connector_module_path":"rdma_hidden_states_connector",
     "kv_role":"kv_producer",
     "kv_connector_extra_config":{"sidecar_port":"18999","pool_slots":"64","max_tokens":"512"}}'

Scope: TP>=1 (hidden states are replicated across TP ranks; only rank 0 owns the
pool + sidecar and serves them), host(pinned) memory (container UCX has no CUDA),
ring-slot reuse (fine when in-flight requests < pool_slots; no credit protocol yet).
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
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, parse_qs

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def extract_from_kv_cache(kv_cache, slot_mapping, num_tokens):
    block_size = kv_cache.shape[1]
    return kv_cache[slot_mapping // block_size, slot_mapping % block_size][:num_tokens]


@dataclass
class ReqMeta:
    req_id: str
    token_ids: torch.Tensor
    slot: int

    @staticmethod
    def make(req_id, token_ids, slot):
        return ReqMeta(req_id=req_id, token_ids=torch.tensor(token_ids), slot=slot)


@dataclass
class RdmaConnMeta(KVConnectorMetadata):
    requests: list = field(default_factory=list)

    def add(self, req_id, token_ids, slot):
        self.requests.append(ReqMeta.make(req_id, token_ids, slot))


class _Sidecar(BaseHTTPRequestHandler):
    connector = None

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
            self._send(200, {
                "ready": True,
                "hs_descs": base64.b64encode(e["descs"]).decode(),
                "hs_shape": list(e["shape"]),
                "hs_dtype": e["dtype"],
                "token_ids": e["token_ids"],
                "slot": e["slot"],
            })
            return
        if u.path == "/done":
            rid = parse_qs(u.query).get("req_id", [None])[0]
            c._bufs.pop(rid, None)  # slot recycled by the ring; pool stays registered
            self._send(200, {"freed": rid})
            return
        self._send(404, {"error": "not found"})


class RdmaHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):
    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config=vllm_config, role=role,
                         kv_cache_config=kv_cache_config)
        self._role = role
        ex = self._kv_transfer_config.get_from_extra_config
        self._sidecar_port = int(ex("sidecar_port", "18999"))
        self._pool_slots = int(ex("pool_slots", "64"))
        self._max_tokens = int(ex("max_tokens", "512"))
        self.cache_layers: list[str] = []
        # TP: hidden states are replicated across ranks, so only rank 0 owns the
        # pool + sidecar (set for real in register_kv_caches). Default True so the
        # scheduler-side instance and TP=1 behave exactly as before.
        self._tp_rank = 0
        self._owner = True
        # worker state
        self._copy_stream = None
        self._nixl = None
        self._agent_meta = b""
        self._pool = None          # [pool_slots, slot_elems] pinned
        self._slot_elems = 0
        self._per_token_elems = 0
        self._bufs: dict[str, dict] = {}
        self._accum_finished: set[str] = set()
        self._lock = threading.Lock()
        # scheduler state
        self._slot_ctr = 0
        logger.info("RdmaHiddenStatesConnector role=%s slots=%d max_tokens=%d port=%d",
                    role, self._pool_slots, self._max_tokens, self._sidecar_port)

    def _cs(self):
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
        return self._copy_stream

    # ---------- worker ----------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )
        from nixl._api import nixl_agent, nixl_agent_config
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        # TP: the captured hidden states are REPLICATED across TP ranks
        # (CacheOnlyAttentionLayer is sized to the full hidden_size, never
        # hidden_size/tp; the residual stream is all-reduced). So only TP rank 0
        # registers the pool + runs the sidecar and serves the (identical) hidden
        # states; other ranks no-op. This avoids the sidecar TCP-port collision
        # and the TPx pinned-memory waste, and needs no trainer-side change.
        self._tp_rank = get_tensor_model_parallel_rank()
        self._owner = self._tp_rank == 0

        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys()))
        self.cache_layers = list(layers.keys())
        assert len(self.cache_layers) == 1
        if not self._owner:
            logger.info("RdmaHiddenStatesConnector tp_rank=%d: non-owner, skipping "
                        "pool/sidecar (hidden states are replicated on rank 0).",
                        self._tp_rank)
            return
        kv = kv_caches[self.cache_layers[0]]
        # per-token feature = everything past [num_blocks, block_size]
        self._per_token_elems = int(prod(kv.shape[2:]))
        self._feat_shape = tuple(kv.shape[2:])
        self._dtype = kv.dtype
        self._slot_elems = self._max_tokens * self._per_token_elems

        self._nixl = nixl_agent(f"hs-producer-{uuid.uuid4()}",
                                nixl_agent_config(backends=["UCX"]))
        # ONE-TIME pool registration (the only ibv_reg_mr).
        t0 = time.time()
        self._pool = torch.empty((self._pool_slots, self._slot_elems),
                                 dtype=self._dtype).pin_memory()
        self._reg = self._nixl.register_memory([self._pool])
        reg_ms = (time.time() - t0) * 1e3
        self._agent_meta = self._nixl.get_agent_metadata()  # static after registration
        gib = self._pool.numel() * self._pool.element_size() / 2**30
        # pre-serialize a transfer descriptor per slot (cheap; no NIC registration)
        self._slot_descs = [
            self._nixl.get_serialized_descs(
                self._nixl.get_xfer_descs([self._pool[s]]))
            for s in range(self._pool_slots)
        ]
        self._start_sidecar()
        logger.info("RDMA pool registered: %d slots x %d elems (%.2f GiB) in %.1f ms; "
                    "per_token_elems=%d feat=%s dtype=%s",
                    self._pool_slots, self._slot_elems, gib, reg_ms,
                    self._per_token_elems, self._feat_shape, self._dtype)

    def _start_sidecar(self):
        h = type("H", (_Sidecar,), {"connector": self})
        self._sidecar = ThreadingHTTPServer(("0.0.0.0", self._sidecar_port), h)
        threading.Thread(target=self._sidecar.serve_forever, daemon=True).start()
        logger.info("RDMA sidecar on %s:%d", socket.gethostname(), self._sidecar_port)

    def start_load_kv(self, *a, **k): pass
    def wait_for_layer_load(self, *a, **k): pass
    def wait_for_save(self): pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if not self._owner:
            return  # non-owner TP ranks hold an identical copy; rank 0 serves it
        if layer_name not in self.cache_layers:
            return
        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionMetadata,
        )
        assert isinstance(attn_metadata, CacheOnlyAttentionMetadata)
        md = self._get_connector_metadata()
        assert isinstance(md, RdmaConnMeta)
        cs = self._cs()
        ready = torch.cuda.Event(); ready.record(); cs.wait_event(ready)
        slot_mapping = get_forward_context().slot_mapping[layer_name]  # type: ignore
        offset = 0
        for req in md.requests:
            n = req.token_ids.shape[0]
            nelems = n * self._per_token_elems
            slot = req.slot
            with torch.cuda.stream(cs):
                rsm = slot_mapping[offset:offset + n]
                offset += n
                hs_gpu = extract_from_kv_cache(kv_layer, rsm, n)  # [n, *feat]
                # copy into the pre-registered pool slot (flattened)
                self._pool[slot, :nelems].copy_(hs_gpu.reshape(-1), non_blocking=True)
            ev = torch.cuda.Event(); ev.record(cs)
            # descriptor for exactly this slot's used sub-region
            sub = self._pool[slot, :nelems]
            descs = self._nixl.get_serialized_descs(self._nixl.get_xfer_descs([sub]))
            with self._lock:
                self._bufs[req.req_id] = {
                    "slot": slot, "descs": descs,
                    "shape": (n, *self._feat_shape),
                    "dtype": str(self._dtype).split(".")[-1],
                    "token_ids": req.token_ids.tolist(),
                    "event": ev,
                }

    # ---------- scheduler ----------
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        assert num_external_tokens == 0

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = RdmaConnMeta()
        for nr in scheduler_output.scheduled_new_reqs:
            slot = self._slot_ctr % self._pool_slots
            self._slot_ctr += 1
            meta.add(nr.req_id, token_ids=nr.prompt_token_ids or [], slot=slot)
        return meta

    def request_finished(self, request, block_ids):
        return True, {"hs_req_id": request.request_id,
                      "hs_sidecar_port": self._sidecar_port}

    def request_finished_all_groups(self, request, block_ids):
        return self.request_finished(request, block_ids[0])

    def get_finished_count(self):
        # Override KVOutputAggregator's default (world_size): only TP rank 0 (the
        # owner) reports finished_sending, so a single completion frees a request.
        # Without this the aggregator would wait for all TP workers and never free.
        return 1

    def get_finished(self, finished_req_ids):
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
                done.add(rid); self._accum_finished.discard(rid)
        return (done or None), None

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config):
        if cls is KVConnectorBase_V1:
            raise TypeError("not on base class")
        return "NHD"
