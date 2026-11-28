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

"""C2 rollout capture: record every drafting round, one RolloutRecord per tagged request.

Enabled by ``MODELOPT_REJMEM_CAPTURE_DIR=<dir>``. A request is captured when its sampling params carry
``extra_args={"rejmem_tag": <tag>}`` (``"vllm_xargs"`` in a ``/v1/completions`` body). After it finishes,
a background thread writes ``<dir>/<tag>.safetensors`` (:func:`.records.save_record`), so the file shows
up shortly after the response. A request that ends without reaching its own stop condition (EOS, a stop
token id, ``max_tokens`` or ``max_model_len``) was aborted or stopped outside the engine (a stop string)
and leaves no file. A preempted request restarts its capture at the resume, whose prefill recomputes all
features, so its record holds the rounds after the last resume. ``MODELOPT_REJMEM_CONSUMER`` is recorded
as ``memory_consumer``.

``aux_layer_ids`` are the drafter's ``target_layer_ids``: feature block ``j`` of a row is the output of
target layer ``aux_layer_ids[j]`` (Hugging Face ``hidden_states[aux_layer_ids[j] + 1]``), in the order the
drafter concatenates them.

Requires vLLM 0.33.0 (Model Runner V2), a DFlash drafter and prefix caching off, and refuses anything
else at startup. The hooks run eagerly around ``DFlashSpeculator.propose``: they queue device-to-host
copies of the step on the current stream, and a background thread reads them after the step's CUDA
event, so the engine never waits on the host.
"""

import functools
import inspect
import os
import queue
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from modelopt import env

from . import vllm_weights
from .records import SCHEMA, RolloutRecord, save_record

__all__ = ["TAG_KEY", "install"]

TAG_KEY = "rejmem_tag"
_VLLM_VERSION = "0.33.0"
_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")
# The capture thread copies steps in batches of this many (or bytes) to keep its GIL time per step low.
_BATCH_STEPS = 32
_BATCH_BYTES = 1 << 30

# Under vllm.*: vLLM configures only its own logger tree, so a modelopt.* logger's INFO is dropped.
logger = init_logger(f"vllm.{__name__}")

_installed = False


def install() -> None:
    """Patch vLLM's model runner and DFlash speculator when ``MODELOPT_REJMEM_CAPTURE_DIR`` is set."""
    global _installed
    capture_dir = env.rejmem_capture_dir()
    if not capture_dir or _installed:
        return
    # Version-pinned vLLM internals: import them only when the capture is on.
    import vllm
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import LiLiCorrSpeculator
    from vllm.v1.worker.gpu_worker import Worker

    if vllm.__version__ != _VLLM_VERSION:
        raise RuntimeError(
            f"rejmem capture supports vLLM {_VLLM_VERSION} only, found {vllm.__version__}"
        )
    os.makedirs(capture_dir, exist_ok=True)
    recorder = _Recorder(capture_dir, env.rejmem_consumer())
    init_device = Worker.init_device
    add_requests = GPUModelRunner.add_requests
    finish_requests = GPUModelRunner.finish_requests
    propose = DFlashSpeculator.propose
    propose_signature = inspect.signature(propose)

    @functools.wraps(init_device)
    def init_device_hook(worker, *args, **kwargs):
        init_device(worker, *args, **kwargs)
        # Without Model Runner V2's DFlash speculator (e.g. on the V1 runner) no hook would ever run.
        # LiLiCorr's speculator inherits the hooked propose unchanged.
        speculator = type(getattr(worker.model_runner, "speculator", None))
        if speculator not in (DFlashSpeculator, LiLiCorrSpeculator):
            raise NotImplementedError(
                "rejmem capture needs Model Runner V2's DFlash or LiLiCorr speculator"
            )
        if worker.cache_config.enable_prefix_caching:
            # A cached prefix skips the rows whose features a record needs.
            raise NotImplementedError("rejmem capture needs --no-enable-prefix-caching")
        if worker.vllm_config.speculative_config.enable_adaptive_verification:
            # It compacts the batch on the device only, so the host spans would be wrong.
            raise NotImplementedError("rejmem capture does not support adaptive verification")

    @functools.wraps(add_requests)
    def add_requests_hook(runner, scheduler_output):
        recorder.admit(scheduler_output.scheduled_new_reqs, runner.max_model_len)
        add_requests(runner, scheduler_output)

    @functools.wraps(finish_requests)
    def finish_requests_hook(runner, scheduler_output):
        recorder.finish(scheduler_output.finished_req_ids, scheduler_output.preempted_req_ids or ())
        finish_requests(runner, scheduler_output)

    @functools.wraps(propose)
    def propose_hook(speculator, *args, **kwargs):
        call = propose_signature.bind(speculator, *args, **kwargs).arguments
        step = None if call.get("dummy_run") else recorder.begin_step(speculator, call)
        draft_tokens = propose(speculator, *args, **kwargs)
        if step is not None:
            recorder.end_step(step, draft_tokens)
        return draft_tokens

    Worker.init_device = init_device_hook
    GPUModelRunner.add_requests = add_requests_hook
    GPUModelRunner.finish_requests = finish_requests_hook
    DFlashSpeculator.propose = propose_hook
    _installed = True
    logger.info("rejmem capture: writing rollout records to %s", capture_dir)


@dataclass
class _Admission:
    tag: str
    prompt_len: int
    prefill_len: int  # tokens the first prefill covers (prompt + replayed output after a resume)
    num_computed_tokens: int
    max_tokens: int
    max_model_len: int
    stop_token_ids: tuple[int, ...]  # EOS included unless ignore_eos, as the scheduler checks them
    weight_version: int


@dataclass
class _Step:
    """One propose call: host metadata and in-flight device-to-host copies of its inputs."""

    reqs: list[tuple[int, str]]  # (batch index, req_id) of the tagged requests
    query_start_loc: np.ndarray  # [num_reqs + 1]
    num_drafts: np.ndarray  # [num_reqs] drafts this step verified
    proposed: bool  # False when vLLM skipped the draft pass (num_speculative_tokens == 0)
    # [A + 1, T, H]: the aux states in the drafter's order, then the final-norm hidden
    rows: torch.Tensor
    positions: torch.Tensor  # [T] int64
    input_ids: torch.Tensor  # [T] int32
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    bonus: torch.Tensor  # [num_reqs] the token each request sampled last
    drafts: torch.Tensor | None = None  # [num_reqs, NS] proposals, set after propose
    event: torch.cuda.Event | None = None
    stream: torch.cuda.Stream | None = None  # where the copies and the event were queued


def _to_host(t: torch.Tensor) -> torch.Tensor:
    """Queue a copy of ``t`` into fresh pinned memory on the current stream."""
    return torch.empty(t.shape, dtype=t.dtype, pin_memory=True).copy_(t, non_blocking=True)


def _ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """The concatenation of ``arange(start, start + length)`` over the pairs."""
    first = np.cumsum(lengths) - lengths
    return np.repeat(starts - first, lengths) + np.arange(int(lengths.sum()))


class _CaptureError(Exception):
    """The request's capture cannot be made exact; its record is dropped."""


class _Capture:
    """One tagged request's rows of every step it took part in, then its record once it finishes.

    The capture thread appends; the writer thread checks and assembles the record.
    """

    def __init__(self, req_id: str, admission: _Admission):
        self.req_id = req_id
        self.adm = admission
        self.num_steps = self.num_rows = 0
        self.rows = torch.empty(0)  # [rows, (A + 1) * H], step after step
        self.rows3 = self.rows  # the same as [rows, A + 1, H]
        self.ids = np.zeros(0, dtype=np.int32)  # [rows]
        # [steps, 7]: first position, rows, sampled, rejected, drafts verified, bonus token, proposed
        self.steps = np.zeros((0, 7), dtype=np.int64)
        self.drafts = np.zeros((0, 0), dtype=np.int64)  # [steps, NS]

    def extend(
        self, views: list[torch.Tensor], ids: np.ndarray, info: np.ndarray, drafts: np.ndarray
    ) -> None:
        """Append steps (``info``, ``drafts``) whose rows are ``views`` ([rows, A + 1, H] each)."""
        k, cur, num_new = self.num_steps, self.num_rows, len(info)
        n = sum(view.shape[0] for view in views)
        if not k:
            _, planes, hidden = views[0].shape
            ns = drafts.shape[1]
            # Each step takes at most NS + 1 rows and commits a token; one step may follow the stop.
            max_steps = self.adm.prefill_len + self.adm.max_tokens + ns + 1
            max_rows = self.adm.prefill_len + (self.adm.max_tokens + ns + 1) * (ns + 1)
            # Untouched pages of these upper bounds cost no memory.
            self.rows = torch.empty(max_rows, planes * hidden, dtype=views[0].dtype)
            self.rows3 = self.rows.view(max_rows, planes, hidden)
            self.ids = np.zeros(max_rows, dtype=np.int32)
            self.steps = np.zeros((max_steps, 7), dtype=np.int64)
            self.drafts = np.zeros((max_steps, ns), dtype=np.int64)
        if k + num_new > len(self.steps) or cur + n > len(self.ids):
            raise _CaptureError("more steps than max_tokens allows")
        torch.cat(views, out=self.rows3[cur : cur + n])
        self.ids[cur : cur + n] = ids
        self.steps[k : k + num_new] = info
        self.drafts[k : k + num_new] = drafts
        self.num_steps, self.num_rows = k + num_new, cur + n

    def to_record(
        self, layout: dict[str, Any], weight_version_end: int, memory_consumer: bool
    ) -> RolloutRecord | None:
        """The record of the finished request; None if it ended before its stop condition."""
        st = self.steps[: self.num_steps]
        p0, n, sampled, rejected, num_drafts, bonus, proposed = st.T
        if not len(st) or not sampled[-1]:
            return None
        start = np.cumsum(n) - n  # first row of each step
        valid = n - rejected  # committed rows of each step
        self._check(st, start, valid)
        ctx = _ranges(start, valid)
        tokens = np.append(self.ids[ctx], bonus[-1]).astype(np.int32)
        if self._stop_len(tokens) is None:
            return None

        anchor_steps = np.flatnonzero(sampled > 0)  # every anchor the drafter ran for
        kept = anchor_steps[proposed[anchor_steps] > 0]  # anchors it drafted for
        nxt = np.minimum(kept + 1, len(st) - 1)
        verified = (kept + 1 < len(st)) & (num_drafts[nxt] > 0)
        mem = _ranges(start[kept] + valid[kept], rejected[kept])
        aux_width = len(layout["aux_layer_ids"]) * layout["hidden_size"]
        rows = self.rows[: self.num_rows]
        tensors = {
            "token_ids": torch.from_numpy(tokens),
            "ctx_aux_hidden": rows[:, :aux_width].index_select(0, torch.from_numpy(ctx)),
            "ctx_last_hidden": rows[:, aux_width:].index_select(0, torch.from_numpy(ctx)),
            "round_anchor_pos": torch.from_numpy(p0[kept] + valid[kept]),
            "round_index": torch.from_numpy(np.searchsorted(anchor_steps, kept)),
            "round_num_accepted": torch.from_numpy(
                np.where(verified, sampled[nxt] - 1, -1).astype(np.int32)
            ),
            "round_draft_token_ids": torch.from_numpy(self.drafts[kept].astype(np.int32)),
            "round_mem_offset": torch.from_numpy(np.cumsum(np.append(0, rejected[kept]))),
            "mem_token_ids": torch.from_numpy(self.ids[mem]),
            "mem_aux_hidden": rows[:, :aux_width].index_select(0, torch.from_numpy(mem)),
        }
        meta = {
            "schema": SCHEMA,
            "tag": self.adm.tag,
            "req_id": self.req_id,
            "prompt_len": self.adm.prompt_len,
            **layout,
            "weight_version_start": self.adm.weight_version,
            "weight_version_end": weight_version_end,
            "memory_consumer": memory_consumer,
            "num_rounds_total": len(anchor_steps),
            "round_keep": "all",
        }
        return RolloutRecord(tensors=tensors, meta=meta)

    def _check(self, st: np.ndarray, start: np.ndarray, valid: np.ndarray) -> None:
        """Raise unless the steps chain into one committed sequence the way spec decoding must."""
        p0, _, sampled, rejected, num_drafts, bonus, proposed = st.T
        if p0[0] != 0 or np.any(p0[1:] != p0[:-1] + valid[:-1]):
            raise _CaptureError("a step's rows do not start at the committed length")
        if np.any(rejected != np.where(sampled > 0, num_drafts + 1 - sampled, 0)):
            raise _CaptureError("rejected rows do not match the verified drafts")
        if np.any((num_drafts > 0) & (p0 < self.adm.prefill_len)):
            raise _CaptureError("draft placeholders inside a prefill are not supported")
        prev = np.flatnonzero(sampled[:-1] > 0)
        if np.any(self.ids[start[prev + 1]] != bonus[prev]):
            raise _CaptureError("an anchor row does not hold the sampled bonus token")
        # A verify checks the drafts the step before it proposed.
        ver = np.flatnonzero((num_drafts[1:] > 0) & (sampled[:-1] > 0) & (proposed[:-1] > 0)) + 1
        cols = np.arange(self.drafts.shape[1])
        got = self.ids[np.minimum(start[ver, None] + 1 + cols, self.num_rows - 1)]
        if np.any((got != self.drafts[ver - 1]) & (cols < num_drafts[ver, None])):
            raise _CaptureError("the verified drafts are not the previous step's proposals")

    def _stop_len(self, tokens: np.ndarray) -> int | None:
        """Output length at which the scheduler stopped the request; None if it never would have."""
        out = tokens[self.adm.prompt_len :]
        limit = min(self.adm.max_tokens, self.adm.max_model_len - self.adm.prompt_len)
        hits = np.flatnonzero(np.isin(out[:limit], self.adm.stop_token_ids))
        if hits.size:
            return int(hits[0]) + 1
        return limit if out.size >= limit else None


class _Recorder:
    """Engine-thread hooks feeding a capture thread, which hands finished requests to a writer thread."""

    def __init__(self, capture_dir: str, memory_consumer: bool):
        self.capture_dir = capture_dir
        self.memory_consumer = memory_consumer
        self.tags: dict[str, str] = {}  # engine thread: req_id -> tag of live captured requests
        self.layout: dict[str, Any] | None = None  # record metadata fixed by the deployment
        self.counts = dict.fromkeys(("written", "aborted", "failed", "restarted"), 0)
        self._counts_lock = threading.Lock()
        # Bounded so a stalled capture thread throttles the engine instead of piling up pinned memory.
        self._events: queue.Queue = queue.Queue(maxsize=64)
        self._finished: queue.Queue = queue.Queue()
        self._caps: dict[str, _Capture] = {}  # capture thread only
        self._threads_started = False

    def _count(self, key: str) -> dict[str, int]:
        with self._counts_lock:
            self.counts[key] += 1
            return dict(self.counts)

    # Engine thread.

    def admit(self, new_reqs: list[Any], max_model_len: int) -> None:
        for new in new_reqs:
            if self.tags.pop(new.req_id, None) is not None:
                # Re-added without a preemption notice (streaming input): its rows restart too.
                self._events.put(("drop", new.req_id, "was re-added"))
            sp = new.sampling_params
            tag = (sp.extra_args or {}).get(TAG_KEY) if sp is not None else None
            if tag is None:
                continue
            self._start_threads()
            if not isinstance(tag, str) or not _TAG_RE.fullmatch(tag) or tag in self.tags.values():
                self._events.put(("reject", new.req_id, repr(tag), "invalid or duplicate tag"))
                continue
            self.tags[new.req_id] = tag
            stop = {*(sp.stop_token_ids or ())}
            if sp.eos_token_id is not None:
                stop.add(sp.eos_token_id)
            admission = _Admission(
                tag=tag,
                prompt_len=new.prompt_len,
                prefill_len=len(new.prefill_token_ids),
                num_computed_tokens=new.num_computed_tokens,
                max_tokens=sp.max_tokens,
                max_model_len=max_model_len,
                stop_token_ids=tuple(sorted(stop)),
                weight_version=vllm_weights.current_weight_version(),
            )
            self._events.put(("admit", new.req_id, admission))

    def finish(self, finished_req_ids: Iterable[str], preempted_req_ids: Iterable[str]) -> None:
        for req_id in preempted_req_ids:
            if self.tags.pop(req_id, None) is not None:
                self._events.put(("drop", req_id, "was preempted"))
        version = vllm_weights.current_weight_version()
        for req_id in finished_req_ids:
            if self.tags.pop(req_id, None) is not None:
                self._events.put(("finish", req_id, version))

    def begin_step(self, speculator: Any, call: dict[str, Any]) -> _Step | None:
        batch = call["input_batch"]
        reqs = [(i, r) for i, r in enumerate(batch.req_ids) if r in self.tags]
        if not reqs:
            return None
        aux_hidden, last_hidden = call["aux_hidden_states"], call["last_hidden_states"]
        if self.layout is None:
            self.layout = self._layout(speculator, aux_hidden, last_hidden)
        n, num_reqs = batch.num_tokens, batch.num_reqs
        rows = torch.empty(
            (len(aux_hidden) + 1, n, last_hidden.shape[-1]),
            dtype=last_hidden.dtype,
            pin_memory=True,
        )
        for plane, h in zip(rows, [*aux_hidden, last_hidden]):
            plane.copy_(h[:n], non_blocking=True)
        num_drafts = batch.num_draft_tokens_per_req
        return _Step(
            reqs=reqs,
            query_start_loc=batch.query_start_loc_np[: num_reqs + 1].copy(),
            num_drafts=np.zeros(num_reqs, np.int32) if num_drafts is None else num_drafts.copy(),
            proposed=call.get("num_speculative_tokens") != 0,
            rows=rows,
            positions=_to_host(batch.positions[:n]),
            input_ids=_to_host(batch.input_ids[:n]),
            num_sampled=_to_host(call["num_sampled"][:num_reqs]),
            num_rejected=_to_host(call["num_rejected"][:num_reqs]),
            bonus=_to_host(call["last_sampled"][batch.idx_mapping, 0]),
        )

    def end_step(self, step: _Step, draft_tokens: torch.Tensor) -> None:
        step.drafts = _to_host(draft_tokens)
        step.event = torch.cuda.Event(blocking=True)
        step.stream = torch.cuda.current_stream()
        step.event.record(step.stream)
        self._events.put(("step", step))

    @staticmethod
    def _layout(speculator: Any, aux_hidden: list[torch.Tensor], last_hidden: torch.Tensor):
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            get_eagle3_aux_layers_from_config,
        )

        spec_config = speculator.speculative_config
        layers = get_eagle3_aux_layers_from_config(spec_config) or ()
        if not aux_hidden or len(layers) != len(aux_hidden):
            raise RuntimeError(
                f"rejmem capture: {len(aux_hidden or ())} aux tensors, layers {layers}"
            )
        if (
            aux_hidden[0].shape[-1] != last_hidden.shape[-1]
            or aux_hidden[0].dtype != last_hidden.dtype
        ):
            raise RuntimeError(
                "rejmem capture: aux and final hidden states differ in width or dtype"
            )
        return {
            "num_spec_tokens": speculator.num_speculative_steps,
            # The target appends aux states in layer order, each the input of layer id + 1.
            "aux_layer_ids": sorted(layer - 1 for layer in layers),
            "hidden_size": last_hidden.shape[-1],
        }

    def _start_threads(self) -> None:
        # Started at the first tagged request, after vLLM's startup is done with forking helpers.
        if not self._threads_started:
            self._threads_started = True
            for target, name in ((self._capture_loop, "capture"), (self._write_loop, "writer")):
                threading.Thread(target=target, name=f"rejmem-{name}", daemon=True).start()

    # Capture thread: it shares the GIL with the engine, so it handles steps in batches.

    def _capture_loop(self) -> None:
        with torch.inference_mode():
            pending: list[_Step] = []
            while True:
                kind, *args = self._events.get()
                if kind == "step":
                    pending.append(args[0])
                    if (
                        len(pending) < _BATCH_STEPS
                        and sum(s.rows.nbytes for s in pending) < _BATCH_BYTES
                    ):
                        continue
                if pending:  # before any other event, which may end a request of these steps
                    steps, pending = pending, []
                    try:
                        self._on_steps(steps)
                    except Exception:
                        logger.exception("rejmem capture: failed to read %d steps", len(steps))
                        # Their requests lost rows, so none of them can get an exact record.
                        lost = {req_id for step in steps for _, req_id in step.reqs}
                        for req_id in lost & self._caps.keys():
                            self._on_reject(req_id, self._caps[req_id].adm.tag, "a step was lost")
                if kind != "step":
                    try:
                        getattr(self, f"_on_{kind}")(*args)
                    except Exception:
                        logger.exception("rejmem capture: failed to handle a %s event", kind)

    def _on_admit(self, req_id: str, admission: _Admission) -> None:
        if admission.num_computed_tokens:
            self._on_reject(req_id, admission.tag, "its prompt hit the prefix cache")
        else:
            self._caps[req_id] = _Capture(req_id, admission)

    def _on_reject(self, req_id: str, tag: str, reason: str) -> None:
        self._caps.pop(req_id, None)
        self._count("failed")
        logger.error("rejmem capture: no record for %s (tag %s): %s", req_id, tag, reason)

    def _on_drop(self, req_id: str, reason: str) -> None:
        if (cap := self._caps.pop(req_id, None)) is not None:
            self._count("restarted")
            logger.warning(
                "rejmem capture: %s (tag %s) %s; its record starts at the resume",
                req_id,
                cap.adm.tag,
                reason,
            )

    def _on_finish(self, req_id: str, weight_version_end: int) -> None:
        if (cap := self._caps.pop(req_id, None)) is not None:
            self._finished.put((cap, weight_version_end))

    def _on_steps(self, steps: list[_Step]) -> None:
        # Every torch call hands the GIL to the engine thread, so work per batch, not per request.
        # Events of one stream complete in order: the last one covers the batch.
        for step in steps:
            if step is steps[-1] or step.stream != steps[-1].stream:
                assert step.event is not None and step.drafts is not None
                step.event.synchronize()
        pos, ids, sampled, rejected, bonus, drafts = (
            torch.cat([getattr(step, name) for step in steps]).numpy()
            for name in ("positions", "input_ids", "num_sampled", "num_rejected", "bonus", "drafts")
        )
        row_off = np.cumsum([0] + [step.rows.shape[1] for step in steps])
        num_reqs = [len(step.query_start_loc) - 1 for step in steps]
        lo = np.concatenate([step.query_start_loc[:-1] + off for step, off in zip(steps, row_off)])
        lens = np.concatenate([np.diff(step.query_start_loc) for step in steps])
        # Each request's rows must cover consecutive positions.
        broken = np.add.reduceat(pos != np.repeat(pos[lo] - lo, lens) + np.arange(len(pos)), lo) > 0
        info = np.stack(
            [
                pos[lo],
                lens,
                sampled,
                rejected,
                np.concatenate([step.num_drafts for step in steps]),
                bonus,
                np.repeat([step.proposed for step in steps], num_reqs),
            ],
            axis=1,
        )
        rows = [step.rows.transpose(0, 1) for step in steps]  # [T, A + 1, H] each
        spans: dict[str, list[tuple[int, int]]] = {}  # req_id -> (step, row of info) per step
        for k, (step, off) in enumerate(zip(steps, np.cumsum([0, *num_reqs]))):
            for i, req_id in step.reqs:
                spans.setdefault(req_id, []).append((k, off + i))
        for req_id, pairs in spans.items():
            if (cap := self._caps.get(req_id)) is None:
                continue
            sel = np.array([j for _, j in pairs])
            try:
                if broken[sel].any():
                    raise _CaptureError("its rows are not at consecutive positions")
                views = [
                    rows[k][lo[j] - row_off[k] : lo[j] - row_off[k] + lens[j]] for k, j in pairs
                ]
                cap.extend(views, ids[_ranges(lo[sel], lens[sel])], info[sel], drafts[sel])
            except Exception as e:  # any surprise leaves the capture inexact: drop the record
                self._on_reject(req_id, cap.adm.tag, f"{type(e).__name__}: {e}")

    # Writer thread.

    def _write_loop(self) -> None:
        with torch.inference_mode():
            while True:
                cap, weight_version_end = self._finished.get()
                try:
                    assert self.layout is not None
                    rec = cap.to_record(self.layout, weight_version_end, self.memory_consumer)
                    if rec is not None:
                        save_record(
                            os.path.join(self.capture_dir, f"{cap.adm.tag}.safetensors"), rec
                        )
                except Exception as e:
                    counts = self._count("failed")
                    logger.error("rejmem capture: no record for %s: %r; %s", cap.req_id, e, counts)
                    continue
                if rec is None:
                    counts = self._count("aborted")
                    logger.warning(
                        "rejmem capture: %s (tag %s) ended before its stop condition (aborted, or "
                        "stopped by a stop string): no record. %s",
                        cap.req_id,
                        cap.adm.tag,
                        counts,
                    )
                elif (counts := self._count("written"))["written"] % 100 == 1:
                    logger.info("rejmem capture: %s", counts)
