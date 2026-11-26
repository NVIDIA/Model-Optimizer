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

"""On-policy rollout records: what a DFlash-family drafter saw while generating one response.

One record per request, written inside vLLM by the capture hook and read by the trainer. It holds the
committed sequence with the target's features, and every drafting round with the target's features on
the drafts the previous verify rejected (the round's memory). Positions are 0-based absolute positions
in the committed sequence. This module must not import vLLM.

A preempted request is admitted again when it resumes, and its record holds only the rounds after the
last admission: ``round_index``, ``num_rounds_total`` and ``weight_version_start`` count from there, and
committed positions before ``round_anchor_pos[0]`` have no round.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = ["SCHEMA", "RolloutRecord", "load_record", "save_record", "validate_record"]

SCHEMA = "modelopt.rejmem.v1"

# name -> (dtype, shape spec). L: committed tokens known to the worker, N: committed rows with features
# (positions 0..N-1, N <= L), R: kept rounds, M: memory rows of the kept rounds, NS: drafts per round,
# AH: concatenated aux width (num aux layers x hidden), H: target hidden size.
TENSOR_SPEC: dict[str, tuple[torch.dtype, str]] = {
    "token_ids": (torch.int32, "L"),
    "ctx_aux_hidden": (torch.bfloat16, "N,AH"),
    "ctx_last_hidden": (torch.bfloat16, "N,H"),  # after the target's final norm (lm_head input)
    "round_anchor_pos": (torch.int64, "R"),  # position of the round's anchor (bonus) token
    "round_index": (torch.int64, "R"),  # ordinal of the round since the request's last admission
    # Drafts of the round accepted by the next verify; -1 if it was never verified.
    "round_num_accepted": (torch.int32, "R"),
    "round_draft_token_ids": (torch.int32, "R,NS"),  # proposals for positions anchor+1 .. anchor+NS
    "round_mem_offset": (torch.int64, "R+1"),  # CSR: round t's memory rows are [off[t], off[t+1])
    "mem_token_ids": (torch.int32, "M"),  # the rejected draft tokens (row i sits at anchor + i)
    "mem_aux_hidden": (torch.bfloat16, "M,AH"),
}

# Required metadata keys (stored as one JSON string under the safetensors metadata key "rejmem").
META_KEYS = (
    "schema",
    "tag",
    "req_id",
    "prompt_len",
    "num_spec_tokens",
    "aux_layer_ids",
    "hidden_size",
    "weight_version_start",
    "weight_version_end",
    "memory_consumer",
    "num_rounds_total",
    "round_keep",
)


@dataclass
class RolloutRecord:
    """Tensors of ``TENSOR_SPEC`` plus metadata; see the module docstring."""

    tensors: dict[str, torch.Tensor]
    meta: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> torch.Tensor:
        tensors = self.__dict__.get("tensors", {})
        if name in tensors:
            return tensors[name]
        raise AttributeError(name)

    @property
    def mem_count(self) -> torch.Tensor:
        """Memory rows per kept round, [R] int64."""
        off = self.tensors["round_mem_offset"]
        return off[1:] - off[:-1]

    def mem_rows(self, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(token ids [r], aux features [r, AH]) of kept round ``t``'s memory."""
        off = self.tensors["round_mem_offset"]
        lo, hi = int(off[t]), int(off[t + 1])
        return self.tensors["mem_token_ids"][lo:hi], self.tensors["mem_aux_hidden"][lo:hi]


def validate_record(rec: RolloutRecord) -> None:
    """Raise ValueError unless ``rec`` satisfies the schema and the round invariants.

    The round checks tie the metadata to what spec decoding must have done: consecutive rounds advance
    the anchor by ``accepted + 1``, a round's memory tokens are the previous round's rejected drafts, and
    a round's accepted drafts are the committed tokens after its anchor.
    """
    t, m = rec.tensors, rec.meta
    missing = [k for k in META_KEYS if k not in m]
    if missing or m.get("schema") != SCHEMA:
        raise ValueError(f"bad metadata (missing {missing}, schema {m.get('schema')!r})")
    for name, (dtype, _) in TENSOR_SPEC.items():
        if name not in t:
            raise ValueError(f"missing tensor {name}")
        if t[name].dtype != dtype:
            raise ValueError(f"{name}: dtype {t[name].dtype}, expected {dtype}")
    ns = int(m["num_spec_tokens"])
    n = t["ctx_aux_hidden"].shape[0]
    length = t["token_ids"].shape[0]
    r = t["round_anchor_pos"].shape[0]
    ah = t["ctx_aux_hidden"].shape[1]
    if n > length or t["ctx_last_hidden"].shape != (n, int(m["hidden_size"])):
        raise ValueError("ctx tensors do not match token_ids / hidden_size")
    if ah != len(m["aux_layer_ids"]) * int(m["hidden_size"]):
        raise ValueError(f"aux width {ah} != {len(m['aux_layer_ids'])} x {m['hidden_size']}")
    for name in ("round_index", "round_num_accepted"):
        if t[name].shape != (r,):
            raise ValueError(f"{name}: shape {tuple(t[name].shape)}, expected ({r},)")
    if t["round_draft_token_ids"].shape != (r, ns):
        raise ValueError("round_draft_token_ids must be [R, NS]")
    off = t["round_mem_offset"]
    if off.shape != (r + 1,) or int(off[0]) != 0 or bool((off[1:] < off[:-1]).any()):
        raise ValueError("round_mem_offset must be [R+1], start at 0 and be non-decreasing")
    mem = int(off[-1])
    if t["mem_token_ids"].shape != (mem,) or t["mem_aux_hidden"].shape != (mem, ah):
        raise ValueError("mem tensors do not match round_mem_offset[-1]")
    count = off[1:] - off[:-1]
    if bool((count > ns).any()):
        raise ValueError("a round has more than NS memory rows")
    acc = t["round_num_accepted"]
    if bool(((acc < -1) | (acc > ns)).any()):
        raise ValueError("round_num_accepted outside [-1, NS]")
    anchor, index = t["round_anchor_pos"], t["round_index"]
    if r and (bool((index[1:] <= index[:-1]).any()) or bool((anchor[1:] <= anchor[:-1]).any())):
        raise ValueError("rounds must be strictly increasing in index and anchor")
    if r and (int(anchor[0]) < int(m["prompt_len"]) or int(anchor[-1]) >= length):
        raise ValueError("anchors must lie in [prompt_len, len(token_ids))")
    if int(m["num_rounds_total"]) < r:
        raise ValueError("num_rounds_total < kept rounds")
    tok = t["token_ids"]
    for i in range(r):
        a, k = int(anchor[i]), int(acc[i])
        if k >= 0 and a + k < length:
            # Accepted drafts are committed right after the anchor.
            if not torch.equal(tok[a + 1 : a + 1 + k], t["round_draft_token_ids"][i, :k]):
                raise ValueError(f"round {i}: accepted drafts differ from committed tokens")
        if i + 1 < r and int(index[i + 1]) == int(index[i]) + 1 and k >= 0:
            if int(anchor[i + 1]) != a + k + 1:
                raise ValueError(f"round {i + 1}: anchor != previous anchor + accepted + 1")
            nxt = int(count[i + 1])
            if nxt > ns - k:
                raise ValueError(f"round {i + 1}: more memory rows than rejected drafts")
            lo = int(off[i + 1])
            # The next round's memory rows are this round's rejected drafts, in order.
            if not torch.equal(
                t["mem_token_ids"][lo : lo + nxt], t["round_draft_token_ids"][i, k : k + nxt]
            ):
                raise ValueError(f"round {i + 1}: memory tokens are not the rejected drafts")


def save_record(path: str, rec: RolloutRecord) -> None:
    """Validate and write ``rec`` atomically (temp file + rename) as safetensors."""
    from safetensors.torch import save_file

    validate_record(rec)
    tmp = f"{path}.tmp.{os.getpid()}"
    tensors = {k: v.detach().contiguous().cpu() for k, v in rec.tensors.items()}
    save_file(tensors, tmp, metadata={"rejmem": json.dumps(rec.meta)})
    os.replace(tmp, path)


def load_record(path: str, validate: bool = True) -> RolloutRecord:
    """Read a record written by :func:`save_record`."""
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        meta = json.loads(f.metadata()["rejmem"])
        tensors = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    rec = RolloutRecord(tensors=tensors, meta=meta)
    if validate:
        validate_record(rec)
    return rec
