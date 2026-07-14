# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Debug helper: hash the calibration samples each rank actually feeds to the model.

Gated by the ``PUZZLE_HASH_SAMPLES`` env var (set it to an output directory). When set, both the
legacy and the AutoModel scoring paths call :func:`log_batch_hashes` right before the model forward
to append a per-sample sha1 of ``input_ids`` to ``<dir>/<tag>_rank<global_rank>.txt``. This lets us
verify, independent of the scoring math, that (a) data-parallel sharding gives each dp rank a
DISJOINT slice and (b) the legacy and AutoModel backends observe the SAME set of samples.

Check it with, e.g.::

    # union of all automodel ranks == union of all legacy ranks (same samples)?
    tmp_dir=$(mktemp -d)
    trap 'rm -rf "$tmp_dir"' EXIT
    cat $DIR/automodel_rank*.txt | grep -o 'hash=[0-9a-f]*' | sort -u > "$tmp_dir/am.txt"
    cat $DIR/legacy_rank*.txt    | grep -o 'hash=[0-9a-f]*' | sort -u > "$tmp_dir/lg.txt"
    diff "$tmp_dir/am.txt" "$tmp_dir/lg.txt" && echo "SAME SAMPLE SET"
    # per dp rank disjoint? (no hash appears for two different dp ranks)
    grep -h . $DIR/automodel_rank*.txt | sed -E 's/.*(dp=[0-9]+).*(hash=[0-9a-f]+)/\2 \1/' | sort | uniq -c | sort -rn | head
"""

import hashlib
import os
from pathlib import Path

import torch
import torch.distributed as dist

__all__ = ["log_batch_hashes", "samples_hashing_enabled"]


def samples_hashing_enabled() -> bool:
    return bool(os.environ.get("PUZZLE_HASH_SAMPLES"))


def _global_rank() -> int:
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0


def log_batch_hashes(input_ids, tag: str, step: int, extra: str = "") -> None:
    """Append a per-sample sha1 of ``input_ids`` (shape ``[N, T]``) to this rank's hash file.

    No-op unless ``PUZZLE_HASH_SAMPLES`` is set. ``tag`` is typically ``"legacy"`` / ``"automodel"``;
    ``extra`` carries diagnostic fields (e.g. ``"dp=0 cp=1"``).
    """
    out_dir = os.environ.get("PUZZLE_HASH_SAMPLES")
    if not out_dir or input_ids is None:
        return
    try:
        ids = input_ids.detach().to("cpu").to(torch.int64)
    except Exception:  # noqa: BLE001 — never let a debug hook break scoring
        return
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    rank = _global_rank()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(ids.shape[0]):
        h = hashlib.sha1(ids[i].contiguous().numpy().tobytes()).hexdigest()[:16]
        lines.append(f"step={step} rank={rank} {extra} sample={i} hash={h}\n")
    with open(Path(out_dir) / f"{tag}_rank{rank}.txt", "a") as f:
        f.writelines(lines)
