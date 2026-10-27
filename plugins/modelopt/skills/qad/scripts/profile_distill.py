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

"""Profile per-stage time and memory of examples/megatron_bridge/distill.py, unchanged.

Run it exactly like distill.py (same CLI, from examples/megatron_bridge), for a few iterations on
mock data. Each rank appends one JSON line per training iteration to ``$PROFILE_OUT/rank<r>.jsonl``;
``plan_layout.py`` reads them. Checkpoint saving and loading are disabled.

Env:
  PROFILE_OUT        output directory (required)
  PROFILE_OPT_OFFLOAD  fraction of optimizer state to offload to CPU, to profile on fewer nodes than
                     training: 1 - 1/k reproduces the optimizer memory of k times the data parallelism
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())

import distill as D
from megatron.bridge.training import train as T
from megatron.core import parallel_state
from megatron.core.pipeline_parallel import schedules as S
from megatron.core.utils import unwrap_model

import modelopt.torch.utils.distributed as dist

OUT = os.environ["PROFILE_OUT"]
GiB = 1024**3
_events = {"fwd": [], "bwd": []}
_stage = {}


def _timed(kind, fn):
    def wrapped(*args, **kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        _events[kind].append((start, end))  # read after the iteration: no sync inside it
        return out

    return wrapped


def _record_stage(model):
    """Layers and effective recompute of the student on this stage, read once from the model."""
    student = unwrap_model(model[0] if isinstance(model, list) else model)
    if hasattr(student, "language_model"):
        student = student.language_model
    config, layers = student.decoder.config, len(student.decoder.layers)
    recomputed = 0
    if config.recompute_granularity == "full":
        recomputed = layers if config.recompute_method == "uniform" else config.recompute_num_layers
    _stage.update(layers=layers, recomputed_layers=min(layers, recomputed))


_fwd = _timed("fwd", S.forward_step)


def _forward_step(*args, **kwargs):
    if not _stage:
        _record_stage(kwargs.get("model", args[2] if len(args) > 2 else None))
    return _fwd(*args, **kwargs)


S.forward_step = _forward_step
S.backward_step = _timed("bwd", S.backward_step)
_train_step = T.train_step


def _profiled_train_step(*args, **kwargs):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    static = torch.cuda.memory_allocated()
    for spans in _events.values():
        spans.clear()
    t0 = time.time()
    out = _train_step(*args, **kwargs)
    torch.cuda.synchronize()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    record = {
        "rank": torch.distributed.get_rank(),
        "pp_rank": parallel_state.get_pipeline_model_parallel_rank(),
        "pp_size": pp_size,
        "iter_s": time.time() - t0,
        "microbatches": len(_events["fwd"]),
        **{
            f"{kind}_ms": sum(s.elapsed_time(e) for s, e in spans)
            for kind, spans in _events.items()
        },
        **_stage,
        "static_gib": static / GiB,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / GiB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GiB,
        "device_gib": torch.cuda.mem_get_info()[1] / GiB,
    }
    with open(os.path.join(OUT, f"rank{record['rank']}.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    return out


T.train_step = _profiled_train_step
_distill = D.distill


def _profiled_distill(cfg):
    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    offload = os.environ.get("PROFILE_OPT_OFFLOAD")
    if offload:
        cfg.optimizer.optimizer_cpu_offload = True
        cfg.optimizer.optimizer_offload_fraction = float(offload)
        cfg.optimizer.use_precision_aware_optimizer = True
    return _distill(cfg)


D.distill = _profiled_distill

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    dist.setup()
    args = D.get_args()
    try:
        with D.distill_run(args, D.DISTILL) as owns_the_run:
            D.main(args, owns_the_run)
    except BaseException:
        dist.abort()
        raise
    finally:
        dist.cleanup()
