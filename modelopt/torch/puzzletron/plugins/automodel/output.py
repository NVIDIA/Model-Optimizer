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

"""Consolidated, layout-invariant output for AutoModel activation scoring.

After gather+reduce every rank in a pipeline stage holds identical scores, so a
single writer per stage persists them. We keep the legacy on-disk contract — one
``rank_<shard>.pth`` per pipeline stage, each a ``{module_name: score_dict}`` map
keyed by the same module names ``get_module_names_to_hook`` returns — so the
existing pruning loader (``pruning_utils._cache_activations_log``, which globs
``rank_*.pth`` and merges by module name) consumes it **unchanged**. With no
pipeline parallel this is a single fully-aggregated ``rank_0.pth``.
"""

import logging
from pathlib import Path

import torch

from .reduction import MeshGroups, is_writer, writer_shard_id

logger = logging.getLogger(__name__)

__all__ = ["write_scores"]


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _copy_containers(value):
    """Copy mutable containers without duplicating activation tensors."""
    if isinstance(value, dict):
        return {key: _copy_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_containers(item) for item in value)
    return value


def _merge_score_dict(target: dict, incoming: dict, *, module_name: str) -> None:
    overlap = set(target) & set(incoming)
    unsupported = overlap - {"magnitude_metadata"}
    if unsupported:
        raise RuntimeError(
            f"Duplicate activation-score writer for {module_name!r}; "
            f"overlapping fields={sorted(unsupported)}"
        )
    if "magnitude_metadata" in overlap:
        metadata_overlap = set(target["magnitude_metadata"]) & set(
            incoming["magnitude_metadata"]
        )
        if metadata_overlap:
            raise RuntimeError(
                f"Duplicate magnitude-score metadata for {module_name!r}; "
                f"fields={sorted(metadata_overlap)}"
            )
        target["magnitude_metadata"].update(incoming["magnitude_metadata"])
    for key, value in incoming.items():
        if key != "magnitude_metadata" or key not in overlap:
            target[key] = value


def write_scores(scorers, activations_log_dir: str, groups: MeshGroups) -> dict:
    """Finalize every scorer and write the owning stage's scores to ``rank_<shard>.pth``.

    ``finalize()`` performs collective gather/reduce, so **every rank must call this
    with the same scorers in the same order**; only the single writer of each
    pipeline stage (``token==0 and tp==0 and ep==0``) saves the file. Returns the
    in-memory ``{module_name: score_dict}`` map (on every rank, device tensors).
    """
    results = {}
    default_results = {}
    ep_sharded_results = {}
    for scorer in scorers:
        assert scorer.name is not None, "scorer.name (canonical module key) must be set"
        score_dict = scorer.finalize()
        if scorer.name in results:
            _merge_score_dict(results[scorer.name], score_dict, module_name=scorer.name)
        else:
            results[scorer.name] = _copy_containers(score_dict)
        if getattr(scorer, "write_all_ep_ranks", False):
            destination = ep_sharded_results
        else:
            destination = default_results
        if scorer.name in destination:
            _merge_score_dict(destination[scorer.name], score_dict, module_name=scorer.name)
        else:
            destination[scorer.name] = _copy_containers(score_dict)

    out_dir = Path(activations_log_dir)
    if default_results and is_writer(groups):
        cpu_results = {name: _to_cpu(score_dict) for name, score_dict in default_results.items()}
        out_dir = Path(activations_log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"rank_{writer_shard_id(groups)}.pth"
        torch.save(cpu_results, path)
        logger.info("Wrote %d module scores to %s", len(cpu_results), path)

    if ep_sharded_results and groups.token_rank == 0 and groups.tp_rank == 0:
        cpu_results = {name: _to_cpu(score_dict) for name, score_dict in ep_sharded_results.items()}
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"rank_{writer_shard_id(groups)}_ep{groups.ep_rank}.pth"
        torch.save(cpu_results, path)
        logger.info("Wrote %d EP-sharded module scores to %s", len(cpu_results), path)

    return results
