#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run sharded persistent-server AIPerf sweeps for a solution registry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

CONCURRENCIES = (1, 4, 16, 64)
# Two full concurrency waves are enough to measure every configured point while
# avoiding redundant multi-hour traffic for long-context campaign workloads.
REQUEST_COUNTS = {1: 8, 4: 8, 16: 32, 64: 128}
TOPOLOGIES = (
    {
        "topology_id": "tp8-pp1-dp1-ep1-pcp1-dcp1",
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "gpu_count": 8,
    },
    {
        "topology_id": "tp4-pp2-dp1-ep1-pcp1-dcp1",
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 2,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "gpu_count": 8,
    },
    {
        "topology_id": "tp2-pp4-dp1-ep1-pcp1-dcp1",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 4,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "gpu_count": 8,
    },
    {
        "topology_id": "tp2-pp2-dp2-ep4-pcp1-dcp1",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 2,
        "data_parallel_size": 2,
        "enable_expert_parallel": True,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "gpu_count": 8,
    },
    {
        "topology_id": "tp1-pp2-dp4-ep4-pcp1-dcp1",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 2,
        "data_parallel_size": 4,
        "enable_expert_parallel": True,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "gpu_count": 8,
    },
    {
        "topology_id": "tp1-pp1-dp8-ep8-pcp1-dcp1",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 8,
        "enable_expert_parallel": True,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "gpu_count": 8,
    },
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def select_registry_solutions(
    registry: dict[str, Any], solution_ids: tuple[str, ...] | list[str]
) -> dict[str, Any]:
    """Return a registry restricted to explicitly requested solution IDs."""

    requested = tuple(str(solution_id) for solution_id in solution_ids)
    if not requested:
        return registry
    if len(requested) != len(set(requested)):
        raise ValueError(f"duplicate AIPerf solution IDs: {requested}")
    by_id = {str(row["solution_id"]): row for row in registry.get("solutions", ())}
    missing = [solution_id for solution_id in requested if solution_id not in by_id]
    if missing:
        raise ValueError(f"unknown AIPerf solution IDs: {missing}")
    return {**registry, "solutions": [by_id[solution_id] for solution_id in requested]}


def build_work_items(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "profile_id": registry["profile_id"],
            "solution_id": solution["solution_id"],
            "checkpoint": solution["checkpoint"],
            "topology_id": topology["topology_id"],
            "gpu_count": topology["gpu_count"],
            "topology": {
                key: value
                for key, value in topology.items()
                if key not in {"topology_id", "gpu_count"}
            },
        }
        for solution in registry["solutions"]
        for topology in TOPOLOGIES
    ]


def shard_work(
    items: list[dict[str, Any]], *, worker_index: int, worker_count: int
) -> list[dict[str, Any]]:
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError(
            f"invalid worker index/count: index={worker_index}, count={worker_count}"
        )
    return list(items[worker_index::worker_count])


def expected_result_count(
    registry: dict[str, Any], *, concurrencies: tuple[int, ...] = CONCURRENCIES
) -> int:
    """Return the complete AIPerf matrix size for a solution registry."""

    return len(registry["solutions"]) * len(TOPOLOGIES) * len(concurrencies)


def _registry(puzzle_dir: Path, profile_id: str) -> dict[str, Any]:
    path = puzzle_dir / "mip" / "profiles" / profile_id / "selected_solutions.json"
    return json.loads(path.read_text())


def _visible_gpu_ids() -> str:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible = [item.strip() for item in value.split(",") if item.strip()]
    if len(visible) == 8:
        return value
    local_id = int(os.environ.get("SLURM_LOCALID", "0"))
    start = 8 * local_id
    selected = visible[start : start + 8]
    if len(selected) != 8:
        raise RuntimeError(
            "each profile AIPerf worker requires a deterministic eight-GPU node; "
            f"SLURM_LOCALID={local_id} CUDA_VISIBLE_DEVICES={value!r}"
        )
    return ",".join(selected)


def run_worker(
    puzzle_dir: Path,
    *,
    profile_id: str,
    worker_index: int,
    worker_count: int,
    input_tokens: int,
    output_tokens: int,
    solution_ids: tuple[str, ...] = (),
    preflight: bool = False,
    concurrencies: tuple[int, ...] | None = None,
    request_count: int | None = None,
    benchmark_timeout: float = 7200,
) -> Path:
    # Worker execution needs the GPU stack; result merging intentionally remains
    # usable by the dependency-light login-node orchestrator.
    from modelopt.torch.puzzletron.benchmarks import run_aiperf_sweep

    registry = select_registry_solutions(_registry(puzzle_dir, profile_id), solution_ids)
    items = build_work_items(registry)
    if preflight:
        items = [
            item for item in items if item["solution_id"] in {"teacher", "h0512-d0"}
        ]
    work = shard_work(
        items,
        worker_index=worker_index,
        worker_count=worker_count,
    )
    workload_id = f"isl-{input_tokens}-osl-{output_tokens}"
    root = puzzle_dir / "artifacts" / "aiperf" / "profiles" / profile_id / workload_id
    if preflight:
        root = root / "preflight"
    rows = []
    for item in work:
        topology = dict(item["topology"])
        topology["extra_vllm_args"] = ["-cc.cudagraph_mode=NONE"]
        active_concurrencies = (1,) if preflight else (concurrencies or CONCURRENCIES)
        request_counts = (
            {1: 2}
            if preflight
            else {
                concurrency: (
                    int(request_count)
                    if request_count is not None
                    else int(REQUEST_COUNTS[concurrency])
                )
                for concurrency in active_concurrencies
            }
        )
        results = run_aiperf_sweep(
            item["checkpoint"],
            artifact_dir=root / item["solution_id"] / item["topology_id"],
            concurrencies=active_concurrencies,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            gpu_ids=_visible_gpu_ids(),
            topology=topology,
            request_counts=request_counts,
            solution_id=item["solution_id"],
            profile_id=profile_id,
            topology_id=item["topology_id"],
            executable=os.environ.get("AIPERF_EXECUTABLE", "aiperf"),
            endpoint_type="completions",
            use_server_token_count=True,
            seed=42,
            gpu_telemetry="pynvml",
            benchmark_timeout=benchmark_timeout,
        )
        rows.extend(result.model_dump(mode="json") for result in results)
    selection_suffix = ""
    if solution_ids:
        safe_ids = [
            "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in value
            )
            for value in solution_ids
        ]
        selection_suffix = "-" + "-".join(safe_ids)
    output = root / "workers" / f"worker_{worker_index:02d}{selection_suffix}.json"
    _atomic_json(
        output,
        {
            "worker_index": worker_index,
            "worker_count": worker_count,
            "work_items": work,
            "results": rows,
        },
    )
    return output


def merge_results(
    puzzle_dir: Path,
    *,
    profile_id: str,
    input_tokens: int,
    output_tokens: int,
    solution_ids: tuple[str, ...] = (),
    concurrencies: tuple[int, ...] = CONCURRENCIES,
) -> Path:
    root = (
        puzzle_dir
        / "artifacts"
        / "aiperf"
        / "profiles"
        / profile_id
        / f"isl-{input_tokens}-osl-{output_tokens}"
    )
    registry = select_registry_solutions(_registry(puzzle_dir, profile_id), solution_ids)
    selected_ids = {str(row["solution_id"]) for row in registry["solutions"]}
    selected_concurrencies = {int(value) for value in concurrencies}
    rows = []
    for path in sorted(root.glob("*/tp*/concurrency_*/puzzletron_aiperf_result.json")):
        row = json.loads(path.read_text())
        if (
            str(row.get("solution_id")) in selected_ids
            and int(row.get("concurrency", -1)) in selected_concurrencies
        ):
            rows.append(row)
    expected = expected_result_count(registry, concurrencies=concurrencies)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} AIPerf results, found {len(rows)} under {root}")
    identities = {
        (row["solution_id"], row["topology_id"], int(row["concurrency"])) for row in rows
    }
    if len(identities) != expected:
        raise RuntimeError("AIPerf result matrix has duplicate identities")
    for row in rows:
        if row["failures"] != 0:
            raise RuntimeError(f"AIPerf result has failures: {row}")
        metrics = row["metrics"]
        if round(metrics.get("input_sequence_length", -1)) != input_tokens:
            raise RuntimeError(f"AIPerf result has incorrect ISL: {row}")
        if round(metrics.get("output_sequence_length", -1)) != output_tokens:
            raise RuntimeError(f"AIPerf result has incorrect OSL: {row}")
    output = root / "aiperf_results.json"
    _atomic_json(
        output,
        {
            "version": 1,
            "profile_id": profile_id,
            "workload": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "concurrencies": list(concurrencies),
            "topologies": list(TOPOLOGIES),
            "results": rows,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle-dir", type=Path, required=True)
    parser.add_argument("--profile-id", default="params-080")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--solution-id", action="append", default=[])
    parser.add_argument("--concurrency", type=int, action="append", default=[])
    parser.add_argument("--request-count", type=int)
    parser.add_argument("--benchmark-timeout", type=float, default=7200)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        output = merge_results(
            args.puzzle_dir,
            profile_id=args.profile_id,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
            solution_ids=tuple(args.solution_id),
            concurrencies=tuple(args.concurrency) or CONCURRENCIES,
        )
    else:
        worker_index = (
            int(os.environ.get("SLURM_PROCID", "0"))
            if args.worker_index is None
            else args.worker_index
        )
        worker_count = (
            int(os.environ.get("SLURM_NTASKS", "1"))
            if args.worker_count is None
            else args.worker_count
        )
        output = run_worker(
            args.puzzle_dir,
            profile_id=args.profile_id,
            worker_index=worker_index,
            worker_count=worker_count,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
            solution_ids=tuple(args.solution_id),
            preflight=args.preflight,
            concurrencies=tuple(args.concurrency) or None,
            request_count=args.request_count,
            benchmark_timeout=args.benchmark_timeout,
        )
    print(output)


if __name__ == "__main__":
    main()
