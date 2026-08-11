# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort progress summaries from stage logs and durable artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "summarize_active_progress",
    "summarize_log_progress",
    "summarize_stage_artifacts",
]

_VLLM_TQDM = re.compile(
    r"Benchmarking runtime shard (?P<shard>\d+)/(?P<shards>\d+) "
    r"\((?P<assigned>\d+)/(?P<total>\d+) specs\).*?"
    r"\|\s*(?P<done>\d+)/(?P=assigned)\b"
)
_VLLM_TOTAL = re.compile(
    r"Computing (?:runtime for (?P<subblocks>\d+) subblocks|"
    r"block-level runtime for (?P<blocks>\d+) block configs) "
    r"\((?P<specs>\d+) unique benchmarks\)"
)
_SUBBLOCK_KINDS = ("attention", "mla", "mamba", "ffn", "moe")
_WIDTH_ITER = re.compile(
    r"\[activation/automodel\] iter (?P<current>\d+)/(?P<total>\d+)\b"
)
_WIDTH_TARGET = re.compile(
    r"\[activation/automodel\] entering calibration loop: target (?P<total>\d+) iteration"
)
_SORT_SHARD_COMPLETE = re.compile(
    r"\[sorted_teacher\] shard complete .*?"
    r"shard=(?P<shard>model-(?P<index>\d+)-of-(?P<total>\d+)\.safetensors)"
)
_PARENT_SWEEP_LOAD = re.compile(
    r"\[solution/automodel\] parent sweep load \| "
    r"role=(?P<role>\S+) .*?solutions=(?P<solutions>\d+) pending=(?P<pending>\d+)"
)
_PARENT_SWEEP_CANDIDATE = re.compile(
    r"\[solution/automodel\] parent sweep candidate \| "
    r"role=(?P<role>\S+) solution=(?P<solution>\d+)\b"
)
_PARENT_SWEEP_EQUIVALENCE = re.compile(
    r"\[solution/automodel\] parent sweep equivalence \| role=(?P<role>\S+)"
)
_BYPASS_PROBE = re.compile(
    r"\[bypass/automodel\] running fixed-batch overfit acceptance probe "
    r"mode=(?P<mode>\S+) \((?P<index>\d+)/(?P<count>\d+)\) for "
    r"(?P<steps>\d+) steps"
)
_BYPASS_STEP = re.compile(
    r"\[bypass/automodel\] step=(?P<current>\d+)/(?P<total>\d+) "
    r"loss=(?P<loss>[^\s]+)"
)
_REPLACEMENT_POOL_READY = re.compile(
    r"\[replacement-pool\] ready workers: (?P<ready>\d+)/(?P<total>\d+)"
)


def _tail_text(path: Path, *, max_bytes: int = 65536) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="ignore")


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _count_files(path: Path, pattern: str) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.glob(pattern) if item.is_file())


def _count_rglob(path: Path, pattern: str) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.rglob(pattern) if item.is_file())


def _nested(config: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _teacher_removable_count(puzzle_dir: Path) -> int | None:
    config = _read_json(puzzle_dir / "ckpts" / "teacher" / "config.json")
    if not isinstance(config, Mapping):
        return None
    blocks = config.get("block_configs")
    if not isinstance(blocks, list):
        return None
    count = 0
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        subblocks = block.get("subblock_configs", block.get("subblocks", []))
        if not isinstance(subblocks, list):
            continue
        for subblock in subblocks:
            if isinstance(subblock, Mapping) and not subblock.get("no_op", False):
                count += 1
    return count or None


def _depth_progress(puzzle_dir: Path, config: Mapping[str, Any] | None) -> str | None:
    root = Path(puzzle_dir)
    iterative = root / "depth" / "iterative"
    if not iterative.is_dir():
        return None

    trajectory = _read_json(iterative / "trajectory.json")
    selected = 0
    max_removals = int(
        _nested(config, "depth_importance", "max_removals", default=0)
        or _nested(config, "depth", "max_removals", default=0)
        or 0
    )
    available = int(
        _nested(config, "depth_importance", "expected_initial_sublayers", default=0) or 0
    )
    if isinstance(trajectory, Mapping):
        selected = len(trajectory.get("selected") or [])
        max_removals = int(trajectory.get("max_removals") or max_removals or 0)
        available = int(trajectory.get("available_count") or available or 0)
    if available <= 0:
        available = _teacher_removable_count(root) or 0
    if max_removals <= 0:
        return None

    iteration_dirs = sorted(
        path for path in iterative.glob("iteration_*") if path.is_dir()
    )
    if selected >= max_removals and iteration_dirs:
        return f"removing layer {max_removals} out of {max_removals}, current progress complete"

    current_iteration = selected
    if iteration_dirs:
        newest = iteration_dirs[-1]
        match = re.fullmatch(r"iteration_(\d+)", newest.name)
        if match is not None:
            current_iteration = int(match.group(1))
        candidates_done = _count_files(newest, "candidate_*.json")
        expected = max(available - current_iteration, candidates_done)
        if (newest / "ranking.json").is_file():
            return (
                f"removing layer {min(current_iteration + 1, max_removals)} out of "
                f"{max_removals}, current progress {expected}/{expected}"
            )
        layer = min(current_iteration + 1, max_removals)
        return (
            f"removing layer {layer} out of {max_removals}, "
            f"current progress {candidates_done}/{expected}"
        )

    return f"removing layer 1 out of {max_removals}, current progress 0/{available or '?'}"


def _width_progress_from_artifacts(puzzle_dir: Path, config: Mapping[str, Any] | None) -> str | None:
    root = Path(puzzle_dir)
    candidates: list[Path] = []
    configured = _nested(config, "pruning", "activations_log_dir")
    if configured:
        candidates.append(Path(str(configured)) / ".native_resume" / "progress.json")
    candidates.extend(
        [
            root / "pruning" / "pruning_scores" / "automodel" / "all_axes" / ".native_resume" / "progress.json",
            root / "activations_log" / ".native_resume" / "progress.json",
            root / "width" / ".native_resume" / "progress.json",
        ]
    )
    for path in candidates:
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            continue
        current = int(payload.get("next_step") or 0)
        total = int(payload.get("total") or 0)
        if total > 0:
            return f"minibatch {current}/{total}"
    return None


def _width_progress_from_logs(log_paths: Sequence[str]) -> str | None:
    best: tuple[int, int] | None = None
    target_total: int | None = None
    for log_path in log_paths:
        text = _tail_text(Path(log_path), max_bytes=131072).replace("\r", "\n")
        if not text:
            continue
        for match in _WIDTH_TARGET.finditer(text):
            target_total = int(match.group("total"))
        for match in _WIDTH_ITER.finditer(text):
            current = int(match.group("current"))
            total = int(match.group("total"))
            if best is None or current > best[0]:
                best = (current, total)
    if best is not None:
        return f"minibatch {best[0]}/{best[1]}"
    if target_total is not None:
        return f"minibatch 0/{target_total}"
    return None


def _combined_log_text(log_paths: Sequence[str], *, max_bytes: int = 1048576) -> str:
    return "\n".join(
        text
        for log_path in log_paths
        if (text := _tail_text(Path(log_path), max_bytes=max_bytes))
    ).replace("\r", "\n")


def _checkpoint_shard_progress(log_paths: Sequence[str], *, label: str) -> str | None:
    completed: set[str] = set()
    total = 0
    for match in _SORT_SHARD_COMPLETE.finditer(_combined_log_text(log_paths)):
        completed.add(match.group("shard"))
        total = max(total, int(match.group("total")))
    if not completed:
        return None
    return f"{label} shards {len(completed)}/{total or '?'}"


def _parent_sweep_progress(log_paths: Sequence[str]) -> str | None:
    text = _combined_log_text(log_paths)
    loads = list(_PARENT_SWEEP_LOAD.finditer(text))
    if not loads:
        return None
    load = loads[-1]
    role = load.group("role")
    solutions = int(load.group("solutions"))
    pending = int(load.group("pending"))
    phase_text = text[load.start() :]
    candidates = [
        match
        for match in _PARENT_SWEEP_CANDIDATE.finditer(phase_text)
        if match.group("role") == role
    ]
    if candidates:
        current = int(candidates[-1].group("solution")) + 1
        return f"{role} parent: scoring case {current}/{solutions}"
    equivalence = [
        match
        for match in _PARENT_SWEEP_EQUIVALENCE.finditer(phase_text)
        if match.group("role") == role
    ]
    if equivalence:
        return f"{role} parent: checking full-width equivalence"
    if pending:
        return f"loading {role} parent ({pending} case(s) pending)"
    return f"validating {role} parent"


def _bypass_progress(log_paths: Sequence[str]) -> str | None:
    text = _combined_log_text(log_paths)
    probes = list(_BYPASS_PROBE.finditer(text))
    steps = list(_BYPASS_STEP.finditer(text))
    if steps:
        step = steps[-1]
        prefix = ""
        if probes:
            probe = probes[-1]
            prefix = (
                f"{probe.group('mode')} probe "
                f"{probe.group('index')}/{probe.group('count')}: "
            )
        return (
            f"{prefix}step {step.group('current')}/{step.group('total')}, "
            f"loss {step.group('loss')}"
        )
    if probes:
        probe = probes[-1]
        return (
            f"{probe.group('mode')} probe "
            f"{probe.group('index')}/{probe.group('count')}: "
            f"preparing {probe.group('steps')} steps"
        )
    if "[bypass/automodel] backend ACTIVE" in text:
        return "loading local-KD model"
    return None


def _replacement_progress(
    puzzle_dir: Path,
    log_paths: Sequence[str],
    config: Mapping[str, Any] | None,
) -> str | None:
    granularity = str(_nested(config, "replacement_scoring", "granularity", default="block"))
    solutions_key = (
        "subblock_solutions_path" if granularity == "subblock" else "block_solutions_path"
    )
    configured_solutions = _nested(config, "replacement_scoring", solutions_key) or _nested(
        config, "replacement_scoring", "solutions_path"
    )
    manifest_name = (
        Path(str(configured_solutions)).name
        if configured_solutions
        else (
            "single_subblock_replacement_solutions.json"
            if granularity == "subblock"
            else "single_sequence_replacement_solutions.json"
        )
    )
    output_key = "subblock_output_dir" if granularity == "subblock" else "block_output_dir"
    configured_output = _nested(config, "replacement_scoring", output_key) or _nested(
        config, "replacement_scoring", "output_dir"
    )
    output_name = (
        Path(str(configured_output)).name
        if configured_output
        else f"{Path(manifest_name).stem}--validation"
    )
    widths = _nested(config, "embedding_pruning", "widths", default=()) or ()
    scenario_roots = [
        puzzle_dir / "scenarios" / f"width-{int(width):04d}" / "depth-00"
        for width in widths
    ]
    if not scenario_roots:
        scenario_roots = sorted((puzzle_dir / "scenarios").glob("width-*/depth-00"))
    manifests = [root / manifest_name for root in scenario_roots]
    if not any(path.is_file() for path in manifests):
        manifests = [puzzle_dir / manifest_name]

    expected = completed = 0
    for manifest_path in manifests:
        solutions = _read_json(manifest_path)
        if not isinstance(solutions, list):
            continue
        expected += len(solutions)
        output_dir = manifest_path.parent / output_name
        completed += sum(
            (output_dir / f"solution_{index}.json").is_file()
            for index in range(len(solutions))
        )
    if expected:
        return f"scored {completed}/{expected} replacement candidates"

    configured = _nested(config, "replacement_scoring", "campaign_dir")
    roots = []
    if configured:
        root = Path(str(configured))
        roots.append(root if root.is_absolute() else Path(puzzle_dir) / root)
    roots.extend(
        [
            Path(puzzle_dir) / "distributed_eval" / "replacement_scoring",
            Path(puzzle_dir) / "distributed_eval" / "replace_block",
        ]
    )
    for root in roots:
        requests = _count_rglob(root / "journal" / "requests", "*.json")
        if requests <= 0:
            continue
        completed = _count_rglob(root / "journal" / "results", "*.json")
        failed = _count_rglob(root / "journal" / "terminal", "*.json")
        suffix = f", {failed} failed" if failed else ""
        return f"scored {completed}/{requests} replacement candidates{suffix}"

    ready = None
    for match in _REPLACEMENT_POOL_READY.finditer(_combined_log_text(log_paths)):
        ready = (int(match.group("ready")), int(match.group("total")))
    if ready is not None:
        return f"worker pool {ready[0]}/{ready[1]} ready"
    return None


def _post_mip_progress(
    puzzle_dir: Path,
    stage_id: str,
    config: Mapping[str, Any] | None,
) -> str | None:
    parts = stage_id.split(".", 2)
    if len(parts) != 3:
        return None
    _, flow_id, node_id = parts
    node_config = _nested(config, "post_mip", "flows", flow_id, "nodes", node_id)
    if not isinstance(node_config, Mapping):
        return None
    node_type = str(node_config.get("type") or "")
    if node_type in {"filter", "manual_filter"}:
        return None

    input_id = node_config.get("input")
    if not input_id:
        return None
    input_root = puzzle_dir / "artifacts" / "post_mip" / "nodes" / str(input_id)
    current = _read_json(input_root / "current.json")
    if not isinstance(current, Mapping) or not current.get("execution_identity"):
        return None
    candidate_set = _read_json(
        input_root
        / "executions"
        / str(current["execution_identity"])
        / "candidate_set.json"
    )
    if not isinstance(candidate_set, Mapping):
        return None
    revision_ids = candidate_set.get("revision_ids")
    if not isinstance(revision_ids, list) or not revision_ids:
        return None

    executions_root = (
        puzzle_dir / "artifacts" / "post_mip" / "nodes" / node_id / "executions"
    )
    executions = [path for path in executions_root.glob("post_mip_execution_*") if path.is_dir()]
    rows_by_revision: dict[str, Mapping[str, Any]] = {}
    if executions:
        execution_root = max(executions, key=lambda path: path.stat().st_mtime_ns)
        for shard_path in sorted((execution_root / "shards").glob("shard_*.json")):
            rows = _read_json(shard_path)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, Mapping) and row.get("input_revision_id"):
                    rows_by_revision[str(row["input_revision_id"])] = row

    labels = {
        "evaluation": "evaluated",
        "downstream_evaluation": "evaluated",
        "aiperf": "benchmarked",
        "global_kd": "distilled",
        "materialize": "materialized",
    }
    completed = len(rows_by_revision)
    failed = sum(row.get("status") == "failed" for row in rows_by_revision.values())
    timed_out = sum(
        row.get("status") == "timed_out" for row in rows_by_revision.values()
    )
    outcomes = []
    if failed:
        outcomes.append(f"{failed} failed")
    if timed_out:
        outcomes.append(f"{timed_out} timed out")
    suffix = f", {', '.join(outcomes)}" if outcomes else ""
    return (
        f"{labels.get(node_type, 'processed')} {completed}/{len(revision_ids)} candidates"
        f"{suffix}"
    )


def _unique_library_subblocks(library: list[Any]) -> tuple[int, int]:
    """Return ``(active_unique_subblocks, library_rows)`` from convert's library JSON.

    Convert writes one row per candidate *block* (paired ``*_config`` columns). Subblock
    mode benchmarks the unique column values, not one entry per library row.
    """

    unique: dict[str, set[str]] = {kind: set() for kind in _SUBBLOCK_KINDS}
    active = 0
    for row in library:
        if not isinstance(row, Mapping):
            continue
        for kind in _SUBBLOCK_KINDS:
            payload = row.get(f"{kind}_config")
            if not isinstance(payload, Mapping):
                continue
            identity = json.dumps(payload, sort_keys=True)
            if identity in unique[kind]:
                continue
            unique[kind].add(identity)
            if not payload.get("no_op", False):
                active += 1
    return active, len(library)


def _estimate_vllm_totals(
    puzzle_dir: Path,
    *,
    granularity: str,
) -> tuple[int, int]:
    """Estimate ``(unique_items, unique_specs)`` before logs publish the official counts."""

    library = _read_json(puzzle_dir / "subblock_library.json")
    if not isinstance(library, list) or not library:
        return 0, 0
    active_subblocks, rows = _unique_library_subblocks(library)
    if granularity in {"block", "blocks"}:
        # Block mode times each library block (or the attn×ffn Cartesian product derived
        # from it). Convert's library is already the block candidate list here.
        return rows, 2 * rows
    # Subblock mode: one short+long slope pair per unique active subblock.
    return active_subblocks, 2 * active_subblocks


def _model_hidden_size(config: Any) -> int | None:
    if not isinstance(config, Mapping):
        return None
    for candidate in (config, config.get("text_config")):
        if isinstance(candidate, Mapping) and candidate.get("hidden_size") is not None:
            try:
                return int(candidate["hidden_size"])
            except (TypeError, ValueError):
                return None
    return None


def _vllm_widths(
    puzzle_dir: Path,
    config: Mapping[str, Any] | None,
) -> tuple[int, ...]:
    """Return configured runtime widths plus the always-measured teacher width."""

    widths: list[int] = []
    for configured in (
        _nested(config, "vllm_stats", "model_hidden_sizes", default=()),
        _nested(config, "embedding_pruning", "widths", default=()),
    ):
        if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
            continue
        for width in configured:
            try:
                widths.append(int(width))
            except (TypeError, ValueError):
                continue
    teacher_width = _model_hidden_size(_read_json(puzzle_dir / "ckpts" / "teacher" / "config.json"))
    if teacher_width is not None:
        widths.append(teacher_width)
    return tuple(dict.fromkeys(widths))


def _completed_runtime_cache_specs(
    cache_dir: Path,
    *,
    generation_seq_len: int,
) -> int:
    """Count benchmark identities with both combined and prefill cache phases."""

    phases_by_spec: dict[str, set[int]] = {}
    if not cache_dir.is_dir():
        return 0
    for path in cache_dir.glob("*.json"):
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            continue
        identity = payload.get("cache_identity")
        if not isinstance(identity, Mapping):
            continue
        model_config = identity.get("model_config")
        benchmark_args = identity.get("benchmark_args")
        if not isinstance(model_config, Mapping) or not isinstance(benchmark_args, Mapping):
            continue
        try:
            output_len = int(benchmark_args["output_len"])
        except (KeyError, TypeError, ValueError):
            continue
        # One logical runtime spec invokes the same finalized model twice:
        # combined prefill+decode, then prefill-only (output_len=1). Strip only
        # phase-specific fields so those two files share one identity.
        phase_independent_args = {
            key: value
            for key, value in benchmark_args.items()
            if key not in {"output_len", "max_model_len", "effective_command"}
        }
        spec_identity = json.dumps(
            {
                "schema_version": identity.get("schema_version"),
                "model_config": model_config,
                "benchmark_args": phase_independent_args,
            },
            sort_keys=True,
            default=str,
        )
        phases_by_spec.setdefault(spec_identity, set()).add(output_len)
    expected_phases = {1, generation_seq_len}
    return sum(1 for phases in phases_by_spec.values() if expected_phases.issubset(phases))


def _vllm_progress(
    puzzle_dir: Path,
    log_paths: Sequence[str],
    config: Mapping[str, Any] | None,
) -> str | None:
    root = Path(puzzle_dir)
    widths = _vllm_widths(root, config)
    width_count = len(widths) or 1
    generation_seq_len = int(_nested(config, "vllm_stats", "generation_seq_len", default=1) or 1)
    validated = _completed_runtime_cache_specs(
        root / "runtime_cache",
        generation_seq_len=generation_seq_len,
    )
    total_specs = 0
    item_count = 0
    granularity = str(
        _nested(config, "vllm_stats", "runtime_stats", "granularity", default="")
        or _nested(config, "vllm_stats", "granularity", default="")
        or "subblock"
    ).lower()
    unit = "blocks" if granularity in {"block", "blocks"} else "subblocks"
    unit_from_config = bool(
        _nested(config, "vllm_stats", "runtime_stats", "granularity", default=None)
        or _nested(config, "vllm_stats", "granularity", default=None)
    )

    done_by_shard: dict[int, int] = {}
    for log_path in log_paths:
        text = _tail_text(Path(log_path), max_bytes=131072).replace("\r", "\n")
        if not text:
            continue
        for match in _VLLM_TOTAL.finditer(text):
            total_specs = int(match.group("specs"))
            if match.group("blocks"):
                item_count = int(match.group("blocks"))
                if not unit_from_config:
                    unit = "blocks"
            elif match.group("subblocks"):
                item_count = int(match.group("subblocks"))
                if not unit_from_config:
                    unit = "subblocks"
        for match in _VLLM_TQDM.finditer(text):
            shard = int(match.group("shard"))
            done_by_shard[shard] = max(
                done_by_shard.get(shard, 0),
                int(match.group("done")),
            )
            total_specs = int(match.group("total"))

    if done_by_shard:
        validated = max(validated, sum(done_by_shard.values()))
    if total_specs <= 0:
        item_count, total_specs = _estimate_vllm_totals(root, granularity=granularity)
    total_specs *= width_count
    if total_specs <= 0 and validated <= 0:
        return None
    if total_specs <= 0:
        return f"validated {validated} {unit}"
    if item_count > 0:
        width_suffix = f" × {width_count} widths" if width_count > 1 else ""
        return f"validated {validated}/{total_specs} specs ({item_count} {unit}{width_suffix})"
    return f"validated {validated}/{total_specs} specs"


def _tokenize_progress(
    puzzle_dir: Path,
    config: Mapping[str, Any] | None,
) -> str | None:
    """Report packed-token rows written so far across configured caches."""

    caches = _nested(config, "tokenize_data", "caches", default=None)
    if not isinstance(caches, list) or not caches:
        return None
    done = 0
    total = 0
    for cache in caches:
        if not isinstance(cache, Mapping):
            continue
        output = Path(str(cache.get("output") or ""))
        if not output.is_absolute():
            output = Path(puzzle_dir) / output
        num_samples = int(cache.get("num_samples") or 0)
        if num_samples <= 0:
            continue
        total += num_samples
        metadata = _read_json(output.with_suffix(output.suffix + ".json"))
        if isinstance(metadata, Mapping) and metadata.get("status") == "complete":
            done += num_samples
            continue
        progress_dir = output.parent / f".{output.name}.progress"
        rows_complete = 0
        if progress_dir.is_dir():
            for path in progress_dir.glob("worker_*.json"):
                payload = _read_json(path)
                if isinstance(payload, Mapping):
                    rows_complete += int(payload.get("rows_complete") or 0)
        done += min(rows_complete, num_samples)
    if total <= 0:
        return None
    return f"{done}/{total} samples tokenized"


def _post_width_progress(
    puzzle_dir: Path,
    stage_id: str,
    log_paths: Sequence[str],
    config: Mapping[str, Any] | None,
) -> str | None:
    if stage_id.startswith("post."):
        return _post_mip_progress(puzzle_dir, stage_id, config) or "preparing post-MIP node"
    if stage_id == "sort":
        return _checkpoint_shard_progress(log_paths, label="sorted checkpoint") or (
            "preparing sorted checkpoint"
        )
    if stage_id == "sort_sanity":
        parent = _parent_sweep_progress(log_paths)
        if parent:
            return parent
        text = _combined_log_text(log_paths)
        last_shard = None
        for last_shard in _SORT_SHARD_COMPLETE.finditer(text):
            pass
        if (
            last_shard is not None
            and (solution_position := text.rfind("[solution/automodel]")) > last_shard.end()
        ):
            return "running full-width equivalence checks"
        shards = _checkpoint_shard_progress(log_paths, label="reverse checkpoint")
        if shards:
            return shards
        if "[solution/automodel]" in text:
            return "running full-width equivalence checks"
        return "preparing sort equivalence"
    if stage_id == "width_sanity":
        return _parent_sweep_progress(log_paths) or "preparing width diagnostic"
    if stage_id == "slicing_sanity":
        return "validating dynamic-to-physical slice equivalence"
    if stage_id in {"bypass_sanity", "bypass"}:
        return _bypass_progress(log_paths) or "preparing local-KD model"
    if stage_id == "build_library":
        return "building replacement candidate library"
    if stage_id == "replacement_scoring":
        return (
            _replacement_progress(puzzle_dir, log_paths, config)
            or "starting replacement-scoring worker pool"
        )
    return None


def summarize_log_progress(work_id: str, log_paths: Sequence[str]) -> str | None:
    """Parse the newest useful progress signal from one attempt's logs."""

    stage_id = work_id.split(":", 1)[0]
    if stage_id == "convert":
        return None
    if stage_id == "tokenize_data":
        return None
    if stage_id == "vllm_stats":
        return _vllm_progress(Path("."), log_paths, None)
    if stage_id == "width_importance":
        return _width_progress_from_logs(log_paths)
    if stage_id == "depth_importance":
        return None
    return _post_width_progress(Path("."), stage_id, log_paths, None)


def summarize_stage_artifacts(
    puzzle_dir: Path,
    stage_id: str,
    *,
    config: Mapping[str, Any] | None = None,
    log_paths: Sequence[str] = (),
) -> str | None:
    """Summarize durable artifacts for stages that publish incremental outputs."""

    root = Path(puzzle_dir)
    if stage_id == "convert":
        return None
    if stage_id == "tokenize_data":
        return _tokenize_progress(root, config)
    if stage_id == "vllm_stats":
        return _vllm_progress(root, log_paths, config)
    if stage_id == "depth_importance":
        return _depth_progress(root, config)
    if stage_id == "width_importance":
        return _width_progress_from_artifacts(root, config) or _width_progress_from_logs(log_paths)
    return _post_width_progress(root, stage_id, log_paths, config)


def summarize_active_progress(
    *,
    puzzle_dir: Path,
    active: Mapping[str, tuple[object, str, str]],
    log_paths_by_work_id: Mapping[str, Sequence[str]],
    config: Mapping[str, Any] | None = None,
) -> list[str]:
    """Build short progress lines for the controller heartbeat."""

    lines: list[str] = []
    by_stage: dict[str, list[str]] = {}
    for _handle_id, (_handle, work_id, _attempt_id) in active.items():
        stage_id = work_id.split(":", 1)[0]
        if stage_id == "convert":
            continue
        by_stage.setdefault(stage_id, []).append(work_id)

    for stage_id, work_ids in sorted(by_stage.items()):
        stage_logs: list[str] = []
        for work_id in sorted(set(work_ids)):
            stage_logs.extend(log_paths_by_work_id.get(work_id, ()))
        artifact = summarize_stage_artifacts(
            puzzle_dir,
            stage_id,
            config=config,
            log_paths=stage_logs,
        )
        if artifact:
            lines.append(f"{stage_id}: {artifact}")
            continue
        for work_id in sorted(set(work_ids)):
            detail = summarize_log_progress(work_id, log_paths_by_work_id.get(work_id, ()))
            if detail:
                lines.append(f"{work_id}: {detail}")
    return lines
