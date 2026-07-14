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

"""Normalize Puzzletron runtime and score artifacts into one-axis sweep records."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "AXIS_SPECS",
    "AxisSpec",
    "SweepRecord",
    "anchor_fields",
    "automatic_anchors",
    "curve_warnings",
    "load_replace_block_records",
    "load_vllm_records",
    "metric_direction",
    "observed_axes",
    "sample_layers",
    "write_records_csv",
]


@dataclass(frozen=True)
class AxisSpec:
    axis_id: str
    label: str
    kind: str
    field: str
    coupled_fields: tuple[str, ...] = ()

    @property
    def ignored_fields(self) -> tuple[str, ...]:
        return tuple(f"{self.kind}.{field}" for field in (self.field, *self.coupled_fields))


@dataclass(frozen=True)
class SweepRecord:
    source_kind: str
    source_path: str
    block_config: dict[str, Any]
    fields: dict[str, Any]
    axes: dict[str, float]
    metrics: dict[str, float]
    layer_idx: int | None = None
    profile_id: str | None = None
    profile: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


AXIS_SPECS: dict[str, AxisSpec] = {
    "ffn_intermediate": AxisSpec(
        "ffn_intermediate", "FFN intermediate size", "ffn", "intermediate_size"
    ),
    "kv_groups": AxisSpec(
        "kv_groups", "Attention KV groups", "attention", "num_kv_heads", ("num_query_heads",)
    ),
    "q_heads_per_group": AxisSpec(
        "q_heads_per_group",
        "Query heads per KV group",
        "attention",
        "q_heads_per_group",
        ("num_query_heads",),
    ),
    "qk_head_dim": AxisSpec(
        "qk_head_dim", "Attention Q/K head dimension", "attention", "qk_head_dim"
    ),
    "moe_experts": AxisSpec("moe_experts", "MoE experts", "moe", "num_experts"),
    "moe_expert_intermediate": AxisSpec(
        "moe_expert_intermediate", "MoE expert intermediate size", "moe", "expert_intermediate_size"
    ),
    "moe_shared_expert_intermediate": AxisSpec(
        "moe_shared_expert_intermediate",
        "MoE shared-expert intermediate size",
        "moe",
        "shared_expert_intermediate_size",
    ),
    "moe_top_k": AxisSpec("moe_top_k", "MoE top-k", "moe", "top_k"),
    "moe_latent_dim": AxisSpec("moe_latent_dim", "MoE latent dimension", "moe", "latent_dim"),
    "mamba_heads": AxisSpec("mamba_heads", "Mamba heads", "mamba", "num_heads"),
    "mamba_head_dim": AxisSpec("mamba_head_dim", "Mamba head dimension", "mamba", "head_dim"),
    "mamba_state_dim": AxisSpec("mamba_state_dim", "Mamba state dimension", "mamba", "state_dim"),
    "gdn_key_groups": AxisSpec(
        "gdn_key_groups", "GDN key groups", "mamba", "gdn_key_groups", ("num_groups", "num_heads")
    ),
    "gdn_value_heads_per_group": AxisSpec(
        "gdn_value_heads_per_group",
        "GDN value heads per group",
        "mamba",
        "gdn_value_heads_per_group",
        ("num_heads",),
    ),
    "gdn_key_head_dim": AxisSpec(
        "gdn_key_head_dim", "GDN key head dimension", "mamba", "state_dim"
    ),
    "gdn_value_head_dim": AxisSpec(
        "gdn_value_head_dim", "GDN value head dimension", "mamba", "head_dim"
    ),
}

_LEGACY_CONSTRUCTORS = {
    "BlockConfig",
    "AttentionConfig",
    "FFNConfig",
    "Llama4AttentionConfig",
    "MambaConfig",
    "MoEConfig",
}
_PROFILE_FIELDS = (
    "gpu",
    "batch_size",
    "prefill_seq_len",
    "generation_seq_len",
    "n_embd",
    "n_head",
    "vocab_size",
    "use_cuda_graph",
    "weights_dtype",
    "activations_dtype",
    "kv_cache_dtype",
    "runtime_granularity",
    "runtime_backend",
    "vllm_args",
    "num_iters",
    "num_warmup_iters",
    "max_num_seqs",
    "repeat_block_n_times",
)
_VLLM_METRIC_FIELDS = (
    "runtime_ms",
    "prefill_runtime_ms",
    "decode_runtime_ms",
    "decode_runtime_ms_per_token",
    "weight_memory_mib",
    "kv_cache_bytes_per_token",
    "state_cache_bytes_per_sequence",
    "prefill_flops",
    "decode_flops",
)


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _stable_id(prefix: str, value: Any) -> str:
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _parse_legacy_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool)) or node.value is None:
            return node.value
        raise ValueError(f"Unsupported literal in legacy block config: {type(node.value).__name__}")
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_parse_legacy_node(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            str(_parse_legacy_node(key)): _parse_legacy_node(value)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _parse_legacy_node(node.operand)
        if isinstance(value, (int, float)):
            return -value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in _LEGACY_CONSTRUCTORS or node.args:
            raise ValueError(f"Unsupported constructor in legacy block config: {name}")
        if any(keyword.arg is None for keyword in node.keywords):
            raise ValueError("Legacy block configs may not contain **kwargs")
        result = {keyword.arg: _parse_legacy_node(keyword.value) for keyword in node.keywords}
        if name != "BlockConfig":
            kind = {
                "AttentionConfig": "attention",
                "FFNConfig": "ffn",
                "MambaConfig": "mamba",
                "MoEConfig": "moe",
            }.get(name)
            if kind is not None:
                result.setdefault("kind", kind)
        return result
    raise ValueError(f"Unsupported syntax in legacy block config: {type(node).__name__}")


def parse_legacy_block_config(value: str) -> dict[str, Any]:
    """Safely decode the historical ``str(BlockConfig)`` runtime-stat key."""

    try:
        expression = ast.parse(value, mode="eval").body
    except SyntaxError as error:
        raise ValueError("Malformed legacy BlockConfig key") from error
    decoded = _parse_legacy_node(expression)
    if not isinstance(decoded, dict) or "subblock_configs" not in decoded:
        raise ValueError("Legacy runtime key is not a BlockConfig")
    return _canonical(decoded)


def _subblocks(block_config: dict[str, Any]) -> list[dict[str, Any]]:
    values = block_config.get("subblock_configs") or []
    return [dict(value) for value in values if isinstance(value, dict)]


def _subblock(block_config: dict[str, Any], kind: str) -> dict[str, Any] | None:
    return next((value for value in _subblocks(block_config) if value.get("kind") == kind), None)


def _flatten_fields(block_config: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for subblock in _subblocks(block_config):
        kind = str(subblock.get("kind", "unknown"))
        for key, value in sorted(subblock.items()):
            if key in {"kind", "name"} or isinstance(value, (dict, list)):
                continue
            fields[f"{kind}.{key}"] = value
        if kind == "attention" and subblock.get("num_kv_heads"):
            fields["attention.q_heads_per_group"] = (
                float(subblock["num_query_heads"]) / float(subblock["num_kv_heads"])
                if subblock.get("num_query_heads") is not None
                else None
            )
        if kind == "mamba" and subblock.get("num_groups"):
            fields["mamba.gdn_key_groups"] = subblock["num_groups"]
            fields["mamba.gdn_value_heads_per_group"] = (
                float(subblock["num_heads"]) / float(subblock["num_groups"])
                if subblock.get("num_heads") is not None
                else None
            )
    return {key: value for key, value in fields.items() if value is not None}


def _axis_value(block_config: dict[str, Any], spec: AxisSpec) -> float | None:
    subblock = _subblock(block_config, spec.kind)
    if subblock is None:
        return None
    if bool(subblock.get("no_op", False)):
        return 0.0
    if spec.field == "q_heads_per_group":
        query = subblock.get("num_query_heads")
        kv = subblock.get("num_kv_heads")
        if query is None or not kv:
            return None
        return float(query) / float(kv)
    if spec.field == "gdn_key_groups":
        value = subblock.get("num_groups")
    elif spec.field == "gdn_value_heads_per_group":
        heads = subblock.get("num_heads")
        groups = subblock.get("num_groups")
        value = None if heads is None or not groups else float(heads) / float(groups)
    else:
        value = subblock.get(spec.field)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _preferred_axis_ids(puzzle_dir: Path | None) -> set[str] | None:
    if puzzle_dir is None:
        return None
    candidate_path = puzzle_dir / "candidate_library.json"
    if not candidate_path.is_file():
        return None
    try:
        raw = json.loads(candidate_path.read_text())
    except (OSError, ValueError):
        return None
    axes = {
        str(axis)
        for candidate in raw.get("candidates", [])
        for axis in ((candidate.get("metadata") or {}).get("slice_axes") or {})
    }
    return axes or None


def _axis_values(block_config: dict[str, Any], preferred: set[str] | None) -> dict[str, float]:
    values: dict[str, float] = {}
    prefer_gdn = preferred is not None and any(axis.startswith("gdn_") for axis in preferred)
    for axis_id, spec in AXIS_SPECS.items():
        if preferred is not None and axis_id not in preferred:
            continue
        if prefer_gdn and axis_id.startswith("mamba_"):
            continue
        if not prefer_gdn and preferred is not None and axis_id.startswith("gdn_"):
            continue
        value = _axis_value(block_config, spec)
        if value is not None and math.isfinite(value):
            values[axis_id] = value
    return values


def _profile(args: dict[str, Any]) -> dict[str, Any]:
    return {key: _canonical(args.get(key)) for key in _PROFILE_FIELDS if args.get(key) is not None}


def load_vllm_records(
    stats_path: str | Path,
    *,
    puzzle_dir: str | Path | None = None,
    issues: list[dict[str, Any]] | None = None,
) -> list[SweepRecord]:
    """Read runtime-enabled block or sparse-subblock metric records."""

    stats_path = Path(stats_path).resolve()
    raw = json.loads(stats_path.read_text())
    entries = raw if isinstance(raw, list) else [raw]
    puzzle_path = Path(puzzle_dir).resolve() if puzzle_dir is not None else stats_path.parent
    preferred = _preferred_axis_ids(puzzle_path)
    records: list[SweepRecord] = []
    for entry_index, entry in enumerate(entries):
        args = dict(entry.get("args") or {})
        if not args.get("runtime_stats"):
            continue
        profile = _profile(args)
        profile_id = _stable_id("runtime", profile)
        structured = entry.get("block_runtime_records") or []
        runtime_rows: list[
            tuple[dict[str, Any], dict[str, float], str, dict[str, Any]]
        ] = []

        def extract_metrics(row: dict[str, Any], fragment: str) -> dict[str, float]:
            metrics: dict[str, float] = {}
            for key in _VLLM_METRIC_FIELDS:
                value = row.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    metrics[key] = float(value)
                elif value is not None and issues is not None:
                    issues.append(
                        {
                            "source_path": f"{stats_path}#entry={entry_index}&{fragment}",
                            "metric": key,
                            "warning": "nonfinite_metric",
                        }
                    )
            return metrics

        for row_index, row in enumerate(structured):
            block_config = row.get("block_config")
            if not isinstance(block_config, dict):
                continue
            fragment = f"record={row_index}"
            metrics = extract_metrics(row, fragment)
            if metrics:
                runtime_rows.append(
                    (
                        block_config,
                        metrics,
                        fragment,
                        dict(row.get("additive_metric_provenance") or {}),
                    )
                )
        for row_index, row in enumerate(entry.get("subblocks") or []):
            subblock_config = row.get("subblock_config")
            if not isinstance(subblock_config, dict):
                continue
            fragment = f"subblock={row_index}"
            metrics = extract_metrics(row, fragment)
            if metrics:
                runtime_rows.append(
                    (
                        {"subblock_configs": [subblock_config]},
                        metrics,
                        fragment,
                        dict(row.get("additive_metric_provenance") or {}),
                    )
                )
        if not runtime_rows:
            for row_index, (key, value) in enumerate((entry.get("block_runtimes") or {}).items()):
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    if issues is not None:
                        issues.append(
                            {
                                "source_path": f"{stats_path}#entry={entry_index}&legacy={row_index}",
                                "metric": "runtime_ms",
                                "warning": "nonfinite_metric",
                            }
                        )
                    continue
                runtime_rows.append(
                    (
                        parse_legacy_block_config(str(key)),
                        {"runtime_ms": float(value)},
                        f"legacy={row_index}",
                        {"runtime_ms": "legacy_vllm_measured"},
                    )
                )
        for block_config, metrics, fragment, metric_provenance in runtime_rows:
            block_config = _canonical(block_config)
            records.append(
                SweepRecord(
                    source_kind="vllm",
                    source_path=f"{stats_path}#entry={entry_index}&{fragment}",
                    block_config=block_config,
                    fields=_flatten_fields(block_config),
                    axes=_axis_values(block_config, preferred),
                    metrics=metrics,
                    profile_id=profile_id,
                    profile=profile,
                    provenance={
                        "stats_entry": entry_index,
                        "metrics": metric_provenance,
                    },
                )
            )
    if not records:
        raise ValueError(f"No block-level vLLM runtime records found in {stats_path}")
    return records


def _finite_metrics(
    raw: dict[str, Any],
    *,
    issues: list[dict[str, Any]] | None = None,
    source_path: str | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    def add_metric(name: str, value: Any) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            metrics[name] = float(value)
        elif value is not None and issues is not None:
            issues.append(
                {
                    "source_path": source_path,
                    "metric": name,
                    "warning": "nonfinite_metric",
                }
            )

    for name, aggregate in raw.items():
        if not isinstance(aggregate, dict) or "avg" not in aggregate:
            continue
        add_metric(str(name), aggregate.get("avg"))
    nested = raw.get("metrics") or {}
    if isinstance(nested, dict):
        for name, value in nested.items():
            if isinstance(value, dict) and "avg" in value:
                value = value.get("avg")
            add_metric(str(name), value)
    return metrics


def _replacement(raw: dict[str, Any]) -> dict[str, Any] | None:
    solution = raw.get("puzzle_solution") or raw.get("solution") or raw
    candidates = [
        solution.get("single_sequence_replacement") if isinstance(solution, dict) else None,
        raw.get("single_sequence_replacement"),
        raw.get("candidate"),
        (raw.get("request") or {}).get("candidate")
        if isinstance(raw.get("request"), dict)
        else None,
    ]
    return next((dict(value) for value in candidates if isinstance(value, dict)), None)


def _replacement_config(replacement: dict[str, Any]) -> dict[str, Any] | None:
    children = replacement.get("child_block_configs") or []
    if children and isinstance(children[0], dict):
        return dict(children[0])
    value = replacement.get("block_config") or replacement.get("config")
    return dict(value) if isinstance(value, dict) else None


def _score_entries(value: Any, pointer: str = "") -> Iterable[tuple[dict[str, Any], str]]:
    if isinstance(value, dict):
        replacement = _replacement(value)
        if replacement is not None and _replacement_config(replacement) is not None:
            yield value, pointer
            return
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                yield from _score_entries(child, f"{pointer}/{escaped}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _score_entries(child, f"{pointer}/{index}")


def load_replace_block_records(
    scores_dir: str | Path,
    *,
    puzzle_dir: str | Path | None = None,
    issues: list[dict[str, Any]] | None = None,
) -> list[SweepRecord]:
    """Read replace-one block or subblock result files with finite aggregate metrics."""

    scores_dir = Path(scores_dir).resolve()
    puzzle_path = Path(puzzle_dir).resolve() if puzzle_dir is not None else scores_dir.parent
    preferred = _preferred_axis_ids(puzzle_path)
    records: list[SweepRecord] = []
    for path in sorted(scores_dir.rglob("*.json")):
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for result, pointer in _score_entries(raw):
            replacement = _replacement(result)
            if replacement is None:
                continue
            block_config = _replacement_config(replacement)
            parents = replacement.get("parent_layer_indices") or []
            source_path = f"{path}#{pointer}" if pointer else str(path)
            metrics = _finite_metrics(result, issues=issues, source_path=source_path)
            if block_config is None or len(parents) != 1 or not metrics:
                continue
            block_config = _canonical(block_config)
            distributed = result.get("distributed_evaluation") or {}
            provenance = {
                key: distributed.get(key, result.get(key))
                for key in ("campaign_id", "candidate_id", "request_id")
                if distributed.get(key, result.get(key)) is not None
            }
            if pointer:
                provenance["artifact_pointer"] = pointer
            records.append(
                SweepRecord(
                    source_kind="replace_block",
                    source_path=source_path,
                    block_config=block_config,
                    fields=_flatten_fields(block_config),
                    axes=_axis_values(block_config, preferred),
                    metrics=metrics,
                    layer_idx=int(parents[0]),
                    provenance=provenance,
                )
            )
    if not records:
        raise ValueError(f"No replace-one result records found in {scores_dir}")
    return records


def observed_axes(records: Iterable[SweepRecord]) -> list[str]:
    values: dict[str, set[float]] = {}
    for record in records:
        for axis, value in record.axes.items():
            values.setdefault(axis, set()).add(float(value))
    return [axis for axis in AXIS_SPECS if len(values.get(axis, ())) >= 2]


def anchor_fields(record: SweepRecord, axis_id: str) -> dict[str, Any]:
    spec = AXIS_SPECS[axis_id]
    ignored = set(spec.ignored_fields) | {f"{spec.kind}.no_op"}
    return {key: value for key, value in record.fields.items() if key not in ignored}


def _is_axis_noop(record: SweepRecord, axis_id: str) -> bool:
    spec = AXIS_SPECS[axis_id]
    return record.fields.get(f"{spec.kind}.no_op") is True


def _matches_anchor(record: SweepRecord, axis_id: str, anchor: dict[str, Any]) -> bool:
    if axis_id not in record.axes:
        return False
    if _is_axis_noop(record, axis_id):
        prefix = f"{AXIS_SPECS[axis_id].kind}."
        return all(
            record.fields.get(key) == value
            for key, value in anchor.items()
            if not key.startswith(prefix)
        )
    return anchor_fields(record, axis_id) == anchor


def automatic_anchors(
    records: Iterable[SweepRecord], axis_id: str, *, count: int = 3
) -> list[dict[str, Any]]:
    groups: dict[str, tuple[dict[str, Any], set[float]]] = {}
    for record in records:
        value = record.axes.get(axis_id)
        if value is None or _is_axis_noop(record, axis_id):
            continue
        anchor = anchor_fields(record, axis_id)
        key = json.dumps(_canonical(anchor), sort_keys=True, separators=(",", ":"))
        groups.setdefault(key, (anchor, set()))[1].add(float(value))
    eligible = [(anchor, values) for anchor, values in groups.values() if len(values) >= 2]
    if not eligible:
        return []

    def capacity(item: tuple[dict[str, Any], set[float]]) -> tuple[float, str]:
        anchor, _ = item
        score = sum(
            float(value)
            for value in anchor.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )
        return score, json.dumps(_canonical(anchor), sort_keys=True)

    ordered = sorted(eligible, key=capacity)
    indices = [len(ordered) - 1, len(ordered) // 2, 0]
    selected: list[dict[str, Any]] = []
    for index in indices:
        anchor = ordered[index][0]
        if anchor not in selected:
            selected.append(anchor)
        if len(selected) >= max(1, count):
            break
    return selected


def records_for_anchor(
    records: Iterable[SweepRecord], axis_id: str, anchor: dict[str, Any]
) -> list[SweepRecord]:
    return [record for record in records if _matches_anchor(record, axis_id, anchor)]


def sample_layers(layers: Iterable[int], count: int = 5) -> list[int]:
    values = sorted(set(int(layer) for layer in layers))
    if len(values) <= count:
        return values
    if count <= 1:
        return values[:1]
    indices = {round(index * (len(values) - 1) / (count - 1)) for index in range(count)}
    return [values[index] for index in sorted(indices)]


def metric_direction(metric: str) -> str:
    return (
        "higher"
        if "accuracy" in metric.lower() or metric.lower().startswith(("hit", "topk"))
        else "lower"
    )


def _rank(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for index in ordered[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    xr, yr = _rank(xs), _rank(ys)
    xm, ym = sum(xr) / len(xr), sum(yr) / len(yr)
    numerator = sum((x - xm) * (y - ym) for x, y in zip(xr, yr))
    denominator = math.sqrt(sum((x - xm) ** 2 for x in xr) * sum((y - ym) ** 2 for y in yr))
    return None if denominator == 0 else numerator / denominator


def curve_warnings(
    points: Iterable[tuple[float, float]],
    *,
    expected: str,
    relative_tolerance: float,
) -> dict[str, Any]:
    grouped: dict[float, list[float]] = {}
    for x, y in points:
        if math.isfinite(x) and math.isfinite(y):
            grouped.setdefault(float(x), []).append(float(y))
    ordered = sorted((x, sum(ys) / len(ys), ys) for x, ys in grouped.items())
    warnings: list[str] = []
    if len(ordered) < 2:
        warnings.append("fewer_than_two_points")
    if 0.0 not in grouped:
        warnings.append("missing_no_op")
    if any(
        max(ys) - min(ys) > max(1e-12, relative_tolerance * max(abs(sum(ys) / len(ys)), 1e-12))
        for _, _, ys in ordered
    ):
        warnings.append("inconsistent_duplicates")
    ys = [value for _, value, _ in ordered]
    if len(ys) >= 2:
        scale = max(max(abs(value) for value in ys), 1e-12)
        if max(ys) - min(ys) <= relative_tolerance * scale:
            warnings.append("flat_curve")
        for previous, current in zip(ys, ys[1:]):
            tolerance = relative_tolerance * max(abs(previous), abs(current), 1e-12)
            violation = (
                current < previous - tolerance
                if expected == "higher"
                else current > previous + tolerance
            )
            if violation:
                warnings.append("direction_violation")
                break
    correlation = _spearman([x for x, _, _ in ordered], ys)
    corrected = (
        correlation if expected == "higher" else (-correlation if correlation is not None else None)
    )
    return {
        "points": [{"x": x, "y": y, "duplicates": len(raw)} for x, y, raw in ordered],
        "warnings": warnings,
        "direction_corrected_spearman": corrected,
    }


def write_records_csv(path: str | Path, records: Iterable[SweepRecord]) -> Path:
    path = Path(path)
    rows = [record.to_dict() for record in records]
    metric_names = sorted({name for row in rows for name in row["metrics"]})
    axis_names = sorted({name for row in rows for name in row["axes"]})
    fieldnames = [
        "source_kind",
        "source_path",
        "layer_idx",
        "profile_id",
        *axis_names,
        *metric_names,
        "block_config",
        "profile",
        "provenance",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_kind": row["source_kind"],
                    "source_path": row["source_path"],
                    "layer_idx": row["layer_idx"],
                    "profile_id": row["profile_id"],
                    **row["axes"],
                    **row["metrics"],
                    "block_config": json.dumps(row["block_config"], sort_keys=True),
                    "profile": json.dumps(row["profile"], sort_keys=True),
                    "provenance": json.dumps(row["provenance"], sort_keys=True),
                }
            )
    return path
