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

"""Hardware-aware AutoQuantize latency look-up table (LUT).

This module implements the ``haq_latency_v1`` cost table used by the
hardware-aware AutoQuantize (HAQ) cost model. It is deliberately decoupled from
the AutoQuantize solver so it can be unit-tested in isolation and reused by the
offline benchmark-to-LUT converter.

Three responsibilities live here:

1. :class:`FixedKernelPolicy` — the versioned, explicit GEMM/fused-MoE kernel
   selector. It declares exactly one benchmarked backend per
   ``(op_kind, recipe_id)`` and never falls back to another backend.
2. :func:`canonicalize_benchmark_csv` — convert the sectioned
   ``combined_results.csv`` produced by the kernel benchmark into
   ``haq_latency_v1`` rows, applying the fixed-kernel policy and preserving
   proxy provenance.
3. :class:`LatencyLUT` — load a canonical ``haq_latency_v1`` CSV, expose a
   deterministic digest, and perform exact ``(deployment_profile, m,
   group_pattern, recipe_id)`` look-ups with strict source-pattern auditing.

No interpolation, nearest-shape fallback, or minimum-across-backends logic is
implemented anywhere; missing coverage is always a fatal, aggregated error.
"""

import csv
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Final

import yaml

__all__ = [
    "FixedKernelPolicy",
    "KernelSelector",
    "LatencyCoverageError",
    "LatencyLUT",
    "LatencyRow",
    "canonicalize_benchmark_csv",
    "load_fixed_kernel_policy",
    "write_canonical_csv",
]

SCHEMA_VERSION: Final = "haq_latency_v1"
SELECTION_POLICY_FIXED_KERNEL: Final = "fixed_kernel"

OP_KIND_GEMM: Final = "gemm"
OP_KIND_MOE: Final = "moe"

# Stable ModelOpt candidate identities. These mirror the quantization config
# names that AutoQuantize search recipes resolve to.
RECIPE_NONE: Final = "NONE"
RECIPE_FP8: Final = "FP8_DEFAULT_CFG"
RECIPE_NVFP4: Final = "NVFP4_DEFAULT_CFG"
RECIPE_W4A16_NVFP4: Final = "W4A16_NVFP4_CFG"

# A candidate is measured with input-activation quantization inside the timed
# region for every quantized recipe; the BF16/no-quant candidate is not.
_RECIPE_WITH_QUANT: Final = {
    RECIPE_NONE: False,
    RECIPE_FP8: True,
    RECIPE_NVFP4: True,
    RECIPE_W4A16_NVFP4: True,
}

# Human-readable runtime format label emitted for each recipe. This is the
# *target* deployed format; the measured format may differ under an explicit
# proxy (see ``measured_runtime_format``).
_RECIPE_RUNTIME_FORMAT: Final = {
    RECIPE_NONE: "BF16",
    RECIPE_FP8: "FP8",
    RECIPE_NVFP4: "NVFP4_W4A4",
    RECIPE_W4A16_NVFP4: "NVFP4_W4A16",
}

# The measured format an explicit W4A4->W4A16 proxy records.
_W4A4_MEASURED_RUNTIME_FORMAT: Final = "NVFP4_W4A4"

REQUIRED_COLUMNS: Final = (
    "schema_version",
    "deployment_profile",
    "group_pattern",
    "source_module_patterns",
    "recipe_id",
    "runtime_format",
    "m",
    "latency_us",
    "backend",
    "with_quant",
    "op_kind",
    "timing_scope",
    "selection_policy",
    "kernel_policy_id",
    "tp",
    "ep",
    "hardware",
)

OPTIONAL_COLUMNS: Final = (
    "n",
    "k",
    "h",
    "f",
    "local_experts",
    "top_k",
    "benchmark_provenance",
    "measured_runtime_format",
    "cost_is_proxy",
    "proxy_reason",
)

ALL_COLUMNS: Final = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

_DECODER_LAYER_INDEX_RE = re.compile(r"(?<=\.)\d+(?=\.)")


class LatencyCoverageError(ValueError):
    """Raised when the fixed-kernel policy cannot be satisfied.

    The error aggregates every missing, failed, ambiguous, or malformed row into
    a single actionable message so coverage gaps surface before any expensive
    calibration or scoring.
    """

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(
            f"Fixed-kernel latency coverage failed ({len(self.problems)} problem(s)):\n  - {joined}"
        )


def normalize_layer_indices(module_name: str) -> str:
    """Replace concrete decoder-layer indices with ``*`` wildcards.

    ``model.layers.5.self_attn.q_proj`` -> ``model.layers.*.self_attn.q_proj``.
    Names already wildcarded by the benchmark are returned unchanged.
    """
    return _DECODER_LAYER_INDEX_RE.sub("*", module_name)


# ---------------------------------------------------------------------------
# Fixed-kernel policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelSelector:
    """One fixed backend selection for an ``(op_kind, recipe_id)`` pair.

    ``enable_w4a4_proxy`` authorizes pricing a W4A16 NVFP4 candidate with a
    measured W4A4 NVFP4 row for the documented batch-size-1, low-``M`` POC. When
    set, ``proxy_reason`` is mandatory and the emitted row records
    ``cost_is_proxy=True`` plus ``measured_runtime_format=NVFP4_W4A4``.
    """

    backend: str
    kernel_source: str | None = None
    enable_w4a4_proxy: bool = False
    proxy_reason: str | None = None

    def __post_init__(self):
        if not self.backend or not isinstance(self.backend, str):
            raise ValueError("KernelSelector.backend must be a non-empty string.")
        if self.enable_w4a4_proxy and not self.proxy_reason:
            raise ValueError("KernelSelector with enable_w4a4_proxy=True requires a proxy_reason.")


@dataclass(frozen=True)
class FixedKernelPolicy:
    """Versioned fixed-kernel selector configuration.

    ``selectors`` maps ``op_kind`` -> ``recipe_id`` -> :class:`KernelSelector`.
    A selector is fixed across all groups, shapes, layers, and ``M`` values for
    its ``(op_kind, recipe_id)`` scope.
    """

    kernel_policy_id: str
    selectors: dict[str, dict[str, KernelSelector]]
    mode: str = SELECTION_POLICY_FIXED_KERNEL

    def __post_init__(self):
        if not self.kernel_policy_id:
            raise ValueError("FixedKernelPolicy.kernel_policy_id must be non-empty.")
        if self.mode != SELECTION_POLICY_FIXED_KERNEL:
            raise ValueError(
                f"Unsupported kernel policy mode {self.mode!r}; only "
                f"{SELECTION_POLICY_FIXED_KERNEL!r} is implemented."
            )
        for op_kind, recipe_selectors in self.selectors.items():
            if op_kind not in (OP_KIND_GEMM, OP_KIND_MOE):
                raise ValueError(f"Unknown op_kind {op_kind!r} in fixed-kernel policy.")
            for recipe_id in recipe_selectors:
                if recipe_id not in _RECIPE_WITH_QUANT:
                    raise ValueError(
                        f"Unknown recipe_id {recipe_id!r} in fixed-kernel policy for "
                        f"op_kind {op_kind!r}."
                    )

    def selector(self, op_kind: str, recipe_id: str) -> KernelSelector | None:
        return self.selectors.get(op_kind, {}).get(recipe_id)


def _parse_selector(raw: dict[str, Any]) -> KernelSelector:
    known = {"backend", "kernel_source", "enable_w4a4_proxy", "proxy_reason"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown kernel selector fields: {sorted(unknown)}.")
    return KernelSelector(
        backend=raw["backend"],
        kernel_source=raw.get("kernel_source"),
        enable_w4a4_proxy=bool(raw.get("enable_w4a4_proxy", False)),
        proxy_reason=raw.get("proxy_reason"),
    )


def load_fixed_kernel_policy(source: str | Path | dict[str, Any]) -> FixedKernelPolicy:
    """Load a fixed-kernel policy from a YAML/JSON file path or an in-memory dict."""
    if isinstance(source, dict):
        data = source
    else:
        text = Path(source).read_text()
        # yaml.safe_load parses JSON as well, so a single loader covers both.
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Fixed-kernel policy must be a mapping.")

    selectors: dict[str, dict[str, KernelSelector]] = {}
    for op_kind, recipe_selectors in (data.get("selectors") or {}).items():
        if not isinstance(recipe_selectors, dict):
            raise ValueError(f"selectors[{op_kind!r}] must be a mapping.")
        selectors[op_kind] = {
            recipe_id: _parse_selector(raw) for recipe_id, raw in recipe_selectors.items()
        }
    return FixedKernelPolicy(
        kernel_policy_id=data.get("kernel_policy_id", ""),
        mode=data.get("mode", SELECTION_POLICY_FIXED_KERNEL),
        selectors=selectors,
    )


# ---------------------------------------------------------------------------
# Raw benchmark CSV parsing
# ---------------------------------------------------------------------------


@dataclass
class _RawRow:
    op_kind: str
    module_name: str
    m: int
    n: int | None
    k: int | None
    backend: str
    with_quant: bool
    latency_us: float | None  # None for an error/failed measurement
    error: str | None


@dataclass
class _RawBenchmark:
    provenance: str
    rows: list[_RawRow]
    moe_shape: dict[str, Any]  # h, f, local_experts, top_k, activation when present


_SECTION_TO_OP_KIND: Final = {"GEMM": OP_KIND_GEMM, "MOE": OP_KIND_MOE, "MoE": OP_KIND_MOE}
_RAW_HEADER = "module_name,M,N,K,backend,with_quant,runtime"


def _parse_int_or_none(value: str) -> int | None:
    value = value.strip()
    return int(value) if value else None


def _parse_moe_shape_line(line: str) -> dict[str, Any]:
    """Parse ``H=2048 F=512 E=256 top_k=8 activation=Swiglu`` metadata lines."""
    shape: dict[str, Any] = {}
    mapping = {"H": "h", "F": "f", "E": "local_experts", "top_k": "top_k"}
    for token in line.split():
        if "=" not in token:
            continue
        key, _, val = token.partition("=")
        canonical = mapping.get(key)
        if canonical is None:
            if key == "activation":
                shape["activation"] = val
            continue
        try:
            shape[canonical] = int(val)
        except ValueError:
            shape[canonical] = val
    return shape


def parse_benchmark_csv(source: str | Path) -> _RawBenchmark:
    """Parse the sectioned ``combined_results.csv`` into typed raw rows."""
    lines = Path(source).read_text().splitlines()
    if not lines:
        raise ValueError("Empty benchmark CSV.")

    provenance = lines[0].strip()
    rows: list[_RawRow] = []
    moe_shape: dict[str, Any] = {}
    current_op_kind: str | None = None

    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in _SECTION_TO_OP_KIND:
            current_op_kind = _SECTION_TO_OP_KIND[stripped]
            continue
        if stripped == _RAW_HEADER:
            continue
        if current_op_kind == OP_KIND_MOE and stripped.startswith("H="):
            moe_shape = _parse_moe_shape_line(stripped)
            continue
        if current_op_kind is None:
            raise ValueError(f"Data row before any section header: {stripped!r}.")

        # runtime may itself contain commas (ERROR messages with paths), so split
        # off the leading six fixed fields and keep the remainder as runtime.
        parts = stripped.split(",", 6)
        if len(parts) != 7:
            raise ValueError(f"Malformed benchmark row: {stripped!r}.")
        module_name, m_str, n_str, k_str, backend, with_quant_str, runtime = parts
        runtime = runtime.strip()
        error = runtime if runtime.startswith("ERROR") else None
        latency_us: float | None = None
        if error is None:
            try:
                latency_us = float(runtime)
            except ValueError as exc:
                raise ValueError(f"Non-numeric runtime {runtime!r} for row {stripped!r}.") from exc
        rows.append(
            _RawRow(
                op_kind=current_op_kind,
                module_name=module_name,
                m=int(m_str),
                n=_parse_int_or_none(n_str),
                k=_parse_int_or_none(k_str),
                backend=backend,
                with_quant=with_quant_str.strip() == "True",
                latency_us=latency_us,
                error=error,
            )
        )
    return _RawBenchmark(provenance=provenance, rows=rows, moe_shape=moe_shape)


# ---------------------------------------------------------------------------
# Canonical row + canonicalizer
# ---------------------------------------------------------------------------


@dataclass
class LatencyRow:
    """One canonical ``haq_latency_v1`` cost row."""

    schema_version: str
    deployment_profile: str
    group_pattern: str
    source_module_patterns: list[str]
    recipe_id: str
    runtime_format: str
    m: int
    latency_us: float
    backend: str
    with_quant: bool
    op_kind: str
    timing_scope: str
    selection_policy: str
    kernel_policy_id: str
    tp: int
    ep: int
    hardware: str
    n: int | None = None
    k: int | None = None
    h: int | None = None
    f: int | None = None
    local_experts: int | None = None
    top_k: int | None = None
    benchmark_provenance: str | None = None
    measured_runtime_format: str | None = None
    cost_is_proxy: bool = False
    proxy_reason: str | None = None

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (self.deployment_profile, self.m, self.group_pattern, self.recipe_id)


def _recipe_ids_for_op(policy: FixedKernelPolicy, op_kind: str) -> list[str]:
    return list(policy.selectors.get(op_kind, {}))


def canonicalize_benchmark_csv(
    source: str | Path,
    policy: FixedKernelPolicy,
    *,
    deployment_profile: str,
    tp: int,
    ep: int,
    hardware: str,
    gemm_timing_scope: str = "gemm_fused",
    moe_timing_scope: str = "moe_grouped_gemm",
) -> tuple[list[LatencyRow], list[str]]:
    """Convert a raw benchmark CSV into canonical rows under a fixed-kernel policy.

    Returns ``(rows, coverage_problems)``. ``rows`` contains exactly one row per
    ``(module group, M, declared recipe)`` for which the policy's selector found
    a single successful measurement. Every missing, failed, ambiguous, or invalid
    selection is appended to ``coverage_problems`` rather than silently dropped
    or substituted. Callers that require complete coverage should raise
    :class:`LatencyCoverageError` when ``coverage_problems`` is non-empty.
    """
    raw = parse_benchmark_csv(source)
    timing_scope = {OP_KIND_GEMM: gemm_timing_scope, OP_KIND_MOE: moe_timing_scope}

    # Index raw rows by (op_kind, module_name, m) for selection.
    by_group: dict[tuple[str, str, int], list[_RawRow]] = {}
    module_ms: dict[str, set[tuple[str, int]]] = {OP_KIND_GEMM: set(), OP_KIND_MOE: set()}
    for row in raw.rows:
        by_group.setdefault((row.op_kind, row.module_name, row.m), []).append(row)
        module_ms[row.op_kind].add((row.module_name, row.m))

    rows: list[LatencyRow] = []
    problems: list[str] = []

    for op_kind in (OP_KIND_GEMM, OP_KIND_MOE):
        recipe_ids = _recipe_ids_for_op(policy, op_kind)
        if not recipe_ids:
            continue
        for module_name, m in sorted(module_ms[op_kind]):
            group_pattern = normalize_layer_indices(module_name)
            source_patterns = module_name.split("|")
            candidates = by_group.get((op_kind, module_name, m), [])
            for recipe_id in recipe_ids:
                selector = policy.selector(op_kind, recipe_id)
                assert selector is not None
                expected_with_quant = _RECIPE_WITH_QUANT[recipe_id]
                # The current raw benchmark CSV carries no kernel_source column, so
                # selector.kernel_source is reserved for future raw formats (e.g. AIC)
                # and does not further filter here.
                selected = [
                    r
                    for r in candidates
                    if r.backend == selector.backend and r.with_quant == expected_with_quant
                ]
                successful = [r for r in selected if r.error is None]
                failed = [r for r in selected if r.error is not None]

                context = (
                    f"op_kind={op_kind} recipe_id={recipe_id} backend={selector.backend} "
                    f"group={group_pattern} M={m}"
                )
                if not selected:
                    problems.append(f"No benchmarked row for {context} (backend not measured).")
                    continue
                if not successful:
                    problems.append(f"All rows failed for {context}: {failed[0].error}")
                    continue
                if len(successful) > 1:
                    problems.append(
                        f"Ambiguous fixed-kernel selection for {context}: "
                        f"{len(successful)} successful rows."
                    )
                    continue

                chosen = successful[0]
                assert chosen.latency_us is not None  # guaranteed by the error is None filter
                is_proxy = selector.enable_w4a4_proxy and recipe_id == RECIPE_W4A16_NVFP4
                if selector.enable_w4a4_proxy and recipe_id != RECIPE_W4A16_NVFP4:
                    problems.append(
                        f"enable_w4a4_proxy is only valid for {RECIPE_W4A16_NVFP4}; got {context}."
                    )
                    continue
                rows.append(
                    LatencyRow(
                        schema_version=SCHEMA_VERSION,
                        deployment_profile=deployment_profile,
                        group_pattern=group_pattern,
                        source_module_patterns=source_patterns,
                        recipe_id=recipe_id,
                        runtime_format=_RECIPE_RUNTIME_FORMAT[recipe_id],
                        m=m,
                        latency_us=chosen.latency_us,
                        backend=chosen.backend,
                        with_quant=chosen.with_quant,
                        op_kind=op_kind,
                        timing_scope=timing_scope[op_kind],
                        selection_policy=SELECTION_POLICY_FIXED_KERNEL,
                        kernel_policy_id=policy.kernel_policy_id,
                        tp=tp,
                        ep=ep,
                        hardware=hardware,
                        n=chosen.n,
                        k=chosen.k,
                        h=raw.moe_shape.get("h") if op_kind == OP_KIND_MOE else None,
                        f=raw.moe_shape.get("f") if op_kind == OP_KIND_MOE else None,
                        local_experts=(
                            raw.moe_shape.get("local_experts") if op_kind == OP_KIND_MOE else None
                        ),
                        top_k=raw.moe_shape.get("top_k") if op_kind == OP_KIND_MOE else None,
                        benchmark_provenance=raw.provenance,
                        measured_runtime_format=(
                            _W4A4_MEASURED_RUNTIME_FORMAT if is_proxy else None
                        ),
                        cost_is_proxy=is_proxy,
                        proxy_reason=selector.proxy_reason if is_proxy else None,
                    )
                )

    return rows, problems


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, list):
        return json.dumps(value)
    return str(value)


def write_canonical_csv(rows: Sequence[LatencyRow], path: str | Path) -> None:
    """Serialize canonical rows to a ``haq_latency_v1`` CSV file."""
    path = Path(path)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ALL_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    _cell(
                        getattr(row, col)
                        if col != "source_module_patterns"
                        else row.source_module_patterns
                    )
                    for col in ALL_COLUMNS
                ]
            )


# ---------------------------------------------------------------------------
# Canonical LUT consumer
# ---------------------------------------------------------------------------


def _parse_bool_cell(value: str) -> bool:
    return value.strip() == "True"


def _parse_int_cell(value: str) -> int | None:
    value = value.strip()
    return int(value) if value else None


class LatencyLUT:
    """Loaded, validated ``haq_latency_v1`` cost table with exact look-up."""

    def __init__(self, rows: Sequence[LatencyRow], digest: str):
        self.digest = digest
        self._rows: list[LatencyRow] = list(rows)
        self._by_key: dict[tuple[str, int, str, str], LatencyRow] = {}
        for row in self._rows:
            if row.key in self._by_key:
                raise LatencyCoverageError([f"Duplicate canonical row for key {row.key}."])
            self._by_key[row.key] = row

    @classmethod
    def from_csv(cls, path: str | Path) -> "LatencyLUT":
        """Load and validate a canonical CSV, computing a deterministic digest."""
        path = Path(path)
        raw_bytes = path.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()

        rows: list[LatencyRow] = []
        problems: list[str] = []
        reader = csv.DictReader(raw_bytes.decode().splitlines())
        missing_cols = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing_cols:
            raise LatencyCoverageError(
                [f"Canonical CSV missing required columns: {sorted(missing_cols)}."]
            )
        for i, record in enumerate(reader, start=2):
            if record.get("schema_version") != SCHEMA_VERSION:
                problems.append(
                    f"Row {i}: schema_version={record.get('schema_version')!r}, "
                    f"expected {SCHEMA_VERSION!r}."
                )
                continue
            try:
                latency_us = float(record["latency_us"])
            except (TypeError, ValueError):
                problems.append(f"Row {i}: non-numeric latency_us={record.get('latency_us')!r}.")
                continue
            if not (latency_us > 0.0 and latency_us < float("inf")):
                problems.append(
                    f"Row {i}: latency_us must be finite and positive, got {latency_us}."
                )
                continue
            try:
                source_patterns = json.loads(record["source_module_patterns"])
            except (TypeError, ValueError, json.JSONDecodeError):
                problems.append(
                    f"Row {i}: source_module_patterns is not valid JSON: "
                    f"{record.get('source_module_patterns')!r}."
                )
                continue
            if not isinstance(source_patterns, list) or not all(
                isinstance(p, str) for p in source_patterns
            ):
                problems.append(f"Row {i}: source_module_patterns must be a JSON list of strings.")
                continue

            cost_is_proxy = _parse_bool_cell(record.get("cost_is_proxy", ""))
            proxy_reason = record.get("proxy_reason") or None
            measured_runtime_format = record.get("measured_runtime_format") or None
            if cost_is_proxy and not (proxy_reason and measured_runtime_format):
                problems.append(
                    f"Row {i}: cost_is_proxy=True requires both measured_runtime_format "
                    "and proxy_reason."
                )
                continue

            rows.append(
                LatencyRow(
                    schema_version=SCHEMA_VERSION,
                    deployment_profile=record["deployment_profile"],
                    group_pattern=record["group_pattern"],
                    source_module_patterns=source_patterns,
                    recipe_id=record["recipe_id"],
                    runtime_format=record["runtime_format"],
                    m=int(record["m"]),
                    latency_us=latency_us,
                    backend=record["backend"],
                    with_quant=_parse_bool_cell(record["with_quant"]),
                    op_kind=record["op_kind"],
                    timing_scope=record["timing_scope"],
                    selection_policy=record["selection_policy"],
                    kernel_policy_id=record["kernel_policy_id"],
                    tp=int(record["tp"]),
                    ep=int(record["ep"]),
                    hardware=record["hardware"],
                    n=_parse_int_cell(record.get("n", "")),
                    k=_parse_int_cell(record.get("k", "")),
                    h=_parse_int_cell(record.get("h", "")),
                    f=_parse_int_cell(record.get("f", "")),
                    local_experts=_parse_int_cell(record.get("local_experts", "")),
                    top_k=_parse_int_cell(record.get("top_k", "")),
                    benchmark_provenance=record.get("benchmark_provenance") or None,
                    measured_runtime_format=measured_runtime_format,
                    cost_is_proxy=cost_is_proxy,
                    proxy_reason=proxy_reason,
                )
            )

        if problems:
            raise LatencyCoverageError(problems)
        return cls(rows, digest)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def deployment_profiles(self) -> set[str]:
        return {row.deployment_profile for row in self._rows}

    def group_patterns(self, deployment_profile: str, m: int) -> set[str]:
        return {
            row.group_pattern
            for row in self._rows
            if row.deployment_profile == deployment_profile and row.m == m
        }

    def lookup(
        self, deployment_profile: str, m: int, group_pattern: str, recipe_id: str
    ) -> LatencyRow:
        """Return the exact row for a key or raise :class:`LatencyCoverageError`."""
        key = (deployment_profile, m, group_pattern, recipe_id)
        row = self._by_key.get(key)
        if row is None:
            raise LatencyCoverageError([f"No latency row for key {key}."])
        return row

    def match_group_pattern(
        self, deployment_profile: str, m: int, source_module_names: Sequence[str]
    ) -> str:
        """Resolve concrete source module names to exactly one ``group_pattern``.

        A candidate group's concrete (per-layer) source modules must be fully and
        exclusively covered by one row-group's ``source_module_patterns``:

        - every concrete source module matches at least one supplied pattern, and
        - every supplied pattern matches at least one concrete source module.

        Zero or multiple matching group patterns raise a coverage error.
        """
        source_module_names = list(source_module_names)
        seen: dict[str, list[str]] = {}
        for row in self._rows:
            if row.deployment_profile != deployment_profile or row.m != m:
                continue
            seen.setdefault(row.group_pattern, row.source_module_patterns)

        matches = [
            group_pattern
            for group_pattern, patterns in seen.items()
            if _sources_fully_covered(source_module_names, patterns)
        ]
        if not matches:
            raise LatencyCoverageError(
                [
                    f"No group_pattern at profile={deployment_profile} M={m} covers source "
                    f"modules {source_module_names}."
                ]
            )
        if len(matches) > 1:
            raise LatencyCoverageError(
                [
                    f"Source modules {source_module_names} match multiple group_patterns "
                    f"{sorted(matches)} at profile={deployment_profile} M={m}."
                ]
            )
        return matches[0]


def _sources_fully_covered(source_module_names: Sequence[str], patterns: Sequence[str]) -> bool:
    if not source_module_names or not patterns:
        return False
    every_source_matches = all(
        any(fnmatch(name, pattern) for pattern in patterns) for name in source_module_names
    )
    every_pattern_matches = all(
        any(fnmatch(name, pattern) for name in source_module_names) for pattern in patterns
    )
    return every_source_matches and every_pattern_matches
