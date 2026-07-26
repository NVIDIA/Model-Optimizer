"""Deterministic sparse views over canonical Puzzletron candidate libraries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from ..candidates import _AXIS_TO_TARGET, Candidate
from ..identity import canonicalize, stable_hash


@dataclass(frozen=True, kw_only=True)
class SparseSamplingPolicy:
    max_pairwise_per_family: int = 4
    replacement_cap: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_pairwise_per_family < 0:
            raise ValueError("max_pairwise_per_family must be non-negative")
        if self.replacement_cap < 1:
            raise ValueError("replacement_cap must be positive")


@dataclass(frozen=True, kw_only=True)
class SparseSampleRecord:
    sample_id: str
    candidate_id: str
    layer_idx: int
    hidden_width: int | None
    subblock_kind: str
    subblock_name: str
    changed_axes: tuple[str, ...]
    block_config: dict[str, Any]
    subblock_config: dict[str, Any] | None
    no_op: bool = False
    reason: str = "eligible"

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(asdict(self))


@dataclass(frozen=True, kw_only=True)
class SparseSampleManifest:
    mode: str
    policy: SparseSamplingPolicy
    eligible: tuple[SparseSampleRecord, ...]
    selected: tuple[SparseSampleRecord, ...]
    excluded: tuple[SparseSampleRecord, ...]

    @property
    def identity(self) -> str:
        return stable_hash(self.to_dict(include_identity=False), prefix="sparse_samples")

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "version": 1,
            "mode": self.mode,
            "policy": canonicalize(asdict(self.policy)),
            "eligible": [row.to_dict() for row in self.eligible],
            "selected": [row.to_dict() for row in self.selected],
            "excluded": [row.to_dict() for row in self.excluded],
        }
        if include_identity:
            payload["identity"] = self.identity
        return payload


def _changed_axes(candidate: Candidate) -> tuple[str, ...]:
    return tuple(sorted(str(axis) for axis in (candidate.metadata.get("slice_axes") or {})))


def _axis_kind(axis_id: str) -> str | None:
    target = _AXIS_TO_TARGET.get(axis_id)
    return None if target is None else str(target[0])


def _json_key(value: Any) -> str:
    return json.dumps(canonicalize(value), sort_keys=True, separators=(",", ":"))


def _record(
    candidate: Candidate,
    *,
    changed_axes: tuple[str, ...],
    subblock_kind: str,
    subblock_name: str,
    subblock_config: dict[str, Any] | None,
    reason: str = "eligible",
) -> SparseSampleRecord:
    data = {
        "candidate_id": candidate.identity.value,
        "layer_idx": candidate.layer_idx,
        "hidden_width": candidate.hidden_width,
        "subblock_kind": subblock_kind,
        "subblock_name": subblock_name,
        "changed_axes": changed_axes,
        "block_config": candidate.block_config.to_dict(),
        "subblock_config": subblock_config,
    }
    return SparseSampleRecord(
        sample_id=stable_hash(data, prefix="sparse_sample"),
        reason=reason,
        no_op=(
            bool(subblock_config.get("no_op", False))
            if subblock_config is not None
            else any(subblock.no_op for subblock in candidate.block_config.subblock_configs)
        ),
        **data,
    )


def _replace_reason(row: SparseSampleRecord, reason: str) -> SparseSampleRecord:
    return SparseSampleRecord(**{**asdict(row), "reason": reason})


def _record_key(row: SparseSampleRecord) -> tuple[Any, ...]:
    return (
        -1 if row.hidden_width is None else int(row.hidden_width),
        row.subblock_kind,
        row.subblock_name,
        row.changed_axes,
        row.layer_idx,
        row.sample_id,
    )


def _candidate_records(candidates: Iterable[Candidate]) -> list[SparseSampleRecord]:
    rows: list[SparseSampleRecord] = []
    for candidate in candidates:
        axes = _changed_axes(candidate)
        kinds = sorted({_axis_kind(axis) or "unknown" for axis in axes})
        rows.append(
            _record(
                candidate,
                changed_axes=axes,
                subblock_kind="+".join(kinds) if kinds else "block",
                subblock_name="block",
                subblock_config=None,
            )
        )
    return sorted(rows, key=_record_key)


def _representative_layers(layers: Iterable[int], *, rotation: int = 0) -> list[int]:
    ordered = sorted(set(int(layer) for layer in layers))
    if not ordered:
        return []
    representatives = []
    for layer in (ordered[0], ordered[len(ordered) // 2], ordered[-1]):
        if layer not in representatives:
            representatives.append(layer)
    if representatives:
        offset = rotation % len(representatives)
        representatives = representatives[offset:] + representatives[:offset]
    return representatives + [layer for layer in ordered if layer not in representatives]


def _dedupe(rows: Iterable[SparseSampleRecord], key) -> tuple[list[SparseSampleRecord], list[SparseSampleRecord]]:
    kept: dict[Any, SparseSampleRecord] = {}
    excluded: list[SparseSampleRecord] = []
    for row in sorted(rows, key=_record_key):
        row_key = key(row)
        if row_key in kept:
            excluded.append(_replace_reason(row, "duplicate_identity"))
        else:
            kept[row_key] = row
    return list(kept.values()), excluded


def sample_subblock_configs(
    candidates: Iterable[Candidate],
    *,
    policy: SparseSamplingPolicy | None = None,
) -> SparseSampleManifest:
    """Select layer-independent teacher, single-axis, and pairwise subblocks."""

    policy = policy or SparseSamplingPolicy()
    candidates = sorted(candidates, key=lambda item: item.identity.value)
    teacher_by_layer_kind: dict[tuple[int | None, int, str, str], str] = {}
    for candidate in candidates:
        if _changed_axes(candidate) or candidate.source_kind != "self":
            continue
        for subblock in candidate.block_config.subblock_configs:
            teacher_by_layer_kind[
                (candidate.hidden_width, candidate.layer_idx, subblock.kind, subblock.name)
            ] = _json_key(subblock.to_dict())

    eligible: list[SparseSampleRecord] = []
    excluded: list[SparseSampleRecord] = []
    family_by_sample: dict[str, tuple[Any, ...]] = {}
    for candidate in candidates:
        axes = _changed_axes(candidate)
        if len(axes) > 2:
            excluded.append(
                _replace_reason(_candidate_records([candidate])[0], "more_than_two_axes")
            )
            continue
        for subblock in candidate.block_config.subblock_configs:
            relevant_axes = tuple(axis for axis in axes if _axis_kind(axis) == subblock.kind)
            if axes and not relevant_axes:
                continue
            if len(relevant_axes) > 2:
                continue
            if subblock.no_op:
                excluded.append(
                    _replace_reason(
                        _record(
                            candidate,
                            changed_axes=relevant_axes,
                            subblock_kind=subblock.kind,
                            subblock_name=subblock.name,
                            subblock_config=subblock.to_dict(),
                        ),
                        "no_op",
                    )
                )
                continue
            teacher_key = teacher_by_layer_kind.get(
                (candidate.hidden_width, candidate.layer_idx, subblock.kind, subblock.name)
            )
            if teacher_key is None:
                excluded.append(
                    _replace_reason(
                        _record(
                            candidate,
                            changed_axes=relevant_axes,
                            subblock_kind=subblock.kind,
                            subblock_name=subblock.name,
                            subblock_config=subblock.to_dict(),
                        ),
                        "missing_teacher_anchor",
                    )
                )
                continue
            row = _record(
                candidate,
                changed_axes=relevant_axes,
                subblock_kind=subblock.kind,
                subblock_name=subblock.name,
                subblock_config=subblock.to_dict(),
            )
            eligible.append(row)
            family_by_sample[row.sample_id] = (
                candidate.hidden_width,
                subblock.kind,
                subblock.name,
                teacher_key,
            )

    eligible, duplicate_rows = _dedupe(
        eligible,
        key=lambda row: (
            row.hidden_width,
            row.subblock_kind,
            row.subblock_name,
            _json_key(row.subblock_config),
        ),
    )
    excluded.extend(duplicate_rows)
    families: dict[tuple[Any, ...], list[SparseSampleRecord]] = {}
    for row in eligible:
        families.setdefault(family_by_sample[row.sample_id], []).append(row)

    selected: list[SparseSampleRecord] = []
    for family in sorted(families, key=str):
        rows = sorted(families[family], key=_record_key)
        anchors = [row for row in rows if not row.changed_axes]
        if anchors:
            selected.append(anchors[0])
        signatures: dict[tuple[str, ...], list[SparseSampleRecord]] = {}
        for row in rows:
            if row.changed_axes:
                signatures.setdefault(row.changed_axes, []).append(row)
        for signature in sorted(sig for sig in signatures if len(sig) == 1):
            selected.append(sorted(signatures[signature], key=_record_key)[0])
        pair_signatures = sorted(sig for sig in signatures if len(sig) == 2)
        for signature in pair_signatures[: policy.max_pairwise_per_family]:
            selected.append(sorted(signatures[signature], key=_record_key)[0])

    selected_ids = {row.sample_id for row in selected}
    for row in eligible:
        if row.sample_id in selected_ids:
            continue
        reason = "pairwise_limit" if len(row.changed_axes) == 2 else "not_selected"
        excluded.append(_replace_reason(row, reason))
    selected = sorted(selected, key=_record_key)
    return SparseSampleManifest(
        mode="subblock_runtime",
        policy=policy,
        eligible=tuple(sorted(eligible, key=_record_key)),
        selected=tuple(selected),
        excluded=tuple(sorted(excluded, key=_record_key)),
    )


def _signature_queues(
    rows: Iterable[SparseSampleRecord],
) -> list[list[SparseSampleRecord]]:
    by_signature: dict[tuple[str, tuple[str, ...]], list[SparseSampleRecord]] = {}
    for row in rows:
        by_signature.setdefault((row.subblock_kind, row.changed_axes), []).append(row)
    queues: list[list[SparseSampleRecord]] = []
    for index, signature in enumerate(sorted(by_signature, key=str)):
        signature_rows = by_signature[signature]
        by_layer = {row.layer_idx: row for row in sorted(signature_rows, key=_record_key)}
        queues.append(
            [
                by_layer[layer]
                for layer in _representative_layers(by_layer, rotation=index)
            ]
        )
    return queues


def _round_robin(queues: list[list[SparseSampleRecord]]) -> list[SparseSampleRecord]:
    output: list[SparseSampleRecord] = []
    cursor = 0
    while any(queues):
        queue = queues[cursor % len(queues)]
        if queue:
            output.append(queue.pop(0))
        cursor += 1
    return output


def sample_replacement_candidates(
    candidates: Iterable[Candidate],
    *,
    policy: SparseSamplingPolicy | None = None,
) -> SparseSampleManifest:
    """Select at most ``replacement_cap`` layer candidates independently per width."""

    policy = policy or SparseSamplingPolicy()
    candidates = sorted(candidates, key=lambda item: item.identity.value)
    raw_rows = _candidate_records(candidates)
    candidate_by_id = {candidate.identity.value: candidate for candidate in candidates}
    teacher_by_layer = {
        (candidate.hidden_width, candidate.layer_idx): candidate.block_config
        for candidate in candidates
        if candidate.source_kind == "self" and not _changed_axes(candidate)
    }
    eligible: list[SparseSampleRecord] = []
    excluded: list[SparseSampleRecord] = []
    for row in raw_rows:
        candidate = candidate_by_id[row.candidate_id]
        teacher = teacher_by_layer.get((row.hidden_width, row.layer_idx))
        teacher_subblocks = {
            (subblock.kind, subblock.name): subblock
            for subblock in (() if teacher is None else teacher.subblock_configs)
        }
        disabled_active_subblock = any(
            subblock.no_op
            and not getattr(
                teacher_subblocks.get((subblock.kind, subblock.name)), "no_op", True
            )
            for subblock in candidate.block_config.subblock_configs
        )
        row = SparseSampleRecord(**{**asdict(row), "no_op": disabled_active_subblock})
        if disabled_active_subblock:
            excluded.append(_replace_reason(row, "no_op"))
        elif not row.changed_axes:
            excluded.append(_replace_reason(row, "teacher_anchor"))
        else:
            eligible.append(row)

    eligible, duplicate_rows = _dedupe(
        eligible,
        key=lambda row: (row.hidden_width, row.layer_idx, _json_key(row.block_config)),
    )
    excluded.extend(duplicate_rows)
    by_width: dict[int | None, list[SparseSampleRecord]] = {}
    for row in eligible:
        by_width.setdefault(row.hidden_width, []).append(row)

    selected: list[SparseSampleRecord] = []
    for width in sorted(by_width, key=lambda value: -1 if value is None else int(value)):
        width_rows = by_width[width]
        singles = [row for row in width_rows if len(row.changed_axes) == 1]
        pairs = [row for row in width_rows if len(row.changed_axes) == 2]
        extremes = [row for row in width_rows if len(row.changed_axes) > 2]
        single_order = _round_robin(_signature_queues(singles))

        pair_by_family: dict[str, list[SparseSampleRecord]] = {}
        for row in pairs:
            pair_by_family.setdefault(row.subblock_kind, []).append(row)
        limited_pairs: list[SparseSampleRecord] = []
        for family in sorted(pair_by_family):
            queues = _signature_queues(pair_by_family[family])
            family_order = _round_robin(queues)
            distinct_signatures: set[tuple[str, ...]] = set()
            for row in family_order:
                if row.changed_axes not in distinct_signatures:
                    if len(distinct_signatures) >= policy.max_pairwise_per_family:
                        continue
                    distinct_signatures.add(row.changed_axes)
                limited_pairs.append(row)

        # Guarantee one representative for every legal single and pair signature,
        # then use remaining budget for additional single-axis layer coverage.
        single_seeds: list[SparseSampleRecord] = []
        seen_single: set[tuple[str, tuple[str, ...]]] = set()
        remaining_singles: list[SparseSampleRecord] = []
        for row in single_order:
            signature = (row.subblock_kind, row.changed_axes)
            if signature not in seen_single:
                seen_single.add(signature)
                single_seeds.append(row)
            else:
                remaining_singles.append(row)
        pair_seeds: list[SparseSampleRecord] = []
        seen_pair: set[tuple[str, tuple[str, ...]]] = set()
        remaining_pairs: list[SparseSampleRecord] = []
        for row in limited_pairs:
            signature = (row.subblock_kind, row.changed_axes)
            if signature not in seen_pair:
                seen_pair.add(signature)
                pair_seeds.append(row)
            else:
                remaining_pairs.append(row)

        extreme_by_layer: dict[int, SparseSampleRecord] = {}
        for row in extremes:
            current = extreme_by_layer.get(row.layer_idx)
            if current is None or (
                len(row.changed_axes), _record_key(row)
            ) > (
                len(current.changed_axes), _record_key(current)
            ):
                extreme_by_layer[row.layer_idx] = row
        extreme_anchors = [
            extreme_by_layer[layer]
            for layer in _representative_layers(extreme_by_layer)
        ]

        chosen = [*single_seeds, *pair_seeds]
        for row in extreme_anchors:
            if len(chosen) >= policy.replacement_cap:
                break
            chosen.append(row)
        for row in [*remaining_singles, *remaining_pairs]:
            if len(chosen) >= policy.replacement_cap:
                break
            chosen.append(row)
        # Output remains single-axis first for transparent diagnostics.
        chosen = sorted(
            chosen[: policy.replacement_cap],
            key=lambda row: (len(row.changed_axes), _record_key(row)),
        )
        selected.extend(chosen)

    selected_ids = {row.sample_id for row in selected}
    excluded.extend(
        _replace_reason(row, "replacement_cap")
        for row in eligible
        if row.sample_id not in selected_ids
    )
    return SparseSampleManifest(
        mode="replacement_scoring",
        policy=policy,
        eligible=tuple(sorted(eligible, key=_record_key)),
        selected=tuple(sorted(selected, key=lambda row: (
            -1 if row.hidden_width is None else int(row.hidden_width),
            len(row.changed_axes),
            _record_key(row),
        ))),
        excluded=tuple(sorted(excluded, key=_record_key)),
    )
