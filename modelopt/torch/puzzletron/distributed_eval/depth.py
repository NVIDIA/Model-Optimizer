# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical requests and orchestration helpers for distributed depth scoring."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..block_config import BlockConfig
from ..depth.iterative import _child_blocks, _metric
from ..depth.schema import DepthScenario, SublayerRemoval
from .campaign import Campaign
from .identity import canonicalize, content_id
from .schema import EvaluationRequest
from .storage import atomic_write_json, read_json

__all__ = [
    "build_depth_request",
    "depth_rpc_context_from_config",
    "run_iterative_depth_rpc",
]


_DEPTH_REVISION = "puzzletron-iterative-depth-v1"


def _canonical_removals(
    removals: Sequence[SublayerRemoval],
) -> tuple[SublayerRemoval, ...]:
    return tuple(sorted(removals, key=lambda item: (item.layer_idx, item.kind)))


def build_depth_request(
    campaign: Campaign,
    *,
    teacher_blocks: Sequence[BlockConfig],
    removals: Sequence[SublayerRemoval],
    hidden_width: int,
) -> EvaluationRequest:
    """Build one cache-stable request for a cumulative depth-removal model."""
    canonical_removals = _canonical_removals(removals)
    layer_replacements = [
        {
            "parent_layer_indices": [layer_idx],
            "child_block_configs": [child.to_dict()],
            "weight_paths": [],
        }
        for layer_idx, child in sorted(
            _child_blocks(list(teacher_blocks), canonical_removals).items()
        )
    ]
    removal_payload = [canonicalize(item) for item in canonical_removals]
    candidate_id = content_id(
        "depth_candidate",
        {
            "hidden_width": int(hidden_width),
            "removals": removal_payload,
            "layer_replacements": layer_replacements,
        },
    )
    return EvaluationRequest(
        campaign_id=campaign.campaign_id,
        handler="depth_candidate",
        payload={
            "candidate_id": candidate_id,
            "hidden_width": int(hidden_width),
            "removals": removal_payload,
            "layer_replacements": layer_replacements,
        },
        model=campaign.manifest.model,
        data=campaign.manifest.data,
        metrics=campaign.manifest.metrics,
        precision=campaign.manifest.precision,
        evaluator_revision=campaign.manifest.evaluator_revision,
    )


def _selected_from_trajectory(path: Path) -> list[SublayerRemoval]:
    if not path.is_file():
        return []
    return [
        SublayerRemoval.model_validate(item)
        for item in read_json(path).get("selected", [])
    ]


def _result_payload(result, **metadata: Any) -> dict[str, Any]:
    return {
        **result.metrics,
        **metadata,
        "distributed_evaluation": {
            "campaign_id": result.campaign_id,
            "request_id": result.request_id,
            "timing": result.timing,
            "provenance": result.provenance,
        },
    }


def _trajectory_payload(
    *,
    selected: Sequence[SublayerRemoval],
    available_count: int,
    max_removals: int,
    source_checkpoint_dir: str | Path,
    parent_checkpoint_identity: str,
    data_identity: str,
    hidden_width: int,
    granularity: str,
) -> dict[str, Any]:
    scenarios = []
    for length in range(len(selected) + 1):
        scenario = DepthScenario(
            parent_checkpoint_identity=parent_checkpoint_identity,
            hidden_width=hidden_width,
            removals=tuple(selected[:length]),
            data_identity=data_identity,
            evaluator_revision=_DEPTH_REVISION,
            granularity=granularity,
        )
        scenarios.append({**scenario.model_dump(mode="python"), "scenario_id": scenario.scenario_id})
    return {
        "version": 1,
        "granularity": granularity,
        "status": "complete" if len(selected) == max_removals else "running",
        "source_checkpoint_dir": str(source_checkpoint_dir),
        "available_count": available_count,
        "max_removals": max_removals,
        "selected": [item.model_dump() for item in selected],
        "scenarios": scenarios,
    }


def depth_rpc_context_from_config(hydra_cfg) -> dict[str, Any]:
    """Resolve the model-independent coordinator inputs from a pipeline config."""
    import json

    from ..anymodel.model_descriptor import ModelDescriptorFactory
    from ..block_config import maybe_cast_block_configs
    from ..depth.iterative import _available_removals, _depth_scoring_config
    from ..granularity import resolve_granularity
    from ..identity import stable_hash
    from ..tools.checkpoint_utils import load_model_config

    cfg = _depth_scoring_config(hydra_cfg)
    depth_cfg = cfg.get("depth", {})
    scoring = cfg.scoring
    output_dir = Path(scoring.output_dir)
    source = Path(scoring.source_checkpoint_dir)
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"iterative-depth source checkpoint is incomplete: {source}")
    descriptor = ModelDescriptorFactory.get(cfg.descriptor)
    model_config = load_model_config(
        source,
        trust_remote_code=descriptor.requires_trust_remote_code(),
    )
    teacher_blocks = list(maybe_cast_block_configs(model_config.block_configs))
    language_config = descriptor.get_language_model_config(model_config)
    granularity = resolve_granularity("depth", depth_cfg)
    available = _available_removals(teacher_blocks, granularity=granularity)
    expected = depth_cfg.get("expected_initial_sublayers", None)
    if expected is not None and len(available) != int(expected):
        raise RuntimeError(
            f"expected {expected} removable sublayers, descriptor exposed {len(available)}"
        )
    data_identity = stable_hash(
        {
            key: scoring.get(key, None)
            for key in (
                "dataset_path",
                "eval_samples",
                "block_size",
                "seed",
                "shuffle_seed",
            )
        },
        prefix="depth_data",
    )
    parent_identity = stable_hash(
        json.loads((source / "config.json").read_text()),
        prefix="depth_parent",
    )
    return {
        "teacher_blocks": teacher_blocks,
        "available": available,
        "hidden_width": int(language_config.hidden_size),
        "output_dir": output_dir,
        "max_removals": int(depth_cfg.get("max_removals", 10)),
        "source_checkpoint_dir": source,
        "parent_checkpoint_identity": parent_identity,
        "data_identity": data_identity,
        "granularity": granularity,
    }


async def run_iterative_depth_rpc(
    campaign: Campaign,
    *,
    client,
    teacher_blocks: Sequence[BlockConfig],
    available: Sequence[SublayerRemoval],
    hidden_width: int,
    output_dir: str | Path,
    max_removals: int,
    source_checkpoint_dir: str | Path,
    parent_checkpoint_identity: str,
    data_identity: str,
    granularity: str = "subblock",
) -> dict[str, Any]:
    """Resume and score an iterative depth trajectory across RPC worker groups."""
    output_dir = Path(output_dir)
    trajectory_path = output_dir / "trajectory.json"
    available = tuple(available)
    max_removals = min(int(max_removals), len(available))
    selected = _selected_from_trajectory(trajectory_path)
    if len(selected) > max_removals:
        raise RuntimeError(f"trajectory already has too many removals: {len(selected)}")

    available_keys = {(item.layer_idx, item.kind) for item in available}
    selected_keys = [(item.layer_idx, item.kind) for item in selected]
    if len(selected_keys) != len(set(selected_keys)) or not set(selected_keys) <= available_keys:
        raise RuntimeError("trajectory contains duplicate or unavailable depth removals")

    for iteration in range(len(selected), max_removals):
        prefix = tuple(selected)
        iteration_dir = output_dir / f"iteration_{iteration:02d}"
        baseline_path = iteration_dir / "baseline.json"
        remaining = [
            item
            for item in available
            if (item.layer_idx, item.kind) not in set(selected_keys)
        ]
        if len(remaining) != len(available) - iteration:
            raise RuntimeError(
                f"depth iteration {iteration} has {len(remaining)} candidates; "
                f"expected {len(available) - iteration}"
            )

        pending: list[tuple[EvaluationRequest, Path, dict[str, Any]]] = []
        if not baseline_path.is_file():
            pending.append(
                (
                    build_depth_request(
                        campaign,
                        teacher_blocks=teacher_blocks,
                        removals=prefix,
                        hidden_width=hidden_width,
                    ),
                    baseline_path,
                    {"removals": [item.model_dump() for item in prefix]},
                )
            )
        for candidate in remaining:
            name = f"candidate_layer_{candidate.layer_idx:03d}_{candidate.kind}"
            result_path = iteration_dir / f"{name}.json"
            if result_path.is_file():
                continue
            pending.append(
                (
                    build_depth_request(
                        campaign,
                        teacher_blocks=teacher_blocks,
                        removals=(*prefix, candidate),
                        hidden_width=hidden_width,
                    ),
                    result_path,
                    {
                        "iteration": iteration,
                        "candidate": candidate.model_dump(),
                        "prefix": [item.model_dump() for item in prefix],
                    },
                )
            )

        if pending:
            pending_by_id = {
                request.request_id: (path, metadata)
                for request, path, metadata in pending
            }
            if len(pending_by_id) != len(pending):
                raise RuntimeError("depth iteration produced duplicate request identities")
            handles = await client.submit_many(request for request, _, _ in pending)
            async for result in client.as_completed(handles):
                try:
                    path, metadata = pending_by_id[result.request_id]
                except KeyError as error:
                    raise RuntimeError(
                        f"RPC returned unknown depth request {result.request_id}"
                    ) from error
                atomic_write_json(path, _result_payload(result, **metadata))

        baseline = _metric(baseline_path)
        ranked = []
        for candidate in remaining:
            name = f"candidate_layer_{candidate.layer_idx:03d}_{candidate.kind}"
            result_path = iteration_dir / f"{name}.json"
            value = _metric(result_path)
            ranked.append(
                {
                    "candidate": candidate.model_dump(),
                    "lm_loss": value,
                    "delta_lm_loss": value - baseline,
                    "result_path": str(result_path),
                }
            )
        ranked.sort(
            key=lambda row: (
                row["delta_lm_loss"],
                row["candidate"]["layer_idx"],
                row["candidate"]["kind"],
            )
        )
        atomic_write_json(iteration_dir / "ranking.json", {"baseline": baseline, "rows": ranked})
        chosen = SublayerRemoval.model_validate(ranked[0]["candidate"])
        selected.append(chosen)
        selected_keys.append((chosen.layer_idx, chosen.kind))
        atomic_write_json(
            trajectory_path,
            _trajectory_payload(
                selected=selected,
                available_count=len(available),
                max_removals=max_removals,
                source_checkpoint_dir=source_checkpoint_dir,
                parent_checkpoint_identity=parent_checkpoint_identity,
                data_identity=data_identity,
                hidden_width=hidden_width,
                granularity=granularity,
            ),
        )

    return {
        "trajectory_path": str(trajectory_path),
        "selected": [item.model_dump() for item in selected],
        "scenario_count": len(selected) + 1,
    }
