"""Read existing Puzzletron configs without changing their schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..anymodel.registry import resolve_descriptor_from_pretrained
from ..plugins.automodel.config import build_stage_recipe_config
from ..scoring_parent import ensure_scoring_parent
from .schema import CampaignManifest, ParallelismSpec

DEFAULT_EVALUATOR_REVISION = "puzzletron-distributed-replace-block-v1"
DEFAULT_DEPTH_EVALUATOR_REVISION = "puzzletron-distributed-depth-v1"


def _replacement_scoring_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the public scoring section with legacy-config compatibility."""

    if "replacement_scoring" in cfg:
        return dict(cfg.get("replacement_scoring") or {})
    return dict(cfg.get("scoring") or {})


def distributed_stage_config(
    cfg: dict[str, Any], *, stage: str = "replace_block"
) -> dict[str, Any]:
    """Expose one pipeline stage through the evaluator's canonical scoring view."""
    if stage == "replace_block":
        return dict(cfg)
    if stage != "depth":
        raise ValueError(f"unsupported distributed evaluation stage {stage!r}")

    result = dict(cfg)
    depth = dict(cfg.get("depth") or cfg.get("depth_importance") or {})
    scoring = _replacement_scoring_config(cfg)
    automodel = {
        **dict(scoring.get("automodel") or {}),
        **dict(depth.get("automodel") or {}),
    }
    scoring["automodel"] = automodel
    for key in (
        "eval_samples",
        "micro_batch_size",
        "block_size",
        "seed",
        "shuffle_seed",
        "varlen",
        "dataset_path",
        "packed_token_cache_path",
        "realized_dataset_cache_dir",
        "val_dataset_name",
        "data_column",
        "load_dataset_fn",
    ):
        if key in depth:
            scoring[key] = depth[key]
    puzzle_dir = Path(cfg.get("puzzle_dir") or (cfg.get("experiment") or {}).get("dir"))
    source = depth.get("source_checkpoint_dir") or str(
        puzzle_dir / "ckpts" / "elastic_sorted_teacher"
    )
    scoring["teacher_dir"] = source
    scoring["source_checkpoint_dir"] = source
    scoring["target_teacher_dir"] = source
    scoring["output_dir"] = depth.get("output_dir") or str(
        puzzle_dir / "depth" / "iterative"
    )
    result.pop("replacement_scoring", None)
    result["scoring"] = scoring
    return result


def load_plain_pipeline_config(
    config_path: str | Path,
    *,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    from ..pipeline_config import pipeline_config_from_path

    return pipeline_config_from_path(config_path, overrides=overrides or [])


def load_runtime_config(
    config_path: str | Path,
    *,
    overrides: list[str] | None = None,
):
    from ..pipeline_config import load_runtime_hydra_config, pipeline_config_from_path

    normalized = pipeline_config_from_path(config_path, overrides=overrides or [])
    return load_runtime_hydra_config(normalized)


def _stage_recipe(cfg: dict[str, Any]) -> dict[str, Any]:
    scoring = _replacement_scoring_config(cfg)
    automodel = dict(scoring.get("automodel") or {})
    return build_stage_recipe_config(automodel)


def parallelism_from_config(cfg: dict[str, Any], *, world_size: int) -> ParallelismSpec:
    recipe = _stage_recipe(cfg)
    distributed = dict(recipe.get("distributed") or {})
    strategy = dict(recipe.get("distributed_config") or {})
    target = str(strategy.get("_target_", ""))
    tp_size = int(distributed.get("tp_size", 1) or 1)
    ep_size = int(distributed.get("ep_size", 1) or 1)
    cp_size = int(distributed.get("cp_size", 1) or 1)
    pp_size = int(distributed.get("pp_size", 1) or 1)
    configured_dp_size = distributed.get("dp_size")
    if configured_dp_size in (None, "none", "None", 0, "0"):
        automodel_dp_size, remainder = divmod(
            int(world_size), tp_size * cp_size * pp_size
        )
        if remainder:
            raise ValueError(
                "AutoModel parallel sizes do not divide the configured world size: "
                f"world_size={world_size}, tp_size={tp_size}, cp_size={cp_size}, "
                f"pp_size={pp_size}"
            )
    else:
        automodel_dp_size = int(configured_dp_size)
        expected_world_size = automodel_dp_size * tp_size * cp_size * pp_size
        if expected_world_size != int(world_size):
            raise ValueError(
                "AutoModel parallel sizes imply "
                f"world_size={expected_world_size}, configured world_size={world_size}"
            )

    # AutoModel overlays EP on its DP/CP/TP FSDP mesh instead of making EP an
    # additional world-size dimension. ParallelismSpec represents independent
    # axes, so expose the residual sample-parallel degree after that overlay.
    dp_size, remainder = divmod(automodel_dp_size, ep_size)
    if remainder:
        raise ValueError(
            "Distributed evaluation cannot represent this AutoModel EP overlay: "
            f"dp_size={automodel_dp_size} must be divisible by ep_size={ep_size}"
        )
    scoring = _replacement_scoring_config(cfg)
    distributed_eval = dict(scoring.get("distributed_eval") or {})
    return ParallelismSpec(
        tp_size=tp_size,
        ep_size=ep_size,
        cp_size=cp_size,
        pp_size=pp_size,
        dp_size=dp_size,
        sequence_parallel=bool(distributed.get("sequence_parallel", False)),
        fsdp="FSDP" in target.upper(),
        distributed_backend=str((recipe.get("dist_env") or {}).get("backend", "nccl")),
        world_size=int(world_size),
        gpus_per_task=int(distributed_eval.get("gpus_per_task", world_size)),
    )


def _small_file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    manifests = {}
    for name in (
        "config.json",
        "block_configs.json",
        "model.safetensors.index.json",
        "modelopt_state.json",
    ):
        digest = _small_file_digest(checkpoint_dir / name)
        if digest is not None:
            manifests[name] = digest
    weight_files = []
    for path in sorted(checkpoint_dir.glob("*.safetensors")):
        stat = path.stat()
        weight_files.append(
            {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )
    payload = {"manifests": manifests, "weights": weight_files}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "fingerprint": fingerprint,
        **payload,
    }


def build_campaign_manifest(
    config_path: str | Path,
    *,
    world_size: int,
    name: str | None = None,
    evaluator_revision: str | None = None,
    overrides: list[str] | None = None,
    stage: str = "replace_block",
) -> CampaignManifest:
    cfg = distributed_stage_config(
        load_plain_pipeline_config(config_path, overrides=overrides),
        stage=stage,
    )
    scoring = _replacement_scoring_config(cfg)
    automodel = dict(scoring.get("automodel") or {})
    model_cfg = dict(cfg.get("model") or {})
    recipe = _stage_recipe(cfg)
    puzzle_dir = Path(cfg.get("puzzle_dir") or (cfg.get("experiment") or {}).get("dir"))
    source_dir = Path(
        scoring.get("source_checkpoint_dir")
        or ensure_scoring_parent(cfg).path
    )
    force_hf = bool(automodel.get("force_hf", model_cfg.get("force_hf", True)))
    descriptor = scoring.get("descriptor") or cfg.get("descriptor") or model_cfg.get(
        "descriptor_override"
    )
    if not descriptor:
        descriptor = resolve_descriptor_from_pretrained(
            str(source_dir),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        ).name
    descriptor = str(descriptor)
    data_keys = (
        "dataset_path",
        "val_dataset_name",
        "data_column",
        "eval_samples",
        "micro_batch_size",
        "block_size",
        "seed",
        "shuffle_seed",
        "bos_rate",
        "source_datasets_to_discard",
        "packed_token_cache_path",
    )
    metric_keys = (
        "calculate_full_score_ablations",
        "automodel",
    )
    data = {
        "canonical": dict(cfg.get("data") or {}),
        "scoring": {key: scoring.get(key) for key in data_keys if key in scoring},
    }
    canonical_path = (cfg.get("data") or {}).get("path")
    data_manifest = None
    if canonical_path:
        candidate = Path(str(canonical_path))
        manifest_path = candidate / "manifest.json" if candidate.is_dir() else candidate
        digest = _small_file_digest(manifest_path)
        if digest is not None:
            data_manifest = {
                "path": str(manifest_path.resolve()),
                "sha256": digest,
            }
    data["materialized_manifest"] = data_manifest
    data["batch_cache_identity"] = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    metrics = {key: scoring.get(key) for key in metric_keys if key in scoring}
    precision = {
        "torch_dtype": (recipe.get("model") or {}).get("torch_dtype"),
        "autocast_dtype": scoring.get("autocast_dtype"),
        "quantization": scoring.get("quantization"),
    }
    model_identity = checkpoint_identity(source_dir)
    bypass_checkpoint_dir = scoring.get("bypass_checkpoint_dir")
    if bypass_checkpoint_dir:
        model_identity["bypass_overlay"] = checkpoint_identity(bypass_checkpoint_dir)
    return CampaignManifest(
        name=name or Path(config_path).stem,
        model=model_identity,
        descriptor=descriptor,
        force_hf=force_hf,
        parallelism=parallelism_from_config(cfg, world_size=world_size),
        precision=precision,
        automodel_recipe=recipe,
        data=data,
        metrics=metrics,
        evaluator_revision=evaluator_revision
        or (
            DEFAULT_DEPTH_EVALUATOR_REVISION
            if stage == "depth"
            else DEFAULT_EVALUATOR_REVISION
        ),
        metadata={
            "config_path": str(Path(config_path).resolve()),
            "overrides": list(overrides or []),
            "evaluation_stage": stage,
        },
    )
