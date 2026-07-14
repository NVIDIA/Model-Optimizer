"""Read existing Puzzletron configs without changing their schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ..anymodel.registry import resolve_descriptor_from_pretrained
from ..scoring_parent import ensure_scoring_parent
from .schema import CampaignManifest, ParallelismSpec

DEFAULT_EVALUATOR_REVISION = "puzzletron-distributed-replace-block-v1"


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


def _load_recipe(cfg: dict[str, Any]) -> dict[str, Any]:
    scoring = dict(cfg.get("scoring") or {})
    automodel = dict(scoring.get("automodel") or {})
    recipe = automodel.get("recipe")
    if recipe is not None:
        return dict(recipe)
    recipe_path = automodel.get("recipe_path") or cfg.get("recipe_path")
    if recipe_path is None:
        raise ValueError(
            "Distributed evaluation needs scoring.automodel.recipe_path or an inline recipe"
        )
    loaded = OmegaConf.load(str(recipe_path))
    return OmegaConf.to_container(loaded, resolve=True)


def parallelism_from_config(cfg: dict[str, Any], *, world_size: int) -> ParallelismSpec:
    recipe = _load_recipe(cfg)
    distributed = dict(recipe.get("distributed") or {})
    strategy = dict(recipe.get("distributed_config") or {})
    target = str(strategy.get("_target_", ""))
    dp_size = distributed.get("dp_size")
    if dp_size in (None, "none", "None", 0, "0"):
        dp_size = None
    scoring = dict(cfg.get("scoring") or {})
    distributed_eval = dict(scoring.get("distributed_eval") or {})
    return ParallelismSpec(
        tp_size=int(distributed.get("tp_size", 1) or 1),
        ep_size=int(distributed.get("ep_size", 1) or 1),
        cp_size=int(distributed.get("cp_size", 1) or 1),
        pp_size=int(distributed.get("pp_size", 1) or 1),
        dp_size=int(dp_size) if dp_size is not None else None,
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
    evaluator_revision: str = DEFAULT_EVALUATOR_REVISION,
    overrides: list[str] | None = None,
) -> CampaignManifest:
    cfg = load_plain_pipeline_config(config_path, overrides=overrides)
    scoring = dict(cfg.get("scoring") or {})
    automodel = dict(scoring.get("automodel") or {})
    model_cfg = dict(cfg.get("model") or {})
    recipe = _load_recipe(cfg)
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
        evaluator_revision=evaluator_revision,
        metadata={
            "config_path": str(Path(config_path).resolve()),
            "overrides": list(overrides or []),
        },
    )
