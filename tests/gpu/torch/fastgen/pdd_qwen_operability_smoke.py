# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staged canonical Qwen-Image PDD operability smoke and result validator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

_THIS_FILE = pathlib.Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
for _path in (_REPO_ROOT, _FASTGEN_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_MODEL_ID = "Qwen/Qwen-Image"
_MODEL_REVISION = "75e0b4be04f60ec59a75f475837eced720f823b6"
_AUTOMODEL_TREE_SHA256 = "b43cb34e04992c66d1888abc0529b760b5b69fc121ff4268b42ecb4a89b1e528"
_AUTOMODEL_WHEEL_SHA256 = "881aebafc5145752842afbbfe0a42e1c33d06847c3e418ad3d6f154ddc8e0f45"
_AUTOMODEL_RELEASE_COMMIT = "d02f49cb314554715aabb97e8dba6599c9f6e9e0"
_AUTOMODEL_RELEASE_TAG = "v0.5.0"
_AUTOMODEL_WHEEL = "nemo_automodel-0.5.0-py3-none-any.whl"
_AUTOMODEL_PACKAGE_FILE_COUNT = 490
_EXPECTED_PAIRS = {"train-one": (0, 63), "resume-one": (124, 127)}
_STAGE_RESULT_KEYS = {
    "schema_version",
    "record_type",
    "stage",
    "pid",
    "world_size",
    "model",
    "pdd",
    "source",
    "config_sha256",
    "automodel",
    "gpu",
    "pair",
    "sample_ids",
    "diagnostics",
    "teacher_calls_per_rank",
    "checkpoint",
    "resume",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train-one", "resume-one", "validate"), required=True)
    parser.add_argument("--run-root", type=pathlib.Path, required=True)
    parser.add_argument("--before-automodel", type=pathlib.Path)
    parser.add_argument("--after-automodel", type=pathlib.Path)
    return parser.parse_args()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} is not a hexadecimal SHA-256 digest") from error
    return value.lower()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"JSON contains non-finite value {token}")


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(
        path.read_bytes(),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _finite_positive(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _absolute_path_without_symlinks(value: pathlib.Path, *, name: str) -> pathlib.Path:
    path = pathlib.Path(os.path.abspath(value))
    for candidate in (*reversed(path.parents), path):
        if candidate.is_symlink():
            raise ValueError(f"{name} cannot traverse a symlink: {candidate}")
    return path


def _regular_directory(value: pathlib.Path, *, name: str) -> pathlib.Path:
    path = _absolute_path_without_symlinks(value, name=name)
    if not path.is_dir():
        raise ValueError(f"{name} must identify a regular directory")
    return path.resolve()


def _regular_file(value: pathlib.Path, *, name: str) -> pathlib.Path:
    path = _absolute_path_without_symlinks(value, name=name)
    if not path.is_file():
        raise ValueError(f"{name} must identify a regular file")
    return path.resolve()


def _create_run_root(value: pathlib.Path) -> pathlib.Path:
    path = _absolute_path_without_symlinks(value, name="smoke run root")
    path.mkdir(parents=True, exist_ok=True)
    return _regular_directory(path, name="smoke run root")


def _relative_regular_file(root: pathlib.Path, value: Any, *, name: str) -> pathlib.Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty relative path")
    relative = pathlib.PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must stay beneath the run root")
    root = _regular_directory(root, name=f"{name} root")
    path = root.joinpath(*relative.parts)
    if any(candidate.is_symlink() for candidate in (path, *path.parents) if candidate != root):
        raise ValueError(f"{name} cannot traverse a symlink")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must stay beneath the run root") from error
    if not resolved.is_file():
        raise ValueError(f"{name} must identify a regular file")
    return resolved


def validate_stage_result(value: Mapping[str, Any], *, stage: str) -> None:
    """Validate one atomic training-stage result without importing GPU dependencies."""
    if set(value) != _STAGE_RESULT_KEYS:
        raise ValueError(f"{stage} result keys are incompatible")
    if value["schema_version"] != 1 or value["record_type"] != "pdd_qwen_smoke_stage":
        raise ValueError(f"{stage} result schema is incompatible")
    if value["stage"] != stage or stage not in _EXPECTED_PAIRS:
        raise ValueError("smoke stage identity is invalid")
    if type(value["pid"]) is not int or value["pid"] <= 0:
        raise ValueError("smoke stage pid is invalid")
    if type(value["world_size"]) is not int or value["world_size"] < 2:
        raise ValueError("full-Qwen smoke requires a multi-GPU world")
    if value["model"] != {"id": _MODEL_ID, "revision": _MODEL_REVISION, "dtype": "bfloat16"}:
        raise ValueError("smoke model identity is invalid")
    if value["pdd"] != {
        "grid_size": 128,
        "grid_max_t": 0.999,
        "flow_shift": 5.0,
        "block_size_min": 4,
        "block_size_max": 64,
        "teacher_integrator": "euler",
        "guidance_scale": 4.0,
        "guidance_rescale": 1.0,
        "guidance_eps": 1e-5,
    }:
        raise ValueError("smoke PDD identity is invalid")
    source = value["source"]
    if (
        not isinstance(source, Mapping)
        or set(source) != {"commit", "dirty"}
        or not isinstance(source["commit"], str)
        or len(source["commit"]) != 40
        or source["dirty"] is not False
    ):
        raise ValueError("smoke source identity is invalid")
    try:
        int(source["commit"], 16)
    except ValueError as error:
        raise ValueError("smoke source commit is not hexadecimal") from error
    _require_sha256(value["config_sha256"], name="config_sha256")
    automodel = value["automodel"]
    expected_automodel_keys = {
        "distribution",
        "version",
        "package_tree_sha256",
        "wheel_sha256",
        "runtime_versions",
    }
    if not isinstance(automodel, Mapping) or set(automodel) != expected_automodel_keys:
        raise ValueError("smoke AutoModel identity is invalid")
    if (
        automodel["distribution"] != "nemo_automodel"
        or automodel["version"] != "0.5.0"
        or automodel["runtime_versions"] != {"diffusers": "0.38.0"}
        or automodel["package_tree_sha256"] != _AUTOMODEL_TREE_SHA256
        or automodel["wheel_sha256"] != _AUTOMODEL_WHEEL_SHA256
    ):
        raise ValueError("smoke AutoModel release identity is invalid")
    _require_sha256(automodel["package_tree_sha256"], name="automodel.package_tree_sha256")
    _require_sha256(automodel["wheel_sha256"], name="automodel.wheel_sha256")
    gpu = value["gpu"]
    if not isinstance(gpu, Mapping) or set(gpu) != {
        "names",
        "total_memory_bytes",
        "host_available_bytes",
        "allocated_before_step_bytes",
        "peak_memory_bytes",
        "student_parameter_bytes",
        "teacher_parameter_bytes",
        "step_seconds",
    }:
        raise ValueError("smoke GPU evidence is invalid")
    if not isinstance(gpu["names"], list) or len(gpu["names"]) != value["world_size"]:
        raise ValueError("smoke GPU inventory does not match world size")
    if any(not isinstance(name, str) or not name for name in gpu["names"]):
        raise ValueError("smoke GPU names are invalid")
    for name in (
        "total_memory_bytes",
        "host_available_bytes",
        "allocated_before_step_bytes",
        "peak_memory_bytes",
    ):
        values = gpu[name]
        if (
            not isinstance(values, list)
            or len(values) != value["world_size"]
            or any(type(item) is not int or item <= 0 for item in values)
        ):
            raise ValueError(f"smoke GPU {name} is invalid")
    for name in ("student_parameter_bytes", "teacher_parameter_bytes"):
        if type(gpu[name]) is not int or gpu[name] <= 0:
            raise ValueError(f"smoke capacity {name} is invalid")
    for allocated, peak, total in zip(
        gpu["allocated_before_step_bytes"],
        gpu["peak_memory_bytes"],
        gpu["total_memory_bytes"],
        strict=True,
    ):
        if not allocated <= peak <= total:
            raise ValueError("smoke GPU allocation evidence is inconsistent")
    _finite_positive(gpu["step_seconds"], name="gpu.step_seconds")
    if value["pair"] != {"n": _EXPECTED_PAIRS[stage][0], "k": _EXPECTED_PAIRS[stage][1]}:
        raise ValueError("smoke explicit support pair is invalid")
    sample_ids = value["sample_ids"]
    if (
        not isinstance(sample_ids, list)
        or len(sample_ids) != value["world_size"]
        or any(not isinstance(item, str) or not item for item in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError("smoke sample IDs are invalid")
    expected_step = 1 if stage == "train-one" else 2
    expected_ids = [
        f"synthetic-pdd-smoke-step-{expected_step}-rank-{rank}"
        for rank in range(value["world_size"])
    ]
    if sample_ids != expected_ids:
        raise ValueError("smoke sample IDs do not match the canonical rank order")
    diagnostics = value["diagnostics"]
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != {
        "completed_step",
        "loss",
        "grad_norm",
        "student_adamw_nominal_update_ratio",
        "pdd_projection_update_ratio",
        "learning_rate",
        "student_velocity_rms",
        "teacher_velocity_rms",
        "student_teacher_velocity_rms_ratio",
        "reconstructed_state_rms",
    }:
        raise ValueError("smoke diagnostics are invalid")
    if diagnostics["completed_step"] != expected_step:
        raise ValueError("smoke completed step is invalid")
    for name in diagnostics:
        if name != "completed_step":
            _finite_positive(diagnostics[name], name=f"diagnostics.{name}")
    calls = value["teacher_calls_per_rank"]
    if calls != [2] * value["world_size"]:
        raise ValueError("guided teacher call structure is invalid")
    checkpoint = value["checkpoint"]
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "path",
        "manifest_sha256",
        "completed_steps",
        "parent_checkpoint",
    }:
        raise ValueError("smoke checkpoint evidence is invalid")
    if checkpoint["completed_steps"] != expected_step:
        raise ValueError("smoke checkpoint step is invalid")
    expected_parent = None if stage == "train-one" else "step_00000001"
    if checkpoint["parent_checkpoint"] != expected_parent:
        raise ValueError("smoke checkpoint lineage is invalid")
    if checkpoint["path"] != f"checkpoints/step_{expected_step:08d}":
        raise ValueError("smoke checkpoint path is invalid")
    _require_sha256(checkpoint["manifest_sha256"], name="checkpoint.manifest_sha256")
    resume = value["resume"]
    if stage == "train-one":
        if resume is not None:
            raise ValueError("first smoke stage cannot have resume evidence")
    elif resume != {
        "selected_checkpoint": "step_00000001",
        "completed_steps": 1,
        "parent_checkpoint": None,
        "first_sample_ids": sample_ids,
        "learning_rate": diagnostics["learning_rate"],
    }:
        raise ValueError("smoke resume evidence is invalid")


def validate_inference_result(value: Mapping[str, Any], *, root: pathlib.Path) -> None:
    """Validate the exact authenticated PDD-4 inference evidence."""
    if value.get("schema_version") != 1 or value.get("record_type") != "pdd_inference":
        raise ValueError("PDD inference result schema is invalid")
    if value.get("condition") != "pdd_4" or value.get("schedule") != "pdd-4":
        raise ValueError("PDD inference schedule identity is invalid")
    if value.get("blocks") != [32, 32, 32, 32]:
        raise ValueError("PDD-4 blocks are invalid")
    if value.get("height") != 1024 or value.get("width") != 1024:
        raise ValueError("PDD-4 smoke output must be exactly 1024x1024")
    if (
        value.get("scheduler_steps") != 4
        or value.get("actual_transformer_invocations") != 4
        or value.get("batch_normalized_transformer_evaluations") != 4
    ):
        raise ValueError("PDD-4 compute counters are invalid")
    _finite_positive(value.get("latency_seconds"), name="inference.latency_seconds")
    output = value.get("output")
    if not isinstance(output, Mapping) or set(output) != {"path", "sha256"}:
        raise ValueError("PDD inference output evidence is invalid")
    image = _relative_regular_file(root, output["path"], name="inference.output.path")
    if image.suffix.lower() != ".png" or image.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("PDD inference output is not a PNG")
    if image.stat().st_size <= 8 or _sha256(image) != _require_sha256(
        output["sha256"], name="inference.output.sha256"
    ):
        raise ValueError("PDD inference PNG hash is invalid")


def _exact_mapping(value: Any, keys: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} must contain exactly {sorted(keys)}")
    return value


def _validate_checkpoint_identity(identity: Any, *, stage: Mapping[str, Any]) -> None:
    from modelopt.torch.fastgen import PDDMetadata

    identity = _exact_mapping(
        identity,
        {
            "schema_version",
            "model",
            "pdd_metadata",
            "guidance",
            "automodel",
            "data",
            "topology",
            "training",
            "optimizer",
            "scheduler",
        },
        name="smoke checkpoint identity",
    )
    if identity["schema_version"] != 1 or identity["model"] != stage["model"]:
        raise ValueError("smoke checkpoint model identity is incompatible")
    metadata = PDDMetadata.from_dict(identity["pdd_metadata"])
    if metadata.to_dict() != identity["pdd_metadata"]:
        raise ValueError("smoke checkpoint PDD metadata is not canonical")
    pdd = stage["pdd"]
    if (
        metadata.grid_size != pdd["grid_size"]
        or metadata.grid_max_t != pdd["grid_max_t"]
        or metadata.flow_shift != pdd["flow_shift"]
        or metadata.block_size_min != pdd["block_size_min"]
        or metadata.block_size_max != pdd["block_size_max"]
        or metadata.teacher_integrator != pdd["teacher_integrator"]
        or metadata.inference_blocks != (32, 32, 32, 32)
        or metadata.layer_spec.to_dict()
        != {
            "projection_path": "transformer.proj_out",
            "head_layout": "channel_major",
            "output_channels": None,
        }
    ):
        raise ValueError("smoke checkpoint PDD metadata does not match the stage")
    if identity["guidance"] != {
        "scale": pdd["guidance_scale"],
        "rescale": pdd["guidance_rescale"],
        "eps": pdd["guidance_eps"],
    }:
        raise ValueError("smoke checkpoint guidance does not match the stage")
    if identity["automodel"] != stage["automodel"]:
        raise ValueError("smoke checkpoint AutoModel identity does not match the stage")
    if identity["topology"] != {
        "world_size": stage["world_size"],
        "pure_data_parallel": True,
    }:
        raise ValueError("smoke checkpoint topology does not match the stage")


def _validate_automodel_snapshot(
    snapshot: Any,
    *,
    expected_automodel: Mapping[str, Any],
) -> None:
    snapshot = _exact_mapping(
        snapshot,
        {
            "distribution",
            "files",
            "import_origin",
            "package_file_count",
            "package_tree_sha256",
            "release_commit",
            "release_tag",
            "root",
            "runtime_versions",
            "version",
            "wheel",
            "wheel_sha256",
        },
        name="AutoModel snapshot",
    )
    for key in ("distribution", "version", "runtime_versions"):
        if snapshot[key] != expected_automodel[key]:
            raise ValueError(f"AutoModel snapshot identity differs for {key}")
    for key in ("package_tree_sha256", "wheel_sha256"):
        if (
            _require_sha256(snapshot[key], name=f"AutoModel snapshot {key}")
            != expected_automodel[key]
        ):
            raise ValueError(f"AutoModel snapshot identity differs for {key}")
    if (
        snapshot["release_commit"] != _AUTOMODEL_RELEASE_COMMIT
        or snapshot["release_tag"] != _AUTOMODEL_RELEASE_TAG
        or snapshot["wheel"] != _AUTOMODEL_WHEEL
        or type(snapshot["package_file_count"]) is not int
        or snapshot["package_file_count"] != _AUTOMODEL_PACKAGE_FILE_COUNT
    ):
        raise ValueError("AutoModel snapshot release identity is invalid")
    root_value = snapshot["root"]
    origin_value = snapshot["import_origin"]
    if not isinstance(root_value, str) or not isinstance(origin_value, str):
        raise TypeError("AutoModel snapshot root and import origin must be strings")
    root = pathlib.Path(root_value)
    import_origin = pathlib.Path(origin_value)
    if not root.is_absolute() or not import_origin.is_absolute():
        raise ValueError("AutoModel snapshot paths must be absolute")
    try:
        import_origin.relative_to(root)
    except ValueError as error:
        raise ValueError("AutoModel snapshot import origin is outside its root") from error
    files = snapshot["files"]
    if not isinstance(files, list) or len(files) != _AUTOMODEL_PACKAGE_FILE_COUNT:
        raise ValueError("AutoModel snapshot file inventory is invalid")
    tree = hashlib.sha256()
    previous_path: str | None = None
    for index, raw_record in enumerate(files):
        record = _exact_mapping(
            raw_record,
            {"path", "sha256", "size"},
            name=f"AutoModel snapshot files[{index}]",
        )
        path = record["path"]
        if (
            not isinstance(path, str)
            or not path
            or pathlib.PurePosixPath(path).is_absolute()
            or "\\" in path
            or any(part in ("", ".", "..") for part in path.split("/"))
            or (previous_path is not None and path <= previous_path)
        ):
            raise ValueError("AutoModel snapshot paths must be sorted normalized references")
        digest = _require_sha256(record["sha256"], name=f"AutoModel snapshot files[{index}].sha256")
        if type(record["size"]) is not int or record["size"] < 0:
            raise ValueError(f"AutoModel snapshot files[{index}].size is invalid")
        tree.update(path.encode())
        tree.update(b"\0")
        tree.update(digest.encode())
        tree.update(b"\0")
        tree.update(str(record["size"]).encode())
        tree.update(b"\n")
        previous_path = path
    if tree.hexdigest() != snapshot["package_tree_sha256"]:
        raise ValueError("AutoModel snapshot inventory does not match its tree digest")


def _validate_bundle_links(
    *,
    stage1: Mapping[str, Any],
    stage2: Mapping[str, Any],
    manifest1: Mapping[str, Any],
    manifest2: Mapping[str, Any],
    export_manifest: Mapping[str, Any],
    automodel_snapshot: Mapping[str, Any],
) -> None:
    identity1 = manifest1.get("identity")
    identity2 = manifest2.get("identity")
    if identity1 != identity2:
        raise ValueError("smoke checkpoint identities differ across resume")
    _validate_checkpoint_identity(identity1, stage=stage1)
    _validate_checkpoint_identity(identity2, stage=stage2)
    if export_manifest.get("identity") != identity2:
        raise ValueError("smoke export identity does not match step 2")
    if export_manifest.get("modelopt_source") != stage2["source"]:
        raise ValueError("smoke export source does not match the training source")
    expected_checkpoint = {
        "name": "step_00000002",
        "manifest_sha256": stage2["checkpoint"]["manifest_sha256"],
        "completed_steps": 2,
    }
    if export_manifest.get("source_checkpoint") != expected_checkpoint:
        raise ValueError("smoke export does not derive from the exact step-2 checkpoint")
    _validate_automodel_snapshot(
        automodel_snapshot,
        expected_automodel=stage2["automodel"],
    )


def _load_matching_automodel_snapshots(
    before_path: pathlib.Path,
    after_path: pathlib.Path,
    *,
    expected_automodel: Mapping[str, Any],
) -> Mapping[str, Any]:
    before_path = _regular_file(before_path, name="before AutoModel snapshot")
    after_path = _regular_file(after_path, name="after AutoModel snapshot")
    before_bytes = before_path.read_bytes()
    if before_bytes != after_path.read_bytes():
        raise ValueError("AutoModel package snapshot changed during full-Qwen smoke")
    snapshot = _read_json(before_path)
    _validate_automodel_snapshot(snapshot, expected_automodel=expected_automodel)
    return snapshot


def _raw_config(run_root: pathlib.Path, world_size: int) -> dict[str, Any]:
    return {
        "model": {
            "pretrained_model_name_or_path": _MODEL_ID,
            "revision": _MODEL_REVISION,
            "torch_dtype": "bfloat16",
            "device": "cuda",
            "transformer_engine_linear": False,
            "peft": None,
            "guidance_embeds": False,
            "fuse_qkv_projections": False,
        },
        "pdd": {
            "pred_type": "flow",
            "num_train_timesteps": None,
            "guidance_scale": 4.0,
            "student_sample_steps": 4,
            "student_sample_type": "ode",
            "grid_size": 128,
            "grid_max_t": 0.999,
            "flow_shift": 5.0,
            "block_size_min": 4,
            "block_size_max": 64,
            "teacher_integrator": "euler",
            "inference_blocks": [32, 32, 32, 32],
            "data_free": False,
        },
        "optim": {
            "learning_rate": 2.0e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
        },
        "guidance": {"rescale": 1.0, "eps": 1e-5},
        "training": {
            "seed": 42,
            "max_steps": 2,
            "max_grad_norm": 1.0,
            "zero_grad_warmup_steps": 0,
            "log_every_steps": 1,
            "checkpoint_every_steps": 1,
            "validation_every_steps": 1000,
            "grad_accumulation_steps": 1,
            "global_batch_size": world_size,
            "validation_seed": 2026,
        },
        "fsdp": {
            "dp_size": world_size,
            "tp_size": 1,
            "cp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "activation_checkpointing": True,
        },
        "data": {
            "all_metadata_index": "synthetic_metadata.json",
            "validation_metadata_index": "synthetic_heldout.json",
            "dataloader": {
                "_target_": "fastgen_data.build_text_to_image_multiresolution_dataloader",
                "cache_dir": "synthetic-unused",
                "metadata_index": "synthetic_train.json",
                "base_resolution": [1024, 1024],
                "batch_size": 1,
                "drop_last": True,
                "shuffle": False,
                "dynamic_batch_size": False,
                "negative_prompt_embedding_path": "synthetic_negative.pt",
            },
        },
        "checkpoint": {
            "enabled": True,
            "checkpoint_dir": str(run_root / "checkpoints"),
            "model_save_format": "torch_save",
            "save_consolidated": False,
            "restore_from": "LATEST",
        },
    }


def _modelopt_source() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if dirty:
        raise RuntimeError("full-Qwen smoke requires a clean ModelOpt checkout")
    return {"commit": commit, "dirty": False}


def _ordered_id_sha256(sample_ids: Sequence[str], *, split: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"modelopt-pdd-ordered-{split}-ids-v1\0".encode())
    for sample_id in sample_ids:
        digest.update(sample_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _training_sample_ids(world_size: int) -> tuple[str, ...]:
    return tuple(
        f"synthetic-pdd-smoke-step-{step}-rank-{rank}"
        for step in (1, 2)
        for rank in range(world_size)
    )


def _build_sampler(world_size: int, rank: int) -> Any:
    import torch
    from fastgen_data import ReplayableBatchSampler
    from torch.utils.data import Sampler

    sample_ids = _training_sample_ids(world_size)

    class _Dataset:
        metadata = [{"sample_id": sample_id} for sample_id in sample_ids]

    class _Sampler(Sampler[list[int]]):
        def __init__(self) -> None:
            self.dataset = _Dataset()
            self.rank = rank
            self.num_replicas = world_size
            self.epoch = 0
            self.batches_yielded = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def load_state_dict(self, state: Mapping[str, Any]) -> None:
            self.epoch = int(state["epoch"])
            self.batches_yielded = int(state["batches_yielded"])

        def __iter__(self):
            batches = ([rank], [world_size + rank])
            for index, batch in enumerate(batches):
                if index >= self.batches_yielded:
                    self.batches_yielded = index + 1
                    yield list(batch)

        def __len__(self) -> int:
            return 2

    assert torch.distributed.get_world_size() == world_size
    return ReplayableBatchSampler(_Sampler())


def _prepared_batch(
    setup: Any, *, rank: int, step: int, device: Any, dtype: Any
) -> tuple[Any, Any]:
    import torch
    from pdd_training import PreparedPDDBatch

    config = getattr(setup.student, "config", None)
    in_channels = getattr(config, "in_channels", None)
    condition_width = getattr(config, "joint_attention_dim", None)
    if type(in_channels) is not int or in_channels <= 0 or in_channels % 4:
        raise RuntimeError("pinned Qwen config has invalid in_channels")
    if type(condition_width) is not int or condition_width <= 0:
        raise RuntimeError("pinned Qwen config has invalid joint_attention_dim")
    generator = torch.Generator(device="cpu").manual_seed(10_000 + rank * 10 + step)
    latent_shape = (1, in_channels // 4, 128, 128)
    data = torch.randn(latent_shape, generator=generator, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    noise = torch.randn(latent_shape, generator=generator, dtype=torch.float32).to(device=device)
    condition = torch.randn((1, 8, condition_width), generator=generator, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    negative = torch.randn((1, 8, condition_width), generator=generator, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], device=device, dtype=torch.long)
    negative_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=torch.long)
    sample_id = f"synthetic-pdd-smoke-step-{step}-rank-{rank}"
    batch = PreparedPDDBatch(
        data=data,
        condition=(condition, mask),
        negative_condition=(negative, negative_mask),
        sample_ids=(sample_id,),
        valid_mask=(True,),
    )
    return batch, noise


def _identity(
    *, setup: Any, training: Any, config: Any, sampler: Any, raw: Mapping[str, Any]
) -> dict[str, Any]:
    from pdd_checkpoint import build_pdd_checkpoint_identity

    train_ids = tuple(item["sample_id"] for item in sampler.dataset.metadata)
    heldout_ids = ("synthetic-pdd-smoke-heldout-not-evaluated",)
    return build_pdd_checkpoint_identity(
        metadata=setup.metadata,
        model_id=config.model_id,
        model_revision=config.model_revision,
        guidance_scale=config.pdd.guidance_scale,
        guidance_rescale=config.guidance.rescale,
        guidance_eps=config.guidance.eps,
        automodel_snapshot=setup.automodel_snapshot,
        ordered_train_id_sha256=_ordered_id_sha256(train_ids, split="train"),
        ordered_heldout_id_sha256=_ordered_id_sha256(heldout_ids, split="heldout"),
        dataset_snapshot_sha256=_canonical_sha256(
            {"domain": "modelopt-pdd-synthetic-smoke-v1", "config": raw["pdd"]}
        ),
        local_batch_size=1,
        grad_accumulation_steps=1,
        training_seed=config.training.seed,
        validation_seed=config.training.validation_seed,
        validation_every_steps=config.training.validation_every_steps,
        max_grad_norm=config.training.max_grad_norm,
        zero_grad_warmup_steps=config.training.zero_grad_warmup_steps,
        activation_checkpointing=config.parallel.activation_checkpointing,
        dtype="bfloat16",
        optimizer=setup.optimizer,
        scheduler=training.scheduler,
    )


def _diagnostics_dict(diagnostics: Any) -> dict[str, Any]:
    return {
        "completed_step": diagnostics.completed_step,
        "loss": diagnostics.loss,
        "grad_norm": diagnostics.grad_norm,
        "student_adamw_nominal_update_ratio": diagnostics.student_adamw_nominal_update_ratio,
        "pdd_projection_update_ratio": diagnostics.pdd_projection_update_ratio,
        "learning_rate": diagnostics.learning_rate,
        "student_velocity_rms": diagnostics.student_velocity_rms,
        "teacher_velocity_rms": diagnostics.teacher_velocity_rms,
        "student_teacher_velocity_rms_ratio": diagnostics.student_teacher_velocity_rms_ratio,
        "reconstructed_state_rms": diagnostics.reconstructed_state_rms,
    }


def _run_training_stage(stage: str, run_root: pathlib.Path) -> None:
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("full-Qwen smoke requires HF_HUB_OFFLINE=1 and a pinned local snapshot")
    import torch
    import torch.distributed as dist
    from export_pdd_qwen_image import host_available_bytes
    from pdd_artifacts import write_canonical_json
    from pdd_checkpoint import PDDCheckpointManager, validate_pdd_training_checkpoint
    from pdd_recipe import (
        build_pdd_setup,
        build_pdd_training_artifacts,
        initialize_pdd_distributed,
        resolve_pdd_recipe_config,
    )

    initialize_pdd_distributed(backend="nccl", timeout_minutes=60)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise RuntimeError("full-Qwen smoke requires a multi-GPU FSDP2 world")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    run_root = _create_run_root(run_root)
    result_path = run_root / ("stage1.json" if stage == "train-one" else "stage2.json")
    if result_path.exists() or result_path.is_symlink():
        raise FileExistsError(f"smoke stage result already exists: {result_path}")

    raw = _raw_config(run_root, world_size)
    config = resolve_pdd_recipe_config(raw)
    setup = build_pdd_setup(config)
    training = build_pdd_training_artifacts(setup, config)
    sampler = _build_sampler(world_size, rank)
    identity = _identity(setup=setup, training=training, config=config, sampler=sampler, raw=raw)
    manager = PDDCheckpointManager(
        root=config.checkpoint.checkpoint_dir,
        checkpointer=setup.checkpointer,
        model=setup.student,
        optimizer=setup.optimizer,
        scheduler=training.scheduler,
        trainer=training.trainer,
        sampler=sampler,
        rng=training.rng,
        identity=identity,
    )
    resume_payload = None
    try:
        if stage == "train-one":
            if manager.resolve("LATEST") is not None:
                raise RuntimeError("first smoke stage requires an empty checkpoint root")
            step = 1
        else:
            resume = manager.load("LATEST")
            if resume is None:
                raise RuntimeError("resume smoke stage found no LATEST checkpoint")
            if (
                resume.checkpoint_path.name != "step_00000001"
                or resume.completed_steps != 1
                or resume.parent_checkpoint is not None
                or training.trainer.completed_steps != 1
            ):
                raise RuntimeError("resume smoke stage restored incompatible lineage")
            expected_ids = sampler.expected_next_sample_ids()
            resume.verify_first_batch(expected_ids)
            resume_payload = {
                "selected_checkpoint": resume.checkpoint_path.name,
                "completed_steps": resume.completed_steps,
                "parent_checkpoint": resume.parent_checkpoint,
                "first_sample_ids": list(expected_ids),
                "learning_rate": float(setup.optimizer.param_groups[0]["lr"]),
            }
            step = 2

        batch, noise = _prepared_batch(
            setup,
            rank=rank,
            step=step,
            device=device,
            dtype=config.dtype,
        )
        if sampler.expected_next_sample_ids() != batch.sample_ids:
            raise RuntimeError("synthetic smoke batch does not match the committed sampler cursor")
        n_value, k_value = _EXPECTED_PAIRS[stage]
        n = torch.tensor([n_value], device=device, dtype=torch.int64)
        k = torch.tensor([k_value], device=device, dtype=torch.int64)
        teacher_calls = 0

        def count_teacher_call(_module: Any, _args: Any, _kwargs: Any) -> None:
            nonlocal teacher_calls
            teacher_calls += 1

        hook = setup.teacher.register_forward_pre_hook(count_teacher_call, with_kwargs=True)
        allocated_before_step = torch.cuda.memory_allocated(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            diagnostics = training.trainer.train_step(
                batch,
                noise=noise,
                n=n,
                k=k,
                measure_updates=True,
            )
            training.scheduler.step()
            sampler.commit(batch.sample_ids)
        finally:
            hook.remove()
        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - started
        calls = [None] * world_size
        dist.all_gather_object(calls, teacher_calls)
        if calls != [2] * world_size:
            raise RuntimeError(f"guided teacher calls differ across ranks: {calls}")
        checkpoint = manager.save()
        manifest = validate_pdd_training_checkpoint(
            checkpoint,
            expected_identity=identity,
            expected_world_size=world_size,
        )

        sample_ids = [None] * world_size
        dist.all_gather_object(sample_ids, batch.sample_ids[0])
        gpu_name = torch.cuda.get_device_name(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        host_available = host_available_bytes()
        peak_memory = torch.cuda.max_memory_allocated(device)
        gpu_names = [None] * world_size
        total_memories = [None] * world_size
        host_memories = [None] * world_size
        allocated_memories = [None] * world_size
        peak_memories = [None] * world_size
        dist.all_gather_object(gpu_names, gpu_name)
        dist.all_gather_object(total_memories, total_memory)
        dist.all_gather_object(host_memories, host_available)
        dist.all_gather_object(allocated_memories, allocated_before_step)
        dist.all_gather_object(peak_memories, peak_memory)
        seconds = torch.tensor(step_seconds, device=device, dtype=torch.float64)
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        automodel = {
            key: setup.automodel_snapshot[key]
            for key in (
                "distribution",
                "version",
                "package_tree_sha256",
                "wheel_sha256",
                "runtime_versions",
            )
        }
        result = {
            "schema_version": 1,
            "record_type": "pdd_qwen_smoke_stage",
            "stage": stage,
            "pid": os.getpid(),
            "world_size": world_size,
            "model": {"id": _MODEL_ID, "revision": _MODEL_REVISION, "dtype": "bfloat16"},
            "pdd": {
                "grid_size": 128,
                "grid_max_t": 0.999,
                "flow_shift": 5.0,
                "block_size_min": 4,
                "block_size_max": 64,
                "teacher_integrator": "euler",
                "guidance_scale": 4.0,
                "guidance_rescale": 1.0,
                "guidance_eps": 1e-5,
            },
            "source": _modelopt_source(),
            "config_sha256": _canonical_sha256(raw),
            "automodel": automodel,
            "gpu": {
                "names": gpu_names,
                "total_memory_bytes": total_memories,
                "host_available_bytes": host_memories,
                "allocated_before_step_bytes": allocated_memories,
                "peak_memory_bytes": peak_memories,
                "student_parameter_bytes": sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in setup.student.parameters()
                ),
                "teacher_parameter_bytes": sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in setup.teacher.parameters()
                ),
                "step_seconds": float(seconds.item()),
            },
            "pair": {"n": n_value, "k": k_value},
            "sample_ids": sample_ids,
            "diagnostics": _diagnostics_dict(diagnostics),
            "teacher_calls_per_rank": calls,
            "checkpoint": {
                "path": checkpoint.relative_to(run_root).as_posix(),
                "manifest_sha256": _sha256(checkpoint / "manifest.json"),
                "completed_steps": manifest["completed_steps"],
                "parent_checkpoint": manifest["parent_checkpoint"],
            },
            "resume": resume_payload,
        }
        validate_stage_result(result, stage=stage)
        if rank == 0:
            write_canonical_json(result_path, result)
        dist.barrier()
    finally:
        setup.checkpointer.close()
        dist.destroy_process_group()


def _validate_bundle(
    run_root: pathlib.Path,
    before_automodel: pathlib.Path,
    after_automodel: pathlib.Path,
) -> pathlib.Path:
    from pdd_artifacts import load_canonical_json, write_canonical_json
    from pdd_checkpoint import validate_pdd_training_checkpoint
    from pdd_export import inspect_pdd_export

    run_root = _regular_directory(run_root, name="smoke run root")
    stage1_path = _relative_regular_file(run_root, "stage1.json", name="stage-1 result")
    stage2_path = _relative_regular_file(run_root, "stage2.json", name="stage-2 result")
    export_root = _regular_directory(run_root / "export", name="smoke export root")
    export_manifest_path = _regular_file(
        export_root / "manifest.json", name="smoke export manifest"
    )
    inference_path = _relative_regular_file(
        run_root, "inference/pdd4.json", name="smoke inference result"
    )
    output_path = run_root / "smoke_result.json"
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"smoke result already exists: {output_path}")

    stage1 = load_canonical_json(stage1_path)
    stage2 = load_canonical_json(stage2_path)
    if not isinstance(stage1, Mapping) or not isinstance(stage2, Mapping):
        raise TypeError("smoke stage results must be JSON objects")
    validate_stage_result(stage1, stage="train-one")
    validate_stage_result(stage2, stage="resume-one")
    if stage1["pid"] == stage2["pid"]:
        raise ValueError("forced-resume stages did not use fresh processes")
    for name in ("world_size", "model", "pdd", "source", "config_sha256", "automodel"):
        if stage1[name] != stage2[name]:
            raise ValueError(f"smoke stages disagree on {name}")
    checkpoint1 = _relative_regular_file(
        run_root,
        pathlib.PurePosixPath(stage1["checkpoint"]["path"]).joinpath("manifest.json").as_posix(),
        name="stage1.checkpoint.manifest",
    ).parent
    checkpoint2 = _relative_regular_file(
        run_root,
        pathlib.PurePosixPath(stage2["checkpoint"]["path"]).joinpath("manifest.json").as_posix(),
        name="stage2.checkpoint.manifest",
    ).parent
    if _sha256(checkpoint1 / "manifest.json") != stage1["checkpoint"]["manifest_sha256"]:
        raise ValueError("stage-1 checkpoint manifest hash changed")
    if _sha256(checkpoint2 / "manifest.json") != stage2["checkpoint"]["manifest_sha256"]:
        raise ValueError("stage-2 checkpoint manifest hash changed")
    manifest1 = validate_pdd_training_checkpoint(
        checkpoint1, expected_world_size=stage1["world_size"]
    )
    manifest2 = validate_pdd_training_checkpoint(
        checkpoint2, expected_world_size=stage2["world_size"]
    )
    if manifest1["completed_steps"] != 1 or manifest2["completed_steps"] != 2:
        raise ValueError("smoke checkpoint steps are invalid")
    if (
        manifest1["parent_checkpoint"] is not None
        or manifest2["parent_checkpoint"] != checkpoint1.name
    ):
        raise ValueError("smoke checkpoint parent lineage is invalid")
    export_descriptor = inspect_pdd_export(export_root)
    if export_descriptor.root != export_root:
        raise ValueError("smoke export descriptor resolved an unexpected root")
    inference = load_canonical_json(inference_path)
    if not isinstance(inference, Mapping):
        raise TypeError("smoke inference result must be a JSON object")
    validate_inference_result(inference, root=inference_path.parent)
    export_sha256 = _sha256(export_manifest_path)
    if inference.get("export_manifest_sha256") != export_sha256:
        raise ValueError("inference does not authenticate the smoke export")
    automodel_snapshot = _load_matching_automodel_snapshots(
        before_automodel,
        after_automodel,
        expected_automodel=stage2["automodel"],
    )
    _validate_bundle_links(
        stage1=stage1,
        stage2=stage2,
        manifest1=manifest1,
        manifest2=manifest2,
        export_manifest=export_descriptor.manifest,
        automodel_snapshot=automodel_snapshot,
    )
    result = {
        "schema_version": 1,
        "record_type": "pdd_qwen_operability_smoke",
        "status": "passed",
        "stage1_sha256": _sha256(stage1_path),
        "stage2_sha256": _sha256(stage2_path),
        "checkpoint_manifest_sha256": [
            stage1["checkpoint"]["manifest_sha256"],
            stage2["checkpoint"]["manifest_sha256"],
        ],
        "export_manifest_sha256": export_sha256,
        "inference_result_sha256": _sha256(inference_path),
        "automodel_snapshot_sha256": _sha256(
            _regular_file(before_automodel, name="before AutoModel snapshot")
        ),
        "model": stage1["model"],
        "pdd": stage1["pdd"],
        "source": stage1["source"],
        "config_sha256": stage1["config_sha256"],
        "world_size": stage1["world_size"],
        "pairs": [stage1["pair"], stage2["pair"]],
        "losses": [stage1["diagnostics"]["loss"], stage2["diagnostics"]["loss"]],
        "inference": {
            "blocks": inference["blocks"],
            "scheduler_steps": inference["scheduler_steps"],
            "actual_transformer_invocations": inference["actual_transformer_invocations"],
            "batch_normalized_transformer_evaluations": inference[
                "batch_normalized_transformer_evaluations"
            ],
            "output_sha256": inference["output"]["sha256"],
            "latency_seconds": inference["latency_seconds"],
        },
    }
    write_canonical_json(output_path, result)
    return output_path


def main() -> None:
    args = _parse_args()
    if args.stage == "validate":
        if args.before_automodel is None or args.after_automodel is None:
            raise ValueError("validate requires --before-automodel and --after-automodel")
        print(_validate_bundle(args.run_root, args.before_automodel, args.after_automodel))
        return
    if args.before_automodel is not None or args.after_automodel is not None:
        raise ValueError("training stages do not accept AutoModel snapshot arguments")
    _run_training_stage(args.stage, args.run_root)


if __name__ == "__main__":
    main()
