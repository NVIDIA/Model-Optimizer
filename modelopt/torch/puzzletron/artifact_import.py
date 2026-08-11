# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receipt-bound, atomic imports of immutable Puzzletron campaign artifacts."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifact_import_contract import (
    IMPORT_CAMPAIGN_MANIFEST,
    canonical_receipt_identity,
    imported_completion_payload,
    imported_stage_manifest_is_complete,
)
from .artifact_inventory import inventory_campaign_artifacts
from .identity import canonicalize, stable_hash
from .manifest import StageManifest, semantic_stage_config
from .pipeline_config import pipeline_config_from_path

__all__ = ["ArtifactImportError", "DEFAULT_BUNDLES", "import_campaign_artifacts"]


DEFAULT_BUNDLES = ("activation", "depth", "vllm_stats", "scoring", "bypass_evidence")
_EXECUTION_BUNDLES = frozenset(("activation", "depth", "vllm_stats", "scoring"))
_CAMPAIGN_ROOT_PLACEHOLDER = "<campaign-root>"
_EXECUTION_ONLY_CONFIG_KEYS = frozenset(
    ("enabled", "execution", "micro_batch_size", "parallel", "sharding", "topology")
)
_RELOCATION_ONLY_CONFIG_SUFFIXES = ("_output_dir", "_solutions_path")


class ArtifactImportError(RuntimeError):
    """Raised when an artifact import cannot prove that publication is safe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _absolute_without_resolving(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path.cwd() / value


def _reject_symlink_ancestors(path: str | Path, *, label: str) -> None:
    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ArtifactImportError(f"{label} has a symlink ancestor: {current}")


def _safe_relative(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ArtifactImportError(f"receipt has no path for {label}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ArtifactImportError(f"receipt path escapes source root for {label}")
    return path


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactImportError(f"cannot read source receipt: {error}") from error
    if not isinstance(payload, dict):
        raise ArtifactImportError("source receipt must be a JSON object")
    if payload.get("state") != "complete":
        raise ArtifactImportError("source receipt must be complete")
    if payload.get("version") != 2:
        raise ArtifactImportError("unsupported source receipt version")
    if not isinstance(payload.get("artifacts"), dict) or not isinstance(
        payload.get("compatibility"), dict
    ):
        raise ArtifactImportError("source receipt is missing artifact or compatibility metadata")
    identity = payload.get("receipt_identity")
    if not isinstance(identity, str) or identity != canonical_receipt_identity(payload):
        raise ArtifactImportError("source receipt identity is invalid")
    if not isinstance(payload.get("artifact_paths"), dict):
        raise ArtifactImportError("source receipt has no canonical artifact paths")
    return payload


def _selected_bundles(bundles: Iterable[str] | None) -> tuple[str, ...]:
    selected = tuple(DEFAULT_BUNDLES if bundles is None else bundles)
    if not selected or len(set(selected)) != len(selected):
        raise ArtifactImportError("artifact selection must be non-empty and unique")
    unknown = sorted(set(selected) - set(DEFAULT_BUNDLES))
    if unknown:
        raise ArtifactImportError(f"unknown artifact bundles: {', '.join(unknown)}")
    return selected


def _validate_receipt_files(
    source_root: Path,
    receipt: Mapping[str, Any],
    selected: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    plans: dict[str, dict[str, Any]] = {}
    for name in selected:
        artifact = receipt["artifacts"].get(name)
        if not isinstance(artifact, dict) or artifact.get("state") != "complete":
            raise ArtifactImportError(f"receipt bundle is not complete: {name}")
        records = artifact.get("files")
        if not isinstance(records, list) or not records:
            raise ArtifactImportError(f"receipt bundle has no file identities: {name}")
        files = []
        seen: set[Path] = set()
        for record in records:
            if not isinstance(record, dict):
                raise ArtifactImportError(f"invalid receipt file identity for {name}")
            relative = _safe_relative(record.get("path"), label=f"{name} file")
            if relative in seen:
                raise ArtifactImportError(f"duplicate receipt file identity for {name}: {relative}")
            seen.add(relative)
            size = record.get("size")
            digest = record.get("sha256")
            if not isinstance(size, int) or size < 0 or not isinstance(digest, str):
                raise ArtifactImportError(f"invalid receipt file identity for {name}: {relative}")
            source_path = source_root / relative
            _reject_symlink_ancestors(source_path, label="source artifact")
            if not source_path.is_file():
                raise ArtifactImportError(f"missing source artifact: {relative}")
            if source_path.stat().st_size != size or _sha256(source_path) != digest:
                raise ArtifactImportError(
                    f"source mutation does not match receipt identity: {relative}"
                )
            files.append(
                {
                    "source": str(relative),
                    "destination": str(relative),
                    "size": size,
                    "sha256": digest,
                }
            )
        plans[name] = {
            "roots": [str(receipt["artifacts"][name]["path"])],
            "files": sorted(files, key=lambda item: item["destination"]),
        }
    return plans


def _recompute_source_receipt(source_root: Path, receipt: Mapping[str, Any]) -> None:
    current = inventory_campaign_artifacts(
        source_root,
        artifact_paths=receipt["artifact_paths"],
    )
    state = current.get("state")
    if state != "complete":
        qualifier = "duplicate" if state == "duplicate_conflicting" else "semantic"
        raise ArtifactImportError(
            f"source changed after receipt: current source inventory is {qualifier} {state}"
        )
    if current.get("receipt_identity") != receipt.get("receipt_identity"):
        raise ArtifactImportError(
            "source changed after receipt: current source inventory does not match receipt identity"
        )


def _load_target_config(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        if path.suffix.lower() == ".json":
            config = json.loads(path.read_text(encoding="utf-8"))
        else:
            config = pipeline_config_from_path(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactImportError(f"cannot load target config: {error}") from error
    if not isinstance(config, dict):
        raise ArtifactImportError("target config must resolve to a mapping")
    return config


def _normalize_compatibility_path(value: str, campaign_root: Path, key: str | None) -> str:
    root = str(campaign_root)
    if value == root:
        return _CAMPAIGN_ROOT_PLACEHOLDER
    if value.startswith(root + os.sep):
        return _CAMPAIGN_ROOT_PLACEHOLDER + value[len(root) :]
    if key == "path" or (key is not None and key.endswith(("_path", "_dir"))):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        return str(path.resolve(strict=False))
    return value


def _normalize_import_compatibility(
    value: Any,
    *,
    campaign_root: Path,
    key: str | None = None,
) -> Any:
    """Remove execution detail and canonicalize paths before import comparison."""

    if isinstance(value, Mapping):
        normalized = {}
        for child_key, child_value in value.items():
            if child_key in _EXECUTION_ONLY_CONFIG_KEYS or str(child_key).endswith(
                _RELOCATION_ONLY_CONFIG_SUFFIXES
            ):
                continue
            child = _normalize_import_compatibility(
                child_value,
                campaign_root=campaign_root,
                key=str(child_key),
            )
            if not (isinstance(child, Mapping) and not child):
                normalized[child_key] = child
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_import_compatibility(item, campaign_root=campaign_root, key=key)
            for item in value
        ]
    if isinstance(value, (str, Path)):
        return _normalize_compatibility_path(str(value), campaign_root, key)
    return canonicalize(value)


def _mapping_is_subset(candidate: Any, reference: Any) -> bool:
    if isinstance(candidate, Mapping) and isinstance(reference, Mapping):
        return all(
            key in reference and _mapping_is_subset(value, reference[key])
            for key, value in candidate.items()
        )
    return candidate == reference


def _compatibility_mismatch(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> str | None:
    for section in ("model", "data", "dataset", "search_space"):
        if source.get(section) != target.get(section):
            return section

    source_embedding = source.get("embedding_pruning")
    target_embedding = target.get("embedding_pruning")
    if target_embedding is not None and not _mapping_is_subset(target_embedding, source_embedding):
        return "semantic config"

    source_remainder = dict(source)
    target_remainder = dict(target)
    source_remainder.pop("embedding_pruning", None)
    target_remainder.pop("embedding_pruning", None)
    if source_remainder != target_remainder:
        return "semantic config"
    return None


def _apply_granularity_evidence(
    source: dict[str, Any],
    target: Mapping[str, Any],
    receipt: Mapping[str, Any],
    stage: str,
) -> None:
    """Fill newly explicit config selectors only from validated artifact metadata."""

    compatibility = receipt.get("compatibility")
    granularities = (
        compatibility.get("granularities", {}) if isinstance(compatibility, Mapping) else {}
    )
    for section in (stage, "scoring"):
        target_section = target.get(section)
        if not isinstance(target_section, Mapping) or "granularity" not in target_section:
            continue
        source_section = source.setdefault(section, {})
        if not isinstance(source_section, dict) or "granularity" in source_section:
            continue
        section_compatibility = (
            compatibility.get(section, {}) if isinstance(compatibility, Mapping) else {}
        )
        evidence = (
            section_compatibility.get("granularity")
            if isinstance(section_compatibility, Mapping)
            else None
        ) or (granularities.get(section) if isinstance(granularities, Mapping) else None)
        if evidence is None and section == "depth":
            depth_path = Path(str(receipt["campaign_root"])) / str(
                receipt["artifact_paths"]["depth"]
            )
            try:
                trajectory = json.loads(depth_path.read_text(encoding="utf-8"))
                removals = [
                    removal
                    for scenario in trajectory.get("scenarios", [])
                    for removal in scenario.get("removals", [])
                ]
            except (KeyError, OSError, TypeError, json.JSONDecodeError):
                removals = []
            if removals and all(
                isinstance(removal, Mapping) and "kind" in removal for removal in removals
            ):
                evidence = "subblock"
        if evidence == target_section["granularity"]:
            source_section["granularity"] = evidence


def _validate_target_compatibility(
    receipt: Mapping[str, Any],
    selected: tuple[str, ...],
    target_config: Mapping[str, Any] | None,
) -> None:
    if target_config is None:
        return
    target = dict(target_config)
    for name in selected:
        if name not in _EXECUTION_BUNDLES:
            continue
        compatibility = receipt["compatibility"].get(name)
        if not isinstance(compatibility, dict):
            raise ArtifactImportError(f"missing source semantic config compatibility: {name}")
        source_config = compatibility.get("source_semantic_config")
        source_identity = compatibility.get("source_semantic_config_identity")
        if not isinstance(source_config, dict) or not isinstance(source_identity, str):
            raise ArtifactImportError(f"missing source semantic config compatibility: {name}")
        target_projection = semantic_stage_config(target, name)
        try:
            source_root = Path(str(receipt["campaign_root"])).resolve()
        except (KeyError, TypeError) as error:
            raise ArtifactImportError("source receipt has no campaign root") from error
        target_root = _target_campaign_root(target)
        if target_root is None:
            raise ArtifactImportError("target config has no campaign root")
        normalized_source = _normalize_import_compatibility(
            source_config,
            campaign_root=source_root,
        )
        normalized_target = _normalize_import_compatibility(
            target_projection,
            campaign_root=target_root,
        )
        _apply_granularity_evidence(normalized_source, normalized_target, receipt, name)
        mismatch = _compatibility_mismatch(normalized_source, normalized_target)
        if mismatch is not None:
            raise ArtifactImportError(
                f"target {mismatch} compatibility is incompatible with source: {name}"
            )


def _target_campaign_root(config: Mapping[str, Any]) -> Path | None:
    value = config.get("puzzle_dir") or (config.get("experiment") or {}).get("dir")
    return Path(str(value)).resolve() if value else None


def _manifest_path(root: Path, name: str) -> Path:
    if name == "bypass_evidence":
        return root / "manifests/imports/bypass_evidence.json"
    return root / f"manifests/{name}.json"


def _manifest_payload(
    name: str,
    source_root: Path,
    receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
    target_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    artifact = receipt["artifacts"][name]
    if name == "bypass_evidence":
        return {
            "stage": name,
            "status": "evidence",
            "report_only": True,
            "source_campaign": str(source_root),
            "source_path": artifact["path"],
            "receipt_identity": receipt["receipt_identity"],
            "compatibility": receipt["compatibility"].get(name) or {"validated": True},
            "counts": artifact["counts"],
            "output_inventory": [
                {key: item[key] for key in ("destination", "size", "sha256")}
                for item in plan["files"]
            ],
        }

    config = dict(target_config or {})
    semantic_config = semantic_stage_config(config, name)
    stage_manifest = StageManifest(
        stage=name,
        status="imported",
        inputs={
            "source_campaign": str(source_root),
            "receipt_identity": receipt["receipt_identity"],
        },
        outputs={"imported_files": [item["destination"] for item in plan["files"]]},
        config=config,
        semantic_config=semantic_config,
    ).to_dict()
    stage_manifest["started_at"] = None
    stage_manifest["ended_at"] = None
    stage_manifest.update(
        {
            "source_campaign": str(source_root),
            "source_path": artifact["path"],
            "receipt_identity": receipt["receipt_identity"],
            "compatibility": receipt["compatibility"].get(name) or {"validated": True},
            "counts": artifact["counts"],
            "output_inventory": [
                {
                    "path": item["destination"],
                    "size": item["size"],
                    "sha256": item["sha256"],
                }
                for item in plan["files"]
            ],
        }
    )
    return stage_manifest


def _file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)}


def _completion_payload(
    root: Path,
    name: str,
    receipt_identity: str,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    target_config: Mapping[str, Any],
    target_config_path: Path,
) -> dict[str, Any]:
    relative_paths = [item["destination"] for item in plan["files"]]
    relative_paths.append(str(_manifest_path(root, name).relative_to(root)))
    required_artifacts = {
        relative: [_file_record(root, relative)] for relative in sorted(relative_paths)
    }
    return imported_completion_payload(
        stage_id=name,
        target_config=str(target_config_path),
        receipt_identity=receipt_identity,
        expected_semantic_config=semantic_stage_config(dict(target_config), name),
        semantic_identity=str(manifest["semantic_identity"]),
        required_artifacts=required_artifacts,
        stable_hash=stable_hash,
    )


def _campaign_manifest_payload(
    source_root: Path,
    receipt: Mapping[str, Any],
    selected: tuple[str, ...],
    target_config: Mapping[str, Any] | None,
    target_config_path: Path | None,
    *,
    include_receipt: bool = True,
) -> dict[str, Any]:
    payload = {
        "version": 1,
        "status": "complete",
        "source_campaign": str(source_root),
        "receipt_identity": receipt["receipt_identity"],
        "bundles": list(selected),
        "target_config": str(target_config_path) if target_config_path else None,
        "target_config_identity": (
            stable_hash(dict(target_config), prefix="artifact_import_target_cfg")
            if target_config is not None
            else None
        ),
    }
    if include_receipt:
        payload.update(receipt_version=2, receipt=dict(receipt))
    return payload


def _same_file(path: Path, item: Mapping[str, Any], *, read_only: bool = False) -> bool:
    _reject_symlink_ancestors(path, label="destination artifact")
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    if _sha256(path) != item["sha256"]:
        return False
    return not read_only or stat.S_IMODE(path.stat().st_mode) & 0o222 == 0


def _validate_payload_files(
    root: Path,
    plans: Mapping[str, Mapping[str, Any]],
    *,
    read_only: bool,
) -> None:
    for plan in plans.values():
        for item in plan["files"]:
            if not _same_file(root / item["destination"], item, read_only=read_only):
                raise ArtifactImportError(f"final destination hash mismatch: {item['destination']}")


def _validate_existing_destination(
    destination: Path,
    source_root: Path,
    receipt: Mapping[str, Any],
    selected: tuple[str, ...],
    plans: Mapping[str, Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
    target_config: Mapping[str, Any] | None,
    target_config_path: Path | None,
) -> bool:
    if not destination.exists():
        return False
    _reject_symlink_ancestors(destination, label="destination root")
    if not destination.is_dir():
        raise ArtifactImportError(f"conflicting destination: {destination}")
    expected_campaign = _campaign_manifest_payload(
        source_root, receipt, selected, target_config, target_config_path
    )
    legacy_campaign = _campaign_manifest_payload(
        source_root,
        receipt,
        selected,
        target_config,
        target_config_path,
        include_receipt=False,
    )
    # Mirror artifact_import_contract._receipt_inventory's pre-receipt
    # compatibility window. The legacy form lacks receipt binding and must be
    # removed in the first ModelOpt release after Puzzletron v2 GA, once
    # migration manifests have been regenerated.
    campaign_manifest_path = destination / IMPORT_CAMPAIGN_MANIFEST
    _reject_symlink_ancestors(campaign_manifest_path, label="destination import manifest")
    try:
        actual_campaign = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactImportError(f"conflicting destination: {destination}") from error
    if actual_campaign not in (expected_campaign, legacy_campaign):
        raise ArtifactImportError(f"conflicting destination: {destination}")
    _validate_payload_files(destination, plans, read_only=True)
    for name, manifest in manifests.items():
        path = _manifest_path(destination, name)
        _reject_symlink_ancestors(path, label="destination manifest")
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactImportError(f"conflicting destination manifest: {path}") from error
        if current != manifest:
            raise ArtifactImportError(f"conflicting destination manifest: {path}")
        if (
            name in _EXECUTION_BUNDLES
            and target_config is not None
            and target_config_path is not None
        ):
            if not imported_stage_manifest_is_complete(
                destination,
                name,
                current,
                expected_semantic_config=semantic_stage_config(dict(target_config), name),
                stable_hash=stable_hash,
            ):
                raise ArtifactImportError(f"invalid imported stage contract: {name}")
            expected_completion = _completion_payload(
                destination,
                name,
                str(receipt["receipt_identity"]),
                plans[name],
                manifest,
                target_config,
                target_config_path,
            )
            completion_path = destination / f"manifests/completions/{name}.json"
            _reject_symlink_ancestors(completion_path, label="destination completion")
            try:
                current_completion = json.loads(completion_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ArtifactImportError(
                    f"conflicting destination completion: {completion_path}"
                ) from error
            if current_completion != expected_completion:
                raise ArtifactImportError(f"conflicting destination completion: {completion_path}")
    return True


def _copy_payload_files(
    source_root: Path,
    transaction_root: Path,
    plans: Mapping[str, Mapping[str, Any]],
) -> None:
    copied: set[str] = set()
    try:
        for plan in plans.values():
            for item in plan["files"]:
                relative = item["destination"]
                if relative in copied:
                    continue
                copied.add(relative)
                target = transaction_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / item["source"], target)
                if not _same_file(target, item):
                    raise ArtifactImportError(f"temporary copy validation failed: {relative}")
                target.chmod(stat.S_IMODE(target.stat().st_mode) & ~0o222)
    except OSError as error:
        raise ArtifactImportError(f"copy failed: {error}") from error


def _prepare_transaction_metadata(
    transaction_root: Path,
    source_root: Path,
    receipt: Mapping[str, Any],
    selected: tuple[str, ...],
    plans: Mapping[str, Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
    target_config: Mapping[str, Any] | None,
    target_config_path: Path | None,
) -> None:
    for name, manifest in manifests.items():
        _write_json(_manifest_path(transaction_root, name), manifest)
    if target_config is not None and target_config_path is not None:
        for name in selected:
            if name not in _EXECUTION_BUNDLES:
                continue
            completion = _completion_payload(
                transaction_root,
                name,
                str(receipt["receipt_identity"]),
                plans[name],
                manifests[name],
                target_config,
                target_config_path,
            )
            _write_json(transaction_root / f"manifests/completions/{name}.json", completion)
    _write_json(
        transaction_root / IMPORT_CAMPAIGN_MANIFEST,
        _campaign_manifest_payload(
            source_root,
            receipt,
            selected,
            target_config,
            target_config_path,
        ),
    )


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename one path while refusing every existing destination."""

    if not sys.platform.startswith("linux"):
        raise OSError(errno.ENOSYS, "atomic rename-no-replace is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def _publish_campaign_root(transaction_root: Path, destination: Path) -> None:
    try:
        _rename_noreplace(transaction_root, destination)
    except OSError as error:
        if error.errno in {errno.EEXIST, errno.ENOTEMPTY}:
            raise ArtifactImportError(
                f"conflicting destination created during destination race: {destination}"
            ) from error
        if error.errno in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
            # Some distributed filesystems (including Lustre configurations) reject
            # renameat2(RENAME_NOREPLACE). Serialize cooperating importers with an
            # exclusive sibling lock, re-check the destination, then use the
            # filesystem's atomic same-directory rename.
            lock_path = destination.parent / f".{destination.name}.artifact-import.lock"
            try:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as lock_error:
                raise ArtifactImportError(
                    f"another import is publishing destination: {destination}"
                ) from lock_error
            try:
                if destination.exists() or destination.is_symlink():
                    raise ArtifactImportError(
                        f"conflicting destination created during destination race: {destination}"
                    )
                os.rename(transaction_root, destination)
            finally:
                os.close(lock_fd)
                lock_path.unlink(missing_ok=True)
            return
        raise ArtifactImportError(f"publish failed: {error}") from error


def import_campaign_artifacts(
    source_root: str | Path,
    destination_root: str | Path,
    receipt_path: str | Path,
    *,
    bundles: Iterable[str] | None = None,
    dry_run: bool = False,
    target_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and physically import selected bundles as one campaign-root transaction.

    The source bytes must exactly match a deterministic version-2 inventory receipt. A new
    destination is assembled in a sibling temporary directory, validated, made read-only, and
    renamed atomically. Supplying ``target_config_path`` additionally publishes imported semantic
    completion evidence for executable stages; bypass remains report-only evidence.
    """

    _reject_symlink_ancestors(source_root, label="source root")
    _reject_symlink_ancestors(destination_root, label="destination root")
    _reject_symlink_ancestors(receipt_path, label="receipt")
    if target_config_path is not None:
        _reject_symlink_ancestors(target_config_path, label="target config")

    source = _absolute_without_resolving(source_root).resolve()
    destination = _absolute_without_resolving(destination_root).resolve()
    receipt_file = _absolute_without_resolving(receipt_path).resolve()
    config_path = (
        _absolute_without_resolving(target_config_path).resolve()
        if target_config_path is not None
        else None
    )
    receipt = _load_receipt(receipt_file)
    try:
        recorded_source = Path(str(receipt["campaign_root"])).resolve()
    except (KeyError, TypeError) as error:
        raise ArtifactImportError("source receipt has no campaign root") from error
    if recorded_source != source:
        raise ArtifactImportError("source root does not match receipt campaign root")
    if source == destination or destination.is_relative_to(source):
        raise ArtifactImportError("destination must be outside the source campaign")

    selected = _selected_bundles(bundles)
    for name in receipt["artifact_paths"]:
        relative = _safe_relative(receipt["artifact_paths"][name], label=name)
        _reject_symlink_ancestors(source / relative, label="source artifact")
    _recompute_source_receipt(source, receipt)
    plans = _validate_receipt_files(source, receipt, selected)

    target_config = _load_target_config(config_path)
    if target_config is not None and _target_campaign_root(target_config) != destination:
        raise ArtifactImportError("target config campaign root does not match destination root")
    _validate_target_compatibility(receipt, selected, target_config)
    manifests = {
        name: _manifest_payload(name, source, receipt, plans[name], target_config)
        for name in selected
    }
    existing = _validate_existing_destination(
        destination,
        source,
        receipt,
        selected,
        plans,
        manifests,
        target_config,
        config_path,
    )
    result = {
        "status": "planned" if dry_run else "noop" if existing else "imported",
        "source_root": str(source),
        "destination_root": str(destination),
        "receipt_identity": receipt["receipt_identity"],
        "bundles": plans,
    }
    if dry_run or existing:
        return result

    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(destination.parent, label="destination parent")
    transaction_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.artifact-import-", dir=destination.parent)
    )
    try:
        _copy_payload_files(source, transaction_root, plans)
        _prepare_transaction_metadata(
            transaction_root,
            source,
            receipt,
            selected,
            plans,
            manifests,
            target_config,
            config_path,
        )
        if target_config is not None and config_path is not None:
            for name in selected:
                if name not in _EXECUTION_BUNDLES:
                    continue
                if not imported_stage_manifest_is_complete(
                    transaction_root,
                    name,
                    manifests[name],
                    expected_semantic_config=semantic_stage_config(target_config, name),
                    stable_hash=stable_hash,
                ):
                    raise ArtifactImportError(f"invalid imported stage contract: {name}")
        _validate_payload_files(transaction_root, plans, read_only=True)
        _recompute_source_receipt(source, receipt)
    except BaseException:
        shutil.rmtree(transaction_root, ignore_errors=True)
        raise
    try:
        _publish_campaign_root(transaction_root, destination)
    except BaseException:
        if transaction_root.exists():
            shutil.rmtree(transaction_root, ignore_errors=True)
        raise
    return result
