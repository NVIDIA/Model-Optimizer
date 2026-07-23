# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

import modelopt.torch.utils.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file
from transformers import AutoConfig

from ..anymodel.converter import ConverterFactory
from ..anymodel.registry import register_native_config_aliases, resolve_descriptor_from_pretrained
from ..identity import model_identity
from ..manifest import StageManifest
from ..tools.checkpoint_utils_hf import save_model_config
from .common import complete_stage, experiment_dir

__all__ = ["convert_stage"]


def _register_automodel_config_aliases() -> None:
    """Backward-compatible stage wrapper around the shared config registry."""

    register_native_config_aliases()


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _teacher_dir(config: dict[str, Any]) -> Path:
    convert_cfg = config.get("convert") or {}
    return Path(convert_cfg.get("teacher_dir") or experiment_dir(config) / "ckpts" / "teacher")


def _world_size_from_env() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


@contextmanager
def _distributed_if_needed() -> Iterator[None]:
    already_initialized = dist.is_initialized()
    should_setup = _world_size_from_env() > 1 and not already_initialized
    if should_setup:
        dist.setup(timeout=timedelta(minutes=10))
    try:
        yield
    finally:
        if should_setup:
            dist.cleanup()


def _is_anymodel_config(config: Any) -> bool:
    architectures = set(_get(config, "architectures", []) or [])
    text_config = _get(config, "text_config")
    return (
        "AnyModel" in architectures
        or _get(config, "block_configs") is not None
        or _get(text_config, "per_layer_config") is not None
    )


def _has_standard_hf_weights(path: Path) -> bool:
    single_files = ("model.safetensors", "pytorch_model.bin")
    if any((path / name).is_file() for name in single_files):
        return True

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text())
            shard_names = set(index.get("weight_map", {}).values())
        except Exception:
            return False
        if shard_names and all((path / shard).is_file() for shard in shard_names):
            return True
    return False


def _is_complete_checkpoint(path: Path, *, trust_remote_code: bool) -> bool:
    if not (path / "config.json").exists():
        return False
    if not _has_standard_hf_weights(path):
        return False
    config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
    if not _is_anymodel_config(config):
        return False
    block_configs = _get(config, "block_configs", None)
    if block_configs is not None:
        lm_config = _get(config, "text_config", None) or config
        expected_layers = _get(lm_config, "num_hidden_layers", None)
        if expected_layers is not None and len(block_configs) != int(expected_layers):
            return False
    return True


def _checkpoint_weight_map(path: Path) -> tuple[dict[str, str], str | None]:
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        return dict(index.get("weight_map", {})), index_path.name

    single_path = path / "model.safetensors"
    if single_path.is_file():
        with safe_open(single_path, framework="pt", device="cpu") as f:
            return {key: single_path.name for key in f.keys()}, None

    raise ValueError(
        f"Cannot untie word embeddings for {path}: only safetensors checkpoints are supported."
    )


def _checkpoint_has_weight(path: Path, key: str) -> bool:
    try:
        weight_map, _ = _checkpoint_weight_map(path)
    except ValueError:
        return False
    return key in weight_map


def _descriptor_checkpoint_layout_complete(path: Path, descriptor, config: Any) -> bool:
    """Reject stale typed-block schemas and descriptor-declared legacy keys."""
    contract_factory = getattr(descriptor, "generic_decoder_contract", None)
    if not callable(contract_factory):
        return True
    contract = contract_factory(config)
    if contract is None:
        # Descriptors may expose the optional hook through a shared base class
        # while retaining a specialized family converter.  Only descriptors
        # that return a concrete contract participate in generic schema/key
        # validation.
        return True
    from ..anymodel.converter.generic_decoder import GenericDecoderConverter
    from ..block_config import maybe_cast_block_configs

    existing = maybe_cast_block_configs(_get(config, "block_configs", None))
    if not existing:
        return False
    expected = GenericDecoderConverter.create_block_configs(descriptor, config)
    if [block.to_dict() for block in existing] != [block.to_dict() for block in expected]:
        return False

    rewrites = tuple(getattr(contract, "checkpoint_key_rewrites", ()) or ())
    if not rewrites:
        return True
    try:
        weight_map, _ = _checkpoint_weight_map(path)
    except ValueError:
        return False
    from ..anymodel.converter.generic_decoder import rewrite_checkpoint_key

    return all(rewrite_checkpoint_key(key, rewrites) == key for key in weight_map)


def _load_checkpoint_tensor(path: Path, key: str):
    weight_map, _ = _checkpoint_weight_map(path)
    filename = weight_map.get(key)
    if filename is None:
        raise KeyError(f"Weight {key!r} is missing from {path}")
    with safe_open(path / filename, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _set_tie_word_embeddings(config: Any, value: bool) -> None:
    if hasattr(config, "tie_word_embeddings"):
        config.tie_word_embeddings = value
    text_config = _get(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "tie_word_embeddings"):
        text_config.tie_word_embeddings = value


def _ensure_untied_word_embeddings(
    checkpoint_dir: Path,
    *,
    descriptor,
    trust_remote_code: bool,
) -> bool:
    """Make tied checkpoints compatible with AutoModel pipeline parallelism.

    NeMo's HF pipeline splitter rejects ``tie_word_embeddings=True`` because the first and last
    PP stages cannot share the same parameter object.  Puzzletron activation and replacement
    scoring only need a numerically equivalent output head, so tied checkpoints are converted to
    an explicit standalone ``lm_head.weight`` shard while preserving the input embedding.
    """
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=trust_remote_code)
    if not bool(_get(config, "tie_word_embeddings", False)):
        return False

    input_key = f"{descriptor.input_embedding_name()}.weight"
    output_key = f"{descriptor.output_embedding_name()}.weight"
    if not _checkpoint_has_weight(checkpoint_dir, output_key):
        embedding_weight = _load_checkpoint_tensor(checkpoint_dir, input_key).contiguous()
        untied_file = "puzzletron_untied_embeddings.safetensors"
        safe_save_file(
            {output_key: embedding_weight},
            checkpoint_dir / untied_file,
            metadata={"format": "pt"},
        )
        weight_map, _ = _checkpoint_weight_map(checkpoint_dir)
        weight_map[output_key] = untied_file
        index = {"metadata": {"format": "pt"}, "weight_map": weight_map}
        (checkpoint_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )

    _set_tie_word_embeddings(config, False)
    save_model_config(config, checkpoint_dir)
    return True


def _resolve_source_path(source: str) -> Path:
    source_path = Path(source)
    if source_path.exists():
        return source_path
    from huggingface_hub import snapshot_download

    if source.startswith("https://huggingface.co/"):
        model_id = "/".join(source.rstrip("/").split("/")[-2:])
    else:
        model_id = source
    return Path(snapshot_download(repo_id=model_id))


def convert_stage(config: dict[str, Any], manifest: StageManifest):
    """Stage 1: convert a local HF checkpoint into canonical Puzzletron AnyModel."""
    _register_automodel_config_aliases()
    model_cfg = config.get("model") or {}
    source = model_cfg.get("source")
    if source is None:
        raise ValueError("model.source is required for the convert stage")

    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    convert_cfg = config.get("convert") or {}
    untie_word_embeddings = bool(convert_cfg.get("untie_word_embeddings", False))
    teacher_dir = _teacher_dir(config)
    already_anymodel = False
    source_identity = None
    descriptor_payload = None
    skipped = False

    with _distributed_if_needed():
        teacher_complete = _is_complete_checkpoint(teacher_dir, trust_remote_code=trust_remote_code)
        if teacher_complete:
            teacher_config = AutoConfig.from_pretrained(
                teacher_dir, trust_remote_code=trust_remote_code
            )
            resolution = resolve_descriptor_from_pretrained(
                str(teacher_dir),
                trust_remote_code=trust_remote_code,
                descriptor_override=model_cfg.get("descriptor_override"),
            )
            teacher_complete = _descriptor_checkpoint_layout_complete(
                teacher_dir, resolution.descriptor, teacher_config
            )
        if teacher_complete and untie_word_embeddings:
            teacher_config = AutoConfig.from_pretrained(teacher_dir, trust_remote_code=trust_remote_code)
            teacher_tied = bool(_get(teacher_config, "tie_word_embeddings", False))
            if teacher_tied:
                teacher_complete = False

        if teacher_complete:
            skipped = True
        else:
            if dist.is_master():
                source_path = _resolve_source_path(str(source))
            else:
                source_path = None
            source_path = Path(dist.broadcast(str(source_path), src=0))
            source_config = AutoConfig.from_pretrained(source_path, trust_remote_code=trust_remote_code)
            source_identity = model_identity(source_config).value
            if dist.is_master():
                if _is_anymodel_config(source_config):
                    teacher_dir.parent.mkdir(parents=True, exist_ok=True)
                    if source_path.resolve() != teacher_dir.resolve():
                        shutil.copytree(source_path, teacher_dir, dirs_exist_ok=True)
                    already_anymodel = True
                else:
                    resolution = resolve_descriptor_from_pretrained(
                        str(source_path),
                        trust_remote_code=trust_remote_code,
                        descriptor_override=model_cfg.get("descriptor_override"),
                    )
                    converter = ConverterFactory.get(resolution.name)
                    converter.convert(
                        descriptor=resolution.descriptor,
                        input_dir=source_path,
                        output_dir=teacher_dir,
                    )
                    descriptor_payload = resolution.to_dict()
                    already_anymodel = False
                if untie_word_embeddings:
                    if "resolution" not in locals():
                        resolution = resolve_descriptor_from_pretrained(
                            str(teacher_dir),
                            trust_remote_code=trust_remote_code,
                            descriptor_override=model_cfg.get("descriptor_override"),
                        )
                    _ensure_untied_word_embeddings(
                        teacher_dir,
                        descriptor=resolution.descriptor,
                        trust_remote_code=trust_remote_code,
                    )
            dist.barrier()

    teacher_config = AutoConfig.from_pretrained(teacher_dir, trust_remote_code=trust_remote_code)
    outputs = {
        "teacher_dir": str(teacher_dir),
        "teacher_identity": model_identity(teacher_config).value,
        "skipped": skipped,
    }
    if source_identity is not None:
        outputs["source_identity"] = source_identity
    if descriptor_payload is not None:
        outputs["descriptor"] = descriptor_payload
    if not skipped:
        outputs["already_anymodel"] = already_anymodel

    # Single-writer runtime candidate list for vLLM stats (not the final build_library).
    if bool((config.get("vllm_stats") or {}).get("enabled", False)):
        from .pipeline import emit_runtime_subblock_library

        library_path = emit_runtime_subblock_library(
            config,
            teacher_dir=teacher_dir,
            puzzle_dir=experiment_dir(config),
        )
        outputs["runtime_subblock_library"] = str(library_path)

    return complete_stage(
        config,
        manifest,
        outputs=outputs,
        status="skipped" if skipped else "success",
        message="Teacher checkpoint already exists." if skipped else None,
    )
