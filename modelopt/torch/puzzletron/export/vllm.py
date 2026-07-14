# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..identity import model_identity, stable_hash
from ..tools.checkpoint_utils_hf import load_model_config, save_model_config
from ..utils.vllm_adapter import convert_block_configs_to_per_layer_config

__all__ = ["prepare_vllm_config", "prepare_vllm_deploy_copy"]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _delete(obj: Any, key: str) -> None:
    if isinstance(obj, dict):
        obj.pop(key, None)
        return
    try:
        delattr(obj, key)
    except AttributeError:
        pass


def _infer_base_architecture(config: Any) -> str:
    base = _get(config, "base_architecture")
    if base:
        return base
    for arch in list(_get(config, "architectures", None) or ()):
        if arch != "AnyModel":
            return arch
    raise ValueError(
        "Cannot infer base_architecture for vLLM export. Set it during conversion "
        "or provide it through anymodel_arch_info."
    )


def prepare_vllm_config(config: Any, *, descriptor_name: str | None = None) -> bool:
    """Mutate a copied HF config into the vLLM AnyModel deploy shape."""
    base_architecture = _infer_base_architecture(config)
    _set(config, "architectures", ["AnyModel"])
    _set(config, "base_architecture", base_architecture)
    arch_info = dict(_get(config, "anymodel_arch_info", None) or {})
    if arch_info and {"decoder_layer_module", "decoder_layer_class"} <= set(arch_info):
        arch_info.setdefault("base_architecture", base_architecture)
        arch_info.setdefault("block_config_schema", "typed_subblocks_v1")
        if descriptor_name is not None:
            arch_info.setdefault("descriptor", descriptor_name)
        _set(config, "anymodel_arch_info", arch_info)
    else:
        _delete(config, "anymodel_arch_info")
    return convert_block_configs_to_per_layer_config(config)


def prepare_vllm_deploy_copy(
    checkpoint_dir: str | Path,
    deploy_root: str | Path,
    *,
    trust_remote_code: bool = False,
    descriptor_name: str | None = None,
) -> Path:
    """Create an immutable-ish deploy copy and rewrite only that copy for vLLM.

    The canonical checkpoint remains in Puzzletron format with typed
    ``block_configs``. The returned directory is safe for vLLM benchmarking or
    serving and may be discarded by the caller.
    """
    checkpoint_dir = Path(checkpoint_dir)
    deploy_root = Path(deploy_root)
    config = load_model_config(checkpoint_dir, trust_remote_code=trust_remote_code)
    source_id = model_identity(config).value
    export_id = stable_hash(
        {
            "source": source_id,
            "format": "vllm_anymodel",
            "descriptor": descriptor_name,
        },
        prefix="vllm_export",
    )
    deploy_dir = deploy_root / export_id
    if not deploy_dir.exists():
        deploy_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(checkpoint_dir, deploy_dir)

    deploy_config = load_model_config(deploy_dir, trust_remote_code=trust_remote_code)
    prepare_vllm_config(deploy_config, descriptor_name=descriptor_name)
    save_model_config(deploy_config, deploy_dir)
    return deploy_dir
