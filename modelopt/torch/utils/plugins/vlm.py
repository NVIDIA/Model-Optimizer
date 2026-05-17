# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Helpers for running Megatron-Bridge prune / distill pipelines on Vision-Language Models.

The Megatron-Core providers behind ``examples/megatron_bridge/{prune_minitron,distill}.py``
expect a plain causal LM, not a VLM container. These helpers let those scripts extract the
language tower from a VLM into a standalone HF causal-LM checkpoint, run the existing
mcore pipeline on it, and reinsert the result into the VLM container — preserving the
vision encoder, projector, and ``lm_head`` byte-for-byte.

``hidden_size`` is enforced as invariant: the vision projector outputs features in the
original embed dim, so changing it would silently break the multimodal alignment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modelopt.torch.export.model_utils import get_language_model_from_vl, is_multimodal_model

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "TEXT_CONFIG_FIELDS",
    "assert_hidden_size_invariant",
    "extract_text_tower_to_hf_causal_lm",
    "get_submodule_by_path",
    "is_vlm_checkpoint",
    "reinsert_pruned_lm_into_vlm",
    "resolve_vlm_lm_path",
]

TEXT_CONFIG_FIELDS = (
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
)

# Maps the ``model_type`` reported by a VLM's ``text_config`` to the corresponding plain
# causal-LM ``model_type`` so AutoModelForCausalLM / AutoBridge can ingest the extracted LM.
# Add entries here when adding support for new VLM families.
VLM_TEXT_TO_CAUSAL_MODEL_TYPE = {
    "qwen3_vl_text": "qwen3",
    "qwen2_5_vl_text": "qwen2",
    "qwen2_vl_text": "qwen2",
    "llava_text": "llama",
}


def _vlm_text_to_causal_model_type(text_model_type: str) -> str:
    """Translate a VLM's text-config ``model_type`` to a plain causal-LM ``model_type``."""
    if text_model_type in VLM_TEXT_TO_CAUSAL_MODEL_TYPE:
        return VLM_TEXT_TO_CAUSAL_MODEL_TYPE[text_model_type]
    # Heuristic: many VLM text configs are named "<base>_vl_text" or "<base>_vl".
    for suffix in ("_vl_text", "_vl"):
        if text_model_type.endswith(suffix):
            return text_model_type[: -len(suffix)]
    return text_model_type


def resolve_vlm_lm_path(model: nn.Module) -> str | None:
    """Return the attribute path of the language tower inside a VLM, or ``None``.

    Wraps :func:`get_language_model_from_vl` and converts the lineage into a dotted path
    relative to ``model``. Returns ``None`` for non-VLM models so callers can fall back to
    operating on the top-level model unchanged.
    """
    if not hasattr(model, "config"):
        return None
    try:
        if not is_multimodal_model(model):
            return None
    except AttributeError:
        return None

    lineage = get_language_model_from_vl(model)
    if lineage is None or len(lineage) < 2:
        return None

    parts: list[str] = []
    parent = lineage[0]
    for child in lineage[1:]:
        attr = _find_attribute_name(parent, child)
        if attr is None:
            return None
        parts.append(attr)
        parent = child
    return ".".join(parts)


def _find_attribute_name(parent: nn.Module, child: nn.Module) -> str | None:
    for name, candidate in parent.named_children():
        if candidate is child:
            return name
    for name in dir(parent):
        if name.startswith("_"):
            continue
        try:
            candidate = getattr(parent, name)
        except AttributeError:
            continue
        if candidate is child:
            return name
    return None


def get_submodule_by_path(model: nn.Module, path: str) -> nn.Module:
    """Return the submodule reachable from ``model`` via ``path`` (dotted attributes)."""
    obj: Any = model
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def assert_hidden_size_invariant(orig_text_cfg: Any, new_text_cfg: Any) -> None:
    """Raise ``ValueError`` if hidden_size differs between two text configs.

    The vision projector outputs features in the original embed dim. If the language tower's
    hidden_size changes, the projector and the LM no longer line up and the VLM cannot be
    used without retraining the projector. We enforce this up front so callers get a clear
    error instead of a shape mismatch deep inside a forward pass.
    """
    orig = getattr(orig_text_cfg, "hidden_size", None)
    new = getattr(new_text_cfg, "hidden_size", None)
    if orig is not None and new is not None and orig != new:
        raise ValueError(
            f"Cannot modify VLM language tower: hidden_size mismatch "
            f"(original={orig}, new={new}). The vision projector outputs features in the "
            f"original hidden_size; changing it would require retraining the projector. "
            f"Prune or distill with hidden_size preserved."
        )


def is_vlm_checkpoint(path: str, trust_remote_code: bool = False) -> bool:
    """Return ``True`` if ``path`` is an HF checkpoint whose top-level architecture is a VLM.

    Reads only ``config.json`` -- never loads weights. Used by example scripts to decide
    whether to wrap the existing pruning / distillation pipeline with extract+reinsert.
    """
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
    except Exception:
        return False
    if getattr(cfg, "vision_config", None) is not None:
        return True
    if getattr(cfg, "vision_lora", None) is not None:
        return True
    if getattr(cfg, "audio_processor", None) is not None:
        return True
    archs = getattr(cfg, "architectures", None) or []
    return any("ConditionalGeneration" in a or "ImageTextToText" in a for a in archs)


def extract_text_tower_to_hf_causal_lm(
    vlm_path: str,
    output_dir: str,
    *,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Materialize a VLM's language tower (with ``lm_head``) as a standalone HF causal LM.

    Loads the VLM, walks to its language tower (auto-detected via
    :func:`resolve_vlm_lm_path`), and writes an ``AutoModelForCausalLM`` checkpoint at
    ``output_dir`` whose state dict combines the inner LM weights with the VLM's top-level
    ``lm_head.*`` tensors. The VLM's tokenizer is saved alongside.

    Returns a metadata dict (original VLM path, resolved LM attribute path, architectures)
    that callers can pass to :func:`reinsert_pruned_lm_into_vlm` after the pipeline runs.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except ImportError:  # pragma: no cover
        from transformers import AutoModelForCausalLM as _AutoVLM  # type: ignore[no-redef]

    vlm = _AutoVLM.from_pretrained(
        vlm_path, dtype=torch.bfloat16, trust_remote_code=trust_remote_code
    )
    lm_path = resolve_vlm_lm_path(vlm)
    if lm_path is None:
        raise RuntimeError(
            f"{vlm_path} does not look like a VLM with a recognisable language tower."
        )

    inner_lm = get_submodule_by_path(vlm, lm_path)
    text_cfg = vlm.config.text_config
    causal_model_type = _vlm_text_to_causal_model_type(text_cfg.model_type)
    text_cfg_dict = {k: v for k, v in text_cfg.to_dict().items() if k != "model_type"}
    causal_cfg = AutoConfig.for_model(causal_model_type, **text_cfg_dict)

    causal = AutoModelForCausalLM.from_config(causal_cfg, trust_remote_code=trust_remote_code)
    state = {f"model.{k}": v for k, v in inner_lm.state_dict().items()}
    lm_head = {k: v for k, v in vlm.state_dict().items() if k.startswith("lm_head.")}
    state.update(lm_head)
    load_out = causal.load_state_dict(state, strict=False)
    if load_out.unexpected_keys:
        raise RuntimeError(
            f"unexpected keys when building causal LM: {load_out.unexpected_keys!r}"
        )

    causal.to(torch.bfloat16)
    causal.save_pretrained(output_dir)

    try:
        processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=trust_remote_code)
        tokenizer = getattr(processor, "tokenizer", processor)
        tokenizer.save_pretrained(output_dir)
    except Exception:
        pass  # downstream load_mbridge_model_from_hf will fall back to vlm_path's tokenizer

    return {
        "original_vlm_path": vlm_path,
        "lm_attribute_path": lm_path,
        "architectures": list(getattr(vlm.config, "architectures", []) or []),
    }


def reinsert_pruned_lm_into_vlm(
    pruned_causal_lm_path: str,
    original_vlm_path: str,
    output_dir: str,
    *,
    trust_remote_code: bool = False,
) -> None:
    """Build a VLM at ``output_dir`` by combining a pruned/distilled LM with the original vision tower.

    Complement of :func:`extract_text_tower_to_hf_causal_lm`. Loads the modified causal LM
    (whose config reflects the new shape), builds a fresh VLM container with
    ``text_config`` aligned to those shapes, copies all non-LM tensors from the original
    VLM (vision encoder, projector, embeddings, lm_head, ...), then loads the modified LM
    weights into ``vlm.model.language_model``. The ``hidden_size`` invariant is enforced.

    Saves the assembled VLM and the original processor under ``output_dir``.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except ImportError:  # pragma: no cover
        from transformers import AutoModelForCausalLM as _AutoVLM  # type: ignore[no-redef]

    pruned = AutoModelForCausalLM.from_pretrained(
        pruned_causal_lm_path, dtype=torch.bfloat16, trust_remote_code=trust_remote_code
    )
    pruned_cfg = pruned.config

    vlm_cfg = AutoConfig.from_pretrained(original_vlm_path, trust_remote_code=trust_remote_code)
    text_cfg = vlm_cfg.text_config
    assert_hidden_size_invariant(text_cfg, pruned_cfg)
    for field in TEXT_CONFIG_FIELDS:
        if hasattr(pruned_cfg, field) and hasattr(text_cfg, field):
            setattr(text_cfg, field, getattr(pruned_cfg, field))

    # Build an empty VLM container with the new text shapes; weights come next.
    vlm_cls = type(
        _AutoVLM.from_pretrained(
            original_vlm_path, dtype=torch.bfloat16, trust_remote_code=trust_remote_code
        )
    )
    vlm = vlm_cls(vlm_cfg)
    vlm = vlm.to(torch.bfloat16)

    # Load non-LM tensors from the original VLM (vision encoder, projector, embeddings, lm_head, ...).
    src_vlm = _AutoVLM.from_pretrained(
        original_vlm_path, dtype=torch.bfloat16, trust_remote_code=trust_remote_code
    )
    lm_path = resolve_vlm_lm_path(src_vlm)
    if lm_path is None:
        raise RuntimeError(f"{original_vlm_path}: cannot resolve language tower path.")
    lm_prefix = f"{lm_path}."
    non_lm = {k: v for k, v in src_vlm.state_dict().items() if not k.startswith(lm_prefix)}
    missing, unexpected = vlm.load_state_dict(non_lm, strict=False)
    # `missing` will include the language_model.* keys -- that is expected; we fill them next.
    if unexpected:
        raise RuntimeError(f"unexpected non-LM keys when rebuilding VLM: {unexpected!r}")
    del src_vlm

    # Drop the leading "model." prefix the standalone causal LM uses, then load into the LM tower.
    pruned_state = {
        (k[len("model.") :] if k.startswith("model.") else k): v
        for k, v in pruned.state_dict().items()
    }
    pruned_state = {k: v for k, v in pruned_state.items() if not k.startswith("lm_head.")}
    inner_lm = get_submodule_by_path(vlm, lm_path)
    load_out = inner_lm.load_state_dict(pruned_state, strict=False)
    if load_out.unexpected_keys:
        raise RuntimeError(f"unexpected LM keys: {load_out.unexpected_keys!r}")

    vlm.save_pretrained(output_dir)
    try:
        processor = AutoProcessor.from_pretrained(
            original_vlm_path, trust_remote_code=trust_remote_code
        )
        processor.save_pretrained(output_dir)
    except Exception:
        pass
