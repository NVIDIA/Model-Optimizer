# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation and observability at the canonical-batch/AutoModel boundary."""

from __future__ import annotations

import hashlib
import inspect
from typing import Any, Mapping

import torch

from ...dataset.batch import DataLayout, Modality, PuzzletronBatch

__all__ = [
    "VisionForwardMonitor",
    "canonicalize_position_ids",
    "prepare_native_cp_inputs",
    "validated_forward_kwargs",
    "validate_native_feature_config",
]


def canonicalize_position_ids(
    batch: PuzzletronBatch,
    *,
    descriptor,
    config,
) -> PuzzletronBatch:
    """Apply the descriptor-owned position-ID shape before CP/PP transforms."""
    axes = int(descriptor.position_id_axes(config)) if descriptor is not None else 1
    if axes <= 1:
        return batch
    position_ids = batch.model_kwargs.get("position_ids")
    if position_ids is None:
        position_ids = (
            torch.arange(
                batch.sequence_length,
                device=batch.input_ids.device,
            )
            .unsqueeze(0)
            .expand(batch.batch_size, -1)
        )
    if not isinstance(position_ids, torch.Tensor):
        raise TypeError("canonical position_ids must be a tensor")
    if position_ids.ndim == 3:
        if int(position_ids.shape[0]) != axes:
            raise ValueError(
                f"descriptor requires {axes} position axes, got {position_ids.shape[0]}"
            )
        return batch
    if position_ids.ndim != 2:
        raise ValueError(
            "canonical position_ids must be [batch, sequence] or [axes, batch, sequence]"
        )
    expanded = position_ids.unsqueeze(0).expand(axes, -1, -1).contiguous()
    return batch.replace_model_kwargs(position_ids=expanded)


def prepare_native_cp_inputs(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    num_chunks: int = 1,
) -> dict[str, Any]:
    """Let an AutoModel own CP-specific input preparation when it provides it.

    Models with non-standard positional coordinates (mRoPE), multimodal token
    replacement, or model-owned CP sharding expose this hook.  The generic
    fallback in ``make_cp_batch_and_ctx`` can only synthesize ordinary
    ``[batch, sequence]`` positions, so bypassing the hook silently breaks those
    models.  Models without the hook retain the descriptor-based canonical
    position-ID fallback.
    """
    prepare = getattr(model, "prepare_model_inputs_for_cp", None)
    if not callable(prepare) or payload.get("input_ids") is None:
        return dict(payload)

    prepare_signature = inspect.signature(prepare)
    accepts_arbitrary = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in prepare_signature.parameters.values()
    )
    accepted = {
        name
        for name, parameter in prepare_signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    native_kwargs = {
        key: value for key, value in payload.items() if accepts_arbitrary or key in accepted
    }
    attention_mask = native_kwargs.get("attention_mask")
    input_ids = native_kwargs.get("input_ids")
    if (
        isinstance(attention_mask, torch.Tensor)
        and isinstance(input_ids, torch.Tensor)
        and attention_mask.ndim == 2
        and input_ids.ndim >= 2
        and tuple(attention_mask.shape) == tuple(input_ids.shape[:2])
        and attention_mask.shape[1] > 0
    ):
        # PP scheduling pads a micro-batch with fully masked rows.  Model-owned
        # mRoPE preparers cannot construct positions for an empty row, so make
        # one padding token visible only while producing embeddings/positions.
        # ``result`` below retains the original mask for attention and losses.
        empty_rows = ~attention_mask.bool().any(dim=1)
        if bool(empty_rows.any()):
            attention_mask = attention_mask.clone()
            attention_mask[empty_rows, 0] = 1
            native_kwargs["attention_mask"] = attention_mask
    if accepts_arbitrary or "num_chunks" in accepted:
        native_kwargs["num_chunks"] = int(num_chunks)

    forward_signature = inspect.signature(model.forward)
    forward_accepts_arbitrary = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in forward_signature.parameters.values()
    )
    forward_accepts_pre_embed = "_pre_embed_only" in forward_signature.parameters
    if not forward_accepts_pre_embed and forward_accepts_arbitrary:
        # Native models commonly consume this control flag from ``**kwargs``.
        # Do not infer support from ``**kwargs`` alone: sharding-only hooks
        # (for example GLM DSA) deliberately do not implement a pre-embed
        # forward path and must keep their direct preparer call.
        try:
            forward_accepts_pre_embed = "_pre_embed_only" in inspect.getsource(model.forward)
        except (OSError, TypeError):
            forward_accepts_pre_embed = False
    if forward_accepts_pre_embed:
        prepared = model(_pre_embed_only=True, **native_kwargs)
    else:
        prepared = prepare(**native_kwargs)
    if not isinstance(prepared, Mapping):
        raise TypeError(
            f"{type(model).__name__}.prepare_model_inputs_for_cp must return a mapping, "
            f"got {type(prepared).__name__}"
        )

    result = dict(payload)
    # An embedding-producing hook has consumed token IDs.  Keeping both keys
    # would violate make_cp_batch_and_ctx's exactly-one-primary-input contract.
    if "inputs_embeds" in prepared:
        result.pop("input_ids", None)
    result.update(prepared)
    return result


def validated_forward_kwargs(model: torch.nn.Module, batch: PuzzletronBatch) -> dict[str, Any]:
    """Return all canonical kwargs accepted by ``model.forward`` or fail loudly.

    Multimodal fields must never disappear because an old stage happened to know
    only ``input_ids`` and ``attention_mask``.
    """
    signature = inspect.signature(model.forward)
    accepts_arbitrary = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    kwargs = dict(batch.model_kwargs)
    if batch.layout is DataLayout.PACKED_VARLEN:
        packed_values = {
            "cu_seqlens": batch.sequence.local_cu_seqlens
            if batch.sequence.local_cu_seqlens is not None
            else batch.sequence.global_cu_seqlens,
            "seq_idx": batch.sequence.seq_ids,
            "max_seqlen": batch.sequence.max_seqlen,
        }
        kwargs.update(
            {
                key: value
                for key, value in packed_values.items()
                if value is not None
                and key not in kwargs
                and (accepts_arbitrary or key in accepted)
            }
        )
    unsupported = [key for key in kwargs if not accepts_arbitrary and key not in accepted]
    if unsupported:
        raise TypeError(
            f"{type(model).__name__}.forward does not accept canonical Puzzletron fields: "
            + ", ".join(sorted(unsupported))
        )
    return kwargs


def _nested(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def validate_native_feature_config(config: Mapping[str, Any]) -> None:
    """Enforce the explicitly native-only boundary for the new feature set."""
    force_hf = bool(_nested(config, "model", "force_hf", default=True))
    if not force_hf:
        return
    layout = DataLayout(_nested(config, "data", "layout", default=DataLayout.FIXED.value))
    modality = Modality(_nested(config, "data", "modality", default=Modality.TEXT.value))
    embedding_enabled = bool(_nested(config, "embedding_pruning", "enabled", default=False))
    if layout is DataLayout.PACKED_VARLEN or modality is Modality.MULTIMODAL or embedding_enabled:
        raise ValueError(
            "packed multimodal and embedding-pruning features require native AutoModel; "
            "set model.force_hf=False"
        )


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _checksum(value: Any) -> str:
    tensor = _first_tensor(value)
    if tensor is None:
        return hashlib.sha256(repr(value).encode()).hexdigest()
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(data)
    return digest.hexdigest()


class VisionForwardMonitor:
    """Context-managed evidence that the real vision tower executed."""

    def __init__(self, vision_module: torch.nn.Module):
        self.vision_module = vision_module
        self.forward_count = 0
        self.output_checksums: list[str] = []
        self._handle = None

    def _hook(self, _module, _inputs, output) -> None:
        self.forward_count += 1
        self.output_checksums.append(_checksum(output))

    def __enter__(self) -> "VisionForwardMonitor":
        if self._handle is not None:
            raise RuntimeError("VisionForwardMonitor is already active")
        self._handle = self.vision_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "vision_forward_count": int(self.forward_count),
            "vision_output_checksums": list(self.output_checksums),
        }
