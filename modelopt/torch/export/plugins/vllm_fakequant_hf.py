# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Export HuggingFace model to vLLM fakequant checkpoint."""

import copy
import logging
import re
import warnings
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

import modelopt.torch.opt as mto
from modelopt.torch.quantization.config import RotateConfig
from modelopt.torch.quantization.conversion import quantizer_state
from modelopt.torch.quantization.model_calib import enable_stats_collection, finish_stats_collection
from modelopt.torch.quantization.nn import QuantModule, SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.utils import get_quantizer_state_dict
from modelopt.torch.quantization.utils.core_utils import enable_weight_access_and_writeback
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
from modelopt.torch.utils import get_unwrapped_name, safe_save

from ..layer_utils import get_experts_list, is_moe
from ..quant_utils import get_quantization_format
from ..unified_export_hf import collect_shared_input_modules

__all__ = [
    "export_hf_vllm_fq_checkpoint",
    "infer_quantizer_prefix_remap",
    "is_weight_quantizer_state_key",
    "merge_amax_tensors_for_group",
]

# Matches ``…weight_quantizer``, ``…weight_quantizer.0``, ``…w13_weight_quantizer.0``,
# and the plural fused-experts form ``…weight_quantizers.0`` (per-expert ModuleList).
_WEIGHT_QUANTIZER_STATE_KEY = re.compile(r"(?:^|\.)(?:\w+_)?weight_quantizers?(?:\.\d+)*$")


def is_weight_quantizer_state_key(key: str) -> bool:
    """Return True for weight-quantizer state keys.

    Includes ``SequentialQuantizer`` entries and fused-experts ``ModuleList``
    entries (``*_weight_quantizers.<idx>``). Matches ``weight_quantizer``,
    ``w13_weight_quantizer``, ``weight_quantizer.0``,
    ``gate_up_proj_weight_quantizers.0``, etc.
    """
    return bool(_WEIGHT_QUANTIZER_STATE_KEY.search(key))


def infer_quantizer_prefix_remap(
    quantizer_keys: dict[str, Any],
    map_fun: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, str]:
    """Infer HF root name → vLLM root (e.g. ``backbone`` → ``model``) for reload/export.

    Map HF root → vLLM root (e.g. ``backbone`` → ``model``) by probing ``map_fun`` with
    synthetic ``<module>.weight`` keys and a 2-D placeholder (quantizer paths are not weight
    keys). Keys under the same HF root must agree on the target root or :exc:`ValueError` is
    raised; failed probes are skipped. Returns ``{hf_root: vllm_root}`` only where the root
    renames; not for arbitrary layer rewrites.

    Args:
        quantizer_keys: HF quantizer state paths as keys (values unused).
        map_fun: HF→vLLM weight ``state_dict`` mapper, same as for ``convert_dict_to_vllm``.

    Returns:
        ``{hf_root: vllm_root}`` for roots that rename; omits identity pairs.
    """
    logger = logging.getLogger(__name__)
    probe_weight = torch.empty((1, 1))
    observed_vllm_root: dict[str, str] = {}

    for key in quantizer_keys:
        first_component = key.split(".")[0]
        last_dot = key.rfind(".")
        if last_dot == -1:
            continue
        probe_key = key[:last_dot] + ".weight"
        try:
            result = map_fun({probe_key: probe_weight})
            if not result:
                continue
            new_key = next(iter(result))
            new_first = new_key.split(".")[0]
        except Exception as e:
            logger.debug("prefix-remap probe failed for %r: %s", probe_key, e)
            continue

        if first_component not in observed_vllm_root:
            observed_vllm_root[first_component] = new_first
        elif observed_vllm_root[first_component] != new_first:
            raise ValueError(
                "Inconsistent HF→vLLM prefix remap for "
                f"{first_component!r}: probes implied "
                f"{observed_vllm_root[first_component]!r} and {new_first!r}. "
                "map_fun must apply one target root per HF root, or use explicit quantizer "
                "key remapping."
            )

    return {
        hf_root: vllm_root
        for hf_root, vllm_root in observed_vllm_root.items()
        if hf_root != vllm_root
    }


def _check_all_weight_quantizers_disabled(model: nn.Module) -> None:
    """Export invariant before writing metadata: every weight quantizer must be off."""
    for _, module in model.named_modules():
        if not isinstance(module, QuantModule):
            continue
        for attr_name, quantizer in module.named_children():
            if attr_name.endswith("weight_quantizer") and isinstance(
                quantizer, (TensorQuantizer, SequentialQuantizer)
            ):
                if quantizer.is_enabled:
                    raise RuntimeError(
                        f"vLLM fakequant export: {attr_name!r} must be disabled before saving "
                        f"quantizer_state (weights already folded). "
                        f"See filter_modelopt_state_quantizer_state_for_model in vllm_reload_utils."
                    )


def disable_rotate(quantizer: TensorQuantizer):
    """Return a disabled copy of the quantizer's ``_rotate`` field, preserving its type."""
    if isinstance(quantizer._rotate, RotateConfig):
        return RotateConfig(enable=False)
    if isinstance(quantizer._rotate, dict):  # backward compat: old checkpoints stored a dict
        return dict(quantizer._rotate, enable=False)
    return False


def _fakequant_fused_experts_weights(
    module: nn.Module,
    module_name: str,
    state_dict: dict | None,
    fakequant_weights: set,
    inplace: bool,
):
    """Apply per-expert fake-quant to a ``_QuantFusedExperts`` module's 3-D weights.

    The base loop in :func:`_fakequant_module_weights` only handles singular
    ``*_weight_quantizer`` attrs (one TensorQuantizer per weight). Fused-experts
    modules expose ``*_weight_quantizers`` (``nn.ModuleList`` with one entry per
    expert) that the base loop skips, leaving the fused 3-D weight unquantized
    in the export and breaking weight-fold round-trips.
    """
    for w_attr, q_attr in (
        ("gate_up_proj", "gate_up_proj_weight_quantizers"),
        ("down_proj", "down_proj_weight_quantizers"),
    ):
        quantizers = getattr(module, q_attr, None)
        if not isinstance(quantizers, nn.ModuleList):
            continue
        if not any(
            isinstance(q, TensorQuantizer) and q.fake_quant and q.is_enabled for q in quantizers
        ):
            continue
        sd_key = f"{module_name}.{w_attr}" if module_name else w_attr
        if sd_key in fakequant_weights:
            raise RuntimeError(f"Weight {sd_key} has already been fakequantized")

        if inplace:
            w = getattr(module, w_attr)
            for idx, q in enumerate(quantizers):
                if not (isinstance(q, TensorQuantizer) and q.fake_quant and q.is_enabled):
                    continue
                slice_ = w.data[idx]
                slice_.copy_(q(slice_.float()).to(w.dtype))
        else:
            if state_dict is None or sd_key not in state_dict:
                continue
            w_3d = state_dict[sd_key].clone()
            for idx, q in enumerate(quantizers):
                if not (isinstance(q, TensorQuantizer) and q.fake_quant and q.is_enabled):
                    continue
                slice_ = w_3d[idx]
                w_3d[idx] = q(slice_.float()).to(slice_.dtype)
            state_dict[sd_key] = w_3d.cpu()
        fakequant_weights.add(sd_key)


def _fakequant_module_weights(
    module: nn.Module,
    module_name: str,
    model: nn.Module,
    state_dict: dict | None,
    input_quantizers_folded_pqs: set,
    fakequant_weights: set,
    requant_weights: set[str],
    inplace: bool,
):
    """Apply fake-quant to a single QuantModule's weights.

    When ``inplace=False``, reads/writes weights from/to ``state_dict``.
    When ``inplace=True``, modifies the module's weight parameters directly.
    """
    if not isinstance(module, QuantModule):
        return
    _fakequant_fused_experts_weights(module, module_name, state_dict, fakequant_weights, inplace)
    for attr_name, quantizer in module.named_children():
        if not (
            attr_name.endswith("weight_quantizer")
            and isinstance(quantizer, TensorQuantizer)
            and quantizer.fake_quant
            and quantizer.is_enabled
        ):
            continue
        weight_name = attr_name.removesuffix("_quantizer")
        prefix = f"{module_name}." if module_name else ""
        sd_key = f"{prefix}{weight_name}"
        if sd_key in fakequant_weights:
            raise RuntimeError(f"Weight {sd_key} has already been fakequantized")

        if inplace:
            w = getattr(module, weight_name)
            if sd_key in requant_weights:
                w_quant = requant_weights_for_export(quantizer, w, copy_quantizer=False)
            else:
                w_quant = quantizer(w.float()).to(w.dtype)
        else:
            if state_dict is None:
                raise RuntimeError("state_dict is required when inplace=False for fakequant export")
            if sd_key not in state_dict:
                continue
            w = state_dict[sd_key]
            if sd_key in requant_weights:
                w_quant = requant_weights_for_export(quantizer, w)
            else:
                w_quant = quantizer(w.float()).to(w.dtype)

        # Fold pre_quant_scale: (x*s)@fake_quant(W) = x@(fake_quant(W)*s)
        # Only valid when input_quantizer does NOT fake-quant activations. If it does
        # fake_quant(x*s), the non-linearity prevents folding s into W.
        inp_attr = attr_name.replace("weight_quantizer", "input_quantizer")
        if hasattr(module, inp_attr):
            inp_q = getattr(module, inp_attr)
            if (
                hasattr(inp_q, "_pre_quant_scale")
                and inp_q._pre_quant_scale is not None
                and not inp_q.is_enabled
            ):
                scale = inp_q._pre_quant_scale.squeeze().to(device=w_quant.device)
                w_quant = (w_quant * scale[None, :]).to(w_quant.dtype)
                inp_q_key = get_unwrapped_name(
                    f"{module_name}.{inp_attr}" if module_name else inp_attr, model
                )
                input_quantizers_folded_pqs.add(inp_q_key)

        if inplace:
            w.data.copy_(w_quant)
        else:
            if state_dict is None:
                raise RuntimeError("state_dict is required when inplace=False for fakequant export")
            state_dict[sd_key] = w_quant.cpu()
        fakequant_weights.add(sd_key)


def _collect_group_pre_quant_scales(
    experts: list[nn.Module],
) -> list[torch.Tensor] | None:
    """Return per-expert ``pre_quant_scale`` tensors if every expert can be averaged; else None.

    Skips groups where any expert has no input quantizer, no pqs (e.g. weight-only AWQ INT4),
    or a disabled input quantizer (pqs already folded / not used).
    """
    pre_quant_scales: list[torch.Tensor] = []
    for expert_module in experts:
        input_quantizer = getattr(expert_module, "input_quantizer", None)
        if (
            input_quantizer is None
            or not input_quantizer.is_enabled
            or input_quantizer.pre_quant_scale is None
        ):
            return None
        pre_quant_scales.append(input_quantizer.pre_quant_scale)
    return pre_quant_scales


def requant_weights_for_export(
    quantizer: TensorQuantizer | SequentialQuantizer,
    weight: torch.Tensor,
    copy_quantizer: bool = True,
) -> torch.Tensor:
    """Requantize folded weights after resmooth (``TensorQuantizer`` or ``SequentialQuantizer``).

    A single ``TensorQuantizer`` is treated as a one-stage chain so the same
    calibrate-then-apply steps cover W4A8-style sequential weights (e.g. INT4→FP8).

    Deepcopy may leave buffers on the original device; ``.to(device=w.device)`` aligns with
    ``w`` (e.g. CPU offload).
    """
    if copy_quantizer:
        copied = copy.deepcopy(quantizer).to(device=weight.device)
    else:
        copied = quantizer
    quantizers: list[TensorQuantizer] = (
        list(copied) if isinstance(copied, SequentialQuantizer) else [copied]
    )

    for quantizer_copy in quantizers:
        quantizer_copy.eval()
        quantizer_copy.reset_amax()
        enable_stats_collection(quantizer_copy)
    weight_quantized = weight
    for quantizer_copy in quantizers:
        weight_quantized = quantizer_copy(weight_quantized)
    for quantizer_copy in quantizers:
        finish_stats_collection(quantizer_copy)
    # Re-run application pass to get the quantized output with the freshly collected amax.
    # The calibration forward above only collected stats; its output is intentionally discarded.
    weight_quantized = weight
    for quantizer_copy in quantizers:
        weight_quantized = quantizer_copy(weight_quantized)
    return weight_quantized.to(weight.dtype)


def merge_amax_tensors_for_group(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Combine `_amax` buffers from a merge group into a single tensor.

    Used when HuggingFace module names are folded to vLLM names (e.g. q/k/v → qkv_proj).

    - If every tensor has the same shape, take the element-wise maximum over the group
      (conservative when each branch carried the same axis layout).
    - If shapes differ: ``torch.cat(..., dim=0)`` assumes **1D per-channel** amaxes in
      fused order (e.g. GQA q/k/v → ``[N_q]`` + ``[N_kv]`` + ``[N_kv]``), matching vLLM’s
      grouped quantizer. Not valid for 2D blockwise amax; on failure, **scalar**
      max (drops channel structure).
    """
    if not tensors:
        raise ValueError("merge_amax_tensors_for_group: expected at least one tensor")
    if len(tensors) == 1:
        return tensors[0]

    first = tensors[0]
    if all(t.shape == first.shape for t in tensors):
        stacked = torch.stack([t.float() for t in tensors], dim=0)
        return torch.amax(stacked, dim=0).to(dtype=first.dtype, device=first.device)

    try:
        return torch.cat(tensors, dim=0).to(dtype=first.dtype, device=first.device)
    except RuntimeError:
        shapes = [tuple(t.shape) for t in tensors]
        warnings.warn(
            f"merge_amax_tensors_for_group: torch.cat failed for shapes {shapes}; "
            "falling back to scalar max which loses per-channel amax structure.",
            stacklevel=2,
        )
        flat = torch.cat([t.reshape(-1).float() for t in tensors])
        return torch.max(flat).to(dtype=first.dtype, device=first.device)


@contextmanager
def _enable_writeback_for_group(
    group: list[nn.Module],
    root_model: nn.Module,
    name_to_module: dict[str, nn.Module],
):
    """Nest ``enable_weight_access_and_writeback`` for every module in ``group`` (one ``with``).

    The stdlib pattern for a *variable* number of context managers is :class:`ExitStack`;
    wrapping it here keeps call sites readable.
    """
    with ExitStack() as stack:
        for m in group:
            stack.enter_context(enable_weight_access_and_writeback(m, root_model, name_to_module))
        yield


def _resmooth_experts_for_export(
    model: nn.Module,
    state_dict: dict[str, Any] | None,
    *,
    inplace: bool = False,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor | None]], set[str]]:
    """Prepare AWQ weights for vLLM fakequant export when several linears share one input quantizer.

    PTQ can assign a different ``pre_quant_scale`` per branch (per expert, or per
    q/k/v projection) even though they see the same activation. vLLM’s fused kernels expose a
    **single** input quantizer for that fused group, so reload must use one scale — otherwise
    activations are scaled wrong for k/v or non-primary experts.

    For each group (MoE experts via ``get_experts_list``; dense shared-input linears
    via ``collect_shared_input_modules`` / hooks), average ``pre_quant_scale``, set weights to
    ``W' = W * old_pqs / avg_pqs`` so the net is unchanged, merge input ``amax`` where needed,
    and return per-``input_quantizer`` tensor overrides for ``modelopt_state_weights``.

    Runs only for AWQ with **enabled** input quantizers (e.g. activation-aware); if inputs are
    disabled and PQS was folded into weights only, there is nothing to unify.

    ``inplace=False`` — adjust a detached ``state_dict`` copy (``state_dict`` required).
    ``inplace=True`` — pass ``state_dict=None``; update live ``nn.Parameter`` data under
    ``_enable_writeback_for_group`` (nested writeback per module so offloaded/meta weights
    materialize before ``copy_``).
    """
    if not inplace and state_dict is None:
        raise ValueError("state_dict is required when inplace=False")
    qfmt = get_quantization_format(model)
    if qfmt is None or "awq" not in qfmt.lower():
        return {}, set()

    name_to_module = dict(model.named_modules()) if inplace else None

    model_type = type(model).__name__.lower()
    id_to_name: dict[int, str] = {id(m): n for n, m in model.named_modules()}
    out: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
    requant_weights: set[str] = set()

    def _process_group(modules: list[nn.Module]) -> None:
        pqs_list = _collect_group_pre_quant_scales(modules)
        if pqs_list is None:
            return

        # Mean and clamp in float32: fp16/bf16 would underflow float32.tiny to 0 and divide by zero.
        pqs_dtype = pqs_list[0].dtype
        avg_pqs = torch.stack([p.float() for p in pqs_list]).mean(0)
        avg_pqs = avg_pqs.clamp(min=torch.finfo(torch.float32).tiny)

        for m in modules:
            nm = id_to_name.get(id(m))
            if nm is None or not hasattr(m, "weight"):
                continue
            w_key = f"{nm}.weight"
            old_pqs = m.input_quantizer._pre_quant_scale
            avg_pqs_dev = avg_pqs.to(device=old_pqs.device, dtype=old_pqs.dtype)
            if torch.equal(old_pqs, avg_pqs_dev):
                continue
            if inplace:
                w_param = m.weight
                ratio = old_pqs.to(dtype=torch.float32, device=w_param.device) / avg_pqs.to(
                    device=w_param.device
                )
                w_param.data.copy_((w_param.to(torch.float32) * ratio).to(w_param.dtype))
            else:
                if state_dict is None:
                    raise RuntimeError(
                        "state_dict is required when inplace=False in _resmooth_experts_for_export"
                    )
                weight = state_dict[w_key]
                ratio = old_pqs.to(dtype=torch.float32, device=weight.device) / avg_pqs.to(
                    device=weight.device
                )
                state_dict[w_key] = (weight.to(torch.float32) * ratio).to(weight.dtype)
            requant_weights.add(w_key)

        synced_amax: torch.Tensor | None = None
        amaxes = [m.input_quantizer.amax for m in modules]
        if all(a is not None for a in amaxes):
            synced_amax = merge_amax_tensors_for_group(amaxes)

        avg_pqs_out = avg_pqs.detach().to(pqs_dtype).clone()
        for m in modules:
            nm = id_to_name.get(id(m))
            if nm is None:
                continue
            out[get_unwrapped_name(f"{nm}.input_quantizer", model)] = (avg_pqs_out, synced_amax)

    # MoE expert groups — must be enumerated by name because MoE routing sends
    # different tokens to each expert, so forward hooks cannot detect them as
    # sharing the same input tensor.
    for _, module in model.named_modules():
        if not is_moe(module):
            continue
        try:
            expert_groups = get_experts_list(module, model_type)
        except NotImplementedError:
            continue
        for experts in expert_groups:
            if not experts:
                continue
            if inplace:
                if name_to_module is None:
                    raise RuntimeError(
                        "name_to_module is required when inplace=True in _resmooth_experts_for_export"
                    )
                with _enable_writeback_for_group(experts, model, name_to_module):
                    _process_group(experts)
            else:
                _process_group(experts)

    # Dense shared-input groups (e.g. q/k/v in GQA attention) — detected via forward
    # hooks so any architecture is covered regardless of projection attribute names.

    dev = next(model.parameters()).device

    def _dummy_forward() -> None:
        # Partial forward is OK: hooks record layers reached before failure.
        with torch.inference_mode():
            try:
                model(torch.ones([1, 2], dtype=torch.long, device=dev))
            except Exception as e:
                logging.getLogger(__name__).debug(
                    "Dummy forward for shared-input detection failed (expected for VLMs): %s", e
                )

    input_to_linear, _ = collect_shared_input_modules(model, _dummy_forward)
    for modules in input_to_linear.values():
        if len(modules) <= 1:
            continue
        if inplace:
            if name_to_module is None:
                raise RuntimeError(
                    "name_to_module is required when inplace=True in _resmooth_experts_for_export"
                )
            with _enable_writeback_for_group(modules, model, name_to_module):
                _process_group(modules)
        else:
            _process_group(modules)

    return out, requant_weights


def export_hf_vllm_fq_checkpoint(
    model: nn.Module,
    export_dir: Path | str,
    inplace_mem_efficient: bool = False,
):
    """Export quantized HF weights + ``vllm_fq_modelopt_state.pth`` for vLLM fake-quant reload.

    Folds fake-quant weights into a ``state_dict()`` copy (optional
    ``pre_quant_scale`` into weight when input fake-quant is off), drops quantizer
    keys from the HF save, briefly disables weight quantizers to snapshot
    ModelOpt/quantizer state, then re-enables them. Weight files are written with an
    explicit ``state_dict`` (and ``hf_quantizer`` cleared during save) so safetensors
    do not pick up live quantizer buffers.

    For MoE models with AWQ quantization, pre_quant_scale is averaged across experts
    and input amax is unified — required because vLLM uses a single input quantizer
    per expert group. By default this updates only a detached ``state_dict`` copy.
    With ``inplace_mem_efficient=True``, resmooth runs **in place** on materialized
    weight parameters only (no ``state_dict``), before the inplace fakequant loop.

    Args:
        model: In-memory quantized model.
        export_dir: Output dir for HF files and ``vllm_fq_modelopt_state.pth``.
        inplace_mem_efficient: When True, applies fake-quant inplace one decoder layer at
            a time using ``enable_weight_access_and_writeback``, avoiding full state
            dict materialization. This is destructive — model weights are permanently
            modified and weight quantizers are not re-enabled after export.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    fakequant_weights: set[str] = set()
    # Input quantizer keys whose _pre_quant_scale was folded into the weight above.
    input_quantizers_folded_pqs: set[str] = set()
    with torch.inference_mode():
        if inplace_mem_efficient:
            # Resmooth shared-input groups, then fakequant (state dict and/or params).
            pqs_overrides, requant_weights = _resmooth_experts_for_export(model, None, inplace=True)
            # Inplace path: iterate decoder layers, one offload<->onload per layer.
            decoder_layers = LayerActivationCollector.get_decoder_layers(model)
            if decoder_layers is None:
                raise RuntimeError(
                    "inplace_mem_efficient=True requires a model with discoverable decoder layers"
                )
            for name, module in model.named_modules():
                if module not in decoder_layers:
                    continue
                with enable_weight_access_and_writeback(module, module):
                    for sub_name, sub_mod in module.named_modules():
                        full_name = f"{name}.{sub_name}" if sub_name else name
                        _fakequant_module_weights(
                            sub_mod,
                            full_name,
                            model,
                            None,
                            input_quantizers_folded_pqs,
                            fakequant_weights,
                            requant_weights,
                            inplace=True,
                        )
            # Meta tensors for offloaded weights (free); offload maps now have
            # fakequanted values via writeback.
            state_dict = model.state_dict()
        else:
            state_dict = model.state_dict()
            # Resmooth shared-input groups, then fakequant (state dict and/or params).
            pqs_overrides, requant_weights = _resmooth_experts_for_export(
                model, state_dict, inplace=False
            )

            # Default path: fakequant into the resmoothed state_dict copy (do not refresh
            # from model.state_dict() or resmooth is lost).
            for module_name, module in model.named_modules():
                with enable_weight_access_and_writeback(module, model):
                    _fakequant_module_weights(
                        module,
                        module_name,
                        model,
                        state_dict,
                        input_quantizers_folded_pqs,
                        fakequant_weights,
                        requant_weights,
                        inplace=False,
                    )

    if inplace_mem_efficient:
        # Let save_pretrained build its own state_dict so offloaded params go through
        # its module_map / get_state_dict_from_offload path (modeling_utils.py:3967+).
        # Passing state_dict= bypasses that path and crashes on meta tensors.
        quantizer_keys = [k for k in state_dict if "quantizer" in k]
        clean_sd = None
    else:
        clean_sd = {k: v for k, v in state_dict.items() if "quantizer" not in k}
        quantizer_keys = None

    # Step 2: Disable weight quantizers, save modelopt state + quantizer state
    # dict, then re-enable. The _disabled=True flag is captured in modelopt_state
    # so that on vLLM reload weight quantizers stay off while input/output/
    # attention quantizers remain active.
    # Rotation is also cleared: the weight was already folded with rotation applied,
    # so if fold_weight is called on reload it must not re-rotate the exported weight.
    wqs_to_restore: list[tuple[TensorQuantizer, Any]] = []
    try:
        for _, module in model.named_modules():
            if isinstance(module, QuantModule):
                for attr_name, quantizer in module.named_children():
                    if not (attr_name.endswith("weight_quantizer") and quantizer.is_enabled):
                        continue
                    if isinstance(quantizer, SequentialQuantizer):
                        quantizer.disable()
                        for sub in quantizer:
                            orig_rotate = sub._rotate
                            if sub.rotate_is_enabled:
                                sub._rotate = disable_rotate(sub)
                            wqs_to_restore.append((sub, orig_rotate))
                    elif isinstance(quantizer, TensorQuantizer):
                        quantizer.disable()
                        orig_rotate = quantizer._rotate
                        if quantizer.rotate_is_enabled:
                            quantizer._rotate = disable_rotate(quantizer)
                        wqs_to_restore.append((quantizer, orig_rotate))

        quantizer_state_dict = get_quantizer_state_dict(model)
        for key in list(quantizer_state_dict):
            if is_weight_quantizer_state_key(key):
                # Fakequant amax is folded into HF weights; do not reload weight quantizer tensors.
                # Reload must force-disable WQs missing from saved state (see
                # ``filter_modelopt_state_quantizer_state_for_model`` assertion in vllm_reload_utils).
                quantizer_state_dict.pop(key)
            elif key in input_quantizers_folded_pqs:
                # pre_quant_scale was folded into the weight; keep the buffer for strict load but
                # save identity so activations are not scaled twice.
                qstate_val = quantizer_state_dict[key]
                if isinstance(qstate_val, dict) and "_pre_quant_scale" in qstate_val:
                    quantizer_state_dict[key]["_pre_quant_scale"] = torch.ones_like(
                        qstate_val["_pre_quant_scale"]
                    )

        # Patch input quantizers with averaged pqs and unified amax so that vLLM's single
        # per-group input quantizer sees consistent values (covers both dense qkv and MoE experts).
        for iq_key, (avg_pqs, max_input_amax) in pqs_overrides.items():
            if iq_key in quantizer_state_dict:
                qstate_val = quantizer_state_dict[iq_key]
                if isinstance(qstate_val, dict):
                    if "_pre_quant_scale" in qstate_val:
                        qstate_val["_pre_quant_scale"] = avg_pqs
                    if max_input_amax is not None and "_amax" in qstate_val:
                        qstate_val["_amax"] = max_input_amax

        modelopt_state = mto.modelopt_state(model)
        _check_all_weight_quantizers_disabled(model)
        # Rebuild quantizer_state from the live model (post-disable) and strip weight-quantizer
        # entries. Apply to every mode that carries quantizer_state so that stale entries from
        # a calibrate pass (which also stores quantizer_state in its metadata) are cleaned up.
        # Reload synthesizes missing WQ rows with ``_disabled`` via
        # ``filter_modelopt_state_quantizer_state_for_model``.
        qstate = quantizer_state(model)
        for key in list(qstate):
            if is_weight_quantizer_state_key(key):
                qstate.pop(key)
        for _mode_str, m_state in modelopt_state.get("modelopt_state_dict", []):
            md = m_state.get("metadata", {})
            if "quantizer_state" in md:
                md["quantizer_state"] = qstate

        # Per-quantizer tensor dict loaded alongside metadata on reload.
        modelopt_state["modelopt_state_weights"] = quantizer_state_dict
        safe_save(modelopt_state, export_dir / "vllm_fq_modelopt_state.pth")

        # Step 3: Save HF weights.
        if inplace_mem_efficient:
            prev_ignore = getattr(model, "_keys_to_ignore_on_save", None)
            model._keys_to_ignore_on_save = quantizer_keys
            try:
                model.save_pretrained(export_dir, save_modelopt_state=False)
            finally:
                model._keys_to_ignore_on_save = prev_ignore
        else:
            model.save_pretrained(export_dir, state_dict=clean_sd, save_modelopt_state=False)

    finally:
        if not inplace_mem_efficient:
            for wq, orig_rotate in wqs_to_restore:
                wq.enable()
                wq._rotate = orig_rotate
