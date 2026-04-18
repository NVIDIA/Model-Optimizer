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

from pathlib import Path

import torch
import torch.nn as nn

import modelopt.torch.opt as mto
from modelopt.torch.quantization.config import RotateConfig
from modelopt.torch.quantization.conversion import quantizer_state
from modelopt.torch.quantization.nn import QuantModule, TensorQuantizer
from modelopt.torch.quantization.utils import get_quantizer_state_dict
from modelopt.torch.quantization.utils.core_utils import enable_weight_access_and_writeback
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
from modelopt.torch.utils import get_unwrapped_name

__all__ = ["export_hf_vllm_fq_checkpoint"]


def disable_rotate(quantizer: TensorQuantizer):
    """Return a disabled copy of the quantizer's ``_rotate`` field, preserving its type."""
    if isinstance(quantizer._rotate, RotateConfig):
        return RotateConfig(enable=False)
    if isinstance(quantizer._rotate, dict):  # backward compat: old checkpoints stored a dict
        return dict(quantizer._rotate, enable=False)
    return False


def _fakequant_module_weights(
    module: nn.Module,
    module_name: str,
    model: nn.Module,
    state_dict: dict | None,
    input_quantizers_folded_pqs: set,
    fakequant_weights: set,
    inplace: bool,
):
    """Apply fake-quant to a single QuantModule's weights.

    When ``inplace=False``, reads/writes weights from/to ``state_dict``.
    When ``inplace=True``, modifies the module's weight parameters directly.
    """
    if not isinstance(module, QuantModule):
        return
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
        assert sd_key not in fakequant_weights, f"Weight {sd_key} has already been fakequantized"

        if inplace:
            w = getattr(module, weight_name)
            w_quant = quantizer(w.float()).to(w.dtype)
        else:
            assert state_dict is not None
            if sd_key not in state_dict:
                continue
            w = state_dict[sd_key]
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
                and inp_q._disabled
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
            assert state_dict is not None
            state_dict[sd_key] = w_quant.cpu()
        fakequant_weights.add(sd_key)


def export_hf_vllm_fq_checkpoint(
    model: nn.Module,
    export_dir: Path | str,
    inplace_mem_efficient: bool = False,
):
    """Export quantized HF weights + ``vllm_fq_modelopt_state.pth`` for vLLM fake-quant reload.

    Folds fake-quant weights into a ``state_dict()`` copy (optional
    ``pre_quant_scale`` into weight when input fake-quant is off), drops quantizer
    keys from the HF save, briefly disables weight quantizers to snapshot
    ModelOpt/quantizer state, then re-enables them. Writes ``export_dir`` via
    ``save_pretrained(..., save_modelopt_state=False)``.

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

    # Step 1: Build the folded HF state dict.
    fakequant_weights = set()
    input_quantizers_folded_pqs = set()
    with torch.inference_mode():
        if inplace_mem_efficient:
            # Inplace path: iterate decoder layers, one offload<->onload per layer.
            decoder_layers = LayerActivationCollector.get_decoder_layers(model)
            assert decoder_layers is not None, (
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
                            inplace=True,
                        )
            # Meta tensors for offloaded weights (free); offload maps now have
            # fakequanted values via writeback.
            state_dict = model.state_dict()
        else:
            # Default path: full state_dict copy, fakequant into the copy.
            state_dict = model.state_dict()
            for module_name, module in model.named_modules():
                with enable_weight_access_and_writeback(module, model):
                    _fakequant_module_weights(
                        module,
                        module_name,
                        model,
                        state_dict,
                        input_quantizers_folded_pqs,
                        fakequant_weights,
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
    wqs_to_restore = []
    for _, module in model.named_modules():
        if isinstance(module, QuantModule):
            for attr_name, quantizer in module.named_children():
                if (
                    attr_name.endswith("weight_quantizer")
                    and isinstance(quantizer, TensorQuantizer)
                    and quantizer.is_enabled
                ):
                    quantizer.disable()
                    orig_rotate = quantizer._rotate
                    if quantizer.rotate_is_enabled:
                        quantizer._rotate = disable_rotate(quantizer)
                    wqs_to_restore.append((quantizer, orig_rotate))

    quantizer_state_dict = get_quantizer_state_dict(model)
    for key in list(quantizer_state_dict):
        if key.endswith("weight_quantizer"):
            # Fakequant amax is folded into HF weights; do not reload weight quantizer tensors.
            quantizer_state_dict.pop(key)
        elif key in input_quantizers_folded_pqs:
            # pre_quant_scale was folded into the weight; keep the buffer for strict load but
            # save identity so activations are not scaled twice.
            qstate_val = quantizer_state_dict[key]
            if isinstance(qstate_val, dict) and "_pre_quant_scale" in qstate_val:
                quantizer_state_dict[key]["_pre_quant_scale"] = torch.ones_like(
                    qstate_val["_pre_quant_scale"]
                )
    modelopt_state = mto.modelopt_state(model)
    # ``modelopt_state`` may be stale if another mode (e.g. calibrate) ran last. Rebuild
    # ``quantizer_state`` and drop disabled weight quantizer entries (weights already folded).
    qstate = quantizer_state(model)
    for key in list(qstate):
        if key.endswith("weight_quantizer") and qstate[key].get("_disabled"):
            qstate.pop(key)

    for mode_str, m_state in modelopt_state.get("modelopt_state_dict", []):
        if mode_str == "quantize" and "metadata" in m_state:
            m_state["metadata"]["quantizer_state"] = qstate
            break

    # Per-quantizer tensor dict loaded alongside metadata on reload.
    modelopt_state["modelopt_state_weights"] = quantizer_state_dict
    torch.save(modelopt_state, export_dir / "vllm_fq_modelopt_state.pth")

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

    if not inplace_mem_efficient:
        for wq, orig_rotate in wqs_to_restore:
            wq.enable()
            wq._rotate = orig_rotate
