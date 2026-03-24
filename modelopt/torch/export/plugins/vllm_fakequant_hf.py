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
from modelopt.torch.quantization.conversion import quantizer_state
from modelopt.torch.quantization.nn import QuantModule, TensorQuantizer
from modelopt.torch.quantization.utils import get_quantizer_state_dict
from modelopt.torch.utils import get_unwrapped_name

__all__ = ["export_hf_vllm_fq_checkpoint"]


def export_hf_vllm_fq_checkpoint(
    model: nn.Module,
    export_dir: Path | str,
):
    """Exports the model with weight quantizers folded into weights.

    Model parameters are never mutated. Folded weights are computed by applying
    each weight quantizer's fake-quant to a copy of the state dict returned by
    model.state_dict(). The only transient in-place change is disabling weight
    quantizers (_disabled=True) while saving modelopt_state, immediately
    restored in a finally block.

    This function:
    1. Builds a clean HF state dict: applies each weight quantizer's fake-quant
       to the corresponding weight tensors in a state dict copy, then filters out
       quantizer tensors
    2. Disables weight quantizers in-place, saves modelopt state and quantizer
       state dict (input/output/attention amaxes; weight quantizers disabled),
       then re-enables weight quantizers
    3. Saves the folded HF weights via save_pretrained

    Args:
        model: The quantized model to export. Not mutated (only _disabled flag on
            weight quantizers is transiently toggled and immediately restored).
        export_dir: Directory to save the checkpoint

    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build the folded HF state dict.
    # model.state_dict() returns detached copies of all tensors, so model
    # parameters are never modified. Apply each weight quantizer's fake-quant
    # to the corresponding weight tensor in the copy.
    state_dict = model.state_dict()
    fakequant_weights = set()
    input_quantizers_folded_pqs = (
        set()
    )  # keys for input_quantizers where pre_quant_scale was folded
    with torch.no_grad():
        for module_name, module in model.named_modules():
            if not isinstance(module, QuantModule):
                continue
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
                assert sd_key not in fakequant_weights, (
                    f"Weight {sd_key} has already been fakequantized"
                )
                if sd_key in state_dict:
                    w = state_dict[sd_key]
                    w_quant = quantizer(w.float()).to(w.dtype).detach()
                    # Fold pre_quant_scale: (x*s)@fake_quant(W) = x@(fake_quant(W)*s)
                    # Only valid when input_quantizer does NOT fake-quant activations. If it does
                    # fake_quant(x*s), the non-linearity prevents folding s into W.
                    inp_attr = attr_name.replace("weight_quantizer", "input_quantizer")
                    if hasattr(module, inp_attr):
                        inp_q = getattr(module, inp_attr)
                        if (
                            hasattr(inp_q, "_pre_quant_scale")
                            and inp_q._pre_quant_scale is not None
                            and (inp_q._disabled or not inp_q._if_quant)
                        ):
                            scale = inp_q._pre_quant_scale.squeeze()
                            w_quant = (w_quant * scale[None, :]).to(w_quant.dtype)
                            inp_q_key = get_unwrapped_name(
                                f"{module_name}.{inp_attr}" if module_name else inp_attr, model
                            )
                            input_quantizers_folded_pqs.add(inp_q_key)
                    state_dict[sd_key] = w_quant
                    fakequant_weights.add(sd_key)

    # Filter quantizer tensors out for a clean HF checkpoint.
    clean_sd = {k: v for k, v in state_dict.items() if "quantizer" not in k}

    # Step 2: Disable weight quantizers, save modelopt state + quantizer state
    # dict, then re-enable. The _disabled=True flag is captured in modelopt_state
    # so that on vLLM reload weight quantizers stay off while input/output/
    # attention quantizers remain active.
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
                    wqs_to_restore.append(quantizer)

    quantizer_state_dict = get_quantizer_state_dict(model)
    for key in list(quantizer_state_dict):
        if key.endswith("weight_quantizer"):
            # Weight quantizer amaxes were folded into weights; clear them so they
            # are not reloaded on the vLLM side.
            quantizer_state_dict[key] = {}
        elif key in input_quantizers_folded_pqs:
            # For input_quantizers in input_quantizers_folded_pqs: we folded pre_quant_scale
            # into weights, so strip it from both tensor state and metadata to avoid double-apply.
            qstate_val = quantizer_state_dict[key]
            if isinstance(qstate_val, dict) and "_pre_quant_scale" in qstate_val:
                qstate_val = {k: v for k, v in qstate_val.items() if k != "_pre_quant_scale"}
                quantizer_state_dict[key] = qstate_val
    modelopt_state = mto.modelopt_state(model)
    # modelopt_state only updates the last mode's metadata; quantize may not be last (e.g. after
    # calibrate). Explicitly refresh quantizer_state so weight_quantizers show _disabled=True.
    # For disabled weight quantizers, keep only minimal metadata (weights are folded, amax unused).
    qstate = quantizer_state(model)
    for key in list(qstate):
        if key.endswith("weight_quantizer") and qstate[key].get("_disabled"):
            qstate[key] = {
                "_disabled": True,
                "_pytorch_state_metadata": {"params": {}, "buffers": {}},
            }
        elif key in input_quantizers_folded_pqs:
            # For input_quantizers in input_quantizers_folded_pqs: we folded pre_quant_scale
            # into weights, so strip it from metadata to avoid double-apply.
            meta = qstate[key].get("_pytorch_state_metadata", {})
            if "_pre_quant_scale" in meta.get("buffers", {}):
                meta["buffers"].pop("_pre_quant_scale")
    for mode_str, m_state in modelopt_state.get("modelopt_state_dict", []):
        if mode_str == "quantize" and "metadata" in m_state:
            m_state["metadata"]["quantizer_state"] = qstate
            break

    modelopt_state["modelopt_state_weights"] = quantizer_state_dict
    torch.save(modelopt_state, export_dir / "vllm_fq_modelopt_state.pth")

    # Step 3: Save HF weights using the pre-built folded state dict.
    model.save_pretrained(export_dir, state_dict=clean_sd, save_modelopt_state=False)

    for wq in wqs_to_restore:
        wq.enable()
