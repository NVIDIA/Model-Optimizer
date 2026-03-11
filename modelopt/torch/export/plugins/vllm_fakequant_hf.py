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
from modelopt.torch.export.layer_utils import is_attention, is_quantlinear
from modelopt.torch.quantization.nn import QuantModule, TensorQuantizer
from modelopt.torch.quantization.utils import get_quantizer_state_dict

__all__ = ["export_hf_vllm_fq_checkpoint"]


def export_hf_vllm_fq_checkpoint(
    model: nn.Module,
    export_dir: Path | str,
):
    """Exports the model with weight quantizers folded into weights.

    This function:
    1. Folds each weight quantizer into its weight (fake-quant applied in-place) and disables it
    2. Saves remaining quantizer states (input/output/attention amaxes) for reload
    3. Saves model weights

    Args:
        model: The quantized model to export
        export_dir: Directory to save the checkpoint

    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Fold weight quantizers into weights in-place and disable them.
    # Weight quantizers must remain in the model until AFTER state is saved so that
    # the disabled state (_disabled=True) is captured in modelopt_state and
    # quantizer_state_dict. Deletion happens in step 3.
    for _, module in model.named_modules():
        if isinstance(module, QuantModule):
            module.fold_weight()
            for attr_name in dir(module):
                if attr_name.endswith("weight_quantizer"):
                    wq = getattr(module, attr_name)
                    if isinstance(wq, TensorQuantizer):
                        wq.disable()

    # Step 2: Save modelopt state with weight quantizers present but disabled.
    # On reload, _disabled=True is restored via set_from_modelopt_state so weight
    # quantizers stay off while input/output/attention quantizers remain active.
    quantizer_state_dict = get_quantizer_state_dict(model)

    modelopt_state = mto.modelopt_state(model)
    modelopt_state["modelopt_state_weights"] = quantizer_state_dict
    torch.save(modelopt_state, export_dir / "vllm_fq_modelopt_state.pth")
    # Step 3: Remove quantizer attrs from model before saving HF weights.
    for _, module in model.named_modules():
        if is_quantlinear(module):
            for attr in ["weight_quantizer", "input_quantizer", "output_quantizer"]:
                if hasattr(module, attr):
                    delattr(module, attr)
            module.export()
        if is_attention(module):
            for attr in [
                "q_bmm_quantizer",
                "k_bmm_quantizer",
                "v_bmm_quantizer",
                "softmax_quantizer",
            ]:
                if hasattr(module, attr):
                    delattr(module, attr)
            module.export()

    # Save model
    model.save_pretrained(export_dir, state_dict=model.state_dict(), save_modelopt_state=False)
