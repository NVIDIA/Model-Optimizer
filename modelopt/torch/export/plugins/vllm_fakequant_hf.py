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
from modelopt.torch.quantization.utils import get_quantizer_state_dict

__all__ = ["export_hf_vllm_fq_checkpoint"]


def export_hf_vllm_fq_checkpoint(
    model: nn.Module,
    export_dir: Path | str,
):
    """Exports the torch model weights and amax values separately.

    This function:
    1. Extracts amax values for calibration
    2. Deletes all quantizer parameters from state dict to store only weights in original dtype
    3. Saves the model weights

    Args:
        model: The quantized model to export
        export_dir: Directory to save the amax values

    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    quantizer_state_dict = get_quantizer_state_dict(model)

    modelopt_state = mto.modelopt_state(model)
    modelopt_state["modelopt_state_weights"] = quantizer_state_dict
    torch.save(modelopt_state, export_dir / "vllm_fq_modelopt_state.pth")
    # remove quantizer from model
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

    # Save model
    model.save_pretrained(export_dir, state_dict=model.state_dict(), save_modelopt_state=False)
