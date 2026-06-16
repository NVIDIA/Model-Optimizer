# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""End-to-end regression guard for the GPT-OSS MXFP4 -> NVFP4 cast path.

This exercises the exact combination that silently regressed (nvbug 6295279 / 6295242):
a transposed-quantize MoE (``GptOssExperts``) with **static-block** NVFP4 weight quantizers
-- which is what ``examples/llm_ptq --cast_mxfp4_to_nvfp4`` produces via
``force_weight_quantizers_static`` -- calibrated with a forward loop.

The regression (the unconditional ``weight_only_quantize`` from #1560 feeding the
*non-transposed* expert weight while the forward feeds the transposed one) raised
``ValueError: Input shape has changed`` during ``mtq.quantize``. Without the
``iter_weights_for_calibration`` fix this test fails at ``mtq.quantize`` below.

Static-block NVFP4 fake quant uses a Triton kernel, so this is GPU-only.
"""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_gpt_oss
from safetensors.torch import save_file

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import NVFP4_DEFAULT_CFG
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer

# The cast helpers live next to the example script, not in the ``modelopt`` package.
_LLM_PTQ_DIR = Path(__file__).resolve().parents[4] / "examples" / "llm_ptq"
if str(_LLM_PTQ_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_PTQ_DIR))

from cast_mxfp4_to_nvfp4 import apply_to_model, force_weight_quantizers_static

_EXPERT_WEIGHT_QUANTIZERS = ("gate_up_proj_weight_quantizer", "down_proj_weight_quantizer")


def _write_lossless_mxfp4_source(model, ckpt_dir: Path) -> None:
    """Write a synthetic MXFP4 source whose ``*_scales``/``*_blocks`` match each expert
    weight quantizer's per-block ``_amax`` count.

    Every E8M0 scale is ``127`` (exponent ``k = 0``), so all blocks are "lossless" / in-range:
    the cast derives a per-block amax of ``6 * 2^0 = 6`` and a per-tensor
    ``global_amax = E2M1_MAX * E4M3_MAX * 2^(k_max - 8) = 6 * 448 * 2^-8 = 10.5``.
    """
    state: dict[str, torch.Tensor] = {}
    for layer_idx in range(model.config.num_hidden_layers):
        experts = model.model.layers[layer_idx].mlp.experts
        for wname in ("gate_up_proj", "down_proj"):
            quantizer = getattr(experts, f"{wname}_weight_quantizer")
            # Two NVFP4 blocks (of 16) share one MXFP4 block (of 32).
            n_mxfp4_blocks = quantizer._amax.numel() // 2
            base = f"model.layers.{layer_idx}.mlp.experts.{wname}"
            state[f"{base}_scales"] = torch.full((n_mxfp4_blocks,), 127, dtype=torch.uint8)
            state[f"{base}_blocks"] = torch.zeros((n_mxfp4_blocks, 16), dtype=torch.uint8)
    save_file(state, str(ckpt_dir / "model-00001-of-00001.safetensors"))
    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {}, "weight_map": dict.fromkeys(state, "model-00001-of-00001.safetensors")}
        )
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="static-block NVFP4 needs a CUDA Triton kernel"
)
def test_gpt_oss_mxfp4_to_nvfp4_cast(tmp_path):
    # intermediate_size != hidden_size so the expert weights are non-square and the
    # transposed-vs-non-transposed calibration orientations are actually distinguishable.
    model = (
        get_tiny_gpt_oss(num_hidden_layers=2, hidden_size=64, intermediate_size=96).cuda().eval()
    )

    # Mirror ``--cast_mxfp4_to_nvfp4``: force the NVFP4 weight quantizers to static block.
    quant_cfg = copy.deepcopy(NVFP4_DEFAULT_CFG)
    force_weight_quantizers_static(quant_cfg["quant_cfg"])

    def forward_loop(m):
        m(torch.randint(0, model.config.vocab_size, (2, 16), device="cuda"))

    # Regression guard: pre-fix this raised "Input shape has changed" because weight-only
    # calibration fed the non-transposed expert weight while the forward fed the transposed one.
    mtq.quantize(model, quant_cfg, forward_loop)

    # Calibration must have promoted every expert weight quantizer to a static NVFP4 quantizer
    # with a per-block ``_amax`` (the cast's precondition), sized from the *transposed* weight.
    for layer_idx in range(model.config.num_hidden_layers):
        experts = model.model.layers[layer_idx].mlp.experts
        for qname in _EXPERT_WEIGHT_QUANTIZERS:
            quantizer = getattr(experts, qname)
            weight = getattr(experts, qname[: -len("_weight_quantizer")])
            assert isinstance(quantizer, NVFP4StaticQuantizer)
            assert quantizer._amax is not None
            assert quantizer._amax.numel() == weight.numel() // 16  # NVFP4 block_size = 16
            assert quantizer._amax.numel() > 1  # per-block, not per-tensor

    # Run the closed-form cast against a matching MXFP4 source and confirm it overrides every
    # expert weight quantizer with the closed-form values (matches names + per-block numel).
    _write_lossless_mxfp4_source(model, tmp_path)
    apply_to_model(model, tmp_path)

    for layer_idx in range(model.config.num_hidden_layers):
        experts = model.model.layers[layer_idx].mlp.experts
        for qname in _EXPERT_WEIGHT_QUANTIZERS:
            quantizer = getattr(experts, qname)
            assert torch.allclose(quantizer._amax, torch.full_like(quantizer._amax, 6.0))
            assert abs(float(quantizer.global_amax) - 10.5) < 1e-3
