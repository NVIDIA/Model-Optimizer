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

"""CPU end-to-end test for NVFP4 Conv3d export on the Wan 2.2 VAE.

Validates that the Wan ``WanCausalConv3d`` modules (registered as
``_QuantDiffusersWanCausalConv3d``) are routed through the conv packing path,
saved to disk via ``save_pretrained`` with the packed flattened-K schema, and
carry no quantizer-state keys on disk. Uses a tiny ``AutoencoderKLWan`` so it
runs on CPU.
"""

import glob
from pathlib import Path

import pytest
import torch

pytest.importorskip("diffusers")

from _test_utils.torch.diffusers_models import get_tiny_wan22_vae
from safetensors.torch import load_file

import modelopt.torch.quantization as mtq
from modelopt.torch.export.diffusers_utils import hide_quantizers_from_state_dict
from modelopt.torch.export.layer_utils import is_quantconv3d
from modelopt.torch.export.unified_export_hf import _process_quantized_modules

BLOCK_SIZE = 16

CONV_NVFP4_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: BLOCK_SIZE, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
            },
            "enable": True,
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: BLOCK_SIZE, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
            },
            "enable": True,
        },
    ],
    "algorithm": "max",
}


def _calibrate_noop(model):
    # Dynamic NVFP4 needs no calibration data; attempt a tiny decode for
    # realism but tolerate any tiny-config/CPU mismatch.
    try:
        with torch.no_grad():
            model.decode(torch.randn(1, model.config.z_dim, 1, 8, 8))
    except Exception:
        pass


def test_wan22_vae_conv3d_export_and_save(tmp_path: Path):
    vae = get_tiny_wan22_vae().to(torch.float32).eval()

    mtq.quantize(vae, CONV_NVFP4_CFG, _calibrate_noop)

    # The Wan VAE convs are WanCausalConv3d -> _QuantDiffusersWanCausalConv3d,
    # which is_quantconv3d must recognize.
    quant_convs = [name for name, m in vae.named_modules() if is_quantconv3d(m)]
    assert quant_convs, "expected at least one quantized Conv3d in the Wan VAE"

    _process_quantized_modules(vae, torch.float32)

    # A representative conv is now packed in-place.
    sample = vae.get_submodule(quant_convs[0])
    assert sample.weight.dtype == torch.uint8

    out_dir = tmp_path / "wan_vae_nvfp4"
    with hide_quantizers_from_state_dict(vae):
        vae.save_pretrained(str(out_dir))

    state_dict: dict[str, torch.Tensor] = {}
    for f in glob.glob(str(out_dir / "*.safetensors")):
        state_dict.update(load_file(f))

    # No quantizer state leaked onto disk.
    leaked = [k for k in state_dict if "_quantizer" in k]
    assert leaked == [], f"quantizer keys leaked into saved checkpoint: {leaked[:5]}"

    # Every exported conv satisfies the flattened-K schema.
    conv_prefixes = [
        k[: -len(".weight_scale_2")] for k in state_dict if k.endswith(".weight_scale_2")
    ]
    assert conv_prefixes, "no NVFP4 conv layers found on disk"
    for prefix in conv_prefixes:
        weight = state_dict[f"{prefix}.weight"]
        scale = state_dict[f"{prefix}.weight_scale"]
        scale_2 = state_dict[f"{prefix}.weight_scale_2"]
        assert weight.dtype == torch.uint8
        assert scale.dtype == torch.float8_e4m3fn
        assert scale_2.dtype == torch.float32
        assert weight.dim() == 2
        # weight is [O, K_pad/2], scale is [O, K_pad/16] -> 8 packed bytes per scale entry.
        assert weight.shape[0] == scale.shape[0]
        assert weight.shape[1] == scale.shape[1] * (BLOCK_SIZE // 2)
