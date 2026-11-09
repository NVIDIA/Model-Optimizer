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

"""Exercise native NVFP4 RCO through AutoQuant and persistence/export APIs."""

import copy
import io

import torch
from torch import nn

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq


def test_auto_quantize_rco_nvfp4_save_restore_and_export():
    """A constrained NVFP4/FP8/BF16 result retains its outputs through save and export."""
    model = (
        nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128)).cuda().bfloat16().eval()
    )
    original = [parameter.detach().clone() for parameter in model.parameters()]
    batch = torch.randn(1, 8, 128, device="cuda", dtype=torch.bfloat16)
    formats = [copy.deepcopy(mtq.NVFP4_DEFAULT_CFG), copy.deepcopy(mtq.FP8_DEFAULT_CFG)]
    for config in formats:
        config["quant_cfg"].append({"quantizer_name": "*input_quantizer", "enable": False})
    model, state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.5},
        quantization_formats=formats,
        data_loader=[batch],
        forward_step=lambda model, batch: model(batch),
        num_calib_steps=1,
        num_score_steps=1,
        method={"method": "rco", "steps": 2, "num_samples": 1, "seed": 42},
    )
    assert state["best"]["is_satisfied"]
    assert state["best"]["constraints"]["effective_bits"] <= 6.5
    for parameter, reference in zip(model.parameters(), original, strict=True):
        torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
    reference_output = model(batch)
    saved = io.BytesIO()
    mto.save(model, saved)
    saved.seek(0)
    restored = (
        nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128)).cuda().bfloat16().eval()
    )
    restored = mto.restore(restored, saved)
    torch.testing.assert_close(restored(batch), reference_output)
    with mtq.utils.export_torch_mode():
        exported = torch.export.export(model, (batch,))
    torch.testing.assert_close(exported.module()(batch), reference_output)
