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

import pytest
import torch
from safetensors.torch import save_file

from modelopt.torch.export.plugins import mcore_custom


@pytest.mark.parametrize(
    "mapping",
    [
        mcore_custom.CustomModuleMapping,
        mcore_custom.NameRemapping,
        mcore_custom.QKVMerging,
        mcore_custom.GroupedMLPMerging,
        mcore_custom.GroupedMLPSlicing,
        mcore_custom.GroupedGatedMLPSlicing,
        mcore_custom.GatedMLPMerging,
        mcore_custom.QKVSlicing,
        mcore_custom.SelfAttentionScaling,
        mcore_custom.GatedMLPSlicing,
        mcore_custom.GatedDeltaNetSlicing,
        mcore_custom.PackNameRemapping,
        mcore_custom.UnpackNameRemapping,
        mcore_custom.PackNameRemappingGPT,
        mcore_custom.UnpackNameRemappingGPT,
    ],
)
def test_mapping_default_kwargs_are_isolated(mapping):
    first = mapping()
    original = dict(first.func_kwargs)
    first.func_kwargs["custom_option"] = True
    assert mapping().func_kwargs == original


def test_split_sizes_keep_tensors_whole_at_tp1(tmp_path, monkeypatch):
    # Mamba's in_proj [z|x|B|C|dt]: its weight_scale_inv has one row per 128 weight rows,
    # so it cannot follow split_sizes.
    sizes = [256, 256, 128, 128, 8]
    weight = torch.randn(sum(sizes), 128).to(torch.float8_e4m3fn)
    scale = torch.rand(7, 1) + 0.5
    conv1d = torch.randn(sum(sizes), 1, 4)  # 3-D, like Mamba's conv1d weight
    save_file(
        {"w.weight": weight, "w.weight_scale_inv": scale, "c.weight": conv1d},
        str(tmp_path / "model.safetensors"),
    )
    monkeypatch.setattr(mcore_custom, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(
        mcore_custom, "get_tensor_model_parallel_world_size", lambda: 1, raising=False
    )

    tensor = mcore_custom.get_safetensor(
        tmp_path, "w.weight", mcore_custom.ParallelConfig(0, "TP", sizes), dequantize=True
    )
    expected = (
        weight.to(torch.bfloat16)
        * scale.to(torch.bfloat16).repeat_interleave(128)[: sum(sizes), None]
    )
    torch.testing.assert_close(tensor, expected)
    torch.testing.assert_close(
        mcore_custom.get_safetensor(
            tmp_path, "c.weight", mcore_custom.ParallelConfig(0, "TP", sizes)
        ),
        conv1d,
    )
    # A weight's split_sizes are still checked at TP=1.
    for bad_sizes in (sizes[:-1], [-8, sum(sizes) + 8]):
        with pytest.raises(ValueError, match="does not split"):
            mcore_custom.get_safetensor(
                tmp_path, "c.weight", mcore_custom.ParallelConfig(0, "TP", bad_sizes)
            )
