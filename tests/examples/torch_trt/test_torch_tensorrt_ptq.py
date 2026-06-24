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
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command

from modelopt.torch.quantization.backends.utils import fp4_compatible

# Recipe variants the example ships. NVFP4's TRT kernels are only generated on
# Blackwell (compute capability >= 10.0), so that case is skipped elsewhere.
_RECIPES = [
    "huggingface/vit/ptq/fp8",
    pytest.param(
        "huggingface/vit/ptq/nvfp4",
        marks=pytest.mark.skipif(
            not fp4_compatible(), reason="NVFP4 requires a Blackwell GPU (SM >= 100)"
        ),
    ),
]


@pytest.mark.parametrize("recipe", _RECIPES)
def test_torch_tensorrt_ptq(recipe, tmp_path):
    """End-to-end: load ViT -> mtq.quantize via recipe -> torch_tensorrt.compile.

    Uses the pretrained ``google/vit-base-patch16-224`` (the example no longer
    exposes a random-weight path). The CLI exits non-zero if any step
    (calibration, quantization, TRT compile) fails; the printed argmax
    comparison is informational only. NVFP4's low-precision kernels require a
    Blackwell GPU, so the nvfp4 case only builds there.
    """
    pytest.importorskip("torch_tensorrt")

    cmd_parts = extend_cmd_parts(
        ["python", "torch_tensorrt_ptq.py"],
        model_id="google/vit-base-patch16-224",
        recipe=recipe,
        calib_samples="4",
        batch_size="1",
        save_dir=str(tmp_path / "ckpt"),
    )
    run_example_command(cmd_parts, "torch_trt")
