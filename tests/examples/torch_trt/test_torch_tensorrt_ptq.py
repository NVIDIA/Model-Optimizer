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
from _test_utils.torch.transformers_models import create_tiny_vit_dir

# Recipe variants the example ships.
_RECIPES = [
    "huggingface/vit/ptq/fp8",
]


@pytest.mark.parametrize("recipe", _RECIPES)
def test_torch_tensorrt_ptq(recipe, tmp_path):
    """End-to-end: load ViT -> mtq.quantize via recipe -> torch_tensorrt.compile.

    Uses a tiny randomly-initialized ViT (saved locally with its image processor)
    rather than downloading a full pretrained checkpoint, so the test stays offline
    and fast while exercising the same module structure (3-channel patch conv,
    attention q/k/v, classifier). The CLI exits non-zero if any step (calibration,
    quantization, TRT compile) fails; the printed argmax comparison is informational
    only.
    """
    pytest.importorskip("torch_tensorrt")

    model_dir = create_tiny_vit_dir(tmp_path)

    cmd_parts = extend_cmd_parts(
        ["python", "torch_tensorrt_ptq.py"],
        model_id=str(model_dir),
        recipe=recipe,
        calib_samples="4",
        batch_size="1",
        save_dir=str(tmp_path / "ckpt"),
    )
    run_example_command(cmd_parts, "torch_trt")
