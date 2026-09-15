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

import importlib.util
import logging
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

CALIBRATION_PATH = Path(__file__).parents[3] / "examples/diffusers/quantization/calibration.py"


class _ModelType(Enum):
    SDXL_BASE = "sdxl-1.0"
    LTX2 = "ltx-2"
    LTX_VIDEO_DEV = "ltx-video-dev"
    WAN22_T2V_14b = "wan2.2-t2v-14b"
    WAN22_T2V_5b = "wan2.2-t2v-5b"


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _load_calibration_module() -> ModuleType:
    dependencies = {
        "models_utils": _module(
            "models_utils",
            MODEL_DEFAULTS={_ModelType.SDXL_BASE: {}},
            ModelType=_ModelType,
        ),
        "pipeline_manager": _module("pipeline_manager", PipelineManager=object),
        "quantize_config": _module("quantize_config", CalibrationConfig=object),
        "utils": _module("utils", load_calib_prompts=lambda *args: []),
    }
    spec = importlib.util.spec_from_file_location("calibration_under_test", CALIBRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


class _LatentOnlyPipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if kwargs.get("output_type") != "latent":
            raise AssertionError("image calibration must skip VAE decoding")
        return SimpleNamespace(images=object())


def test_image_calibration_requests_latent_output() -> None:
    calibration = _load_calibration_module()
    pipe = _LatentOnlyPipeline()
    calibrator = calibration.Calibrator(
        pipeline_manager=SimpleNamespace(pipe=pipe, pipe_upsample=None),
        config=SimpleNamespace(prompts_dataset=Path("prompts.txt"), num_batches=1, n_steps=2),
        model_type=calibration.ModelType.SDXL_BASE,
        logger=logging.getLogger(__name__),
    )

    calibrator.run_calibration([["a prompt"]])

    assert pipe.calls == [
        {
            "prompt": ["a prompt"],
            "num_inference_steps": 2,
            "output_type": "latent",
        }
    ]
