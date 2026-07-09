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

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"


def _import_hf_ptq(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("hf_ptq")


def _run_vlm_image_quantize(monkeypatch, quant_cfg):
    hf_ptq = _import_hf_ptq(monkeypatch)
    args = SimpleNamespace(calib_with_images=True, qformat="nvfp4", specdec_offline_dataset=None)
    language_model = SimpleNamespace(name="language_model")
    parent = SimpleNamespace(language_model=language_model)
    full_model = SimpleNamespace(parent=parent)
    calibrate_loop = Mock(name="calibrate_loop")

    monkeypatch.setattr(hf_ptq, "is_quantized", Mock(return_value=False))
    monkeypatch.setattr(hf_ptq, "need_calibration", Mock(return_value=True))
    monkeypatch.setattr(hf_ptq, "create_vlm_calibration_loop", Mock(return_value=calibrate_loop))
    monkeypatch.setattr(
        hf_ptq,
        "get_language_model_from_vl",
        Mock(side_effect=lambda model: [parent, parent.language_model]),
    )
    monkeypatch.setattr(
        hf_ptq.mtq, "quantize", Mock(side_effect=lambda model, *_args, **_kwargs: model)
    )

    hf_ptq.mono_quantize(
        args=args,
        quant_cfg=quant_cfg,
        full_model=full_model,
        language_model=language_model,
        model_type=None,
        calibration_only=False,
        calib_dataloader=Mock(name="calib_dataloader"),
        is_nemotron_vl_model=True,
    )

    return hf_ptq, full_model, language_model, calibrate_loop


def test_layerwise_vlm_image_calibration_quantizes_full_model(monkeypatch):
    quant_cfg = {"quant_cfg": [], "algorithm": {"layerwise": {"enable": True}}}

    hf_ptq, full_model, _, calibrate_loop = _run_vlm_image_quantize(monkeypatch, quant_cfg)

    hf_ptq.mtq.quantize.assert_called_once_with(full_model, quant_cfg, forward_loop=calibrate_loop)


def test_non_layerwise_vlm_image_calibration_quantizes_language_model(monkeypatch):
    quant_cfg = {"quant_cfg": [], "algorithm": "max"}

    hf_ptq, _, language_model, calibrate_loop = _run_vlm_image_quantize(monkeypatch, quant_cfg)

    hf_ptq.mtq.quantize.assert_called_once_with(
        language_model, quant_cfg, forward_loop=calibrate_loop
    )
