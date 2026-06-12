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

import sys
import types

from modelopt.onnx.quantization import ort_utils
from modelopt.onnx.quantization.ort_utils import create_input_shapes_profile


def _raise_trt_unavailable():
    raise RuntimeError("trt unavailable")


def test_create_input_shapes_profile_forwards_trust_remote_code(monkeypatch):
    calls = []

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return types.SimpleNamespace(
                hidden_size=128,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_hidden_layers=1,
            )

    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(AutoConfig=FakeAutoConfig)
    )

    default_profiles = create_input_shapes_profile("local-config.json", ["NvTensorRtRtx", "cpu"])
    trusted_profiles = create_input_shapes_profile(
        "custom-model", ["NvTensorRtRtx"], trust_remote_code=True
    )

    assert calls == [
        ("local-config.json", {"trust_remote_code": False}),
        ("custom-model", {"trust_remote_code": True}),
    ]
    assert default_profiles[0]["nv_profile_min_shapes"].startswith("input_ids:1x1")
    assert default_profiles[1] == {}
    assert trusted_profiles[0]["nv_profile_opt_shapes"].startswith("input_ids:1x512")


def test_prepare_ep_list_keeps_profiles_aligned_when_ep_is_disabled(monkeypatch):
    monkeypatch.setattr(ort_utils, "_check_for_tensorrt", _raise_trt_unavailable)
    monkeypatch.setattr(ort_utils, "_check_for_nv_tensorrt_rtx_libs", lambda: True)

    providers = ort_utils._prepare_ep_list(
        ["trt", "NvTensorRtRtx", "cpu"],
        [
            {"trt_profile_min_shapes": "trt_profile"},
            {"nv_profile_min_shapes": "rtx_profile"},
            {},
        ],
    )

    assert providers == [
        ("NvTensorRTRTXExecutionProvider", {"nv_profile_min_shapes": "rtx_profile"}),
        "CPUExecutionProvider",
    ]


def test_create_inference_session_filters_profile_with_disabled_ep(monkeypatch):
    captured_kwargs = {}

    class FakeInferenceSession:
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(ort_utils, "_check_for_tensorrt", _raise_trt_unavailable)
    monkeypatch.setattr(ort_utils.ort, "InferenceSession", FakeInferenceSession)

    ort_utils.create_inference_session(
        "model.onnx",
        ["trt", "cpu"],
        [{"trt_profile_min_shapes": "trt_profile"}, {}],
    )

    assert captured_kwargs["providers"] == ["CPUExecutionProvider"]
    assert "provider_options" not in captured_kwargs
