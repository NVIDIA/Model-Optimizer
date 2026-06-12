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

"""Negative-path loading tests for Qwen-Image in the diffusers quantization example.

These cover the AC-1 negative criteria without a GPU or a real model:
- selecting Qwen-Image when diffusers lacks the Qwen classes raises a clear,
  actionable error (not an opaque failure);
- Qwen loading does not pass ``trust_remote_code``.
"""

import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("diffusers")
pytest.importorskip("torch")

_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "diffusers" / "quantization"
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

import models_utils  # noqa: E402
import pipeline_manager  # noqa: E402
from models_utils import ModelType  # noqa: E402
from quantize_config import ModelConfig  # noqa: E402


def _qwen_pipeline_manager() -> "pipeline_manager.PipelineManager":
    config = ModelConfig(model_type=ModelType.QWEN_IMAGE, backbone=["transformer"])
    return pipeline_manager.PipelineManager(config, logging.getLogger("qwen-loading-test"))


def test_missing_qwen_pipeline_raises_actionable_error(monkeypatch):
    # Simulate a diffusers version without QwenImagePipeline.
    monkeypatch.setitem(models_utils.MODEL_PIPELINE, ModelType.QWEN_IMAGE, None)
    manager = _qwen_pipeline_manager()
    with pytest.raises(ImportError, match="Qwen-Image requires"):
        manager.create_pipeline()


def test_qwen_loading_does_not_pass_trust_remote_code(monkeypatch):
    captured_kwargs: dict = {}

    class _FakeQwenPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            captured_kwargs.update(kwargs)
            return cls()

        def set_progress_bar_config(self, **kwargs):
            pass

    monkeypatch.setitem(models_utils.MODEL_PIPELINE, ModelType.QWEN_IMAGE, _FakeQwenPipeline)
    manager = _qwen_pipeline_manager()
    manager.create_pipeline()

    # Qwen-Image must load without trust_remote_code.
    assert captured_kwargs.get("trust_remote_code") is not True
    assert "trust_remote_code" not in captured_kwargs
