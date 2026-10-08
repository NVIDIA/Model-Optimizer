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

"""The Nemotron VL calibration fallback for wrappers without ``extract_feature``."""

import importlib
from pathlib import Path

import pytest
import torch

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"


@pytest.fixture
def calib(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("nemotron_vl_calib")


class _OmniWrapper(torch.nn.Module):
    """An omni-style wrapper: merges vision inside forward, so no ``extract_feature``.

    Deliberately also has no ``img_context_token_id`` and no ``language_model`` -- touching
    either before the fallback fires is the regression this guards.
    """

    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(1)


def _batch():
    return {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "pixel_values": torch.zeros(2, 3, 8, 8),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "position_ids": None,
    }


def test_fallback_forwards_batch_when_extract_feature_is_absent(calib):
    model = _OmniWrapper()
    calib.safe_nemotron_vl_forward(model, _batch())

    assert len(model.calls) == 1, "the wrapper's own forward should be called exactly once"
    passed = model.calls[0]
    # every non-None batch key reaches the model
    assert passed["input_ids"].shape == (1, 4)
    assert passed["pixel_values"].shape == (2, 3, 8, 8)
    assert passed["attention_mask"].shape == (1, 4)
    # None-valued keys are dropped, not forwarded
    assert "position_ids" not in passed
    # calibration must not build a KV cache
    assert passed["use_cache"] is False
    # a synthesized image_flags must NOT be injected: it is built on pixel_values.device, and a
    # sharded wrapper indexes its vision output with it from another device (seen in a real
    # 4-GPU run as "indices should be either on cpu or on the same device as the indexed tensor")
    assert "image_flags" not in passed


def test_fallback_forwards_image_flags_the_batch_already_carried(calib):
    """A batch-supplied image_flags is device-consistent with the rest, so it passes through."""
    model = _OmniWrapper()
    batch = _batch()
    batch["image_flags"] = torch.ones(2, 1, dtype=torch.long)
    calib.safe_nemotron_vl_forward(model, batch)

    assert model.calls[0]["image_flags"].shape == (2, 1)


def test_fallback_drops_kwargs_a_narrow_forward_cannot_accept(calib):
    """A wrapper without ``**kwargs`` must not get a TypeError instead of the old AttributeError."""

    class _Narrow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[dict] = []

        def forward(self, input_ids=None, pixel_values=None):
            self.calls.append({"input_ids": input_ids, "pixel_values": pixel_values})
            return torch.zeros(1)

    model = _Narrow()
    calib.safe_nemotron_vl_forward(model, _batch())  # must not raise

    assert len(model.calls) == 1
    assert model.calls[0]["pixel_values"].shape == (2, 3, 8, 8)


def test_accepted_kwargs_passes_everything_to_a_var_keyword_callable(calib):
    def takes_anything(**kwargs):
        pass

    payload = {"a": 1, "b": 2}
    assert calib._accepted_kwargs(takes_anything, payload) == payload


def test_accepted_kwargs_filters_a_fixed_signature(calib):
    def takes_only_a(a=None):
        pass

    assert calib._accepted_kwargs(takes_only_a, {"a": 1, "b": 2}) == {"a": 1}
