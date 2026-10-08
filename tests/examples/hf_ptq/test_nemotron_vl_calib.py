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

"""The Nemotron VL calibration fallback for wrappers without ``extract_feature``.

Two layers are covered: the helper itself, and the dispatch in
``create_vlm_calibration_loop()`` that decides which wrappers reach it. The dispatch half is
the one that matters in production -- a wrapper can only benefit from the fallback if the
calibration loop routes it there.
"""

from types import SimpleNamespace

import torch
from _test_utils.examples.hf_ptq_example_utils import example_utils, nemotron_vl_calib


class _OmniWrapper(torch.nn.Module):
    """An omni-style wrapper: merges vision inside forward, so no ``extract_feature``.

    Deliberately also has no ``img_context_token_id`` and no ``language_model`` -- touching
    either before the fallback fires is the regression this guards.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.calls: list[dict] = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(1)


def _omni_config():
    """A config that ``is_nemotron_vl()`` recognises: multimodal, Nemotron architecture."""
    return SimpleNamespace(
        architectures=["NemotronH_Omni_ForConditionalGeneration"],
        vision_config=SimpleNamespace(torch_dtype=torch.bfloat16),
        is_encoder_decoder=False,
    )


def _batch():
    return {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "pixel_values": torch.zeros(2, 3, 8, 8),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "position_ids": None,
    }


def test_fallback_forwards_batch_when_extract_feature_is_absent():
    model = _OmniWrapper()
    nemotron_vl_calib.safe_nemotron_vl_forward(model, _batch())

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


def test_fallback_forwards_image_flags_the_batch_already_carried():
    """A batch-supplied image_flags is device-consistent with the rest, so it passes through."""
    model = _OmniWrapper()
    batch = _batch()
    batch["image_flags"] = torch.ones(2, 1, dtype=torch.long)
    nemotron_vl_calib.safe_nemotron_vl_forward(model, batch)

    assert model.calls[0]["image_flags"].shape == (2, 1)


def test_fallback_drops_kwargs_a_narrow_forward_cannot_accept():
    """A wrapper without ``**kwargs`` must not get a TypeError instead of the old AttributeError."""

    class _Narrow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[dict] = []

        def forward(self, input_ids=None, pixel_values=None):
            self.calls.append({"input_ids": input_ids, "pixel_values": pixel_values})
            return torch.zeros(1)

    model = _Narrow()
    nemotron_vl_calib.safe_nemotron_vl_forward(model, _batch())  # must not raise

    assert len(model.calls) == 1
    assert model.calls[0]["pixel_values"].shape == (2, 3, 8, 8)


def test_accepted_kwargs_passes_everything_to_a_var_keyword_callable():
    def takes_anything(**kwargs):
        pass

    payload = {"a": 1, "b": 2}
    assert nemotron_vl_calib._accepted_kwargs(takes_anything, payload) == payload


def test_accepted_kwargs_filters_a_fixed_signature():
    def takes_only_a(a=None):
        pass

    assert nemotron_vl_calib._accepted_kwargs(takes_only_a, {"a": 1, "b": 2}) == {"a": 1}


def test_calibration_loop_routes_an_omni_wrapper_through_the_fallback():
    """The production dispatch, not the helper directly.

    An omni wrapper has no ``img_context_token_id``, so the loop used to call its forward with
    the raw batch -- bypassing the dtype cast and the ``use_cache=False`` the fallback supplies.
    """
    model = _OmniWrapper(config=_omni_config())
    model.vision_model = torch.nn.Module()
    model.vision_model.config = SimpleNamespace(torch_dtype=torch.bfloat16)

    example_utils.create_vlm_calibration_loop(model, [_batch()])(model)

    assert len(model.calls) == 1
    passed = model.calls[0]
    assert passed["use_cache"] is False
    assert passed["pixel_values"].dtype is torch.bfloat16, "the fallback carries the dtype cast"
    assert "position_ids" not in passed


def test_calibration_loop_keeps_the_plain_forward_for_encoder_decoder_vlms():
    """Nemotron-Parse is a Nemotron VL model too, but must not be rerouted.

    Its batch is renamed to ``decoder_input_ids``; the helper looks for ``input_ids`` and would
    return without a forward pass, silently skipping calibration.
    """
    model = _OmniWrapper(
        config=SimpleNamespace(
            architectures=["NemotronParseForConditionalGeneration"],
            is_encoder_decoder=True,
        )
    )

    example_utils.create_vlm_calibration_loop(model, [_batch()])(model)

    assert len(model.calls) == 1
    passed = model.calls[0]
    assert "decoder_input_ids" in passed, "the encoder-decoder rename reached the model"
    assert "input_ids" not in passed
    # the helper's fallback was not applied
    assert passed["pixel_values"].dtype is torch.float32
