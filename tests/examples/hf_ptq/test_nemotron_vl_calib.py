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


# --- sharded InternVL path -----------------------------------------------------------------
#
# The omni tests above exit through the fallback and never reach the InternVL branch, where two
# bool masks index tensors that a sharded ``device_map`` can place on other GPUs. This group
# runs CPU-only, so the topology is simulated rather than allocated -- the convention
# ``test_example_utils.py`` already uses for ``torch.cuda.device_count``.


def _device_of(tensor):
    return getattr(tensor, "_fake_device", torch.device("cpu"))


def _shard(tensor, device):
    """Tag a CPU tensor as living on ``device``."""
    out = tensor.as_subclass(_ShardedTensor)
    out._fake_device = torch.device(device)
    return out


class _ShardedTensor(torch.Tensor):
    """CPU-backed tensor that reports a fake device and enforces CUDA's indexing rule.

    Storage stays on CPU; only the reported device, ``.to(device)`` and the cross-device
    indexing error are faked. CUDA raises when a bool mask lives on another device, and that
    is the failure this reproduces without needing two GPUs.
    """

    _fake_device = torch.device("cpu")

    @property
    def device(self):
        return self._fake_device

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.Tensor.__getitem__, torch.Tensor.__setitem__):
            target, index = args[0], args[1]
            if isinstance(index, torch.Tensor) and _device_of(index) != _device_of(target):
                raise RuntimeError(
                    "indices should be either on cpu or on the same device as the indexed "
                    f"tensor (got {_device_of(index)} vs {_device_of(target)})"
                )
        if func is torch.Tensor.to:
            target = kwargs.get("device", args[1] if len(args) > 1 else None)
            if isinstance(target, (str, torch.device)):
                return _shard(args[0].clone(), target)
        out = super().__torch_function__(func, types, args, kwargs)
        device = next((a._fake_device for a in args if isinstance(a, _ShardedTensor)), None)
        if device is not None:
            for tensor in out if isinstance(out, (list, tuple)) else [out]:
                if isinstance(tensor, _ShardedTensor):
                    tensor._fake_device = device
        return out


class _Embedding:
    """Callable embedding table that answers on ``device``, as ``.weight.device`` does."""

    def __init__(self, device, hidden):
        self.weight = _shard(torch.zeros(16, hidden), device)
        self._device = device
        self._hidden = hidden

    def __call__(self, input_ids):
        b, n = input_ids.shape
        return _shard(torch.zeros(b, n, self._hidden), self._device)


class _LanguageModel(torch.nn.Module):
    def __init__(self, device, hidden):
        super().__init__()
        self.config = SimpleNamespace(torch_dtype=torch.float32)
        self.calls: list[dict] = []
        self._embedding = _Embedding(device, hidden)

    def get_input_embeddings(self):
        return self._embedding

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return (torch.zeros(1),)


class _ShardedInternVL(torch.nn.Module):
    """InternVL-style wrapper with vision tower, embedding table and batch on three devices."""

    def __init__(self, vision_device, embed_device, tokens_per_image=1, hidden=4):
        super().__init__()
        self.config = SimpleNamespace(
            architectures=["NemotronVLForConditionalGeneration"], is_encoder_decoder=False
        )
        self.vision_model = SimpleNamespace(config=SimpleNamespace(torch_dtype=torch.float32))
        self.language_model = _LanguageModel(embed_device, hidden)
        self.img_context_token_id = 7
        self._vision_device = vision_device
        self._tokens_per_image = tokens_per_image
        self._hidden = hidden

    def extract_feature(self, pixel_values):
        return _shard(
            torch.ones(pixel_values.shape[0], self._tokens_per_image, self._hidden),
            self._vision_device,
        )


def _sharded_batch(device):
    """Two images and four tokens, of which two are image-context tokens.

    ``image_flags`` is supplied rather than left to the helper's synthesis step: that step
    allocates on ``pixel_values.device``, which under this harness is a *fake* device name, so
    it would escape the bookkeeping. A processor that emits ``image_flags`` puts it on the
    batch device anyway, which is exactly the placement the synthesis step reproduces.
    """
    return {
        "input_ids": _shard(torch.tensor([[7, 7, 1, 1]], dtype=torch.long), device),
        "pixel_values": _shard(torch.zeros(2, 3, 8, 8), device),
        "attention_mask": _shard(torch.ones(1, 4, dtype=torch.long), device),
        "image_flags": _shard(torch.ones(2, 1, dtype=torch.long), device),
        "position_ids": None,
    }


def test_sharded_internvl_aligns_both_index_masks():
    """``image_flags_s`` must follow the vision output and ``selected`` the embedding table.

    Pre-fix this raised on ``vit_embeds[image_flags_s == 1]``, before any alignment ran.
    """
    model = _ShardedInternVL(vision_device="cuda:1", embed_device="cuda:2")

    nemotron_vl_calib.safe_nemotron_vl_forward(model, _sharded_batch("cuda:0"))

    assert len(model.language_model.calls) == 1, "the LLM forward drives the activation stats"
    passed = model.language_model.calls[0]
    assert passed["inputs_embeds"].device == torch.device("cuda:2")
    assert passed["attention_mask"].device == torch.device("cuda:2")
    assert passed["use_cache"] is False


def test_sharded_internvl_retry_path_also_uses_aligned_masks():
    """The ``except`` branch re-indexes with ``selected``, so it needs the aligned mask too.

    Two images of two tokens each give four vision rows for two selected positions, so the
    first assignment raises on shape and the retry runs -- under the same sharding.
    """
    model = _ShardedInternVL(vision_device="cuda:1", embed_device="cuda:2", tokens_per_image=2)

    nemotron_vl_calib.safe_nemotron_vl_forward(model, _sharded_batch("cuda:0"))

    assert len(model.language_model.calls) == 1, "the retry still reaches the LLM forward"


def test_unsharded_internvl_is_unaffected():
    """Everything on one device: no mask is moved and the merge still happens."""
    model = _ShardedInternVL(vision_device="cuda:0", embed_device="cuda:0")

    nemotron_vl_calib.safe_nemotron_vl_forward(model, _sharded_batch("cuda:0"))

    assert len(model.language_model.calls) == 1
    assert model.language_model.calls[0]["inputs_embeds"].device == torch.device("cuda:0")


def test_sharded_internvl_aligns_the_token_mask_when_only_the_embedding_moves():
    """Isolates the second mask, which the first test cannot reach.

    When the vision tower shares the batch device, ``vit_embeds[image_flags_s == 1]`` succeeds
    and execution gets as far as ``flat_embeds[selected]``, where ``selected`` -- built from
    ``input_ids`` -- meets ``flat_embeds`` on the embedding device. Pre-fix that raises, and the
    ``except`` branch retries with the same mask and raises again.
    """
    model = _ShardedInternVL(vision_device="cuda:0", embed_device="cuda:2")

    nemotron_vl_calib.safe_nemotron_vl_forward(model, _sharded_batch("cuda:0"))

    assert len(model.language_model.calls) == 1
    assert model.language_model.calls[0]["inputs_embeds"].device == torch.device("cuda:2")
