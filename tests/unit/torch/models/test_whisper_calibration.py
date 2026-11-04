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

"""Whisper's calibration support: the processor, the speech data and the batch-probing sample."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
from transformers import WhisperProcessor

from modelopt.torch.models.whisper import calibration


def test_the_processor_is_left_padded_and_gets_a_pad_token(monkeypatch):
    tokenizer = SimpleNamespace(pad_token=None, eos_token="</s>")
    loaded = {}

    def from_pretrained(path, **kwargs):
        loaded.update(path=path, **kwargs)
        return SimpleNamespace(tokenizer=tokenizer)

    monkeypatch.setattr(calibration.AutoProcessor, "from_pretrained", from_pretrained)

    processor = calibration.load_calibration_processor("ckpt", True)

    assert loaded == {"path": "ckpt", "padding_side": "left", "trust_remote_code": True}
    assert processor.tokenizer.pad_token == "</s>"


@pytest.mark.parametrize(
    ("dataset_names", "expected"), [(None, "peoples_speech"), (["other"], "other")]
)
def test_the_dataloader_is_built_from_the_first_dataset(monkeypatch, dataset_names, expected):
    built = {}
    monkeypatch.setattr(
        calibration,
        "get_speech_dataset_dataloader",
        lambda **kwargs: built.update(kwargs) or ("loader", "transcript"),
    )
    model = SimpleNamespace(dtype=torch.float16)

    result = calibration.make_calibration_dataloader(
        model, object.__new__(WhisperProcessor), dataset_names, [8], 2, "cpu"
    )

    assert result == ("loader", "transcript")
    assert built["dataset_name"] == expected
    assert (built["num_samples"], built["batch_size"], built["dtype"]) == (8, 2, torch.float16)


def test_the_dataloader_refuses_what_it_cannot_serve():
    model = SimpleNamespace(dtype=torch.float32)
    with pytest.raises(AssertionError, match="AutoProcessor must be set"):
        calibration.make_calibration_dataloader(model, object(), None, [8], 2, "cpu")
    with pytest.raises(AssertionError, match="only supports one dataset"):
        calibration.make_calibration_dataloader(
            model, object.__new__(WhisperProcessor), None, [8, 8], 2, "cpu"
        )


def test_the_sample_input_is_one_float_mel_spectrogram():
    model = SimpleNamespace(
        config=SimpleNamespace(num_mel_bins=80), dtype=torch.float32, device=torch.device("cpu")
    )

    sample = calibration.calibration_sample_input(model)

    assert sample.shape == (1, 80, 3000)
    assert sample.dtype is torch.float32
    assert bool((sample == 100).all())
