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

"""Calibration support for Whisper: speech features, not token ids."""

import torch
from transformers import AutoProcessor, WhisperProcessor

from modelopt.torch.utils.speech_dataset_utils import get_speech_dataset_dataloader

__all__ = [
    "calibration_sample_input",
    "load_calibration_processor",
    "make_calibration_dataloader",
]

# The encoder takes mel-spectrogram input features of length 3000.
_MEL_FRAMES = 3000


def load_calibration_processor(checkpoint_path, trust_remote_code):
    """Load the processor, left-padded and with a pad token, that turns audio into features."""
    processor = AutoProcessor.from_pretrained(
        checkpoint_path, padding_side="left", trust_remote_code=trust_remote_code
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    assert processor.tokenizer.pad_token is not None, (
        f"Pad token for {checkpoint_path} cannot be set!"
    )
    return processor


def make_calibration_dataloader(model, processor, dataset_names, calib_size, batch_size, device):
    """Build the speech calibration data: ``(dataloader, transcript of the first sample)``."""
    assert isinstance(processor, WhisperProcessor), "The AutoProcessor must be set."
    assert len(calib_size) == 1, (
        "whisper only supports one dataset for calibration, can extend this in the future"
    )
    return get_speech_dataset_dataloader(
        dataset_name=dataset_names[0] if dataset_names else "peoples_speech",
        processor=processor,
        batch_size=batch_size,
        num_samples=calib_size[0],
        device=device,
        dtype=model.dtype,
    )


def calibration_sample_input(model):
    """One sample of the shape the encoder takes, ``(1, num_mel_bins, 3000)``, for batch probing.

    The encoder has no embedding layer, so the input has to be a float tensor.
    """
    mel_bins = model.config.num_mel_bins
    return torch.ones([1, mel_bins, _MEL_FRAMES], dtype=model.dtype).to(model.device) * 100
