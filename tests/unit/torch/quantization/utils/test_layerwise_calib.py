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

import os
import pickle

import pytest
import torch

from modelopt.torch.quantization.utils.layerwise_calib import _CheckpointState


class _UnsupportedCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def test_resume_rejects_unsupported_pickle_even_with_unsafe_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    layer_dir = tmp_path / "layer_0000"
    layer_dir.mkdir()
    marker = tmp_path / "unpickled"
    torch.save(_UnsupportedCheckpoint(marker), layer_dir / "output_meta.pt")
    state = _CheckpointState(str(tmp_path), num_layers=2, start_layer=1)
    layers = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
    with pytest.raises(pickle.UnpicklingError):
        state.setup_resume(layers)
    assert not marker.exists()
