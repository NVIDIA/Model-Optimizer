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
import os
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).parents[3] / "examples/windows/accuracy_benchmark/fvd_metrics/i3d_model.py"
_SPEC = importlib.util.spec_from_file_location("sonar_i3d_model", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
i3d_model = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(i3d_model)
InceptionI3d = i3d_model.InceptionI3d
load_i3d = i3d_model.load_i3d


class _UnsupportedCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


@pytest.mark.parametrize("trusted_archive", [False, True])
def test_i3d_rejects_pickle_objects_without_executing_them(tmp_path, trusted_archive):
    marker = tmp_path / "executed"
    path = tmp_path / "unsupported.pt"
    torch.save(_UnsupportedCheckpoint(marker), path)
    with pytest.raises(RuntimeError, match="Checkpoint"):
        load_i3d(str(path), torch.device("cpu"), allow_unsafe_pickle=trusted_archive)
    assert not marker.exists()


@pytest.mark.parametrize("torchscript", [False, True])
def test_i3d_loads_supported_formats(tmp_path, torchscript):
    model = InceptionI3d().eval()
    path = tmp_path / "weights.pt"
    if torchscript:
        torch.jit.save(torch.jit.script(model), str(path))
        with pytest.raises(RuntimeError, match="restricted state dict"):
            load_i3d(str(path), torch.device("cpu"))
    else:
        torch.save(model.state_dict(), path)
    restored = load_i3d(str(path), torch.device("cpu"), allow_unsafe_pickle=torchscript)
    assert not restored.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
