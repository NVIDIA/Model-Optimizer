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

import torch

from examples.onnx_ptq.petr.petr_utils import run_backbone


class FakeBackbone:
    def __init__(self, name, events, outputs):
        self.name = name
        self.events = events
        self.outputs = outputs

    def __call__(self, stream, **inputs):
        self.events.append((self.name, inputs))
        return self.outputs


def test_run_backbone_v1_uses_one_context():
    events = []
    outputs = {"out.0": torch.ones(1)}
    backbone = FakeBackbone("current", events, outputs)
    images = torch.arange(6).reshape(1, 6, 1)

    result = run_backbone("v1", backbone, None, None, images)

    assert result is outputs
    assert [name for name, _ in events] == ["current"]
    assert set(events[0][1]) == {"img"}
    torch.testing.assert_close(events[0][1]["img"], images.squeeze(0))


def test_run_backbone_v2_uses_history_features_from_matching_context():
    events = []
    history_outputs = {
        "out.0": torch.arange(12).reshape(1, 12, 1),
        "out.1": torch.arange(100, 112).reshape(1, 12, 1),
    }
    current_outputs = {
        "out.0": torch.full((1, 12, 1), 200),
        "out.1": torch.full((1, 12, 1), 300),
    }
    history_backbone = FakeBackbone("history", events, history_outputs)
    backbone = FakeBackbone("current", events, current_outputs)
    images = torch.arange(12).reshape(1, 12, 1)

    result = run_backbone("v2", backbone, history_backbone, None, images)

    assert result is current_outputs
    assert [name for name, _ in events] == ["history", "current"]
    assert set(events[0][1]) == {"img"}
    torch.testing.assert_close(events[0][1]["img"], images[:, 6:12].squeeze(0))
    torch.testing.assert_close(events[1][1]["img"], images[:, :6].squeeze(0))
    for index, name in enumerate(("out.0", "out.1")):
        previous = events[1][1][f"prev.{index}"]
        torch.testing.assert_close(previous, history_outputs[name][:, :6])
        assert previous.is_contiguous()
