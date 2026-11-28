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

"""SGLANGModel must pass the CUDA-graph batch size under the name the installed SGLang accepts."""

import dataclasses
import importlib
from types import SimpleNamespace

import pytest
from specdec_bench.models import SGLANGModel


@dataclasses.dataclass
class _LegacyServerArgs:
    """SGLang <= 0.5.13: a dataclass with a single cuda_graph_max_bs."""

    cuda_graph_max_bs: int | None = None


class _SplitServerArgs:
    """SGLang >= 0.5.14 (not a dataclass): per-phase cuda_graph_max_bs_decode."""

    def __init__(self, *, cuda_graph_max_bs_decode=None):
        pass


@pytest.mark.parametrize(
    ("version", "server_args", "accepted", "rejected"),
    [
        ("0.5.13", _LegacyServerArgs, "cuda_graph_max_bs", "cuda_graph_max_bs_decode"),
        ("0.5.23", _SplitServerArgs, "cuda_graph_max_bs_decode", "cuda_graph_max_bs"),
        # Source and dev-image builds report 0.0.0.devN.
        (
            "0.0.0.dev1+g797af7170",
            _SplitServerArgs,
            "cuda_graph_max_bs_decode",
            "cuda_graph_max_bs",
        ),
    ],
)
def test_engine_gets_cuda_graph_max_bs(monkeypatch, version, server_args, accepted, rejected):
    engine_calls = []
    fake_sgl = SimpleNamespace(
        __version__=version,
        ServerArgs=server_args,
        Engine=lambda **kwargs: engine_calls.append(kwargs),
    )
    module = importlib.import_module(SGLANGModel.__module__)
    monkeypatch.setattr(module, "sgl", fake_sgl, raising=False)

    SGLANGModel("model", max_concurrent_requests=32, sampling_kwargs={})

    (kwargs,) = engine_calls
    assert kwargs[accepted] == 32
    assert rejected not in kwargs
