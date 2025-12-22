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

from contextlib import contextmanager
from itertools import chain, repeat
from typing import Any
from unittest import mock

import torch
from numpy import ndarray

from modelopt.torch._deploy._runtime import DetailedResults, RuntimeClient, RuntimeRegistry


class FakeClient(RuntimeClient):
    def _profile(self, compiled_model: bytes) -> tuple[float, DetailedResults]:
        return 1234, {}

    def _inference(self, compiled_model: bytes, inputs: list[ndarray]) -> list[ndarray]:
        raise NotImplementedError

    def _ir_to_compiled(
        self, ir_bytes: bytes, compilation_args: dict[str, Any] | None = None
    ) -> bytes:
        return ir_bytes

    @property
    def default_deployment(self):
        return {k: v[0] for k, v in self.deployment_table.items()}

    @property
    def deployment_table(self):
        return {
            "runtime": [__file__ + ".fake"],
        }

    def _compile(self, model: torch.nn.Module) -> bytes:
        pass


FAKE_DEPLOYMENT = {"runtime": __file__ + ".fake"}


@contextmanager
def fake_latency(latencies):
    if isinstance(latencies, (int, float)):
        latencies = [latencies]

    latencies = ((x, {}) for x in chain(latencies, repeat(latencies[-1])))
    identifier = __file__ + ".fake"
    RuntimeRegistry.register(identifier)(FakeClient)

    try:
        with mock.patch.object(
            FakeClient,
            "_profile",
        ) as _mock_method:
            _mock_method.side_effect = latencies
            yield
    finally:
        RuntimeRegistry.unregister(identifier)
