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
"""Which GPU architectures a JIT extension is built for, without a GPU."""

import os

from modelopt.torch.utils import cpp_extension


def test_builds_for_every_visible_gpu(monkeypatch):
    """A build triggered from one GPU also runs on the others, whatever GPU is current."""
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "")  # restored after the test
    monkeypatch.setattr(cpp_extension.torch.version, "cuda", "13.0")
    monkeypatch.setattr(cpp_extension.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpp_extension.torch.cuda, "device_count", lambda: 3)
    capabilities = [(10, 0), (8, 0), (10, 0)]
    monkeypatch.setattr(
        cpp_extension.torch.cuda,
        "get_device_capability",
        lambda device=None: capabilities[1 if device is None else device],  # cuda:1 is current
    )
    built_for = []

    def load(name, sources, **kwargs):
        built_for.append(os.environ["TORCH_CUDA_ARCH_LIST"])
        return object()

    monkeypatch.setattr(cpp_extension, "load", load)

    assert cpp_extension.load_cpp_extension("ext", [], None) is not None
    assert built_for == ["8.0;10.0"]
