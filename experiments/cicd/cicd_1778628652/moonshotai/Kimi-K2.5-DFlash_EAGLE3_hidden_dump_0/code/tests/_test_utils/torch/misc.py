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

import random

import pytest
import torch

from modelopt.torch.utils import flatten_tree


def compare_outputs(out1, out2, rtol=1e-5, atol=1e-8):
    out1, _ = flatten_tree(out1)
    out2, _ = flatten_tree(out2)
    for i, (t1, t2) in enumerate(zip(out1, out2)):
        try:
            assert torch.allclose(t1.to(torch.float32), t2.to(torch.float32), rtol, atol)
        except AssertionError:  # noqa: PERF203
            diff = torch.abs(t1 - t2)
            print(f"\n{i=}")
            print(f"{t1=}")
            print(f"{t2=}")
            print(f"{diff=}")
            print(f"{diff.shape=}")
            print(f"{diff.min()=}")
            print(f"{diff.max()=}")
            print(f"{diff.mean()=}")
            raise


def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed_value)


def minimum_sm(sm):
    major, minor = sm // 10, sm % 10
    return pytest.mark.skipif(
        torch.cuda.get_device_capability() < (major, minor),
        reason=f"Requires sm{sm} or higher",
    )


def minimum_gpu(n):
    return pytest.mark.skipif(
        torch.cuda.device_count() < n,
        reason=f"Requires at least {n} GPUs",
    )
