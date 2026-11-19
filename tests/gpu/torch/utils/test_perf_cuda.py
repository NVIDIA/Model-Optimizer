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
"""Clearing the CUDA cache while another GPU still has work queued."""

import os
import subprocess
import sys
import textwrap

# Kernels queued on cuda:1 read a tensor the host has freed when the cache is cleared from cuda:0.
_CLEAR_WHILE_ANOTHER_GPU_WORKS = textwrap.dedent(
    """
    import torch

    from modelopt.torch.utils import clear_cuda_cache

    torch.cuda.set_device(0)
    for _ in range(50):
        a = torch.randn(4096, 4096, device="cuda:1")
        b = torch.empty_like(a)
        for _ in range(20):
            torch.mm(a, a, out=b)
        del a
        clear_cuda_cache()
    torch.cuda.synchronize(1)
    """
)


def test_clearing_the_cache_waits_for_every_gpu(need_2_gpus):
    # With expandable segments, torch.cuda.empty_cache() from cuda:0 unmapped the freed tensor
    # while cuda:1 still read it: an illegal memory access. The allocator reads the setting once,
    # when CUDA initializes, so the check runs in its own process.
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    subprocess.run(
        [sys.executable, "-c", _CLEAR_WHILE_ANOTHER_GPU_WORKS], env=env, check=True, timeout=300
    )
