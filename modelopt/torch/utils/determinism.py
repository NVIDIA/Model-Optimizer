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

"""Best-effort determinism for reproducible runs."""

import os
import random

import numpy as np
import torch

from .logging import print_rank_0

__all__ = ["enable_determinism"]


def enable_determinism(seed: int) -> None:
    """Seed every random number generator and ask PyTorch for deterministic kernels.

    An operation without a deterministic kernel only warns, so repeated runs on the same hardware,
    software and parallel layout agree as closely as the remaining kernels allow. Call this at
    startup, before the first CUDA matmul: cuBLAS reads ``CUBLAS_WORKSPACE_CONFIG`` when it creates
    its handle.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    print_rank_0(f"Deterministic mode: seeded every RNG with {seed}, deterministic torch kernels.")
