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

"""CPU smoke tests for the Triton flash attention module.

The ``@triton.jit`` kernels and the ``attention`` / ``attention_calibrate``
Python wrappers require a GPU and are fully exercised in
``tests/gpu/torch/sparsity/attention_sparsity/test_triton_fa*.py``.

This file only verifies that the module is importable on CPU-only CI runners,
so upstream code paths that conditionally import it don't break.
"""

import pytest


def test_triton_fa_importable_on_cpu():
    """Module imports cleanly without CUDA; exports the public API names."""
    try:
        import triton  # noqa: F401
    except ImportError:
        pytest.skip("triton is not installed")

    from modelopt.torch.kernels import triton_fa

    assert "attention" in triton_fa.__all__
    assert "attention_calibrate" in triton_fa.__all__
