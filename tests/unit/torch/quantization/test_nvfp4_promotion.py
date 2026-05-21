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

"""Tests for promote_nvfp4_static_quantizers — class swap must not require _amax."""

import pytest
import torch
import torch.nn as nn

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer, TensorQuantizer
from modelopt.torch.quantization.utils.core_utils import promote_nvfp4_static_quantizers


def _make_nvfp4_static_quantizer() -> TensorQuantizer:
    cfg = QuantizerAttributeConfig(
        num_bits=(2, 1),
        block_sizes={-1: 16, "type": "static", "scale_bits": (4, 3)},
        axis=None,
        fake_quant=True,
    )
    return TensorQuantizer(cfg)


class _Model(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.quantizers = nn.ModuleList([_make_nvfp4_static_quantizer() for _ in range(n)])


@pytest.mark.parametrize(
    ("amax", "expected_global_amax"),
    [(None, None), (torch.tensor([2.0, 4.0]), 4.0)],
    ids=["no_amax", "with_amax"],
)
def test_promotes_nvfp4_static(amax, expected_global_amax):
    """Quantizers in NVFP4-static format must be promoted regardless of _amax.

    Per-expert quantizers in fused MoEs that received no calibration tokens
    (amax=None) must still become NVFP4StaticQuantizer so a later forward —
    once MSE populates _amax — dispatches via two-level scaling instead of
    the parent's generic E4M3 path.
    """
    model = _Model(n=2)
    if amax is not None:
        for q in model.quantizers:
            q.register_buffer("_amax", amax.clone())

    converted = promote_nvfp4_static_quantizers(model)

    assert converted == 2
    for q in model.quantizers:
        assert isinstance(q, NVFP4StaticQuantizer)
        if expected_global_amax is None:
            assert q.global_amax is None
        else:
            assert torch.allclose(q.global_amax, torch.tensor(expected_global_amax))


def test_skips_disabled_and_dynamic_quantizers():
    """Disabled or dynamic quantizers must not be promoted even with matching format."""
    model = _Model(n=2)
    model.quantizers[0]._disabled = True
    model.quantizers[1]._dynamic = True

    converted = promote_nvfp4_static_quantizers(model)

    assert converted == 0
    for q in model.quantizers:
        assert not isinstance(q, NVFP4StaticQuantizer)
