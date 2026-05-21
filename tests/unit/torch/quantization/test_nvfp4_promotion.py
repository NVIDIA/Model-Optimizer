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


class _ModelWithQuantizers(nn.Module):
    def __init__(self, n_quantizers: int):
        super().__init__()
        self.quantizers = nn.ModuleList(
            [_make_nvfp4_static_quantizer() for _ in range(n_quantizers)]
        )


def test_promotes_quantizer_without_amax():
    """Per-expert quantizers that received no calibration tokens must still be promoted."""
    model = _ModelWithQuantizers(n_quantizers=3)
    for q in model.quantizers:
        assert q.is_nvfp4_static
        assert not hasattr(q, "_amax")

    converted = promote_nvfp4_static_quantizers(model)

    assert converted == 3
    for q in model.quantizers:
        assert isinstance(q, NVFP4StaticQuantizer)
        assert q.global_amax is None  # no amax means no global_amax computed


def test_promotes_quantizer_with_amax_sets_global_amax():
    """When _amax is set, promotion also seeds _global_amax."""
    model = _ModelWithQuantizers(n_quantizers=1)
    q = model.quantizers[0]
    q.register_buffer("_amax", torch.tensor([2.0, 4.0]))

    promote_nvfp4_static_quantizers(model)

    assert isinstance(q, NVFP4StaticQuantizer)
    assert q.global_amax is not None
    assert torch.allclose(q.global_amax, torch.tensor(4.0))


def test_idempotent_on_already_promoted():
    """Re-running promotion on a model that's already been promoted is a no-op."""
    model = _ModelWithQuantizers(n_quantizers=2)
    first = promote_nvfp4_static_quantizers(model)
    second = promote_nvfp4_static_quantizers(model)

    assert first == 2
    assert second == 0


def test_skips_disabled_and_dynamic_quantizers():
    """Disabled or dynamic quantizers must not be promoted even with matching format."""
    model = _ModelWithQuantizers(n_quantizers=2)
    model.quantizers[0]._disabled = True
    model.quantizers[1]._dynamic = True

    converted = promote_nvfp4_static_quantizers(model)

    assert converted == 0
    for q in model.quantizers:
        assert not isinstance(q, NVFP4StaticQuantizer)
