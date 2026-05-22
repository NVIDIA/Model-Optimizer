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

"""Tests of QuantEmbedding module."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelopt.torch.quantization import tensor_quant
from modelopt.torch.quantization import utils as quant_utils
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.conversion import set_quantizer_attributes_partial
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.nn.modules.quant_embedding import _UnsettableInputQuantizer

VOCAB_SIZE = 16
EMBED_DIM = 32


def _make_quant_embedding(**kwargs) -> nn.Module:
    """Build an nn.Embedding and convert it through QuantModuleRegistry."""
    return QuantModuleRegistry.convert(nn.Embedding(VOCAB_SIZE, EMBED_DIM, **kwargs))


class TestQuantEmbedding:
    """Forward-path behavior of the registered QuantEmbedding wrapper."""

    def test_default_state_and_no_quant(self):
        """Default state: input quant locked-disabled, output quant disabled, weight quant on;
        with weight quant also off the wrapper matches plain F.embedding."""
        qemb = _make_quant_embedding()
        assert isinstance(qemb.input_quantizer, _UnsettableInputQuantizer)
        assert not qemb.input_quantizer.is_enabled
        assert not qemb.output_quantizer.is_enabled
        assert qemb.weight_quantizer.is_enabled

        qemb.weight_quantizer.disable()
        ids = torch.randint(0, VOCAB_SIZE, (4, 6))
        assert torch.allclose(qemb(ids), F.embedding(ids, qemb.weight), rtol=0, atol=0)

    @pytest.mark.parametrize("axis", [None, 0])
    def test_weight_fake_quant(self, axis):
        """Per-tensor (axis=None) and per-row (axis=0) weight fake quant match the manual ref."""
        qemb = _make_quant_embedding()
        set_quantizer_attributes_partial(
            qemb, "*weight_quantizer", QuantizerAttributeConfig(axis=axis).model_dump()
        )

        ids = torch.randint(0, VOCAB_SIZE, (4, 6))
        weight = qemb.weight.detach().clone()
        amax = (
            torch.max(torch.abs(weight))
            if axis is None
            else quant_utils.reduce_amax(weight, axis=1, keepdims=True)
        )
        ref = F.embedding(ids, tensor_quant.fake_tensor_quant(weight, amax))
        assert torch.allclose(qemb(ids), ref, rtol=0, atol=0)

    def test_output_quantizer_applied_when_enabled(self):
        """Enabling output_quantizer makes forward equivalent to applying it to the lookup."""
        qemb = _make_quant_embedding()
        qemb.weight_quantizer.disable()
        qemb.output_quantizer.enable()
        ids = torch.randint(0, VOCAB_SIZE, (4, 6))
        with torch.no_grad():
            qemb(ids)  # calibrate

        ref = qemb.output_quantizer(F.embedding(ids, qemb.weight))
        assert torch.allclose(qemb(ids), ref, rtol=0, atol=0)

    @pytest.mark.parametrize("method", ["enable", "enable_quant", "enable_calib"])
    def test_input_quantizer_mutators_raise(self, method):
        """Each public enable/enable_quant/enable_calib API on input_quantizer raises."""
        qemb = _make_quant_embedding()
        with pytest.raises(RuntimeError, match="nn.Embedding"):
            getattr(qemb.input_quantizer, method)()

    def test_wildcard_config_keeps_input_quantizer_disabled(self):
        """set_from_attribute_config absorbs any cfg but force-disables input_quantizer.

        Stock recipes' ``*input_quantizer`` wildcard (and the default ``QuantizeConfig``
        ``"*"`` rule) target every quantizer including the embedding's input slot.
        The quantizer must end up disabled regardless of what the cfg said.
        """
        qemb = _make_quant_embedding()
        set_quantizer_attributes_partial(
            qemb,
            "*input_quantizer",
            QuantizerAttributeConfig(num_bits=8, axis=None).model_dump(),
        )
        assert not qemb.input_quantizer.is_enabled
        # Forward still works — input_quantizer is disabled and never applied.
        qemb(torch.randint(0, VOCAB_SIZE, (4, 6)))


# Export-path tests for QuantEmbedding live in tests/gpu/torch/export/test_export_embedding.py
# because _export_quantized_weight bottoms out in torch.cuda.empty_cache(), which raises on
# CPU-only CI runners that have a CUDA-enabled torch build but no NVIDIA driver.
