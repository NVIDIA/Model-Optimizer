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

import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_hf import _process_quantized_modules
from modelopt.torch.quantization import tensor_quant
from modelopt.torch.quantization import utils as quant_utils
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.conversion import set_quantizer_attributes_partial
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.nn.modules.quant_embedding import _UnsettableInputQuantizer
from modelopt.torch.quantization.utils import quantizer_attr_names

VOCAB_SIZE = 16
EMBED_DIM = 32  # multiple of the NVFP4 block size (16) so export tests can pack


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


def _embedding_nvfp4_cfg() -> dict:
    """Stock-NVFP4-style cfg that opts the embedding's weight quantizer in."""
    nvfp4 = {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    }
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "parent_class": "nn.Embedding",
                "quantizer_name": "*weight_quantizer",
                "cfg": dict(nvfp4),
            },
        ],
        "algorithm": "max",
    }


class _EmbeddingOnly(nn.Module):
    """Single-embedding wrapper exposing forward + named_modules iteration."""

    def __init__(self):
        """Build the lone embedding submodule."""
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

    def forward(self, ids):
        """Look up embeddings for the given token IDs."""
        return self.embedding(ids)


class _TiedEmbeddingLM(nn.Module):
    """Embedding + Linear lm_head with tied weights (lm_head.weight is embedding.weight)."""

    def __init__(self):
        """Build embedding + lm_head and tie their weight Parameters."""
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.embedding.weight  # Python-level tie

    def forward(self, ids):
        """Embed then project to vocab logits with the tied weight."""
        return self.lm_head(self.embedding(ids))


class TestQuantEmbeddingExport:
    """Export-path coverage: weight packing and tied-weight guard."""

    def test_quantized_weight_is_packed_and_scales_registered(self):
        """End-to-end: _process_quantized_modules packs the embedding weight and
        registers ``weight_scale`` + ``weight_scale_2`` buffers."""
        model = _EmbeddingOnly()
        model = mtq.quantize(
            model, _embedding_nvfp4_cfg(), lambda m: m(torch.randint(0, VOCAB_SIZE, (2, 4)))
        )
        _process_quantized_modules(model, dtype=torch.float16)

        attrs = quantizer_attr_names("weight")
        assert model.embedding.weight.dtype == torch.uint8
        assert hasattr(model.embedding, attrs.weight_scale)
        assert hasattr(model.embedding, attrs.weight_scale_2)
        # input_scale is not registered (input_quantizer is permanently disabled).
        assert not hasattr(model.embedding, attrs.input_scale)

    def test_tied_embedding_export_skips_packing(self):
        """When the embedding weight is shared with lm_head, packing is skipped
        with a warning so the tie survives the export."""
        model = _TiedEmbeddingLM()
        assert model.lm_head.weight is model.embedding.weight  # sanity

        model = mtq.quantize(
            model, _embedding_nvfp4_cfg(), lambda m: m(torch.randint(0, VOCAB_SIZE, (2, 4)))
        )
        orig_dtype = model.embedding.weight.dtype
        with pytest.warns(UserWarning, match="tied"):
            _process_quantized_modules(model, dtype=torch.float16)

        # Weight Parameter unchanged (not packed to uint8) and still tied.
        assert model.embedding.weight.dtype == orig_dtype
        assert model.lm_head.weight is model.embedding.weight
