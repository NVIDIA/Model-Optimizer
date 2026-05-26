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

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import tensor_quant
from modelopt.torch.quantization import utils as quant_utils
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.conversion import set_quantizer_attributes_partial
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.nn.modules.tensor_quantizer import HardDisabledTensorQuantizer

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
        assert isinstance(qemb.input_quantizer, HardDisabledTensorQuantizer)
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
        with pytest.raises(RuntimeError, match="hard-disabled"):
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


class _EmbeddingForwardModel(nn.Module):
    """Embedding + tiny Linear so the model has a normal float forward to assert against."""

    def __init__(self):
        """Build a single embedding and a small linear head."""
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, 4)

    def forward(self, ids):
        """Look up embeddings then project to a fixed 4-dim output."""
        return self.head(self.embedding(ids))


def _embedding_quant_cfg() -> dict:
    """Per-tensor weight quant opt-in for the embedding, every other quantizer disabled."""
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "parent_class": "nn.Embedding",
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": 8, "axis": None},
            },
        ],
        "algorithm": "max",
    }


class TestQuantEmbeddingSaveRestore:
    """Save → restore preserves the HardDisabledTensorQuantizer type and behavior."""

    def test_save_restore_preserves_hard_disabled_input_quantizer(self):
        """After modelopt save/restore the embedding's input_quantizer is still a
        HardDisabledTensorQuantizer with locked-mutator semantics, not a plain
        TensorQuantizer that happens to be `_disabled = True`."""
        ids = torch.randint(0, VOCAB_SIZE, (2, 4))

        model_quant = _EmbeddingForwardModel()
        mtq.quantize(model_quant, _embedding_quant_cfg(), lambda m: m(ids))

        # Sanity: the original module has the hard-disabled type.
        assert isinstance(model_quant.embedding.input_quantizer, HardDisabledTensorQuantizer)

        state_dict = mto.modelopt_state(model_quant)

        model_restored = _EmbeddingForwardModel()
        mto.restore_from_modelopt_state(model_restored, state_dict)
        model_restored.load_state_dict(model_quant.state_dict())

        # 1. Type is preserved.
        assert isinstance(model_restored.embedding.input_quantizer, HardDisabledTensorQuantizer), (
            "input_quantizer regressed to a non-HardDisabled type after restore — "
            "the hard-disable guarantees (enable() raise, set_from_attribute_config "
            "force-disable) would be lost."
        )

        # 2. Behavior is preserved: direct enable still raises after restore.
        for method in ("enable", "enable_quant", "enable_calib"):
            with pytest.raises(RuntimeError, match="hard-disabled"):
                getattr(model_restored.embedding.input_quantizer, method)()

        # 3. Forward output matches the original (weight quantization properties round-tripped).
        assert torch.allclose(model_quant(ids), model_restored(ids))
