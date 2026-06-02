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

"""Export-path tests for QuantEmbedding (CUDA-only).

These tests drive ``_process_quantized_modules`` end-to-end. The export weight-packing
path bottoms out in ``torch.cuda.empty_cache()``, which raises on systems without an
NVIDIA driver, so the suite lives under ``tests/gpu/`` rather than ``tests/unit/``.
"""

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_hf import _process_quantized_modules
from modelopt.torch.quantization.utils import quantizer_attr_names

VOCAB_SIZE = 16
EMBED_DIM = 32  # multiple of the NVFP4 block size (16) so export packs cleanly


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
