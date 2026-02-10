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

"""Tests for DMS chunked prefill."""

import pytest
import torch

from experimental.dms.tests.utils import add_dms_to_path

try:
    from dms.core import dms_chunked_prefill
except ImportError:
    add_dms_to_path()
    from dms.core import dms_chunked_prefill


class IdentityDecoderLayer(torch.nn.Module):
    """Decoder layer that returns hidden states unchanged."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: torch.Tensor,
        use_cache: bool,
        cache_position: torch.Tensor,
        position_embeddings: torch.Tensor,
        **kwargs: dict,
    ):
        """Return hidden states unchanged, matching the HF decoder layer interface.

        All arguments besides hidden_states are accepted but ignored, so that
        this layer can be used as a drop-in replacement in dms_chunked_prefill.
        """
        return hidden_states, None


@pytest.fixture
def decoder_layers():
    """Create a list of identity decoder layers."""
    return [IdentityDecoderLayer() for _ in range(2)]


class TestChunkedPrefill:
    """Tests for dms_chunked_prefill with identity decoder layers."""

    @pytest.mark.parametrize("seed", range(10))
    def test_chunked_matches_unchunked(self, decoder_layers, seed):
        """Chunked prefill output should match non-chunked output."""
        torch.manual_seed(seed)
        batch_size = torch.randint(1, 10, (1,)).item()
        seq_len = torch.randint(1, 100, (1,)).item()
        chunk_size = torch.randint(1, 10, (1,)).item()
        hidden_dim = torch.randint(1, 16, (1,)).item()

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

        output_no_chunking, _ = dms_chunked_prefill(
            decoder_layers=decoder_layers,
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=True,
            cache_position=None,
            position_embeddings=None,
            dms_manual_inference_mode=False,
            dms_chunked_prefill=None,
        )

        output_chunking, _ = dms_chunked_prefill(
            decoder_layers=decoder_layers,
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=True,
            cache_position=None,
            position_embeddings=None,
            dms_manual_inference_mode=False,
            dms_chunked_prefill=chunk_size,
        )

        torch.testing.assert_close(output_no_chunking, output_chunking)
