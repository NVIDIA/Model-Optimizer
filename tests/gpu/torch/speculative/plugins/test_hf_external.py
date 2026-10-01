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

"""GPU tests for the external (standalone) draft plugin.

These tests require a CUDA GPU. CPU-only tests are in tests/unit/.

The base lm_head is attached with ``object.__setattr__`` so it stays off the draft's
module tree and out of its optimiser and gradients. The cost of that is that
``model.cuda()`` does not move it, which no CPU test can observe: every tensor is on
the same device there. That is the class of bug these tests exist for.
"""

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama

import modelopt.torch.speculative as mtsp

SEQ_LEN = 16


def _external_draft(**overrides):
    model = get_tiny_llama()
    mtsp.convert(
        model,
        [("external", {"external_offline": True, "external_loss": "tvd_deploy", **overrides})],
    )
    return model


class TestExternalDraftDevicePlacement:
    """The base lm_head lives off the module tree, so it needs its own device handling."""

    def test_teacher_logits_with_base_head_left_on_cpu(self):
        """Attaching a CPU head then moving the draft to GPU must still train."""
        draft = _external_draft()
        base = get_tiny_llama()
        # Deliberately attach before .cuda(), the order main.py uses.
        draft.attach_base_lm_head(
            base.get_output_embeddings(), getattr(getattr(base, "model", None), "norm", None)
        )
        draft = draft.cuda()

        hidden = torch.randn(
            1, SEQ_LEN, draft.config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        logits = draft._teacher_logits(hidden, {})
        assert logits.device.type == "cuda"
        assert logits.shape == (1, SEQ_LEN, draft.config.vocab_size)

    def test_base_head_stays_out_of_the_draft_parameters(self):
        """Gradient isolation: the teacher must not appear in the draft's optimiser."""
        draft = _external_draft()
        base = get_tiny_llama()
        draft.attach_base_lm_head(base.get_output_embeddings())
        draft = draft.cuda()

        param_ids = {id(p) for p in draft.parameters()}
        assert all(id(p) not in param_ids for p in base.get_output_embeddings().parameters())
        assert not any("_base_lm_head" in k for k in draft.state_dict())


def test_dense_path_rejects_a_missing_teacher_head():
    """Without the head the dense objective cannot be formed; fail loudly, not silently."""
    draft = _external_draft().cuda()
    with pytest.raises(RuntimeError, match="Base lm_head is not attached"):
        draft._teacher_logits(
            torch.randn(1, SEQ_LEN, draft.config.hidden_size, device="cuda", dtype=torch.bfloat16),
            {},
        )
