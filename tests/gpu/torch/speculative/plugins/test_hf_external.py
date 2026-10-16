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
TOPK = 8


def _external_draft(**overrides):
    model = get_tiny_llama()
    mtsp.convert(
        model,
        [("external", {"external_loss": "tvd_deploy", **overrides})],
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


class TestExternalDraftSparseGPU:
    """What the sparse path needs a GPU to prove: that its dedup is order-stable there.

    The forward/backward and the no-teacher-head checks are device-independent and are
    covered on CPU in tests/unit; repeating them here buys nothing. The dedup below is
    different: it relies on ``sort(..., stable=True)`` keeping the first occurrence of a
    repeated id, and unstable tie order is genuinely unspecified on CUDA.
    """

    def test_ragged_row_dedup_is_stable_on_cuda(self):
        """A padded row repeating a real token must be charged once, not once per slot.

        The builder pads ragged rows by repeating the trailing id, so the same token can
        occupy several slots with the teacher's probability stored only in the first.
        Keeping any other occurrence pairs the teacher's mass with the wrong q, which no
        loss-magnitude check would catch -- so compare against the deduplicated reference.
        """
        draft = _external_draft(external_loss="tvd", external_report_acc=False).cuda()
        vocab = draft.config.vocab_size
        torch.manual_seed(0)
        logits = torch.randn(1, 3, vocab, device="cuda")
        # row = [7, 11, 11, 11, ...]: one real entry then pads repeating id 11
        tok = torch.full((1, 3, TOPK), 11, dtype=torch.long, device="cuda")
        tok[..., 0] = 7
        prob = torch.zeros(1, 3, TOPK, device="cuda")
        prob[..., 0] = 0.6
        prob[..., 1] = 0.4  # the first 11 carries the mass; later slots are pads

        loss, _ = draft.compute_sparse_loss(logits, tok, prob, torch.ones(1, 3, device="cuda"))

        q_all = torch.softmax(logits[:, :-1].float(), dim=-1)
        p = prob[:, 1:] / prob[:, 1:].sum(-1, keepdim=True)
        q = q_all.gather(-1, tok[:, 1:])
        keep = torch.zeros_like(q, dtype=torch.bool)
        keep[..., :2] = True  # ids 7 and 11 once each; the rest are duplicates
        q = q * keep
        expected = 0.5 * ((p - q).abs().sum(-1) + (1.0 - q.sum(-1)).clamp_min(0.0))
        assert torch.allclose(loss, expected.mean(), atol=1e-5), (
            f"dedup kept the wrong occurrence on CUDA: {loss} vs {expected.mean()}"
        )


def test_dense_path_rejects_a_missing_teacher_head():
    """Without the head the dense objective cannot be formed; fail loudly, not silently."""
    draft = _external_draft().cuda()
    with pytest.raises(RuntimeError, match="Base lm_head is not attached"):
        draft._teacher_logits(
            torch.randn(1, SEQ_LEN, draft.config.hidden_size, device="cuda", dtype=torch.bfloat16),
            {},
        )
