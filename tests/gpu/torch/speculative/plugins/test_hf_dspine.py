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

"""GPU tests for the DSpine speculative decoding plugin. CPU-only tests are in tests/unit/."""

import pytest
import torch
from _test_utils.torch.speculative.dflash import get_dflash_config
from _test_utils.torch.transformers_models import get_tiny_llama
from accelerate import dispatch_model
from accelerate.hooks import AlignDevicesHook, add_hook_to_module

import modelopt.torch.speculative as mtsp


@pytest.mark.parametrize("offload", [False, True])
def test_generate_picks_the_best_head_row_wherever_the_head_is(offload):
    """``device_map="auto"``, as AR validation loads, can move lm_head or offload it to meta."""
    config = get_dflash_config(block_size=4, num_layers=2)
    config.update(dflash_draft_attention="causal", dflash_loss_objective="decay")
    config["dflash_architecture_config"].update(
        projector_type="dspine", code_size=8, code_rank=8, refine_top_k=4, teacher_forcing_layers=1
    )
    model = get_tiny_llama(num_hidden_layers=4)
    mtsp.convert(model, [("dflash", config)])
    model = model.float().to("cuda").eval()
    draft = model.dflash_module
    torch.manual_seed(0)
    # From its identity start, refinement would only repeat the backbone's top candidate.
    for weight in draft.selected_prefix_bridge.parameters():
        torch.nn.init.normal_(weight, std=0.5)
    rows = model.lm_head.weight.detach().clone()

    if offload:
        dispatch_model(model, {"model": 0, "dflash_module": 0, "lm_head": "cpu"})
        assert model.lm_head.weight.device.type == "meta"
    else:
        # As accelerate's dispatch hooks a module placed on another device.
        add_hook_to_module(model.lm_head.to("cpu"), AlignDevicesHook(execution_device="cpu"))

    draft_outputs, refine_calls = [], []
    draft.register_forward_hook(lambda _module, _args, output: draft_outputs.append(output))
    refine = draft.refine

    def spy_refine(state, prev_token):
        refine_calls.append((state, prev_token, refine(state, prev_token)))
        return refine_calls[-1][-1]

    draft.refine = spy_refine
    input_ids = torch.randint(1, 32, (8, 4), device="cuda")
    base_token, draft_tokens = model.pseudo_speculative_generate(input_ids, steps=3)

    assert draft_tokens.shape == (8, 3)
    assert base_token.device == draft_tokens.device == input_ids.device
    assert len(refine_calls) == draft_tokens.shape[1]
    out = draft_outputs[0]
    backbone_logits = out.hidden_states @ rows.T
    expected_prev = base_token.squeeze(-1)
    for t, (state, prev_token, refined) in enumerate(refine_calls, start=1):
        assert torch.equal(state, out.pre_injection[:, t])
        assert torch.equal(prev_token, expected_prev)
        candidates = backbone_logits[:, t].topk(draft.head_topk, dim=-1).indices
        scores = torch.einsum("bh,bkh->bk", refined, rows[candidates])
        expected_prev = candidates.gather(-1, scores.argmax(dim=-1, keepdim=True)).squeeze(-1)
        assert torch.equal(draft_tokens[:, t - 1], expected_prev)
    assert (draft_tokens != backbone_logits[:, 1:].argmax(dim=-1)).any()
