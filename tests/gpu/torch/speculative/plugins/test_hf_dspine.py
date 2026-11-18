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

import torch
from _test_utils.torch.speculative.dflash import get_dflash_config
from _test_utils.torch.transformers_models import get_tiny_llama
from accelerate.hooks import AlignDevicesHook, add_hook_to_module

import modelopt.torch.speculative as mtsp


def test_generate_with_the_head_on_another_device():
    """``device_map="auto"``, as AR validation loads, can place lm_head off the draft's device."""
    config = get_dflash_config(block_size=4, num_layers=2)
    config.update(dflash_draft_attention="causal", dflash_loss_objective="decay")
    config["dflash_architecture_config"].update(
        projector_type="dspine", code_size=8, code_rank=8, refine_top_k=4, teacher_forcing_layers=1
    )
    model = get_tiny_llama(num_hidden_layers=4)
    mtsp.convert(model, [("dflash", config)])
    model = model.to("cuda").eval()
    # As accelerate's dispatch hooks a module placed on another device.
    add_hook_to_module(model.lm_head.to("cpu"), AlignDevicesHook(execution_device="cpu"))

    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    base_token, draft_tokens = model.pseudo_speculative_generate(input_ids, steps=3)

    assert draft_tokens.shape == (1, 3)
    assert base_token.device == draft_tokens.device == input_ids.device
