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

"""Offline AR must draft exactly what the online DFlash generate drafts.

``offline_ar_validate.py`` carries its own copy of the draft half of
``HFDFlashModel.pseudo_speculative_generate``, fed from a dump; this pins the copy to it.
"""

import importlib.util

import torch
from _test_utils.examples.run_command import MODELOPT_ROOT
from _test_utils.torch.speculative.dflash import get_dflash_config
from _test_utils.torch.transformers_models import get_tiny_llama

import modelopt.torch.speculative as mtsp

_SPEC = importlib.util.spec_from_file_location(
    "offline_ar_validate",
    MODELOPT_ROOT / "examples" / "speculative_decoding" / "offline_ar" / "offline_ar_validate.py",
)
assert _SPEC is not None and _SPEC.loader is not None
offline_ar = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(offline_ar)


def test_dump_fed_drafts_match_online_generate():
    torch.manual_seed(0)
    model = get_tiny_llama(num_hidden_layers=4)
    config = get_dflash_config()
    config["dflash_swa_window_size"] = 5  # shorter than the context, so the window bites
    mtsp.convert(model, [("dflash", config)])
    model = model.double().eval()  # full and prefix forwards then agree far below argmax ties
    for p in model.dflash_module.parameters():  # at init scale every draft token is the same
        p.data.normal_(0, 0.5)

    prompt_len, ids = 6, torch.randint(1, model.config.vocab_size, (1, 6))
    for _ in range(8):  # the target's greedy reply, as the online validator builds it
        token, _ = model.pseudo_speculative_generate(ids, steps=0)
        ids = torch.cat([ids, token], dim=-1)

    # One forward over the whole conversation, as compute_hidden_states_hf.py dumps it.
    with torch.no_grad():
        hidden = model._base_model(input_ids=ids, output_hidden_states=True).hidden_states
    aux = torch.cat([hidden[lid + 1][0] for lid in model.target_layer_ids], dim=-1)

    steps = model.dflash_block_size - 1
    for pos in range(prompt_len, ids.shape[1] - 1):
        _, online = model.pseudo_speculative_generate(ids[:, :pos], steps=steps)
        assert torch.equal(offline_ar.draft_block(model, ids[0, pos], aux[:pos]), online[0])
