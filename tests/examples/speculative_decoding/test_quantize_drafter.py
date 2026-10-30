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
"""Run the drafter PTQ script on a tiny EAGLE-3-shaped checkpoint and check its export."""

import json

import safetensors.torch
import torch
from _test_utils.examples.run_command import run_example_command


def test_quantize_drafter_keeps_vocab_maps(tmp_path):
    """Test drafter PTQ keeps an EAGLE-3 draft vocab's int64 d2t and bool t2d exact."""
    hidden, draft_vocab, vocab = 128, 64, 2048
    d2t = torch.arange(draft_vocab) * 13 + 257
    t2d = torch.zeros(vocab, dtype=torch.bool)
    t2d[torch.arange(draft_vocab) + d2t] = True
    drafter_dir = tmp_path / "drafter"
    drafter_dir.mkdir()
    safetensors.torch.save_file(
        {
            "fc.weight": torch.randn(hidden, 3 * hidden, dtype=torch.bfloat16),
            "midlayer.self_attn.q_proj.weight": torch.randn(
                hidden, 2 * hidden, dtype=torch.bfloat16
            ),
            "lm_head.weight": torch.randn(draft_vocab, hidden, dtype=torch.bfloat16),
            "d2t": d2t,
            "t2d": t2d,
        },
        drafter_dir / "model.safetensors",
    )
    (drafter_dir / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLMEagle3"]})
    )

    run_example_command(
        [
            "python",
            "./scripts/quantize_drafter.py",
            "--drafter_path",
            drafter_dir,
            "--qformat",
            "fp8",
            "--export_path",
            tmp_path / "export",
        ],
        "speculative_decoding",
    )
    state_dict = safetensors.torch.load_file(tmp_path / "export" / "model.safetensors")
    assert state_dict["d2t"].dtype == torch.int64 and torch.equal(state_dict["d2t"], d2t)
    assert state_dict["t2d"].dtype == torch.bool and torch.equal(state_dict["t2d"], t2d)
    assert state_dict["midlayer.self_attn.q_proj.weight"].dtype == torch.float8_e4m3fn
