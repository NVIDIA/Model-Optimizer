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

"""Native MTP training with a compiled loss and an eager decoder."""

from copy import deepcopy

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from modelopt.torch.speculative.config import MTPConfig
from modelopt.torch.speculative.plugins.modeling_fakebase import FakeBaseConfig, FakeBaseModel
from modelopt.torch.speculative.plugins.modeling_mtp import (
    MTPModule,
    MTPModuleOutput,
    create_mtp_model,
)


class _NativeMTP(MTPModule):
    def __init__(self, config):
        super().__init__(config)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def _forward(
        self,
        hidden_states,
        inputs_embeds,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        input_ids,
    ):
        assert not torch.compiler.is_compiling(), "The native decoder must remain eager"
        hidden = torch.tanh(self.proj(hidden_states) + inputs_embeds)
        return MTPModuleOutput(hidden, hidden, past_key_values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("loss_type", "length"), [("lk_lambda", 17), ("lk_lambda", 1026), ("soft_target_ce", 17)]
)
def test_native_mtp_compiles_only_loss(dtype, loss_type, length):
    torch.manual_seed(31)
    torch._dynamo.reset()
    architecture = FakeBaseConfig(
        hidden_size=16,
        vocab_size=128,
        num_hidden_layers=1,
        dtype=dtype,
    )
    base = FakeBaseModel(architecture).to(device="cuda", dtype=dtype)
    native = _NativeMTP(architecture).to(device="cuda", dtype=dtype)
    cfg = MTPConfig(mtp_ttt_steps=3, mtp_mix_hidden_states=True, mtp_loss_type=loss_type)
    eager = create_mtp_model(
        deepcopy(native), deepcopy(base), cfg.model_copy(update={"mtp_use_torch_compile": False})
    )
    compiled = create_mtp_model(native, base, cfg)
    batch = {
        "input_ids": torch.randint(128, (2, length), device="cuda"),
        "loss_mask": torch.randint(2, (2, length), device="cuda"),
        "base_model_outputs": {
            "draft_hidden_states": torch.randn(2, length, 16, device="cuda", dtype=dtype),
            "base_model_hidden_states": torch.randn(2, length, 16, device="cuda", dtype=dtype),
        },
    }
    rng = torch.cuda.get_rng_state()
    expected = eager(**batch)
    expected.loss.backward()
    torch.cuda.set_rng_state(rng)
    graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    actual = compiled(**batch)
    actual.loss.backward()
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > graphs_before
    # Fused CE avoids some eager BF16 rounding; LK computes its statistics in FP32.
    loss_rtol = 1e-3 if dtype == torch.bfloat16 and loss_type == "soft_target_ce" else 1e-4
    torch.testing.assert_close(actual.loss, expected.loss, atol=1e-5, rtol=loss_rtol)
    torch.testing.assert_close(actual.train_acc, expected.train_acc)
    for parameter, reference in zip(
        compiled.eagle_module.parameters(), eager.eagle_module.parameters()
    ):
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-4, rtol=0.02)
    assert all(
        parameter.grad is None
        for name, parameter in compiled.named_parameters()
        if not name.startswith("eagle_module.")
    )
