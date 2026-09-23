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

import copy
from contextlib import nullcontext

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    linear_attention_training_phase,
)
from modelopt.torch.quantization.nn import TensorQuantizer

KimiDeltaAttention = pytest.importorskip("fla.layers.kda").KimiDeltaAttention


def _layer(head_dim=16):
    return (
        KimiDeltaAttention(hidden_size=32, head_dim=head_dim, num_heads=2, use_short_conv=True)
        .cuda()
        .train()
    )


def _forward(model, hidden):
    policy = getattr(model, "linear_attention_config", None)
    phase = (
        linear_attention_training_phase(model, [min(31, hidden.shape[1])])
        if policy is not None and policy.decode is not None
        else nullcontext()
    )
    with phase, torch.autocast("cuda", dtype=torch.bfloat16):
        return model(hidden)[0]


@pytest.mark.parametrize(
    "mode",
    [
        "state",
        "state_int8",
        "w",
        "prefill_fp8",
        "prefill_nvfp4",
        "arithmetic",
        "decode_token",
        "decode_replay",
        "decode_token_int8",
        "decode_replay_int8",
        "decode_decay",
        "decode_replay_hadamard_int8",
    ],
)
@pytest.mark.timeout(180)
def test_fla_layer_qat_restore_and_optimizer(tmp_path, mode):
    torch.manual_seed(73)
    head_dim = 32 if "hadamard" in mode else 16
    model = _layer(head_dim)
    hidden = torch.randn(1, 73, 32, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        baseline = _forward(model, hidden)
    attributes = {"num_bits": (4, 3), "type": "dynamic", "axis": (0, 1, 2)}
    if mode == "prefill_nvfp4":
        attributes = {
            "num_bits": (2, 1),
            "type": "dynamic",
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        }
    cfg = {
        "quant_cfg": [{"quantizer_name": "*", "enable": False}],
        "algorithm": None,
        "linear_attention": [{"module_name": "*", "cfg": {"backend": "matmul"}}],
    }
    if mode in ("w", "prefill_fp8", "prefill_nvfp4"):
        cfg["quant_cfg"].append({"quantizer_name": "*kda_w_quantizer", "cfg": attributes})
    if mode.startswith("prefill"):
        cfg["quant_cfg"].append({"quantizer_name": "*linear_attn_sites.*", "cfg": attributes})
    state_attributes = {**attributes, "axis": (0, 1)}
    if mode.endswith("int8"):
        state_attributes.update(num_bits=8, unsigned=False, narrow_range=True)
    if mode.startswith("state"):
        cfg["quant_cfg"].append({"quantizer_name": "*kda_state_quantizer", "cfg": state_attributes})
    if mode == "arithmetic":
        cfg["linear_attention"][0]["cfg"]["elementwise"] = {"value_residual": "bfloat16"}
    if mode.startswith("decode"):
        cfg["quant_cfg"].append({"quantizer_name": "*kda_state_quantizer", "cfg": state_attributes})
        policy_cfg = cfg["linear_attention"][0]["cfg"]
        policy_cfg["state"] = {"block_v": 16}
        policy_cfg["decode"] = {
            "mode": "replay" if mode.startswith("decode_replay") else "token",
            "implementation": "triton",
        }
        if mode.startswith("decode_replay"):
            policy_cfg["decode"]["replay"] = {"window": 5}
        if mode == "decode_decay":
            policy_cfg["decode"]["decay_log_step"] = 1 / 256
        if "hadamard" in mode:
            policy_cfg["state"]["block_v"] = 32
            policy_cfg["decode"]["state_codec"] = "int8_hadamard32"
    mtq.quantize(model, cfg)
    output = _forward(model, hidden)
    assert torch.isfinite(output).all()
    enabled = [q for q in model.modules() if isinstance(q, TensorQuantizer) and q.is_enabled]
    policy = model.linear_attention_config
    for q in enabled:
        q.disable()
    model.linear_attention_config = LinearAttentionConfig()
    with torch.no_grad():
        torch.testing.assert_close(_forward(model, hidden), baseline, rtol=0, atol=0)
    for q in enabled:
        q.enable()
    model.linear_attention_config = policy
    checkpoint = tmp_path / "model.pth"
    mto.save(model, checkpoint)
    restored = mto.restore(_layer(head_dim), checkpoint)
    actual = _forward(restored, hidden)
    torch.testing.assert_close(actual, output, rtol=0, atol=0)
    assert restored.linear_attention_config == policy
    actual.float().square().mean().backward()
    assert restored.q_proj.weight.grad is not None
    for parameter in restored.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    if mode == "prefill_fp8":
        copied = copy.deepcopy(restored)
        calls = []
        hook = copied.kda_w_quantizer.register_forward_hook(lambda *args: calls.append(True))
        with torch.no_grad():
            torch.testing.assert_close(_forward(copied, hidden), actual, rtol=0, atol=0)
        hook.remove()
        assert calls, "a copied layer must use its own quantizer handles"
    before = restored.q_proj.weight.detach().clone()
    torch.optim.SGD(restored.parameters(), lr=0.1).step()
    assert not torch.equal(before, restored.q_proj.weight)
    restored.eval()
    if mode.startswith("decode"):
        with (
            linear_attention_training_phase(restored, [0]),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            assert torch.isfinite(restored(hidden[:, :1])[0]).all()
            with pytest.raises(NotImplementedError, match="use_cache=False"):
                restored(hidden[:, :1], use_cache=True)
        with (
            pytest.raises(ValueError, match="explicit"),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            restored(hidden)
    else:
        with pytest.raises(NotImplementedError, match="chunk path"):
            _forward(restored, hidden[:, :1])


@pytest.mark.parametrize("hadamard", [False, True])
def test_decode_phase_spans_activation_checkpoint_backward(hadamard):
    model = _layer(32 if hadamard else 16)
    quant_cfg = [{"quantizer_name": "*", "enable": False}]
    if hadamard:
        quant_cfg.append(
            {
                "quantizer_name": "*kda_state_quantizer",
                "cfg": {
                    "num_bits": 8,
                    "type": "dynamic",
                    "axis": (0, 1),
                    "unsigned": False,
                    "narrow_range": True,
                },
            }
        )
    mtq.quantize(
        model,
        {
            "quant_cfg": quant_cfg,
            "algorithm": None,
            "linear_attention": [
                {
                    "module_name": "*",
                    "cfg": {
                        "backend": "matmul",
                        "decode": {
                            "implementation": "triton",
                            "state_codec": "int8_hadamard32" if hadamard else "tile",
                        },
                    },
                }
            ],
        },
    )
    hidden = torch.randn(1, 73, 32, device="cuda", requires_grad=True)
    with linear_attention_training_phase(model, [31]):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = checkpoint(lambda x: model(x)[0], hidden, use_reentrant=False)
        actual.square().mean().backward()
    assert model._linear_attention_prefill_lengths is None
    assert torch.isfinite(hidden.grad).all()
    assert torch.count_nonzero(model.q_proj.weight.grad) > 0
    with linear_attention_training_phase(model, [0]):
        with linear_attention_training_phase(model, [73]):
            assert model._linear_attention_prefill_lengths == (73,)
        assert model._linear_attention_prefill_lengths == (0,)
    assert model._linear_attention_prefill_lengths is None
