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

"""Optional AutoModel integration: packed parity, gradients, and FSDP2 resume."""

import copy

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("nemo_automodel")
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.distributed.fsdp import fully_shard

from examples.llm_qat.nemo_automodel.nvfp4 import NVFP4GroupedExperts, export_hf_experts
from modelopt.torch.quantization.qtensor import NVFP4QTensor


def _make_model(device):
    torch.manual_seed(123)
    cfg = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=32,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=True,
        apply_router_weight_after_down=True,
        dtype=torch.float32,
    )
    native = GroupedExperts(cfg).to(device)
    with torch.no_grad():
        for p in native.parameters():
            p.normal_(std=0.02)
    state = {}
    for e, (up, down) in enumerate(zip(native.gate_and_up_projs, native.down_projs)):
        for name, w in zip(("gate_proj", "up_proj", "down_proj"), (*up.T.chunk(2), down.T)):
            # This scalar loses precision if passed through a BF16 module dtype cast.
            scale2 = torch.tensor(0.00010172526526730508, device=device)
            packed, scale, _ = NVFP4QTensor.quantize(
                w.detach().bfloat16().contiguous(), 16, weights_scaling_factor_2=scale2
            )
            stem = f"{e}.{name}."
            state.update(
                {
                    stem + "weight": packed._quantized_data,
                    stem + "weight_scale": scale,
                    stem + "weight_scale_2": scale2,
                    stem + "input_scale": torch.tensor(0.0031415927, device=device),
                }
            )
    return native, state


def _inputs(device, dtype=torch.bfloat16):
    x = torch.randn(8, 32, device=device, dtype=dtype, requires_grad=True)
    probs = torch.full((8, 2), 0.5, device=device, requires_grad=True)
    # Experts 2 and 3 are deliberately unrouted.
    ids = torch.tensor([[0, 1]], device=device).expand(8, 2)
    mask = torch.tensor([True] * 6 + [False] * 2, device=device)
    return x, mask, probs, ids


def _dequant(w, scale, scale2):
    packed, scale, scale2 = NVFP4QTensor.quantize(w.contiguous(), 16, scale, scale2)
    return packed.dequantize(scale=scale, double_scale=scale2, block_sizes={-1: 16}, dtype=w.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_published_parity_and_gradients(dtype):
    native, state = _make_model("cuda")
    adapter = NVFP4GroupedExperts(native.bfloat16(), state).to(dtype)
    assert adapter.gate_and_up_projs is native.gate_and_up_projs
    exported = export_hf_experts(adapter.state_dict(), dtype=dtype)
    for key, value in state.items():
        assert torch.equal(
            exported[key].reshape(-1).view(torch.uint8), value.reshape(-1).view(torch.uint8)
        ), key
    x, mask, probs, ids = _inputs("cuda", dtype)
    output = adapter(x, mask, probs, ids)
    reference = torch.zeros_like(output, dtype=torch.float32)
    for e in range(2):

        def linear(inputs, name):
            stem = f"{e}.{name}."
            weight = NVFP4QTensor((32, 32), x.dtype, state[stem + "weight"]).dequantize(
                scale=state[stem + "weight_scale"],
                double_scale=state[stem + "weight_scale_2"],
                block_sizes={-1: 16},
                dtype=x.dtype,
            )
            return F.linear(_dequant(inputs, None, state[stem + "input_scale"]), weight)

        h = F.silu(linear(x.detach()[mask], "gate_proj")) * linear(x.detach()[mask], "up_proj")
        reference[mask] += linear(h, "down_proj").float() * probs[mask, e, None].detach()
    reference[~mask] = 0
    torch.testing.assert_close(output, reference.to(output.dtype), rtol=0, atol=0)
    output.float().square().sum().backward()
    for value in (x, probs, adapter.gate_and_up_projs, adapter.down_projs):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    assert torch.count_nonzero(adapter.gate_and_up_projs.grad[2:]) == 0
    for empty in (False, True):
        adapter.zero_grad(set_to_none=True)
        xx, mm, pp, ii = _inputs("cuda", dtype)
        mm.zero_()
        xx.detach().fill_(65000)  # Finite FP16 values whose sum would overflow.
        if empty:
            xx, mm, pp, ii = xx[:0], mm[:0], pp[:0], ii[:0]
        result = adapter(xx, mm, pp, ii)
        result.sum().backward()
        assert torch.count_nonzero(result) == 0
        assert all(
            p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in adapter.parameters()
        )


@pytest.mark.parametrize("invalid", ["backend", "routing", "scale"])
def test_refusal_leaves_source_unchanged(invalid):
    native, state = _make_model("cuda")
    before = {k: v.clone() for k, v in native.state_dict().items()}
    if invalid == "backend":
        native.use_torch_mm = True
    elif invalid == "routing":
        native.config.apply_router_weight_after_down = False
    else:
        state["0.gate_proj.weight_scale_2"] = torch.tensor(float("nan"), device="cuda")
    with pytest.raises(ValueError):
        NVFP4GroupedExperts(native, state)
    for name, value in native.state_dict().items():
        assert torch.equal(value, before[name])


def _fsdp_resume(rank, size):
    native, state = _make_model(torch.device("cuda", rank))
    model = NVFP4GroupedExperts(native, state)
    fully_shard(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = _inputs(torch.device("cuda", rank))
    if rank == 0:
        inputs[1].zero_()  # A rank with no routed tokens must still participate in backward.

    def step(module, opt):
        opt.zero_grad(set_to_none=True)
        module(*inputs).float().square().mean().backward()
        opt.step()

    step(model, optimizer)
    options = StateDictOptions(full_state_dict=True)
    sd, od = get_state_dict(model, optimizer, options=options)
    # Isolate the checkpoint: restore can mutate mappings and alias optimizer step tensors.
    exported = export_hf_experts(sd)
    new_native, _ = _make_model(torch.device("cuda", rank))
    restored = NVFP4GroupedExperts(new_native, state)
    fully_shard(restored)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    set_state_dict(
        restored,
        restored_optimizer,
        model_state_dict=copy.deepcopy(sd),
        optim_state_dict=copy.deepcopy(od),
        options=options,
    )
    torch.testing.assert_close(model(*inputs), restored(*inputs), rtol=0, atol=0)
    step(model, optimizer)
    step(restored, restored_optimizer)
    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.equal(a.to_local(), b.to_local())
    assert len(exported) == 4 * 3 * 4
    with pytest.raises(ValueError):
        export_hf_experts(model.state_dict())


def test_fsdp2_resume_and_export(dist_workers):
    dist_workers.run(_fsdp_resume)
