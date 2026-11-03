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


"""Real calibration, AWQ smoothing, and native GSQ checkpoint/export contracts."""

from copy import deepcopy
from importlib import import_module

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _test_utils.torch.transformers_models import get_tiny_llama
from safetensors.torch import load_file

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.export.quant_utils import preprocess_linear_fusion
from modelopt.torch.export.unified_export_hf import _export_quantized_weight, export_hf_checkpoint
from modelopt.torch.quantization.config import GSQCalibConfig
from modelopt.torch.quantization.gsq import _LocalShift, gsq
from modelopt.torch.quantization.qtensor import NVFP4QTensor


class _Projections(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 32, bias=False)
        self.k_proj = nn.Linear(64, 32, bias=False)
        self.v_proj = nn.Linear(64, 32, bias=False)

    def forward(self, x):
        return torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)


def _config(initializer):
    config = load_recipe("general/ptq/nvfp4_gsq-gptq").quantize.model_dump(exclude_unset=True)
    config["algorithm"].update(
        reconstruction="linear",
        initializer=initializer,
        steps=4,
        eval_every=2,
        max_tokens=64,
        batch_size=32,
        layerwise={"enable": False},
    )
    return config


@pytest.mark.parametrize("method", ["max", "gptq", "awq_lite", "awq_full", "mse"])
def test_gsq_quality_restore_and_native_export(method, tmp_path):
    torch.manual_seed(9)
    model = _Projections().cuda().bfloat16().eval()
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    x[:, :16] *= 20
    reference = model(x).detach()
    baseline = deepcopy(model)
    initializer = {"method": method}
    if "awq" in method:
        initializer["alpha_step"] = 0.5
    if method == "gptq":
        initializer.update(perc_damp=0.1, block_size=64)
    config = _config(initializer)
    mtq.quantize(baseline, {**config, "algorithm": ["max", initializer]}, lambda m: m(x))
    preprocess_linear_fusion([baseline.q_proj, baseline.k_proj, baseline.v_proj])
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    mtq.quantize(model, config, lambda m: m(x))
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    actual = model(x).detach()
    actual_loss = F.mse_loss(actual.float(), reference.float())
    assert torch.isfinite(actual_loss)
    assert all(not m.input_quantizer.is_enabled for m in (model.q_proj, model.k_proj, model.v_proj))
    if "awq" in method:
        for name in ("q_proj", "k_proj", "v_proj"):
            torch.testing.assert_close(
                getattr(model, name).input_quantizer.pre_quant_scale,
                getattr(baseline, name).input_quantizer.pre_quant_scale,
            )
    shared = [m.weight_quantizer.global_amax for m in (model.q_proj, model.k_proj, model.v_proj)]
    assert all(s is shared[0] for s in shared)
    checkpoint = tmp_path / "gsq.pth"
    mto.save(model, checkpoint)
    restored = mto.restore(_Projections().cuda().bfloat16(), checkpoint).eval()
    torch.testing.assert_close(restored(x), actual, rtol=0, atol=0)
    for name in ("q_proj", "k_proj", "v_proj"):
        original, reloaded = getattr(model, name), getattr(restored, name)
        hard = original.weight.detach().clone()
        _export_quantized_weight(original, torch.bfloat16)
        _export_quantized_weight(reloaded, torch.bfloat16)
        for attr in ("weight", "weight_scale", "weight_scale_2"):
            assert torch.equal(
                getattr(original, attr).reshape(-1).view(torch.uint8),
                getattr(reloaded, attr).reshape(-1).view(torch.uint8),
            )
        dequant = NVFP4QTensor(hard.shape, hard.dtype, original.weight).dequantize(
            scale=original.weight_scale,
            double_scale=original.weight_scale_2,
            block_sizes={-1: 16},
            dtype=hard.dtype,
        )
        torch.testing.assert_close(dequant, hard, rtol=0, atol=0)


@pytest.mark.parametrize("reconstruction", ["linear", "staged"])
def test_gsq_retains_final_state_even_when_initializer_has_lower_loss(reconstruction):
    torch.manual_seed(17)
    if reconstruction == "staged":
        model = (
            get_tiny_llama(
                hidden_size=64,
                intermediate_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_hidden_layers=1,
            )
            .cuda()
            .eval()
        )
        original = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        config = _config("max")
        config["algorithm"].update(
            reconstruction="staged",
            qk_steps=1,
            epochs=1,
            initializer_samples=1,
            train_samples=1,
            batch_size=1,
            scale_lr=20.0,
            layerwise={"enable": True},
        )
        baseline = deepcopy(model)
        mtq.quantize(baseline, {**config, "algorithm": "max"})
        projection = baseline.model.layers[0].self_attn.q_proj
        baseline_weight = projection.weight_quantizer(projection.weight).detach()
        ids = torch.arange(16, device="cuda").reshape(1, -1)
        mtq.quantize(model, config, lambda m: [m(ids, use_cache=False) for _ in range(2)])
        actual = model.model.layers[0].self_attn.q_proj.weight
        assert F.mse_loss(actual.float(), original.float()) > 10 * F.mse_loss(
            baseline_weight.float(), original.float()
        )
        return
    model = nn.Linear(64, 16, bias=False, device="cuda").eval()
    baseline = deepcopy(model)
    x = torch.randn(64, 64, device="cuda")
    target = model(x).detach()
    config = _config("max")
    mtq.quantize(baseline, {**config, "algorithm": "max"})
    config["algorithm"].update(steps=1, scale_lr=20.0)
    mtq.quantize(model, config, lambda m: m(x))
    assert F.mse_loss(model(x), target) > 100 * F.mse_loss(baseline(x), target)


def test_staged_gsq_rejects_preexisting_awq_scaling_without_mutation():
    model = (
        get_tiny_llama(
            hidden_size=64,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=1,
        )
        .cuda()
        .eval()
    )
    config = _config("max")
    config["algorithm"].update(
        reconstruction="staged",
        qk_steps=1,
        epochs=1,
        initializer_samples=1,
        train_samples=1,
        batch_size=1,
        layerwise={"enable": True},
    )
    mtq.quantize(model, {**config, "algorithm": None})
    for module in model.model.layers[0].modules():
        if hasattr(module, "input_quantizer") and isinstance(module, nn.Linear):
            module.input_quantizer.pre_quant_scale = torch.linspace(
                0.5, 2, 64, device="cuda", dtype=torch.bfloat16
            )
    before = deepcopy(model.state_dict())
    ids = torch.arange(16, device="cuda").reshape(1, -1)
    with pytest.raises(ValueError):
        mtq.calibrate(
            model, config["algorithm"], lambda m: [m(ids, use_cache=False) for _ in range(2)]
        )
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["gptq", "awq_lite", "awq_full"])
@pytest.mark.parametrize("change", ["tokens", "mask", "positions"])
def test_staged_gsq_isolates_initializer_data_and_trains_both_joint_regions(method, change):
    model = (
        get_tiny_llama(
            hidden_size=64,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=1,
        )
        .cuda()
        .eval()
    )
    other = deepcopy(model)
    config = _config({"method": method})
    config["algorithm"].update(
        reconstruction="staged",
        qk_steps=8,
        epochs=2,
        initializer_samples=1,
        train_samples=3,
        batch_size=2,
        microbatch_size=2,
        assignment_lr=0.01,
        scale_lr=0.003,
        layerwise={"enable": True},
    )
    initializer = torch.arange(16, device="cuda").reshape(1, -1)

    def loop(m, offset):
        m(initializer, use_cache=False)
        for i in range(3):
            mask = torch.ones_like(initializer)
            positions = torch.arange(16, device="cuda").reshape(1, -1)
            if offset and change == "mask":
                mask[:, 4::2] = 0
            if offset and change == "positions":
                positions = positions * 2
            tokens = (initializer * (i + 1) + (offset if change == "tokens" else 0)) % 32
            m(tokens, use_cache=False, attention_mask=mask, position_ids=positions)

    mtq.quantize(model, config, lambda m: loop(m, 0))
    mtq.quantize(other, config, lambda m: loop(m, 13))
    first, second = model.model.layers[0], other.model.layers[0]
    for name in ("q_proj", "k_proj"):
        torch.testing.assert_close(
            getattr(first.self_attn, name).weight,
            getattr(second.self_attn, name).weight,
            rtol=0,
            atol=0,
        )
    for region, names in [
        ("self_attn", ("v_proj", "o_proj")),
        ("mlp", ("gate_proj", "up_proj", "down_proj")),
    ]:
        assert any(
            not torch.equal(
                getattr(getattr(first, region), name).weight,
                getattr(getattr(second, region), name).weight,
            )
            for name in names
        )


@pytest.mark.parametrize("method", ["max", "awq_lite"])
def test_staged_gsq_qk_responds_to_correlations_between_hessian_blocks(method):
    base = (
        get_tiny_llama(
            hidden_size=256,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=1,
            dtype=torch.float32,
        )
        .cuda()
        .eval()
    )
    layer = base.model.layers[0]
    x = torch.randn(1, 16, 1, device="cuda").expand(-1, -1, 256).clone()
    x += 0.02 * torch.randn_like(x)
    flipped = x.clone()
    flipped[..., :128].neg_()
    positions = torch.arange(16, device="cuda").reshape(1, -1)
    with torch.no_grad():
        rotary = base.model.rotary_emb(x, positions)
    initializer = {"method": method}
    if method == "awq_lite":
        # A single alpha candidate makes initialization independent of input signs.
        initializer["alpha_step"] = 1.0
    config = _config(initializer)
    config["algorithm"].update(
        reconstruction="staged",
        qk_steps=32,
        epochs=1,
        initializer_samples=1,
        train_samples=1,
        batch_size=1,
        assignment_lr=0.01,
        scale_lr=0.003,
    )
    initialized, trained = [], []
    for inputs in (x, flipped):
        baseline = deepcopy(layer)
        mtq.quantize(
            baseline,
            {**config, "algorithm": ["max", initializer]},
            lambda m: m(inputs, position_embeddings=rotary),
        )
        preprocess_linear_fusion(
            [baseline.self_attn.q_proj, baseline.self_attn.k_proj, baseline.self_attn.v_proj]
        )
        initialized.append(baseline)
        candidate = deepcopy(layer)
        mtq.quantize(
            candidate, config, lambda m: [m(inputs, position_embeddings=rotary) for _ in range(2)]
        )
        trained.append(candidate)
    # Flipping an entire 128-feature block preserves all within-block Grams.
    # Only the full Hessian can distinguish the Q/K objectives for these inputs.
    for name in ("q_proj", "k_proj"):
        first, second = [getattr(model.self_attn, name) for model in initialized]
        torch.testing.assert_close(first.weight, second.weight, rtol=0, atol=0)
        torch.testing.assert_close(
            first.weight_quantizer.amax, second.weight_quantizer.amax, rtol=0, atol=0
        )
        if method == "awq_lite":
            torch.testing.assert_close(
                first.input_quantizer.pre_quant_scale,
                second.input_quantizer.pre_quant_scale,
                rtol=0,
                atol=0,
            )
        assert not torch.equal(
            getattr(trained[0].self_attn, name).weight, getattr(trained[1].self_attn, name).weight
        )


def test_local_shift_uses_fixed_indices_and_masks_invalid_shifts():
    model = nn.Linear(64, 16, bias=False, device="cuda", dtype=torch.float32)
    config = _config("max")
    mtq.quantize(model, {**config, "algorithm": "max"})
    scale = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(model.weight_quantizer)
    assignment = _LocalShift(model, nn.Parameter(scale.log()), GSQCalibConfig())
    assignment.center.copy_(
        (torch.arange(model.weight.numel(), device="cuda") % 15).reshape_as(assignment.center)
    )
    center = assignment.center.clone()
    indices = center.long().unsqueeze(-1) + torch.arange(-2, 3, device="cuda")
    valid = (indices >= 0) & (indices < 15)
    F.mse_loss(assignment().float(), model.weight.float()).backward()
    for parameter in (assignment.logits, assignment.log_block, assignment.log_global):
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    assert torch.count_nonzero(assignment.logits.grad[~valid]) == 0
    with torch.no_grad():
        assignment.logits.zero_()
        assignment.logits[..., 2] = 10
        assignment.logits.masked_fill_(~valid, 1000)
    hard = assignment(hard=True)
    block, global_scale = assignment.scales()
    expected = assignment.grid[center.long()].reshape(*block.shape, 16) * (
        block * global_scale
    ).unsqueeze(-1)
    torch.testing.assert_close(hard, expected.reshape_as(hard), rtol=0, atol=0)
    assert torch.equal(assignment.center, center)
    assert assignment.logits.shape[-1] == 5


@pytest.mark.parametrize("steps", [1, 5])
def test_gsq_uses_cosine_rates_and_no_scale_decay(monkeypatch, steps):
    gsq_module = import_module("modelopt.torch.quantization.gsq")
    config = _config("max")
    config["algorithm"].update(steps=steps)
    algorithm = GSQCalibConfig(**config["algorithm"])
    reference = torch.optim.SGD(
        [
            {"params": [nn.Parameter(torch.zeros(()))], "lr": algorithm.assignment_lr},
            {"params": [nn.Parameter(torch.zeros(()))], "lr": algorithm.scale_lr},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(reference, T_max=max(steps - 1, 1))
    # Both groups share a relative 10% floor, unlike PyTorch's absolute eta_min.
    calls = []
    step_fn = gsq_module._lion_step

    def observe_step(groups, momentum, lr_multiplier):
        cosine = scheduler.get_last_lr()
        expected = [0.1 * base + 0.9 * rate for base, rate in zip(scheduler.base_lrs, cosine)]
        assert [lr * lr_multiplier for _, lr, _ in groups] == pytest.approx(expected)
        assert [decay for _, _, decay in groups] == [algorithm.weight_decay, 0.0]
        step_fn(groups, momentum, lr_multiplier)
        calls.append(lr_multiplier)
        reference.step()
        scheduler.step()

    if steps == 1:
        reference.step()
        scheduler.step()
    monkeypatch.setattr(gsq_module, "_lion_step", observe_step)
    model = nn.Linear(64, 16, bias=False, device="cuda").eval()
    x = torch.randn(64, 64, device="cuda")
    mtq.quantize(model, config, lambda m: m(x))
    assert len(calls) == steps


@pytest.mark.parametrize("method", ["gptq", "awq_full"])
def test_gsq_rolls_back_mutating_initializer_failure(method):
    model = _Projections().cuda().eval()
    initializer = (
        {"method": "gptq", "block_size": 64}
        if method == "gptq"
        else {"method": "awq_full", "alpha_step": 0.5}
    )
    config = _config(initializer)
    mtq.quantize(model, {**config, "algorithm": None})
    before = deepcopy(model.state_dict())
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    calls = 0

    def forward(m):
        nonlocal calls
        calls += 1
        # Interrupt GPTQ after promotion, or AWQ clipping after Lite has changed weights.
        if calls > (1 if method == "gptq" else 3):
            raise RuntimeError("calibration interrupted")
        m(torch.arange(1, 65, device="cuda", dtype=torch.float32).expand(4, -1))

    with pytest.raises(RuntimeError, match="calibration interrupted"):
        gsq(model, forward, initializer=config["algorithm"]["initializer"])
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert before.keys() == model.state_dict().keys()
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    assert all(p.requires_grad for p in model.parameters())
    assert all(not hasattr(m, "awq_lite") and not hasattr(m, "awq_clip") for m in model.modules())
    mtq.calibrate(model, "max")
    model(torch.ones(4, 64, device="cuda"))


@pytest.mark.parametrize("method", ["gptq", "awq_lite"])
@pytest.mark.parametrize("reconstruction", ["linear", "staged"])
def test_layerwise_gsq_hf_export_preserves_hard_codes(method, reconstruction, tmp_path):
    model = (
        get_tiny_llama(
            hidden_size=64, intermediate_size=64, num_attention_heads=4, num_key_value_heads=4
        )
        .cuda()
        .eval()
    )
    x = torch.randint(0, 32, (2, 16), device="cuda")
    config = _config({"method": method})
    config["algorithm"].update(steps=2, layerwise={"enable": True})
    if reconstruction == "staged":
        config["algorithm"].update(
            reconstruction="staged",
            qk_steps=2,
            epochs=1,
            initializer_samples=1,
            train_samples=1,
            batch_size=1,
        )

    def forward(m):
        for row in x.split(1):
            m(row, use_cache=False, attention_mask=torch.ones_like(row))

    mtq.quantize(model, config, forward)
    actual = model(x, use_cache=False).logits.detach()
    assert torch.isfinite(actual).all()
    state = tmp_path / "staged.pth"
    mto.save(model, state)
    restored = mto.restore(
        get_tiny_llama(
            hidden_size=64, intermediate_size=64, num_attention_heads=4, num_key_value_heads=4
        ).cuda(),
        state,
    ).eval()
    torch.testing.assert_close(restored(x, use_cache=False).logits, actual, rtol=0, atol=0)
    expected = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight_quantizer") and module.weight_quantizer.is_enabled:
            copy = deepcopy(module)
            _export_quantized_weight(copy, torch.bfloat16)
            expected[name] = {
                key: getattr(copy, key).detach().cpu().clone()
                for key in ("weight", "weight_scale", "weight_scale_2")
            }
    export_hf_checkpoint(model, export_dir=tmp_path)
    tensors = {}
    for shard in tmp_path.glob("*.safetensors"):
        tensors.update(load_file(str(shard)))
    assert expected
    for name, state in expected.items():
        for key, value in state.items():
            torch.testing.assert_close(
                tensors[f"{name}.{key}"].reshape(-1).view(torch.uint8),
                value.reshape(-1).view(torch.uint8),
                rtol=0,
                atol=0,
            )
