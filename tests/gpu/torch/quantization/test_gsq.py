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


"""Real calibration and native GSQ checkpoint/export contracts."""

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
from modelopt.torch.quantization.nn import StaticBlockScaleQuantizer
from modelopt.torch.quantization.qtensor import NVFP4QTensor


def _config(initializer):
    config = load_recipe("general/ptq/nvfp4_gsq-gptq").quantize.model_dump(exclude_unset=True)
    config["algorithm"].update(
        initializer=initializer,
        qk_steps=2,
        epochs=1,
        initializer_samples=1,
        train_samples=3,
        batch_size=2,
        microbatch_size=1,
        layerwise={"enable": True},
    )
    return config


@pytest.mark.parametrize("method", ["gptq", "max", "mse"])
@pytest.mark.parametrize("baseline", [False, True])
def test_staged_gsq_streaming_matches_device_cache(method, baseline):
    torch.manual_seed(42)
    reference = (
        get_tiny_llama(
            hidden_size=64,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
        )
        .cuda()
        .bfloat16()
        .eval()
    )
    streamed = deepcopy(reference)
    ids = torch.arange(16, device="cuda").reshape(1, -1)

    def forward(m):
        for i in range(4):
            mask = torch.ones_like(ids)
            mask[:, 2 + i] = 0
            m((ids + i) % 32, use_cache=False, attention_mask=mask, position_ids=ids * (i + 1))

    config = _config({"method": method})
    config["algorithm"].update(
        qk_steps=2,
        epochs=1,
        initializer_samples=1,
        train_samples=3,
        batch_size=2,
        microbatch_size=1,
        layerwise={"enable": True},
    )
    if baseline:
        for prepared in (reference, streamed):
            mtq.quantize(prepared, {**config, "algorithm": "max"}, forward)
        config["algorithm"] = {"method": method, "layerwise": {"enable": True}}

    def apply(model):
        if baseline:
            return mtq.calibrate(model, config["algorithm"], forward)
        return mtq.quantize(model, config, forward)

    apply(reference)
    config["algorithm"]["layerwise"]["activation_cache_device"] = "cpu"
    apply(streamed)
    torch.testing.assert_close(streamed(ids).logits, reference(ids).logits, rtol=0, atol=0)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(streamed.state_dict()[name], value, rtol=0, atol=0)


def test_gsq_retains_final_state_even_when_initializer_has_lower_loss():
    torch.manual_seed(17)
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


@pytest.mark.parametrize("method", ["gptq", "max", "mse"])
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


def test_staged_gsq_qk_responds_to_correlations_between_hessian_blocks():
    method = "max"
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
    config = _config(initializer)
    config["algorithm"].update(
        qk_steps=32,
        epochs=1,
        initializer_samples=1,
        train_samples=1,
        batch_size=1,
        assignment_lr=0.01,
        scale_lr=0.003,
    )
    config["algorithm"]["layerwise"] = {"enable": False}
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
def test_gsq_uses_constant_qk_and_cosine_joint_rates(monkeypatch, steps):
    gsq_module = import_module("modelopt.torch.quantization.gsq")
    config = _config("max")
    config["algorithm"].update(qk_steps=1, train_samples=steps, batch_size=1)
    algorithm = GSQCalibConfig(**config["algorithm"])
    calls = []
    step_fn = gsq_module._lion_step

    def observe_step(groups, momentum, multiplier):
        assert [lr for _, lr, _ in groups] == [algorithm.assignment_lr, algorithm.scale_lr]
        assert [decay for _, _, decay in groups] == [algorithm.weight_decay, 0.0]
        calls.append(multiplier)
        step_fn(groups, momentum, multiplier)

    monkeypatch.setattr(gsq_module, "_lion_step", observe_step)
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
    ids = torch.arange(8, device="cuda").reshape(1, -1)
    mtq.quantize(model, config, lambda m: [m(ids, use_cache=False) for _ in range(steps + 1)])
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(()))], lr=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps - 1, 1))
    expected = []
    for _ in range(steps):
        expected.append(0.1 + 0.9 * scheduler.get_last_lr()[0] if steps > 1 else 0.1)
        optimizer.step()
        scheduler.step()
    assert calls == pytest.approx([1.0, 1.0] + expected * 2)


def test_gsq_rolls_back_mutating_initializer_failure():
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
    config = _config({"method": "gptq", "block_size": 64})
    mtq.quantize(model, {**config, "algorithm": None})
    layer = model.model.layers[0]
    before = deepcopy(layer.state_dict())
    hidden = torch.randn(1, 8, 64, device="cuda", dtype=layer.self_attn.q_proj.weight.dtype)
    rotary = model.model.rotary_emb(hidden, torch.arange(8, device="cuda").reshape(1, -1))
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()

    def forward(m):
        for _ in range(2):
            m(hidden, position_embeddings=rotary)

    def interrupt(module, inputs):
        if isinstance(module.weight_quantizer, StaticBlockScaleQuantizer):
            raise RuntimeError("calibration interrupted")

    handle = layer.self_attn.q_proj.register_forward_pre_hook(interrupt)
    try:
        with pytest.raises(RuntimeError, match="calibration interrupted"):
            gsq(
                layer,
                forward,
                initializer=config["algorithm"]["initializer"],
                initializer_samples=1,
                train_samples=1,
            )
    finally:
        handle.remove()
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert before.keys() == layer.state_dict().keys()
    for name, value in before.items():
        torch.testing.assert_close(layer.state_dict()[name], value, rtol=0, atol=0)
    assert all(p.requires_grad for p in layer.parameters())
    mtq.calibrate(model, "max")
    assert torch.isfinite(model(torch.ones(1, 8, device="cuda", dtype=torch.long)).logits).all()


@pytest.mark.parametrize("method", ["gptq", "max", "mse"])
def test_layerwise_gsq_hf_export_preserves_hard_codes(method, tmp_path):
    model = (
        get_tiny_llama(
            hidden_size=64, intermediate_size=64, num_attention_heads=4, num_key_value_heads=4
        )
        .cuda()
        .eval()
    )
    x = torch.randint(0, 32, (2, 16), device="cuda")
    config = _config({"method": method})
    config["algorithm"].update(initializer_samples=1, train_samples=1, batch_size=1)

    def forward(m):
        for row in x.split(1):
            m(row, use_cache=False, attention_mask=torch.ones_like(row))

    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    mtq.quantize(model, config, forward)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert not model.training
    assert all(p.requires_grad for p in model.parameters())
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
            dequant = NVFP4QTensor(
                module.weight.shape, module.weight.dtype, copy.weight
            ).dequantize(
                scale=copy.weight_scale,
                double_scale=copy.weight_scale_2,
                block_sizes={-1: 16},
                dtype=module.weight.dtype,
            )
            torch.testing.assert_close(dequant, module.weight, rtol=0, atol=0)
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
