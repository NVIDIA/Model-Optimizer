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

import json
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from _test_utils.torch.transformers_models import get_tiny_llama
from safetensors.torch import load_file

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.export.layerwise_export import LayerwiseExporter
from modelopt.torch.export.quant_utils import preprocess_linear_fusion
from modelopt.torch.export.unified_export_hf import _export_quantized_weight, export_hf_checkpoint
from modelopt.torch.quantization.config import GSQCalibConfig
from modelopt.torch.quantization.gsq import _LocalShift, gsq
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from modelopt.torch.utils.distributed import (
    DistributedProcessGroup,
    DistributionPlan,
    broadcast_from,
    set_distribution_plan,
)


def _distributed_staged_gsq(rank, size, method, train_samples):
    torch.manual_seed(42)
    model = (
        get_tiny_llama(
            hidden_size=64,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
        )
        .to(f"cuda:{rank}")
        .eval()
    )
    set_distribution_plan(model, DistributionPlan(world_size=size))
    ids = torch.arange(8, device=f"cuda:{rank}").reshape(1, -1)

    def forward(m):
        for i in range(rank, 1, size):
            m(ids + i, use_cache=False)
        for i in range(rank, train_samples, size):
            m(ids + i + 1, use_cache=False)

    config = _config({"method": method})
    config["algorithm"].update(
        reconstruction="staged",
        initializer_samples=1,
        train_samples=train_samples,
        batch_size=2,
        microbatch_size=1,
        qk_steps=2,
        epochs=1,
        layerwise={"enable": True, "activation_cache_device": "cpu"},
    )
    mtq.quantize(model, config, forward)
    for value in model.state_dict().values():
        expected = value.clone()
        dist.broadcast(expected, 0)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert torch.isfinite(model(ids).logits).all()


@pytest.mark.parametrize("method", ["gptq", "max", "mse"])
@pytest.mark.parametrize("train_samples", [1, 3])
def test_distributed_staged_gsq_with_uneven_shards(
    need_2_gpus, dist_workers, method, train_samples
):
    dist_workers.run(_distributed_staged_gsq, method=method, train_samples=train_samples)


def _distributed_gsq_resume(rank, size, checkpoint_dir, train_samples, failure_mode):
    torch.manual_seed(42)
    base = (
        get_tiny_llama(
            hidden_size=64,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=3,
        )
        .to(f"cuda:{rank}")
        .eval()
    )
    set_distribution_plan(base, DistributionPlan(world_size=size))
    ids = torch.arange(8, device=f"cuda:{rank}").reshape(1, -1)

    def forward(m):
        for count in (1, train_samples):
            for i in range(rank, count, size):
                m(ids + i, use_cache=False)

    config = _config({"method": "max"})
    config["algorithm"].update(
        reconstruction="staged",
        initializer_samples=1,
        train_samples=train_samples,
        batch_size=2,
        microbatch_size=1,
        qk_steps=2,
        epochs=1,
        layerwise={"enable": True, "activation_cache_device": "cpu"},
    )
    reference = deepcopy(base)
    mtq.quantize(reference, config, forward)
    config["algorithm"]["layerwise"]["checkpoint_dir"] = checkpoint_dir
    interrupted = deepcopy(base)
    manifest_path = Path(checkpoint_dir) / "manifest.json"

    def interrupt(module, inputs):
        if failure_mode == "forward" and rank == 0 and manifest_path.exists():
            raise RuntimeError("interrupted layer")

    original_save = torch.save

    def interrupted_save(value, path, *args, **kwargs):
        if rank == 1 and failure_mode == "write" and "layer_0001/next_inputs" in str(path):
            Path(path).write_bytes(b"incomplete")
            raise OSError("interrupted write")
        return original_save(value, path, *args, **kwargs)

    handle = interrupted.model.layers[1].register_forward_pre_hook(interrupt)
    failing_rank = 0 if failure_mode == "forward" else 1
    message = (
        f"interrupted {failure_mode if failure_mode == 'write' else 'layer'}"
        if rank == failing_rank
        else "another rank"
    )
    with (
        patch("torch.save", interrupted_save),
        pytest.raises((RuntimeError, OSError), match=message),
    ):
        mtq.quantize(interrupted, config, forward)
    handle.remove()
    assert json.loads(manifest_path.read_text())["last_completed_layer"] == 0
    for r in range(size):
        assert (Path(checkpoint_dir) / "layer_0000" / f"next_inputs_rank_{r:05d}.pt").is_file()
    resumed = deepcopy(base)
    mtq.quantize(resumed, config, forward)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
    complete = deepcopy(base)
    mtq.quantize(complete, config, lambda m: pytest.fail("Completed resume must not replay data"))
    torch.testing.assert_close(complete(ids).logits, reference(ids).logits, rtol=0, atol=0)
    native = Path(checkpoint_dir) / "native.pth"
    broadcast_from("save native GSQ", lambda: mto.save(complete, native))
    restored = mto.restore(deepcopy(base), native).eval()
    torch.testing.assert_close(restored(ids).logits, reference(ids).logits, rtol=0, atol=0)
    config["algorithm"]["seed"] = 10
    with pytest.raises(ValueError):
        mtq.quantize(deepcopy(base), config, forward)


@pytest.mark.parametrize("train_samples", [1, 3])
@pytest.mark.parametrize("failure_mode", ["forward", "write"])
def test_distributed_gsq_checkpoint_resume(
    need_2_gpus, dist_workers, tmp_path, train_samples, failure_mode
):
    dist_workers.run(
        _distributed_gsq_resume,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        train_samples=train_samples,
        failure_mode=failure_mode,
    )
    model = (
        get_tiny_llama(
            hidden_size=64,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=3,
        )
        .cuda()
        .eval()
    )
    config = _config({"method": "max"})
    config["algorithm"].update(
        reconstruction="staged",
        initializer_samples=1,
        train_samples=train_samples,
        batch_size=2,
        microbatch_size=1,
        qk_steps=2,
        epochs=1,
        layerwise={
            "enable": True,
            "activation_cache_device": "cpu",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    )
    with pytest.raises(ValueError):
        mtq.quantize(
            model, config, lambda m: pytest.fail("GPU count mismatch must refuse before replay")
        )


def _distributed_gsq_layerwise_export(rank, size, export_dir):
    torch.manual_seed(42)
    base = (
        get_tiny_llama(
            hidden_size=64,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
        )
        .to(f"cuda:{rank}")
        .bfloat16()
        .eval()
    )
    set_distribution_plan(base, DistributionPlan(world_size=size))
    ids = torch.arange(8, device=f"cuda:{rank}").reshape(1, -1)

    def forward(m):
        for _ in range(2) if rank == 0 else ():
            m(ids, use_cache=False)

    config = _config({"method": "max"})
    config["algorithm"].update(
        reconstruction="staged",
        initializer_samples=1,
        train_samples=1,
        batch_size=1,
        microbatch_size=1,
        qk_steps=2,
        epochs=1,
        layerwise={"enable": True, "activation_cache_device": "cpu"},
    )
    reference = deepcopy(base)
    mtq.quantize(reference, config, forward)
    expected = deepcopy(reference.model.layers[0].self_attn.q_proj)
    _export_quantized_weight(expected, torch.bfloat16)
    config["algorithm"]["layerwise"].update(
        export_dir=export_dir,
        checkpoint_dir=export_dir + "-checkpoints",
    )
    for _ in range(2):
        candidate = deepcopy(base)
        exporter = LayerwiseExporter(candidate, export_dir, dtype=torch.bfloat16)
        mtq.quantize(candidate, config, forward)
        broadcast_from("finalize GSQ export", exporter.finalize)
        tensors = {}
        for shard in Path(export_dir).glob("*.safetensors"):
            tensors.update(load_file(shard))
        for key in ("weight", "weight_scale", "weight_scale_2"):
            actual = tensors[f"model.layers.0.self_attn.q_proj.{key}"]
            assert torch.equal(
                actual.reshape(-1).view(torch.uint8),
                getattr(expected, key).cpu().reshape(-1).view(torch.uint8),
            )


def test_distributed_gsq_layerwise_export(need_2_gpus, dist_workers, tmp_path):
    dist_workers.run(_distributed_gsq_layerwise_export, export_dir=str(tmp_path / "export"))


def _distributed_gsq_refusal(rank, size, refusal):
    torch.manual_seed(42)
    model = (
        get_tiny_llama(
            hidden_size=64,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=1,
        )
        .to(f"cuda:{rank}")
        .eval()
    )
    config = _config({"method": "max"})
    config["algorithm"].update(
        reconstruction="staged",
        initializer_samples=1,
        train_samples=3,
        batch_size=2,
        microbatch_size=1,
        qk_steps=2,
        epochs=1,
        layerwise={"enable": True, "activation_cache_device": "cpu"},
    )
    mtq.quantize(model, {**config, "algorithm": None})
    set_distribution_plan(
        model, DistributionPlan(world_size=size, dense="fsdp" if refusal == "plan" else "replicate")
    )
    group = None
    if refusal == "subgroup":
        group = dist.new_group()
        linear = model.model.layers[0].self_attn.q_proj
        linear.parallel_state.data_parallel_group = DistributedProcessGroup(group)
    if refusal == "linear":
        config["algorithm"].update(reconstruction="linear", layerwise={"enable": False})
    if refusal == "rank_config":
        config["algorithm"]["seed"] = rank
    before = deepcopy(model.state_dict())

    def forward(m):
        if refusal != "nonfinite":
            raise AssertionError("Invalid placement or config must refuse before data replay")
        for count in (1, 3):
            for _ in range(rank, count, size):
                hidden = torch.full(
                    (1, 8, 64),
                    float("nan") if rank == 1 else 0.25,
                    device=f"cuda:{rank}",
                    dtype=next(m.parameters()).dtype,
                )
                m(inputs_embeds=hidden, use_cache=False)

    try:
        expected = (FloatingPointError, RuntimeError) if refusal == "nonfinite" else ValueError
        message = "Nonfinite|another rank" if refusal == "nonfinite" else None
        with pytest.raises(expected, match=message):
            mtq.calibrate(model, config["algorithm"], forward)
        assert before.keys() == model.state_dict().keys()
        for name, value in before.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        assert all(p.requires_grad for p in model.parameters())
    finally:
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("refusal", ["plan", "linear", "subgroup", "rank_config", "nonfinite"])
def test_distributed_gsq_refuses_or_rolls_back(need_2_gpus, dist_workers, refusal):
    dist_workers.run(_distributed_gsq_refusal, refusal=refusal)


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


@pytest.mark.parametrize("method", ["gptq", "max", "mse"])
@pytest.mark.parametrize("cache", ["cpu", "mmap"])
@pytest.mark.parametrize("baseline", [False, True])
def test_staged_gsq_streaming_matches_device_cache_and_checkpoint(
    method, cache, baseline, tmp_path
):
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
    fresh = deepcopy(reference)
    ids = torch.arange(16, device="cuda").reshape(1, -1)

    def forward(m):
        for i in range(4):
            mask = torch.ones_like(ids)
            mask[:, 2 + i] = 0
            m((ids + i) % 32, use_cache=False, attention_mask=mask, position_ids=ids * (i + 1))

    config = _config({"method": method})
    config["algorithm"].update(
        reconstruction="staged",
        qk_steps=2,
        epochs=1,
        initializer_samples=1,
        train_samples=3,
        batch_size=2,
        microbatch_size=1,
        layerwise={"enable": True},
    )
    if baseline:
        for prepared in (reference, streamed, fresh):
            mtq.quantize(prepared, {**config, "algorithm": "max"}, forward)
        config["algorithm"] = {"method": method, "layerwise": {"enable": True}}

    def apply(model):
        if baseline:
            return mtq.calibrate(model, config["algorithm"], forward)
        return mtq.quantize(model, config, forward)

    apply(reference)
    config["algorithm"]["layerwise"].update(
        activation_cache_device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        activation_cache_dir=str(tmp_path / "cache") if cache == "mmap" else None,
    )
    visits = 0

    def interrupt(module, inputs):
        nonlocal visits
        visits += 1
        if visits == 5:
            raise RuntimeError("interrupt after the first committed block")

    handle = streamed.model.layers[1].register_forward_pre_hook(interrupt)
    with pytest.raises(RuntimeError):
        apply(streamed)
    handle.remove()
    manifest = json.loads((tmp_path / "checkpoints" / "manifest.json").read_text())
    assert manifest["last_completed_layer"] == 0
    apply(fresh)
    torch.testing.assert_close(fresh(ids).logits, reference(ids).logits, rtol=0, atol=0)
    if cache == "mmap":
        assert not list((tmp_path / "cache").iterdir())


@pytest.mark.parametrize("method", ["max", "gptq", "mse"])
def test_gsq_quality_restore_and_native_export(method, tmp_path):
    torch.manual_seed(9)
    model = _Projections().cuda().bfloat16().eval()
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    x[:, :16] *= 20
    reference = model(x).detach()
    baseline = deepcopy(model)
    initializer = {"method": method}
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


def test_gsq_rolls_back_mutating_initializer_failure():
    model = _Projections().cuda().eval()
    initializer = {"method": "gptq", "block_size": 64}
    config = _config(initializer)
    mtq.quantize(model, {**config, "algorithm": None})
    before = deepcopy(model.state_dict())
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    calls = 0

    def forward(m):
        nonlocal calls
        calls += 1
        # Interrupt GPTQ after promotion.
        if calls > 1:
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
    mtq.calibrate(model, "max")
    model(torch.ones(4, 64, device="cuda"))


@pytest.mark.parametrize("method", ["gptq", "max", "mse"])
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
