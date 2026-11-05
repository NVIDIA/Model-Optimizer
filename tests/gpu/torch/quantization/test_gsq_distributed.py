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
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.transformers_models import get_tiny_llama
from test_gsq import _config

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
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
        filename = f"next_inputs_rank_{r:05d}.pt" if size > 1 else "next_inputs.pt"
        assert (Path(checkpoint_dir) / "layer_0000" / filename).is_file()
    resumed = deepcopy(base)
    mtq.quantize(resumed, config, forward)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
    complete = resumed
    if size > 1:
        complete = deepcopy(base)
        mtq.quantize(
            complete, config, lambda m: pytest.fail("Completed resume must not replay data")
        )
    torch.testing.assert_close(complete(ids).logits, reference(ids).logits, rtol=0, atol=0)
    native = Path(checkpoint_dir) / "native.pth"
    broadcast_from("save native GSQ", lambda: mto.save(complete, native))
    restored = mto.restore(deepcopy(base), native).eval()
    torch.testing.assert_close(restored(ids).logits, reference(ids).logits, rtol=0, atol=0)
    if size > 1:
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


@pytest.mark.parametrize("refusal", ["plan", "subgroup", "rank_config", "nonfinite"])
def test_distributed_gsq_refuses_or_rolls_back(need_2_gpus, dist_workers, refusal):
    dist_workers.run(_distributed_gsq_refusal, refusal=refusal)


def test_single_process_gsq_checkpoint_resume(tmp_path):
    _distributed_gsq_resume(
        rank=0,
        size=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        train_samples=3,
        failure_mode="forward",
    )
