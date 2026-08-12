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

"""Focused tests for the PDD loss seam used by AutoModel's diffusion recipe."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))

from pdd import compat as pdd_compat
from pdd.recipe import _validate_prepared_student
from pdd.training import PDDFlowMatchingStepAdapter

from modelopt.torch.fastgen import PDDConfig


class _PreparedStudent(nn.Module):
    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(in_channels=4)
        self.proj_out = nn.Linear(3, out_features)


class _PreparedQwenStudent(_PreparedStudent):
    def __init__(self, out_features: int) -> None:
        super().__init__(out_features)
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.to_add_out = nn.Linear(3, 3)
        block.txt_mlp = nn.Module()
        block.txt_mlp.net = nn.ModuleList(
            [
                nn.ModuleDict({"proj": nn.Linear(3, 4)}),
                nn.Identity(),
                nn.Linear(4, 3),
            ]
        )
        self.transformer_blocks = nn.ModuleList([block])
        self.backbone = nn.Linear(3, 3)


class _LossPipeline:
    def __init__(self, *, guidance_scale: float | None = None) -> None:
        self.config = SimpleNamespace(guidance_scale=guidance_scale, data_free=False)
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.last_call = None

    def compute_loss(self, data, *, condition, negative_condition, collect_metrics):
        self.last_call = (data, condition, negative_condition, collect_metrics)
        per_sample = (data * self.scale).square().flatten(1).mean(1)
        return per_sample.mean(), {"student_target_mse": per_sample}


class _DataFreeLossPipeline:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            guidance_scale=None,
            data_free=True,
            grid_max_t=0.999,
            grid_size=4,
        )
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.calls = []

    def compute_data_free_loss(
        self,
        state,
        *,
        n,
        condition,
        negative_condition,
        collect_metrics,
    ):
        self.calls.append(
            {
                "state": state.detach().clone(),
                "n": n.detach().clone(),
                "condition": tuple(value.detach().clone() for value in condition),
                "negative_condition": negative_condition,
                "collect_metrics": collect_metrics,
            }
        )
        per_sample = (state * self.scale).square().flatten(1).mean(1)
        return (
            per_sample.mean(),
            {"student_target_mse": per_sample},
            state.detach() + 1,
            n + 2,
        )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "image_latents": torch.ones(2, 1, 2, 2),
        "text_embeddings": torch.ones(2, 3, 4),
        "text_embeddings_mask": torch.ones(2, 3, dtype=torch.long),
        "negative_text_embeddings": torch.zeros(2, 3, 4),
        "negative_text_embeddings_mask": torch.ones(2, 3, dtype=torch.long),
    }


def test_prepared_student_width_is_validated_before_training() -> None:
    config = PDDConfig(
        grid_size=8,
        block_size_min=1,
        block_size_max=8,
        inference_blocks=[4, 4],
        student_sample_steps=2,
    )
    _validate_prepared_student(_PreparedStudent(out_features=32), config)

    with pytest.raises(ValueError, match="Prepare the Qwen PDD student"):
        _validate_prepared_student(_PreparedStudent(out_features=4), config)


def test_pdd_fsdp_preserves_fp32_timestep_inputs(monkeypatch) -> None:
    def build_manager_args(**kwargs):
        del kwargs
        return {"_manager_type": "fsdp2"}

    monkeypatch.setattr(
        pdd_compat.automodel_diffusion_train,
        "_build_diffusion_parallel_manager_args",
        build_manager_args,
    )
    with pdd_compat.automodel_pdd_setup():
        manager_args = pdd_compat.automodel_diffusion_train._build_diffusion_parallel_manager_args(
            dtype=torch.float32,
            compute_dtype=torch.bfloat16,
            lora_enabled=False,
        )

    policy = manager_args["mp_policy"]
    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.float32
    assert policy.output_dtype == torch.bfloat16
    assert policy.cast_forward_inputs is False
    assert (
        pdd_compat.automodel_diffusion_train._build_diffusion_parallel_manager_args
        is build_manager_args
    )


def test_unused_final_text_outputs_are_frozen_before_optimizer_collection(monkeypatch) -> None:
    class _Pipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del cls, args, kwargs
            return SimpleNamespace(transformer=_PreparedQwenStudent(out_features=32)), {}

    monkeypatch.setattr(
        pdd_compat.automodel_diffusion_train,
        "NeMoAutoDiffusionPipeline",
        _Pipeline,
    )

    original_descriptor = _Pipeline.__dict__["from_pretrained"]
    with pdd_compat.automodel_pdd_setup():
        setup_pipeline = pdd_compat.automodel_diffusion_train.NeMoAutoDiffusionPipeline
        student, _ = setup_pipeline.from_pretrained("student", load_for_training=True)
        teacher, _ = setup_pipeline.from_pretrained("teacher", load_for_training=False)

    frozen_names = {
        name
        for name, parameter in student.transformer.named_parameters()
        if not parameter.requires_grad
    }
    assert frozen_names == {
        "transformer_blocks.0.attn.to_add_out.weight",
        "transformer_blocks.0.attn.to_add_out.bias",
        "transformer_blocks.0.txt_mlp.net.0.proj.weight",
        "transformer_blocks.0.txt_mlp.net.0.proj.bias",
        "transformer_blocks.0.txt_mlp.net.2.weight",
        "transformer_blocks.0.txt_mlp.net.2.bias",
    }
    assert all(parameter.requires_grad for parameter in teacher.transformer.parameters())
    assert _Pipeline.__dict__["from_pretrained"] is original_descriptor
    assert pdd_compat.automodel_diffusion_train.NeMoAutoDiffusionPipeline is _Pipeline


def test_automodel_setup_rejects_an_unvalidated_release(monkeypatch) -> None:
    monkeypatch.setattr(pdd_compat.nemo_automodel, "__version__", "0.6.0")

    with (
        pytest.raises(RuntimeError, match=r"requires nemo_automodel release 0\.5\.0"),
        pdd_compat.automodel_pdd_setup(),
    ):
        pass


@pytest.mark.parametrize("guidance_scale", [None, 4.0])
def test_step_adapter_returns_native_tuple_and_preserves_pdd_gradient(guidance_scale) -> None:
    pipeline = _LossPipeline(guidance_scale=guidance_scale)
    adapter = PDDFlowMatchingStepAdapter(pipeline)

    per_sample, loss, prediction, metrics = adapter.step(
        model=nn.Identity(),
        batch=_batch(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    loss.backward()

    assert prediction is None
    torch.testing.assert_close(per_sample, torch.full((2,), 4.0))
    assert metrics["student_target_mse"] is per_sample
    torch.testing.assert_close(pipeline.scale.grad, torch.tensor(4.0))
    _, _, negative_condition, collect_metrics = pipeline.last_call
    assert (negative_condition is not None) is (guidance_scale is not None)
    assert collect_metrics is True


def test_data_free_step_uses_independent_accumulation_slots_and_reuses_prompts() -> None:
    pipeline = _DataFreeLossPipeline()
    optimizer_step = [0]
    adapter = PDDFlowMatchingStepAdapter(
        pipeline,
        grad_acc_steps=2,
        latent_shape=(1, 2, 2),
        optimizer_step_getter=lambda: optimizer_step[0],
    )

    def prompt_batch(value: float) -> dict[str, torch.Tensor]:
        return {
            "text_embeddings": torch.full((1, 3, 4), value),
            "text_embeddings_mask": torch.ones(1, 3, dtype=torch.long),
        }

    adapter.step(
        nn.Identity(),
        prompt_batch(1.0),
        device=torch.device("cpu"),
        dtype=torch.float32,
        global_step=0,
    )
    adapter.step(
        nn.Identity(),
        prompt_batch(2.0),
        device=torch.device("cpu"),
        dtype=torch.float32,
        global_step=0,
    )
    first_states = [call["state"] for call in pipeline.calls]
    optimizer_step[0] = 1

    adapter.step(
        nn.Identity(),
        prompt_batch(101.0),
        device=torch.device("cpu"),
        dtype=torch.float32,
        global_step=0,
    )
    adapter.step(
        nn.Identity(),
        prompt_batch(102.0),
        device=torch.device("cpu"),
        dtype=torch.float32,
        global_step=0,
    )

    assert torch.equal(pipeline.calls[0]["n"], torch.tensor([0]))
    assert torch.equal(pipeline.calls[1]["n"], torch.tensor([0]))
    assert torch.equal(pipeline.calls[2]["n"], torch.tensor([2]))
    assert torch.equal(pipeline.calls[3]["n"], torch.tensor([2]))
    torch.testing.assert_close(pipeline.calls[2]["state"], first_states[0] + 1)
    torch.testing.assert_close(pipeline.calls[3]["state"], first_states[1] + 1)
    assert torch.all(pipeline.calls[2]["condition"][0] == 1.0)
    assert torch.all(pipeline.calls[3]["condition"][0] == 2.0)

    optimizer_step[0] = 2
    adapter.step(
        nn.Identity(),
        prompt_batch(3.0),
        device=torch.device("cpu"),
        dtype=torch.float32,
        global_step=0,
    )
    assert torch.equal(pipeline.calls[4]["n"], torch.tensor([0]))
    assert torch.all(pipeline.calls[4]["condition"][0] == 3.0)
