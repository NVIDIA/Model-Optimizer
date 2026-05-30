# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Checklist §7 recipe-setup smoke as pytest.

The full :meth:`DMD2DiffusionRecipe.setup()` runs :meth:`super().setup()` which
requires NCCL/FSDP2 + real Qwen weights, so we can't call it directly on CPU.
Instead we exercise the four helpers it consumes — ``_load_frozen_teacher``,
``_load_fake_score``, ``_resolve_dmd_config``, ``_resolve_pipeline_cls``,
``_build_fake_score_optimizer`` — after monkeypatching
``NeMoAutoDiffusionPipeline.from_pretrained`` to return a tiny
``ToyTransformer`` stub. Together they cover every assertion §7 names:

- ``_teacher`` is frozen + eval (§7.a).
- ``_fake_score`` is trainable + train mode (§7.b).
- ``_dmd_config.num_train_timesteps is None`` (§7.c).
- ``_dmd_pipeline`` resolves to ``QwenImageDMDPipeline`` (§7.d).
- ``_fake_score_optimizer`` has trainable fake-score params (§7.e).

The YAML-parse and mock-dataloader bullets stay self-contained at the top.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

YAML_PATH = "examples/diffusers/fastgen/configs/dmd2_qwen_image_smoke.yaml"

# The recipe file isn't a regular package — import it via path. ``dmd2_recipe`` is
# what ``dmd2_finetune.py`` does too (``from dmd2_recipe import DMD2DiffusionRecipe``).
RECIPE_DIR = Path(__file__).resolve().parents[4] / "examples" / "diffusers" / "fastgen"


@pytest.fixture
def dmd2_recipe_module():
    """Import ``dmd2_recipe.py`` once per test. Adds the example dir to sys.path."""
    sys.path.insert(0, str(RECIPE_DIR))
    try:
        mod = importlib.import_module("dmd2_recipe")
        yield mod
    finally:
        sys.path.remove(str(RECIPE_DIR))


# ---------------------------------------------------------------------------- #
# Tiny modules + mock pipeline used in monkeypatched from_pretrained            #
# ---------------------------------------------------------------------------- #


class _ToyTransformer(nn.Module):
    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.linear(hidden_states)


class _MockPipe:
    """Minimal stand-in for ``NeMoAutoDiffusionPipeline``: only ``.transformer`` is read."""

    def __init__(self, transformer: nn.Module) -> None:
        self.transformer = transformer


@pytest.fixture
def cfg():
    """Parse the Qwen DMD2 YAML. Skips if AutoModel isn't on PYTHONPATH."""
    arg_parser = pytest.importorskip("nemo_automodel.components.config._arg_parser")
    # parse_args_and_load_config consumes sys.argv; clear it so it doesn't try
    # to pick up pytest's flags.
    old_argv = sys.argv
    sys.argv = ["pytest"]
    try:
        return arg_parser.parse_args_and_load_config(
            str(RECIPE_DIR / "configs" / "dmd2_qwen_image_smoke.yaml")
        )
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------- #
# §7.1 — YAML parses without launching training                                #
# ---------------------------------------------------------------------------- #


def test_yaml_parses_without_launch(cfg):
    assert cfg.get("dmd2.recipe_path") == "general/distillation/dmd2_qwen_image"
    assert cfg.get("dmd2.pipeline_plugin") == "qwen_image"
    # ``_target_`` is auto-resolved from the string path to the actual callable
    # by AutoModel's argparse layer. The resolved ``__module__`` may be the source
    # module (``...diffusion.mock_dataloader``) rather than the YAML's re-export
    # shortcut (``...diffusion``), so accept either by matching the function name
    # and the prefix.
    target = cfg.get("data.dataloader._target_")
    assert callable(target)
    assert target.__name__ == "build_mock_t2i_dataloader"
    assert target.__module__.startswith("nemo_automodel.components.datasets.diffusion")


# ---------------------------------------------------------------------------- #
# §7.2 — mock dataloader instantiates and yields one batch                     #
# ---------------------------------------------------------------------------- #


def test_mock_dataloader_yields_one_batch(cfg):
    dl_mod = pytest.importorskip("nemo_automodel.components.datasets.diffusion")
    build_mock_t2i_dataloader = dl_mod.build_mock_t2i_dataloader

    data_cfg = cfg.get("data.dataloader")
    kwargs = data_cfg.to_dict() if hasattr(data_cfg, "to_dict") else dict(data_cfg)
    kwargs.pop("_target_", None)
    kwargs["batch_size"] = 1
    kwargs["dp_rank"] = 0
    kwargs["dp_world_size"] = 1
    dl, _sampler = build_mock_t2i_dataloader(**kwargs)
    batch = next(iter(dl))
    assert "image_latents" in batch
    assert "text_embeddings" in batch
    assert batch["image_latents"].shape[0] == 1
    assert batch["text_embeddings"].shape[0] == 1


# ---------------------------------------------------------------------------- #
# Helper: build a recipe instance + patch ``from_pretrained`` to return tiny  #
# transformer copies. We deliberately don't call ``setup()`` (which would     #
# run the parent's FSDP2 path and need NCCL).                                  #
# ---------------------------------------------------------------------------- #


@pytest.fixture
def stub_recipe(cfg, dmd2_recipe_module, monkeypatch):
    recipe = dmd2_recipe_module.DMD2DiffusionRecipe(cfg)
    recipe.model_id = cfg.get("model.pretrained_model_name_or_path", "Qwen/Qwen-Image")
    recipe.bf16 = torch.bfloat16
    recipe.device = "cpu"
    recipe.learning_rate = float(cfg.get("optim.learning_rate", 1.0e-5))

    def _fake_from_pretrained(*_args, **_kwargs):
        return _MockPipe(transformer=_ToyTransformer(d=8)), {}

    monkeypatch.setattr(
        dmd2_recipe_module.NeMoAutoDiffusionPipeline,
        "from_pretrained",
        _fake_from_pretrained,
    )
    return recipe


# ---------------------------------------------------------------------------- #
# §7.a — _load_frozen_teacher returns a frozen + eval module                  #
# ---------------------------------------------------------------------------- #


def test_load_frozen_teacher_eval_and_no_grad(stub_recipe):
    teacher = stub_recipe._load_frozen_teacher()
    assert teacher.training is False
    assert not any(p.requires_grad for p in teacher.parameters())


# ---------------------------------------------------------------------------- #
# §7.b — _load_fake_score returns a trainable + train-mode module             #
# ---------------------------------------------------------------------------- #


def test_load_fake_score_trainable_and_train_mode(stub_recipe):
    fake_score = stub_recipe._load_fake_score()
    assert fake_score.training is True
    assert all(p.requires_grad for p in fake_score.parameters())


def test_fake_score_initializes_with_teacher_weights(stub_recipe, dmd2_recipe_module, monkeypatch):
    base = _ToyTransformer(d=8)
    base_state = {name: tensor.detach().clone() for name, tensor in base.state_dict().items()}

    def _fake_from_pretrained(*_args, **_kwargs):
        transformer = _ToyTransformer(d=8)
        transformer.load_state_dict(base_state)
        return _MockPipe(transformer=transformer), {}

    monkeypatch.setattr(
        dmd2_recipe_module.NeMoAutoDiffusionPipeline,
        "from_pretrained",
        _fake_from_pretrained,
    )

    teacher = stub_recipe._load_frozen_teacher()
    fake_score = stub_recipe._load_fake_score()

    assert teacher is not fake_score
    assert teacher.state_dict().keys() == fake_score.state_dict().keys()
    for name, teacher_tensor in teacher.state_dict().items():
        assert torch.equal(teacher_tensor, fake_score.state_dict()[name]), name

    assert teacher.training is False
    assert not any(p.requires_grad for p in teacher.parameters())
    assert fake_score.training is True
    assert all(p.requires_grad for p in fake_score.parameters())


def test_teacher_stays_frozen_across_phase_toggles(stub_recipe):
    teacher = stub_recipe._load_frozen_teacher()
    stub_recipe.__dict__["_teacher"] = teacher
    stub_recipe.model = _ToyTransformer(d=8)
    stub_recipe.__dict__["_fake_score"] = stub_recipe._load_fake_score()
    stub_recipe.__dict__["_discriminator"] = None

    for is_student_phase in (True, False, True):
        stub_recipe._set_grad_requirements(is_student_phase)
        assert teacher.training is False
        assert not any(p.requires_grad for p in teacher.parameters())


# ---------------------------------------------------------------------------- #
# §7.c — _resolve_dmd_config returns DMDConfig with num_train_timesteps=None  #
# ---------------------------------------------------------------------------- #


def test_resolve_dmd_config_num_train_timesteps_none(stub_recipe):
    dmd_cfg = stub_recipe._resolve_dmd_config()
    assert dmd_cfg.num_train_timesteps is None


# ---------------------------------------------------------------------------- #
# §7.d — _resolve_pipeline_cls returns QwenImageDMDPipeline                   #
# ---------------------------------------------------------------------------- #


def test_resolve_pipeline_cls_is_qwen_image(stub_recipe):
    cls = stub_recipe._resolve_pipeline_cls()
    assert cls.__name__ == "QwenImageDMDPipeline"


# ---------------------------------------------------------------------------- #
# §7.e — _build_fake_score_optimizer has trainable fake-score params          #
# ---------------------------------------------------------------------------- #


def test_build_fake_score_optimizer_has_trainable_params(stub_recipe):
    # _build_fake_score_optimizer reads self._fake_score, so populate it via the
    # production helper (which we already trust per the test above).
    stub_recipe.__dict__["_fake_score"] = stub_recipe._load_fake_score()
    optimizer = stub_recipe._build_fake_score_optimizer()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) >= 1
    params = optimizer.param_groups[0]["params"]
    assert len(params) > 0
    assert all(p.requires_grad for p in params)
