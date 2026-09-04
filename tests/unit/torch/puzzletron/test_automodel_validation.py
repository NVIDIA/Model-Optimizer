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

"""CPU contracts for native AutoModel realized-checkpoint validation."""

from omegaconf import OmegaConf

from modelopt.torch.puzzletron.plugins.automodel import validation


def test_realized_validation_propagates_multimodal_data_contract(tmp_path, monkeypatch):
    teacher = tmp_path / "teacher"
    candidate = tmp_path / "candidate"
    for checkpoint in (teacher, candidate):
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}\n")

    data_cfg = {
        "path": "/materialized/vlm",
        "revision": "dataset-commit",
        "modality": "multimodal",
        "layout": "padded_varlen",
        "max_sample_length": 512,
    }
    cfg = OmegaConf.create(
        {
            "data": data_cfg,
            "scoring": {
                "eval_samples": 1,
                "micro_batch_size": 1,
                "automodel": {"teacher_cache_device": "cpu"},
            },
        }
    )
    args = OmegaConf.create({"teacher_dir": str(teacher), "eval_samples": 1, "micro_batch_size": 1})
    launches = []

    class Recipe:
        _puzzletron_output_writer = False

        @staticmethod
        def teardown_capture():
            return None

    def run_recipe(recipe, scoring, eval_iters, use_puzzletron_dataloader, data_cfg=None):
        launches.append((recipe, eval_iters, use_puzzletron_dataloader, data_cfg))
        return Recipe()

    monkeypatch.setattr(validation, "apply_patch", lambda: None)
    monkeypatch.setattr(
        validation, "build_solution_recipe_config", lambda _cfg, path: {"path": str(path)}
    )
    monkeypatch.setattr(validation, "TeacherTargetCache", lambda device: object())
    monkeypatch.setattr(validation, "_run_recipe", run_recipe)
    monkeypatch.setattr(validation, "_extract_teacher_targets", lambda *args: None)
    monkeypatch.setattr(validation, "_score_candidate", lambda *args, **kwargs: None)
    monkeypatch.setattr(validation, "_free_scoring_memory", lambda recipe: None)
    monkeypatch.setattr(validation.dist, "barrier", lambda: None)

    validation.validate_realized_checkpoints_automodel(
        cfg,
        args,
        [(1, candidate, {"candidate": 1})],
        tmp_path / "output",
    )

    assert [launch[0]["path"] for launch in launches] == [str(teacher), str(candidate)]
    assert [launch[1] for launch in launches] == [1, 1]
    assert [launch[2] for launch in launches] == [False, False]
    assert [launch[3] for launch in launches] == [data_cfg, data_cfg]
