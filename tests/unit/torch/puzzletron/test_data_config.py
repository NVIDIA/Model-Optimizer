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

"""Tests for Puzzletron data and experiment configuration normalization."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.dataset.config import PuzzletronDataSpec
from modelopt.torch.puzzletron.pipeline_config import (
    adapt_runtime_hydra_config,
    normalize_pipeline_config,
    pipeline_config_from_path,
)
from puzzletron_orchestrator.config import load_experiment_config


def test_canonical_packed_multimodal_data_spec():
    spec = PuzzletronDataSpec.from_mapping(
        {
            "modality": "multimodal",
            "layout": "packed_varlen",
            "max_sample_length": 1536,
            "packing": {
                "pack_size": 2048,
                "packing_ratio": 0.9,
                "drop_long_samples": True,
            },
        }
    )

    assert spec.modality.value == "multimodal"
    assert spec.layout.value == "packed_varlen"
    assert spec.packing.pack_size == 2048
    assert spec.packing.packing_ratio == 0.9


def test_legacy_varlen_has_actionable_migration_error():
    with pytest.raises(ValueError, match="data.layout.*packed_varlen"):
        PuzzletronDataSpec.from_mapping({"varlen": True})


def test_packed_layout_requires_valid_packing_contract():
    with pytest.raises(ValueError, match="data.packing"):
        PuzzletronDataSpec.from_mapping(
            {"modality": "text", "layout": "packed_varlen", "max_sample_length": 128}
        )
    with pytest.raises(ValueError, match="packing_ratio"):
        PuzzletronDataSpec.from_mapping(
            {
                "modality": "text",
                "layout": "packed_varlen",
                "max_sample_length": 128,
                "packing": {"pack_size": 128, "packing_ratio": 1.1},
            }
        )


def test_fixed_layout_does_not_require_packing():
    spec = PuzzletronDataSpec.from_mapping(
        {"modality": "text", "layout": "fixed", "max_sample_length": 512}
    )
    assert spec.packing is None


def test_pipeline_boundary_rejects_stage_local_legacy_varlen():
    with pytest.raises(ValueError, match="pruning.varlen.*data.layout"):
        normalize_pipeline_config(
            {
                "data": {
                    "modality": "text",
                    "layout": "fixed",
                    "max_sample_length": 32,
                },
                "pruning": {"varlen": False},
            }
        )


def test_pipeline_defaults_sort_sanity_to_include_reverse():
    canonical = normalize_pipeline_config({})

    assert canonical["sort_sanity"]["include_reverse"] is True


def test_pipeline_preserves_explicit_reverse_sort_opt_out():
    canonical = normalize_pipeline_config({"sort_sanity": {"include_reverse": False}})

    assert canonical["sort_sanity"]["include_reverse"] is False


@pytest.mark.parametrize(
    "loader",
    [pipeline_config_from_path, load_experiment_config],
    ids=["pipeline", "controller"],
)
@pytest.mark.parametrize(
    ("legacy_path", "canonical_section", "canonical_key", "legacy", "canonical", "preferred"),
    [
        ("puzzle_dir", "experiment", "dir", "first", "second", "puzzle_dir"),
        (
            "input_hf_model_path",
            "model",
            "source",
            "model-a",
            "model-b",
            "input_hf_model_path",
        ),
        ("teacher_dir", "convert", "teacher_dir", "first", "second", "teacher_dir"),
        ("dataset_path", "data", "path", "first", "second", "dataset_path"),
        (
            "trust_remote_code",
            "model",
            "trust_remote_code",
            False,
            True,
            "model.trust_remote_code",
        ),
    ],
)
def test_pipeline_and_controller_loaders_reject_conflicting_compatibility_aliases(
    tmp_path: Path,
    loader,
    legacy_path: str,
    canonical_section: str,
    canonical_key: str,
    legacy,
    canonical,
    preferred: str,
) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    legacy_path: legacy,
                    canonical_section: {canonical_key: canonical},
                }
            )
        )
    )

    with pytest.raises(ValueError, match=rf"override '{preferred}'.*stay synchronized"):
        loader(experiment)


@pytest.mark.parametrize(
    "override",
    ["runtime_annotations.reason=ad-hoc", "++runtime_annotations.reason=ad-hoc"],
)
def test_pipeline_and_controller_loaders_apply_overrides_with_parity(
    tmp_path: Path,
    override: str,
) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("experiment:\n  dir: run\n")

    pipeline = pipeline_config_from_path(experiment, overrides=[override])
    controller = load_experiment_config(experiment, overrides=[override])

    assert (
        pipeline["runtime_annotations"] == controller["runtime_annotations"] == {"reason": "ad-hoc"}
    )


def test_runtime_adapter_derives_legacy_loader_fields_from_canonical_data():
    canonical = normalize_pipeline_config(
        {
            "experiment": {"dir": "/tmp/run"},
            "data": {
                "modality": "multimodal",
                "layout": "packed_varlen",
                "max_sample_length": 24,
                "packing": {"pack_size": 32, "packing_ratio": 0.9},
            },
            "pruning": {},
            "scoring": {},
            "realize_model": {},
        }
    )
    runtime = adapt_runtime_hydra_config(
        OmegaConf.create({"pruning": {}, "scoring": {}, "realize_model": {}}),
        canonical,
    )

    assert runtime.pruning.varlen is True
    assert runtime.pruning.block_size == 32
    assert runtime.scoring.varlen is True
    assert runtime.realize_model.varlen is True


def test_runtime_adapter_routes_inferred_descriptor_to_all_legacy_stage_sections():
    canonical = normalize_pipeline_config(
        {
            "experiment": {"dir": "/tmp/run"},
            "model": {"source": "/checkpoint", "force_hf": False},
            "pruning": {},
            "scoring": {},
            "realize_model": {},
        }
    )
    canonical["_runtime"] = {"descriptor": "qwen3_5"}

    runtime = adapt_runtime_hydra_config(
        OmegaConf.create({"pruning": {}, "scoring": {}, "realize_model": {}}),
        canonical,
    )

    assert runtime.descriptor == "qwen3_5"
    assert runtime.pruning.descriptor == "qwen3_5"
    assert runtime.scoring.descriptor == "qwen3_5"
    assert runtime.realize_model.descriptor == "qwen3_5"
