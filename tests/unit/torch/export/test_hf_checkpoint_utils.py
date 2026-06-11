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

"""Tests for modelopt/torch/export/plugins/hf_checkpoint_utils.py"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("huggingface_hub")
hf_hub_errors = pytest.importorskip("huggingface_hub.errors")
LocalEntryNotFoundError = hf_hub_errors.LocalEntryNotFoundError

from modelopt.torch.export import copy_hf_ckpt_remote_code, sanitize_hf_config_for_deployment


def test_copy_hf_ckpt_remote_code_local_dir(tmp_path):
    """copy_hf_ckpt_remote_code copies top-level .py files from a local directory."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "modeling_custom.py").write_text("# custom model")
    (src_dir / "configuration_custom.py").write_text("# custom config")
    (src_dir / "not_python.txt").write_text("not python")
    (src_dir / "subdir").mkdir()
    (src_dir / "subdir" / "nested.py").write_text("# nested — should not be copied")

    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()

    copy_hf_ckpt_remote_code(src_dir, dst_dir)

    assert (dst_dir / "modeling_custom.py").read_text() == "# custom model"
    assert (dst_dir / "configuration_custom.py").read_text() == "# custom config"
    assert not (dst_dir / "not_python.txt").exists(), "non-.py files should not be copied"
    assert not (dst_dir / "nested.py").exists(), "nested .py files should not be copied"


def test_copy_hf_ckpt_remote_code_local_dir_no_py_files(tmp_path):
    """copy_hf_ckpt_remote_code is a no-op when the local directory has no .py files."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "config.json").write_text("{}")

    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()

    copy_hf_ckpt_remote_code(src_dir, dst_dir)  # should not raise

    assert list(dst_dir.iterdir()) == [], "no files should be copied"


def test_copy_hf_ckpt_remote_code_hub_id(tmp_path, monkeypatch):
    """copy_hf_ckpt_remote_code copies .py files from the resolved Hub snapshot."""
    dst_dir = tmp_path / "dst"
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "modeling_custom.py").write_text("# custom model")
    (snapshot_dir / "not_python.txt").write_text("not python")

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with patch(
        "modelopt.torch.export.plugins.hf_checkpoint_utils.snapshot_download",
        return_value=str(snapshot_dir),
    ) as mock_sd:
        copy_hf_ckpt_remote_code("nvidia/NVIDIA-Nemotron-Nano-12B-v2", dst_dir)

    mock_sd.assert_called_once_with(
        repo_id="nvidia/NVIDIA-Nemotron-Nano-12B-v2",
        allow_patterns=["*.py"],
        local_files_only=False,
    )
    assert (dst_dir / "modeling_custom.py").read_text() == "# custom model"
    assert not (dst_dir / "not_python.txt").exists(), "non-.py files should not be copied"


def test_copy_hf_ckpt_remote_code_hub_id_offline_uses_cache(tmp_path, monkeypatch):
    """copy_hf_ckpt_remote_code resolves cached Hub snapshots when HF_HUB_OFFLINE is set."""
    dst_dir = tmp_path / "dst"
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "nemotron_reasoning_parser.py").write_text("# parser")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with patch(
        "modelopt.torch.export.plugins.hf_checkpoint_utils.snapshot_download",
        return_value=str(snapshot_dir),
    ) as mock_sd:
        copy_hf_ckpt_remote_code("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", dst_dir)

    mock_sd.assert_called_once_with(
        repo_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        allow_patterns=["*.py"],
        local_files_only=True,
    )
    assert (dst_dir / "nemotron_reasoning_parser.py").read_text() == "# parser"


def test_copy_hf_ckpt_remote_code_hub_id_offline_missing_cache_raises(tmp_path, monkeypatch):
    """copy_hf_ckpt_remote_code raises a clear error when offline cache is missing."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with (
        patch(
            "modelopt.torch.export.plugins.hf_checkpoint_utils.snapshot_download",
            side_effect=LocalEntryNotFoundError("missing"),
        ),
        pytest.raises(RuntimeError, match="HF_HUB_OFFLINE"),
    ):
        copy_hf_ckpt_remote_code("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", tmp_path / "dst")


def test_sanitize_hf_config_for_deployment_trims_nextn_layer_types():
    """Drop MTP/next-token-prediction layer types from exported config.json."""
    hidden_layer_types = ["full_attention"] * 45
    nextn_layer_types = ["nextn_predict"] * 3
    config_data = {
        "num_hidden_layers": 45,
        "num_nextn_predict_layers": 3,
        "layer_types": hidden_layer_types + nextn_layer_types,
    }

    with pytest.warns(UserWarning, match="Trimming config.layer_types"):
        sanitize_hf_config_for_deployment(config_data, model=SimpleNamespace())

    assert config_data["layer_types"] == hidden_layer_types


def test_sanitize_hf_config_for_deployment_adds_rope_theta_to_llama3_rope_parameters():
    """Transformers 5.x requires rope_theta inside llama3 rope_parameters."""
    config_data = {
        "rope_theta": 500000,
        "rope_parameters": {
            "rope_type": "llama3",
            "factor": 8.0,
            "original_max_position_embeddings": 4096,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        },
    }

    sanitize_hf_config_for_deployment(config_data, SimpleNamespace(config=SimpleNamespace()))

    assert config_data["rope_parameters"]["rope_theta"] == 500000


def test_sanitize_hf_config_for_deployment_uses_model_rope_theta_for_rope_parameters():
    """Use model.config.rope_theta when save_pretrained omits the top-level field."""
    config_data = {
        "rope_parameters": {
            "rope_type": "llama3",
            "factor": 8.0,
            "original_max_position_embeddings": 4096,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        },
    }
    model = SimpleNamespace(config=SimpleNamespace(rope_theta=500000))

    sanitize_hf_config_for_deployment(config_data, model)

    assert config_data["rope_parameters"]["rope_theta"] == 500000


def test_sanitize_hf_config_for_deployment_adds_rope_theta_to_llama3_rope_scaling():
    """Legacy rope_scaling metadata is normalized for llama3 configs as well."""
    config_data = {
        "rope_scaling": {
            "type": "llama3",
            "factor": 8.0,
            "original_max_position_embeddings": 4096,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        },
    }
    model = SimpleNamespace(config=SimpleNamespace(rope_theta=500000))

    sanitize_hf_config_for_deployment(config_data, model)

    assert config_data["rope_scaling"]["rope_theta"] == 500000


def test_sanitize_hf_config_for_deployment_keeps_existing_rope_theta():
    """Existing rope_theta in rope metadata is not overwritten."""
    config_data = {
        "rope_theta": 500000,
        "rope_parameters": {
            "rope_type": "llama3",
            "rope_theta": 1000000,
        },
    }

    sanitize_hf_config_for_deployment(config_data, SimpleNamespace(config=SimpleNamespace()))

    assert config_data["rope_parameters"]["rope_theta"] == 1000000


def test_sanitize_hf_config_for_deployment_ignores_non_llama3_rope_parameters():
    """Only llama3 RoPE parameters need the Transformers 5.x compatibility fix."""
    config_data = {
        "rope_theta": 500000,
        "rope_parameters": {
            "rope_type": "default",
        },
    }

    sanitize_hf_config_for_deployment(config_data, SimpleNamespace(config=SimpleNamespace()))

    assert "rope_theta" not in config_data["rope_parameters"]


def test_sanitize_hf_config_for_deployment_uses_model_config_nextn_count():
    """Handle exports where save_pretrained omits num_nextn_predict_layers."""
    config_data = {
        "num_hidden_layers": 2,
        "layer_types": ["full_attention", "linear_attention", "nextn_predict"],
    }
    model = SimpleNamespace(config=SimpleNamespace(num_nextn_predict_layers=1))

    with pytest.warns(UserWarning, match="Trimming config.layer_types"):
        sanitize_hf_config_for_deployment(config_data, model=model)

    assert config_data["layer_types"] == ["full_attention", "linear_attention"]


def test_sanitize_hf_config_for_deployment_counts_mtp_layer_prefixes():
    """Do not count broad MTP exclude prefixes as prediction layers."""
    config_data = {
        "num_hidden_layers": 2,
        "layer_types": ["full_attention", "linear_attention", "nextn_predict"],
    }
    model = SimpleNamespace(_mtp_layer_prefixes=["mtp", "mtp.layers.0"])

    with pytest.warns(UserWarning, match="Trimming config.layer_types"):
        sanitize_hf_config_for_deployment(config_data, model=model)

    assert config_data["layer_types"] == ["full_attention", "linear_attention"]


def test_sanitize_hf_config_for_deployment_ignores_broad_mtp_prefix_only():
    """Do not infer prediction-layer count from a broad exclude prefix alone."""
    config_data = {
        "num_hidden_layers": 2,
        "layer_types": ["full_attention", "linear_attention", "nextn_predict"],
    }
    model = SimpleNamespace(_mtp_layer_prefixes=["mtp"])

    sanitize_hf_config_for_deployment(config_data, model=model)

    assert config_data["layer_types"] == ["full_attention", "linear_attention", "nextn_predict"]


def test_sanitize_hf_config_for_deployment_keeps_unexplained_layer_type_mismatch():
    """Do not rewrite config when extra layer types are not explained by nextn metadata."""
    config_data = {
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 1,
        "layer_types": ["full_attention", "linear_attention", "extra_a", "extra_b"],
    }

    sanitize_hf_config_for_deployment(config_data, model=SimpleNamespace())

    assert config_data["layer_types"] == [
        "full_attention",
        "linear_attention",
        "extra_a",
        "extra_b",
    ]
