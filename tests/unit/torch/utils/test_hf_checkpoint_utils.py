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

"""Tests for modelopt/torch/utils/plugins/hf_checkpoint_utils.py"""

import json
import sys
import warnings
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from safetensors.torch import save_file

from modelopt.torch.utils.plugins.hf_checkpoint_utils import (
    copy_off_index_safetensors,
    indexed_weight_map,
    off_index_safetensors_files,
    read_safetensors_subset,
)

pytest.importorskip("huggingface_hub")
hf_hub_errors = pytest.importorskip("huggingface_hub.errors")
LocalEntryNotFoundError = hf_hub_errors.LocalEntryNotFoundError

from modelopt.torch.utils.mlflow import EXPERIMENT_JSON
from modelopt.torch.utils.plugins import hf_checkpoint_utils
from modelopt.torch.utils.plugins.hf_checkpoint_utils import (
    copy_hf_ckpt_remote_code,
    copy_non_safetensor_files_from_ckpt,
    sanitize_hf_config_for_deployment,
)


def test_copy_non_safetensor_files_from_ckpt_supports_additional_exclusions(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "model.safetensors").write_text("weights")
    (src_dir / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    (src_dir / "pytorch_model.bin").write_text("weights")
    (src_dir / "stats.npy").write_text("stats")
    (src_dir / "reasoning_parser.py").write_text("parser")

    default_dst = tmp_path / "default"
    copy_non_safetensor_files_from_ckpt(src_dir, default_dst)
    assert not (default_dst / "model.safetensors").exists()
    assert not (default_dst / "model.safetensors.index.json").exists()
    assert (default_dst / "pytorch_model.bin").exists()
    assert (default_dst / "stats.npy").exists()

    filtered_dst = tmp_path / "filtered"
    copy_non_safetensor_files_from_ckpt(
        src_dir,
        filtered_dst,
        exclude_patterns=("*.bin", "*.npy"),
    )
    assert (filtered_dst / "reasoning_parser.py").exists()
    assert not (filtered_dst / "pytorch_model.bin").exists()
    assert not (filtered_dst / "stats.npy").exists()


def test_copy_non_safetensor_files_from_ckpt_continues_after_copy_failure(tmp_path, monkeypatch):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "bad.py").write_text("bad")
    (src_dir / "good.py").write_text("good")

    original_copy2 = hf_checkpoint_utils.shutil.copy2

    def copy2(source, *args, **kwargs):
        if source.endswith("bad.py"):
            raise PermissionError("unreadable")
        return original_copy2(source, *args, **kwargs)

    monkeypatch.setattr(hf_checkpoint_utils.shutil, "copy2", copy2)
    with pytest.warns(UserWarning, match="bad.py"):
        copied_files = copy_non_safetensor_files_from_ckpt(src_dir, tmp_path / "dst")

    assert copied_files == ["good.py"]


def test_copy_non_model_files_continues_after_copy_failure(tmp_path, monkeypatch):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "bad.py").write_text("bad")
    (src_dir / "good.py").write_text("good")

    original_copy2 = hf_checkpoint_utils.shutil.copy2

    def copy2(source, *args, **kwargs):
        if str(source).endswith("bad.py"):
            raise PermissionError("unreadable")
        return original_copy2(source, *args, **kwargs)

    monkeypatch.setattr(hf_checkpoint_utils.shutil, "copy2", copy2)
    with pytest.warns(UserWarning, match="bad.py"):
        copied_files = hf_checkpoint_utils.copy_non_model_files(src_dir, tmp_path / "dst")

    assert copied_files == ["good.py"]


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
        "modelopt.torch.utils.plugins.hf_checkpoint_utils.snapshot_download",
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
        "modelopt.torch.utils.plugins.hf_checkpoint_utils.snapshot_download",
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
            "modelopt.torch.utils.plugins.hf_checkpoint_utils.snapshot_download",
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


# --- off-index safetensors: files model loading never opens --------------------------------------


def _shard(path, name):
    (path / name).write_text("tensors")


def test_no_index_means_only_model_safetensors_is_read(tmp_path):
    _shard(tmp_path, "model.safetensors")
    assert hf_checkpoint_utils.off_index_safetensors_files(tmp_path) == []


def test_a_standalone_sidecar_is_off_index(tmp_path):
    """GLM-4.7 ships its MTP head as mtp.safetensors, which the loader never opens."""
    _shard(tmp_path, "model.safetensors")
    _shard(tmp_path, "mtp.safetensors")
    assert hf_checkpoint_utils.off_index_safetensors_files(tmp_path) == ["mtp.safetensors"]


def test_indexed_shards_are_read_and_sidecars_are_not(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00002.safetensors",'
        ' "b": "model-00002-of-00002.safetensors"}}'
    )
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        _shard(tmp_path, name)
    _shard(tmp_path, "mtp.safetensors")

    assert hf_checkpoint_utils.off_index_safetensors_files(tmp_path) == ["mtp.safetensors"]


@pytest.mark.parametrize(
    "index",
    [
        '{"weight_map": {}}',
        "{}",
        '{"weight_map": {"a": "model-00001-of-00002.safetensors"}}',
    ],
    ids=["empty-map", "no-map-key", "partial-map"],
)
def test_main_weight_shards_are_never_off_index_whatever_the_index_says(tmp_path, index):
    """An empty, partial or malformed index must not make the real weights look like sidecars.

    Copying those into an export would leave the unquantized source weights sitting beside the
    quantized ones -- a checkpoint that loads and is silently wrong.
    """
    (tmp_path / "model.safetensors.index.json").write_text(index)
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        _shard(tmp_path, name)

    assert hf_checkpoint_utils.off_index_safetensors_files(tmp_path) == []


def test_unsharded_main_weights_are_never_off_index(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    _shard(tmp_path, "model.safetensors")
    assert hf_checkpoint_utils.off_index_safetensors_files(tmp_path) == []


def test_results_are_sorted(tmp_path):
    _shard(tmp_path, "model.safetensors")
    for name in ("zeta.safetensors", "alpha.safetensors", "mtp.safetensors"):
        _shard(tmp_path, name)
    assert hf_checkpoint_utils.off_index_safetensors_files(tmp_path) == [
        "alpha.safetensors",
        "mtp.safetensors",
        "zeta.safetensors",
    ]


def test_a_missing_directory_is_not_an_error(tmp_path):
    assert hf_checkpoint_utils.off_index_safetensors_files(tmp_path / "nope") == []


def test_copy_moves_only_the_sidecars_and_preserves_bytes(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00001.safetensors"}}'
    )
    _shard(src, "model-00001-of-00001.safetensors")
    (src / "mtp.safetensors").write_text("mtp-bytes")

    assert hf_checkpoint_utils.copy_off_index_safetensors(src, dst) == ["mtp.safetensors"]
    assert (dst / "mtp.safetensors").read_text() == "mtp-bytes"
    # the real weights are the export's job, not a verbatim copy
    assert not (dst / "model-00001-of-00001.safetensors").exists()


def test_copy_does_not_overwrite_what_the_export_already_wrote(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    _shard(src, "model.safetensors")
    (src / "mtp.safetensors").write_text("source")
    (dst / "mtp.safetensors").write_text("already exported")

    assert hf_checkpoint_utils.copy_off_index_safetensors(src, dst) == []
    assert (dst / "mtp.safetensors").read_text() == "already exported"


def _write_st(path, tensors):
    """Minimal real safetensors file so header reads work."""
    save_file({k: torch.zeros(1) for k in tensors}, str(path))


def test_off_index_skips_mistral_consolidated_copy(tmp_path):
    """Mistral ships consolidated.safetensors: a SECOND full copy of the indexed weights.

    Copying it into an export puts unquantized weights beside the quantized ones, and vLLM's
    mistral load-format looks for that filename specifically -- so it can be served instead of
    what we quantized.
    """

    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}'
    )
    _write_st(tmp_path / "model-00001-of-00001.safetensors", ["a.weight"])
    _write_st(tmp_path / "consolidated.safetensors", ["a.weight"])

    assert off_index_safetensors_files(tmp_path) == []


def test_off_index_skips_peft_adapter(tmp_path):
    """A PEFT adapter's tensor names do NOT overlap the index, so only the name rule catches it."""

    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}'
    )
    _write_st(tmp_path / "model-00001-of-00001.safetensors", ["a.weight"])
    _write_st(tmp_path / "adapter_model.safetensors", ["base_model.a.lora_A.weight"])

    assert off_index_safetensors_files(tmp_path) == []


def test_off_index_skips_unknown_name_that_reships_indexed_weights(tmp_path):
    """The name rules only know the conventions we have seen; overlap catches the rest."""

    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a.weight": "model-00001-of-00001.safetensors",'
        ' "b.weight": "model-00001-of-00001.safetensors"}}'
    )
    _write_st(tmp_path / "model-00001-of-00001.safetensors", ["a.weight", "b.weight"])
    _write_st(tmp_path / "backup-copy.safetensors", ["a.weight", "b.weight"])

    assert off_index_safetensors_files(tmp_path) == []


def test_off_index_still_keeps_a_genuine_mtp_sidecar(tmp_path):
    """The whole point: a real sidecar holds names the index does NOT have, and must be kept."""

    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}'
    )
    _write_st(tmp_path / "model-00001-of-00001.safetensors", ["a.weight"])
    _write_st(tmp_path / "mtp.safetensors", ["model.mtp.eh_proj.weight"])

    assert off_index_safetensors_files(tmp_path) == ["mtp.safetensors"]


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX symlink support")
def test_copies_a_symlinked_sidecar_from_a_hub_cache_layout(tmp_path):
    """A hub-downloaded checkpoint stores EVERY file as a symlink into ``../../blobs/<sha>``.

    Rejecting symlinks outright therefore skips the sidecar of every checkpoint loaded by hub id
    -- including the GLM-4.7 ``mtp.safetensors`` this path exists to carry -- which is the
    silent-missing-MTP failure the carry-over was written to prevent. The guard must look at what
    the link resolves to, not at whether it is a link.
    """
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "deadbeef"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)

    save_file({"model.mtp.eh_proj.weight": torch.zeros(1)}, str(blobs / "sha123"))
    save_file({"a.weight": torch.zeros(1)}, str(blobs / "sha456"))
    (snapshot / "mtp.safetensors").symlink_to("../../blobs/sha123")
    (snapshot / "model-00001-of-00001.safetensors").symlink_to("../../blobs/sha456")
    (snapshot / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}'
    )

    assert off_index_safetensors_files(snapshot) == ["mtp.safetensors"]

    dst = tmp_path / "export"
    dst.mkdir()
    assert copy_off_index_safetensors(snapshot, dst) == ["mtp.safetensors"]
    assert (dst / "mtp.safetensors").is_file()


def _indexed_ckpt(src):
    src.mkdir(parents=True, exist_ok=True)
    save_file({"a.weight": torch.zeros(1)}, str(src / "model-00001-of-00001.safetensors"))
    (src / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}'
    )
    return src


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX symlink support")
def test_skips_a_sidecar_whose_link_dangles(tmp_path):
    """A link to nothing copies nothing rather than raising out of the export."""
    src = _indexed_ckpt(tmp_path / "ckpt")
    (src / "mtp.safetensors").symlink_to(tmp_path / "does-not-exist")

    dst = tmp_path / "export"
    dst.mkdir()
    with pytest.warns(UserWarning, match="not a readable regular file"):
        assert copy_off_index_safetensors(src, dst) == []
    assert not (dst / "mtp.safetensors").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX symlink support")
def test_skips_a_sidecar_pointing_outside_the_checkpoint(tmp_path):
    """The original hardening, restored: a link out of the tree is refused, not followed.

    Accepting hub blobs must not mean accepting any target at all -- a checkpoint shipping
    ``mtp.safetensors -> /etc/passwd`` would otherwise land that file in the export under a name
    that looks like model weights.
    """
    outside = tmp_path / "secret.txt"
    outside.write_text("not model weights")
    src = _indexed_ckpt(tmp_path / "ckpt")
    (src / "mtp.safetensors").symlink_to(outside)

    dst = tmp_path / "export"
    dst.mkdir()
    with pytest.warns(UserWarning, match="outside the checkpoint directory"):
        assert copy_off_index_safetensors(src, dst) == []
    assert not (dst / "mtp.safetensors").exists()


# --- indexed_weight_map / read_safetensors_subset (moved from model_load_utils.py, which used
# to duplicate indexed_weight_map's own index/single-file logic) --------------------------------


def test_indexed_weight_map_sharded(tmp_path):
    save_file({"a.weight": torch.zeros(2)}, str(tmp_path / "shard1.safetensors"))
    save_file({"b.weight": torch.zeros(2)}, str(tmp_path / "shard2.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"a.weight": "shard1.safetensors", "b.weight": "shard2.safetensors"}}
        )
    )

    assert indexed_weight_map(str(tmp_path)) == {
        "a.weight": "shard1.safetensors",
        "b.weight": "shard2.safetensors",
    }


def test_indexed_weight_map_single_file(tmp_path):
    save_file(
        {"a.weight": torch.zeros(2), "b.weight": torch.zeros(2)},
        str(tmp_path / "model.safetensors"),
    )

    assert indexed_weight_map(str(tmp_path)) == {
        "a.weight": "model.safetensors",
        "b.weight": "model.safetensors",
    }


def test_indexed_weight_map_missing_returns_empty(tmp_path):
    """Neither an index nor a single-file checkpoint: {}, not an exception.

    Right for indexed_weight_map's own callers (e.g. locate_source_keys), which treat "nothing
    recorded" as legitimate. Callers for whom a missing checkpoint is a genuine error (FSDP2
    parallel loading, the structural unplaced-keys fallback, DFlash's precision reload) check for
    the empty result and raise themselves.
    """
    assert indexed_weight_map(str(tmp_path)) == {}


def test_read_safetensors_subset(tmp_path):
    save_file(
        {"a.weight": torch.tensor([1.0, 2.0]), "a.bias": torch.tensor([3.0])},
        str(tmp_path / "shard1.safetensors"),
    )
    save_file({"b.weight": torch.tensor([4.0])}, str(tmp_path / "shard2.safetensors"))
    weight_map = {
        "a.weight": "shard1.safetensors",
        "a.bias": "shard1.safetensors",
        "b.weight": "shard2.safetensors",
    }

    result = read_safetensors_subset(str(tmp_path), weight_map, lambda n: n.startswith("a."))

    assert set(result.keys()) == {"a.weight", "a.bias"}
    assert torch.equal(result["a.weight"], torch.tensor([1.0, 2.0]))
    assert torch.equal(result["a.bias"], torch.tensor([3.0]))


def test_copy_non_model_files_copies_everything_the_export_does_not_own(tmp_path):
    source_dir = tmp_path / "source"
    export_dir = tmp_path / "export"
    source_dir.mkdir()
    export_dir.mkdir()

    source_files = {
        "super_v3_reasoning_parser.py": "class Parser: pass\n",
        "modeling_custom.py": "class Model: pass\n",
        "README.md": "# Source model\n",
        "LICENSE": "license text\n",
        ".gitattributes": "*.safetensors filter=lfs\n",
        "chat_template.jinja": "{{ messages }}\n",
        "tokenizer_config.json": '{"chat_template": "source"}\n',
        "assets/audio/config.json": '{"nested": "config"}\n',
        "generation_config.json": '{"source": "generation"}\n',
        "config.json": '{"source": "config"}\n',
        "hf_quant_config.json": '{"source": "quant"}\n',
        "quant_config.json": '{"source": "stale quant"}\n',
        "quantize_config.json": '{"source": "stale quant"}\n',
        "recipe.yaml": "quantize: {}\n",
        ".experiment.json": '{"run_id": "source"}\n',
        "model.safetensors.index.json": '{"weight_map": {}}\n',
        "model-00001-of-00001.safetensors": "source weights\n",
        "mtp.safetensors": "off-index weights\n",
        "pytorch_model.bin": "source weights\n",
        "original/consolidated.00.pth": "source weights\n",
        ".git/lfs/objects/blob": "source weights\n",
        ".cache/huggingface/download/README.md.metadata": "download state\n",
    }
    for file_name, contents in source_files.items():
        (source_dir / file_name).parent.mkdir(parents=True, exist_ok=True)
        (source_dir / file_name).write_text(contents)

    export_files = {
        "config.json": '{"export": "config"}\n',
        "generation_config.json": '{"export": "generation"}\n',
    }
    for file_name, contents in export_files.items():
        (export_dir / file_name).write_text(contents)

    copied = hf_checkpoint_utils.copy_non_model_files(str(source_dir), str(export_dir))

    expected = [
        ".gitattributes",
        "LICENSE",
        "README.md",
        "assets/audio/config.json",
        "chat_template.jinja",
        "modeling_custom.py",
        "super_v3_reasoning_parser.py",
        "tokenizer_config.json",
    ]
    assert sorted(copied) == expected
    for file_name in expected:
        assert (export_dir / file_name).read_text() == source_files[file_name]
    for file_name, contents in export_files.items():
        assert (export_dir / file_name).read_text() == contents
    exported = {str(p.relative_to(export_dir)) for p in export_dir.rglob("*") if p.is_file()}
    assert exported == {*expected, *export_files}


def test_ensure_local_checkpoint_returns_a_local_dir_untouched(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise AssertionError("a local checkpoint must not hit the hub")

    monkeypatch.setattr(hf_checkpoint_utils, "snapshot_download", fail)
    assert hf_checkpoint_utils.ensure_local_checkpoint(str(tmp_path)) == (None, str(tmp_path))


class _FakeHfApi:
    """``HfApi`` whose ``repo_info`` raises ``error``, or succeeds when it is None."""

    error: Exception | None = None

    def repo_info(self, repo_id):
        if self.error is not None:
            raise self.error


def _fake_hub(monkeypatch, tmp_path, error=None):
    # tests/unit/conftest.py sets HF_HUB_OFFLINE for the session; these tests model a live Hub.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    calls = []

    def fake_snapshot_download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(hf_checkpoint_utils, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hf_checkpoint_utils, "HfApi", type("Api", (_FakeHfApi,), {"error": error}))
    return calls


def test_ensure_local_checkpoint_fetches_a_hub_id_in_full(monkeypatch, tmp_path):
    calls = _fake_hub(monkeypatch, tmp_path)

    assert hf_checkpoint_utils.ensure_local_checkpoint("org/model") == ("org/model", str(tmp_path))
    assert calls == [(("org/model",), {})]


def test_ensure_local_checkpoint_warns_when_only_the_cache_is_left(monkeypatch, tmp_path):
    """snapshot_download falls back to the cache silently; the fallback must be visible."""
    calls = _fake_hub(monkeypatch, tmp_path, error=ConnectionError("no route to host"))

    with pytest.warns(UserWarning, match="cached snapshot"):
        result = hf_checkpoint_utils.ensure_local_checkpoint("org/model")

    assert result == ("org/model", str(tmp_path))
    assert calls == [(("org/model",), {"local_files_only": True})]


def test_ensure_local_checkpoint_uses_the_cache_quietly_when_offline(monkeypatch, tmp_path):
    """HF_HUB_OFFLINE is the user's choice: do not ask the Hub, and do not warn on every run."""
    calls = _fake_hub(monkeypatch, tmp_path, error=AssertionError("the Hub must not be asked"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = hf_checkpoint_utils.ensure_local_checkpoint("org/model")

    assert result == ("org/model", str(tmp_path))
    assert calls == [(("org/model",), {"local_files_only": True})]


def test_ensure_local_checkpoint_offline_and_uncached_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def not_cached(*args, **kwargs):
        raise hf_hub_errors.LocalEntryNotFoundError("not in cache")

    monkeypatch.setattr(hf_checkpoint_utils, "snapshot_download", not_cached)

    with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE is set"):
        hf_checkpoint_utils.ensure_local_checkpoint("org/model")


def test_ensure_local_checkpoint_names_the_hub_error_when_nothing_is_cached(monkeypatch, tmp_path):
    _fake_hub(monkeypatch, tmp_path, error=ConnectionError("no route to host"))

    def not_cached(*args, **kwargs):
        raise hf_hub_errors.LocalEntryNotFoundError("not in cache")

    monkeypatch.setattr(hf_checkpoint_utils, "snapshot_download", not_cached)

    with pytest.raises(RuntimeError, match=r"no route to host.*not in the local cache"):
        hf_checkpoint_utils.ensure_local_checkpoint("org/model")


def test_ensure_local_checkpoint_raises_for_a_missing_repo(monkeypatch, tmp_path):
    # Its constructor wants an HTTP response; only the type matters here.
    missing = hf_hub_errors.RepositoryNotFoundError.__new__(hf_hub_errors.RepositoryNotFoundError)
    calls = _fake_hub(monkeypatch, tmp_path, error=missing)

    with pytest.raises(hf_hub_errors.RepositoryNotFoundError):
        hf_checkpoint_utils.ensure_local_checkpoint("org/typo")
    assert calls == []


def test_copy_non_model_files_never_fetches_from_the_hub(monkeypatch, tmp_path):
    """Exporters read the source from local disk only; a Hub ID is the caller's to download."""

    def fail(*args, **kwargs):
        raise AssertionError("copy_non_model_files must not download")

    monkeypatch.setattr(hf_checkpoint_utils, "snapshot_download", fail)

    with pytest.raises(ValueError, match="ensure_local_checkpoint"):
        hf_checkpoint_utils.copy_non_model_files("org/model", tmp_path / "export")


def test_experiment_json_is_export_owned():
    """copy_non_model_files copies dotfiles, so without this the source checkpoint's MLflow pointer
    would follow it into every derived checkpoint."""
    assert EXPERIMENT_JSON in hf_checkpoint_utils._EXPORT_OWNED_FILES


def test_copy_non_model_files_follows_links_only_within_the_checkpoint(tmp_path):
    src_dir, export = tmp_path / "src", tmp_path / "export"
    src_dir.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("host file")
    (src_dir / "README.md").write_text("readme")
    (src_dir / "LICENSE").symlink_to(src_dir / "README.md")
    (src_dir / "notes.txt").symlink_to(outside)
    (src_dir / "gone.txt").symlink_to(tmp_path / "missing.txt")

    with pytest.warns(UserWarning) as record:
        copied = hf_checkpoint_utils.copy_non_model_files(src_dir, export)

    assert sorted(copied) == ["LICENSE", "README.md"]
    assert (export / "LICENSE").read_text() == "readme"
    assert not (export / "notes.txt").exists()
    assert not (export / "gone.txt").exists()
    warned = " ".join(str(w.message) for w in record)
    assert "notes.txt" in warned and "gone.txt" in warned


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX symlink support")
def test_copy_non_model_files_follows_hub_snapshot_links_into_blobs(tmp_path):
    repo = tmp_path / "models--org--model"
    snapshot, blobs = repo / "snapshots" / "abc123", repo / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (blobs / "sha-tokenizer").write_text("{}")
    (snapshot / "tokenizer.json").symlink_to("../../blobs/sha-tokenizer")

    copied = hf_checkpoint_utils.copy_non_model_files(snapshot, tmp_path / "export")

    assert copied == ["tokenizer.json"]
    assert (tmp_path / "export" / "tokenizer.json").read_text() == "{}"


def _ensure_local_checkpoint_job(tmp_path, fail, group_ranks, rank, size):
    import os

    import torch.distributed as dist

    from modelopt.torch.utils.plugins import hf_checkpoint_utils as utils

    snapshot = tmp_path / "snapshot"

    def fake_snapshot_download(repo_id, **kwargs):
        (tmp_path / f"downloaded-by-rank{rank}").write_text(repo_id)
        if fail:
            raise OSError("hub unreachable")
        snapshot.mkdir(exist_ok=True)
        return str(snapshot)

    utils.snapshot_download = fake_snapshot_download
    utils.HfApi = _FakeHfApi
    os.environ.pop("HF_HUB_OFFLINE", None)  # inherited from tests/unit/conftest.py
    # new_group is collective over every rank, members or not.
    group = None if group_ranks is None else dist.new_group(group_ranks, backend="gloo")
    if group_ranks is not None and rank not in group_ranks:
        return
    if fail:
        with pytest.raises(RuntimeError, match="hub unreachable"):
            utils.ensure_local_checkpoint("org/model", group=group)
    else:
        result = utils.ensure_local_checkpoint("org/model", group=group)
        assert result == ("org/model", str(snapshot))


def _ensure_local_checkpoint_local_path_job(tmp_path, hidden_from_rank1, rank, size):
    import os

    from modelopt.torch.utils.plugins import hf_checkpoint_utils as utils

    def fail(*args, **kwargs):
        raise AssertionError("a local path must not be downloaded")

    utils.snapshot_download = fail
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    if hidden_from_rank1 and rank == 1:
        # As if the directory sat on rank 0's node-local disk only.
        isdir = os.path.isdir
        utils.os.path.isdir = lambda path: path != str(checkpoint) and isdir(path)
        with pytest.raises(RuntimeError, match="this rank cannot see"):
            utils.ensure_local_checkpoint(str(checkpoint))
    else:
        assert utils.ensure_local_checkpoint(str(checkpoint)) == (None, str(checkpoint))


@pytest.mark.parametrize("hidden_from_rank1", [False, True])
def test_ensure_local_checkpoint_lets_rank_0_resolve_a_local_path(tmp_path, hidden_from_rank1):
    """Ranks that each checked a local path could disagree about it and strand one another in the
    collective; rank 0 decides, and a rank that cannot see its answer raises instead."""
    job = partial(_ensure_local_checkpoint_local_path_job, tmp_path, hidden_from_rank1)
    spawn_multiprocess_job(2, job, backend="gloo")


@pytest.mark.parametrize(
    ("fail", "group_ranks", "downloader"),
    [
        (False, None, "rank0"),
        (True, None, "rank0"),
        (False, [0, 1], "rank0"),
        # Rank 0 of the group, not of the world: global rank 1.
        (False, [1], "rank1"),
    ],
)
def test_ensure_local_checkpoint_downloads_once_per_group(tmp_path, fail, group_ranks, downloader):
    """Rank 0 of the group downloads once and every member gets its result -- including its
    failure, so the waiting ranks raise instead of hanging."""
    job = partial(_ensure_local_checkpoint_job, tmp_path, fail, group_ranks)
    spawn_multiprocess_job(2, job, backend="gloo")
    assert [p.name for p in tmp_path.glob("downloaded-by-*")] == [f"downloaded-by-{downloader}"]
