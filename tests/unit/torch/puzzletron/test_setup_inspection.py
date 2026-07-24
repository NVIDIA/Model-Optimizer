# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused source-normalization tests for the Puzzletron setup wizard."""

import pytest

from puzzletron_setup import SetupError
from puzzletron_setup.inspection import normalize_dataset_source, normalize_model_source


def test_normalizes_hugging_face_web_urls():
    assert normalize_model_source("https://huggingface.co/Qwen/Qwen3.5-0.8B") == (
        "Qwen/Qwen3.5-0.8B"
    )
    assert normalize_model_source("huggingface.com/Qwen/Qwen3.5-0.8B/") == (
        "Qwen/Qwen3.5-0.8B"
    )
    assert normalize_dataset_source(
        "https://huggingface.com/datasets/nvidia/Some-Dataset"
    ) == "nvidia/Some-Dataset"


def test_normalizes_existing_local_paths_and_rejects_other_uris(tmp_path):
    model = tmp_path / "Model"
    dataset = tmp_path / "Dataset"
    model.mkdir()
    dataset.mkdir()

    assert normalize_model_source(str(model)) == str(model.resolve())
    assert normalize_dataset_source(str(dataset)) == str(dataset.resolve())
    with pytest.raises(SetupError, match="Unsupported model source"):
        normalize_model_source("s3://bucket/model")
    with pytest.raises(SetupError, match="does not exist"):
        normalize_dataset_source("../missing-dataset")
