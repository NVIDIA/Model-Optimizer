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

"""Shared fixtures for Puzzletron CPU unit tests."""

import json
import platform
from pathlib import Path

import pytest

from puzzletron_orchestrator.identity import stable_hash
from puzzletron_orchestrator.stages import semantic_stage_config


@pytest.fixture
def write_terminal_manifest():
    """Return a dependency-light terminal-manifest writer with a success default."""

    def write(
        root: Path,
        stage: str,
        *,
        config: dict[str, object],
        **extra: object,
    ) -> None:
        semantic_config = semantic_stage_config(config, stage)
        semantic_config_identity = stable_hash(semantic_config, prefix=f"{stage}_semantic_cfg")
        capability_snapshot = extra.get("capability_snapshot")
        semantic_identity = stable_hash(
            {
                "stage": stage,
                "semantic_config_identity": semantic_config_identity,
                "capability_snapshot": capability_snapshot,
            },
            prefix=f"{stage}_semantic",
        )
        path = root / "manifests" / f"{stage}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "stage": stage,
                    "status": "success",
                    "semantic_config": semantic_config,
                    "semantic_config_identity": semantic_config_identity,
                    "semantic_identity": semantic_identity,
                    **extra,
                }
            )
            + "\n"
        )

    return write


@pytest.fixture
def write_token_cache():
    """Return a writer for one valid fixed-token cache and its metadata receipt."""

    def write(config: dict, cache: dict) -> dict[str, str]:
        output = Path(cache["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        num_samples = int(cache["num_samples"])
        seq_length = int(cache["seq_length"])
        expected_bytes = num_samples * (seq_length + 1) * 4
        output.write_bytes(bytes(expected_bytes))
        metadata_path = output.with_suffix(output.suffix + ".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "version": 1,
                    "dataset_path": str(Path(config["dataset_path"]).expanduser().resolve()),
                    "tokenizer_path": str(
                        Path(config["convert"]["teacher_dir"]).expanduser().resolve()
                    ),
                    "split": str(cache["split"]),
                    "content_field": str(config["tokenize_data"].get("content_field", "messages")),
                    "num_samples": num_samples,
                    "seq_length": seq_length,
                    "shuffle_seed": int(cache["shuffle_seed"]),
                    "dtype": "uint32",
                    "bytes": expected_bytes,
                }
            )
        )
        return {
            "path": str(output),
            "metadata": str(metadata_path),
            "split": str(cache["split"]),
        }

    return write


# `import fcntl` fails on Windows
def pytest_ignore_collect(collection_path, config):
    return platform.system() == "Windows"
