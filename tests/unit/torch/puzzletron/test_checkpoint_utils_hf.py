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

from types import SimpleNamespace

import torch

from modelopt.torch.puzzletron.tools import checkpoint_utils_hf as cuhf


def test_save_checkpoint_uses_descriptor_language_model_config(tmp_path, monkeypatch):
    calls = {}

    class Descriptor:
        @staticmethod
        def get_language_model_config(config):
            return config.text_config

        @staticmethod
        def get_weight_groups(layer_names, num_hidden_layers):
            calls["num_hidden_layers"] = num_hidden_layers
            return {"weights": list(layer_names)}

        @staticmethod
        def output_embedding_name():
            return "lm_head"

    monkeypatch.setattr(cuhf, "save_model_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(cuhf, "save_subblocks", lambda *args, **kwargs: None)

    cfg = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=7),
        tie_word_embeddings=False,
    )
    cuhf._save_checkpoint(cfg, {"some.weight": torch.zeros(1)}, tmp_path, Descriptor)

    assert calls["num_hidden_layers"] == 7


def test_copy_auto_map_code_files_copies_valid_local_code_references(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    checkpoint_dir = tmp_path / "checkpoint"
    source_dir.mkdir()
    checkpoint_dir.mkdir()
    (source_dir / "modeling_custom.py").write_text("# modeling\n")
    (source_dir / "tokenization_custom.py").write_text("# tokenizer\n")

    monkeypatch.setattr(cuhf.inspect, "getfile", lambda _cls: source_dir / "configuration.py")

    cfg = SimpleNamespace(
        auto_map={
            "AutoConfig": "configuration_custom.CustomConfig",
            "AutoModelForCausalLM": "org/repo--modeling_custom.CustomModel",
            "AutoTokenizer": [None, "tokenization_custom.CustomTokenizer"],
        }
    )

    cuhf._copy_auto_map_code_files(cfg, checkpoint_dir)

    assert (checkpoint_dir / "modeling_custom.py").exists()
    assert (checkpoint_dir / "tokenization_custom.py").exists()
