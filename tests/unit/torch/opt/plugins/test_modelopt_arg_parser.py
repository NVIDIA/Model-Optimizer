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

"""Unit tests for ModelOptArgParser."""

from dataclasses import field
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from modelopt.torch.opt.plugins.transformers import ModelOptArgParser, ModelOptHFArguments


class _ModelArgs(ModelOptHFArguments):
    model_name: str = field(default="test-model", metadata={"help": "The model name."})
    hidden_size: int = field(default=128, metadata={"help": "Hidden size."})


class _TrainArgs(ModelOptHFArguments):
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate."})
    epochs: int = field(default=3, metadata={"help": "Number of epochs."})


class TestModelOptArgParser:
    """Tests for ModelOptArgParser --config and --generate_docs features."""

    def test_cli_args_only(self):
        parser = ModelOptArgParser((_ModelArgs, _TrainArgs))
        model_args, train_args = parser.parse_args_into_dataclasses(
            args=["--model_name", "my-model", "--learning_rate", "0.01"]
        )
        assert model_args.model_name == "my-model"
        assert train_args.learning_rate == 0.01
        assert train_args.epochs == 3  # default

    def test_yaml_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_name: yaml-model\nepochs: 10\n")

        parser = ModelOptArgParser((_ModelArgs, _TrainArgs))
        model_args, train_args = parser.parse_args_into_dataclasses(
            args=["--config", str(config_file)]
        )
        assert model_args.model_name == "yaml-model"
        assert train_args.epochs == 10

    def test_cli_overrides_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_name: yaml-model\nlearning_rate: 0.001\n")

        parser = ModelOptArgParser((_ModelArgs, _TrainArgs))
        model_args, train_args = parser.parse_args_into_dataclasses(
            args=["--config", str(config_file), "--learning_rate", "0.01"]
        )
        assert model_args.model_name == "yaml-model"  # from yaml
        assert train_args.learning_rate == 0.01  # CLI override

    def test_empty_yaml_config(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        parser = ModelOptArgParser((_ModelArgs, _TrainArgs))
        model_args, train_args = parser.parse_args_into_dataclasses(
            args=["--config", str(config_file)]
        )
        assert model_args.model_name == "test-model"  # defaults
        assert train_args.epochs == 3

    def test_generate_docs(self, tmp_path):
        output_path = tmp_path / "ARGUMENTS.md"
        parser = ModelOptArgParser((_ModelArgs, _TrainArgs))

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args_into_dataclasses(args=["--generate_docs", str(output_path)])
        assert exc_info.value.code == 0

        content = output_path.read_text()
        assert "## _ModelArgs" in content
        assert "## _TrainArgs" in content
        assert "--model_name" in content
        assert "--learning_rate" in content
        assert "--epochs" in content

    def test_generate_docs_default_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        parser = ModelOptArgParser((_ModelArgs, _TrainArgs))

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args_into_dataclasses(args=["--generate_docs"])
        assert exc_info.value.code == 0

        content = Path("ARGUMENTS.md").read_text()
        assert "# Argument Reference" in content

    def test_docs_table_format(self, tmp_path):
        output_path = tmp_path / "ARGUMENTS.md"
        parser = ModelOptArgParser((_ModelArgs, _TrainArgs))

        with pytest.raises(SystemExit):
            parser.parse_args_into_dataclasses(args=["--generate_docs", str(output_path)])

        content = output_path.read_text()
        # Check table headers
        assert "| Argument | Type | Default | Description |" in content
        # Check a specific row
        assert "`--model_name`" in content
        assert '`"test-model"`' in content
        assert "The model name." in content
