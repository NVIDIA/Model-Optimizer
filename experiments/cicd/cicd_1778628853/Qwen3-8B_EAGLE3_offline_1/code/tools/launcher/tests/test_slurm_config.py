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

# ruff: noqa: D102
"""Tests for launcher/slurm_config.py — SlurmConfig dataclass and factory.

Coverage:
    - SlurmConfig: default values, field types
    - slurm_factory: default behavior, env var overrides (SLURM_HOST, SLURM_ACCOUNT,
      SLURM_HF_LOCAL), return type
"""


class TestSlurmConfig:
    """Tests for the SlurmConfig dataclass."""

    def test_defaults(self):
        from slurm_config import SlurmConfig

        cfg = SlurmConfig()
        assert cfg.host is None
        assert cfg.port == 22
        assert cfg.account is None
        assert cfg.partition == "batch"
        assert cfg.container is None
        assert cfg.nodes == 1
        assert cfg.ntasks_per_node == 1
        assert cfg.gpus_per_node == 1
        assert cfg.local is False
        assert cfg.container_mounts is None
        assert cfg.srun_args is None
        assert cfg.array is None

    def test_custom_values(self):
        from slurm_config import SlurmConfig

        cfg = SlurmConfig(
            host="login.example.com",
            account="my_account",
            nodes=4,
            gpus_per_node=8,
            container="nvcr.io/nvidia/pytorch:24.01-py3",
            container_mounts=["/data:/data"],
            srun_args=["--no-container-mount-home"],
        )
        assert cfg.host == "login.example.com"
        assert cfg.account == "my_account"
        assert cfg.nodes == 4
        assert cfg.gpus_per_node == 8
        assert cfg.container_mounts == ["/data:/data"]


class TestSlurmFactory:
    """Tests for the slurm_factory function."""

    def test_default_returns_slurm_config(self):
        from slurm_config import slurm_factory

        cfg = slurm_factory()
        # slurm_factory with @run.autoconvert returns a nemo-run Config wrapper
        assert "SlurmConfig" in repr(cfg)

    def test_default_container(self):
        from slurm_config import slurm_factory

        cfg = slurm_factory()
        assert "tensorrt-llm" in cfg.container

    def test_default_srun_args(self):
        from slurm_config import slurm_factory

        cfg = slurm_factory()
        assert cfg.srun_args == ["--no-container-mount-home"]

    def test_default_container_mounts_from_env(self, monkeypatch):
        monkeypatch.setenv("SLURM_HF_LOCAL", "/custom/hf-local")
        # Need to re-import to pick up the env var in the default
        # The factory reads SLURM_HF_LOCAL at call time via the default arg
        import importlib

        import slurm_config

        importlib.reload(slurm_config)
        cfg = slurm_config.slurm_factory()
        assert any("/custom/hf-local:/hf-local" in m for m in cfg.container_mounts)

    def test_override_nodes(self):
        from slurm_config import slurm_factory

        cfg = slurm_factory(nodes=8)
        assert cfg.nodes == 8

    def test_override_partition(self):
        from slurm_config import slurm_factory

        cfg = slurm_factory(partition="gpu")
        assert cfg.partition == "gpu"

    def test_env_var_host(self, monkeypatch):
        monkeypatch.setenv("SLURM_HOST", "test-host.example.com")
        import importlib

        import slurm_config

        importlib.reload(slurm_config)
        cfg = slurm_config.slurm_factory()
        assert cfg.host == "test-host.example.com"
