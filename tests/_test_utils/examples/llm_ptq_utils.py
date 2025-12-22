# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib.metadata as metadata
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest
import torch

PTQ_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "llm_ptq"


@dataclass
class PTQCommand:
    quant: str
    tasks: str = "quant"
    calib: int = 16
    sparsity: str | None = None
    kv_cache_quant: str | None = None
    trust_remote_code: bool = False
    calib_batch_size: int | None = None
    auto_quantize_bits: float | None = None
    tp: int | None = None
    pp: int | None = None
    min_sm: int | None = None
    max_sm: int | None = None
    min_gpu: int | None = None
    batch: int | None = None

    def run(self, model_path: str):
        if self.min_sm and torch.cuda.get_device_capability() < (
            self.min_sm // 10,
            self.min_sm % 10,
        ):
            pytest.skip(reason=f"Requires sm{self.min_sm} or higher")
            return

        if self.max_sm and torch.cuda.get_device_capability() > (
            self.max_sm // 10,
            self.max_sm % 10,
        ):
            pytest.skip(reason=f"Requires sm{self.max_sm} or lower")
            return

        if self.min_gpu and torch.cuda.device_count() < self.min_gpu:
            pytest.skip(reason=f"Requires at least {self.min_gpu} GPUs")
            return

        param_dict = asdict(self)

        param_dict.pop("min_sm", None)
        param_dict.pop("min_gpu", None)

        trust_remote_code = param_dict.pop("trust_remote_code", False)

        args = ["--model", model_path]
        for key, value in param_dict.items():
            if value is not None:
                args.append(f"--{key}")
                args.append(f"{value}")

        if trust_remote_code:
            args.append("--trust_remote_code")

        self.command = ["scripts/huggingface_example.sh", "--no-verbose", *args]
        subprocess.run(self.command, cwd=PTQ_EXAMPLE_DIR, check=True)

    def param_str(self):
        param_dict = asdict(self)
        param_dict.pop("trust_remote_code", False)
        return "_".join(str(value) for value in param_dict.values() if value is not None).replace(
            ",", "_"
        )


class WithRequirements:
    requirements = []

    @pytest.fixture(scope="class", autouse=True)
    def install(self):
        save_deps = []
        for mod, ver in self.requirements:
            try:
                save_ver = metadata.version(mod)
            except metadata.PackageNotFoundError:
                save_ver = None

            save_deps.append((mod, save_ver))

            spec = f"{mod}=={ver}" if ver else mod
            subprocess.run(["pip", "install", spec], check=True)

        yield

        for mod, ver in save_deps:
            if ver:
                subprocess.run(["pip", "install", f"{mod}=={ver}"], check=True)
            else:
                subprocess.run(["pip", "uninstall", "--yes", mod], check=True)
