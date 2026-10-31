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

"""Local inputs and artifact checks for real Megatron Bridge README commands."""

import json
from pathlib import Path

__all__ = [
    "prepare_bridge_full_scale",
    "verify_bridge_hf_export",
]


def prepare_bridge_full_scale(ctx):
    """Set the Bridge working/output directories; callers supply full-scale inputs."""
    ctx.cwd = ctx.repo / "examples/megatron_bridge"
    ctx.env["OUTPUT_ROOT"] = str(ctx.tmp / "outputs")
    Path(ctx.env["OUTPUT_ROOT"]).mkdir()


def verify_bridge_hf_export(path):
    """Check config and weight-file presence for full-scale Bridge HF exports."""
    assert json.loads((path / "config.json").read_text())["model_type"]
    weights = list(path.glob("*.safetensors"))
    assert weights and all(weight.stat().st_size > 0 for weight in weights)
