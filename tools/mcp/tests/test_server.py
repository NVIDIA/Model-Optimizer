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

"""Unit tests for the modelopt-mcp tool schema."""

from modelopt_mcp.server import _build_server


def test_submit_job_exposes_resolved_slurm_list_fields():
    """Cluster inventory lists must be accepted by the typed submit tool."""
    tool = _build_server()._tool_manager.get_tool("submit_job")
    properties = tool.parameters["properties"]

    for field in ("container_mounts", "srun_args"):
        assert properties[field]["anyOf"][0] == {
            "items": {"type": "string"},
            "type": "array",
        }


def test_wait_for_experiment_includes_logs_by_default():
    """Terminal waits should return the final task log without a second tool call."""
    tool = _build_server()._tool_manager.get_tool("wait_for_experiment")
    properties = tool.parameters["properties"]

    assert properties["include_log"]["default"] is True
    assert properties["log_job_idx"]["default"] == 0
