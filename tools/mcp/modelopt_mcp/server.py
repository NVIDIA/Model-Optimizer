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

"""modelopt-mcp server entry point.

Stdio transport; codex / Claude Code launch this as a subprocess and
talk to it over stdin/stdout. See OMNIML-5123 for the design.

Phase 1 tool surface:
  * list_examples       — discover bundled launcher YAMLs
  * verify_setup        — fail-fast probe (docker or slurm)
  * submit_job          — submit a launcher YAML; mode by args
  * job_status          — filesystem-based status
  * job_logs            — filesystem-based logs

All tools return JSON-friendly dicts with explicit ``ok`` / ``reason`` /
``diagnostic`` fields so the calling LLM can route on structured
outcomes instead of free-form prose.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from modelopt_mcp import bridge

logger = logging.getLogger("modelopt_mcp")


def _build_server() -> FastMCP:
    """Construct the MCP server with Phase 1 tools registered.

    Factored out so tests can build an isolated instance without
    stdio plumbing.
    """
    mcp = FastMCP("modelopt")

    @mcp.tool(
        name="list_examples",
        description=(
            "List all bundled launcher YAML examples under "
            "tools/launcher/examples/, with model + description "
            "metadata extracted from each YAML. Use BEFORE submit_job "
            "when you don't know which YAML to launch — this gives the "
            "agent the discovery surface needed to pick one."
        ),
    )
    def list_examples() -> dict:
        return bridge.list_examples_impl()

    @mcp.tool(
        name="verify_setup",
        description=(
            "Probe whether the named executor is reachable from THIS "
            "host. Run BEFORE submit_job to fail fast — the actual "
            "submission burns 30+ seconds (slurm) or starts a Docker "
            "container (docker) before discovering setup is broken.\n\n"
            "Docker mode: checks `docker info` (daemon up) + GPU "
            "passthrough (`docker run --gpus all nvidia-smi`). Set "
            "MODELOPT_MCP_SKIP_GPU_CHECK=1 in the env to skip the GPU "
            "check on CPU-only hosts.\n\n"
            "Slurm mode: ssh -o BatchMode=yes -o ConnectTimeout=5 "
            "to the cluster login node. Refuses to prompt for "
            "password, so key-auth failure is detected immediately."
        ),
    )
    def verify_setup(
        executor: Annotated[
            Literal["docker", "slurm"],
            Field(
                description=(
                    "Which executor to probe: 'docker' for local GPU or 'slurm' for remote cluster."
                )
            ),
        ],
        cluster_host: Annotated[
            str | None,
            Field(description=("Slurm cluster login hostname. Required when executor='slurm'.")),
        ] = None,
        cluster_user: Annotated[
            str | None, Field(description=("SSH user for the cluster. None uses ssh's default."))
        ] = None,
        identity: Annotated[
            str | None,
            Field(
                description=("SSH identity file (-i) override. None uses default key / ssh-agent.")
            ),
        ] = None,
    ) -> dict:
        if executor == "docker":
            return bridge.verify_docker_setup_impl()
        if executor == "slurm":
            if not cluster_host:
                return {
                    "ok": False,
                    "executor": "slurm",
                    "reason": "missing_cluster_host",
                    "diagnostic": ("executor='slurm' requires cluster_host=<hostname>."),
                }
            return bridge.verify_slurm_setup_impl(
                cluster_host=cluster_host,
                cluster_user=cluster_user,
                identity=identity,
            )
        # Pydantic Literal already constrains; this is a defensive fallback.
        return {"ok": False, "reason": "unknown_executor"}

    @mcp.tool(
        name="submit_job",
        description=(
            "Submit a ModelOpt launcher YAML for execution. Mode is "
            "determined by mutually-exclusive args:\n"
            "  - hf_local=<path>      → Docker (local GPU)\n"
            "  - cluster_host=<host>  → Slurm (remote SSH)\n\n"
            "Returns the experiment_id (Slurm) or PID (Docker, "
            "experiment_id captured in Phase 2) immediately; the actual "
            "job runs detached. Poll status via job_status, fetch "
            "output via job_logs.\n\n"
            "Auto-verifies the executor first by default (skip_verify="
            "False is recommended unless you just called verify_setup)."
        ),
    )
    def submit_job(
        yaml_path: Annotated[
            str,
            Field(
                description=(
                    "Launcher YAML to submit. Pass an absolute path, a path "
                    "relative to tools/launcher/examples/, or one of the paths "
                    "returned by list_examples."
                )
            ),
        ],
        hf_local: Annotated[
            str | None,
            Field(
                description=(
                    "Local HF cache directory — when set, dispatches via "
                    "Docker. Mutually exclusive with cluster_host."
                )
            ),
        ] = None,
        cluster_host: Annotated[
            str | None,
            Field(
                description=(
                    "Slurm cluster login hostname — when set, dispatches via "
                    "remote SSH. Mutually exclusive with hf_local."
                )
            ),
        ] = None,
        cluster_user: Annotated[
            str | None,
            Field(description=("SSH user for the cluster. None uses launcher's default.")),
        ] = None,
        identity: Annotated[
            str | None,
            Field(description=("SSH identity file (-i). None uses ssh-agent / default key.")),
        ] = None,
        job_dir: Annotated[
            str | None,
            Field(
                description=(
                    "Override the experiment output directory. None uses the "
                    "launcher's per-mode default."
                )
            ),
        ] = None,
        job_name: Annotated[
            str | None,
            Field(description=("Override the job_name in the YAML. None uses the YAML's default.")),
        ] = None,
        extra_overrides: Annotated[
            dict[str, str] | None,
            Field(
                description=(
                    "Additional nemo-run-style overrides as a flat dict, e.g. "
                    "{'task.slurm_config.nodes': '2'}."
                )
            ),
        ] = None,
        skip_verify: Annotated[
            bool,
            Field(
                description=(
                    "If True, skip the verify_setup probe before submission. "
                    "Default False — the probe takes ~1s and saves you from "
                    "30+s of wasted submission time on bad config."
                )
            ),
        ] = False,
    ) -> dict:
        return bridge.submit_job_impl(
            yaml_path=yaml_path,
            hf_local=hf_local,
            cluster_host=cluster_host,
            cluster_user=cluster_user,
            identity=identity,
            job_dir=job_dir,
            job_name=job_name,
            extra_overrides=extra_overrides,
            skip_verify=skip_verify,
        )

    @mcp.tool(
        name="job_status",
        description=(
            "Read filesystem-based status from a nemo_run experiment "
            "dir. Returns 'done' / 'failed' / 'running' based on "
            "presence of _DONE and contents of status_*.out files. "
            "Per-task statuses also surfaced for multi-task pipelines."
        ),
    )
    def job_status(
        experiment_id: Annotated[
            str,
            Field(
                description=(
                    "The experiment id returned by submit_job (Slurm) or the "
                    "name nemo_run assigned to the experiment dir."
                )
            ),
        ],
    ) -> dict:
        return bridge.job_status_impl(experiment_id)

    @mcp.tool(
        name="job_logs",
        description=(
            "Read log_<task>.out files from the experiment dir. If "
            "task=None, returns logs for all tasks. If tail=N, returns "
            "only the last N lines per task."
        ),
    )
    def job_logs(
        experiment_id: Annotated[str, Field(description=("The experiment id."))],
        task: Annotated[
            str | None,
            Field(description=("Specific task name to filter logs by. None returns all.")),
        ] = None,
        tail: Annotated[
            int | None,
            Field(description=("Return only the last N lines per task. None returns full.")),
        ] = None,
    ) -> dict:
        return bridge.job_logs_impl(experiment_id, task, tail)

    return mcp


def main() -> None:
    """Entry point for the `modelopt-mcp` console_script."""
    logging.basicConfig(
        stream=sys.stderr,
        level=os.environ.get("MODELOPT_MCP_LOG", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    mcp = _build_server()
    mcp.run()  # stdio by default


if __name__ == "__main__":
    main()
