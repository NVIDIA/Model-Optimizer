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

"""ModelOpt launcher MCP server.

Anchor design: OMNIML-5123. Exposes the launcher's submit / status / logs
operations as typed MCP tools that codex / Claude Code agents can call
directly, instead of shelling out to ``uv run launch.py --yaml ...``.

Tool surface (Phase 1):

* ``list_examples`` — discover bundled example YAMLs under
  ``tools/launcher/examples/`` with their model + description metadata.
* ``verify_setup`` — fail-fast probe for the named executor (docker or
  slurm) BEFORE the user burns minutes on a misconfigured submission.
* ``submit_job`` — submit a launcher YAML; mode determined by args
  (``hf_local`` → Docker; ``cluster_host`` → Slurm). Returns
  immediately: Slurm returns ``experiment_id`` (parsed from launch.py's
  detach-mode stdout), Docker always returns the background subprocess
  ``pid`` plus ``experiment_id`` when it appears during the short
  launcher-output tail; on timeout Docker returns ``experiment_id=None``
  and a persistent ``stdout_log`` diagnostic path.
* ``job_status`` — filesystem-based status from nemo_run's experiment
  dir (``_DONE``, ``status_*.out``). No in-memory registry; survives
  MCP server restarts.
* ``job_logs`` — read ``log_*.out`` per task, with optional tail.

Two design constants:

1. **Mode determined by args, not by tool choice.** Single
   ``submit_job`` rather than separate ``submit_docker`` / ``submit_slurm``.
   The mutually-exclusive arg shape is a runtime check; the LLM catalog
   stays compact.
2. **Filesystem is the source of truth.** Status + logs both read from
   nemo_run's experiment dir. The MCP server carries no per-job state
   across calls — survives restarts cleanly.
"""

from modelopt_mcp.server import main

__all__ = ["main"]
