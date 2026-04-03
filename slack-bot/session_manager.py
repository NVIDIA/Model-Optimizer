#!/usr/bin/env python3

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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code subprocess runner for jobs.

Runs Claude CLI in a job's working directory with the user's auth credentials.
Uses streaming output so progress is visible in real time.
"""

import asyncio
import contextlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# If no output for this long, assume the process is stuck.
# Must be generous — agent may sleep/poll for long-running SLURM jobs.
IDLE_TIMEOUT = int(os.environ.get("CLAUDE_IDLE_TIMEOUT", "7200"))  # 2 hours

# Track sessions that have been created (--session-id used once).
# Subsequent calls use --resume to avoid "session already in use" error.
_known_sessions: set[str] = set()


@dataclass
class StreamChunk:
    """A chunk of Claude's streamed output."""

    text: str
    is_final: bool = False
    is_error: bool = False


async def run_claude_streaming(
    prompt: str,
    cwd: Path,
    env: dict[str, str],
    session_id: str | None = None,
    idle_timeout: int = IDLE_TIMEOUT,
    system_prompt_extra: str | None = None,
):
    """Run Claude CLI with streaming output.

    Yields StreamChunk objects as Claude produces output.
    No total timeout — only kills if idle for idle_timeout seconds.
    """
    cmd = _build_cmd(prompt, session_id, streaming=True, system_prompt_extra=system_prompt_extra)

    logger.info("Running claude (streaming) in %s: %.80s...", cwd, prompt)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except Exception as e:
        yield StreamChunk(text=f"Error starting Claude: {e}", is_final=True, is_error=True)
        return

    buffer = ""
    try:
        while True:
            try:
                assert proc.stdout is not None  # guaranteed by PIPE
                chunk = await asyncio.wait_for(proc.stdout.read(4096), timeout=idle_timeout)
            except asyncio.TimeoutError:
                # No output for idle_timeout — kill process
                logger.error("Claude idle for %ds, killing", idle_timeout)
                with contextlib.suppress(Exception):
                    proc.kill()
                yield StreamChunk(
                    text=f"\n\nNo output for {idle_timeout // 60}m — process appears stuck. Killed.",
                    is_final=True,
                    is_error=True,
                )
                return

            if not chunk:
                break  # EOF — process finished

            buffer += chunk.decode(errors="replace")

            # Parse complete JSON lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    text = _extract_text_from_event(event)
                    if text:
                        yield StreamChunk(text=text)
                except json.JSONDecodeError:
                    yield StreamChunk(text=line)

    except Exception as e:
        logger.error("Stream read error: %s", e)
        yield StreamChunk(
            text=f"\nStream error: {e}",
            is_final=True,
            is_error=True,
        )
        with contextlib.suppress(Exception):
            proc.kill()
        return

    await proc.wait()

    if proc.returncode != 0:
        assert proc.stderr is not None  # guaranteed by PIPE
        stderr = await proc.stderr.read()
        stderr_text = stderr.decode(errors="replace")[:500]
        yield StreamChunk(
            text=f"\nClaude CLI error (exit {proc.returncode}): {stderr_text}",
            is_final=True,
            is_error=True,
        )
    else:
        yield StreamChunk(text="", is_final=True)


def _build_cmd(
    prompt: str,
    session_id: str | None = None,
    streaming: bool = False,
    system_prompt_extra: str | None = None,
) -> list[str]:
    """Build the claude CLI command."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise FileNotFoundError("`claude` CLI not found in PATH")

    cmd = [
        claude_bin,
        "--print",
        "--dangerously-skip-permissions",
        "--disallowed-tools", "CronCreate,CronDelete,CronList,Agent",
    ]

    if system_prompt_extra:
        cmd.extend(["--append-system-prompt", system_prompt_extra])

    if streaming:
        cmd.extend(["--output-format", "stream-json", "--verbose"])

    if session_id:
        if session_id in _known_sessions:
            cmd.extend(["--resume", session_id])
        else:
            cmd.extend(["--session-id", session_id])
            _known_sessions.add(session_id)

    cmd.extend(["-p", prompt])
    return cmd


def _extract_text_from_event(event: dict) -> str:
    """Extract displayable text from a stream-json event."""
    # Assistant text messages
    if event.get("type") == "assistant" and "message" in event:
        content = event["message"].get("content", [])
        parts = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block["text"])
            elif block.get("type") == "tool_use":
                # Show tool usage so user sees activity during long waits
                tool = block.get("name", "")
                if tool == "Bash":
                    cmd = block.get("input", {}).get("command", "")
                    if cmd:
                        # Truncate long commands
                        cmd_short = cmd[:120] + "..." if len(cmd) > 120 else cmd
                        parts.append(f"\n`$ {cmd_short}`\n")
                elif tool == "Read":
                    path = block.get("input", {}).get("file_path", "")
                    parts.append(f"\n_Reading {path}_\n")
                elif tool == "Edit":
                    path = block.get("input", {}).get("file_path", "")
                    parts.append(f"\n_Editing {path}_\n")
                elif tool == "Write":
                    path = block.get("input", {}).get("file_path", "")
                    parts.append(f"\n_Writing {path}_\n")
        return "".join(parts)

    # Result message (final)
    if event.get("type") == "result":
        return ""

    return ""
