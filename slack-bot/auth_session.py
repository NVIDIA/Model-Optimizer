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

"""Interactive Claude session for authentication.

Spawns a temporary interactive `claude` session via pexpect to handle the
/login flow (theme selection → Console login → paste code). After auth
succeeds, the session is killed and subsequent requests use --print mode.

The credentials are stored in a per-user CLAUDE_CONFIG_DIR.
"""

import asyncio
import contextlib
import logging
import os
import re
import shutil
import tempfile

logger = logging.getLogger(__name__)


class AuthSession:
    """Manages a temporary interactive Claude session for login."""

    def __init__(self, user_id: str, data_dir: str):
        """Initialize the auth session for the given user."""
        self.user_id = user_id
        self.config_dir = tempfile.mkdtemp(prefix=f"claude-auth-{user_id}-")
        self._data_dir = data_dir
        self._child = None
        self._url = None

    async def start_and_get_url(self) -> tuple[str | None, int | None]:
        """Start interactive session, navigate to Console login, return OAuth URL and port.

        Returns (url, local_port) — the URL the user opens after setting up
        an SSH tunnel to local_port. Returns (None, None) on failure.
        """
        import pexpect

        claude_bin = shutil.which("claude")
        if not claude_bin:
            raise FileNotFoundError("`claude` CLI not found in PATH")

        env = os.environ.copy()
        env["CLAUDE_CONFIG_DIR"] = self.config_dir

        self._child = pexpect.spawn(
            f"{claude_bin} --no-chrome",
            timeout=30,
            env=env,
            encoding="utf-8",
            dimensions=(50, 200),
        )

        def _navigate_to_login():
            import time

            assert self._child is not None
            # Wait for theme picker
            time.sleep(4)
            with contextlib.suppress(Exception):
                self._child.read_nonblocking(16384, timeout=3)

            # Select default theme (press Enter)
            self._child.send("\r")
            time.sleep(3)
            with contextlib.suppress(Exception):
                self._child.read_nonblocking(16384, timeout=3)

            # At login menu — select option 2 (Console account)
            # Press down arrow once, then Enter
            self._child.send("\x1b[B")  # down arrow
            time.sleep(0.5)
            self._child.send("\r")  # enter
            time.sleep(5)

            # Read output in a loop until we get the complete URL.
            # The URL gets line-wrapped by the PTY (\r\n mid-URL), so we
            # must strip all whitespace before extracting.
            buf = ""
            for _ in range(15):
                try:
                    chunk = self._child.read_nonblocking(32768, timeout=2)
                    buf += chunk
                except Exception:
                    pass
                # Strip ANSI escapes, then remove all \r\n to rejoin wrapped lines
                clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", buf)
                clean = re.sub(r"[\r\n]+", "", clean)
                urls = re.findall(
                    r"https://platform\.claude\.com/oauth/authorize[^\s'\"<>]+",
                    clean,
                )
                if urls and "&state=" in urls[0]:
                    url = urls[0]
                    # State is always 43 chars (base64url of 32 bytes, no padding).
                    # Truncate after &state=<43 chars> to remove any trailing text.
                    m = re.search(r"&state=", url)
                    if m:
                        url = url[: m.end() + 43]
                    return url
                time.sleep(1)

            return None

        url = await asyncio.to_thread(_navigate_to_login)
        self._url = url

        # Find the local port the CLI is listening on
        port = None
        if url and self._child:
            try:
                import subprocess as _sp  # nosec B404

                pid = self._child.pid
                ss = _sp.run(  # nosec B603 B607
                    ["ss", "-tlnp"], capture_output=True, text=True
                ).stdout
                for line in ss.split("\n"):
                    if str(pid) in line:
                        m = re.search(r":(\d+)\s", line)
                        if m:
                            port = int(m.group(1))
                            break
            except Exception:
                pass

        if url:
            logger.info("Auth URL captured for %s (port=%s)", self.user_id, port)
        else:
            logger.error("Failed to capture auth URL for %s", self.user_id)

        return url, port

    async def wait_for_auth(self, timeout: int = 300) -> bool:
        """Wait for auth to complete (config.json gets written with API key).

        The user completes the browser auth with an SSH tunnel. The CLI's
        localhost listener receives the callback and writes credentials.

        Returns True if auth succeeded within timeout.
        """
        import json
        from pathlib import Path

        config_file = Path(self.config_dir) / ".claude.json"

        def _poll():
            import time

            start = time.time()
            while time.time() - start < timeout:
                if config_file.exists():
                    try:
                        data = json.loads(config_file.read_text())
                        if data.get("primaryApiKey"):
                            return True
                    except Exception:
                        pass
                # Also check if process exited
                if self._child and not self._child.isalive():
                    # Check one more time
                    if config_file.exists():
                        try:
                            data = json.loads(config_file.read_text())
                            if data.get("primaryApiKey"):
                                return True
                        except Exception:
                            pass
                    return False
                time.sleep(2)
            return False

        return await asyncio.to_thread(_poll)

    async def submit_code(self, code: str) -> bool:
        """Paste the auth code into the interactive session.

        Sends characters one-by-one (Ink TUI uses raw mode and can't handle
        bulk sendline), then presses Enter to submit.

        Returns True if login succeeded.
        """
        if not self._child or not self._child.isalive():
            logger.error("Auth session not alive for %s", self.user_id)
            return False

        def _submit():
            import json
            import time
            from pathlib import Path

            # Send code char-by-char (Ink raw mode requires this)
            for ch in code:
                self._child.send(ch)
                time.sleep(0.02)
            time.sleep(0.3)
            self._child.send("\r")  # Enter to submit

            config_file = Path(self.config_dir) / ".claude.json"

            # Poll for up to 30s: check config.json and process output
            for attempt in range(15):
                time.sleep(2)

                # Check if API key was written
                if config_file.exists():
                    try:
                        data = json.loads(config_file.read_text())
                        if data.get("primaryApiKey"):
                            logger.info("API key found in config for %s", self.user_id)
                            return True
                    except Exception:
                        pass

                # Read any output (non-blocking)
                try:
                    buf = self._child.read_nonblocking(16384, timeout=1)
                    # Strip ALL ANSI escape sequences
                    clean = re.sub(r"\x1b\[[\?0-9;]*[a-zA-Z]", "", buf)
                    clean = re.sub(r"\x1b[><=()][0-9]*", "", clean)
                    clean = re.sub(r"[\r\n\s]+", " ", clean).strip()
                    # Filter out char echoes (single chars or masked *)
                    words = [w for w in clean.split() if len(w) > 5 and not w.startswith("*")]
                    meaningful = " ".join(words)
                    if meaningful:
                        logger.info("Auth output for %s: %s", self.user_id, meaningful[:200])
                    if "successful" in meaningful.lower() or "logged" in meaningful.lower():
                        # Press Enter to continue past the success prompt
                        logger.info("Login successful for %s, pressing Enter", self.user_id)
                        time.sleep(1)
                        self._child.send("\r")
                        time.sleep(3)
                        # Check config now
                        if config_file.exists():
                            try:
                                data = json.loads(config_file.read_text())
                                if data.get("primaryApiKey"):
                                    return True
                            except Exception:
                                pass
                        # Give more time after Enter
                        time.sleep(5)
                        if config_file.exists():
                            try:
                                data = json.loads(config_file.read_text())
                                if data.get("primaryApiKey"):
                                    return True
                            except Exception:
                                pass
                        # Even without config.json, login was successful
                        return True
                    if "oauth error" in meaningful.lower() or "login failed" in meaningful.lower():
                        logger.error("Auth failed for %s: %s", self.user_id, meaningful[:200])
                        return False
                except Exception:
                    pass

                # Check if process died
                if not self._child.isalive():
                    # Final check
                    if config_file.exists():
                        try:
                            data = json.loads(config_file.read_text())
                            if data.get("primaryApiKey"):
                                return True
                        except Exception:
                            pass
                    return False

            return False

        return await asyncio.to_thread(_submit)

    def get_config_dir(self) -> str:
        """Return the temporary config directory path."""
        return self.config_dir

    def close(self):
        """Kill the interactive session."""
        if self._child:
            with contextlib.suppress(Exception):
                self._child.close(force=True)
            self._child = None

    def __del__(self):
        self.close()
