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

"""Per-user data management: auth, cluster config, onboarding state.

Directory layout per user:
    <data_dir>/users/<slack_uid>/
        auth.json          — auth method + encrypted credentials
        clusters.yaml      — SSH/cluster configs
        jobs/              — per-job working directories
"""

import json
import logging
import os
from enum import Enum
from pathlib import Path

from key_store import KeyStore

logger = logging.getLogger(__name__)


class AuthMethod(str, Enum):
    """Authentication method choices for a user."""

    SHARED_KEY = "shared_key"  # Use the server's default ANTHROPIC_API_KEY
    OWN_KEY = "own_key"  # User provided their own sk-ant-... key
    LOGIN = "login"  # User authenticated via `claude auth login` (headless browser flow)


class UserStore:
    """Manages per-user data: auth credentials, cluster configs, onboarding state."""

    def __init__(self, data_dir: str | Path, key_store: KeyStore):
        """Initialize the user store with the given data directory and key store."""
        self._data_dir = Path(data_dir)
        self._users_dir = self._data_dir / "users"
        self._users_dir.mkdir(parents=True, exist_ok=True)
        self._key_store = key_store

        # Shared cache for large model downloads (all users share this)
        self._shared_cache = self._data_dir / "shared-cache"
        self._shared_cache.mkdir(parents=True, exist_ok=True)
        (self._shared_cache / "huggingface").mkdir(exist_ok=True)
        (self._shared_cache / "torch").mkdir(exist_ok=True)

    # ── User Directory ───────────────────────────────────────────────

    def user_dir(self, user_id: str) -> Path:
        """Return the base directory for a user's data."""
        return self._users_dir / user_id

    def jobs_dir(self, user_id: str) -> Path:
        """Return (and create) the jobs/workspace root directory for a user."""
        d = self.user_dir(user_id) / "jobs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def is_registered(self, user_id: str) -> bool:
        """Return True if the user has completed onboarding."""
        return (self.user_dir(user_id) / "auth.json").exists()

    # ── Auth ─────────────────────────────────────────────────────────

    def setup_shared_key(self, user_id: str) -> None:
        """Register user with the shared/default API key."""
        self._ensure_user_dir(user_id)
        self._write_auth(user_id, {"method": AuthMethod.SHARED_KEY})
        logger.info("User %s registered with shared key", user_id)

    def setup_own_key(self, user_id: str, api_key: str) -> None:
        """Register user with their own Anthropic API key."""
        self._ensure_user_dir(user_id)
        self._key_store.store_key(user_id, api_key)
        self._write_auth(user_id, {"method": AuthMethod.OWN_KEY})
        logger.info("User %s registered with own API key", user_id)

    def setup_login_auth(self, user_id: str, config_dir: str) -> None:
        """Register user who authenticated via `claude auth login`.

        The config_dir contains the .credentials.json from the login flow.
        We store this path so we can set CLAUDE_CONFIG_DIR when running Claude.
        """
        self._ensure_user_dir(user_id)
        # Copy the credentials into the user's persistent dir
        import shutil

        user_auth_dir = self.user_dir(user_id) / "claude-config"
        if user_auth_dir.exists():
            shutil.rmtree(user_auth_dir)
        shutil.copytree(config_dir, str(user_auth_dir))
        self._write_auth(user_id, {"method": AuthMethod.LOGIN, "config_dir": str(user_auth_dir)})
        logger.info("User %s registered with claude login auth", user_id)

    def get_auth_method(self, user_id: str) -> AuthMethod | None:
        """Return the auth method for this user, or None if not registered."""
        auth = self._read_auth(user_id)
        if auth is None:
            return None
        return AuthMethod(auth["method"])

    def get_api_key(self, user_id: str) -> str | None:
        """Get the API key to use for this user's Claude session."""
        auth = self._read_auth(user_id)
        if auth is None:
            return None

        method = AuthMethod(auth["method"])
        if method == AuthMethod.SHARED_KEY:
            # Use server's default key
            return os.environ.get("ANTHROPIC_API_KEY")
        elif method == AuthMethod.OWN_KEY:
            return self._key_store.get_key(user_id)
        return None

    def get_claude_env(self, user_id: str) -> dict[str, str]:
        """Build environment variables for this user's Claude subprocess.

        Loads:
        1. System env
        2. Shared cache paths (HF_HOME, TORCH_HOME — shared across users)
        3. User's personal env file (HF_TOKEN, NGC credentials, etc.)
        4. Claude auth credentials

        Note: user env vars can override shared cache paths if needed.
        """
        env = os.environ.copy()

        # Set shared cache dirs (large model downloads shared across users)
        shared_cache = str(self._shared_cache)
        env.setdefault("HF_HOME", f"{shared_cache}/huggingface")
        env.setdefault("TORCH_HOME", f"{shared_cache}/torch")
        env.setdefault("TRANSFORMERS_CACHE", f"{shared_cache}/huggingface/hub")

        # Load user's personal env vars (can override cache paths)
        env_file = self.user_dir(user_id) / "env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    env[key.strip()] = value.strip()

        # Apply Claude auth
        auth = self._read_auth(user_id)
        if auth is None:
            return env

        method = AuthMethod(auth["method"])
        if method == AuthMethod.SHARED_KEY:
            # ANTHROPIC_API_KEY already in env (server default)
            pass
        elif method == AuthMethod.OWN_KEY:
            key = self._key_store.get_key(user_id)
            if key:
                env["ANTHROPIC_API_KEY"] = key
        elif method == AuthMethod.LOGIN:
            # Point Claude CLI at the user's stored credentials
            config_dir = auth.get("config_dir", "")
            if config_dir:
                env["CLAUDE_CONFIG_DIR"] = config_dir

        return env

    def set_env_var(self, user_id: str, key: str, value: str):
        """Set a personal env var for this user."""
        env_file = self.user_dir(user_id) / "env"
        existing = {}
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    existing[k.strip()] = v.strip()
        existing[key] = value
        env_file.write_text("\n".join(f"{k}={v}" for k, v in sorted(existing.items())) + "\n")

    def get_env_vars(self, user_id: str) -> dict[str, str]:
        """List user's personal env vars (values masked)."""
        env_file = self.user_dir(user_id) / "env"
        result = {}
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    result[k.strip()] = v.strip()[:4] + "..." if len(v.strip()) > 4 else v.strip()
        return result

    def remove_env_var(self, user_id: str, key: str) -> bool:
        """Remove a personal env var."""
        env_file = self.user_dir(user_id) / "env"
        if not env_file.exists():
            return False
        lines = []
        removed = False
        for line in env_file.read_text().splitlines():
            if line.strip().startswith(f"{key}="):
                removed = True
            else:
                lines.append(line)
        if removed:
            env_file.write_text("\n".join(lines) + "\n" if lines else "")
        return removed

    def get_claude_config_dir(self, user_id: str) -> str:
        """Return the path to this user's Claude config directory."""
        auth = self._read_auth(user_id)
        if auth and auth.get("config_dir"):
            return auth["config_dir"]
        return str(self.user_dir(user_id) / "claude-config")

    def remove_auth(self, user_id: str) -> bool:
        """Remove user's auth credentials."""
        self._key_store.remove_key(user_id)
        auth_file = self.user_dir(user_id) / "auth.json"
        if auth_file.exists():
            auth_file.unlink()
            return True
        return False

    # ── Cluster Config ───────────────────────────────────────────────

    def get_clusters_yaml_path(self, user_id: str) -> Path:
        """Return the path to the user's cluster config file."""
        return self.user_dir(user_id) / "clusters.yaml"

    def has_clusters(self, user_id: str) -> bool:
        """Return True if the user has a cluster config file."""
        return self.get_clusters_yaml_path(user_id).exists()

    def save_clusters_yaml(self, user_id: str, content: str) -> None:
        """Write cluster config for a user."""
        self._ensure_user_dir(user_id)
        path = self.get_clusters_yaml_path(user_id)
        path.write_text(content, encoding="utf-8")
        logger.info("Saved cluster config for user %s", user_id)

    def read_clusters_yaml(self, user_id: str) -> str | None:
        """Read and return the user's cluster config, or None if not set."""
        path = self.get_clusters_yaml_path(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    # ── User Info ────────────────────────────────────────────────────

    def user_info(self, user_id: str) -> dict | None:
        """Get summary info about a user."""
        if not self.is_registered(user_id):
            return None
        auth = self._read_auth(user_id)
        jobs_path = self.user_dir(user_id) / "jobs"
        job_count = len(list(jobs_path.iterdir())) if jobs_path.exists() else 0
        return {
            "user_id": user_id,
            "auth_method": auth.get("method", "unknown") if auth else "unknown",
            "has_clusters": self.has_clusters(user_id),
            "job_count": job_count,
        }

    def list_users(self) -> list[str]:
        """List all registered user IDs."""
        if not self._users_dir.exists():
            return []
        return [
            d.name for d in self._users_dir.iterdir() if d.is_dir() and (d / "auth.json").exists()
        ]

    # ── Credential Import ──────────────────────────────────────────

    def scan_local_credentials(self, home_dir: str) -> dict[str, str]:
        """Scan a local home directory for known credentials.

        Returns dict of {ENV_VAR_NAME: value} for found credentials.
        Only reads, never modifies.
        """
        found: dict[str, str] = {}
        home = Path(home_dir)
        if not home.is_dir():
            return found

        # HuggingFace token
        for hf_path in [
            home / ".cache" / "huggingface" / "token",
            home / ".huggingface" / "token",
        ]:
            if hf_path.exists():
                token = hf_path.read_text(encoding="utf-8").strip()
                if token:
                    found["HF_TOKEN"] = token
                    break

        # NGC API key
        ngc_config = home / ".ngc" / "config"
        if ngc_config.exists():
            for line in ngc_config.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("apikey"):
                    _, _, val = line.partition("=")
                    val = val.strip()
                    if val:
                        found["NGC_API_KEY"] = val
                        break

        # Docker config (for registry auth — just note it exists, don't extract)
        docker_config = home / ".docker" / "config.json"
        if docker_config.exists():
            found["_DOCKER_CONFIG"] = str(docker_config)

        return found

    def import_credentials(self, user_id: str, creds: dict[str, str]) -> list[str]:
        """Import scanned credentials into user's env file.

        Returns list of imported variable names.
        """
        imported = []
        for key, value in creds.items():
            if key.startswith("_"):
                continue  # Skip metadata entries like _DOCKER_CONFIG
            self.set_env_var(user_id, key, value)
            imported.append(key)
        return imported

    def resolve_home_dir(self, username: str) -> str | None:
        """Resolve a local username to their home directory."""
        import pwd

        try:
            pw = pwd.getpwnam(username)
            if Path(pw.pw_dir).is_dir():
                return pw.pw_dir
        except KeyError:
            pass

        # Fallback: check common paths
        for prefix in ["/home", "/home/scratch." + username]:
            candidate = Path(prefix) / username
            if candidate.is_dir():
                return str(candidate)
        return None

    # ── Internal ─────────────────────────────────────────────────────

    def _ensure_user_dir(self, user_id: str):
        d = self.user_dir(user_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "jobs").mkdir(exist_ok=True)

    def _write_auth(self, user_id: str, auth: dict):
        path = self.user_dir(user_id) / "auth.json"
        path.write_text(json.dumps(auth, indent=2), encoding="utf-8")
        path.chmod(0o600)

    def _read_auth(self, user_id: str) -> dict | None:
        path = self.user_dir(user_id) / "auth.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
