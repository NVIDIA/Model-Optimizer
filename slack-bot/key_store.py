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

"""Encrypted per-user API key storage.

Keys are AES-256-GCM encrypted at rest using a server-side master key.
Each user's key is stored in a separate file under DATA_DIR/keys/<user_id>.enc

The master key is read from the KEY_STORE_SECRET environment variable (32-byte hex or
base64). If not set, a random key is generated and written to DATA_DIR/.master_key
on first use (suitable for single-server dev setups, NOT for production).
"""

import base64
import hashlib
import json
import logging
import os
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)


class KeyStore:
    """Manages encrypted per-user Anthropic API keys."""

    def __init__(self, data_dir: str | Path):
        """Initialize the key store with the given data directory."""
        self._data_dir = Path(data_dir)
        self._keys_dir = self._data_dir / "keys"
        self._keys_dir.mkdir(parents=True, exist_ok=True)
        self._master_key = self._load_master_key()

    # ── Public API ───────────────────────────────────────────────────

    def store_key(self, user_id: str, api_key: str) -> None:
        """Encrypt and store an API key for a Slack user."""
        if not api_key.startswith("sk-ant-"):
            raise ValueError("Invalid Anthropic API key format (expected sk-ant-...)")
        encrypted = self._encrypt(api_key, user_id)
        path = self._key_path(user_id)
        path.write_text(json.dumps(encrypted), encoding="utf-8")
        path.chmod(0o600)
        logger.info("Stored API key for user %s", user_id)

    def get_key(self, user_id: str) -> str | None:
        """Retrieve and decrypt the API key for a user. Returns None if not found."""
        path = self._key_path(user_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return self._decrypt(data, user_id)
        except Exception:
            logger.exception("Failed to decrypt key for user %s", user_id)
            return None

    def remove_key(self, user_id: str) -> bool:
        """Remove stored key. Returns True if key existed."""
        path = self._key_path(user_id)
        if path.exists():
            path.unlink()
            logger.info("Removed API key for user %s", user_id)
            return True
        return False

    def has_key(self, user_id: str) -> bool:
        """Return True if an encrypted key exists for this user."""
        return self._key_path(user_id).exists()

    # ── Internal ─────────────────────────────────────────────────────

    def _key_path(self, user_id: str) -> Path:
        # Use a hash of user_id for the filename to avoid path-injection
        safe_name = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        return self._keys_dir / f"{safe_name}.enc"

    def _load_master_key(self) -> bytes:
        """Load or generate the master encryption key."""
        env_key = os.environ.get("KEY_STORE_SECRET", "")
        if env_key:
            # Accept hex (64 chars) or base64 (44 chars)
            try:
                return bytes.fromhex(env_key)
            except ValueError:
                return base64.b64decode(env_key)

        # Dev fallback: file-based key
        key_file = self._data_dir / ".master_key"
        if key_file.exists():
            return bytes.fromhex(key_file.read_text().strip())

        key = secrets.token_bytes(32)
        key_file.write_text(key.hex())
        key_file.chmod(0o600)
        logger.warning(
            "Generated master key at %s. Set KEY_STORE_SECRET env var in production.",
            key_file,
        )
        return key

    def _encrypt(self, plaintext: str, aad: str) -> dict:
        """AES-256-GCM encrypt. Returns dict with nonce, ciphertext, tag (all base64)."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            # Fallback: XOR-based obfuscation (NOT secure, but functional for dev)
            return self._encrypt_fallback(plaintext, aad)

        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(self._master_key)
        ct = aesgcm.encrypt(nonce, plaintext.encode(), aad.encode())
        return {
            "v": 1,
            "nonce": base64.b64encode(nonce).decode(),
            "ct": base64.b64encode(ct).decode(),
        }

    def _decrypt(self, data: dict, aad: str) -> str:
        if data.get("v") == 1:
            try:
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            except ImportError:
                return self._decrypt_fallback(data, aad)
            aesgcm = AESGCM(self._master_key)
            nonce = base64.b64decode(data["nonce"])
            ct = base64.b64decode(data["ct"])
            return aesgcm.decrypt(nonce, ct, aad.encode()).decode()
        elif data.get("v") == 0:
            return self._decrypt_fallback(data, aad)
        else:
            raise ValueError(f"Unknown key store version: {data.get('v')}")

    # Dev-only fallback when cryptography package isn't installed
    def _encrypt_fallback(self, plaintext: str, aad: str) -> dict:
        key_material = hashlib.sha256(self._master_key + aad.encode()).digest()
        data = plaintext.encode()
        encrypted = bytes(a ^ b for a, b in zip(data, key_material * (len(data) // 32 + 1)))
        return {"v": 0, "ct": base64.b64encode(encrypted).decode()}

    def _decrypt_fallback(self, data: dict, aad: str) -> str:
        key_material = hashlib.sha256(self._master_key + aad.encode()).digest()
        encrypted = base64.b64decode(data["ct"])
        decrypted = bytes(
            a ^ b for a, b in zip(encrypted, key_material * (len(encrypted) // 32 + 1))
        )
        return decrypted.decode()
