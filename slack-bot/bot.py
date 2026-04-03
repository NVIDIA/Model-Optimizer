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

"""ModelOpt Slack Bot — centralized bot with per-user sessions.

Architecture:
    - Single shared Model-Optimizer repo (upstream source)
    - Per-user workspaces (agent decides when to create/reuse)
    - User's cluster config injected into workspaces
    - Claude CLI runs with user's auth credentials

Auth options (per user):
    1. Shared team key (no setup needed)
    2. Anthropic Console account (interactive login via PTY)

Usage:
    @modelopt <prompt>           — run a prompt
    /modelopt setup              — onboard (auth + optional cluster config)
    /modelopt add-cluster        — interactive cluster setup
    /modelopt clusters           — list configured clusters
    /modelopt workspaces         — list your workspaces
    /modelopt cleanup            — remove old workspaces
    /modelopt status             — show session info
    /modelopt help               — show commands
"""

import asyncio
import logging
import os
import re
import uuid
from pathlib import Path

from job_manager import WorkspaceManager
from key_store import KeyStore
from session_manager import run_claude_streaming
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from user_store import UserStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]

REPO_DIR = os.environ.get(
    "REPO_DIR",
    str(Path(__file__).resolve().parent.parent),
)

DATA_DIR = os.environ.get(
    "DATA_DIR", os.path.join(os.path.expanduser("~"), ".local", "share", "modelopt")
)

MAX_SLACK_LENGTH = 3900

# ─── Initialize Components ───────────────────────────────────────────

app = AsyncApp(token=SLACK_BOT_TOKEN)

key_store = KeyStore(data_dir=DATA_DIR)
user_store = UserStore(data_dir=DATA_DIR, key_store=key_store)
workspace_mgr = WorkspaceManager(repo_dir=REPO_DIR)

# Onboarding state machines
onboarding_state: dict[str, str] = {}
cluster_setup_state: dict[str, dict] = {}

# Store last full response per user for /modelopt logs
_last_response: dict[str, str] = {}

# Pending credential imports awaiting user confirmation
_pending_cred_import: dict[str, dict] = {}

# Per-session locks to prevent concurrent Claude processes on the same session
_session_locks: dict[str, asyncio.Lock] = {}

# Keep strong references to background tasks to prevent GC
_background_tasks: set = set()

# ─── Helpers ─────────────────────────────────────────────────────────


def strip_bot_mention(text: str) -> str:
    """Remove @bot mention prefix from a message."""
    return re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()


def truncate(text: str, limit: int = MAX_SLACK_LENGTH) -> str:
    """Truncate text to the given limit, appending a notice if cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n... (truncated, full output in job dir)"


async def send_long_response(say, text: str, thread_ts: str, channel: str):
    """Send response, uploading as file if too long."""
    if len(text) <= MAX_SLACK_LENGTH:
        await say(text=text, thread_ts=thread_ts)
    else:
        await say(
            text=truncate(text) + "\n\n_Full output uploaded as file._",
            thread_ts=thread_ts,
        )
        await app.client.files_upload_v2(
            channel=channel,
            content=text,
            filename="claude_response.md",
            title="Full Claude Response",
            thread_ts=thread_ts,
        )


def is_dm(event: dict) -> bool:
    """Return True if the event is a direct message."""
    return event.get("channel_type") == "im"


# ─── Onboarding ─────────────────────────────────────────────────────

WELCOME_MSG = """Welcome to *ModelOpt Bot*!

I need to set you up first. How would you like to authenticate with Claude?

*1️⃣* Use shared team key (no setup needed)
*2️⃣* Log in with your own Anthropic Console account

Reply with `1` or `2`."""

HELP_MSG = """*ModelOpt Bot Commands*

*Run prompts:*
• `@modelopt <prompt>` — run a ModelOpt task (PTQ, deploy, eval)

*Setup:*
• `/modelopt setup` — onboard (auth + cluster config)
• `/modelopt add-cluster` — configure a remote cluster
• `/modelopt clusters` — list your configured clusters
• `/modelopt set-env KEY=VALUE` — set personal env var (DM only, e.g. `HF_TOKEN`, `NGC_API_KEY`)
• `/modelopt env` — list your env vars
• `/modelopt unset-env KEY` — remove an env var

*Workspaces & Logs:*
• `/modelopt workspaces` — list your workspaces
• `/modelopt logs` — upload full output of last task as a file
• `/modelopt cleanup` — remove old workspaces
• `/modelopt status` — show your current status

*Examples:*
```
@modelopt quantize Qwen3-0.6B with nvfp4
@modelopt quantize and evaluate Llama-3.1-8B with fp8
@modelopt deploy ./my-checkpoint with vLLM
```"""

# ─── Auth via interactive Claude session ─────────────────────────────

_auth_sessions: dict = {}  # {user_id: AuthSession}


async def _start_interactive_login(user_id: str, say, thread_ts: str | None):
    """Start interactive Claude session, navigate to Console login, send URL to user."""
    from auth_session import AuthSession

    try:
        session = AuthSession(user_id, DATA_DIR)
        await say(text="Starting login... this takes a few seconds.", thread_ts=thread_ts)
        url, port = await session.start_and_get_url()

        if url:
            _auth_sessions[user_id] = session
            onboarding_state[user_id] = "awaiting_auth_code"
            await say(
                text=(
                    "Open this link in your browser to sign in with your Anthropic Console account:\n\n"
                    f"{url}\n\n"
                    "After signing in, you'll see a code on the page. Paste that code back here."
                ),
                thread_ts=thread_ts,
            )
        else:
            session.close()
            onboarding_state.pop(user_id, None)
            await say(
                text="Could not start login flow. Try `/modelopt setup` again.",
                thread_ts=thread_ts,
            )
    except Exception as e:
        logger.error("Interactive login error for %s: %s", user_id, e)
        onboarding_state.pop(user_id, None)
        await say(text=f"Login error: {e}\nTry `/modelopt setup` again.", thread_ts=thread_ts)


# ─── Onboarding Response Handler ────────────────────────────────────


async def _finish_onboarding(say, thread_ts):
    """Send the final onboarding message."""
    await say(
        text=(
            "You're all set! Try: `@modelopt quantize Qwen3-0.6B with nvfp4`\n\n"
            "Use `/modelopt help` for all commands."
        ),
        thread_ts=thread_ts,
    )


async def _start_creds_import(user_id, say, thread_ts):
    """Ask user if they want to import credentials from their home dir."""
    onboarding_state[user_id] = "awaiting_creds_choice"
    await say(
        text=(
            "Would you like to import credentials (HF token, NGC key, etc.)?\n\n"
            "*1️⃣* I have a user account on this server — scan my home dir\n"
            "*2️⃣* Skip for now (set later with `/modelopt set-env`)\n\n"
            "Reply with `1` or `2`."
        ),
        thread_ts=thread_ts,
    )


async def handle_onboarding_response(event, say):
    """Handle responses during the onboarding flow."""
    user_id = event["user"]
    text = event.get("text", "").strip()
    thread_ts = event.get("thread_ts", event["ts"])
    state = onboarding_state.get(user_id)

    if state == "awaiting_auth_choice":
        if text == "1":
            user_store.setup_shared_key(user_id)
            del onboarding_state[user_id]
            await say(
                text=(
                    "Using shared team key. No setup needed!\n\n"
                    "Would you like to configure a remote cluster? Reply `yes` or `no`."
                ),
                thread_ts=thread_ts,
            )
            onboarding_state[user_id] = "awaiting_cluster_choice"
        elif text == "2":
            await _start_interactive_login(user_id, say, thread_ts)
        else:
            await say(text="Please reply with `1` or `2`.", thread_ts=thread_ts)
        return True

    if state == "awaiting_auth_code":
        # User pasted the code from the browser callback page
        code = text.strip()
        session = _auth_sessions.pop(user_id, None)
        if not session:
            onboarding_state.pop(user_id, None)
            await say(
                text="Login session expired. Try `/modelopt setup` again.", thread_ts=thread_ts
            )
            return True

        try:
            await say(text="Verifying code...", thread_ts=thread_ts)
            success = await session.submit_code(code)

            if success:
                user_store.setup_login_auth(user_id, session.get_config_dir())
                onboarding_state.pop(user_id, None)
                session.close()
                await say(
                    text=(
                        "Logged in successfully!\n\n"
                        "Would you like to configure a remote cluster? Reply `yes` or `no`."
                    ),
                    thread_ts=thread_ts,
                )
                onboarding_state[user_id] = "awaiting_cluster_choice"
            else:
                onboarding_state.pop(user_id, None)
                session.close()
                await say(
                    text="Login failed. The code may be invalid or expired.\nTry `/modelopt setup` again.",
                    thread_ts=thread_ts,
                )
        except Exception as e:
            logger.error("Auth code error for %s: %s", user_id, e)
            onboarding_state.pop(user_id, None)
            session.close()
            await say(text=f"Login error: {e}\nTry `/modelopt setup` again.", thread_ts=thread_ts)
        return True

    if state == "awaiting_reconfig_choice":
        del onboarding_state[user_id]
        if text == "1":
            # Re-authenticate
            onboarding_state[user_id] = "awaiting_auth_choice"
            await say(text=WELCOME_MSG, thread_ts=thread_ts)
        elif text == "2":
            await start_cluster_setup(user_id, say, thread_ts)
        elif text == "3":
            await _start_creds_import(user_id, say, thread_ts)
        else:
            await say(
                text="Keeping current setup. Use `/modelopt help` for commands.",
                thread_ts=thread_ts,
            )
        return True

    if state == "awaiting_cluster_choice":
        del onboarding_state[user_id]
        if text.lower() in ("yes", "y"):
            await start_cluster_setup(user_id, say, thread_ts, from_onboarding=True)
        else:
            onboarding_state[user_id] = "awaiting_creds_choice"
            await say(
                text=(
                    "Would you like to import credentials (HF token, NGC key, etc.)?\n\n"
                    "*1️⃣* I have a user account on this server — scan my home dir\n"
                    "*2️⃣* Skip for now (set later with `/modelopt set-env`)\n\n"
                    "Reply with `1` or `2`."
                ),
                thread_ts=thread_ts,
            )
        return True

    if state == "awaiting_creds_choice":
        del onboarding_state[user_id]
        if text == "1":
            onboarding_state[user_id] = "awaiting_username"
            await say(
                text="What is your username on this server?",
                thread_ts=thread_ts,
            )
        else:
            await _finish_onboarding(say, thread_ts)
        return True

    if state == "awaiting_username":
        del onboarding_state[user_id]
        username = text.strip()
        home_dir = user_store.resolve_home_dir(username)
        if not home_dir:
            await say(
                text=(
                    f"Could not find home directory for `{username}`.\n"
                    "You can set tokens manually with `/modelopt set-env HF_TOKEN=...`"
                ),
                thread_ts=thread_ts,
            )
            await _finish_onboarding(say, thread_ts)
            return True

        creds = user_store.scan_local_credentials(home_dir)
        if not creds:
            await say(
                text=(
                    f"No credentials found in `{home_dir}`.\n"
                    "You can set tokens manually with `/modelopt set-env HF_TOKEN=...`"
                ),
                thread_ts=thread_ts,
            )
            await _finish_onboarding(say, thread_ts)
            return True

        # Show what was found
        lines = ["*Found credentials:*"]
        for key, val in creds.items():
            if key.startswith("_"):
                if key == "_DOCKER_CONFIG":
                    lines.append(f"• Docker config: `{val}`")
            else:
                masked = val[:6] + "..." if len(val) > 6 else val
                lines.append(f"• `{key}` = `{masked}`")

        # Store found creds temporarily for confirmation
        _pending_cred_import[user_id] = creds
        onboarding_state[user_id] = "awaiting_creds_confirm"
        await say(
            text="\n".join(lines) + "\n\nImport these? Reply `yes` or `no`.",
            thread_ts=thread_ts,
        )
        return True

    if state == "awaiting_creds_confirm":
        del onboarding_state[user_id]
        creds = _pending_cred_import.pop(user_id, {})
        if text.lower() in ("yes", "y") and creds:
            imported = user_store.import_credentials(user_id, creds)
            await say(
                text=f"Imported: {', '.join(f'`{k}`' for k in imported)}",
                thread_ts=thread_ts,
            )
        else:
            await say(
                text="Skipped. Use `/modelopt set-env` to add tokens later.",
                thread_ts=thread_ts,
            )
        await _finish_onboarding(say, thread_ts)
        return True

    # Handle cluster setup flow
    if user_id in cluster_setup_state:
        return await handle_cluster_setup_response(user_id, text, say, thread_ts)

    return False


# ─── Cluster Setup ───────────────────────────────────────────────────


async def start_cluster_setup(user_id, say, thread_ts, *, from_onboarding=False):
    """Begin interactive cluster configuration.

    Shows existing clusters and offers to add/update/remove.
    """
    existing_yaml = user_store.read_clusters_yaml(user_id)
    if existing_yaml:
        try:
            import yaml

            existing = yaml.safe_load(existing_yaml) or {}
            clusters = existing.get("clusters", {})
            default = existing.get("default_cluster", "")
            names = list(clusters.keys())
            cluster_list = "\n".join(
                f"• `{n}`" + (" _(default)_" if n == default else "") for n in names
            )
            cluster_setup_state[user_id] = {
                "step": "action",
                "existing": existing,
                "from_onboarding": from_onboarding,
            }
            await say(
                text=(
                    f"You have {len(names)} cluster(s) configured:\n{cluster_list}\n\n"
                    "*1️⃣* Add a new cluster\n"
                    "*2️⃣* Update an existing cluster\n"
                    "*3️⃣* Remove a cluster\n"
                    "*4️⃣* Done — keep current config\n\n"
                    "Reply with `1`, `2`, `3`, or `4`."
                ),
                thread_ts=thread_ts,
            )
            return
        except Exception:
            pass  # Fall through to fresh setup

    cluster_setup_state[user_id] = {"step": "name", "from_onboarding": from_onboarding}
    await say(
        text=(
            "Let's set up a remote cluster.\n\n*Step 1/6:* What would you like to call this"
            " cluster? (e.g., `cw-dfw`, `selene`, `my-workstation`)"
        ),
        thread_ts=thread_ts,
    )


async def handle_cluster_setup_response(user_id, text, say, thread_ts):
    """Handle multi-step cluster configuration with add/update/remove support."""
    import yaml as _yaml

    state = cluster_setup_state[user_id]
    step = state["step"]

    # ── Action selection (existing clusters) ──
    if step == "action":
        existing = state.get("existing", {})
        clusters = existing.get("clusters", {})
        if text == "1":
            state["step"] = "name"
            await say(
                text="*Step 1/6:* Name for the new cluster?"
                " (e.g., `cw-dfw`, `selene`, `my-workstation`)",
                thread_ts=thread_ts,
            )
        elif text == "2":
            names = list(clusters.keys())
            state["step"] = "pick_update"
            await say(
                text="Which cluster to update?\n"
                + "\n".join(f"• `{n}`" for n in names)
                + "\n\nReply with the cluster name.",
                thread_ts=thread_ts,
            )
        elif text == "3":
            names = list(clusters.keys())
            state["step"] = "pick_remove"
            await say(
                text="Which cluster to remove?\n"
                + "\n".join(f"• `{n}`" for n in names)
                + "\n\nReply with the cluster name.",
                thread_ts=thread_ts,
            )
        else:
            del cluster_setup_state[user_id]
            await say(text="Keeping current cluster config.", thread_ts=thread_ts)
        return True

    if step == "pick_update":
        name = text.strip()
        existing = state.get("existing", {})
        if name not in existing.get("clusters", {}):
            await say(text=f"Cluster `{name}` not found. Try again.", thread_ts=thread_ts)
            return True
        state["name"] = name
        state["step"] = "login_node"
        old = existing["clusters"][name]
        await say(
            text=f"Updating *{name}*. Type `skip` to keep current value.\n\n"
            f"*Step 2/6:* Login node? (current: `{old.get('login_node', '?')}`)",
            thread_ts=thread_ts,
        )
        return True

    if step == "pick_remove":
        name = text.strip()
        existing = state.get("existing", {})
        clusters = existing.get("clusters", {})
        if name not in clusters:
            await say(text=f"Cluster `{name}` not found. Try again.", thread_ts=thread_ts)
            return True
        del clusters[name]
        if existing.get("default_cluster") == name and clusters:
            existing["default_cluster"] = next(iter(clusters))
        del cluster_setup_state[user_id]
        yaml_content = _yaml.dump(existing, default_flow_style=False)
        user_store.save_clusters_yaml(user_id, yaml_content)
        await say(
            text=f"Removed cluster `{name}`.\n\n```{yaml_content}```",
            thread_ts=thread_ts,
        )
        return True

    # ── Shared steps for add/update ──
    existing = state.get("existing", {})
    old = existing.get("clusters", {}).get(state.get("name", ""), {})

    if step == "name":
        state["name"] = text.strip().replace(" ", "-")
        state["step"] = "login_node"
        await say(
            text=f"Cluster alias: *{state['name']}*\n\n"
            "*Step 2/6:* Login node hostname?"
            " (e.g., `cluster-login.example.com`)",
            thread_ts=thread_ts,
        )
    elif step == "login_node":
        val = text.strip()
        state["login_node"] = old.get("login_node", "") if val.lower() == "skip" else val
        state["step"] = "user"
        cur = old.get("user", "")
        await say(
            text="*Step 3/6:* SSH username?"
            + (f" (current: `{cur}`, type `skip` to keep)" if cur else ""),
            thread_ts=thread_ts,
        )
    elif step == "user":
        val = text.strip()
        state["user"] = old.get("user") if val.lower() == "skip" else (val or None)
        state["step"] = "ssh_key"
        cur = old.get("ssh_key", "")
        await say(
            text="*Step 4/6:* SSH private key path? Must be an absolute path"
            " (e.g., `/home/username/.ssh/id_rsa`)."
            + (f" (current: `{cur}`, type `skip` to keep)" if cur else "")
            + " Type `skip` to use SSH default.",
            thread_ts=thread_ts,
        )
    elif step == "ssh_key":
        val = text.strip()
        if val.lower() == "skip":
            state["ssh_key"] = old.get("ssh_key")
        else:
            state["ssh_key"] = val
        state["step"] = "workspace"
        cur = old.get("workspace", "")
        await say(
            text="*Step 5/6:* Remote working directory?"
            + (f" (current: `{cur}`, type `skip` to keep)" if cur else ""),
            thread_ts=thread_ts,
        )
    elif step == "workspace":
        val = text.strip()
        state["workspace"] = old.get("workspace", "") if val.lower() == "skip" else val
        state["step"] = "gpu_type"
        cur = old.get("gpu_type", "")
        await say(
            text="*Step 6/6:* GPU type?"
            + (f" (current: `{cur}`, type `skip` to keep)" if cur else "")
            + " Type `skip` if unknown.",
            thread_ts=thread_ts,
        )
    elif step == "gpu_type":
        gpu_text = text.strip()
        if gpu_text.lower() == "skip":
            gpu = old.get("gpu_type")
        else:
            gpu = gpu_text or old.get("gpu_type")
        del cluster_setup_state[user_id]

        name = state["name"]
        new_cluster = {"login_node": state["login_node"]}
        if state.get("user"):
            new_cluster["user"] = state["user"]
        if state.get("ssh_key"):
            new_cluster["ssh_key"] = state["ssh_key"]
        new_cluster["workspace"] = state["workspace"]
        if gpu:
            new_cluster["gpu_type"] = gpu

        # Merge into existing config
        config = existing if existing else {}
        if "clusters" not in config:
            config["clusters"] = {}
        config["clusters"][name] = new_cluster
        if not config.get("default_cluster"):
            config["default_cluster"] = name

        yaml_content = _yaml.dump(config, default_flow_style=False)
        user_store.save_clusters_yaml(user_id, yaml_content)
        await say(
            text=f"Cluster *{name}* configured!\n\n```{yaml_content}```",
            thread_ts=thread_ts,
        )
        # Only chain to credentials import during initial onboarding
        if state.get("from_onboarding"):
            await _start_creds_import(user_id, say, thread_ts)

    return True


# ─── Slash Command: /modelopt ────────────────────────────────────────


@app.command("/modelopt")
async def handle_slash_command(ack, command, say, respond):
    """Handle /modelopt slash commands."""
    await ack()
    user_id = command["user_id"]
    text = command.get("text", "").strip()
    channel = command["channel_id"]

    parts = text.split(maxsplit=1)
    subcmd = parts[0].lower() if parts else ""
    args = parts[1] if len(parts) > 1 else ""

    if subcmd == "setup":
        if user_store.is_registered(user_id):
            # User already set up — show current config and offer options
            info = user_store.user_info(user_id)
            env_vars = user_store.get_env_vars(user_id)
            has_clusters = user_store.has_clusters(user_id)

            lines = ["You're already set up! Current config:\n"]
            lines.append(f"• *Auth:* {info['auth_method'] if info else 'unknown'}")
            lines.append(f"• *Clusters:* {'configured' if has_clusters else 'none'}")
            if env_vars:
                env_list = ", ".join(f"`{k}`" for k in env_vars)
                lines.append(f"• *Env vars:* {env_list}")
            else:
                lines.append("• *Env vars:* none")

            lines.append("\nWhat would you like to do?\n")
            lines.append("*1️⃣* Re-authenticate (new Claude login)")
            lines.append("*2️⃣* Reconfigure cluster")
            lines.append("*3️⃣* Import/update credentials (HF, NGC, etc.)")
            lines.append("*4️⃣* Keep current setup — nothing to change")
            lines.append("\nReply with `1`, `2`, `3`, or `4`.")

            onboarding_state[user_id] = "awaiting_reconfig_choice"
            await respond(text="\n".join(lines))
        else:
            onboarding_state[user_id] = "awaiting_auth_choice"
            await respond(text=WELCOME_MSG)

    elif subcmd == "add-cluster":
        await start_cluster_setup(user_id, respond, None)

    elif subcmd == "clusters":
        yaml = user_store.read_clusters_yaml(user_id)
        if yaml:
            await respond(text=f"Your cluster config:\n```{yaml}```")
        else:
            await respond(text="No clusters configured. Use `/modelopt add-cluster` to set one up.")

    elif subcmd == "set-env":
        if command.get("channel_name") != "directmessage":
            await respond(text=":warning: Use this command in a DM with me (contains secrets).")
            return
        if not args or "=" not in args:
            await respond(
                text=(
                    "Usage: `/modelopt set-env HF_TOKEN=hf_abc123...`\n\n"
                    "Common variables: `HF_TOKEN`, `NGC_API_KEY`, `DOCKER_TOKEN`"
                )
            )
            return
        key, _, value = args.partition("=")
        user_store.set_env_var(user_id, key.strip(), value.strip())
        await respond(text=f"`{key.strip()}` saved.")

    elif subcmd == "env":
        env_vars = user_store.get_env_vars(user_id)
        if env_vars:
            lines = [f"• `{k}` = `{v}`" for k, v in env_vars.items()]
            await respond(
                text="*Your env vars* (values masked):\n"
                + "\n".join(lines)
                + "\n\nUse `/modelopt set-env KEY=VALUE` to add/update, `/modelopt unset-env KEY` to remove."
            )
        else:
            await respond(
                text="No personal env vars set.\n\nUse `/modelopt set-env HF_TOKEN=hf_abc...` to add one."
            )

    elif subcmd == "unset-env":
        if not args:
            await respond(text="Usage: `/modelopt unset-env HF_TOKEN`")
            return
        if user_store.remove_env_var(user_id, args.strip()):
            await respond(text=f"`{args.strip()}` removed.")
        else:
            await respond(text=f"`{args.strip()}` not found.")

    elif subcmd == "workspaces":
        if not user_store.is_registered(user_id):
            await respond(text="Not registered yet. Use `/modelopt setup` first.")
            return
        ws_root = user_store.jobs_dir(user_id)
        workspaces = workspace_mgr.list_workspaces(ws_root)
        if not workspaces:
            await respond(
                text="No workspaces yet. They'll be created when you run your first task."
            )
            return
        lines = ["*Your workspaces:*"]
        for w in workspaces[:15]:
            lines.append(f"• `{w['name']}` — {w['size_mb']}MB (modified {w['modified']})")
        await respond(text="\n".join(lines))

    elif subcmd == "cleanup":
        if not user_store.is_registered(user_id):
            await respond(text="Not registered yet.")
            return
        ws_root = user_store.jobs_dir(user_id)
        removed = await workspace_mgr.cleanup_old(ws_root)
        await respond(text=f"Cleaned up {removed} old workspace(s).")

    elif subcmd == "status":
        info = user_store.user_info(user_id)
        if info:
            ws_root = user_store.jobs_dir(user_id)
            workspaces = workspace_mgr.list_workspaces(ws_root)
            clusters_str = "configured" if info["has_clusters"] else "none"
            msg = (
                f"*Auth:* {info['auth_method']}\n"
                f"*Clusters:* {clusters_str}\n"
                f"*Workspaces:* {len(workspaces)}"
            )
            await respond(text=msg)
        else:
            await respond(text="Not registered yet. Use `/modelopt setup` first.")

    elif subcmd in ("help", ""):
        await respond(text=HELP_MSG)

    elif subcmd == "logs":
        last = _last_response.get(user_id)
        if not last:
            await respond(text="No recent task output. Run a task first.")
            return
        await app.client.files_upload_v2(
            channel=channel,
            content=last,
            filename="modelopt_task_log.md",
            title="Last Task Output",
        )

    else:
        # Treat as a prompt
        await respond(text="Processing...")
        await _run_job(user_id, text, say_func=respond, channel=channel, thread_ts=None)


# ─── Event Handlers ──────────────────────────────────────────────────


@app.event("app_mention")
async def handle_mention(event, say):
    """Handle @bot mentions in channels."""
    user_id = event["user"]
    text = strip_bot_mention(event.get("text", ""))
    channel = event["channel"]
    thread_ts = event.get("thread_ts", event["ts"])

    if not text:
        await say(
            text="How can I help? Try: `@modelopt quantize Qwen3-0.6B with nvfp4`",
            thread_ts=thread_ts,
        )
        return

    if not user_store.is_registered(user_id):
        onboarding_state[user_id] = "awaiting_auth_choice"
        await say(text=WELCOME_MSG, thread_ts=thread_ts)
        return

    await _run_job(user_id, text, say_func=say, channel=channel, thread_ts=thread_ts)


@app.event("message")
async def handle_dm(event, say):
    """Handle direct messages."""
    if event.get("bot_id") or event.get("subtype"):
        return
    if event.get("channel_type") != "im":
        return

    user_id = event.get("user")
    if not user_id:
        return
    text = event.get("text", "").strip()
    if not text:
        return

    thread_ts = event.get("thread_ts", event["ts"])
    channel = event["channel"]

    if user_id in onboarding_state or user_id in cluster_setup_state:
        handled = await handle_onboarding_response(event, say)
        if handled:
            return

    if not user_store.is_registered(user_id):
        onboarding_state[user_id] = "awaiting_auth_choice"
        await say(text=WELCOME_MSG, thread_ts=thread_ts)
        return

    await _run_job(user_id, text, say_func=say, channel=channel, thread_ts=thread_ts)


# ─── Core Job Execution ─────────────────────────────────────────────


async def _run_job(user_id: str, prompt: str, say_func, channel: str, thread_ts: str | None):
    """Ensure a workspace exists and run Claude in it."""
    clusters_yaml = user_store.read_clusters_yaml(user_id)
    ws_root = user_store.jobs_dir(user_id)

    try:
        workspace = await workspace_mgr.ensure_default_workspace(ws_root, clusters_yaml)
    except Exception as e:
        logger.error("Failed to set up workspace for user %s: %s", user_id, e)
        await say_func(
            text=f":x: Failed to set up workspace: {e}",
            **({"thread_ts": thread_ts} if thread_ts else {}),
        )
        return

    env = user_store.get_claude_env(user_id)
    env["MODELOPT_WORKSPACE_ROOT"] = str(ws_root)
    env["MODELOPT_REPO_DIR"] = str(workspace_mgr.repo_dir)

    bot_context = (
        f"You are running via the ModelOpt Slack bot in --print mode. "
        f"Workspace root: {ws_root} (contains per-model workspaces). "
        f"Upstream repo: {workspace_mgr.repo_dir} (read-only, use for fresh copies). "
        f"Read skills/common/workspace-management.md before creating workspaces. "
        f"Check existing workspaces with: ls $MODELOPT_WORKSPACE_ROOT/ "
        f"IMPORTANT RESTRICTIONS: "
        f"1. You are running in --print mode. CronCreate, background tasks, and Agent "
        f"subagents are NOT available. They will silently fail. "
        f"To monitor SLURM jobs, use sleep-based polling in a bash loop "
        f"(e.g., while squeue -j JOBID | grep -q RUNNING; do sleep 60; done). "
        f"2. SAFETY: You are running unattended — no human can approve actions. "
        f"NEVER run destructive commands (rm -rf /, kill -9, fdisk, mkfs, etc.). "
        f"NEVER modify files outside your workspace ({ws_root}) or the user's "
        f"remote home directory. "
        f"Do NOT modify the upstream repo ({workspace_mgr.repo_dir}). "
        f"Do NOT modify system files, global configs, or other users' data. "
        f"If a task seems risky or ambiguous, output a warning instead of proceeding."
    )

    # Session per Slack thread: messages in the same thread share context,
    # new top-level messages start fresh sessions.
    # thread_ts is the parent message ts (or the message's own ts if it IS the parent).
    session_key = f"modelopt-slack-{user_id}-{thread_ts or 'ephemeral'}"
    session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, session_key))

    # Acquire per-session lock to prevent concurrent Claude on same session
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    lock = _session_locks[session_id]

    if lock.locked():
        kwargs = {"thread_ts": thread_ts} if thread_ts else {}
        await say_func(
            text=":hourglass: Previous request still running. Waiting for it to finish...",
            **kwargs,
        )

    async with lock:
        logger.info("Session %s acquired for user %s", session_id[:8], user_id)
        return await _run_job_inner(
            user_id,
            prompt,
            say_func,
            channel,
            thread_ts,
            workspace,
            env,
            bot_context,
            session_id,
        )


async def _run_job_inner(
    user_id,
    prompt,
    say_func,
    channel,
    thread_ts,
    workspace,
    env,
    bot_context,
    session_id,
):
    """Run Claude (called under session lock)."""
    # Delayed "working on it" — only show if response takes > 5 seconds
    working_msg_sent = False

    async def _send_delayed_notice():
        nonlocal working_msg_sent
        await asyncio.sleep(5)
        if thread_ts:
            working_msg_sent = True
            await say_func(
                text=(
                    ":rocket: Working on it — this may take a while."
                    " I'll let you know when it's done."
                ),
                thread_ts=thread_ts,
            )

    notice_task = asyncio.create_task(_send_delayed_notice())

    # Stream internally to keep idle detection alive. Only send final result to Slack.
    full_response = ""
    try:
        async for chunk in run_claude_streaming(
            prompt=prompt,
            cwd=workspace,
            env=env,
            session_id=session_id,
            system_prompt_extra=bot_context,
        ):
            full_response += chunk.text
            if chunk.is_final:
                break

    except Exception as e:
        full_response += f"\n\n:x: Failed: {e}"
        logger.error("Request failed for user %s: %s", user_id, e)
    finally:
        notice_task.cancel()

    logger.info("Session %s done for user %s", session_id[:8], user_id)

    # Send final response
    if not full_response.strip():
        full_response = "No response from Claude."

    # Save for /modelopt logs
    _last_response[user_id] = full_response

    kwargs = {"thread_ts": thread_ts} if thread_ts else {}
    if channel and thread_ts and len(full_response) > MAX_SLACK_LENGTH:
        await send_long_response(say_func, full_response, thread_ts, channel)
    else:
        await say_func(text=truncate(full_response), **kwargs)


# ─── Auto Cleanup ────────────────────────────────────────────────────

SESSION_MAX_AGE_DAYS = int(os.environ.get("SESSION_MAX_AGE_DAYS", "30"))
CLEANUP_INTERVAL_HOURS = int(os.environ.get("CLEANUP_INTERVAL_HOURS", "6"))


async def _auto_cleanup_loop():
    """Periodically clean up old sessions and workspaces."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_HOURS * 3600)
        try:
            import time

            cutoff = time.time() - SESSION_MAX_AGE_DAYS * 86400
            total_removed = 0

            for uid in user_store.list_users():
                # Clean old Claude sessions
                config_dir = Path(user_store.get_claude_config_dir(uid))
                sessions_dir = config_dir / "projects"
                if sessions_dir.exists():
                    for entry in sessions_dir.iterdir():
                        if entry.is_dir() and entry.stat().st_mtime < cutoff:
                            import shutil

                            shutil.rmtree(entry, ignore_errors=True)
                            total_removed += 1

                # Clean old workspaces (older than 7 days, not the default)
                ws_root = user_store.jobs_dir(uid)
                removed = await workspace_mgr.cleanup_old(
                    ws_root, max_age_days=SESSION_MAX_AGE_DAYS
                )
                total_removed += removed

            if total_removed:
                logger.info("Auto-cleanup: removed %d old sessions/workspaces", total_removed)
        except Exception as e:
            logger.error("Auto-cleanup error: %s", e)


# ─── Main ────────────────────────────────────────────────────────────


async def main():
    """Start the ModelOpt Slack bot."""
    logger.info("Starting ModelOpt Slack Bot...")
    logger.info("Repo dir: %s", REPO_DIR)
    logger.info("Data dir: %s", DATA_DIR)

    if not Path(REPO_DIR).exists():
        logger.error("Repo dir not found: %s", REPO_DIR)
        return

    skills_path = Path(REPO_DIR) / ".claude" / "skills"
    if skills_path.exists():
        skills = [d.name for d in skills_path.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]
        logger.info("Found skills: %s", ", ".join(skills))

    import shutil

    claude_bin = shutil.which("claude")
    if claude_bin:
        logger.info("Claude CLI: %s", claude_bin)
    else:
        logger.error("Claude CLI not found in PATH — bot will not work")

    logger.info("Registered users: %d", len(user_store.list_users()))
    logger.info(
        "Auto-cleanup: every %dh, sessions older than %dd",
        CLEANUP_INTERVAL_HOURS,
        SESSION_MAX_AGE_DAYS,
    )

    # Start background cleanup task (keep reference to prevent GC)
    _cleanup_task = asyncio.create_task(_auto_cleanup_loop())
    _background_tasks.add(_cleanup_task)
    _cleanup_task.add_done_callback(_background_tasks.discard)

    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
