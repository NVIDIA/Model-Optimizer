#!/usr/bin/env bash
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

# remote_exec.sh — Remote execution utility for ModelOpt agent skills
#
# Usage:
#   source .claude/skills/common/remote_exec.sh
#   remote_load_cluster <cluster_name>     # or: remote_load_cluster (uses default)
#   remote_check_ssh
#   remote_detect_env                       # detect SLURM vs Docker vs bare metal
#   remote_run "command"
#   remote_sync_to <local_path> [remote_subdir]
#   remote_sync_from <remote_subdir> <local_path>
#   remote_submit_job <job_script>          # SLURM only
#   remote_poll_job <job_id>                # SLURM only
#   remote_wait_job <job_id> [interval=30]  # SLURM only
#   remote_docker_run "<docker_cmd>"        # Docker only
#   remote_tail_log <remote_log_path> [lines=50]
#
# After remote_load_cluster, these env vars are set:
#   REMOTE_HOST, REMOTE_USER, REMOTE_SSH_KEY, REMOTE_SSH_PROXY,
#   REMOTE_WORKSPACE, REMOTE_GPU_TYPE, REMOTE_ENV_TYPE,
#   REMOTE_CONTAINER_IMAGE, REMOTE_SLURM_ACCOUNT, REMOTE_SLURM_PARTITION

# NOTE: This file is designed to be sourced. It does NOT set shell options
# (set -euo pipefail) to avoid mutating the caller's environment.

# ── Helpers ──────────────────────────────────────────────────────────────────

_remote_config_file() {
    # Find clusters.yaml: user-level > project-level
    local user_config="${HOME}/.config/modelopt/clusters.yaml"
    local project_config
    # Walk up from pwd looking for .claude/clusters.yaml
    local dir="$PWD"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/.claude/clusters.yaml" ]]; then
            project_config="$dir/.claude/clusters.yaml"
            break
        fi
        dir="$(dirname "$dir")"
    done

    if [[ -f "$user_config" ]]; then
        echo "$user_config"
    elif [[ -n "${project_config:-}" && -f "$project_config" ]]; then
        echo "$project_config"
    else
        echo ""
    fi
}

_parse_yaml_value() {
    # Simple YAML value extractor: _parse_yaml_value <file> <dot.path>
    # Handles simple scalar values only (not arrays/nested objects)
    # Uses sys.argv to avoid shell injection via file paths or YAML keys.
    local file="$1" path="$2"
    python3 - "$file" "$path" <<'PYEOF' 2>/dev/null || true
import yaml, sys
with open(sys.argv[1]) as f:
    data = yaml.safe_load(f)
for k in sys.argv[2].split('.'):
    if isinstance(data, dict) and k in data:
        data = data[k]
    else:
        sys.exit(0)
if data is not None:
    print(data)
PYEOF
}

_ssh_control_path() {
    # Return the path for the SSH ControlMaster socket
    # Use a per-host socket so multiple clusters don't collide
    # Try multiple writable locations (sandbox may restrict /tmp)
    local tmpdir
    for candidate in "${TMPDIR:-}" /tmp/claude-*/ssh-ctl /tmp; do
        if [[ -n "$candidate" && -d "$candidate" && -w "$candidate" ]]; then
            tmpdir="$candidate"
            break
        fi
    done
    # Fallback: create in home dir
    tmpdir="${tmpdir:-$HOME/.cache/ssh-ctl}"
    mkdir -p "$tmpdir" 2>/dev/null || true
    # Use short name to avoid Unix socket path length limit (108 chars)
    local host_hash
    host_hash=$(echo "${REMOTE_USER}@${REMOTE_HOST}" | md5sum | cut -c1-12)
    echo "${tmpdir}/ctl-${host_hash}"
}

_ssh_base_opts() {
    # Build SSH options (without the ssh command itself or user@host)
    local opts="-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new"
    # ControlMaster multiplexing: reuse a single persistent SSH connection
    local ctl_path
    ctl_path="$(_ssh_control_path)"
    opts+=" -o ControlPath='${ctl_path}'"
    # If master is already running, just reuse it; otherwise don't try to become master
    # (remote_start_session handles starting the master)
    opts+=" -o ControlMaster=auto"
    if [[ -n "${REMOTE_SSH_KEY:-}" ]]; then
        opts+=" -i $REMOTE_SSH_KEY"
    fi
    if [[ -n "${REMOTE_SSH_PROXY:-}" ]]; then
        opts+=" -o ProxyCommand='${REMOTE_SSH_PROXY}'"
    fi
    echo "$opts"
}

_ssh_base_cmd() {
    # Build the full SSH command
    echo "ssh $(_ssh_base_opts) ${REMOTE_USER}@${REMOTE_HOST}"
}

# ── Session Management ───────────────────────────────────────────────────────

remote_start_session() {
    # Start a persistent SSH ControlMaster connection in the background.
    # All subsequent remote_run / remote_sync_* / scp calls reuse this connection.
    # Call this once after remote_load_cluster + remote_check_ssh.
    local ctl_path
    ctl_path="$(_ssh_control_path)"

    # If a master is already running, skip
    if ssh -o ControlPath="$ctl_path" -O check "${REMOTE_USER}@${REMOTE_HOST}" 2>/dev/null; then
        echo "SSH session already active (reusing existing connection)."
        return 0
    fi

    echo "Starting persistent SSH session to ${REMOTE_USER}@${REMOTE_HOST}..."
    local opts="-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new"
    opts+=" -o ControlMaster=yes -o ControlPath='${ctl_path}' -o ControlPersist=3600"
    if [[ -n "${REMOTE_SSH_KEY:-}" ]]; then
        opts+=" -i $REMOTE_SSH_KEY"
    fi
    if [[ -n "${REMOTE_SSH_PROXY:-}" ]]; then
        opts+=" -o ProxyCommand='${REMOTE_SSH_PROXY}'"
    fi

    # Start the master in the background (-f -N: go background, no command)
    eval "ssh $opts -f -N ${REMOTE_USER}@${REMOTE_HOST}" 2>&1
    local rc=$?
    if (( rc == 0 )); then
        echo "SSH session established. All commands will reuse this connection."
        echo "Call 'remote_enable_cleanup_trap' to auto-close on exit, or 'remote_stop_session' manually."
    else
        echo "WARNING: Failed to start persistent SSH session (rc=$rc). Commands will use individual connections." >&2
    fi
    return $rc
}

remote_stop_session() {
    # Gracefully close the persistent SSH connection
    local ctl_path
    ctl_path="$(_ssh_control_path)"
    if [[ -S "$ctl_path" ]]; then
        ssh -o ControlPath="$ctl_path" -O exit "${REMOTE_USER}@${REMOTE_HOST}" 2>/dev/null || true
        echo "SSH session closed."
    fi
}

remote_enable_cleanup_trap() {
    # Opt-in: register an EXIT trap to auto-close the SSH session.
    # Chains with any existing EXIT trap to avoid breaking the caller.
    local existing_trap
    existing_trap=$(trap -p EXIT | sed "s/^trap -- '//;s/' EXIT$//")
    if [[ -n "$existing_trap" ]]; then
        trap "${existing_trap}; remote_stop_session 2>/dev/null" EXIT
    else
        trap 'remote_stop_session 2>/dev/null' EXIT
    fi
}

# ── Core Functions ───────────────────────────────────────────────────────────

remote_load_cluster() {
    # Load cluster config by name. If no name given, use default_cluster.
    local cluster_name="${1:-}"
    local config_file
    config_file="$(_remote_config_file)"

    if [[ -z "$config_file" ]]; then
        echo "ERROR: No clusters.yaml found. Provide cluster info interactively or create one." >&2
        echo "  User config:    ~/.config/modelopt/clusters.yaml" >&2
        echo "  Project config: .claude/clusters.yaml" >&2
        return 1
    fi

    # Get default cluster if none specified
    if [[ -z "$cluster_name" ]]; then
        cluster_name="$(_parse_yaml_value "$config_file" "default_cluster")"
        if [[ -z "$cluster_name" ]]; then
            echo "ERROR: No cluster name specified and no default_cluster in config." >&2
            return 1
        fi
    fi

    # Parse cluster config
    REMOTE_HOST="$(_parse_yaml_value "$config_file" "clusters.${cluster_name}.login_node")"
    REMOTE_USER="$(_parse_yaml_value "$config_file" "clusters.${cluster_name}.user")"
    REMOTE_SSH_KEY="$(_parse_yaml_value "$config_file" "clusters.${cluster_name}.ssh_key")"
    REMOTE_SSH_PROXY="$(_parse_yaml_value "$config_file" "clusters.${cluster_name}.ssh_proxy")"
    REMOTE_WORKSPACE="$(_parse_yaml_value "$config_file" "clusters.${cluster_name}.workspace")"
    REMOTE_GPU_TYPE="$(_parse_yaml_value "$config_file" "clusters.${cluster_name}.gpu_type")"
    REMOTE_CONTAINER_IMAGE="$(_parse_yaml_value "$config_file" "clusters.${cluster_name}.container_image")"
    REMOTE_ENV_TYPE="$(_parse_yaml_value "$config_file" "clusters.${cluster_name}.env_type")"

    # SLURM-specific
    REMOTE_SLURM_ACCOUNT="$(_parse_yaml_value "$config_file" "clusters.${cluster_name}.slurm.default_account")"
    REMOTE_SLURM_PARTITION="$(_parse_yaml_value "$config_file" "clusters.${cluster_name}.slurm.default_partition")"

    # Expand ~ in paths
    if [[ "${REMOTE_SSH_KEY:-}" == "~/"* ]]; then
        REMOTE_SSH_KEY="${HOME}/${REMOTE_SSH_KEY#\~/}"
    fi
    if [[ "${REMOTE_WORKSPACE:-}" == "~/"* ]]; then
        REMOTE_WORKSPACE="${HOME}/${REMOTE_WORKSPACE#\~/}"
    fi

    # Validate required fields
    if [[ -z "$REMOTE_HOST" ]]; then
        echo "ERROR: Cluster '$cluster_name' has no login_node defined." >&2
        return 1
    fi
    if [[ -z "${REMOTE_WORKSPACE:-}" || "$REMOTE_WORKSPACE" == "/" ]]; then
        echo "ERROR: Cluster '$cluster_name' must define a non-root workspace." >&2
        return 1
    fi

    # Default user to current user
    REMOTE_USER="${REMOTE_USER:-$USER}"

    export REMOTE_HOST REMOTE_USER REMOTE_SSH_KEY REMOTE_SSH_PROXY
    export REMOTE_WORKSPACE REMOTE_GPU_TYPE REMOTE_CONTAINER_IMAGE
    export REMOTE_ENV_TYPE REMOTE_SLURM_ACCOUNT REMOTE_SLURM_PARTITION

    echo "Loaded cluster: $cluster_name (${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_WORKSPACE})"
}

remote_check_ssh() {
    # Validate SSH connectivity and start a persistent session.
    # After this call, all remote_run / remote_sync_* commands reuse one connection.
    echo "Checking SSH connectivity to ${REMOTE_USER}@${REMOTE_HOST}..."
    # Start persistent session (also validates connectivity)
    if remote_start_session 2>&1; then
        return 0
    fi
    # Fallback: try a one-off connection
    local result
    if result=$(eval "$(_ssh_base_cmd)" '"echo SSH_OK"' 2>&1); then
        if echo "$result" | grep -q "SSH_OK"; then
            echo "SSH connection OK (no persistent session — commands will be slower)."
            return 0
        fi
    fi
    echo "ERROR: SSH connection failed:" >&2
    echo "$result" >&2
    return 1
}

remote_detect_env() {
    # Auto-detect remote environment: slurm, docker, or bare
    # Sets REMOTE_ENV_TYPE and discovers GPU info
    if [[ -n "${REMOTE_ENV_TYPE:-}" && "$REMOTE_ENV_TYPE" != "auto" ]]; then
        echo "Environment type: $REMOTE_ENV_TYPE (from config)"
        return 0
    fi

    echo "Detecting remote environment..."
    local info
    info=$(remote_run "
        echo ENV_DETECT_START;
        # Check SLURM
        if command -v sbatch &>/dev/null; then
            echo 'HAS_SLURM=yes';
            sacctmgr show associations user=\$USER format=account%30,partition%20,cluster%20 -n 2>/dev/null | head -20;
            echo 'SLURM_PARTITIONS_START';
            sinfo -o '%P %a %l %D %G' 2>/dev/null | head -30;
            echo 'SLURM_PARTITIONS_END';
        else
            echo 'HAS_SLURM=no';
        fi;
        # Check Docker
        if command -v docker &>/dev/null; then
            echo 'HAS_DOCKER=yes';
            # Check if nvidia-container-cli is available (GPU support without pulling an image)
            if command -v nvidia-container-cli &>/dev/null || docker info 2>/dev/null | grep -qi nvidia; then
                echo 'DOCKER_GPU=yes';
            else
                echo 'DOCKER_GPU=no';
            fi;
        else
            echo 'HAS_DOCKER=no';
        fi;
        # Check bare metal GPU
        if command -v nvidia-smi &>/dev/null; then
            echo 'HAS_BARE_GPU=yes';
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null;
        else
            echo 'HAS_BARE_GPU=no';
        fi;
        echo ENV_DETECT_END;
    " 2>&1)

    echo "$info"

    if echo "$info" | grep -q "HAS_SLURM=yes"; then
        REMOTE_ENV_TYPE="slurm"
    elif echo "$info" | grep -q "DOCKER_GPU=yes"; then
        REMOTE_ENV_TYPE="docker"
    elif echo "$info" | grep -q "HAS_BARE_GPU=yes"; then
        REMOTE_ENV_TYPE="bare"
    elif echo "$info" | grep -q "HAS_DOCKER=yes"; then
        # Docker available but no GPU support — fall back to bare
        REMOTE_ENV_TYPE="bare"
    else
        REMOTE_ENV_TYPE="unknown"
    fi

    export REMOTE_ENV_TYPE
    echo "Detected environment: $REMOTE_ENV_TYPE"
}

remote_run() {
    # Run a command on the remote machine
    # Usage: remote_run "command"
    # Uses base64 encoding to avoid all quoting/escaping issues.
    # Retries up to 3 times on SSH connection failures.
    local cmd="$1"
    local ws="${REMOTE_WORKSPACE:-\$HOME}"
    local full_cmd="cd \"$ws\" && $cmd"
    local encoded
    encoded=$(printf '%s' "$full_cmd" | base64 -w0)

    local attempt=0 max_attempts=3 result rc
    while (( attempt < max_attempts )); do
        result=$(eval "$(_ssh_base_cmd)" "'echo $encoded | base64 -d | bash'" 2>&1) && rc=$? || rc=$?
        if (( rc != 255 )); then
            # rc=255 is SSH connection failure; anything else is the remote command's exit code
            echo "$result"
            return $rc
        fi
        attempt=$((attempt + 1))
        if (( attempt < max_attempts )); then
            echo "SSH connection failed (attempt $attempt/$max_attempts), retrying in 10s..." >&2
            sleep 10
        fi
    done
    echo "$result"
    return $rc
}

remote_sync_to() {
    # Sync local path to remote workspace
    # Usage: remote_sync_to <local_path> [remote_subdir]
    local local_path="$1"
    local remote_subdir="${2:-}"
    local remote_dest="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_WORKSPACE}/${remote_subdir}"

    local rsync_cmd="rsync -avz --progress"
    # Add default excludes
    for excl in .git __pycache__ "*.pyc" .claude node_modules "*.egg-info"; do
        rsync_cmd+=" --exclude='$excl'"
    done
    # Reuse the shared SSH options (including ControlMaster)
    rsync_cmd+=" -e \"ssh $(_ssh_base_opts)\""
    rsync_cmd+=" '${local_path}/' '${remote_dest}'"

    echo "Syncing ${local_path} → ${remote_dest} ..."
    eval "$rsync_cmd"
}

remote_sync_from() {
    # Sync from remote to local
    # Usage: remote_sync_from <remote_subdir> <local_path>
    local remote_subdir="$1"
    local local_path="$2"
    local remote_src="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_WORKSPACE}/${remote_subdir}"

    mkdir -p "$local_path"
    echo "Fetching ${remote_src} → ${local_path} ..."
    eval "rsync -avz --progress -e \"ssh $(_ssh_base_opts)\" '${remote_src}/' '${local_path}/'"
}

# ── SLURM Functions ──────────────────────────────────────────────────────────

remote_submit_job() {
    # Submit a SLURM job script that's already on the remote machine
    # Usage: remote_submit_job <remote_script_path>
    # Returns: job ID on stdout
    local script_path="$1"
    local output
    output=$(remote_run "sbatch '$script_path'" 2>&1)
    local jobid
    jobid=$(echo "$output" | grep -o '[0-9]\+' | tail -1)
    if [[ -z "$jobid" ]]; then
        echo "ERROR: Failed to submit job:" >&2
        echo "$output" >&2
        return 1
    fi
    echo "$jobid"
}

remote_poll_job() {
    # Check SLURM job state
    # Usage: remote_poll_job <job_id>
    # Returns: PENDING, RUNNING, COMPLETED, FAILED, TIMEOUT, CANCELLED, etc.
    local jobid="$1"
    local state
    state=$(remote_run "squeue -j $jobid -h -o %T 2>/dev/null" 2>&1 | grep -v "^$" | tail -1)
    if [[ -z "$state" ]]; then
        # Job no longer in queue — check sacct
        state=$(remote_run "sacct -j $jobid --format=State -n -X 2>/dev/null" 2>&1 | awk '{print $1}' | head -1)
    fi
    echo "${state:-UNKNOWN}"
}

remote_wait_job() {
    # Wait for a SLURM job to complete
    # Usage: remote_wait_job <job_id> [poll_interval_seconds=30]
    local jobid="$1"
    local interval="${2:-30}"
    echo "Waiting for job $jobid (polling every ${interval}s)..."
    while true; do
        local state
        state=$(remote_poll_job "$jobid")
        echo "$(date '+%H:%M:%S') Job $jobid: $state"
        case "$state" in
            COMPLETED)
                echo "Job $jobid completed successfully."
                return 0
                ;;
            FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL)
                echo "ERROR: Job $jobid ended with state: $state" >&2
                remote_job_result "$jobid"
                return 1
                ;;
            UNKNOWN)
                echo "WARNING: Could not determine job state. Checking sacct..." >&2
                remote_job_result "$jobid"
                return 1
                ;;
        esac
        sleep "$interval"
    done
}

remote_job_result() {
    # Get job result details from sacct
    # Usage: remote_job_result <job_id>
    local jobid="$1"
    remote_run "sacct -j $jobid --format=JobID,State,ExitCode,Elapsed,MaxRSS -n 2>/dev/null"
}

# ── Docker Functions ─────────────────────────────────────────────────────────

remote_docker_run() {
    # Run a command inside a Docker container on the remote machine
    # Usage: remote_docker_run <container_or_image> "<command>"
    # If container_or_image matches a running container name, uses docker exec.
    # Otherwise, uses docker run with the given image.
    local container_or_image="$1"
    local cmd="$2"

    # Check if it's a running container
    local is_running
    is_running=$(remote_run "docker ps --format '{{.Names}}' | grep -x '$container_or_image' 2>/dev/null" 2>&1 || true)

    if [[ -n "$is_running" ]]; then
        echo "Executing in running container: $container_or_image"
        remote_run "docker exec $container_or_image bash -c '$cmd'"
    else
        echo "Running in new container: $container_or_image"
        remote_run "docker run --rm --gpus all -v ${REMOTE_WORKSPACE}:${REMOTE_WORKSPACE} -w ${REMOTE_WORKSPACE} $container_or_image bash -c '$cmd'"
    fi
}

# ── Log Functions ────────────────────────────────────────────────────────────

remote_tail_log() {
    # Tail a log file on the remote machine
    # Usage: remote_tail_log <remote_log_path> [num_lines=50]
    local log_path="$1"
    local lines="${2:-50}"
    remote_run "tail -n $lines '$log_path' 2>/dev/null || echo 'Log file not found: $log_path'"
}

# ── Workspace Functions ──────────────────────────────────────────────────────

remote_ensure_workspace() {
    # Create the remote workspace directory if it doesn't exist
    remote_run "mkdir -p '${REMOTE_WORKSPACE}'"
    echo "Remote workspace ready: ${REMOTE_WORKSPACE}"
}

remote_workspace_info() {
    # Print useful info about the remote workspace
    remote_run "
        echo '=== Workspace: ${REMOTE_WORKSPACE} ===';
        echo '--- Disk usage ---';
        du -sh '${REMOTE_WORKSPACE}' 2>/dev/null || echo 'N/A';
        echo '--- Contents ---';
        ls -la '${REMOTE_WORKSPACE}/' 2>/dev/null | head -20;
    "
}
