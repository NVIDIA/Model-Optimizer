---
name: monitor
description: Monitor submitted jobs (PTQ, evaluation, deployment) on SLURM clusters. Use when the user asks "check job status", "is my job done", "monitor my evaluation", "what's the status of the PTQ", "check on job <slurm_job_id>", or after any skill submits a long-running job. Also triggers on "nel status", "squeue", or any request to check progress of a previously submitted job.
---

# Job Monitor

Monitor jobs submitted to SLURM clusters — PTQ quantization, NEL evaluation, model deployment, or raw SLURM jobs.

## When to use

1. **Auto-monitor** — another skill (PTQ, evaluation, deployment) just submitted a job. Register the job and set up monitoring immediately.
2. **User-initiated** — user asks about a job status. Check the current session registry first; if the job is not registered there, use the discovery steps below.

---

## Job Registry

Active jobs are tracked in per-session registries under `.claude/agents/`.
This avoids multiple agents clobbering one shared registry when they run at
the same time.

Use the current agent session id as `<session_id>`:

- Claude Code: `$CLAUDE_CODE_SESSION_ID`, or the `session_id` field from hook input
- Codex: `$CODEX_THREAD_ID`
- If no session id is available, create a stable id for the current terminal session and reuse it for every job registered by that agent

Registry layout:

```text
.claude/agents/
  <session_id>/
    active_jobs.json
```

Each session's `active_jobs.json` is a JSON array:

```json
[
  {
    "type": "nel",
    "id": "<invocation_id or slurm_job_id>",
    "host": "<cluster_hostname>",
    "user": "<ssh_user>",
    "submitted": "YYYY-MM-DD HH:MM",
    "description": "<what this job does>",
    "last_status": "<last known status>",
    "owner": {
      "agent": "claude-code|codex|manual",
      "session_id": "<session_id>"
    }
  }
]
```

`type` is one of: `nel`, `slurm`, `launcher`.

---

## On Job Submission

Every time a job is submitted (by any skill or manually):

1. **Add an entry** to `.claude/agents/<session_id>/active_jobs.json`. Create the session directory and file if they don't exist.
2. **Start a durable monitor** (if one isn't already watching the registry) that polls this session's registered jobs until they reach terminal status. Prefer the Claude Code `Monitor` tool when it is available: write a small watcher that reads `.claude/agents/<session_id>/active_jobs.json`, checks every job with the appropriate method below, prints state-change events, updates `last_status`, removes terminal jobs from the session registry, and exits when no active jobs remain for this session.

The monitor should terminate naturally when every registered job has reached a terminal state. If the `Monitor` tool is not available in the current harness, run an equivalent background process that implements the same loop and lets the agent resume/restart when the process exits.

Always do both steps. Don't try to predict job duration.

---

## On Monitor Event / Status Check

Whether triggered by monitor output or by the user asking "check status":

1. **Read the registry** from `.claude/agents/<session_id>/active_jobs.json`
2. **Check each job** using the appropriate method (see below)
3. **Report only state changes** — compare against `last_status` in registry
4. **Update `last_status`** in the session registry
5. **Remove completed jobs** — any job in a terminal state (COMPLETED, FAILED, CANCELLED, KILLED, TIMEOUT, NODE_FAIL, OUT_OF_MEMORY, PREEMPTED, BOOT_FAIL, DEADLINE)
6. **If no active jobs remain** — let the monitor exit

---

## How to Check Each Job Type

Each check method has its **own** status vocabulary. A watcher that mixes them
(e.g. uses SLURM's `COMPLETED` terminal-state regex against `nel status` output)
will silently never fire terminal transitions. Always match against the
vocabulary of the source you're polling.

### NEL jobs (`type: nel`)

- **Check:** `nel status <id>`.

```bash
extract_nel_state() {
  local jid="$1" nel_bin="${NEL:-nel}" output state_col
  output=$("$nel_bin" status "$jid" 2>&1)
  state_col=$(echo "$output" \
    | awk -F'|' -v prefix="$jid." 'index($1, prefix) == 1 { print $2; exit }')
  [ -z "$state_col" ] && state_col="$output"
  echo "$state_col" \
    | LC_ALL=C tr '[:lower:]' '[:upper:]' \
    | awk 'match($0, /(PENDING|RUNNING|SUCCESS|FAILED|KILLED|ERROR|NOT[[:space:]]+FOUND)/) { print substr($0, RSTART, RLENGTH); exit }' \
    | sed 's/[[:space:]][[:space:]]*/ /g'
}

is_nel_terminal() {
  case "$(extract_nel_state "$1")" in
    SUCCESS|FAILED|KILLED|ERROR|"NOT FOUND") return 0 ;;
    *) return 1 ;;
  esac
}
```

- **On completion:** `nel info <id>` to fetch results.
- **On failure:** `nel info <id> --logs` then inspect server/client/SLURM logs via SSH.

### Launcher jobs (`type: launcher`)

- **Check:** Tail the launcher's background output file for key events.
- **Key events:** experiment ID, SLURM job ID, container import, calibration progress, export path, final status.
- **On failure:** Look for `Traceback`, `Error`, or `FAILED` in the output.

### Raw SLURM jobs (`type: slurm`)

- **Check:** `sacct`; use `sacct` for the termination check because `squeue`
  can lag in `COMPLETING` after `sacct` reports a terminal state.

```bash
extract_slurm_state() {
  local jid="$1" host="$2"
  ssh "$host" "sacct -j $jid -X --format=State --noheader -P 2>/dev/null | head -1" \
    | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' \
    | sed 's/^CANCELLED by .*/CANCELLED/'
}

is_slurm_terminal() {
  case "$(extract_slurm_state "$1" "$2")" in
    COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE) return 0 ;;
    *) return 1 ;;
  esac
}
```

- **On completion:** `ssh <host> "sacct -j <id> --format=State,ExitCode,Elapsed -n"`.
- **On failure:** Check the job's output log file.

---

## Identifying Jobs (user-initiated, no ID given)

When the user asks about a job without specifying an ID, check in order:

1. `.claude/agents/<current_session_id>/active_jobs.json` — current agent's jobs
2. `nel ls runs --since 1d` — recent NEL runs
3. `ssh <host> "squeue -u <user>"` — active SLURM jobs
4. `ls -lt tools/launcher/experiments/cicd/ | head -10` — recent launcher experiments

---

## Reporting Guidelines

- **Report state changes proactively** — PENDING → RUNNING, or job completes
- **Aggregate multiple jobs** — "2 of 4 completed (MMLU-Pro: 42.3%, GSM8K: 67.1%), 1 running, 1 pending"
- **Summarize, don't echo** — interpret events ("Calibration complete, exporting checkpoint") not raw logs
- **On failure, diagnose immediately** — check logs and report root cause without waiting for user to ask
- **Minimize noise** — don't report "still running" unless the user is actively asking
