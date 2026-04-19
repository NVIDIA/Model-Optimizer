---
name: monitor
description: Monitor submitted jobs (PTQ, evaluation, deployment) on SLURM clusters. Use when the user asks "check job status", "is my job done", "monitor my evaluation", "what's the status of the PTQ", "check on job <slurm_job_id>", or after any skill submits a long-running job. Also triggers on "nel status", "squeue", or any request to check progress of a previously submitted job.
---

# Job Monitor

Monitor jobs submitted to SLURM clusters — PTQ quantization, NEL evaluation, model deployment, or raw SLURM jobs.

## When to use

1. **Auto-monitor** — another skill (PTQ, evaluation, deployment) just submitted a job. Register the job and set up monitoring immediately.
2. **User-initiated** — user asks about a job status, possibly in a new conversation. Check the registry, identify the job, and report.

---

## Job Registry

All active jobs are tracked in `.claude/active_jobs.json`. This file is the single source of truth for what's being monitored.

```json
[
  {
    "type": "nel",
    "id": "<invocation_id or slurm_job_id>",
    "host": "<cluster_hostname>",
    "user": "<ssh_user>",
    "submitted": "YYYY-MM-DD HH:MM",
    "description": "<what this job does>",
    "last_status": "<last known status>"
  }
]
```

`type` is one of: `nel`, `slurm`, `launcher`.

---

## On Job Submission

Every time a job is submitted (by any skill or manually):

1. **Add an entry** to `.claude/active_jobs.json`. Create the file if it doesn't exist.
2. **Set up a durable recurring cron** (if one isn't already running) that polls all registered jobs every 15 minutes. The cron prompt should: read the registry, check each job, report state changes to the user, remove completed jobs, and delete itself when the registry is empty.

Always do both steps. Don't try to predict job duration.

---

## On Cron Fire / Status Check

Whether triggered by the cron or by the user asking "check status":

1. **Read the registry** from `.claude/active_jobs.json`
2. **Check each job** using the appropriate method (see below)
3. **Report only state changes** — compare against `last_status` in registry
4. **Update `last_status`** in the registry
5. **Remove completed jobs** — any job in a terminal state (COMPLETED, FAILED, CANCELLED, KILLED)
6. **If registry is empty** — delete the recurring cron

---

## How to Check Each Job Type

### NEL jobs (`type: nel`)

- **Check:** `nel status <id>`
- **On completion:** `nel info <id>` to fetch results
- **On failure:** `nel info <id> --logs` then inspect server/client/SLURM logs via SSH

### Launcher jobs (`type: launcher`)

- **Check:** Tail the launcher's background output file for key events
- **Key events:** experiment ID, SLURM job ID, container import, calibration progress, export path, final status
- **On failure:** Look for `Traceback`, `Error`, or `FAILED` in the output

### Raw SLURM jobs (`type: slurm`)

- **Check:** `ssh <host> "squeue -j <id> -h -o '%T %M %R'"` — if empty, job left the queue
- **On completion:** `ssh <host> "sacct -j <id> --format=State,ExitCode,Elapsed -n"`
- **On failure:** Check the job's output log file

---

## Identifying Jobs (user-initiated, no ID given)

When the user asks about a job without specifying an ID, check in order:

1. `.claude/active_jobs.json` — most reliable, has context
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
