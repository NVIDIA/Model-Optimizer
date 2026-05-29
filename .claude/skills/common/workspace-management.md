# Workspace Management

Organize work by session id and model name so concurrent agents do not
clobber each other, while outputs (checkpoints, logs) stay easy to find and
reuse across PTQ → deploy → eval pipelines within the same session.

## Session Workspaces

Use the same `<session_id>` convention as the monitor skill:

- Claude Code: `$CLAUDE_CODE_SESSION_ID`, or the `session_id` field from hook input
- Codex: `$CODEX_THREAD_ID`
- If no session id is available, create a stable id for the current terminal session and reuse it for every local and remote path created by that agent

## When to Reuse vs Create

**Before starting any task**, check for an existing workspace in the current
session:

```bash
ls ./workspaces/<session_id>/ 2>/dev/null
```

**Reuse** when:

- The matching model workspace already exists under `./workspaces/<session_id>/`
- Task requires output from a previous step (e.g., eval requires the PTQ checkpoint)
- User says "deploy the model I just quantized"

**Create new** when:

- No matching model workspace exists under `./workspaces/<session_id>/`
- User explicitly asks for a fresh start

## Model Workspace Names

Within `./workspaces/<session_id>/`, create one model workspace per model or
model variant. Include meaningful variant details in the model workspace name,
for example quantization format or checkpoint role:

```bash
mkdir -p ./workspaces/<session_id>/<model-name>
```

Use descriptive model workspace names, not timestamps:

```text
# Good
workspaces/<session_id>/qwen3-0.6b-nvfp4/
workspaces/<session_id>/qwen3-0.6b-fp8/
workspaces/<session_id>/qwen3-0.6b-baseline/

# Bad
workspaces/<session_id>/ptq-20260318-143022/
workspaces/<session_id>/job-001/
```

Store outputs (checkpoints, logs) inside the model workspace:

```text
workspaces/<session_id>/qwen3-0.6b-nvfp4/
  output/          # quantized checkpoint
  logs/            # job logs
  scripts/         # custom PTQ scripts (if unsupported model)
```

## Remote execution

When using a remote machine (clusters.yaml configured), create matching workspaces on **both** local and remote:

- **Local** `./workspaces/<session_id>/<model>/` — write and edit scripts here
- **Remote** `<remote_workspace>/<session_id>/<model>/` — model downloads, execution, outputs

Session-scope newly created remote run directories, logs, response caches,
temporary configs, and output artifacts. Shared read-only or concurrency-safe
caches, such as Hugging Face model caches and prebuilt container image caches,
can remain outside the session directory.

Before running, sync the local ModelOpt source and scripts to the remote workspace:

```bash
# Sync ModelOpt source (first time or after local changes)
remote_sync_to ./ <session_id>/<model>/Model-Optimizer/

# Sync custom scripts
remote_sync_to ./workspaces/<session_id>/<model>/scripts/ <session_id>/<model>/scripts/
```

Download the model on the **remote** machine (avoids transferring large model files):

```bash
remote_run "python -c \"from huggingface_hub import snapshot_download; snapshot_download('<model_id>', local_dir='<remote_workspace>/<session_id>/<model>/model')\""
```

Inspect remote files with `remote_run "cat ..."` — read README, config.json, tokenizer_config.json to understand requirements before writing scripts locally.

## Multi-user / Slack bot

When `MODELOPT_WORKSPACE_ROOT` is set, use it instead of `./workspaces/`:

- `MODELOPT_WORKSPACE_ROOT` — user's workspace root (set by the bot); use `$MODELOPT_WORKSPACE_ROOT/<session_id>/<name>/`
- `MODELOPT_REPO_DIR` — shared upstream repo (read-only, use for fresh copies)

To create a workspace, copy the upstream repo (without `.git`):

```bash
rsync -a --quiet \
    --exclude .git --exclude __pycache__ --exclude '*.pyc' \
    --exclude node_modules --exclude '*.egg-info' --exclude '*.sqsh' \
    "$MODELOPT_REPO_DIR/" "$MODELOPT_WORKSPACE_ROOT/<session_id>/<name>/"
```

## Cross-Skill Workspace Flow

Workspaces carry over across the PTQ → Deploy → Eval pipeline. Each stage adds to the same directory:

```text
workspaces/<session_id>/model-name-format/
  output/              ← PTQ: quantized checkpoint
  eval_results/        ← Evaluation: NEL artifacts (results.yml per task)
  eval_config.yaml     ← Evaluation: NEL config
  scripts/             ← Deployment/PTQ: custom run scripts
  logs/                ← All: SLURM job logs
```

## Example Flow

```text
User: "quantize Qwen3-0.6B with nvfp4"
Agent: ls workspaces/<session_id>/ → no "qwen3-0.6b-nvfp4"
       → mkdir workspaces/<session_id>/qwen3-0.6b-nvfp4
       → run PTQ, output to workspaces/<session_id>/qwen3-0.6b-nvfp4/output/

User: "deploy the model I just quantized"
Agent: ls workspaces/<session_id>/ → sees "qwen3-0.6b-nvfp4"
       → reuse, find checkpoint at workspaces/<session_id>/qwen3-0.6b-nvfp4/output/

User: "evaluate the quantized model on MMLU and GSM8K"
Agent: ls workspaces/<session_id>/ → sees "qwen3-0.6b-nvfp4"
       → reuse, write eval_config.yaml, results to workspaces/<session_id>/qwen3-0.6b-nvfp4/eval_results/

User: "now quantize Llama-3.1-8B with fp8"
Agent: ls workspaces/<session_id>/ → no llama
       → mkdir workspaces/<session_id>/llama-3.1-8b-fp8
```
