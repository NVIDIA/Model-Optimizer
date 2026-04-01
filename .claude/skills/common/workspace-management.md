# Workspace Management

Organize work by model name so outputs (checkpoints, logs) are easy to find and reuse across PTQ → deploy → eval pipelines.

## Single-user (default)

Create a work directory named after the model in the current project:

```bash
mkdir -p ./workspaces/<model-name>
```

Use descriptive names, not timestamps:

```bash
# Good
workspaces/qwen3-0.6b-nvfp4/
workspaces/llama-3.1-8b-fp8/

# Bad
workspaces/ptq-20260318-143022/
workspaces/job-001/
```

Store outputs (checkpoints, logs) inside the workspace:

```bash
workspaces/qwen3-0.6b-nvfp4/
  output/          # quantized checkpoint
  logs/            # job logs
  scripts/         # custom PTQ scripts (if unsupported model)
```

## When to Reuse vs Create

**Before starting any task**, check for an existing workspace:

```bash
ls ./workspaces/ 2>/dev/null
```

**Reuse** when:

- Same model (e.g., deploying a model you just quantized)
- Task requires output from a previous step (e.g., eval requires the PTQ checkpoint)
- User says "deploy the model I just quantized"

**Create new** when:

- New model not seen before
- User explicitly asks for a fresh start
- Different quantization format for same model (e.g., `qwen3-0.6b-fp8` vs `qwen3-0.6b-nvfp4`)

## Remote execution

When using a remote machine (clusters.yaml configured), create matching workspaces on **both** local and remote:

- **Local** `./workspaces/<model>/` — write and edit scripts here
- **Remote** `<remote_workspace>/workspaces/<model>/` — model downloads, execution, outputs

Before running, sync the local ModelOpt source and scripts to the remote workspace:

```bash
# Sync ModelOpt source (first time or after local changes)
remote_sync_to ./ workspaces/<model>/Model-Optimizer/

# Sync custom scripts
remote_sync_to ./workspaces/<model>/scripts/ workspaces/<model>/scripts/
```

Download the model on the **remote** machine (avoids transferring large model files):

```bash
remote_run "python -c \"from huggingface_hub import snapshot_download; snapshot_download('<model_id>', local_dir='<remote_workspace>/workspaces/<model>/model')\""
```

Inspect remote files with `remote_run "cat ..."` — read README, config.json, tokenizer_config.json to understand requirements before writing scripts locally.

## Multi-user / Slack bot

When `MODELOPT_WORKSPACE_ROOT` is set, use it instead of `./workspaces/`:

- `MODELOPT_WORKSPACE_ROOT` — user's workspace root (set by the bot)
- `MODELOPT_REPO_DIR` — shared upstream repo (read-only, use for fresh copies)

To create a workspace, copy the upstream repo (without `.git`):

```bash
rsync -a --quiet \
    --exclude .git --exclude __pycache__ --exclude '*.pyc' \
    --exclude node_modules --exclude '*.egg-info' --exclude '*.sqsh' \
    "$MODELOPT_REPO_DIR/" "$MODELOPT_WORKSPACE_ROOT/<name>/"
```

## Example Flow

```text
User: "quantize Qwen3-0.6B with nvfp4"
Agent: ls workspaces/ → no "qwen3-0.6b-nvfp4"
       → mkdir workspaces/qwen3-0.6b-nvfp4
       → run PTQ, output to workspaces/qwen3-0.6b-nvfp4/output/

User: "deploy the model I just quantized"
Agent: ls workspaces/ → sees "qwen3-0.6b-nvfp4"
       → reuse, find checkpoint at workspaces/qwen3-0.6b-nvfp4/output/

User: "now quantize Llama-3.1-8B with fp8"
Agent: ls workspaces/ → no llama
       → mkdir workspaces/llama-3.1-8b-fp8
```
