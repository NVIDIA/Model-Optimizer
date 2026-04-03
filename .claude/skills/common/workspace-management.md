# Workspace Management

When running via the Slack bot (or any multi-user environment), each user has a **workspace root** containing model-specific workspaces. Each workspace is a copy of the Model-Optimizer repo where the agent can freely modify code.

## Environment Variables

The bot sets these env vars before launching Claude:

- `MODELOPT_WORKSPACE_ROOT` — user's workspace root (e.g., `/data/modelopt/users/U123/jobs/`)
- `MODELOPT_REPO_DIR` — path to the shared upstream repo (read-only source for copies)

If these are not set, you are running locally — skip workspace management.

## When to Create vs Reuse a Workspace

**Before starting any task**, check for an existing workspace that matches:

```bash
# List existing workspaces
ls "$MODELOPT_WORKSPACE_ROOT/" 2>/dev/null
```

**Reuse** an existing workspace when:
- The task involves the same model (e.g., deploying a model you just quantized)
- The task needs output from a previous step (e.g., eval needs the PTQ checkpoint)
- The user says "deploy the model I just quantized" or similar

**Create a new workspace** when:
- This is a new model not seen before
- The user explicitly asks for a fresh start
- The existing workspace's code modifications are incompatible (rare)

## Creating a New Workspace

Name workspaces by model/purpose, not timestamps:

```bash
# Good names
qwen3-0.6b
llama-3.1-8b-fp8
deepseek-v3-nvfp4

# Bad names (don't use)
ptq-20260318-143022
job-001
```

To create:

```bash
rsync -a --quiet \
    --exclude .git --exclude __pycache__ --exclude '*.pyc' \
    --exclude node_modules --exclude '*.egg-info' --exclude '*.sqsh' \
    "$MODELOPT_REPO_DIR/" "$MODELOPT_WORKSPACE_ROOT/<name>/"
```

Then `cd` into the new workspace and continue with the task.

## Injecting Cluster Config

If `.claude/clusters.yaml` exists in the current workspace, it was injected by the bot. When creating a new workspace, copy it over:

```bash
cp "$MODELOPT_WORKSPACE_ROOT/default/.claude/clusters.yaml" \
   "$MODELOPT_WORKSPACE_ROOT/<new-name>/.claude/clusters.yaml" 2>/dev/null
```

## Example Flow

```
User: "quantize Qwen3-0.6B with nvfp4"
Agent: ls $MODELOPT_WORKSPACE_ROOT/  → empty or no "qwen3-0.6b"
       → create workspace "qwen3-0.6b"
       → run PTQ, output to qwen3-0.6b/output/

User: "deploy the model I just quantized"
Agent: ls $MODELOPT_WORKSPACE_ROOT/  → sees "qwen3-0.6b"
       → reuse workspace, find checkpoint at qwen3-0.6b/output/
       → deploy from there

User: "now quantize Llama-3.1-8B with fp8"
Agent: ls $MODELOPT_WORKSPACE_ROOT/  → sees "qwen3-0.6b", no llama
       → create workspace "llama-3.1-8b-fp8"
```
