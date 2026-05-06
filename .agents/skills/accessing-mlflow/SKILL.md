---
name: accessing-mlflow
description: Query and browse evaluation results stored in MLflow. Use when the user wants to look up runs by invocation ID, compare metrics across models, fetch artifacts (configs, logs, results), or set up the MLflow MCP server. ALWAYS triggers on mentions of MLflow, experiment results, run comparison, invocation IDs in the context of results, or MLflow MCP setup.
license: Apache-2.0
# Vendored verbatim from NVIDIA NeMo Evaluator (commit 8fa16b2)
# https://github.com/NVIDIA-NeMo/Evaluator/tree/8fa16b237d11e213ea665d5bad6b44d393762317/packages/nemo-evaluator-launcher/.claude/skills/accessing-mlflow
# To re-sync: .claude/scripts/sync-upstream-skills.sh
# Note: this skill depends on the mlflow-mcp MCP server (https://github.com/kkruglik/mlflow-mcp)
# configured in the user's Claude Code setup.
---

# Accessing MLflow

## MCP Server

[mlflow-mcp](https://github.com/kkruglik/mlflow-mcp) gives agents direct access to MLflow — query runs, compare metrics, browse artifacts, all through natural language.

## ID Convention

When the user provides a hex ID (e.g. `71f3f3199ea5e1f0`) without specifying what it is, assume it is an **invocation_id** (not an MLflow run_id). An invocation_id identifies a launcher invocation and is stored as both a tag and a param on MLflow runs. One invocation can produce multiple MLflow runs (one per task). You may need to search across multiple experiments if you don't know which experiment the run belongs to.

## Querying Runs

```python
# Find runs by invocation_id
MLflow:search_runs_by_tags(experiment_id, {"invocation_id": "<invocation_id>"})

# Query for example model/task runs
MLflow:query_runs(experiment_id, "tags.model LIKE '%<model>%'")
MLflow:query_runs(experiment_id, "tags.task_name LIKE '%<task_name>%'")

# Get a config from run's artifacts
MLflow:get_artifact_content(run_id, "config.yml")

# Get nested stats from run's artifacts
MLflow:get_artifact_content(run_id, "artifacts/eval_factory_metrics.json")
```

NOTE: You WILL NOT find PENDING, RUNNING, KILLED, or FAILED runs in MLflow! Only SUCCESSFUL runs are exported to MLflow.

## Workflow Tips

When comparing metrics across runs, fetch the data via MCP, then run the computation in Python for exact results rather than doing math in-context:

```bash
uv run --with pandas python3 << 'EOF'
import pandas as pd
# ... compute deltas, averages, etc.
EOF
```

## Artifacts Structure

```
<harness>.<task>/
├── artifacts/
│   ├── config.yml                # Fully resolved config used during the evaluation
│   ├── launcher_unresolved_config.yaml # Unresolved config passed to the launcher
│   ├── results.yml               # All results in YAML format
│   ├── eval_factory_metrics.json # Runtime stats (latency, tokens count, memory)
│   ├── report.html               # Request-Response Pairs samples in HTML format (if enabled)
│   └── report.json               # Request-Response Pairs samples in JSON format (if enabled)
└── logs/
    ├── client-*.log              # Evaluation client
    ├── server-*-N.log            # Deployment per node
    ├── slurm-*.log               # Slurm job
    └── proxy-*.log               # Request proxy
```

## Troubleshooting

If the MLflow MCP server fails to load or its tools are unavailable:

1. **`uvx` not found** — install [uv](https://docs.astral.sh/uv/getting-started/installation/):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **MCP server not configured** — add the config and restart the agent:

   **For Claude Code** — add to `.claude/settings.json` (project or user level), under `"mcpServers"`:
   ```json
   "MLflow": {
     "command": "uvx",
     "args": ["mlflow-mcp"],
     "env": {
       "MLFLOW_TRACKING_URI": "https://<your-mlflow-server>/"
     }
   }
   ```

   **For Cursor** — edit `~/.cursor/mcp.json` (Settings > Tools & MCP > New MCP Server):
   ```json
   {
     "mcpServers": {
       "MLflow": {
         "command": "uvx",
         "args": ["mlflow-mcp"],
         "env": {
           "MLFLOW_TRACKING_URI": "https://<your-mlflow-server>/"
         }
       }
     }
   }
   ```
