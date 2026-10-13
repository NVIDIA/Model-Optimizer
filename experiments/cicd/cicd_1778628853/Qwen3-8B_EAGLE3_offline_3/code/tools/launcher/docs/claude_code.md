# Using Claude Code with the Launcher

Claude Code creates a tight feedback loop for model optimization: configure → submit → monitor → diagnose → fix → resubmit.

## Setup

```bash
npm install -g @anthropic-ai/claude-code
cd Model-Optimizer/tools/launcher
git submodule update --init --recursive
```

## Workflows

### Submit and Monitor

```text
> Run Qwen3-8B quantization on OCI-HSG and wait for it to finish

Claude will:
1. Run: uv run launch.py --yaml examples/Qwen/Qwen3-8B/megatron_lm_ptq.yaml --yes
2. Monitor: NEMORUN_HOME=$(pwd) uv run nemo experiment status <id>
3. Fetch logs: NEMORUN_HOME=$(pwd) uv run nemo experiment logs <id> 0
4. Report the MMLU score and pass/fail status
```

### Diagnose Failures

```text
> /review-logs

Claude will:
1. Find all experiments in experiments/
2. Fetch logs via nemo experiment logs
3. Analyze error tracebacks
4. Produce a structured report with root cause and suggested fix
5. Write a JUnit XML for CI integration
```

### Add a New Model

```text
> Add Llama-3.1-70B quantization config. It needs 2 nodes with 4 GPUs each.

Claude will:
1. Create examples/Meta/Llama-3.1-70B/megatron_lm_ptq.yaml
2. Set appropriate TP/EP based on model size
3. Reference the correct service script
4. Test with --dryrun to verify the config
```

### Iterate on Failures

```text
> The job failed with CUDA OOM. Try reducing the sequence length to 4096 and resubmit.

Claude will:
1. Edit the YAML config
2. Resubmit with uv run launch.py --yaml <config> --yes
3. Monitor and report results
```

### Reproduce and Compare

```text
> Dump the resolved config for Qwen3-8B, then run it on both OCI-HSG and CW-DFW

Claude will:
1. Dump: uv run launch.py --yaml config.yaml --to-yaml resolved.yaml
2. Run on OCI-HSG: SLURM_CLUSTER=oci_hsg uv run slurm.py --yaml resolved.yaml --yes
3. Run on CW-DFW: SLURM_CLUSTER=cw_dfw uv run slurm.py --yaml resolved.yaml --yes
4. Compare MMLU results
```

## Skills

Available skills:

| Skill | Trigger | Description |
|---|---|---|
| `/review-logs` | After job completion/failure | Analyze logs, diagnose failures, JUnit XML |
| `/wait-for-jobs` | After detached submission | Poll experiment status |
| `/eagle3-new-model` | Adding a new EAGLE3 model | Generate pipeline YAML |

## CI Integration

In CI, Claude Code runs automatically to:

1. Fetch and analyze experiment logs
2. Generate `claude_analysis.md` with findings
3. Write `claude_review_rspec.xml` for GitLab test reporting
4. Post failure summaries as MR comments
5. Create/update GitLab issues for `allow_to_fail` jobs
