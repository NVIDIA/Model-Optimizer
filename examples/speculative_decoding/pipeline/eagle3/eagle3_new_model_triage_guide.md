# EAGLE3 New Model Support — Triage Guide for Claude Code

This document describes how to triage EAGLE3 pipeline failures when adding a new model.
Follow these steps in order. Stop at the first failure, diagnose, and document findings.

## Pipeline Overview

The EAGLE3 pipeline has 4 stages (mapped to task_0 through task_3 in the YAML):

| Task | Stage | Container | Script | What it does |
|------|-------|-----------|--------|-------------|
| task_0 | Data synthesis | vllm/vllm-openai | `common/vllm/query.sh` | Serve model with vLLM, generate synthetic conversations |
| task_1 | Hidden state dump | vllm/vllm-openai | `common/eagle3/dump_offline_data*.sh` | Dump hidden states from generated conversations |
| task_2 | Training + Export | tensorrt-llm/release | `common/eagle3/train_eagle.sh` | Train EAGLE3 draft model, export HF checkpoint |
| task_3 | Benchmark | vllm/vllm-openai | `common/specdec_bench/quick_check.sh` | Run speculative decoding benchmark |

Some configs combine task_0+task_1 into a single vLLM dump step, or skip task_0 if data already exists.

## Step 1: Locate the pipeline config

```text
tools/launcher/examples/<Org>/<Model>/eagle3_quick_check.yaml
```

If it doesn't exist, create one by copying an existing `eagle3_quick_check.yaml` and adjusting:
- `HF_MODEL_CKPT` — the HF model path on `/hf-local/`
- GPU/node counts based on model size
- `--trust_remote_code` / `--trust-remote-code` if needed
- Container images

## Step 2: Submit the pipeline

```bash
cd tools/launcher
uv run launch.py --yaml examples/<Org>/<Model>/eagle3_quick_check.yaml --yes -v
```

The rsync can take several minutes. Experiment ID is printed as `cicd_<timestamp>`.

## Step 3: Check experiment output

Experiment directory:

```text
experiments/cicd/cicd_<id>/
```

Each task has a directory `<JobName>_<N>/` containing:
- `sbatch_<JobName>_<N>_<SlurmJobID>.out` — the main log
- `code/` — snapshot of the code at submission time

Check logs:

```bash
tail -100 experiments/cicd/cicd_<id>/<JobName>_<N>/sbatch_*.out
```

## Step 4: Diagnose failures by stage

### task_0/task_1 failures (vLLM data generation / hidden state dump)

Common issues:
- **Server never starts** → Check for OOM, unsupported architecture, or missing `--trust_remote_code`
- **`HarmonyError: vocab file`** → gated model, tokenizer not available offline
- **`TypeError: 'NoneType' object is not iterable`** → vLLM doesn't support this model architecture yet
- **`CANCELLED DUE TO TIME LIMIT`** → Model too slow for the time limit; increase wall time or reduce data
- **Server starts but queries fail** → Check prompt format, connection errors

### task_2 failures (training + export)

Common issues:
- **`No such file or directory: service_utils.sh`** → pipeline infra issue (older experiment)
- **`ValueError: Unrecognized configuration class ... for AutoModelForCausalLM`** → VLM model not detected as VLM. Check if `load_vlm_or_llm` in `modelopt/torch/speculative/utils.py` handles this model type. Look for `text_config`/`llm_config` attributes.
- **`FileNotFoundError` on shard files** → Checkpoint has unusual format (e.g., missing HF shards, has consolidated.safetensors instead). Check `FakeBaseModel._load_weights`.
- **OOM during training** → Reduce `--train_bs` or `--training_seq_len`
- **NaN loss** → Reduce `--lr`, check data quality

### task_3 failures (benchmark)

Common issues:
- **`/scratchspace/export` doesn't exist** → task_2 failed; fix training first
- **`StrictDataclassFieldValidationError`** → exported `config.json` has `null` where a typed field is expected (e.g., `use_cache`). Fix the export template in `modelopt/torch/export/plugins/hf_spec_configs.py`.
- **`KeyError: '<model_type>'`** → transformers version in container doesn't recognize the model type
- **`trust_remote_code=True` required** → add to benchmark config
- **vLLM resolves model as wrong architecture** → VLM wrapper model needs special handling

## Step 5: Applying fixes

### Repo fixes (for merged modelopt)

Edit files in `/home/yeyu/Documents/TensorRT-Model-Optimizer/modelopt/torch/speculative/`.
The key files:
- `utils.py` — `load_vlm_or_llm()` for model loading
- `plugins/modeling_fakebase.py` — `FakeBaseModel` for offline training weight loading
- `plugins/hf_eagle.py` — EAGLE model definition
- `../export/plugins/hf_spec_configs.py` — export config templates
- `../export/plugins/hf_spec_export.py` — export logic

### Container patches (for pipeline)

A container may ship a pre-installed modelopt that can't be easily upgraded (CUDA build issues).
If a fix is needed against such an installed library, apply a runtime patch in the relevant
task script (e.g. the training script `common/eagle3/train_eagle.sh`) using a Python heredoc
that find-and-replaces the exact code pattern in the installed file.

> Note: the vLLM dump path previously relied on source-patching the `speculators` library.
> That dependency was removed in favor of vLLM's native `extract_hidden_states` extractor, so
> no speculators patches are applied anymore.

When adding a new patch:
1. Find the exact `old` string in the installed file (must be unique)
2. Write the `new` replacement string
3. Add a `python3 << 'PYEOF' || true` block in the task script before `set -eo pipefail`

## Step 6: Document results

Update `examples/speculative_decoding/pipeline/eagle3/eagle3_triage_chart.md`:
1. Update the model row in the **Model Test Matrix** (status + per-task results)
2. Add a **Per-Model Test Results** entry with experiment IDs, errors, and fixes
3. Add new failure patterns to the **Observed Failure Catalog**

## Known Model-Specific Issues

| Model Type | Issue | Where | Fix |
|-----------|-------|-------|-----|
| `mistral3` (Ministral-3-*) | Not detected as VLM by `"vl"` check | `utils.py` | Check `text_config`/`llm_config` attrs |
| `mistral3` (Ministral-3-8B) | Missing HF shard 1, has `consolidated.safetensors` | `modeling_fakebase.py` | Fallback to consolidated with key aliases |
| All models via FakeBaseModel | `use_cache=null` in exported config | `hf_spec_configs.py` | Set `use_cache: True` in templates |
| `gpt-oss-20b` | Tokenizer requires `openai_harmony` | task_0 | Gated/special tokenizer setup |
| `MiniMax-M2.5` | Custom model code | task_3 | `--trust_remote_code` |
| `ministral3` | `KeyError: 'ministral3'` in older transformers | task_3 | Needs transformers >= 5.3.0 |
