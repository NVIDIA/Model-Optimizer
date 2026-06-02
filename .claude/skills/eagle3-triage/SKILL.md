---
name: eagle3-triage
description: >
  Triage a failed EAGLE3 pipeline run. Identifies which step failed (data synthesis,
  hidden state dump, training, or benchmark), diagnoses root cause from logs, and
  suggests fixes. Use when user reports an EAGLE3 pipeline failure or asks why a
  specific step failed. Also helps debug new model support issues.
user_invocable: true
---

# EAGLE3 Pipeline Triage

Diagnose failures in the 4-step EAGLE3 offline pipeline. This skill walks through
each step, identifies the failure point, and provides actionable fixes.

## Pipeline Overview

| Step | Script | Purpose | Common failure area |
|------|--------|---------|---------------------|
| task_0 | `common/vllm/query.sh` | Data synthesis via vLLM server | Server startup, model loading, OOM |
| task_1 | `common/eagle3/dump_offline_data_vllm.sh` (or `_hf.sh` / `.sh`) | Dump hidden states | Backend selection, OOM, unsupported arch |
| task_2 | `common/eagle3/train_eagle.sh` | Train EAGLE3 draft head | Dependencies, training crash, export |
| task_3 | `common/specdec_bench/quick_check.sh` | Benchmark acceptance rate | Engine startup, draft model loading |

## Step 0 — Locate the experiment

Ask the user for one of:
- Experiment directory (e.g., the `--job-dir` passed to `launch.py` or `slurm.py`)
- The model name / YAML they ran

Find recent experiments under the job directory:

```bash
ls -td experiments/cicd/cicd_* | head -10
# or wherever --job-dir was pointed
```

Each experiment directory contains one subdirectory per task (task_0 through task_3),
each with a `sbatch_*.out` log file.

## Step 1 — Fetch logs for the failed task

Locate and read the Slurm output file for the failed task:

```bash
find experiments/ -name "sbatch_*.out" | sort
```

Read the last 200 lines — errors appear at the end:

```bash
tail -200 experiments/<exp_id>/<task_dir>/sbatch_<name>_<slurm_id>.out
```

Look for the first task with a non-zero exit code or error message.

## Step 2 — Diagnose by step

### task_0 failures (Data Synthesis)

**How it works:** Launches a vLLM OpenAI-compatible server, polls `/health` until ready,
then runs `query.py` to generate synthetic prompt/response pairs.
Output goes to `/scratchspace/data/`.

| Error pattern | Root cause | Fix |
|---|---|---|
| Server never becomes healthy (hangs at health check) | Model too large for allocated GPUs, or vLLM startup crash | Check BF16 weight size vs GPU memory. GB200: 192 GB/GPU × 4 GPUs/node = 768 GB. Increase TP. |
| `CUDA out of memory` during model load | Insufficient GPU memory | Reduce `--max-model-len` or increase `--tensor-parallel-size` |
| `trust_remote_code` error | Model requires custom code but flag not set | Add `--trust-remote-code` before the `--` separator in task_0 args |
| Vocab / tokenizer error | Missing tokenizer cache (e.g., GPT-OSS-20B needs `TIKTOKEN_RS_CACHE_DIR`) | Set `TIKTOKEN_RS_CACHE_DIR` to a pre-populated cache path in the environment |
| Architecture not supported | vLLM version doesn't support this model | Try a newer vLLM container (`vllm/vllm-openai:latest`) |
| `CANCELLED ... DUE TO TIME LIMIT` | Job wall-clock limit too short | Increase Slurm `--time`. Note: `afterany` deps let task_1 still start. |
| Empty `/scratchspace/data/` | query.py ran but produced no output | Check `--data` path exists and contains prompts. Check query.py logs. |

### task_1 failures (Hidden State Dump)

**How it works:** Loads the target model and runs a forward pass on each conversation,
saving hidden states as `.pt` files in `/scratchspace/offline_hidden_states/`.

Three backends are available:

| Backend | Script | When to use |
|---------|--------|-------------|
| vLLM | `dump_offline_data_vllm.sh` | Broad model coverage; uses `speculators.VllmHiddenStatesGenerator` |
| HF | `dump_offline_data_hf.sh` | VLMs, custom-code models, SWA attention; uses `device_map="auto"` |
| TRT-LLM | `dump_offline_data.sh` | Pure-text models with TRT-LLM support; needs `--tp`/`--moe-ep` args |

| Error pattern | Root cause | Fix |
|---|---|---|
| `No such file or directory: dump_offline_data_vllm.sh` | Wrong script path in YAML | Use the correct path under `common/eagle3/` |
| `FileNotFoundError: /scratchspace/data` | task_0 failed or produced no output | Re-run task_0 first, or point `--input-data` to existing data |
| `CUDA out of memory` | Model too large | Switch to `_hf.sh` (device_map="auto") or increase TP |
| `RuntimeError` / unsupported arch | Model not supported by TRT-LLM backend | Switch to `dump_offline_data_hf.sh` or `dump_offline_data_vllm.sh` |
| `NCCL timeout` / `NCCL error` | Multi-node communication failure | Retry. Reduce EP. |
| No `.pt` files in output dir | Script ran but extraction produced nothing | Check `--max-seq-len` and input data format |
| `pyxis: child terminated with signal 15` | SIGTERM — likely OOM | Increase TP or switch backends |

### task_2 failures (Training)

**How it works:** Installs requirements, runs `launch_train.sh` (Accelerate + FSDP) with the
config from `modelopt_recipes/general/speculative_decoding/eagle3.yaml`, then exports via
`export_hf_checkpoint.py`. Output: `/scratchspace/eagle3/` and `/scratchspace/export/`.

| Error pattern | Root cause | Fix |
|---|---|---|
| `pip install` failure | Network issue or incompatible dependency | Check container has network access |
| `ImportError: modelopt` | ModelOpt not installed or path issue | Check container version |
| `FileNotFoundError: /scratchspace/offline_hidden_states` | task_1 failed or produced no output | Re-run task_1 first |
| `CUDA out of memory` during training | Batch size too large | Reduce `training.train_bs` or `training.training_seq_len` |
| `KeyError` / `AttributeError` in model loading | Model architecture not recognized by EAGLE3 | Check `eagle_decoder_type` in config. Model may need code changes in modelopt. |
| `HFValidationError: Repo id must be in the form...` | Old `offline_training.sh` trying to upload to HF Hub | Use `train_eagle.sh` which does local export only |
| Loss is NaN or diverges | LR too high or data quality issue | Reduce `training.lr`. Check hidden state data. |
| `export_hf_checkpoint.py` fails | Training produced incomplete checkpoint | Check `/scratchspace/eagle3/` for `model.safetensors` |

### task_3 failures (Benchmark)

**How it works:** Launches vLLM with the target + draft model, runs acceptance rate and
throughput benchmarks. Output: JSON files.

| Error pattern | Root cause | Fix |
|---|---|---|
| `FileNotFoundError: /scratchspace/export` | task_2 failed or export step failed | Re-run task_2. Check export output. |
| `trust_remote_code` error at benchmark | Model requires it but `quick_check.sh` doesn't forward the flag | Pass `--trust-remote-code` in task_3 args |
| Server fails with draft model | Draft model config incompatible with engine | Check `eagle_config.json` and engine version |
| AR below threshold / exit code 1 | Draft model quality too low | More epochs, data, or hyperparameter tuning |
| `CUDA out of memory` | Target + draft exceeds GPU memory | Increase TP |
| vLLM EAGLE3 not supported | vLLM version too old | Use `vllm/vllm-openai:latest` (≥ v0.15.0 for NVFP4) |

## Step 3 — Check for new-model-specific issues

If the user is adding support for a new model, also check:

1. **Is the model a VLM?** → Use `dump_offline_data_hf.sh` (text-only path, no vision encoder invoked)
2. **Does the model use sliding window attention (SWA)?** → TRT-LLM backend won't work; use HF or vLLM
3. **Does the model need `trust_remote_code`?** → Add to task_0 args AND task_3 args
4. **Is the model MoE?** → Check `eagle_config.json` `intermediate_size` matches model's `moe_intermediate_size`
5. **Is the model architecture recognized by EAGLE3 training?** → Check `modelopt/torch/speculative/` for the model type
6. **Custom tokenizer?** → May need additional environment vars (e.g., `TIKTOKEN_RS_CACHE_DIR`)

## Step 4 — Suggest fix and next steps

After diagnosis, provide:

1. **Root cause** — one-line summary
2. **Fix** — specific config change, code edit, or command to run
3. **How to re-run** — skip earlier successful steps by pointing to existing scratchspace artifacts

To skip task_0 and task_1 and re-run from task_2:
```bash
uv run launch.py --yaml examples/<Org>/<Model>/hf_offline_eagle3.yaml \
    pipeline.task_0.skip=true \
    pipeline.task_1.skip=true \
    --yes
```

To run only task_1 standalone (using existing task_0 data):
```bash
uv run launch.py --yaml examples/<Org>/<Model>/hf_offline_eagle3.yaml \
    pipeline.task_0.skip=true \
    pipeline.task_2.skip=true \
    pipeline.task_3.skip=true \
    --yes
```

If the fix requires code changes in ModelOpt (e.g., adding a new `eagle_decoder_type`),
note that a separate PR in the modelopt repo is needed.

## Step 5 — Update triage chart

If you encounter a failure pattern not in the triage chart at
`tools/launcher/examples/EAGLE3_TRIAGE.md`, add it:

1. Add a new branch in the Mermaid flowchart under the relevant step node
2. Add a new issue entry in the "Known Issues" section
3. Update the model's row in the test matrix

This keeps the chart current for the next engineer debugging the same issue.
