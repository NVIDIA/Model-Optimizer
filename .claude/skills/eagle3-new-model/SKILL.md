---
name: eagle3-new-model
description: >
  Add a new model to the EAGLE3 offline pipeline. Generates an hf_offline_eagle3.yaml
  launcher config for a new model checkpoint, choosing the right hidden state dump
  backend (TRT-LLM / HF / vLLM) and GPU configuration.
  Use when user wants to run EAGLE3 on a model that does not yet have a YAML in
  tools/launcher/examples/ or asks how to configure the pipeline for a new checkpoint.
---

# EAGLE3 New Model Configuration

This skill guides you through creating `tools/launcher/examples/<Org>/<Model>/hf_offline_eagle3.yaml`
for a new model.

## Step 1 — Look up the model architecture

Determine these values from the HuggingFace model card, `config.json`, and vLLM docs:

| Property | Where to find it |
|---|---|
| Total / active parameters | Model card |
| Dense or MoE? | `config.json` → `num_experts`, `num_experts_per_tok` |
| Attention type (MHA / GQA / MLA / SWA) | Model card |
| Multimodal? (vision encoder) | Model card |
| BF16 weight size (GB) | `total_params × 2 bytes` |
| Special serving flags | vLLM docs, model README (`--trust-remote-code`, parsers) |

## Step 2 — Calculate GPU requirements (OCI-HSG / GB200)

OCI-HSG nodes: **4 GPUs × 192 GB HBM3e = 768 GB per node**

```
BF16 weight size  = total_params × 2 bytes
GPUs needed       = ceil(weight_size_GB / 192)
nodes             = ceil(gpus_needed / 4)
tp                = min(gpus_needed, 4)
```

| Model | Weights (BF16) | GPUs | nodes | tp |
|---|---|---|---|---|
| 8B dense | ~16 GB | 1 | 1 | 4 |
| 70B dense | ~140 GB | 1 | 1 | 4 |
| 685B MoE | ~340 GB | 2 | 1 | 4 |
| 1T MoE | ~595 GB | 4 | 1 | 4 |

## Step 3 — Choose the hidden state dump backend

| Backend | Script | When to use |
|---------|--------|-------------|
| vLLM | `common/eagle3/dump_offline_data_vllm.sh` | Default; broad coverage via vLLM + speculators |
| HF | `common/eagle3/dump_offline_data_hf.sh` | VLMs, custom-code models, SWA attention |
| TRT-LLM | `common/eagle3/dump_offline_data.sh` | Pure-text models with TRT-LLM support (needs `--tp`/`--moe-ep`) |

Use **HF** when the model is a VLM or uses sliding window attention (TRT-LLM does not support these).
Use **vLLM** for everything else as the default.

## Step 4 — Write the YAML

Create `tools/launcher/examples/<Org>/<Model>/hf_offline_eagle3.yaml`.
Use an existing config as a reference (e.g., `tools/launcher/examples/Qwen/Qwen3.5-35B-A3B/hf_offline_eagle3.yaml`).

### Header comment

```yaml
# EAGLE3 offline speculative decoding pipeline for <org>/<model>.
#
# <Model> is a <size> <dense|MoE> model. <brief notes: attention type, special reqs>
# BF16 weights ~<size> GB — fits on <N> GB200 node(s) (<N> × 192 GB).
#
# <Special requirements (if any)>
#
# 4-step pipeline:
#   task_0: Data synthesis — query vLLM server to generate prompt samples
#   task_1: Dump hidden states — run target model to capture hidden states
#   task_2: Offline training — train the EAGLE3 draft head
#   task_3: Benchmark — evaluate speculative decoding speedup via VLLM
#
# Usage:
#   uv run launch.py --yaml examples/<Org>/<Model>/hf_offline_eagle3.yaml --yes
#   uv run slurm.py --yaml modules/Model-Optimizer/tools/launcher/examples/<Org>/<Model>/hf_offline_eagle3.yaml --yes

job_name: <Model>_EAGLE3_offline
pipeline:
  allow_to_fail: false
  skip: false
  note:

  global_vars:
    hf_model: /hf-local/<org>/<model>
```

### task_0 — Data synthesis (`common/vllm/query.sh`)

Args before `--` go to the vLLM server; args after `--` go to `query.py`.

```yaml
  task_0:
    script: common/vllm/query.sh
    args:
      - --model <<global_vars.hf_model>>
      - --tensor-parallel-size <TP>
      - --trust-remote-code          # add only if required
      - --                           # separator
      - --data /hf-local/modelopt/Speculative-Decoding-Dataset-v2-default
      - --save /scratchspace/data
    environment:
      - HF_LOCAL: /hf-local
    slurm_config:
      _factory_: "slurm_factory"
      nodes: <nodes>
      ntasks_per_node: 1
      gpus_per_node: 4
      container: vllm/vllm-openai:latest
```

### task_1 — Hidden states (vLLM backend, default)

```yaml
  task_1:
    script: common/eagle3/dump_offline_data_vllm.sh
    args:
      - --input-data /scratchspace/data
      - --output-dir /scratchspace/offline_hidden_states
      - --max-seq-len 8192
    environment:
      - HF_MODEL_CKPT: <<global_vars.hf_model>>
    slurm_config:
      _factory_: "slurm_factory"
      nodes: <nodes>
      ntasks_per_node: 1
      gpus_per_node: 4
      container: vllm/vllm-openai:latest
```

For **HF backend** (VLMs, SWA models), use `dump_offline_data_hf.sh` instead — same args, no TP flags needed.

For **TRT-LLM backend**, use `dump_offline_data.sh` and add `--tp <TP>` and `--moe-ep 1` (or appropriate EP).

### task_2 — Offline training (`common/eagle3/train_eagle.sh`)

```yaml
  task_2:
    script: common/eagle3/train_eagle.sh
    args:
      - --config modules/Model-Optimizer/modelopt_recipes/general/speculative_decoding/eagle3.yaml
      - model.model_name_or_path=<<global_vars.hf_model>>
      - data.offline_data_path=/scratchspace/offline_hidden_states
      - training.output_dir=/scratchspace/eagle3
      - training.training_seq_len=4096
      - training.disable_tqdm=true
      - training.ar_validate_steps=500000
    slurm_config:
      _factory_: "slurm_factory"
      nodes: 1
      ntasks_per_node: 1
      gpus_per_node: 4
      container: nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10
```

> **MoE note:** For MoE models with large per-expert hidden dims, consider increasing
> `intermediate_size` in `eagle_config.json` to match the model's `moe_intermediate_size`.

### task_3 — Benchmark (`common/specdec_bench/quick_check.sh`)

```yaml
  task_3:
    script: common/specdec_bench/quick_check.sh
    args:
      - --draft_model_dir /scratchspace/export
      - --draft_length 3
      - --output_length 4096
      - --engine VLLM
      - --tp_size <TP>
      - --ep_size 1
      - --speculative_algorithm EAGLE3
      - --mtbench /hf-local/HuggingFaceH4/mt_bench_prompts/raw/question.jsonl
      - --concurrency 1
    environment:
      - HF_LOCAL: /hf-local
      - HF_MODEL_CKPT: <<global_vars.hf_model>>
    slurm_config:
      _factory_: "slurm_factory"
      nodes: <nodes>
      ntasks_per_node: 1
      gpus_per_node: 4
      container: vllm/vllm-openai:latest
```

## Step 5 — Common model-specific adjustments

| Situation | What to change |
|---|---|
| Requires `--trust-remote-code` | Add to task_0 vLLM args (before `--`) |
| VLM / multimodal | Use `dump_offline_data_hf.sh` for task_1 |
| Sliding window attention | Use `dump_offline_data_hf.sh` or `_vllm.sh` for task_1 |
| MoE with large expert hidden dim | Increase `intermediate_size` in eagle_config.json |
| Non-standard attention (MLA) | Verify `eagle_decoder_type` in the eagle3 recipe YAML |
| Custom tokenizer (e.g., tiktoken) | Set `TIKTOKEN_RS_CACHE_DIR` env var in task_0 and task_1 |
| NVFP4 quant model | task_0/task_3 use quant container; task_1/task_2 use BF16 base model — add `hf_model_bf16` global_var |
| Model needs `trust_remote_code` at benchmark | Add `--trust-remote-code` to task_3 args |

## Step 6 — Test with dry run

Preview the resolved config before submitting:

```bash
uv run launch.py --yaml examples/<Org>/<Model>/hf_offline_eagle3.yaml --dryrun --yes -v
```

## Step 7 — Update triage chart

After adding a new model, add a row to the test matrix in
`tools/launcher/examples/EAGLE3_TRIAGE.md` with status 🔲 (not yet tested).
Fill in results after running.
