---
name: eagle3-new-model
description: >
  Add a new model to the EAGLE3 offline pipeline. Generates an hf_offline_eagle3.yaml
  launcher config for a new model checkpoint, choosing the right hidden state dump
  backend (TRT-LLM / HF / vLLM) and GPU configuration.
  Use when user wants to run EAGLE3 on a model that does not yet have a YAML in
  tools/launcher/examples/ or asks how to configure the pipeline for a new checkpoint.
user_invocable: true
---

# EAGLE3 New Model Configuration

Create `tools/launcher/examples/<Org>/<Model>/hf_offline_eagle3.yaml` by **copying the
closest existing example and adapting it**. Pick a reference with the same shape as the
target (dense vs MoE, similar size) from `tools/launcher/examples/` — e.g. the Qwen3-8B
config for a dense model.

The pipeline is a 4-task config (`task_0` data synthesis → `task_1` hidden-state dump →
`task_2` train → `task_3` benchmark). The task structure, args, containers, and GPU/node
sizing are all visible in the existing examples — infer them from a reference rather than
hand-rolling. This file documents only the two things that are **not** obvious from the
examples: which dump backend to pick, and the model-specific gotchas.

## Choosing the `task_1` hidden-state dump backend

| Backend | Script | When to use |
|---------|--------|-------------|
| vLLM | `common/eagle3/dump_offline_data_vllm.sh` | **Default.** Broad coverage via vLLM's native hidden-state extractor. |
| HF | `common/eagle3/dump_offline_data_hf.sh` | VLMs / multimodal, custom-code models, sliding-window attention (TRT-LLM can't serve these). |
| TRT-LLM | `common/eagle3/dump_offline_data.sh` | Pure-text models with TRT-LLM support; pass `--tp <TP>` and `--moe-ep <EP>`. |

Rule of thumb: **HF** if the model is a VLM or uses sliding-window attention; **vLLM**
otherwise. TRT-LLM only when you specifically want its kernels for a supported plain-text model.

## Model-specific adjustments

These are the non-obvious knobs that vary per model:

| Situation | What to change |
|---|---|
| Requires `--trust-remote-code` | Add to `task_0` vLLM args (before the `--` separator) and to `task_3` benchmark args |
| MoE with large expert hidden dim | Increase `intermediate_size` in `eagle_config.json` to match `moe_intermediate_size` |
| Custom tokenizer (e.g. tiktoken) | Set `TIKTOKEN_RS_CACHE_DIR` env var in `task_0` and `task_1` |

After adapting the config, preview it with `--dryrun` before submitting.
