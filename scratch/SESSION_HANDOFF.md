<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVFP4 activation calibration study — session handoff

This file captures where we are so a future session (potentially against a
different debugger / Docker server, e.g. vLLM instead of TRT-LLM) can
continue without losing context.

## Where we are

- **Branch:** `chenjiel/nvfp4-activation-calib-study` (this branch).
- **Draft PR:** <https://github.com/NVIDIA/Model-Optimizer/pull/1545>.
- **Last commit:** `Scratch: NVFP4 activation input_scale calibration study`.
- **Main report:** [`nvfp4_activation_calib_report.md`](nvfp4_activation_calib_report.md)
  — self-contained, includes all results so far. Read this first.

## What the study has answered

The PR-1545 report's TL;DR, condensed:

1. **Calibration size on real Qwen3.5-9B MLP inputs is essentially flat.**
   Going from 128 → 1024 sequences moves SNR by ≤ 0.01 dB on layer 0 / 31.
2. **The amax default is optimal-enough** — within 0.017–0.018 dB of the
   MSE-optimal oracle on the clean shared-test experiment. No percentile
   policy below p99.999 matches it.
3. **Calibration dataset choice doesn't matter.** With a properly disjoint
   shared held-out test tensor, `cnn_nemotron_v2_mix` and
   `nemotron-post-training-v3` produce input_scale values within ~3% of
   each other, translating to ≤ 0.009 dB SNR spread on layer 31 (0.002 dB
   on layer 0).
4. **Calibration size *can* hurt in principle** — confirmed only on
   pathological synthetic distributions (rare giant 1e6 spike;
   log-normal σ=3). Not observed on any real Qwen3.5-9B / Nemotron-family
   combination.

All experiments use a **bf16-loaded model with PyTorch forward hooks on
`model.layers[{0,31}].mlp`**. The model itself was **not quantized** at
any point — we only studied NVFP4 reconstruction MSE on the captured
bf16 activations in isolation.

## Artifacts in `scratch/` (all committed to PR #1545)

| file | purpose |
|---|---|
| `nvfp4_activation_calib_report.md` | the comprehensive write-up |
| `nvfp4_activation_calib_mse.py` | synthetic distributions sweep (Part A) |
| `nvfp4_activation_calib_results.json` | raw curves for synthetic study |
| `capture_qwen35_mlp_activations.py` | per-combo activation capture (legacy / per-dataset path) |
| `nvfp4_real_activation_calib_mse.py` | per-combo sweep against same combo's own holdout |
| `nvfp4_real_activation_calib_results.json` | raw curves for the per-combo sweep |
| `capture_calib_and_test_split.py` | **clean experiment**: single-model-load capture of disjoint calib + held-out test from both combos |
| `nvfp4_shared_test_sweep.py` | **clean experiment**: applies each combo's amax to one shared held-out test tensor |
| `nvfp4_shared_test_sweep_results.json` | raw curves for the shared-test experiment |
| `_cross_dataset_amax_compare.py` | legacy: replay recorded amaxes against v3's own holdout (kept for transparency) |
| `SESSION_HANDOFF.md` | this file |

`.pt` activation files (~17 GB) are **not** committed — they're easy to
regenerate by running the capture scripts (~12 min each on one GPU). They
also live in `scratch/` if you stay in the same machine:

- `qwen35_cnn_nemotron_v2_mix_{calib,test}_layer{0,31}.pt`
- `qwen35_nemotron_post_training_v3_{calib,test}_layer{0,31}.pt`
- (older) `qwen35_9b_mlp_input_layer{0,31}.pt`

## Open follow-up question (where this hands off)

The chat ended with a discussion of whether to extend the study to a
**quantized** model — specifically NVFP4 W4A4 — so we can see how
input_scale calibration behaves when prior layers' weights and
activations are also NVFP4'd, not just bf16. Three possible paths:

### Option A — eager-mode PyTorch with modelopt quantization (in-place, TRT-LLM docker)

Apply modelopt's NVFP4 W4A4 to the bf16 model via `mtq.quantize`, then
reuse the existing forward-hook capture. Lowest setup cost. Lacks
production-kernel realism but gets activation-level numbers that are
right by construction.

### Option B — vLLM with `enforce_eager=True` + Python hooks (**recommended**)

Launch a vLLM docker running the NVFP4 W4A4 checkpoint with
`--enforce-eager`. Reach `llm.llm_engine.model_executor.driver_worker.model_runner.model`,
register `forward_pre_hook` on the equivalent `model.layers[{0,31}].mlp`,
run a single prompt at a time, save the captured tensors. ~10 lines of
Python; no source modification. Gives real NVFP4 kernels.

Gotchas to remember:
- vLLM uses a flattened token layout (`(total_tokens, hidden)`, no batch
  dim). Hook input shape differs from HF's `(B, S, H)`.
- Hooks fire on prefill AND each decode step. For a static study, run
  with `max_tokens=1` and capture only the prefill activations (filter
  by shape inside the hook).
- vLLM internals shift across versions — `print(model)` once to confirm
  the path to the MLP module.

### Option C — vLLM source patch (in-place, mounted repo)

`git clone vllm @ <docker-image-commit>`, `-v $PWD:/workspace/vllm`,
`pip uninstall vllm && pip install -e /workspace/vllm`. Edit
`vllm/model_executor/models/qwen3*.py`'s `DecoderLayer.forward` to dump
tensors at chosen points; restart vLLM. Heaviest but works under
CUDAGraph. Overkill for our question.

## To continue in a new session

If the new session is **another TRT-LLM / bf16 docker** (same setup as
this one), nothing changes. You can re-run any of the scripts directly:

```bash
# Re-capture the activations
python scratch/capture_calib_and_test_split.py --n_calib 1024 --n_test 256
python scratch/nvfp4_shared_test_sweep.py
```

If the new session is a **vLLM docker** to do Option B:

1. Confirm the NVFP4 W4A4 checkpoint path (the user will provide one — we
   never specified one in this session).
2. Inside vLLM container:

   ```python
   from vllm import LLM, SamplingParams
   llm = LLM(model=CKPT_PATH, quantization="modelopt_fp4",
             enforce_eager=True, dtype="bfloat16")
   model = llm.llm_engine.model_executor.driver_worker.model_runner.model
   # print(model) once to confirm the module path
   for lid in (0, 31):
       model.model.layers[lid].mlp.register_forward_pre_hook(make_hook(lid))
   llm.generate([PROMPT], SamplingParams(max_tokens=1))
   ```

3. Save the captured tensors to `scratch/qwen35_w4a4_vllm_layer{0,31}.pt`
   (or similar naming).
4. Adapt `nvfp4_shared_test_sweep.py` to load these new captures and
   compare against the existing bf16 results — same MSE/SNR table format
   but the underlying activations are now propagated through an NVFP4
   model.

## Environment notes (for the next session)

- **Model path:** `/models/Qwen/Qwen3.5-9B` (host path; mounted in the
  Docker container we've been using).
- **transformers version**: 5.9.0 (installed via
  `pip install -U 'transformers>=5.4,<6.0'`). Required for Qwen3.5.
- **PyTorch:** 2.11 nvidia-prebuilt. CUDA 13.1, NVIDIA RTX 6000 Ada.
- **modelopt:** 0.45.0.dev (the repo checkout we're working from).
- **Datasets:**
  - `abisee/cnn_dailymail` v3.0.0, `article` field, no chat template.
  - `nvidia/Nemotron-Post-Training-Dataset-v2`, `messages` field, chat
    template applied.
  - `nvidia/Nemotron-{Science,Math,SWE,Multilingual,...}-v*` for the v3
    combo. **Caveat:** `nvidia/Nemotron-SFT-Agentic-v2` (search split)
    emits no rows in streaming mode in this environment — known gotcha,
    documented in the report.

## TL;DR for the next agent

> The PR is a complete write-up. If the next task is "see what happens
> with a quantized model under vLLM," use Option B from the report.
> Otherwise, the study is done.
