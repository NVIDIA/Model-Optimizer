# EAGLE3 Automation Triage

This document captures the failure modes observed when running the 4-step EAGLE3 offline
pipeline against a selection of 10 new models. It is structured so Claude (or any contributor)
can update the status table and diagram as new models are tested.

---

## Pipeline Overview

```
Model checkpoint (HuggingFace)
        │
        ▼
┌──────────────────┐
│  Task 0: Query   │  vLLM server generates prompt/response pairs
│  (data synthesis)│  Script: common/vllm/query.sh
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Task 1: Dump    │  Target model runs forward pass, saves hidden states
│  (hidden states) │  Script: common/eagle3/dump_offline_data.sh
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Task 2: Train   │  Draft head trained on hidden states (Accelerate + FSDP)
│  (EAGLE3 head)   │  Script: common/eagle3/train_eagle.sh
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Task 3: Bench   │  Speculative decoding benchmark via vLLM
│  (benchmark)     │  Script: common/specdec_bench/quick_check.sh
└──────────────────┘
```

---

## Triage Diagram

The following Mermaid diagram maps each step to its possible failure modes.
Each leaf node indicates a known issue and its root cause category.

```mermaid
flowchart TD
    START([Start: run pipeline]) --> T0

    T0[Task 0: Data synthesis\nvllm/query.sh] --> T0_OK{Success?}
    T0_OK -- Yes --> T1
    T0_OK -- No --> T0_FAIL

    T0_FAIL{Failure mode?}
    T0_FAIL -- Time limit exceeded --> T0_TIMEOUT[⚠ TIMEOUT\nJob wall-clock limit too short\nfor full dataset synthesis.\nFix: increase Slurm time limit\nor reduce dataset size.]
    T0_FAIL -- Vocab / tokenizer error --> T0_TOKENIZER[⚠ TOKENIZER\nMissing or misconfigured tokenizer cache.\ne.g. GPT-OSS-20B requires TIKTOKEN_RS_CACHE_DIR\nto be pre-populated.\nFix: set env var to valid cache path.]
    T0_FAIL -- vLLM model not supported --> T0_VLLM[⚠ VLLM_SUPPORT\nModel architecture not yet\nsupported in vllm/vllm-openai:latest.\nFix: use a newer vLLM version or\nswitch to TRT-LLM for task_0.]
    T0_FAIL -- trust_remote_code error --> T0_TRUST[⚠ TRUST_REMOTE_CODE\nModel requires custom code but\n--trust-remote-code was not passed.\nFix: add flag to task_0 args.]

    T1[Task 1: Dump hidden states\neagle3/dump_offline_data.sh] --> T1_OK{Success?}
    T1_OK -- Yes --> T2
    T1_OK -- No --> T1_FAIL

    T1_FAIL{Failure mode?}
    T1_FAIL -- Script not found --> T1_SCRIPT[⚠ MISSING_SCRIPT\ndump_offline_data_vllm.sh does not exist.\nFix: use dump_offline_data.sh or\ncreate the vllm variant script.]
    T1_FAIL -- Model not supported by TRT-LLM --> T1_TRTLLM[⚠ TRTLLM_SUPPORT\nTarget model architecture not\nsupported in the TRT-LLM container.\nFix: use vLLM-based dump script or\nupgrade TRT-LLM version.]
    T1_FAIL -- OOM / memory --> T1_OOM[⚠ OOM\nModel weights exceed GPU memory\nwith current TP/EP configuration.\nFix: increase TP, add EP, or use\na node with more GPUs.]
    T1_FAIL -- MLA / SWA attention unsupported --> T1_ARCH[⚠ ARCH\nSpecial attention (MLA, SWA) not\nsupported by dump script backend.\nFix: ensure vLLM backend supports\nthe model's attention variant.]

    T2[Task 2: Train EAGLE3 head\neagle3/train_eagle.sh] --> T2_OK{Success?}
    T2_OK -- Yes --> T3
    T2_OK -- No --> T2_FAIL

    T2_FAIL{Failure mode?}
    T2_FAIL -- HF upload on local path --> T2_HFUPLOAD[⚠ HF_UPLOAD_BUG\noffline_training.sh calls HF Hub upload\nwith a local path like /scratchspace/eagle3.\nFix: remove HF upload call; save locally only.\nTracked: offline_training.sh line 18.]
    T2_FAIL -- eagle_config mismatch --> T2_CONFIG[⚠ EAGLE_CONFIG\neagle_config.json intermediate_size\ntoo small for MoE model's expert hidden dim.\nFix: increase intermediate_size in\neagle_config.json for MoE targets.]
    T2_FAIL -- eagle_decoder_type incompatible --> T2_DECODER[⚠ DECODER_TYPE\nDefault llama decoder type incompatible\nwith target model attention (e.g. MLA).\nFix: set eagle_decoder_type in\neagle_config.json to match target.]
    T2_FAIL -- OOM during training --> T2_OOM[⚠ OOM\nFSDP training OOM with default batch size.\nFix: reduce train_bs or\nuse gradient checkpointing.]

    T3[Task 3: Benchmark\nspecdec_bench/quick_check.sh] --> T3_OK{Success?}
    T3_OK -- Yes --> END([Pipeline complete])
    T3_OK -- No --> T3_FAIL

    T3_FAIL{Failure mode?}
    T3_FAIL -- Export dir missing cascade --> T3_EXPORT[⚠ CASCADE\n/scratchspace/export does not exist\nbecause Task 2 failed.\nFix: resolve Task 2 failure first.]
    T3_FAIL -- trust_remote_code missing --> T3_TRUST[⚠ TRUST_REMOTE_CODE\nBenchmark script did not pass\n--trust-remote-code to vLLM.\nFix: add flag in quick_check.sh\nor pipeline config.]
    T3_FAIL -- vLLM spec decode not supported --> T3_VLLM[⚠ VLLM_SPECDEC\nSpeculative decoding with EAGLE3\nnot yet supported in vLLM for\nthis model architecture.\nFix: check vLLM release notes.]
    T3_FAIL -- NVFP4 requires newer vLLM --> T3_NVFP4[⚠ NVFP4\nNVFP4 inference requires\nvllm/vllm-openai:v0.15.0+\nand Blackwell GPU.\nFix: pin container version.]
```

---

## Model Test Results

Tests were run on the OCI-HSG cluster (GB200 nodes, 4 × 192 GB HBM3e per node).

| Model | Size | Task 0 | Task 1 | Task 2 | Task 3 | Notes |
|-------|------|--------|--------|--------|--------|-------|
| Ministral-3-8B | 8B dense | ⏱ TIMEOUT (near complete, 3277/3295 samples) | ❌ MISSING_SCRIPT | ❌ HF_UPLOAD_BUG | ❌ CASCADE | Tokenizer regex warning (non-fatal) |
| Ministral-3-14B | 14B dense | ⏱ TIMEOUT | ❌ MISSING_SCRIPT | ❌ HF_UPLOAD_BUG | ❌ (no log) | — |
| GPT-OSS-20B | 20B dense | ❌ TOKENIZER (tiktoken cache missing) | ❌ MISSING_SCRIPT | ❌ HF_UPLOAD_BUG | ❌ CASCADE | vLLM tried to start but vocab load failed |
| MiniMax-M2.5 | 230B MoE | ⏱ TIMEOUT | ❌ MISSING_SCRIPT | ❌ HF_UPLOAD_BUG (config.json variant) | ❌ TRUST_REMOTE_CODE | trust_remote_code needed at benchmark |
| Qwen3.5-35B-A3B | 35B MoE | ⏱ TIMEOUT | ❌ MISSING_SCRIPT | ❌ HF_UPLOAD_BUG | ❌ CASCADE | — |
| Step-3.5-Flash | 197B MoE | ⏱ TIMEOUT | ❌ MISSING_SCRIPT | ❌ HF_UPLOAD_BUG | ❌ CASCADE | SWA attention - untested past task_1 |
| DeepSeek-V3.2 | 685B MoE | 🔍 (no log, tarball only) | ❌ MISSING_SCRIPT | ❌ HF_UPLOAD_BUG | ❌ CASCADE | 2-node job, task_1 also OOM-terminated (signal 15) |
| Kimi-K2.5 | 1T MoE | 🔲 Not tested | 🔲 Not tested | 🔲 Not tested | 🔲 Not tested | MLA attention needs decoder_type check |
| GLM-5 | 744B MoE | 🔲 Not tested | 🔲 Not tested | 🔲 Not tested | 🔲 Not tested | Gated model, 2-node |
| Kimi-K2.5-NVFP4 | NVFP4 quant | 🔲 Not tested | 🔲 Not tested | 🔲 Not tested | 🔲 Not tested | Blackwell required; tasks 1-2 use BF16 base |

**Legend:** ✅ Pass · ❌ Fail · ⏱ Timeout · 🔍 Inconclusive · 🔲 Not yet tested

---

## Known Issues

### Issue 1: `dump_offline_data_vllm.sh` does not exist (Task 1 — universal)

**Symptom:** `/usr/bin/bash: services/pipeline/eagle3/dump_offline_data_vllm.sh: No such file or directory`

**Affected:** All 7 models tested.

**Root cause:** Quick-fail configs reference a vLLM-based hidden state dump script that was
planned but never created. The existing script `dump_offline_data.sh` uses TRT-LLM for inference.

**Fix options:**
- (a) Create `dump_offline_data_vllm.sh` using vLLM for hidden state extraction (enables models
  not supported by TRT-LLM, e.g. GPT-OSS-20B, Ministral3, Qwen3.5-VLMs).
- (b) For models supported by TRT-LLM: switch configs to use `dump_offline_data.sh` with
  appropriate `--tp` and `--moe-ep` flags.

---

### Issue 2: `offline_training.sh` HuggingFace Hub upload bug (Task 2 — universal)

**Symptom:**
```
huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name' or
'namespace/repo_name': '/scratchspace/eagle3'.
services/pipeline/eagle3/offline_training.sh: line 18: error_handler: command not found
```

**Affected:** All 7 models tested.

**Root cause:** `offline_training.sh` calls a HuggingFace CLI upload command with
`/scratchspace/eagle3` as the repo ID. This path should be a local output directory, not a
Hub repo. The `error_handler` function is also referenced but not defined.

**Fix:** Remove or gate the Hub upload call in `offline_training.sh`. Save the trained draft
head to a local path only during CI. Also define `error_handler` or remove the reference.

---

### Issue 3: Time limit exceeded (Task 0 — most models)

**Symptom:** `STEP CANCELLED AT ... DUE TO TIME LIMIT`

**Affected:** Ministral-3-8B (near complete), Ministral-3-14B, MiniMax-M2.5, Qwen3.5-35B-A3B,
Step-3.5-Flash.

**Root cause:** The default Slurm time limit is too short for synthesizing the full dataset
with a large model. Ministral-3-8B reached 3277/3295 samples before cancellation.

**Fix options:**
- (a) Increase Slurm `--time` in the job config.
- (b) Reduce the dataset size for quick-fail validation (e.g., use a 100-sample subset).
- (c) Add `--max-samples N` flag to `query.py` / `query.sh`.

---

### Issue 4: GPT-OSS-20B tokenizer cache missing (Task 0 — model-specific)

**Symptom:**
```
openai_harmony.HarmonyError: error downloading or loading vocab file:
failed to download or load vocab file
```

**Affected:** GPT-OSS-20B only.

**Root cause:** GPT-OSS-20B uses a custom OpenAI tokenizer (`openai_harmony`) that reads from
a tiktoken cache directory. The env var `TIKTOKEN_RS_CACHE_DIR` must point to a pre-populated
directory on the cluster.

**Fix:** Ensure `TIKTOKEN_RS_CACHE_DIR` is set to a valid, pre-populated tiktoken cache path
in the cluster environment before running task_0.

---

### Issue 5: `trust_remote_code` not passed to benchmark (Task 3 — MiniMax)

**Symptom:**
```
ValueError: The repository ... contains custom code which must be executed to correctly
load the model. Please pass trust_remote_code=True.
```

**Affected:** MiniMax-M2.5 task_3.

**Root cause:** `quick_check.sh` does not forward `--trust-remote-code` to the vLLM benchmark
process for models that require it.

**Fix:** Add `--trust-remote-code` to the `quick_check.sh` vLLM invocation when
`trust_remote_code` is set in the pipeline config.

---

### Issue 6: DeepSeek-V3.2 task_1 OOM (Task 1 — model-specific)

**Symptom:** `pyxis: child terminated with signal 15` (SIGTERM, likely OOM-triggered)

**Affected:** DeepSeek-V3.2 only (685B MoE).

**Root cause:** The 2-node job for DeepSeek-V3.2 task_1 may have been OOM-killed. The model
requires careful TP/EP configuration across nodes.

**Fix:** Verify TP=8 across 2 nodes is correctly configured; check for `dump_offline_data.sh`
multi-node support. Also blocked by Issue 1 (missing vllm script).

---

## How to Update This Document

When a new model completes testing:

1. Update the **Model Test Results** table — add a row or change status symbols.
2. If a new failure mode is found not in the diagram, add a new leaf node to the Mermaid chart
   under the appropriate step.
3. If a new issue pattern is discovered, add a new **Issue N** section with symptom, affected
   models, root cause, and fix.
4. If an issue is resolved, mark it as ✅ in both the table and the issue section.
