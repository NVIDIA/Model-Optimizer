# EAGLE3 Automation Triage Chart

This document tracks failure modes discovered when running the 4-step EAGLE3 offline
pipeline against 10 new models. Updated as models are tested.
Claude can update the status table, diagram, and issue catalog when new results arrive.

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
         │ (afterany — downstream tasks run even if this times out)
         ▼
┌──────────────────┐
│  Task 1: Dump    │  Target model runs forward pass, saves hidden states
│  (hidden states) │  Script: common/eagle3/dump_offline_data.sh  (TRT-LLM)
└────────┬─────────┘        or  dump_offline_data_hf.sh  (HF/vLLM fallback)
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

## Triage Decision Tree

```mermaid
flowchart TD
    START([EAGLE3 Pipeline Failed]) --> WHICH_STEP{Which step failed?}

    WHICH_STEP -->|task_0: Data synthesis| T0_CHECK{Server started?}
    WHICH_STEP -->|task_1: Hidden states| T1_CHECK{Script found?}
    WHICH_STEP -->|task_2: Training| T2_CHECK{Dependencies installed?}
    WHICH_STEP -->|task_3: Benchmark| T3_CHECK{Engine started?}

    %% ── task_0 ──────────────────────────────────────────────────
    T0_CHECK -->|No - hangs at health check| T0_OOM{CUDA OOM in log?}
    T0_CHECK -->|Yes - server up, query fails| T0_QUERY[Check query.py errors:\nbad prompt format,\nconnection timeout,\nempty response]
    T0_OOM -->|Yes| T0_FIX_OOM[⚠ OOM\nReduce max_num_tokens\nor increase TP]
    T0_OOM -->|No| T0_ARCH{Error type?}
    T0_ARCH -->|vocab / tokenizer error| T0_TOKENIZER[⚠ TOKENIZER\nMissing tokenizer cache.\ne.g. GPT-OSS-20B needs\nTIKTOKEN_RS_CACHE_DIR pre-populated]
    T0_ARCH -->|Architecture / RuntimeError| T0_FIX_ARCH[⚠ VLLM_SUPPORT\nModel arch not supported\nin this vLLM version.\nTry newer container.]
    T0_ARCH -->|trust_remote_code| T0_FIX_TRC[⚠ TRUST_REMOTE_CODE\nAdd --trust-remote-code\nbefore -- separator in args]
    T0_CHECK -->|Cancelled - time limit| T0_TIMEOUT[⚠ TIMEOUT\nJob wall-clock limit too short.\nNote: afterany deps ensure\ntask_1 still runs.\nFix: increase time limit\nor reduce dataset size.]

    %% ── task_1 ──────────────────────────────────────────────────
    T1_CHECK -->|No - script not found| T1_SCRIPT[⚠ MISSING_SCRIPT\ndump_offline_data_vllm.sh does not exist.\nUse dump_offline_data_hf.sh\n(HF device_map=auto, no TP/EP flags)\nor dump_offline_data.sh\n(TRT-LLM, needs --tp / --moe-ep).]
    T1_CHECK -->|Yes| T1_RUN{Runs OK?}
    T1_RUN -->|No - OOM| T1_OOM[⚠ OOM\nIncrease TP, add EP,\nor switch to _hf script.]
    T1_RUN -->|No - NCCL error| T1_NCCL[⚠ NCCL\nNetwork/multi-node issue.\nRetry or reduce EP.]
    T1_RUN -->|No - arch unsupported| T1_ARCH[⚠ ARCH\nModel not supported by TRT-LLM.\nSwitch to dump_offline_data_hf.sh.]
    T1_RUN -->|Yes - no .pt output| T1_DATA[Check --input-data path\nand data format from task_0]

    %% ── task_2 ──────────────────────────────────────────────────
    T2_CHECK -->|No - pip install fails| T2_FIX_DEPS[Network issue in container.\nCheck proxy/mirror.]
    T2_CHECK -->|Yes| T2_TRAIN{Training starts?}
    T2_TRAIN -->|No - ImportError| T2_FIX_IMPORT[modelopt not installed\nor wrong version]
    T2_TRAIN -->|No - FileNotFoundError| T2_FIX_DATA[task_1 output missing.\nRe-run task_1.]
    T2_TRAIN -->|Yes but crashes| T2_CRASH{Error type?}
    T2_CRASH -->|OOM| T2_FIX_OOM[⚠ OOM\nReduce train_bs\nor training_seq_len]
    T2_CRASH -->|NaN loss| T2_FIX_NAN[Reduce lr.\nCheck data quality.]
    T2_CRASH -->|KeyError / arch| T2_FIX_EAGLE[⚠ ARCH\nModel type not recognized\nby EAGLE3 training code.\nNeeds code change in modelopt.\nCheck eagle_decoder_type in config.]
    T2_TRAIN -->|Yes - export fails| T2_FIX_EXPORT[Check /scratchspace/eagle3\nhas model.safetensors]

    %% ── task_3 ──────────────────────────────────────────────────
    T3_CHECK -->|No - export dir missing| T3_EXPORT[⚠ CASCADE\nTask 2 failed or timed out.\nResolve task_2 first.]
    T3_CHECK -->|No - engine crash| T3_ENGINE{Engine type?}
    T3_CHECK -->|Yes - AR below threshold| T3_AR[AR too low:\nneed more epochs, data,\nor larger draft head]
    T3_CHECK -->|Yes - wrong output| T3_FORMAT[Check draft model\nconfig.json vs engine version]
    T3_ENGINE -->|vLLM - trust_remote_code| T3_TRUST[⚠ TRUST_REMOTE_CODE\nAdd --trust-remote-code\nto quick_check.sh invocation]
    T3_ENGINE -->|vLLM - spec decode unsupported| T3_VLLM[⚠ VLLM_SPECDEC\nvLLM version too old.\nUse latest container.]
    T3_ENGINE -->|NVFP4 - unsupported| T3_NVFP4[⚠ NVFP4\nRequires vllm-openai:v0.15.0+\nand Blackwell GPU.]
    T3_ENGINE -->|OOM| T3_FIX_OOM[Target + draft too large.\nIncrease TP.]
```

---

## Model Test Matrix

Tests run on OCI-HSG cluster (GB200 nodes, 4 × 192 GB HBM3e per node).

| # | Model | Type | Size | task_0 | task_1 | task_2 | task_3 | Notes |
|---|-------|------|------|--------|--------|--------|--------|-------|
| 1 | Ministral-3-8B | Dense | 8B | ⏱ TIMEOUT (3277/3295) | ❌ MISSING_SCRIPT | ❌ (no data from t1) | ❌ CASCADE | Tokenizer regex warning (non-fatal) |
| 2 | Ministral-3-14B | Dense | 14B | ⏱ TIMEOUT | ❌ MISSING_SCRIPT | ❌ (no data from t1) | 🔍 (no log) | — |
| 3 | GPT-OSS-20B | Dense | 20B | ❌ TOKENIZER | ❌ MISSING_SCRIPT | ❌ (no data from t1) | ❌ CASCADE | TIKTOKEN_RS_CACHE_DIR not populated |
| 4 | MiniMax-M2.5 | MoE | 230B/10B | ⏱ TIMEOUT | ❌ MISSING_SCRIPT | ❌ (no data from t1) | ❌ TRUST_REMOTE_CODE | trust_remote_code needed at bench |
| 5 | Qwen3.5-35B-A3B | MoE | 35B/3B | ⏱ TIMEOUT | ❌ MISSING_SCRIPT | ❌ (no data from t1) | ❌ CASCADE | — |
| 6 | Step-3.5-Flash | MoE/SWA | 197B/11B | ⏱ TIMEOUT | ❌ MISSING_SCRIPT | ❌ (no data from t1) | ❌ CASCADE | SWA attention — untested past t1 |
| 7 | DeepSeek-V3.2 | MoE/MLA | 685B/37B | 🔍 (tarball only) | ❌ MISSING_SCRIPT + OOM | ❌ (no data from t1) | ❌ CASCADE | 2-node, t1 OOM-killed (SIGTERM) |
| 8 | Kimi-K2.5 | MoE/MLA | 1T/32B | 🔲 | 🔲 | 🔲 | 🔲 | MLA attention: verify eagle_decoder_type |
| 9 | GLM-5 | MoE/DSA | 744B/40B | 🔲 | 🔲 | 🔲 | 🔲 | Gated, 2-node |
| 10 | Kimi-K2.5-NVFP4 | NVFP4 | ~591GB | 🔲 | 🔲 | 🔲 | 🔲 | Blackwell required; t1/t2 use BF16 base |

**Legend:** ✅ Pass · ❌ Fail · ⏱ Timeout · 🔍 Inconclusive · 🔲 Not yet tested

---

## Known Issues

### Issue 1: Missing `dump_offline_data_vllm.sh` (Task 1 — universal) — OPEN

**Symptom:** `/usr/bin/bash: .../dump_offline_data_vllm.sh: No such file or directory`

**Affected:** All 7 models tested (root cause of universal task_1 failure).

**Root cause:** Quick-fail pipeline configs reference `dump_offline_data_vllm.sh`, which was
planned but not created. Two scripts exist: `dump_offline_data.sh` (TRT-LLM based, requires
`--tp`/`--moe-ep`) and `dump_offline_data_hf.sh` (HF `device_map="auto"`, no parallelism args,
works for any model supported by HF Transformers).

**Status:** `dump_offline_data_hf.sh` was created as a fallback and is working for standalone
task_1 re-runs (Ministral-3-8B, MiniMax-M2.5, Qwen3.5-35B-A3B, Step-3.5-Flash). The
quick-fail pipeline configs still reference the non-existent `_vllm` script.

**Fix:** Update quick-fail configs to use `dump_offline_data_hf.sh` for models not supported
by TRT-LLM, or rename `_hf` → `_vllm` if it covers the intended use case.

---

### Issue 2: `offline_training.sh` HuggingFace Hub upload bug — FIXED ✅

**Was:** `HFValidationError: Repo id must be in the form 'repo_name': '/scratchspace/eagle3'`

**Fix applied:** `offline_training.sh` was rewritten to call `launch_train.sh` followed by
`export_hf_checkpoint.py` for local export only. No HF Hub upload. The `error_handler` is
now properly sourced from `service_utils.sh`.

---

### Issue 3: Task 0 time limit (most models) — PARTIALLY ADDRESSED ⚠

**Symptom:** `STEP CANCELLED AT ... DUE TO TIME LIMIT`

**Affected:** Ministral-3-8B (3277/3295 samples — nearly complete), Ministral-3-14B,
MiniMax-M2.5, Qwen3.5-35B-A3B, Step-3.5-Flash.

**Status:** `afterany` Slurm dependencies were added so downstream tasks (task_1, 2, 3)
run even when task_0 times out. The data synthesis timeout itself is not yet resolved.

**Fix options:**
- Increase Slurm `--time` limit for task_0.
- Add `--max-samples N` to limit dataset size for quick-fail validation.

---

### Issue 4: GPT-OSS-20B tokenizer cache missing (Task 0) — OPEN

**Symptom:** `openai_harmony.HarmonyError: error downloading or loading vocab file`

**Affected:** GPT-OSS-20B only. vLLM started (model loaded) but vocab download failed.

**Root cause:** GPT-OSS-20B uses the `openai_harmony` tokenizer backed by tiktoken, which
requires `TIKTOKEN_RS_CACHE_DIR` to point to a pre-populated local cache. The cluster did
not have this directory populated.

**Fix:** Ensure `TIKTOKEN_RS_CACHE_DIR` is set to a valid pre-populated tiktoken cache
path before submitting task_0.

---

### Issue 5: MiniMax-M2.5 missing `trust_remote_code` at benchmark (Task 3) — OPEN

**Symptom:**
```
ValueError: The repository ... contains custom code... Please pass trust_remote_code=True
```

**Affected:** MiniMax-M2.5 task_3.

**Root cause:** `quick_check.sh` does not forward `--trust-remote-code` to vLLM for models
that require it.

**Fix:** Pass `--trust-remote-code` in the `quick_check.sh` vLLM invocation when
`trust_remote_code` is set in the pipeline environment.

---

### Issue 6: DeepSeek-V3.2 task_1 OOM (Task 1) — OPEN

**Symptom:** `pyxis: child terminated with signal 15` (SIGTERM, likely OOM-triggered)

**Affected:** DeepSeek-V3.2 only (685B MoE, 2-node job).

**Root cause:** Task_1 was also blocked by Issue 1 (missing vllm script); the SIGTERM may
indicate OOM during the brief moment before the script-not-found failure propagated. Needs
further investigation with `dump_offline_data_hf.sh`.

---

## How to Update This Document

When a new model completes testing:

1. **Status table**: Update the row — fill in ✅/❌/⏱/🔍 and brief notes.
2. **Decision tree**: If a new failure mode appears that has no matching leaf, add a new
   branch under the appropriate step node.
3. **Issue catalog**: Add a new numbered section with symptom, affected models, root cause,
   fix, and status (OPEN / FIXED / PARTIALLY ADDRESSED).
4. Mark resolved issues as **FIXED ✅** and update the status in the table.

Per-model results template:
```markdown
#### Model: <name>
- **Date tested:** YYYY-MM-DD
- **task_0:** PASS/FAIL/TIMEOUT — <notes>
- **task_1:** PASS/FAIL — <notes>
- **task_2:** PASS/FAIL — <notes>
- **task_3:** PASS/FAIL — <notes>
- **AR speedup:** <value> (target ≥ 2.1×)
- **New failure pattern:** Yes/No — <description if yes>
```
