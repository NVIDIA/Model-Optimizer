# EAGLE3 Automation Triage Chart

This document tracks failure modes discovered when running the 4-step EAGLE3 offline
pipeline against 10 new models. Updated as models are tested.
Claude can update the status table, diagram, and issue catalog when new results arrive.

---

## Pipeline Overview

```text
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
│  (hidden states) │  Script: common/eagle3/dump_offline_data.sh       (TRT-LLM)
└────────┬─────────┘        or  dump_offline_data_hf.sh   (HF device_map=auto)
                           or  dump_offline_data_vllm.sh  (vLLM native extractor)
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
    T1_CHECK -->|No - script not found| T1_SCRIPT[⚠ MISSING_SCRIPT\nVerify script path. Three backends:\n• dump_offline_data_vllm.sh (vLLM native extractor)\n• dump_offline_data_hf.sh (HF device_map=auto)\n• dump_offline_data.sh (TRT-LLM, --tp/--moe-ep)]
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
| 1 | Ministral-3-8B | Dense | 8B | 🔁 RERUNNING (--num-shards 3) | 🔁 RERUNNING (vLLM native extractor) | 🔲 | 🔲 | Issues 6+7 fixed; re-run in progress |
| 2 | Ministral-3-14B | Dense | 14B | ⏱ TIMEOUT | 🔁 NEEDS RERUN (_vllm) | 🔲 | 🔲 | — |
| 3 | GPT-OSS-20B | Dense | 20B | ❌ TOKENIZER | 🔁 NEEDS RERUN (_vllm) | 🔲 | 🔲 | Fix: populate TIKTOKEN_RS_CACHE_DIR first |
| 4 | MiniMax-M2.5 | MoE | 230B/10B | ⏱ TIMEOUT | 🔁 NEEDS RERUN (_vllm) | 🔲 | ❌ TRUST_REMOTE_CODE | trust_remote_code needed at bench |
| 5 | Qwen3.5-35B-A3B | MoE | 35B/3B | ⏱ TIMEOUT | 🔁 NEEDS RERUN (_vllm) | 🔲 | 🔲 | — |
| 6 | Step-3.5-Flash | MoE/SWA | 197B/11B | ⏱ TIMEOUT | 🔁 NEEDS RERUN (_vllm) | 🔲 | 🔲 | SWA: use _vllm or_hf script |
| 7 | DeepSeek-V3.2 | MoE/MLA | 685B/37B | 🔍 (tarball only) | 🔁 NEEDS RERUN (_vllm, 2-node) | 🔲 | 🔲 | 2-node; previous t1 OOM-killed |
| 8 | Kimi-K2.5 | MoE/MLA | 1T/32B | 🔲 | 🔲 | 🔲 | 🔲 | MLA attention: verify eagle_decoder_type |
| 9 | GLM-5 | MoE/DSA | 744B/40B | 🔲 | 🔲 | 🔲 | 🔲 | Gated, 2-node |
| 10 | Kimi-K2.5-NVFP4 | NVFP4 | ~591GB | 🔲 | 🔲 | 🔲 | 🔲 | Blackwell required; t1/t2 use BF16 base |

**Legend:** ✅ Pass · ❌ Fail · ⏱ Timeout · 🔍 Inconclusive · 🔲 Not yet tested · 🔁 Rerun needed

---

## Known Issues

### Issue 1: Missing `dump_offline_data_vllm.sh` (Task 1 — universal) — FIXED ✅

**Symptom:** `/usr/bin/bash: .../dump_offline_data_vllm.sh: No such file or directory`

**Affected:** All 7 models tested (root cause of universal task_1 failure in first round).

**Root cause:** Quick-fail pipeline configs referenced `dump_offline_data_vllm.sh`, which had
not yet been created. Only two scripts existed: `dump_offline_data.sh` (TRT-LLM) and
`dump_offline_data_hf.sh` (HF `device_map="auto"`).

**Fix applied:** `dump_offline_data_vllm.sh` and its backing script
`compute_hidden_states_vllm.py` were added. The vLLM script drives vLLM's built-in
`extract_hidden_states` speculative method (via the `ExampleHiddenStatesConnector` KV
connector) and saves output in the same `.pt` format as the HF variant. No third-party
data-generation dependency is required. Both files are now in:
- `tools/launcher/common/eagle3/dump_offline_data_vllm.sh`
- `examples/speculative_decoding/collect_hidden_states/compute_hidden_states_vllm.py`

Three backends now available for task_1:

| Backend | Script | When to use |
|---------|--------|-------------|
| TRT-LLM | `dump_offline_data.sh` | Pure-text models with TRT-LLM support; needs `--tp`/`--moe-ep` |
| HF | `dump_offline_data_hf.sh` | VLMs, custom-code models, SWA; `device_map="auto"` |
| vLLM | `dump_offline_data_vllm.sh` | Broad coverage via vLLM model implementations; uses vLLM's native extractor |

---

### Issue 2: Training-step HuggingFace Hub upload bug — FIXED ✅

**Was:** `HFValidationError: Repo id must be in the form 'repo_name': '/scratchspace/eagle3'`

**Fix applied:** The training step (`common/eagle3/train_eagle.sh`) trains and then exports the
HF checkpoint to a local path only — no HF Hub upload — and sources `error_handler` from
`service_utils.sh`.

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

```text
ValueError: The repository ... contains custom code... Please pass trust_remote_code=True
```

**Affected:** MiniMax-M2.5 task_3.

**Root cause:** `quick_check.sh` does not forward `--trust-remote-code` to vLLM for models
that require it.

**Fix:** Pass `--trust-remote-code` in the `quick_check.sh` vLLM invocation when
`trust_remote_code` is set in the pipeline environment.

---

### Issue 6: `speculators` dependency removed — superseded by vLLM's native extractor ✅

**History:** The vLLM dump path originally used `VllmHiddenStatesGenerator` from the
`speculators` library. This was brittle: `speculators==0.5.0` removed that class
(`ImportError: cannot import name 'VllmHiddenStatesGenerator'`), forcing a
`pip install "speculators<0.5.0"` pin plus several runtime source-patches of the
installed library for vLLM/pydantic compatibility.

**Resolution:** `dump_offline_data_vllm.sh` / `compute_hidden_states_vllm.py` now use
vLLM's built-in `extract_hidden_states` speculative method (via the
`ExampleHiddenStatesConnector` KV connector). The `speculators` dependency and all
runtime source-patches were removed, so the version pin and these compatibility
patches no longer apply.

---

### Issue 7: `query.py` auto-downgrades shards → empty data on timeout (Task 0) — FIXED ✅

**Symptom:** task_0 times out; `/scratchspace/data/` is empty despite partial generation.

**Affected:** Models with datasets ≤ 33,000 samples where `num_shards * 100 > dataset_size`.

**Root cause:** `query.py` auto-downgrades `--num-shards` to `min(16, dataset_size//100)`
when the default of 1000 is too large relative to dataset size. For 3295 samples this
becomes 1 shard, meaning all data is processed in one batch and nothing is saved until
the entire map completes. A timeout yields zero data.

**Fix applied:** Pass `--num-shards 3` explicitly in task_0 args. Since `3*100=300 < 3295`,
the auto-downgrade is bypassed. Data is saved incrementally across 3 shard files (~1100
samples each). Partial data survives a timeout.

---

### Issue 8: DeepSeek-V3.2 task_1 OOM (Task 1) — OPEN

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
