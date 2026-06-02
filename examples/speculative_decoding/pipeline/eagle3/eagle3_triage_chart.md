# EAGLE3 Automation Triage Chart

This document catalogs failure modes observed when running the EAGLE3 pipeline across
different model architectures. Updated as new models are tested.

## Model Test Matrix

| # | Model | Type | Params | Status | task_0 | task_1 | task_2 | task_3 | Notes |
|---|-------|------|--------|--------|--------|--------|--------|--------|-------|
| 1 | Qwen3-8B | Dense | 8B | Existing | - | - | - | - | Reference model |
| 2 | Kimi-K2.5 | MoE | 1T/32B | Existing | - | - | - | - | GB200 required |
| 3 | Qwen3.5-9B | Dense (VLM) | 9B | Not run | - | - | - | - | Text-only path |
| 4 | Qwen3.5-27B | Dense (VLM) | 27B | Not run | - | - | - | - | Text-only path |
| 5 | Qwen3.5-35B-A3B | MoE (VLM) | 35B/3B | **Blocked** | TIMEOUT | FAIL | FAIL | FAIL | Data synth too slow; infra issues |
| 6 | MiniMax-M2.5 | MoE | 230B/10B | **Blocked** | TIMEOUT | FAIL | FAIL | FAIL | trust_remote_code needed |
| 7 | Ministral-3-8B | Dense (VLM) | 8B | **WIP** | SKIP | PASS | PASS | FAIL | `use_cache=null` in export; see below |
| 8 | Ministral-3-14B | Dense (VLM) | 14B | **Blocked** | FAIL | FAIL | FAIL | FAIL | vLLM engine init fails (NoneType) |
| 9 | DeepSeek-V3.2 | MoE (MLA) | 685B/37B | **Blocked** | no log | FAIL | FAIL | FAIL | No task_0 log; infra issues |
| 10 | gpt-oss-20b | Dense | 20B | **Blocked** | FAIL | FAIL | FAIL | FAIL | Tokenizer `HarmonyError` |
| 11 | Step-3.5-Flash | MoE (SWA) | 197B/11B | **Blocked** | TIMEOUT | FAIL | FAIL | FAIL | Data synth hit time limit |
| 12 | GLM-5 | MoE (DSA) | 744B/40B | Not run | - | - | - | - | 2 nodes, gated |

Legend: PASS / FAIL-{code} / SKIP / Pending

## Triage Decision Tree

```mermaid
flowchart TD
    START([EAGLE3 Pipeline Failed]) --> WHICH_STEP{Which step failed?}

    WHICH_STEP -->|task_0: Data Synthesis| T0_CHECK{Server started?}
    WHICH_STEP -->|task_1: Hidden States| T1_CHECK{TRT-LLM launched?}
    WHICH_STEP -->|task_2: Training| T2_CHECK{Dependencies installed?}
    WHICH_STEP -->|task_3: Benchmark| T3_CHECK{Engine started?}

    %% task_0 branch
    T0_CHECK -->|No - hangs at health check| T0_OOM{CUDA OOM in log?}
    T0_CHECK -->|Yes - server up but query fails| T0_QUERY[Check query.py errors:<br/>bad prompt format,<br/>connection timeout,<br/>empty response]
    T0_OOM -->|Yes| T0_FIX_OOM[Reduce max_num_tokens<br/>or increase TP]
    T0_OOM -->|No| T0_ARCH{Architecture error?}
    T0_ARCH -->|Yes - RuntimeError / KeyError| T0_FIX_ARCH[Model not supported by<br/>this TRT-LLM version.<br/>Try newer container.]
    T0_ARCH -->|No - trust_remote_code| T0_FIX_TRC[Add --trust_remote_code<br/>before -- separator]

    %% task_1 branch
    T1_CHECK -->|No - launch fails| T1_TRTLLM{Error type?}
    T1_CHECK -->|Yes - but no .pt output| T1_DATA[Check --input-data path<br/>and data format]
    T1_TRTLLM -->|OOM| T1_FIX_OOM[Increase TP or nodes]
    T1_TRTLLM -->|NCCL error| T1_FIX_NCCL[Network issue.<br/>Retry or reduce EP.]
    T1_TRTLLM -->|Unsupported model| T1_FIX_MODEL[Model arch not in TRT-LLM.<br/>Check release notes.]

    %% task_2 branch
    T2_CHECK -->|No - pip install fails| T2_FIX_DEPS[Network issue in container.<br/>Check proxy/mirror.]
    T2_CHECK -->|Yes| T2_TRAIN{Training starts?}
    T2_TRAIN -->|No - ImportError| T2_FIX_IMPORT[modelopt not installed<br/>or wrong version]
    T2_TRAIN -->|No - FileNotFoundError| T2_FIX_DATA[task_1 output missing.<br/>Re-run task_1.]
    T2_TRAIN -->|Yes but crashes| T2_CRASH{Error type?}
    T2_CRASH -->|OOM| T2_FIX_OOM[Reduce train_bs<br/>or training_seq_len]
    T2_CRASH -->|NaN loss| T2_FIX_NAN[Reduce lr.<br/>Check data quality.]
    T2_CRASH -->|KeyError / arch error| T2_FIX_EAGLE[Model type not recognized<br/>by EAGLE3 training code.<br/>Needs code change in modelopt.]
    T2_TRAIN -->|Yes - export fails| T2_FIX_EXPORT[Check /scratchspace/eagle3<br/>has model.safetensors]

    %% task_3 branch
    T3_CHECK -->|No - engine crash| T3_ENGINE{Engine type?}
    T3_CHECK -->|Yes - but AR below threshold| T3_AR[AR too low: need more<br/>epochs, data, or<br/>larger draft head]
    T3_CHECK -->|Yes - but wrong output| T3_FORMAT[Check draft model<br/>config.json compatibility<br/>with engine version]
    T3_ENGINE -->|vLLM - ImportError eagle| T3_FIX_VLLM[vLLM version too old.<br/>Use latest container.]
    T3_ENGINE -->|TRT-LLM - build error| T3_FIX_TRTLLM[Draft model format<br/>incompatible. Check export.]
    T3_ENGINE -->|OOM| T3_FIX_OOM[Target + draft too large.<br/>Increase TP.]
```

## Observed Failure Catalog

This section is updated as models are tested. Each entry records the model, step,
error, root cause, and resolution.

### Architecture-Level Failures

| Category | Affected Models | Step | Error | Root Cause | Resolution |
|----------|----------------|------|-------|------------|------------|
| VLM text-only | Qwen3.5-*, Ministral-3-* | task_0 | TBD | VLM models may load vision encoder unnecessarily | TBD — may need --language-model-only |
| VLM detection miss | Ministral-3-* | task_2 | `ValueError: Unrecognized config for AutoModelForCausalLM` | `load_vlm_or_llm` only checks `"vl"` in model_type; `"mistral3"` missed | Check `text_config`/`llm_config` attrs — fixed in repo |
| Missing HF shard | Ministral-3-8B | task_2 | `FileNotFoundError: model-00001-of-00004.safetensors` | Incomplete HF shards + Mistral native `consolidated.safetensors` | Fallback to consolidated with key aliases — fixed in repo |
| Exported config validation | All (via FakeBaseModel) | task_3 | `StrictDataclassFieldValidationError: use_cache` | Template placeholder `None` not filled; strict `huggingface_hub` rejects | Set `use_cache: True` in template — fixed in repo |
| MLA attention | DeepSeek-V3.2 | task_2 | TBD | EAGLE3 decoder type may not support MLA | TBD — verify eagle_decoder_type |
| Custom model code | MiniMax-M2.5 | task_0 | TBD | Non-standard architecture needs trust_remote_code | Add --trust_remote_code |
| Sliding window attn | Step-3.5-Flash | task_1 | TBD | SWA may not be supported in TRT-LLM hidden state extraction | TBD |
| Large MoE (>1 node) | DeepSeek-V3.2, GLM-5 | task_0/1 | TBD | Multi-node EP coordination | TBD — verify NCCL config |
| Gated models | DeepSeek-V3.2, GLM-5 | task_0 | FileNotFoundError | Model not mirrored to /hf-local | Request HF local mirror |

### Per-Model Test Results

#### Model: Ministral-3-8B-Instruct-2512-BF16

- **Date tested:** 2026-05-26
- **Config:** `examples/speculative_decoding/pipeline/eagle3/quick_fail_check_ministral-3-8b-from-task2.yaml`
- **Experiments:** `cicd_1779312692` (dump), `cicd_1779829129` (train+bench), `cicd_1779901409` (retry w/ fixes)
- **task_0 (data synth):** SKIP — used vLLM dump path instead (`dump_offline_data_vllm.sh`)
- **task_1 (hidden states):** PASS — 330/330 conversations via vLLM dump (`cicd_1779312692`)
- **task_2 (training + export):** PASS — required 2 runtime patches (see issues below). `train_loss=31.93`, epoch 1, 278s total. Export to `/scratchspace/export` succeeded.
- **task_3 (benchmark):** FAIL — `StrictDataclassFieldValidationError: use_cache expected bool, got None`
- **AR:** Not measured (benchmark didn't complete)
- **New failure patterns?** Yes — 3 issues:

  1. **VLM detection miss** — `model_type="mistral3"` is a VLM (`Mistral3ForConditionalGeneration`) but `load_vlm_or_llm` only checks `"vl" in model_type`. Fix: also check `text_config`/`llm_config` attrs. Applied in `modelopt/torch/speculative/utils.py` + runtime patch.

  2. **Missing HF shard** — Checkpoint has shards 2-4 + `consolidated.safetensors` but shard 1 is absent. `FakeBaseModel._load_weights` fails. Fix: fallback to `consolidated.safetensors` with Mistral native key aliases (`tok_embeddings.weight`, `output.weight`). Applied in `modelopt/torch/speculative/plugins/modeling_fakebase.py` + runtime patch.

  3. **`use_cache=null` in exported config** — Export template placeholder stays `None` when `FakeBaseConfig` doesn't define `use_cache`. Newer `huggingface_hub` strict validation rejects it. Fix: set `"use_cache": True` in export template (draft models always use cache). Applied in `modelopt/torch/export/plugins/hf_spec_configs.py` + post-export fixup in pipeline.

  4. **(Potential) vLLM Pixtral resolution** — vLLM resolves base model as `PixtralForConditionalGeneration` (VLM). May cause further issues loading the EAGLE3 draft. Needs investigation.

- **Repo fixes (branch `yeyu/speculative-lora-cotrain`):**
  - `modelopt/torch/speculative/utils.py` — VLM detection via `text_config`/`llm_config`
  - `modelopt/torch/speculative/plugins/modeling_fakebase.py` — consolidated.safetensors fallback
  - `modelopt/torch/export/plugins/hf_spec_configs.py` — `use_cache: True` in templates
- **Pipeline fixes (`offline_training.sh`):** 3 runtime patches matching above

---

#### Model: gpt-oss-20b

- **Date tested:** 2026-04-15
- **Config:** `examples/speculative_decoding/pipeline/eagle3/quick_fail_check_gpt-oss-20b.yaml`
- **Experiment:** `cicd_1776272530`
- **task_0 (data synth):** FAIL — `openai_harmony.HarmonyError: error downloading or loading vocab file`. vLLM server starts loading model but tokenizer fails. Likely a gated/proprietary tokenizer issue.
- **task_1 (hidden states):** FAIL — `dump_offline_data_vllm.sh: No such file or directory` (script didn't exist at time of run)
- **task_2 (training):** FAIL — `service_utils.sh: No such file or directory` (infra issue at time of run)
- **task_3 (benchmark):** FAIL — `Error retrieving file list: Repo id must be in the form 'repo_name'` — no exported model
- **Blocker:** Tokenizer loading. Needs special tokenizer setup or newer vLLM with OpenAI model support.

#### Model: Qwen3.5-35B-A3B

- **Date tested:** 2026-04-15
- **Config:** `examples/speculative_decoding/pipeline/eagle3/quick_fail_check_qwen3.5-35b-a3b.yaml`
- **Experiment:** `cicd_1776272531`
- **task_0 (data synth):** TIMEOUT — Server started successfully, data synthesis was running (5%/3295 at 38min), cancelled at time limit. `TCPTransport closed` errors during generation.
- **task_1:** FAIL — script not found (infra issue)
- **task_2:** FAIL — infra issue
- **task_3:** FAIL — no exported model
- **Blocker:** Data synthesis too slow. Needs longer wall time or reduced dataset size. Server itself works.

#### Model: Step-3.5-Flash

- **Date tested:** 2026-04-15
- **Config:** `examples/speculative_decoding/pipeline/eagle3/quick_fail_check_step-3.5-flash.yaml`
- **Experiment:** `cicd_1776272532`
- **task_0 (data synth):** TIMEOUT — `CANCELLED AT 2026-04-15 DUE TO TIME LIMIT`
- **task_1:** FAIL — script not found (infra issue)
- **task_2:** FAIL — infra issue
- **task_3:** FAIL — no exported model
- **Blocker:** Data synthesis hit time limit. Needs investigation of whether server started successfully.

#### Model: MiniMax-M2.5

- **Date tested:** 2026-04-15
- **Config:** `examples/speculative_decoding/pipeline/eagle3/quick_fail_check_minimax-m2.5.yaml`
- **Experiment:** `cicd_1776272524`
- **task_0 (data synth):** TIMEOUT — `CANCELLED DUE TO TIME LIMIT`
- **task_1:** FAIL — script not found (infra issue)
- **task_2:** FAIL — infra issue
- **task_3:** FAIL — `trust_remote_code=True` required for custom model code
- **Blocker:** Time limit on data synth + `trust_remote_code` needed for benchmark.

#### Model: Ministral-3-14B

- **Date tested:** 2026-04-15
- **Config:** `examples/speculative_decoding/pipeline/eagle3/quick_fail_check_ministral-3-14b.yaml`
- **Experiment:** `cicd_1776272522`
- **task_0 (data synth):** FAIL — `TypeError: 'NoneType' object is not iterable` on all TP workers during engine core init. vLLM cannot load this model architecture.
- **task_1:** FAIL — script not found (infra issue)
- **task_2:** FAIL — infra issue
- **task_3:** FAIL — `KeyError: 'ministral3'` — transformers in vLLM container doesn't recognize `ministral3` model type
- **Blocker:** vLLM engine fails to initialize. Same `mistral3` model type issue as 8B variant. Needs newer vLLM + transformers.

#### Model: DeepSeek-V3.2

- **Date tested:** 2026-04-15
- **Config:** `examples/speculative_decoding/pipeline/eagle3/quick_fail_check_deepseek-v3.2.yaml`
- **Experiment:** `cicd_1776275945`
- **task_0 (data synth):** No log file — job may not have started (gated model?)
- **task_1 (hidden states):** FAIL — script not found (infra issue)
- **task_2:** FAIL — infra issue
- **task_3:** FAIL — `Error retrieving file list` — no exported model
- **Blocker:** Model may not be mirrored to `/hf-local`. Needs 2 nodes for MLA architecture.

---

*Use the following template for additional models:*

```markdown
#### Model: <name>
- **Date tested:** YYYY-MM-DD
- **Config:** examples/speculative_decoding/pipeline/eagle3/quick_fail_check_<model>.yaml
- **task_0:** PASS/FAIL — <notes>
- **task_1:** PASS/FAIL — <notes>
- **task_2:** PASS/FAIL — <notes>
- **task_3:** PASS/FAIL — <notes>
- **AR:** <value> (threshold: >= 2.1)
- **New failure pattern?** Yes/No — <description if yes>
```

## Revision History

| Date | Author | Change |
|------|--------|--------|
| 2026-04-02 | Ye Yu | Initial chart with 12 models, triage decision tree |
| 2026-05-27 | Claude Code | Updated with results from initial batch (`cicd_1776272*`) and Ministral-3-8B deep dive. Added per-model test results for 7 models. Added 4 new failure catalog entries. |
