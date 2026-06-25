# Export: Model-Specific Logic Refactor — Action Plan

**Goal:** Extract model-specific logic out of `modelopt/torch/export` into a new
modeling library (`modelopt/torch/export/modeling/`), organized by model family,
leaving the export code generic and model-agnostic.

**Scope (this effort):** HF / TRT-LLM export path only. The Megatron path
(`plugins/mcore_*`) already follows the target registry pattern and is kept as the
reference north star, not refactored here.

---

## 1. Inventory of Model-Specific Logic

Counts are grep-derived and reproducible. "Branches" = decision points
(`decoder_type ==/in`, class-name / `module_match_name_list` string matches).
"Line footprint" = lines referencing a model-family name literal (approximate
measure of model-specific contamination).

### 1.1 HF / TRT-LLM path (refactor target)

| Category | Branches | Line footprint | Main location |
|---|---|---|---|
| Model ID table `MODEL_NAME_TO_TYPE` | 41 entries | ~50 | `model_utils.py` |
| `decoder_type ==/in` imperative branches | 51 | — | layer_utils 25, model_config_export 15, tensorrt_llm_utils 11 |
| Class-name / arch string branches (`type().__name__`, `module_match_name_list`) | ~74 | — | layer_utils 42, unified_export_hf 20, quant_utils 4, diffusers_utils 5, model_utils 2, transformer_engine 1 |
| Config field map `HF_CONFIG_MAP` | 36 entries | ~36 | `hf_config_map.py` |
| AWQ fusion table `PQS_FUSE_MODULE_MAPPING` | 2 groups | ~12 | `quant_utils.py:1141` |
| Speculative-decoding templates | 2 configs + 3 Exporter classes | ~150 | `plugins/hf_spec_configs.py`, `plugins/hf_spec_export.py` |

**Totals:** ≈ 125 imperative branch points + ≈ 79 data-table entries.
Concentrated in 4 files — `layer_utils.py` (67, the heaviest), `model_config_export.py`,
`tensorrt_llm_utils.py`, `unified_export_hf.py` — plus two tables in
`model_utils.py` and `hf_config_map.py`.

### 1.2 Breakdown by functional category (HF / TRT-LLM path)

| Functional category | ~Branches | Representative location |
|---|---|---|
| MoE expert naming / router / expert stacking | ~20 | `get_expert_linear_names`, `build_moe_config` |
| Attention variants (MQA/GQA/MLA/cross-attn/clip_qkv) | ~15 | `build_decoder_config`, `build_attention_config` |
| LayerNorm fixups (weight+1, etc.) | ~3 | `build_layernorm_config` |
| Activation overrides per family | ~4 | `build_mlp_config` |
| MLP gate/fc naming swap, fused chunk order | ~8 | `build_mlp_config` |
| Positional / rope / config field translation | ~36 + a few | `HF_CONFIG_MAP`, Phi3 longrope, etc. |
| Embedding scaling / share / tied-weight dedup | ~6 | `model_config_export`, `model_utils` |
| Encoder-decoder routing (T5/BART/Whisper) | ~12 | `model_config_export`, `tensorrt_llm_utils` |
| Multimodal VLM language-tower extraction | ~5 | `model_utils`, `unified_export_hf` |
| Quant-format specific (AWQ fusion / NVFP4 3D experts / GPT-OSS) | ~8 | `quant_utils`, `unified_export_hf` |
| Speculative-decoding export | ~5 classes/templates | `plugins/hf_spec_*` |

### 1.3 Megatron path (out of scope — reference only)

Already a declarative registry; not refactored here.

| Item | Amount | Location |
|---|---|---|
| Per-family mapping files (≈100% model-specific) | 6 files / 771 lines | `plugins/mcore_{llama,qwen,qwen3vl,deepseek,gptoss,nemotron}.py` |
| Family registry | 14 export + 10 import entries | `plugins/mcore_common.py:44` |
| `MCORE_CONFIG_MAP` field map | 3 entries | `mcore_config_map.py` |
| Arch branches in export driver (MLA / VLM / packed-expert) | ~3 | `unified_export_megatron.py` |

### 1.4 Key takeaways

- The real backlog to absorb is the HF/TRT-LLM path: **≈125 scattered imperative
  branches + ≈79 data-table entries**.
- ~80% of the imperative branches live in a single file: `layer_utils.py` (67).
- ~79 entries are **already data tables** (`MODEL_NAME_TO_TYPE`, `HF_CONFIG_MAP`,
  `PQS_FUSE_MODULE_MAPPING`) — moving them into the modeling lib is near-mechanical.
- Of the ~125 imperative branches, only a minority are true control-flow
  (attention variants, enc-dec); most are "look up a value by model".

---

## 2. Rationale & Trade-offs

This is a `operation × model` matrix. Organize by operation (today) and changing an
operation is cheap but adding a model is shotgun surgery; organize by model and the
reverse. The right axis is the one that changes most.

**Evidence — export files touched per recent model-support PR:**

| PR | export files touched |
|---|---|
| NemotronH non-gated MoE (#1756) | 5 (layer_utils, moe_utils, quant_utils, unified_export_hf, vllm_fakequant) |
| DiffusionGemma enc-dec (#1707) | 4 (model_utils, moe_utils, quant_utils, unified_export_hf) |
| Gemma4 MoE (#1219) | 1 (layer_utils) |
| Llama4 MoE fix (#1744) | 1 (unified_export_hf) |

Adding a model is frequent and often spans 4–5 files — the cost a by-model split removes.

**Scope discipline (the red line):** move per-model *deltas*, not generic algorithms.

| Move into modeling lib (by model) | Keep in functional files (by operation) |
|---|---|
| Data: expert names, norm+1, activation, embed scale, share_embed, AWQ fusion table, config field maps | Generic algorithms: scale compute, weight packing, TP/PP split |
| Most "look up a value by model" branches | True control-flow oddballs: enc-dec routing, MLA |

**Net:** function-cohesion for generic code is correct and stays; only the per-model
deltas leaking into those files get extracted. Target = generic engine (by operation)
- model-descriptor registry (by model), mirroring the Megatron `plugins/mcore_*` pattern.

## 3. Target Architecture

Two layers, split along the §2 red line:

- **Generic engine** (the existing files, organized *by operation*): `layer_utils`,
  `quant_utils`, `moe_utils`, `model_config_export`, … — keep all algorithms here.
- **Modeling library** (`modeling/`, organized *by model*): per-model **data only**,
  read by the engine through registry lookups.

### 3.1 Layout

```text
modeling/
  base.py        # ModelSpec dataclass — the per-model contract
  registry.py    # register() + lookups; returns None when unmatched
  __init__.py    # re-exports; importing it registers all families
  families/
    __init__.py  # imports each family module (import == registration)
    qwen.py deepseek.py gemma.py mixtral.py dbrx.py gptoss.py nemotron.py ...
```

`modeling/` depends only on `torch.nn` + stdlib. It must **never** import from the
parent export package (no `from ..layer_utils import ...`), so it stays at the bottom
of the dependency graph and cannot reintroduce the `unified_export_hf ↔ plugins` cycle.

### 3.2 `ModelSpec` — the contract

A frozen-ish data record per model family. Two kinds of fields:

- **Matching keys** (how a spec is resolved):
  - `architectures: tuple[str, ...]` — exact HF architecture class names (model-level).
  - `moe_block_names: tuple[str, ...]` — MoE block class-name substrings, matched
    case-insensitively (sub-module-level), reproducing the legacy `module_match_name_list`.
- **Per-model data** (grows one migration step at a time; **values, never callables/algorithms**):
  - `expert_linear_names` *(P1, landed)*
  - P2/P3 add fields here (e.g. expert-stacking names, `norm_weight_plus_one`,
    `default_activation`, `scale_embedding_by_sqrt_hidden`, `share_embedding_table`).

Rule: if a field would hold *behavior* (an attention builder, enc-dec routing), it does
**not** belong on `ModelSpec` — that is the OUT bucket (§4).

### 3.3 Lookup conventions

Two resolvers in `registry.py`, both returning `ModelSpec | None`:

- `match_by_architecture(arch)` — model-level data, keyed on `model.config.architectures[0]`.
- `match_moe_block(module)` — sub-module-level data, keyed on a module's class name.

Specs are matched in registration order; families must keep their match keys mutually
exclusive (no class name may match two specs), so order is never load-bearing.

### 3.4 Call-site integration pattern (fallback-first)

Every migrated call site follows the same shape — look up, use if present, else fall
through to the **unchanged** legacy code:

```python
spec = match_moe_block(module)               # or match_by_architecture(arch)
if spec is not None and spec.<field> is not None:
    return spec.<field>
# ... legacy branch preserved verbatim as fallback ...
```

This is what makes migration incremental: a family not yet in `modeling/` returns
`None` and behaves exactly as before. Once every family for a category is migrated, the
legacy branch is dead but kept until the category is fully retired.

### 3.5 Conventions for contributors

- **Add a model:** create/extend `families/<family>.py`, register a `ModelSpec`, done —
  no engine edits.
- **Add a category (P2/P3):** add a field to `ModelSpec`, populate it in the families,
  convert the one engine call site to the fallback-first pattern, add an equivalence test.
- **Never** put algorithms in `modeling/`; if tempted, the logic belongs in the engine
  and only its per-model *parameters* (if any) come from the spec.

## 4. Migration Plan & Priority

Priority is set by two tests: **(a) how pure-data it is** (purer → sooner) and
**(b) how central it is to add-a-model shotgun surgery** (the §2 evidence points at
MoE). Each step keeps a fallback to the legacy path, so migration is incremental and
never breaks un-migrated models.

| Step | What | Why this order | Risk |
|---|---|---|---|
| **P0** ✅ | Registry skeleton: `modeling/` + `ModelSpec` + lookups (`match_moe_block` / `match_mlp_block` / `match_by_decoder_type` / `match_by_architecture`, return `None` → fallback). Do **not** touch `MODEL_NAME_TO_TYPE`. | Everything keys off it; purely additive, no behavior change. | very low |
| **P1** (pilot) ✅ | MoE expert naming: `get_expert_linear_names` if-chain → `spec.expert_linear_names`. | Purest data + the #1 shotgun-surgery driver. Proves the mechanism end-to-end with near-zero blast radius. | very low |
| **P2** ✅ | The duplicate expert-naming table in `get_experts_list` → `spec.expert_linear_names` + `spec.has_iterable_experts` (the grouped-export capability flag that reproduces the legacy "supported 4 families, else `NotImplementedError`"). | Removes the second copy of expert-naming data → "add a MoE model" stops touching this site. | low |
| **P3** (partial) | Non-MoE per-model data. Done: activation override (`forced_activation` ✅), shared-embedding (`force_share_embedding_table` ✅), MLP keyword roles (`mlp_keyword_roles` ✅ — Arctic/InternLM2 remap, MPT/Phi3Small/TLGv4/Nemotron up_proj→fc). Deferred: norm+1 (mixes the generic `LayerNorm1P` which stays in the engine; needs a case-sensitive norm-class resolver) and embed √scale (Gemma1-only + `is_tensorrt_llm_0_8_or_9()` version gate) — low value. | Mechanical bool/str/dict moves; clears scattered `layer_utils` / `model_config_export` branches. | low |
| **P4** (partial) | `PQS_FUSE_MODULE_MAPPING` (AWQ fusion) → `spec.pqs_fuse_rules`, aggregated via `iter_pqs_fuse_rules` (✅, llama/qwen families). `HF_CONFIG_MAP` deliberately **kept**: it is a shared *alias* table applied generically to every config (`for keys, name in HF_CONFIG_MAP`), not a per-model branch — splitting it by family would be artificial and more fragile. | PQS rules are genuinely per-model (keyed by module class); the alias table is not. | low |
| **OUT** | Control-flow oddballs: enc-dec routing, MLA, VLM language-tower extraction, spec-decoding (already separate), the ChatGLM/Phi3 fused gate/fc chunk-swap. | Forcing into `ModelSpec` creates leaky hooks; revisit after P1–P3 fix the interface. | — |

**Guardrails:** fallback-first (un-migrated models keep the old path); one category per
PR with an export-test equivalence check; hard line — never move *algorithms*, only the
per-model *values*.

> **P2 note:** an earlier draft scoped P2 as "expert amax fallback / gate-up amax sync."
> On inspection `set_expert_quantizer_amax` and `sync_moe_gate_up_amax` are generic
> *algorithms* (the gate/up pairs in `_GATE_UP_PAIRS` are a flat, non-per-model list), so
> by the red line they stay in the engine. The only per-model *data* left in this area was
> the duplicate naming table in `get_experts_list`, which is what P2 actually migrated.
