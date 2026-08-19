# Export: Model-Specific Logic Refactor — Action Plan

**Goal:** Give the unified HF export path a per-model data registry
(top-level `modelopt/models/`, one file per HF model type), so that supporting a
new model means adding one declarative spec file instead of editing if/elif chains
across the export engine.

**Scope:** The unified HF export path (`unified_export_hf.py` + its helpers) only.

- The **Megatron** path (`plugins/mcore_*`) already follows a per-model registry
  pattern and is untouched.
- The **TRT-LLM** checkpoint path (`model_config_export.py`, the `build_*`
  functions in `layer_utils.py`, `tensorrt_llm_utils.py`) is legacy: the plan is
  to move it out as-is (separate track), not to refactor it. Its per-model
  branches stay where they are.

## 1. Where the HF path stands after PR #1939

PR #1939 gave the HF path registry-based **module dispatch**: `ExportModuleRegistry`
and `PrepareMoEInputsRegistry` (`registry.py`, `hf_export_handlers.py`) select
*which handler processes a module* by class/predicate match, replacing the if/elif
chains that used to live in `unified_export_hf.py`.

What is still hardcoded is the **per-model data** those handlers (and other HF-path
helpers) consume. Inventory of model-specific logic reachable from the HF path:

| Item | Location | Kind |
|---|---|---|
| MoE expert linear names (Qwen/DeepSeek/Mixtral/DBRX/GptOss/NemotronH/Gemma4) | `layer_utils.get_expert_linear_names` | data — **migrated (P1)** |
| Duplicate expert-naming table + iterable-experts support gate | `layer_utils.get_experts_list` | data — **migrated (P1)** |
| MoE block class-name list | `layer_utils.is_moe` | data — **migrated (P3)** |
| AWQ `pre_quant_scale` fusion rules (Llama/Qwen3) | `quant_utils.PQS_FUSE_MODULE_MAPPING` | data — **migrated (P2)** |
| weight+1 layernorm class names (Gemma RMSNorm, LayerNorm1P) | `quant_utils._layernorm_uses_weight_plus_one` | data — **migrated (NormSpec)** |
| MoE gate/up fusion pairs (`_GATE_UP_PAIRS`, also privately imported by `quantization/model_calib.py`) | `layer_utils.sync_moe_gate_up_amax` | data — **migrated (MoESpec.gate_up_pair)** |
| BMM-style expert class list (`Llama4TextExperts`/`GptOssExperts`) inline copy for weight transpose | `quant_utils` (dispatch copies in `hf_export_handlers.py` stay) | data — P4 |
| VLM detection gates (`phi4mm` model_type, `nemotronparse` architecture, `"nemotron" in model_type` tower special case) | `model_utils.py`, `unified_export_hf.py` | data (gates) + behavior (extraction) — collect until worth a spec flag |
| Handler match keys (Llama4TextExperts, GptOssExperts, DbrxExperts, QuantMoELinear) | `hf_export_handlers.py` | dispatch (stays: structural, per-module) |
| Fused-expert gated/non-gated split (`gate_up_proj` vs `up_proj`) | `moe_utils.py` | data + structure |
| dummy-forward special cases (Whisper input, Nemotron-VL tower) | `unified_export_hf.requantize_resmooth_fused_llm_layers` | behavior |
| VLM language-tower extraction, DiffusionGemma tied-key reorder | `model_utils.py` | behavior |

## 2. Design

Two layers:

- **Engine** (existing files, organized by operation): owns all algorithms and the
  module walk; consults the registry for per-model values.
- **Modeling library** (top-level `modelopt/models/`, one file per HF model type):
  declarative per-model **data only**. No export logic, stdlib-only imports (not even
  torch), so it sits at the bottom of the dependency graph and any modelopt subsystem
  can depend on it.

```text
modelopt/models/
  specs.py       # ModelSpec — the ONE global per-model descriptor, composed
                 # from section mixins: topic sections (MoESpec: MoE layouts as
                 # MoEVariant tuples; NormSpec) + subsystem sections (ExportSpec);
                 # future sections mix in the same way
  registry.py    # register() + lookups queried by spec type (None when unmatched)
                 # + the MRO exact-name matching core (match_class_names)
  __init__.py    # re-exports; importing it registers all specs
  <model_type>.py  # one small file per HF model type (mirrors
                 # transformers.models); import == registration
```

Model type names mirror
[`transformers.models`](https://github.com/huggingface/transformers/tree/main/src/transformers/models)
(e.g. `qwen3_moe.py`, `gpt_oss.py`, `nemotron_h.py`); trust-remote-code models
(`arctic`, `deepseek`) use their config `model_type`.

Each model registers exactly ONE `ModelSpec` (registry enforces uniqueness), so
`get_spec(model_type)` is a dict lookup and consumers call methods directly on the
global spec (e.g. `spec.expert_linear_names_for(module)`).
Resolution is by model type, mirroring HF's own indexing: the engine reads the
model's root HF type (`model.config.model_type`) once per export and passes it
down. The lookup is strict: only the model's own spec is consulted, so a model
whose model_type has no spec fails loudly (register a spec) instead of inheriting
a neighbor's data through a coincidental class-name match. Sub-config model types
of composite models are not walked today — a composite whose MoE lives under a
tower type registers the root type too (`gemma4` + `gemma4_text`, following the
`gemma3`/`gemma3_text` precedent); a recursive config walk is a future extension
if a real VLM tower needs it. Within the spec, each `MoESpec` nests one
`MoEVariant` per concrete block layout — several when the same checkpoint
materializes with different classes and projection names (Mixtral across
transformers generations); variant `block_names` (matched against the module
MRO) identify MoE blocks (`is_moe`) and pick the variant.
`get_expert_linear_names` doesn't need the block class at all when a model's
variants agree on one naming. Without a model type (no config available: unit
tests, the TRT-LLM path), lookups search all specs by class name.

During migration, call sites kept the legacy branches as a fallback behind the
spec lookup. Once the specs covered every family the legacy chains served, the
chains — and the silent ``w1/w2/w3`` guess for unknown models — were deleted:
expert-name resolution is now *structural detection -> spec -> raise*, so a new
MoE model fails loudly, asking for a spec, instead of inheriting another model's
naming. Generic detection that is not per-model data (the ``*SparseMoeBlock`` /
``*MoeLayer`` conventions and the router+experts structural check in ``is_moe``,
the fused-experts quantizer probe in ``get_expert_linear_names``) stays in the
engine, ahead of or beside the spec lookup.

Note on naming: `modelopt/models/registry.py` (per-model **data**, "what are
this model's values") is distinct from the export-path `registry.py` from PR #1939
(per-module **dispatch**, "which handler processes this module"). The two layers
compose: handlers look up model data through `modelopt.models`.

## 3. Migration plan

Each step is one PR with a fallback to the legacy path and an equivalence check
against existing export tests.

| Step | What | Status |
|---|---|---|
| **P1** | Registry skeleton + MoE expert naming: `get_expert_linear_names` and `get_experts_list` read `spec.expert_linear_names` / `spec.has_iterable_experts`. The #1 "add a MoE model" shotgun-surgery driver. | this PR |
| **P2** | `PQS_FUSE_MODULE_MAPPING` → `spec.pqs_fuse_rules`, aggregated via `iter_pqs_fuse_rules` (llama/qwen3 specs). | this PR |
| **P3** | `is_moe` explicit class-name list → `spec.moe_block_names` (arctic/dbrx_ffn are identification-only specs: no expert naming, so expert-name lookups keep the engine default). The generic `*SparseMoeBlock`/`*MoeLayer` conventions and the structural router+experts check stay in the engine. | this PR |
| **P4** | HF handlers consume specs directly; fold remaining `moe_utils` naming data into specs; share the matcher machinery with the export dispatch registry (#1939). Model_type-scoped resolution (`collect_model_types` + scoped `match_moe_block`, threaded via `ExportContext.model_types`) is already in place from this PR. | planned |
| **P5** | Cross-subsystem pilot: unify the remaining copies of linear-fusion-group knowledge (see §4) into spec fields. Partially done: `_GATE_UP_PAIRS` became `MoESpec.gate_up_pair` and `model_calib.py`'s private import of it is gone — the first quantization consumer of `modelopt.models`. Remaining: `shared_input.SHARED_PATTERNS`, `algorithms.quant_grouping_rules`. | in progress |
| **P6** | Migrate remaining quantization-side data: default disabled-quantizer patterns, on-the-fly conversion gates, AutoQuantize grouping rules (see §4). | planned |
| **OUT** | TRT-LLM path branches (`decoder_type` chains in `build_*`, `model_config_export.py`, `tensorrt_llm_utils.py`): frozen, moved out unchanged on a separate track. Candidates for deletion on that track: `adjust_attn_amax_values`, `update_experts_avg_prequant_scale` (unused). NOTE: `MODEL_NAME_TO_TYPE` / `get_model_type` are NOT dead — `examples/hf_ptq/hf_ptq.py` and `multinode_ptq.py` still call them; migrate the examples before removing. | separate track |

**Guardrails:** one data category per PR (fallback-first while a category is
partially migrated, explicit-error once specs cover it); the engine keeps the
algorithms — model specs supply values only, never fork functions.

## 4. Beyond export: per-model data in quantization (P5/P6 inventory)

The same three kinds of model-specific logic exist on the quantization side. Only
kind (a) migrates into `modelopt/modeling`; (b) stays in each subsystem's module
registry (a spec may hold pointer data, never the surgery code); (c) stays in the
engine behind structural checks.

| Item | Location | Kind |
|---|---|---|
| Linear fusion groups (q/k/v, gate/up, `w1/w3`) — was duplicated 3x; the `_GATE_UP_PAIRS` copy is now `MoESpec.gate_up_pair` | `quantization/utils/shared_input.SHARED_PATTERNS`, `quantization/algorithms.quant_grouping_rules` (remaining) | data — P5 |
| Model-class gates for on-the-fly conversion (`"DbrxForCausalLM"`, `("Step3p5ForCausalLM", ...)`) | `quantization/plugins/huggingface.py` | data — P6 |
| Default disabled-quantizer patterns (`*router*`, `*vision_tower*`; per-model, NVBug-gated) | `modelopt_recipes/.../default_disabled_quantizers.yaml` | data — P6 |
| AutoQuantize grouping regexes (llama q/k/v, Mixtral `w1/w2/w3`, NemotronH mixer) | `quantization/algorithms.py` | data — P6 |
| Quant wrapper classes (`_QuantDbrxExperts` splits `w1/v1/w2` into per-expert linears) | `quantization/plugins/huggingface.py` via `QuantModuleRegistry` | dispatch (stays) |
| Structural MoE detection (`gate`+`experts`+`top_k` attrs; 3-D `gate_up_proj` -> gated) | `quantization/plugins/huggingface.py` | behavior (stays) |
