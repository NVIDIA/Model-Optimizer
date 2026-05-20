# vllm_calib — handoff note

This file is intended to be **complete enough that anyone can resume the work
from this document alone**. Read sequentially; later sections assume earlier
ones.

---

## TL;DR — Where to start

1. The pipeline calibrates `compressed-tensors`-INT4 MoE checkpoints (Kimi-K2)
   to NVFP4 by running calibration **inside vLLM** instead of HF eager mode,
   then exporting a unified-HF NVFP4 checkpoint.
2. The **calibration half** works end-to-end on single-node K2-Thinking with
   dummy weights. Real-weights run hasn't been driven yet.
3. The **export half** (`export_hf_from_vllm_state.py`) gets through HF load,
   modelopt wrap, and quantizer-state load, then **OOMs in
   `_export_transformers_checkpoint`** when compressed-tensors'
   `decompress_model` materialises every routed-expert INT4 weight as BF16
   in HBM at once. **This is the active blocker.**
4. The **multi-node calibration** (needed for K2.6) is broken on vLLM 0.20.1
   for unrelated reasons (`collective_rpc should not be called on follower
   node`). **Second blocker** — independent of the export issue.
5. We have made changes to **modelopt source** (under
   `Model-Optimizer/modelopt/...`) that this pipeline depends on. They live
   alongside the modelopt repo, not in this experimental dir. See the
   "Modelopt source patches" section.

If you only have time to read three sections of this doc, read **TL;DR**,
**Known Issues**, and **Suggested Next Moves**.

---

## Goal

Calibrate Kimi-K2-class MoE models (K2-Thinking ~554 GB, K2.6 ~555 GB; both
ship pack-quantized INT4 routed experts via `compressed-tensors`) for NVFP4
by running the calibration forward **inside vLLM** (fast multi-node TP,
native compressed-tensors handling), and export a **unified-HF NVFP4
checkpoint**.

The slow HF-eager `from_pretrained` path takes ~28 min (cache-hot) for
K2-Thinking and is the reason vLLM-side calibration is preferred — vLLM
loads via its own optimised path and runs the calibration forward on the
final tensor-parallel topology. The slow HF load happens **only** once at
export time.

The flow:

```
        Stage 0 (per-rank, on vLLM)
   vLLM model running calibration forward
                  │
                  ▼
   examples/vllm_serve/fakequant_worker.py
   _save_vllm_calibrated_state(model)        ── reads MODELOPT_VLLM_STATE_DIR
                  │                              + MODELOPT_VLLM_STATE_TAG
                  ▼
   <state_dir>/<TAG>-rank<NN>-of<MM>.pth × world_size

        Stage 1 (CPU, seconds)
   experimental/vllm_calib/merge_and_convert_vllm_state.py
   • merge per-rank `_amax` (max for scalar, concat for per-channel)
   • rewrite vLLM fused-MoE quantizer keys
       …mlp.experts.w13_input_quantizer  →  …mlp.experts.<i>.{gate,up}_proj.input_quantizer
       …mlp.experts.w2_input_quantizer   →  …mlp.experts.<i>.down_proj.input_quantizer
                  │
                  ▼
   <state_dir>/<TAG>-merged-hf.pth

        Stage 2 (1×8×H100, ~30 min for K2-Thinking)
   experimental/vllm_calib/export_hf_from_vllm_state.py
   • get_model(ckpt_path)                       ~28 min HF load
   • mtq.quantize(model, NVFP4_EXPERTS_ONLY_CFG, forward_loop=None)
                                                ~8 min wrap
   • per-quantizer non-strict load_state_dict   <1 s
   • export_hf_checkpoint(...)                  ❌ OOM here
                  │
                  ▼
   unified-HF NVFP4 checkpoint  (not yet produced)
```

Why **not** `mto.restore_from_modelopt_state`: the saved `modelopt_state`
metadata carries vLLM-side fused names (`experts.w13_*`, `mla_attn.mla_attn.*`)
that don't match HF's per-expert structure, so its strict
`unmatched_keys = state_keys - model_keys` check fires. We instead reapply the
same `quant_cfg` fresh against the HF model and inject our own per-quantizer
state.

---

## Cluster setup

These are the absolute paths and accounts. Pin them when reading; everything
else assumes them.

### Paths

| Resource | Path |
|----------|------|
| Project root | `/lustre/fs1/portfolios/adlr/projects/adlr_psx_numerics/users/jingyux/kimi-k2` |
| HF source models | `${ROOT}/models/Kimi-K2-Thinking`, `${ROOT}/models/Kimi-K2.6` |
| HF token | `${ROOT}/token.txt` (key/value form: `HF_TOKEN: <value>` — must be parsed) |
| Modelopt source (editable) | `${ROOT}/source/Model-Optimizer` |
| vLLM source (reference only — DO NOT modify) | `${ROOT}/source/vllm` |
| Container | `${ROOT}/containers/vllm-openai-modelopt-kimi.sqsh` |
| Container build script | `${ROOT}/containers/_layer_vllm_modelopt.sh` |
| HF model download cache | `${ROOT}/hf_cache` |
| Calibration experiment (single-node K2-Thinking) | `${ROOT}/experiments/vllm_calib_kimi-k2-thinking/` |
| Calibration experiment (multi-node K2.6) | `${ROOT}/experiments/vllm_calib_kimi-k2.6/` |
| Export experiment | `${ROOT}/experiments/export_hf_kimi-k2-thinking/` |

### Token format quirk

`token.txt` is `HF_TOKEN: <value>` (47 bytes for jingyux's token), **not** a
bare token. The launcher and `fakequant_worker.py` both parse the
`HF_TOKEN: ` prefix off before exporting `HF_TOKEN`. Don't paste raw token
contents anywhere.

### SLURM accounts and partitions

User has access to: `adlr_psx_numerics`, `adlr_psx_sparsity`,
`coreai_chef_numerics`, `coreai_dlalgo_modelopt`, `llmservice_fm_vision`,
`nemotron_n3_quant`, `nemotron_n4_quant`, `nemotron_quant_dev`.

We've been using `coreai_dlalgo_modelopt` as the canonical billing account
for this work. The launchers default to that.

Partitions: launchers default to `interactive,batch_block1` (comma-separated
— SLURM picks whichever has resources first). 4 hr cap on both. For long
downloads we used `cpu_long` (7 day cap, no GPUs).

### Container

The container is `vllm/vllm-openai:latest` + `pip install -e .[hf]` against
`${ROOT}/source/Model-Optimizer`. Built once via
`${ROOT}/containers/_layer_vllm_modelopt.sh`. Versions baked in:

- vLLM 0.20.1
- transformers 5.7.0 (Kimi modeling code emits a deprecation warning under
  this; it's been benign so far)
- compressed-tensors 0.15.0.1
- modelopt is editable — source edits in `${ROOT}/source/Model-Optimizer/`
  land immediately, no rebuild needed.

Container is 22 GB. If it gets corrupted or needs rebuilding:

```bash
srun -A coreai_dlalgo_modelopt -p interactive,batch_block1 -N 1 --gres=gpu:1 -t 90 \
    --container-image=vllm/vllm-openai:latest \
    --container-save=${ROOT}/containers/vllm-openai-modelopt-kimi.sqsh \
    --container-mounts=/lustre:/lustre --no-container-mount-home --no-container-remap-root \
    bash ${ROOT}/containers/_layer_vllm_modelopt.sh
```

(Existing log of the successful build:
`${ROOT}/containers/layer_vllm_modelopt.log`.)

---

## Reproduce from scratch (single-node K2-Thinking dummy-weights smoke)

The known-good smoke that produced the per-rank state files we've been
working with is **JOBID 4240786**. Reproducing:

```bash
cd ${ROOT}/experiments/vllm_calib_kimi-k2-thinking
bash launch.sh --calib-size 4 --max-seq-len 2048 --suffix dummyN --dummy-weights
```

Expected wall time: ~5 min (dummy weights skip the slow vLLM load). Watch
the log for:

- `[modelopt] vLLM FusedMoE wrapped with compressed source quant_method=CompressedTensorsWNA16MarlinMoEMethod`
- `Quantizing model...` (the calibration prolog)
- `[vllm-calib] rank<N>/8: saved quantizer state -> .../<JOBID>-rank0N-of08.pth`

Note: the worker doesn't `sys.exit()` cleanly after state save — vLLM continues
into `mtq.fold_weight` (no-op for compressed source) and post-fold checks,
which on dummy weights eventually CUDA-OOM. The state files are already on
disk by then.

State files land in `${ROOT}/experiments/vllm_calib_kimi-k2-thinking/state/`
and are ~640 KB each.

Then merge + export:

```bash
cd ${ROOT}/experiments/export_hf_kimi-k2-thinking
bash launch.sh --tag <JOBID> --suffix smoke
```

Expected wall time: ~40 min (28 min HF load + 8 min wrap + 1 s state load,
then OOM in `export_hf_checkpoint`). The current OOM is the active blocker.

### What success would look like

`${ROOT}/experiments/export_hf_kimi-k2-thinking/export_out/<TAG>-smoke/`
populated with `model-*.safetensors`, `config.json` (with NVFP4 quant_config),
`tokenizer.json`, etc. — a checkpoint that vLLM/SGLang/TRT-LLM can load
natively. We have not produced this yet.

---

## Files in this directory

| File | Role |
|------|------|
| `README.md` | Intended workflow, run instructions, summary of caveats. |
| `handsoff.md` | This file. |
| `merge_and_convert_vllm_state.py` | **Stage 1**: fold per-rank state files, max/concat-merge `_amax` across TP, rewrite vLLM fused-MoE quantizer keys to HF per-expert names. CPU-only, runs in seconds. |
| `export_hf_from_vllm_state.py` | **Stage 2**: load source HF model via `examples/llm_ptq/example_utils.get_model`, `mtq.quantize` with original quant_cfg (`forward_loop=None`), non-strict per-quantizer `load_state_dict`, `export_hf_checkpoint`. The OOM blocker is in the last call. |

---

## Modelopt source patches (NOT in this dir — in `${ROOT}/source/Model-Optimizer/modelopt/`)

These are required by the workflow. Review / narrow before any upstream PR.

### Required for the calibration path

1. **`modelopt/torch/quantization/plugins/vllm.py`** —
   `_QuantFusedMoEBase` updated to handle compressed-tensors source models:
   - `_setup` no longer asserts
     `type(self.quant_method) is UnquantizedFusedMoEMethod`. Sets
     `self._compressed_source = (type is not UnquantizedFusedMoEMethod)`.
     Logs `[modelopt] vLLM FusedMoE wrapped with compressed source
     quant_method=…` when triggered.
   - `forward` branches: for `_compressed_source`, calls
     `w13_input_quantizer(hidden_states)` (input amax collection) then
     `super().forward(hidden_states, router_logits)` (untouched Marlin
     dispatch). Does **not** monkey-patch
     `vllm_fused_moe_package.invoke_fused_moe_kernel` because Marlin's MoE
     dispatches through `fused_marlin_moe`, a different kernel function.
   - `fold_weight` is a no-op for compressed source: just disables the
     weight quantizers so `fakequant_worker.py`'s post-fold check passes.
     We never want to fold compressed-tensors weights — the export side
     handles weight quantization fresh from source.

   **Why this is needed:** without it, `mtq.quantize` fails with
   `AssertionError: quant_method is <class 'CompressedTensorsWNA16MarlinMoEMethod'>`
   on every routed-expert MoE block.

### Required for the export path

2. **`modelopt/torch/quantization/nn/modules/quant_module.py:248`** — gated
   `self._register_dynamic_attribute("weight", self._get_quantized_weight)`
   on `hasattr(self, "weight")`. Without this, `replace_quant_module` raises
   `AttributeError: weight is not a valid attribute on QuantLinear` on every
   routed-expert `CompressedLinear` (which exposes `.weight_packed`, not
   `.weight`).

   **Blast radius is wider than necessary** — should narrow the gate to
   `isinstance(self, CompressedLinear)` (or
   `hasattr(self, "weight_packed")`) so a regular `nn.Linear` that for some
   other reason is missing `.weight` errors loudly instead of silently
   skipping fake-quant.

### Debug residue (remove before upstream)

3. **`modelopt/torch/quantization/conversion.py`** — `_replace_quant_module`
   wrapped with try/except + `_path` arg to print the offending module path
   when conversion fails. Useful for the next debugger; should be reverted
   before any PR.

4. **`modelopt/torch/opt/dynamic.py:441`** — improved `AttributeError`
   message to include `type(self).__module__.__name__`. Optional retain
   depending on upstream's preference for diagnostic context.

---

## Files outside this dir that we modified (also under `${ROOT}/source/Model-Optimizer/`)

- **`examples/vllm_serve/fakequant_worker.py`**:
  - Added `_save_vllm_calibrated_state(model)` helper.
  - Added a call to it inside `_fakequant_run_prolog_worker`, **before**
    `mtq.fold_weight(model)` — so the saved state captures the calibrated
    `_amax` values before fold disables them.
  - Reads `MODELOPT_VLLM_STATE_DIR` + `MODELOPT_VLLM_STATE_TAG` env vars.
    Each rank writes `<TAG>-rank<NN>-of<MM>.pth` containing
    `mto.modelopt_state(model)` + `mtq.utils.get_quantizer_state_dict(model)`
    + a `quant_config_summary`.
  - Added `QUANT_MAX_SAMPLE_LEN` env passthrough into
    `get_dataset_dataloader(max_sample_length=…)`. Default 512 was too
    small for K2.6's 262k context.

These changes are part of the experimental flow but live in `examples/`
because they extend the upstream-tested `fakequant_worker.py` rather than
introducing a new file. The new files (`merge_and_convert_vllm_state.py`,
`export_hf_from_vllm_state.py`) sit here in `experimental/vllm_calib/` per
the project convention that new modelopt features land under `experimental/`,
not `examples/`.

---

## Naming convention crash course

vLLM **fuses** MoE experts into per-layer `w13` (concatenation of all
experts' gate+up projections) and `w2` (concatenation of all experts' down
projections). modelopt's vLLM-side wrapper exposes:

- `…mlp.experts.w13_input_quantizer`     ← scalar, observed during calibration
- `…mlp.experts.w13_weight_quantizer`    ← created but never invoked (disabled by our patch)
- `…mlp.experts.w2_input_quantizer`      ← created but **never observed** (kernel-internal)
- `…mlp.experts.w2_weight_quantizer`     ← created but never invoked

HF's `DeepseekV3MoE` (used by Kimi-K2-Thinking via auto_map) wraps each
expert as a separate `DeepseekV3MLP` with `gate_proj`, `up_proj`,
`down_proj`. When modelopt wraps this on the HF side via
`_QuantSparseSequentialMoe`, the per-expert linears get individual
quantizers:

- `…mlp.experts.<i>.gate_proj.input_quantizer`
- `…mlp.experts.<i>.gate_proj.weight_quantizer`
- (and the same for `up_proj`, `down_proj`, all i in 0..N_experts-1)

`merge_and_convert_vllm_state.py` does the rename from the vLLM fused names
to the HF per-expert names. The same `_amax` is duplicated across each
expert × HF projection (Marlin assumes a shared input scale across all
experts in a layer, so this is faithful).

For K2-Thinking: 60 layers × 384 experts × 3 projections × 3 quant kinds
(input, weight, output) → 207,360 expanded entries. Original 1458 vLLM
entries → 208,458 HF entries. Of those, 46,080 carry non-empty `_amax`
data after the dummy-weight smoke (`60 layers × 384 experts × 2 (gate+up)
= 46,080` for `w13_input` only).

---

## Quant config

The configured calibration target is `mtq.NVFP4_EXPERTS_ONLY_CFG`:

```python
NVFP4_EXPERTS_ONLY_CFG = _nvfp4_selective_quant_cfg(
    ["*mlp.experts*", "*block_sparse_moe*"]
)
# matches *both* weight and input quantizers, both with
# {num_bits: (2, 1), block_sizes: {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
```

`type: dynamic` means **per-block scales computed at runtime from the live
tensor**, not stored as a static buffer. Implication: `_amax` collected during
calibration is *advisory* for dynamic NVFP4 — the runtime kernel doesn't read
it. It IS used during export to bound block-scale ranges; how much that
matters in practice for NVFP4 is an open question we have not measured.

If you switch to a static / AWQ NVFP4 variant, the calibration
data becomes load-bearing and the `w2_input` blind spot becomes a real
quality issue.

---

## Status

| Stage | Single-node K2-Thinking | Multi-node K2.6 |
|------|-----|-----|
| Container build | ✅ | ✅ (same container) |
| vLLM calibration (`Quantizing model…` → `[vllm-calib] saved quantizer state`) | ✅ on dummy weights | ❌ vLLM 0.20.1 multi-node bug — never reached calibration |
| Per-rank state dump on disk | ✅ 8 files written | — |
| `merge_and_convert_vllm_state.py` (CPU) | ✅ 1458 → 208458 entries, 46080 non-empty | — |
| `export_hf_from_vllm_state.py`: HF load (~28 min) | ✅ | — |
| `export_hf_from_vllm_state.py`: `mtq.quantize` wrap (~8 min, 209010 quantizers) | ✅ (with `quant_module.py` patch) | — |
| `export_hf_from_vllm_state.py`: per-quantizer non-strict `load_state_dict` (46080 entries) | ✅ 0 skipped | — |
| `export_hf_from_vllm_state.py`: `export_hf_checkpoint` writes unified-HF NVFP4 | ❌ CUDA OOM during compressed-tensors `decompress_model` | — |

Net: single-node calibration → state dump → merge → wrap → state load works.
Final write-to-disk OOMs.

We have NOT yet run a real-weights end-to-end pass — only dummy weights for
path validation. The saved `_amax` values are noise.

---

## Known issues (severity-ordered)

### 1. Export-side OOM on K2-class MoEs *(BLOCKING for production)*

`unified_export_hf._export_transformers_checkpoint` calls
`requantize_resmooth_fused_llm_layers(model)` (line 781) which calls
`collect_shared_input_modules(model, dummy_forward_fn)` (line 222) which
runs a `dummy_forward_fn(model)` (line 387 — `llm_dummy_forward`). That
forward triggers compressed-tensors' `ct_decompress_hook`, which calls
`decompress_model(model)`, which materialises **every** routed-expert INT4
weight as BF16 in HBM at once. Approximately 4× expansion. K2-Thinking +
8×H100-80G OOMs at "Decompressing model: 1%". K2.6 won't fit at all.

The `[Compressing model: 100%]` and `[Loading checkpoint shards: 100%]`
progress bars in the log before the OOM are red herrings — those are
internal compressed-tensors processing of the *source* checkpoint, not
the export's own work.

Fix candidates, easiest first:
- **Skip `requantize_resmooth_fused_llm_layers` when AWQ is not
  configured.** This pass is mostly there for AWQ resmoothing
  (`_resmooth_experts_for_export`). For `nvfp4_experts_only` without
  AWQ, the dummy forward might be unnecessary. Read
  `unified_export_hf.py:222 collect_shared_input_modules`,
  `:387 llm_dummy_forward`, `:781 (call site)`. If the call is gated
  by an AWQ-specific config bit, see if our config bypasses it; if
  not, propose a gate for it.
- **Multi-node export** (16+ ranks) using `multinode_ptq.py`. Currently
  blocked because that file is not compressed-tensors-aware (its
  `from_pretrained` call at lines 151-153 lacks
  `patch_compressed_linear_loading`); would need to apply the same
  treatment we did for `examples/llm_ptq/example_utils.get_model`.
- **Layer-wise streaming export** (load → decompress → quantise →
  save → free → next layer). Requires a new code path inside
  `_export_transformers_checkpoint`. Largest scope; biggest payoff.

### 2. Multi-node vLLM calibration broken on 0.20.1 *(BLOCKING K2.6)*

Follower rank's `EngineCore.__init__` calls `_initialize_kv_caches()`
(`vllm/v1/engine/core.py:283`) which calls
`model_executor.get_kv_cache_specs()` (`vllm/v1/executor/abstract.py:124`)
which calls `collective_rpc("get_kv_cache_spec")`. The follower's
`rpc_broadcast_mq` is None (only the head has it), so the assertion at
`vllm/v1/executor/multiproc_executor.py:351` fires:

```
AssertionError: collective_rpc should not be called on follower node
```

Reproduces with both real and dummy weights, on the user's reference flags
(`-cc.pass_config.fuse_allreduce_rms=False`,
`--mm-encoder-tp-mode data`, etc.) — confirmed not flag-related at our
level.

Fix candidates:
- `--distributed-executor-backend ray`. Requires a Ray cluster setup
  inside the SLURM job (head + workers via `ray start --head` / `ray
  start --address`). vLLM's container has Ray installed.
- Pin to a different vLLM version (0.19.x might not have this bug; the
  fakequant example README says it's "tested with 0.9.0 and 0.19.1").
- Patch vLLM. We've been told **not** to modify vLLM source, but the bug
  is in vLLM, so a clean fix probably has to come from there.

### 3. `w2_input_quantizer._amax` never observed *(quality — only matters for non-dynamic configs)*

Our wrapper hooks the activation entering the FusedMoE block (`w13_input`).
The activation between `gate*silu*up` and `down` (i.e., input to `w2`) is
internal to Marlin's kernel and not observable from a wrapper at the
FusedMoE level. Saved state has empty `{}` for every `w2_input_quantizer`
key. For dynamic NVFP4 (the configured default) this doesn't matter at
runtime; for static / AWQ variants it would silently degrade quality.

A deeper hook would intercept inside the per-token expert dispatch — likely
requires either a modelopt patch to vLLM's MoE forward (not desirable) or
running a parallel unfused forward path purely for stats collection.

### 4. Source-format coverage *(robustness)*

Only smoked against `CompressedTensorsWNA16MarlinMoEMethod` (Kimi-K2 INT4).
Other source formats (FP8 MoE, MXFP4 MoE, NVFP4 MoE) hit our
`_compressed_source = True` branch but the `forward` path assumes the
underlying Marlin kernel preserves the weights — that may not be true for
all formats. Untested.

### 5. Patches with wider blast radius than the use case *(cleanup before upstream)*

- `quant_module.py:248`: narrow the `hasattr(self, "weight")` gate to
  compressed-tensors source modules (e.g. `isinstance(self,
  CompressedLinear)`).
- `conversion.py`: remove the debug print + `_path` argument to
  `_replace_quant_module`.
- `dynamic.py:441`: keep or revert depending on whether the improved error
  message is wanted upstream.

### 6. Slow iteration cycle *(operational)*

Each end-to-end smoke is ~30 min on cache-hot /lustre (1.5 hr cold load):
~28 min `from_pretrained` + ~8 min `mtq.quantize` re-wrap on a real-weight
model. The meta-build path in
`${ROOT}/experiments/export_hf_kimi-k2-thinking/debug_wrap_meta.py` gets
the wrap step in ~30 s but is only useful for *path validation* —
`_export_transformers_checkpoint` needs real weights. Plan iteration cost
accordingly.

---

## What's been validated end-to-end

In `${ROOT}/experiments/vllm_calib_kimi-k2-thinking/state/4240786-rank{00..07}-of08.pth`:
8 files, 641 KB each, containing:
- `modelopt_state` (mode chain + version)
- `quantizer_state_dict` (1458 entries per rank, all keys present, only
  `experts.w13_input_quantizer._amax` populated as scalar bf16)
- `quant_config_summary` = `{qformat: NVFP4_EXPERTS_ONLY_CFG, dataset:
  nemotron-post-training-dataset-v2, calib_size: 4, max_sample_length:
  2048}`

These are noise (from dummy weights), but the file shape is the contract
the export side expects.

In `${ROOT}/experiments/vllm_calib_kimi-k2-thinking/state/4240786-merged-hf.pth`:
single merged HF-keyed file, 208458 entries (46080 of them non-empty),
46080 successfully loaded into a real K2-Thinking model on single-node
TP=8 by the export pipeline.

---

## Suggested next moves (lowest cost first)

1. **Address #5 (cleanup patches).** Free, needs to happen regardless of
   what comes next.
2. **Address #1 (export OOM).** Try the "skip the resmooth pass for
   non-AWQ configs" angle first. Read
   `${ROOT}/source/Model-Optimizer/modelopt/torch/export/unified_export_hf.py`
   lines 222 (`collect_shared_input_modules`), 387
   (`requantize_resmooth_fused_llm_layers`), 781 (call site). If the pass
   is gated by something AWQ-specific, see if `nvfp4_experts_only` trips
   it. If not, the OOM might disappear by adding a gate. If it can't be
   skipped, try multi-node export or a layer-wise streaming export.
3. **Address #2 (multi-node vLLM)** by trying
   `--distributed-executor-backend ray` once we have a working
   single-node export. If Ray is the answer, we also need a Ray cluster
   setup in SLURM (head + workers via `ray start --head` / `ray start
   --address`). Until #2 is solved, K2.6 calibration is blocked.
4. **Real-weights single-node end-to-end run** — once #1 is fixed, replace
   the dummy-weights smoke with `--calib-size 512 --max-seq-len 20000`
   on real weights, no `--dummy-weights`. ~4 hr first time; subsequent
   iterations cheaper.
5. **Address #3 (`w2_input` calibration)** — only worth it once the rest
   is unblocked AND we want production-quality calibration for
   non-dynamic configs. If we stay on dynamic NVFP4 it never matters.

---

## Pointers

- Tested vllm-openai container build script:
  `${ROOT}/containers/_layer_vllm_modelopt.sh` +
  `${ROOT}/containers/layer_vllm_modelopt.log`.
- Calibration smoke that succeeded (the source of the `4240786` state):
  `${ROOT}/experiments/vllm_calib_kimi-k2-thinking/logs/k2t-vllm-calib-dummy5-4240786-0.{out,err}`.
- Export smoke that hit the OOM (most informative log):
  `${ROOT}/experiments/export_hf_kimi-k2-thinking/logs/export-k2t-4240786-smoke-4241681.{out,err}`.
- vLLM 0.20.1 multi-node startup failure log:
  `${ROOT}/experiments/vllm_calib_kimi-k2.6/logs/k26-vllm-calib-dummy-4239300-*.err`.
- Meta-build wrap debug:
  `${ROOT}/experiments/export_hf_kimi-k2-thinking/debug_wrap_meta.py`
  (single-GPU, ~30 s, useful for fast iteration on the wrap + state-load
  steps without paying the HF load cost).

---

## Verifying the modelopt patches are still applied

After any modelopt update or rebase, run:

```bash
# 1. _QuantFusedMoEBase compressed-source branch
grep -n "_compressed_source" \
    ${ROOT}/source/Model-Optimizer/modelopt/torch/quantization/plugins/vllm.py

# 2. quant_module.py weight-hasattr gate
grep -n 'hasattr(self, "weight")' \
    ${ROOT}/source/Model-Optimizer/modelopt/torch/quantization/nn/modules/quant_module.py

# 3. fakequant_worker.py state dump
grep -n "_save_vllm_calibrated_state\|MODELOPT_VLLM_STATE_DIR" \
    ${ROOT}/source/Model-Optimizer/examples/vllm_serve/fakequant_worker.py
```

All three should produce non-empty output. If any is missing, the
calibration pipeline will break with the original assertions at the relevant
points (see "Known issues" → "Required for the calibration / export path").
