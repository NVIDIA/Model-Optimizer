# Compatibility findings

These are experiment-local interventions, not upstream fixes or claims of compatibility across
arbitrary checkpoints. Keep each path's source/worker provenance, original failure, patch fingerprint,
and validation result. [source_manifest.yaml](source_manifest.yaml) identifies the omitted external
implementations by filename and SHA-256. Never infer the effective ModelOpt source from its version
string alone: experiment PYTHONPATH overrides changed behavior without updating package metadata.

## Distributed common state

The QAD checkpoint stores shared state in distributed `.distcp` files as a `common_state` object;
a separate `modelopt_state/common.pt` is absent. Older Core's common reader and Bridge's
`has_modelopt_state()` probe assume the legacy file. This concerns ModelOpt quantization metadata,
not a requirement that every NeMo RL input be a `.pt` model.

For the release-stack path, `checkpoint_compat.py` keeps legacy `common.pt` precedence and otherwise
loads `ShardedObject('common_state', None, (1,), (0,)).unique_key` via PyTorch distributed checkpoint's
FileSystemReader. Decode the single-element serialized object and require a dictionary. Patch both
`strategies.common.load_common` and `serialization.load_common`, because the latter imports the
function by value. Replace the Bridge probe's direct `torch.load(...COMMON_STATE_FNAME...)` with
`dist_checkpointing.load_common_state_dict()` and assert that exactly one expected call changed.

Validate legacy-only, distributed-only, mixed-layout, real source root state, and ModelOpt state;
missing or incomplete checkpoints must still fail. The QAD-aligned Core already supports the newer
layout, so avoid installing the old reader there. Never write fabricated `common.pt` files or
modify the source checkpoint to satisfy a filename check.

## Quantization configuration and model construction

The container-installed older ModelOpt rejected saved `effective_bits=4.5` and `constant_amax`.
Use the actual source that trained the QAD checkpoint and validate **every serialized mode's config**.
A successful `QuantizeConfig` import alone is insufficient. Path 6's narrow null-only `export_dir`
migration is a separate schema issue; preserve all meaningful settings.

The saved checkpoint is language-model-only, while the HF configuration describes a VLM. Prefix-only
mapping matched 5,519 LM state descriptors but exposed 138 requested vision descriptors per rank.
Use Bridge's `provide_language_model()` and `build_language_spec()`; do not invent vision weights or
load a partial generic VLM. Preserve Qwen's native gated-delta/full-attention hybrid pattern; a generic
quantized all-attention layer specification is not equivalent.

The [architecture record](architecture.yaml) stores the verified values. Finalize the HF-derived
provider and compare saved architecture fields: layer count, hidden/FFN
sizes, attention heads/groups/head dimension, MoE expert count and widths/router top-k, linear
attention heads/dimensions/convolution/pattern, vocabulary/tied weights, and rotary settings. Expand
integer linear-attention frequency to its per-layer pattern before comparing. If full Bridge YAML
instantiation fails under a different Transformers schema, using the HF-config/native-weight route
requires this parity audit; do not silently ignore configuration mismatches.

For a known LM-root checkpoint, `route1_key_compat.py` removes the `language_model.` prefix only from
requested descriptors when metadata confirms the checkpoint lacks that prefix. Include descriptors
produced by `ShardedTensorFactory.build_fn`, and use ShardedObject unique keys. Audit every request
against metadata and all-reduce the failure flag; any rank with a missing key stops the load.
Restore descriptor keys/factories afterward. Restrict mapping to the explicitly permitted checkpoint
paths. The source checkpoint remains unchanged.

For text-only MRoPE, expand 2-D position IDs to three identical axes. Install the native LM provider
both before and after generic quantization setup, which can overwrite its layer-spec factory.
Patch the worker's quantized checkpoint detection to check the actual supplied iteration in addition
to an assumed iteration zero. Immediately audit policy and reference TensorQuantizers on every rank;
require active quantizers rather than accepting a BF16 model that merely loaded weights.

## Attention and log-probability memory

On GB300/SM103, TE 2.17.1 omitted capability `(10, 3)` from FlashAttention eligibility for head
dimension > 192. Qwen3.6 uses 256. The experiment-local `enable_sm103_flash.py` corrects the eligibility
check without changing installed files or CUDA kernels. Verify forward/backward and backend selection.
Install once per process before model construction; double patching broke source introspection.
BF16 control workers need this correction too, not just quantized workers.

Separate from attention, unchunked log probabilities allocated approximately 39.58 GiB for FP32
full-vocabulary logits. On the release path, use `use_linear_ce_fusion_loss` and
`linear_ce_fusion_chunk_size: 128`. Old similarly named settings were silently ignored.

The BF16 Qwen subclass overrides generic GPT forward and rejected its chunked keyword. The
`vl_chunked_adapter.py` import hook preserves native preprocessing/decoder work and intercepts
`_postprocess` to call the official NeMo RL chunked log-probability routine. Temporarily bypass
auxiliary MTP on that path, restore the flag in `finally`, and restrict the verified adapter to TP=1.
The first call compares eight positions against explicit full projection/log-softmax; observed
maximum error 1.65e-6, tolerance 2e-3. Do not claim a generic architecture-independent adapter.

## Optimizer and optional dependencies

TE 2.17.1 FusedAdam's fixed tensor-handle pool can exhaust before GPU memory does. Path 4 uses
`te_optimizer_groups.py` to split groups at 512 tensors, preserving parameter order and all
hyperparameters. A structural check covered 5,600 parameters; full training and separate-process
optimizer reload then passed. Reapply the same grouping when restoring that optimizer state.

Disable gradient accumulation fusion where the required Apex fused gradient extension is absent;
path 3 needs this for both teacher and student. Add missing import dependencies to isolated targets,
not globally. Preserve Qwen-specific exclusions of unrelated tokenizer patches in provenance. Verify
imports in every Ray worker, since driver PYTHONPATH alone does not establish worker compatibility.

## Current-main output-head state

The latest pinned Core rejects `language_model.output_layer._extra_state` during sharded ModelOpt
restoration. The temporary guard accepts legacy omitted TE state only if the output head has no
active quantizers, no FP8/FP4 configuration, and ordinary FP32/FP16/BF16 weights. The quantized retry
still fails its assertion. This is unresolved, not permission to remove output-head quantization
state. The null-only schema migration and this extra-state guard are distinct patches.
