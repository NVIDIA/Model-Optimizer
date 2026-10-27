# Path 1: Release NeMo RL with compatibility fixes

The numbered items include environment setup and compatibility fixes; they are not a count of
independent bugs. Validation results and unresolved work are listed separately below.

## Setup and fixes

1. **Restore the matching ModelOpt source.** Keep the release container's Bridge/Core, Torch, and
   Transformers, but overlay the actual round-3 QAD ModelOpt source. The installed older package
   rejected `effective_bits=4.5` and `constant_amax`; preserve those saved settings.
2. **Read distributed common state.** Apply the dual-layout reader and Bridge state-probe fix.
   The old stack expects `modelopt_state/common.pt`, but this checkpoint stores `common_state`
   inside `.distcp`. Preserve legacy-file precedence; do not fabricate a `common.pt` file.
   See [distributed common state](../../compatibility/README.md#distributed-common-state).
3. **Reconstruct the native language model.** If the serialized Bridge configuration cannot be
   instantiated, finalize the provider from HF configuration, audit every saved architecture field,
   and select `megatron_lm`. Build the native language-only hybrid provider/spec rather than a full
   VLM or a generic all-attention model. This format label retains distributed ModelOpt state.
4. **Map and audit checkpoint keys.** Apply strict LM-root prefix mapping only to explicitly allowed
   checkpoints, including factories and shared objects. Check requested shards on every rank and
   fail on missing keys. See [model construction](../../compatibility/README.md#quantization-configuration-and-model-construction).
5. **Provide text MRoPE positions.** Expand 2-D positions to three identical axes for text-only inputs.
6. **Correct checkpoint detection and audit restoration.** Check the supplied iteration, not only
   `iter_0000000`. Audit restored policy/reference quantizers in each worker before optimization;
   a successful weight load alone is insufficient.
7. **Enable SM103 attention.** Apply the verified attention eligibility correction before model
   construction to avoid the memory-heavy fallback.
8. **Chunk log-probability computation.** Use the recognized `use_linear_ce_fusion_loss` option and
   `linear_ce_fusion_chunk_size: 128`; the similarly named earlier options were ignored.
   See [attention and memory](../../compatibility/README.md#attention-and-log-probability-memory).

## Validation and remaining work

Full training completed 13 steps on 104 pairs, followed by a separate-process policy/optimizer reload.
Diagnostic pair-validation loss after reload was **0.0002726738578419743**. The successful full
training plus reload suite segment took about 19m45s; this is not an end-to-end generation estimate.
No path-1 NVFP4 export or SciCode outcome is recorded here.

The source runner is `route1/run_full.sh`, invoking `aligned_stack/probe_stack.py` and
`aligned_stack/run_dpo.py`, then the same driver with `--reload`. Its runtime selector is
`QWEN36_DPO_STACK_MODE=stock-hf-qad-modelopt`. Verify the actual selector in the source snapshot
before rerunning; do not replace the package overlay with a version string alone.

See [path.yaml](path.yaml) for the configuration record and [shared fixes](../../compatibility/README.md).
