# Path 2: QAD Bridge/Core inside NeMo RL

The numbered items include environment setup and compatibility fixes; they are not a count of
independent bugs. Validation results and unresolved work are listed separately below.

## Setup and fixes

1. **Overlay the QAD source stack.** Keep the release image's Torch, Transformers, Python, and worker
   environments. Prepend the QAD Bridge, its vendored Core, and the actual QAD ModelOpt source.
   Verify resolved module paths in both driver and Ray workers.
2. **Add isolated dependencies.** Supply xattr, lark, wcmatch, bracex, braceexpand, filetype, mfusepy,
   rapidyaml, webdataset, deprecation, matching multi-storage-client 0.51.0 metadata, and Energon
   dependencies. Preserve the dependency manifest instead of upgrading packages globally.
3. **Use the existing distributed-state reader.** QAD's Core already reads `common_state` inside
   `.distcp`; no path-1 common-reader patch or fabricated `common.pt` is needed.
4. **Reconstruct from HF configuration.** Serialized YAML reconstruction failed on
   `Qwen3_5MoeVisionConfig.deepstack_visual_indexes` and `model_type` under Transformers 5.8.1.
   Finalize an HF-derived provider, compare every recorded architecture field, and select
   `megatron_lm`; this does not establish full serialized-config compatibility.
5. **Build and restore the native LM.** Use the language-only hybrid provider/spec and strict,
   explicitly allowed LM prefix mapping. Audit shard coverage and restored quantizers on all ranks,
   and check the actual pretrained iteration. See [model construction](../../compatibility/README.md#quantization-configuration-and-model-construction).
6. **Provide text MRoPE positions.** Expand 2-D positions to three identical axes.
7. **Enable SM103 attention.** Install the verified eligibility correction before model construction.
8. **Chunk log probabilities.** Use the recognized chunked-linear options with chunk size 128.
   See [attention and memory](../../compatibility/README.md#attention-and-log-probability-memory).

## Validation and remaining work

Both the earlier smoke and the full 13-step run passed. A separate-process reload passed with
pair-validation loss **0.00017418462911337598**. Full training plus reload took about 15m17s in the
shared suite. Export and SciCode evaluation remain unvalidated for this path.

Use `route12/run_full.sh`, stack mode `qad-bridge-core-modelopt`, and separate output directories
from path 1. Never interpret a successful overlay as an unmodified release-container result.

See [path.yaml](path.yaml) for the configuration record and [shared fixes](../../compatibility/README.md).
