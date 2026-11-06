# Transformers evidence and measurements

[Return to the API guide](README.md).

## Focused implementation measurements

November 6, 2026 on NVIDIA GB300 / pinned HF 5.23 and Torch 2.14. These are bounded
random-prefix/component measurements, not full-model forecasts.

| Case | Original | Improved path | Numerical check |
|---|---:|---:|---|
| KL extra peak memory, INT8 candidate |536 MiB|44 MiB|Identical KL and cross-entropy|
| KL elapsed, INT8 candidate |72.3 ms|84.2 ms|Same case as above|
| KL extra peak memory, direct linear heads |536 MiB|32 MiB|KL difference below 1e-8|
| KL elapsed, direct linear heads |75.9 ms|101.4 ms|Same case as above|
| Hybrid packed teacher forwards |8|1|Identical captured targets|
| Hybrid packed teacher elapsed |41.4 ms|8.42 ms|Same case as above|

The KL fixture is a BF16 Llama prefix with two 256-wide layers, one 4096-token
sequence and 32768 vocabulary entries; chunks are 128 tokens / 8,192 vocabulary entries.
The memory figures are extra peak allocations above resident models. Chunking
saves memory here and increases elapsed time. Quantized/custom/TP heads retain
their actual module forward with token chunks; plain eligible heads additionally
slice the vocabulary projection.

The packing fixture is a BF16 Qwen3.5 hybrid prefix with two 128-wide layers and
eight 128-token documents. Batching raises extra peak allocation from 1.02 MiB to
2.81 MiB while remaining inside the original packed-token budget. This is why
batching remains family-owned and guarded by masking/statistics capabilities.

Eight grouped reconstruction comparisons across FP8, NVFP4 and IQ2_S preserved
exact outputs/gradients. Measured speed ratios reach 1.11× with at most 147,456
retained constant bytes in these cases. These are component timings and do not
establish a full repair speedup. Expanded weights and trainable values are never
cached. Raw results and launch/source evidence are retained in the validation
workspace under `runs/bypass-hardening-20261106`.

## Current compatibility evidence

The November 6, 2026 release matrix completed 58 cases: 29 reduced random text
models in IQ2_XXS and NVFP4 W4A4, with three scale-only updates, four unique
128-token sequences per update and max PTQ. The runtime was pinned Transformers
5.23/Torch 2.14 with native BF16 precision policies. Eight-GPU runs combined CP2,
dense FSDP4 and EP2/expert FSDP2 for routed models.

Every case reported `replicated_full_context` CP. This verifies masked objective
and gradient ownership while replicating full forward/replay work; it does not
establish CP compute or activation-memory savings.

Fifty-seven cases meet the original 0.3% relative-L2 native-field criterion.
Muse Spark NVFP4 uses a documented 2.5% scalar bound; observed variation is
1.20%, with identical codes/block scales. Its FP32 eager oracle is within
0.00021% on these scalars, but the BF16 path uses grouped experts. Other scalar
and all vector field bounds remain 0.3%. These short random-model comparisons
are not pretrained quality, long-run equivalence or original-kernel parity.

The validation workspace retains `COMPATIBILITY.md`, case JSON and diagnostic
source review under `runs/transformers-aa-compatibility-20261106`.
See [model coverage](MODEL_COVERAGE.md) and [validation](VALIDATION.md).

## Historical standalone Transformers prefix timings

These retained Transformers measurements were collected on NVIDIA GB300 on
November 5, 2026, using the earlier Transformers 5.18 implementation. They are
historical measurements, not reruns of the current native mesh-manager code.

The model is a random four-layer, full-width prefix of Qwen3-30B-A3B, retaining
128 experts, top-8 routing and the original vocabulary. Global batch is four,
microbatch one, with five warmup and eight measured updates. Base weights are
BF16, PTQ uses max, and repair uses Adagrad scale-only updates. FP8/NVFP4 native
scale reconstruction is batched. All rows below use **one GPU**.

| Tokens/sequence | Format | Seconds/update | Peak allocated (GiB) |
|---:|---|---:|---:|
| 8,192 | IQ2_S | 0.275 | 17.84 |
| 8,192 | FP8 W8A16 | 0.233 | 17.08 |
| 8,192 | NVFP4 W4A16 | 0.406 | 19.68 |
| 131,072 | IQ2_S | 6.847 | 42.35 |
| 131,072 | FP8 W8A16 | 6.824 | 41.59 |
| 131,072 | NVFP4 W4A16 | 6.965 | 43.50 |

The 128K experiment explicitly raised the source's 40,960-position configured
range while retaining RoPE theta. It makes no long-context quality claim.
Timing excludes construction, PTQ and finalization. Allocated peaks include
teacher, base student and repair state but exclude deployment export and
external CUDA/NCCL allocation. No model checkpoint was written. Synthetic prefix
measurements do not forecast full-model throughput or pretrained quality.

The current [benchmark entry point](benchmark_transformers.py) writes source
revisions, runtime versions, physical parameter layouts, data lanes and execution
modes with its timing report. Remeasure a changed runtime/topology before making
a performance comparison. No current pretrained-model KLD result is claimed here.
