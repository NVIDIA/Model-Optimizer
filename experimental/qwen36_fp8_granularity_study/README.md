# Qwen3.6 FP8 granularity study

This experiment compares FP8 fake-quantization granularity on exactly two official checkpoints:

- `Qwen/Qwen3.6-35B-A3B`
- `Qwen/Qwen3.6-27B`

It runs one model and one candidate per process. It does not contain benchmark results and never
generates placeholder scores. Each run first records BF16 reference logits, then applies ModelOpt
quantization in place and measures the changed outputs.

## Candidate matrix

| `--recipe` | Weights | Activations | Purpose |
| --- | --- | --- | --- |
| `per_tensor_fp8` | static per-tensor E4M3 | static per-tensor E4M3 | Built-in `mtq.FP8_DEFAULT_CFG` W8A8 baseline |
| `per_tensor_fp8_weight_only_control` | static per-tensor E4M3 | BF16 | Weight-only control for the baseline |
| `block128_static_weight_only` | static 128x128 E4M3 | BF16 | Built-in `mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG` diagnostic ablation |
| `block128_dynamic_w8a8_research` | 128x128 E4M3, dynamic FP32 amax | last-axis block-128 E4M3, dynamic FP32 amax | Research W8A8 branch |
| `block128_dynamic_weight_only_control` | 128x128 E4M3, dynamic FP32 amax | BF16 | Weight-only control for the research branch |
| `mxfp8` | last-axis block-32 MXFP8/E8M0 | last-axis block-32 MXFP8/E8M0 | Built-in `mtq.MXFP8_DEFAULT_CFG` W8A8 baseline |
| `mxfp8_weight_only_control` | last-axis block-32 MXFP8/E8M0 | BF16 | Weight-only MXFP8 control |

Every candidate starts from the corresponding ModelOpt preset, including its base/default module
exclusions. The study also excludes MTP and shared-expert-gate modules. Quantization targets the
extracted language-model component, while the original architecture-aware wrapper produces logits.
This matters because Qwen3.6 checkpoints can expose a conditional-generation wrapper rather than a
plain `AutoModelForCausalLM`.

The block-128 research candidates deliberately use **top-level**
`QuantizerAttributeConfig.type: dynamic`. Their `block_sizes` dictionaries do not contain `type`.
This selects TensorQuantizer's static block reshape plus dynamically computed full-precision amax
and scaled-E4M3 fake quantization. It is different from MXFP8's nested
`block_sizes: {-1: 32, type: dynamic, scale_bits: e8m0}` kernel path. The research candidates are
fake-quant diagnostics and are marked non-deployable. Because this TensorQuantizer path caches its
first input shape, the study installs a research-only pre-hook that clears reshape state before
every invocation. This is required for shared fused-MoE input quantizers whose routed-token count
changes between experts and batches. The launcher also keeps outer batch and padded sequence
shapes matched for a controlled comparison; neither behavior is an export claim.

## Run

Use a fresh output directory for each candidate. Defaults are intentionally small (16 calibration
rows, 8 evaluation rows, sequence length 128, and batch size 1):

```bash
python experimental/qwen36_fp8_granularity_study/study.py \
  --model Qwen/Qwen3.6-27B \
  --recipe per_tensor_fp8 \
  --output-dir /path/to/results/qwen36-27b/per_tensor_fp8 \
  --reference-cache /path/to/results/qwen36-27b/reference-cache
```

Inspect the fully resolved configuration without loading a tokenizer/model, contacting a dataset,
or requiring a GPU:

```bash
python experimental/qwen36_fp8_granularity_study/study.py \
  --model Qwen/Qwen3.6-27B \
  --recipe block128_dynamic_w8a8_research \
  --output-dir /tmp/qwen36-plan \
  --dry-run-plan
```

For a larger job, set the dataset, sizes, lengths, and batches explicitly. Calibration rows are
packed by default; pass `--no-pack-calibration` to disable packing. Evaluation is never packed. When
both phases use the same dataset, evaluation starts after calibration's raw-source prefix. That is
`calib-size * 8` with the default packed calibration (matching dataset-utils' 8x raw-document
oversampling), or `calib-size` with unpacked calibration. This keeps evaluation source rows
disjoint. `--eval-offset` overrides the derived value; when datasets differ its default is zero.

```bash
python experimental/qwen36_fp8_granularity_study/study.py \
  --model Qwen/Qwen3.6-35B-A3B \
  --revision main \
  --recipe mxfp8 \
  --output-dir /path/to/results/qwen36-a3b/mxfp8 \
  --reference-cache /path/to/results/qwen36-a3b/reference-cache \
  --calib-dataset cnn_dailymail --calib-size 128 --calib-seq-len 512 \
  --eval-dataset cnn_dailymail --eval-size 32 --eval-seq-len 512 \
  --activation-mse-size 32 \
  --calib-batch-size 1 --eval-batch-size 1 \
  --dtype bfloat16 --seed 1234 --device-map auto
```

## Exact reference cache

The first candidate writes BF16 reference-logit batches. Later candidates can reuse them only when
the canonical manifest matches exactly. The key covers model/revision/model dtype and config,
tokenizer identity and special-token IDs, evaluation snapshot/offset/size/shape, exact token-derived
sample IDs and batch hashes, source commit and study-script hash, package/CUDA versions, container
identity, GPU type, backend flags, resolved device map, seed, and reference storage dtype. Every
batch file also has a SHA-256 digest.

If a cache already contains entries but none match, the run fails before quantization. Pass
`--recompute-reference-cache` to create a new hash-addressed entry without deleting old entries.
This strict behavior prevents a superficially similar run from comparing against the wrong BF16
forward pass.

## Measurements and artifacts

`results.json` is replaced atomically at phase boundaries and contains wall times plus failure
tracebacks when a run aborts. A successful run reports:

- batch-streamed logit MSE, global RMSE, MAE, centered MSE, and reference-variance-normalized MSE;
- `KL(reference || quantized)`, reverse KL, and Jensen-Shannon divergence;
- target-token log-probability error, NLL delta, top-1 agreement, and symmetric top-5 set overlap;
- per-token distributions/quantiles and per-sample values/distributions;
- equal-document paired bootstrap intervals over the 32-document deterministic screen;
- direct ModelOpt `TensorQuantizer` MSE for every mapped weight and fused-expert slice, plus
  `mtq.compute_quantization_mse` for input quantizers over a configurable calibration prefix,
  retaining every executed quantizer name and family summary;
- declared and observed quantizer coverage, amax/scale metadata, and logical payload-plus-scale cost
  estimates;
- git, Python/package, CUDA/GPU, model/tokenizer, dataset/sample, and reference-cache provenance.

Artifacts in each candidate directory are:

```text
plan.json
reference_manifest.json
reference_logits/        # files, or a symlink to the exact shared-cache entry
results.json
```

The cost model is an arithmetic estimate for the intended compressed representation. ModelOpt fake
quantization retains original tensors and therefore does not realize that process-memory saving.

## Tests

The focused tests are CPU-only:

```bash
pytest -q experimental/qwen36_fp8_granularity_study/test_study.py
```

They cover metric identities/orientation, left-padding, gain-sensitive variance normalization,
token- and document-level aggregation, top-level versus nested dynamic recipe structure, routed
shape refresh, fused-expert quantizer naming/cost mapping, weight-only controls, scale-cost
arithmetic, held-out row selection, runtime-sensitive cache keys, and the network/model/GPU-free
dry-run schema.
