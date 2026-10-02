# Serve fakequant models with vLLM

This is a simple example to demonstrate calibrating and serving ModelOpt fakequant models in vLLM.

Compared with realquant, fakequant is 2-5x slower, but doesn't require dedicated kernel support and facilitates research.

The general fakequant example is tested with vLLM 0.19.1, 0.26.0, 0.28.0, 0.29.0 and 0.30.0. The
compact NVFP4 attention worker documented below requires vLLM 0.15.0 or newer.

The `vllm` CLI shim does not support vLLM 0.9.0. Use one of the tested releases above.

## Prepare environment

Run the commands below from the ModelOpt repository root (`/workspace/Model-Optimizer`
in the Docker image).

Use the Dockerfile to build an environment with vLLM 0.30.0:

```bash
docker build -f examples/vllm_serve/Dockerfile -t vllm-modelopt:v0.30.0 .
```

The Dockerfile defaults to vLLM 0.30.0. Override `VLLM_VERSION` only when
building for a different tested release.

For a direct installation from the ModelOpt repository root, install the tested vLLM
release and the ModelOpt extras used by this example:

```bash
python3 -m pip install "vllm==0.30.0"
python3 -m pip install -e ".[all,mlflow]"
```

See the [ModelOpt installation guide](../../docs/source/getting_started/_installation_for_Linux.rst)
for details about installing partial dependency sets.

## Calibrate and serve fake quant model in vLLM

Step 1: Configure fakequant with ModelOpt CLI flags. Each flag falls back to its
corresponding environment variable when omitted:

| CLI flag | Environment fallback | Description |
| --- | --- | --- |
| `--modelopt-quant-cfg` | `QUANT_CFG` | Weight/activation quantization config |
| `--modelopt-kv-quant-cfg` | `KV_QUANT_CFG` | KV-cache quantization config |
| `--modelopt-quant-file-path` | `QUANT_FILE_PATH` | Path to `quantizer_state.pth` in a Megatron (MCore) vLLM fakequant export; requires a quantization config or recipe |
| `--modelopt-state-path` | `MODELOPT_STATE_PATH` | Path to `vllm_fq_modelopt_state.pth` in an HF vLLM fakequant export (full ModelOpt state) |
| `--modelopt-recipe-path` | `RECIPE_PATH` | ModelOpt PTQ recipe YAML |
| `--modelopt-quant-dataset` | `QUANT_DATASET` | Calibration dataset |
| `--modelopt-quant-calib-size` | `QUANT_CALIB_SIZE` | Calibration sample count |
| `--modelopt-calib-batch-size` | `CALIB_BATCH_SIZE` | Calibration batch size |

Install the shim to expose these options through the ordinary `vllm` command:

```bash
python3 -m pip install -e examples/vllm_serve
```

Step 2: Serve with any stock vLLM options plus the ModelOpt flags:

```bash
vllm serve <model_path> -tp 8 --host 0.0.0.0 --port 8000 \
  --modelopt-quant-cfg NVFP4_DEFAULT_CFG \
  --modelopt-quant-dataset cnn_dailymail \
  --modelopt-quant-calib-size 512
```

When fakequant is requested, the shim selects `fakequant_worker.FakeQuantWorker`
unless `--worker-cls` is supplied. Without ModelOpt quantization settings,
the shim delegates to the stock vLLM CLI. Existing direct-script workflows can still invoke
`python3 examples/vllm_serve/vllm_serve_fakequant.py` from the repository root.

Hybrid attention/Mamba models such as Nemotron 3 Nano are supported on vLLM 0.26.0, 0.28.0, 0.29.0 and
0.30.0. For example, calibrate and serve with NVFP4 KV-cache fakequant as follows:

```bash
vllm serve <nemotron3_nano_model_path> -tp 8 \
  --modelopt-kv-quant-cfg NVFP4_KV_CFG --modelopt-quant-calib-size 512 \
  --max-model-len 8192 --enforce-eager --host 0.0.0.0 --port 8000
```

Calibration uses dedicated scratch KV-cache blocks, so reducing `--max-num-batched-tokens`
is not required to avoid NaNs.

For vLLM versions that expose `--moe-backend`, this launcher defaults to `--moe-backend triton`.
ModelOpt expert fakequant needs a decomposed MoE backend so both expert GEMMs are visible during
calibration.

A pre-quantized checkpoint (for example FP8) can be served with a recipe that leaves its quantized
layers alone, such as a KV-cache-only recipe: those layers run unchanged, and a recipe that
fake-quantizes their weights or activations raises an error instead. Pass `--moe-backend auto` for
such MoE checkpoints: the `triton` default is only needed to fake-quantize experts, and vLLM
rejects it for NVFP4 experts.

Step 3: test the API server with curl:

```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions"     -H "Content-Type: application/json"     -d '{
          "model": "<model_path>",
          "messages": [
              {"role": "user", "content": "Hi, what is your name"}
          ],
          "max_tokens": 8
        }'

```

Step 4 (Optional): using lm_eval to run evaluation

```bash
lm_eval --model local-completions --tasks gsm8k --model_args model=<model_name>,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,batch_size=128,tokenizer_backend=None
```

## Fake-quantize the MLA KV cache

MLA models such as DeepSeek-V3 and GLM-5.3-Flash cache one latent vector per token instead of
separate keys and values; RoPE models such as DeepSeek-V3 also cache a small RoPE key. Their fake
quantizers are `kv_c_bmm_quantizer` and `k_pe_bmm_quantizer` on vLLM's `MLAAttention`.
`KV_QUANT_CFG` presets (e.g. `NVFP4_KV_CFG`) are extended to both automatically, with the same
format for both. The KV-cache units in a recipe (`*[kv]_bmm_quantizer`) do not match them, so a
recipe imports the `configs/ptq/units/kv_nvfp4_mla` unit instead, which uses NVFP4 for the latent
and FP8 for the RoPE key. For example, `kv_nvfp4_mla_only.yaml` quantizes the MLA KV cache alone
(weights and activations stay unquantized):

```yaml
# modelopt-schema: modelopt.recipe.config.ModelOptPTQRecipe
imports:
  base_disable_all: configs/ptq/units/base_disable_all
  kv_nvfp4_mla: configs/ptq/units/kv_nvfp4_mla

metadata:
  description: Fake quantization of the MLA KV cache only (NVFP4 latent, FP8 RoPE key).
quantize:
  algorithm: max
  quant_cfg:
    - $import: base_disable_all
    - $import: kv_nvfp4_mla
```

```bash
RECIPE_PATH=kv_nvfp4_mla_only.yaml python vllm_serve_fakequant.py <model_path> -tp 8 \
  --host 0.0.0.0 --port 8000
```

This recipe also runs on the FP8 GLM-5.3-Flash release, since it leaves the FP8 layers alone (add
`--moe-backend auto`, see above).

Notes:

- The latent and RoPE key are fake-quantized before vLLM writes them to the cache, so attention
  over the tokens of the current step sees the quantized values too.
- Serve with a BF16 KV cache: an FP8 cache would quantize the fake-quantized latent a second time.
  `--kv-cache-dtype auto` is BF16 unless the checkpoint declares a quantized KV cache (e.g. a
  ModelOpt export with an FP8 KV cache); then pass `--kv-cache-dtype bfloat16`.

## Tracking a serve with MLflow

Pass `--mlflow <tracking-uri>`, or set MLflow's own `MLFLOW_TRACKING_URI`, to record what
this server actually quantized, so the numbers an evaluation produces can be traced back to
a recipe:

```bash
vllm serve <model_path> -tp 8 --modelopt-recipe-path <PATH_TO_RECIPE> \
  --host 0.0.0.0 --port 8000 \
  --mlflow https://<your-mlflow-server>/
```

This is the *quantization* tracking server. It is unrelated to any tracking server an
evaluation harness exports its scores to — NeMo Evaluator Launcher, for instance, has its
own `export.mlflow.tracking_uri`. Keep the two separate.

Quantization runs in the vLLM **worker**, not in the CLI shim, so that is where
the run is recorded: the launcher validates the URI and hands the settings to the workers
through the environment, and global rank 0 opens the run. It opens *before the weights
load*, so a bad URI or a missing token fails within seconds rather than after a load and a
full calibration, and it closes `FINISHED` once the model is quantized and warmed up —
serving itself is not tracked.

<details>
<summary>Uploaded artifacts</summary>

| Artifact | Contents |
| --- | --- |
| `command.txt` | The launcher's full invocation, copy-pasteable, with credentials masked |
| `version.txt` | The ModelOpt version that ran |
| `recipe/resolved_recipe.yaml` | `RECIPE_PATH` with its `$import`s expanded, so it stands alone |
| `recipe/quant_cfg.yaml` | `QUANT_CFG` and `KV_QUANT_CFG` merged, plus any MLA fixup — only when no recipe is used, since a recipe's config is already in `resolved_recipe.yaml` |
| `logs/<script>.log` | The rank-0 worker's Python stdout/stderr, including the traceback if it crashed |
| `summary/quant_summary.txt` | The per-quantizer summary |

</details>

The quantization settings from the table above are logged as searchable params, alongside
the serving settings (`tensor_parallel_size`, `max_model_len`, `dtype`, `kv_cache_dtype`, …)
and `user` / `hostname` / `modelopt_version` / `git_sha` / `vllm_version` tags. The
`checkpoint_path` tag is the checkpoint being served, which is the same key
`examples/hf_ptq/hf_ptq.py` tags its runs with — so the PTQ run that produced a checkpoint
and every serve of it can be found together.

Other flags:

- `--mlflow-experiment` — defaults to
  `$USER/vllm_serve_fakequant/<model basename>-<recipe name>`, falling back to
  `$QUANT_CFG`/`$KV_QUANT_CFG` when no recipe is used.
- `--mlflow-run-name` — defaults to the UTC start time, `YYYYmmdd-HHMMSS`.
- `$MLFLOW_TRACKING_URI` enables tracking on its own; `--mlflow` overrides it. A URI taken
  from the environment is best-effort — if the client is missing or the server is
  unreachable the server warns and serves untracked. An explicit `--mlflow` fails loudly
  instead.

Tracking needs the client: `pip install nvidia-modelopt[mlflow]` (already in this example's
`Dockerfile`). Authentication uses MLflow's own environment variables
(`MLFLOW_TRACKING_TOKEN`, or `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`); with
`--distributed-executor-backend ray` those are forwarded to the workers along with the
tracking settings, since a Ray worker starts with a clean environment.

## Load QAT/PTQ model and serve in vLLM (WIP)

Step 1: export the model with bf16 weights and quantizer state. To export the model:

- For **HF** models, use `examples/hf_ptq/hf_ptq.py` with `--vllm_fakequant_export`:

```bash
python3 examples/hf_ptq/hf_ptq.py \
  --pyt_ckpt_path <MODEL_PATH> \
  --recipe <PATH_TO_RECIPE> \
  --calib_size 512 \
  --export_path <EXPORT_DIR> \
  --vllm_fakequant_export \
  --trust_remote_code
```

  This creates `<EXPORT_DIR>/vllm_fq_modelopt_state.pth` (ModelOpt quantizer state for vLLM fake-quant reload) and saves the HF-exported model under `<EXPORT_DIR>` (config/tokenizer/weights).

  Note: `--pyt_ckpt_path` can point to either an HF checkpoint or a ModelOpt-saved checkpoint (e.g., a QAT/QAD checkpoint produced by `examples/llm_qat/train.py`). If the input checkpoint is already quantized, the script will **skip re-quantization** and only export artifacts for vLLM fakequant reload.

- For **MCore** models, export the model with flag `--export-vllm-fq` as described in [Megatron-LM README](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt#-nvfp4-quantization-qauntization-aware-training-and-model-export). This generates `quantizer_state.pth`, which contains quantizer tensors for vLLM reload via `QUANT_FILE_PATH`.

Step 2: use the exported artifacts when serving:

- **HF export**: pass the exported `vllm_fq_modelopt_state.pth` via `--modelopt-state-path`

```bash
# HF
vllm serve <model_path> -tp 8 --modelopt-state-path <vllm_fq_modelopt_state.pth> \
  --host 0.0.0.0 --port 8000
```

- **MCore export**: pass the exported `quantizer_state.pth` via `--modelopt-quant-file-path` and set `--modelopt-quant-cfg` to match the MCore quantization recipe

```bash
# MCore
vllm serve <model_path> -tp 8 --modelopt-quant-cfg <quant_cfg> \
  --modelopt-quant-file-path <quantizer_state.pth> --host 0.0.0.0 --port 8000
```

## Fake-quantize the sparse-attention indexer query and K cache

The sparse-attention models DeepSeek-V4 and GLM-5.3-Flash keep a separate indexer key cache next
to the attention KV cache and score it against an indexer query. Their fake quantizers are
`indexer_k_quantizer` and `indexer_q_quantizer` on the indexer module; the KV-cache presets
(`*[kv]_bmm_quantizer`) leave them disabled, so enable them by importing the
`configs/ptq/units/indexer_k_nvfp4` and `configs/ptq/units/indexer_q_nvfp4` units into a recipe.
For example, `indexer_nvfp4_only.yaml` quantizes the indexer key cache and query alone (weights,
activations and the attention KV cache stay unquantized):

```yaml
# modelopt-schema: modelopt.recipe.config.ModelOptPTQRecipe
imports:
  base_disable_all: configs/ptq/units/base_disable_all
  indexer_k_nvfp4: configs/ptq/units/indexer_k_nvfp4
  indexer_q_nvfp4: configs/ptq/units/indexer_q_nvfp4

metadata:
  description: NVFP4 fake quantization of the sparse-attention indexer key cache and query only.
quantize:
  algorithm: max
  quant_cfg:
    - $import: base_disable_all
    - $import: indexer_k_nvfp4
    - $import: indexer_q_nvfp4
```

Save the recipe above as `indexer_nvfp4_only.yaml` in the repository root, then serve:

```bash
vllm serve <model_path> -tp 8 --modelopt-recipe-path indexer_nvfp4_only.yaml \
  --host 0.0.0.0 --port 8000
```

Drop one of the two units to quantize only the key cache or only the query. To add them to an
existing recipe, append the imports and their `$import` entries to that recipe's `quant_cfg`.

Notes:

- vLLM computes the indexer key and query inside fused kernels that quantize them to FP8, so the
  FP8 results (the cache entries each step wrote, and the query) are dequantized, fake-quantized
  and quantized to FP8 again. The QDQ input therefore carries the FP8 rounding (at most 2^-4
  relative).
- This requires vLLM's FP8 indexer cache, the default (`indexer_kv_dtype` in the attention
  config); enabling the quantizers with DeepSeek-V4's MXFP4 indexer cache
  (`indexer_kv_dtype="mxfp4"`) is rejected.
- vLLM quantizes the DeepSeek-V4 indexer key and query without a Hadamard rotation and the
  GLM-5.3-Flash ones after one, so the fake quantization applies in that basis.

## Serve a model with sparse attention in vLLM

Apply ModelOpt sparse attention at serve time. Right after model load, the launcher replaces each native attention implementation with its matching ModelOpt adapter: `ModelOptSparseAttentionImpl` for FlashAttention or `ModelOptSparseFlashInferImpl` for FlashInfer. Both adapters use the same Triton kernel with paged KV cache support.

The configuration is read from the checkpoint's `config.json` `sparse_attention_config` block, written by ModelOpt's HF export. The launcher restores calibrated skip-softmax metadata and N:M sparse-softmax metadata (`sparsity_n`, `sparsity_m`, `dense_sink_tokens`, `dense_recent_tokens`). Checkpoints exported with both metadata entries use ModelOpt Triton for sparse prefill launches; launches without active sparse work delegate back to the native backend selected by vLLM.

Workflow:

1. Calibrate and export the model with `examples/llm_sparsity/attention_sparsity/hf_sa.py`. This writes `sparse_attention_config` into the exported checkpoint's `config.json`.
2. Serve the exported checkpoint with `--enforce-eager` (CUDA graph capture is not yet validated with the sparse attention kernel — see Known Problems):

   ```bash
   python3 examples/vllm_serve/vllm_serve_sparse_attn.py <EXPORT_DIR> --enforce-eager -tp 8 --host 0.0.0.0 --port 8000
   ```

If the checkpoint has no `sparse_attention_config`, the sparse-only installer passes through and vLLM runs unchanged. Whole-model fakequant flows use `vllm serve` with ModelOpt flags; the compact attention-only path is below.

### Calibrate skip-softmax thresholds through vLLM

Instead of the HF path in step 1, thresholds can be calibrated directly through vLLM — over the paged KV cache, for both prefill and decode, with tensor parallelism. Pipeline and data parallelism are not supported by calibration.

```bash
# One-time: fetch the RULER essay haystack
bash examples/llm_sparsity/attention_sparsity/download_ruler_data.sh

python3 examples/vllm_serve/calibrate_sparse_attn.py <CKPT> \
  --calib_data_dir examples/llm_sparsity/attention_sparsity/data \
  --target_sparse_ratio 0.5 \
  --decode_tokens 32 --tensor_parallel_size 8 --update_checkpoint_config
```

Calibration always writes `sparse_attention_config.json` in the current directory.
`--update_checkpoint_config` also merges that configuration into `<CKPT>/config.json` in
place, which lets `vllm_serve_sparse_attn.py` load it automatically. This option requires
`<CKPT>` to be a local checkpoint directory; without it, merge the generated configuration
into the checkpoint manually before serving.

Calibration prompts default to the **RULER dataset** via the same `RulerDatasetBuilder` the HF calibration path uses (`--calib_samples` / `--calib_max_seqlen` mirror the HF defaults of 24 / 32768), so vLLM- and PyTorch-calibrated thresholds are fit on identical data. `--prompts_file` (one prompt per line) substitutes custom calibration data.

`install_vllm_skip_softmax_calibration` (called by `sparse_attn_worker.SkipSoftmaxCalibWorker` at model load) swaps calibration adapters onto each attention layer not listed in the checkpoint's existing skip-softmax `ignore` policy after validating all selected layers — eager execution is required, model and KV-cache dtypes must be fp16/bf16, and no attention Q/K/P/V fakequant may be active. During `llm.generate`, the paged Triton calibration kernel computes full dense attention — no sparsification is applied to generation, though the dense kernel's numerics differ slightly from the native backend's — while counting, per candidate threshold, how many KV tiles the skip criterion would drop. The driver then collects **raw tile counts from every TP rank** (each rank only measures its head shard), merges them, fits `scale_factor = a * exp(b * sparsity)` once per phase, and writes the same canonical `sparse_attention_config` block the HF export produces — preserving the existing skip-softmax layer policy and any exported N:M sparse-softmax groups — so the serving workflow above picks it up unchanged.

Calibration and serving use the same 128-token KV-tile skip granularity and the same 128-row Q tile for prefill, so serving realizes the calibrated skip decision. One-token decode uses a 16-row Q compute tile because its padding rows cannot affect the decision. Serving autotunes only the execution schedule (`num_warps` / `num_stages`); measurement remains a single fixed launch because its counters have side effects.

The reusable serving policies live in `modelopt/torch/sparsity/attention_sparsity/plugins/vllm_runtime.py`. `install_vllm_sparse_attention_from_checkpoint` installs checkpoint-driven sparse-only attention, while `install_vllm_nvfp4_attention` installs fixed NVFP4 Q/K/P/V with optional checkpoint sparsity. Both validate every selected layer before publishing any replacement implementation and return a `VllmAttentionInstallReport` with the installed layer names and backend counts.

`sparse_attn_worker.py` only invokes these APIs after vLLM loads the model. It retains `SparseAttnWorker` as the launcher's default and provides `QuantSparseAttnWorker` for the compact NVFP4 policy. Other vLLM integrations can invoke the same library APIs directly:

```python
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm_runtime import (
    install_vllm_nvfp4_attention,
)

report = install_vllm_nvfp4_attention(model_runner, sparse_cfg="checkpoint")
```

Limitations:

- vLLM V1 chunked prefill and prefix-cache suffix attention are supported by offsetting query positions into the longer KV span. This applies to sparse-only serving; quantized attention installs and skip-softmax calibration reject `enable_prefix_caching` (quantize-on-write and per-request measurement both require uncached prefills).
- Skip-softmax calibration requires pipeline- and data-parallel size 1 because raw count records align only across tensor-parallel head shards; data-parallel replicas serve different requests.
- `SparseAttnWorker` CUDA graph capture is not validated yet — use `--enforce-eager`. Checkpoints with a calibrated `decode` `threshold_scale_factor` are rejected at install under a FULL decode CUDA graph mode (including vLLM's default `FULL_AND_PIECEWISE`): the captured graph would replay one request's stale threshold.
- Sparse-only installs validate engine-level compatibility like quantized installs do: decode context parallelism, DBO, speculative decoding, and FULL mixed-batch CUDA graphs are rejected (prefix caching remains supported, per the bullet above).

### Compact NVFP4 attention worker

vLLM 0.15.0 or newer is required when either worker activates a ModelOpt attention transform. Importing `SparseAttnWorker`, or using it with no checkpoint sparse metadata, does not resolve quant-only APIs.

Use the same launcher with the compact worker. By default, vLLM selects the backend for the model and platform; NemotronH on Blackwell selects FlashInfer:

```bash
python3 examples/vllm_serve/vllm_serve_sparse_attn.py <MODEL_PATH> -tp 8 \
  --no-enable-prefix-caching \
  --worker-cls sparse_attn_worker.QuantSparseAttnWorker
```

The installer supports both FlashInfer and FlashAttention, and the worker prints the installed adapter counts. Pass `--attention-backend FLASHINFER` or `--attention-backend FLASH_ATTN` only when an explicit override is needed.

This attention-only path applies a fixed dynamic block-16 NVFP4 fakequant format to Q/K/P/V. Q is dynamic; missing K/V scales default to global scale 1.0, and P defaults to amax 1.0. Existing scalar attention amax values are preserved, but this path does not calibrate or restore them itself. It does not re-quantize realquant Linear or MoE weights. An optional checkpoint `sparse_attention_config` is still honored for N:M sparse softmax; calibrated skip-softmax groups are rejected in combination with attention quantization, because quantized Q/K/P change the score distribution the skip thresholds were calibrated on.

Decode uses a fixed 32-split, 128-key-tile schedule. P QDQ consumes split-local,
unnormalized online-softmax probabilities, so changing that schedule can change
quantized results; split count is part of the numerical contract.

K is QDQ before its cache write, while V is written pristine. Complete 16-token V groups are finalized once in cache; an incomplete tail remains pristine and is QDQ on read. P@V therefore sees uniform fakequant values without re-quantizing the tail.

Supported configurations are regular decoder self-attention with FlashInfer or FlashAttention, fp16/bf16 model and KV cache, equal Q/K/V head dimensions that are multiples of 16, and DCP 1. The FlashInfer adapter preserves both NHD and HND cache strides and separates mixed decode/prefill launches so each phase keeps its own kernel contract. The default `FULL_AND_PIECEWISE` mode remains enabled for fixed N:M and attention-only NVFP4; checkpoints with calibrated decode `threshold_scale_factor` must use a non-`FULL` decode graph mode such as `--enforce-eager` because the live sequence length is not replayed as a Python scalar.

Unsupported features are sliding window, ALiBi, softcap, sinks, FP8 KV cache, cross/encoder/MLA attention, KV sharing or transfer, prefix caching, speculative decoding, DBO/ubatching, and `FULL` mixed/prefill CUDA graphs.

## Known Problems

1. **MCore reload does not use `MODELOPT_STATE_PATH`**; use `QUANT_FILE_PATH` and make sure `QUANT_CFG` matches the quantization recipe used for the original MCore model (otherwise quantizer keys/config won’t align).
2. KV cache quantization export and reload is not supported in MCore yet.
3. **Keep vLLM's torch.compile cache off** (`VLLM_DISABLE_COMPILE_CACHE=1`, which this launcher sets by default). The cache is not keyed on the fake quant, so a graph compiled earlier for the same model without it is reused and the fake quant is silently skipped. If you run `FakeQuantWorker` without this launcher, set it yourself or pass `--enforce-eager`.
