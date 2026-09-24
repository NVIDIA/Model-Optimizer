# Linear-attention numerical studies

These examples study differentiable numerical emulation. The matmul backend
materializes intermediates and can be substantially slower than fused FLA.

## GDN overhead

```bash
PYTHONPATH=. python examples/llm_qat/linear_attention/benchmark_prefill.py \
  --batch 1 --length 1024 --heads 4 --dim 64 --repeats 20 \
  --output prefill-overhead.json
```

The JSON contains all interleaved forward/backward timing samples and extra tensor
memory measurements. Compilation is excluded; CPU submission overhead is included.
Add `--attention kda` to measure the per-channel KDA implementation.

## KDA checkpoint and data

The initial model is [arcee-ai/AFM-4.5B-Base-KDA-Only](https://huggingface.co/arcee-ai/AFM-4.5B-Base-KDA-Only)
at revision `01ad2e06ee4f1214193c17b69e09105a9b257e80`. Its implementation uses
FLA's `KimiDeltaAttention` directly. The dataset is
[Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), configuration
`wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`.

Install the example dependencies, then stage those exact revisions with model
weights, tokenizer, audited model code, and the three Parquet splits. Supply local
paths to the runner; it does not fetch weights or data. The checkpoint's custom
model code requires an explicit `--trust-remote-code` opt-in. Review its pinned
Python files before enabling that flag.

```bash
pip install -r examples/llm_qat/linear_attention/requirements.txt

PYTHONPATH=. python examples/llm_qat/linear_attention/quality_study.py \
  --model /path/to/arcee-kda \
  --train-data /path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --eval-data /path/to/wikitext-2-raw-v1/validation-00000-of-00001.parquet \
  --eval-split validation --trust-remote-code \
  --quant-config examples/llm_qat/linear_attention/configs/kda_prefill_fp8.json \
  --length 128 --train-steps 1 --eval-blocks 4 --seed 2026 \
  --output fp8-smoke.json
```

The one-step command is an integration smoke test. For matched studies, run the
same workload from the same original checkpoint with each numerical configuration:

| Trial | Configuration |
| --- | --- |
| Exact fused FLA control | Omit `--quant-config` |
| Exact materialized control | `configs/kda_exact_matmul.json` |
| FP8 prefill operands | `configs/kda_prefill_fp8.json` |
| NVFP4 prefill operands | `configs/kda_prefill_nvfp4.json` |

The exact materialized control explicitly rounds the already-FP32 `output_add`
to FP32 to select the emulation path. It introduces no lower-precision rounding.
First establish agreement with the fused control before interpreting a candidate's
quality. Candidate `before` measurements isolate its untrained numerical change;
compare trained candidates against an equally trained exact control. Choose
settings using validation data, then run the fixed selected settings on the test
split. Never choose approximation degrees or training hyperparameters on test data.

### Protocol

- Join Parquet text rows with two newlines and tokenize without added special
  tokens. Use the first requested contiguous blocks, each containing `length+1`
  tokens and predicting `length` tokens. Blocks do not overlap; cache/state reset
  for each block. This protocol is for paired comparisons, not a claim of the
  checkpoint's published WikiText perplexity.
- Shuffle the training blocks with the recorded seed. Use the same seed, block
  count, token hashes, length, and learning rate across all trials.
- Train KDA attention parameters only. Keep them and AdamW moments in FP32, use
  BF16 autocast and activation checkpointing, and freeze other model parameters
  in BF16. Preserve the checkpoint's FP32 gate parameters when loading. AdamW has
  zero weight decay, default learning rate `1e-5`, and gradient clipping at 1.
- Record per-block NLL, predicted token counts, source/data/token hashes, package
  versions, training losses, finite-gradient checks, optimizer updates, step times,
  and peak allocated training memory. The output contains no generated answers or
  source document text.

A short matched study can screen numerical sensitivity. It does not establish
broad downstream-task quality, long-context equivalence, or native serving speed.

Approximate inverse is a subsequent delivery. Decode recipes are documented in
[the decode guide](../../../docs/linear_attention_decode.md).

## Decode and state-only serving

`configs/kda_decode_replay_int8_hadamard.json` selects value-axis Hadamard rotation
with per-key-channel, per-32-value INT8 scales stored in FP16. It disables replay
factor QDQ to isolate state quantization. The value dimension must be divisible
by 32. Model-quality evaluation remains pending.

`benchmark_decode.py` measures matched-policy training overhead.
`compare_quality.py` compares completed paired receipts with confidence bounds.
See [the decode guide](../../../docs/linear_attention_decode.md) for the numerical
contract, phase context, checkpoint behavior, and current limitations.

Delivery order is decode, prefill operand matmuls, then approximate inverse.
See the
[vLLM fakequant guide](../../../docs/linear_attention_vllm.md) for the separate
state-only serving plugin and its invocation-boundary quantization semantics.
