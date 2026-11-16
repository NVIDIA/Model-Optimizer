# Qwen3.6-35B-A3B sub-4-bit AutoQuantize recipes

These recipes search the routed experts of
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) at 2.75, 3.0 and 3.25 effective bits.
The candidates are IQ2_XXS, IQ2_XS, IQ2_S, weight-only NVFP4, weight-only FP8 and BF16. The budget
covers every linear except embeddings, `lm_head` and vision, and the other linears stay BF16. The
recipes share `iq2_gptq_nvfp4_fp8_experts.search.yaml`:

- **KL scoring in a low-bit context.** Each recipe uses the IQ2 format nearest its budget: `lowest`
  (IQ2_XXS) at 2.75 bits, `1` (IQ2_XS) at 3.0 and `2` (IQ2_S) at 3.25. `lowest` measured the same
  at 3.0 and 3.25 bits (within 1.7%).
- **Local GPTQ + imatrix** for the IQ2 candidates, on BF16 inputs, so a layer's score doesn't depend
  on the other layers' picks.
- **Sequential GPTQ + imatrix** to finalize the selected formats.

Run the command once to search, then run it again to finalize and export:

```bash
python examples/hf_ptq/hf_ptq.py \
  --pyt_ckpt_path Qwen/Qwen3.6-35B-A3B \
  --recipe models/Qwen/Qwen3.6-35B-A3B/auto_quantize/iq2_gptq_nvfp4_fp8_experts_at_3p0bits \
  --auto_quantize_checkpoint <search.pth> \
  --export_path <output-path>
```

To score the six formats in parallel, run format shards and merge them as described in
[Scoring and finalizing AutoQuantize in separate runs](../../../../../examples/hf_ptq/README.md#scoring-and-finalizing-autoquantize-in-separate-runs).
A search finalizes only with the recipe that scored it, since each budget scores in its own context.

## Results

Held-out KL against BF16, over 128 documents (63.5k tokens; SE ≈ 0.007–0.010). The AutoQuant
column was measured with `context: lowest` at every budget:

| Recipe | Bits | AutoQuant | Uniform IQ2 + sequential GPTQ |
|---|---|---|---|
| `iq2_gptq_nvfp4_fp8_experts_at_2p75bits` | 2.75 | 0.100 | ≈0.106 (interpolated) |
| `iq2_gptq_nvfp4_fp8_experts_at_3p0bits` | 2.99 | 0.077 | ≈0.078 (interpolated) |
| `iq2_gptq_nvfp4_fp8_experts_at_3p25bits` | 3.25 | 0.063 | 0.064 (IQ2_S, 3.13 bits) |

The search beats the uniform curve by 2–5% at 2.75 and 3.0 bits, and matches uniform IQ2_S at 3.25.
The default gradient search, run with W4A4/W8A8 NVFP4/FP8 candidates, measured 0.105, 0.094 and
0.093. The same candidates with context-scored KL measured 0.101, 0.076 and 0.065.
