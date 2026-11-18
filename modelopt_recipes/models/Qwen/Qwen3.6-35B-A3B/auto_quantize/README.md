# Qwen3.6-35B-A3B sub-4-bit AutoQuantize recipe

`iq2_gptq_nvfp4_fp8_experts_at_2p75bits` searches the routed experts of
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) at 2.75 effective bits. The
candidates are IQ2_XXS, IQ2_XS, IQ2_S, weight-only NVFP4, weight-only FP8 and BF16. The budget
covers every linear except embeddings, `lm_head` and vision, and the other linears stay BF16.

- **KL scoring in a low-bit context:** `context_quant_level: lowest` scores each candidate with
  every other expert layer at IQ2_XXS.
- **Local GPTQ + imatrix** for the IQ2 candidates, on BF16 inputs, so a layer's score doesn't depend
  on the other layers' picks.
- **Sequential GPTQ + imatrix** to finalize the selected formats.

Run the command once to search, then run it again to finalize and export:

```bash
python examples/hf_ptq/hf_ptq.py \
  --pyt_ckpt_path Qwen/Qwen3.6-35B-A3B \
  --recipe models/Qwen/Qwen3.6-35B-A3B/auto_quantize/iq2_gptq_nvfp4_fp8_experts_at_2p75bits \
  --auto_quantize_checkpoint <search.pth> \
  --export_path <output-path>
```

To score the five quantized formats in parallel, run up to five format shards and merge them as
described in
[Scoring and finalizing AutoQuantize in separate runs](../../../../../examples/hf_ptq/README.md#scoring-and-finalizing-autoquantize-in-separate-runs).

## Results

Held-out KL against BF16 at 2.75 bits, over 128 documents (63.5k tokens; SE ≈ 0.01):

| Search | KL |
|---|---|
| This recipe | **0.100** |
| Uniform IQ2 + sequential GPTQ (interpolated between IQ2_XXS and IQ2_XS) | ≈0.106 |
| Default gradient search, W4A4/W8A8 NVFP4/FP8 candidates | 0.105 |
| Context-scored KL, W4A4/W8A8 NVFP4/FP8 candidates | 0.101 |
