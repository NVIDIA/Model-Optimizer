# NVIDIA Nemotron 3.5 Bolt 74B-A9B sub-4-bit AutoQuantize recipe

`iq1_iq2_gptq_nvfp4_fp8_experts_at_2p0625bits` searches the routed and shared experts of
[NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16)
at 2.0625 bits per expert weight, the size of the
[24GB release](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB)'s IQ2_XXS
experts. The candidates are IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, weight-only NVFP4, weight-only FP8 and
BF16. The other linears keep the release's formats and stay out of the budget: Mamba `in_proj` and
`out_proj` at W4A16 NVFP4, and attention, the MoE latent projections and `lm_head` at Q8_0.

- **KL scoring in a low-bit context:** `context_quant_level: 1` scores each candidate with every
  other expert layer at IQ2_XXS, the second-lowest candidate, and the other linears at their
  release formats.
- **Local GPTQ + imatrix** for the IQ candidates, on BF16 inputs, so a layer's score doesn't depend
  on the other layers' picks.
- **Sequential GPTQ + imatrix** to finalize the selected formats.

Run the command once to search, then run it again to finalize and export:

```bash
python examples/hf_ptq/hf_ptq.py \
  --pyt_ckpt_path nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 \
  --trust_remote_code \
  --recipe models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16/auto_quantize/iq1_iq2_gptq_nvfp4_fp8_experts_at_2p0625bits \
  --auto_quantize_checkpoint <search.pth> \
  --export_path <output-path>
```

To score the formats in parallel, run format shards and merge them as described in
[Scoring and finalizing AutoQuantize in separate runs](../../../../../examples/hf_ptq/README.md#scoring-and-finalizing-autoquantize-in-separate-runs).

## Results

Held-out KL of the language model against BF16, over 128 documents of Nemotron post-training data
(62.2k tokens; SE 0.002–0.008). Every row keeps the release's formats outside the experts:

| Expert bits | Uniform IQ2_XXS + sequential GPTQ | Default KL search | This recipe's search | 24GB release (QAD) |
|---|---|---|---|---|
| 2.0625 | 0.103 | 0.146 | **0.076** | 0.071 |
| 2.25 | – | 0.137 | **0.060** | – |
| 2.5 | – | – | **0.045** | – |
| 2.75 | – | – | **0.038** | – |

The other budgets use the same search scores, solved at their expert bits. At the release's size,
the search is within one SE of the quantization-aware-distilled release, without training. The
default KL search (no `context_quant_level`) scores each layer with the others in BF16. It
measures worse than uniform IQ2_XXS and leaves part of the budget unspent (2.04 and 2.22 expert
bits); solving its scores with the LP instead still measured 0.125 and 0.094.
