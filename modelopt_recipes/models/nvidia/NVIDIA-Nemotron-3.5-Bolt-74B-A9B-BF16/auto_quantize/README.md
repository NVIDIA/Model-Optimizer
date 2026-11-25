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

Score the six quantized expert formats as six format shards, one per GPU, then merge them: the
merge run solves, finalizes and exports. This is the workflow the recipe was tested with end to
end, shown with that test's calibration size, 16 samples of 512 tokens; raise `--calib_size` for a
real run, and keep one explicit `--batch_size` for every shard
([format shards](../../../../../examples/hf_ptq/README.md#scoring-and-finalizing-autoquantize-in-separate-runs)):

```bash
RECIPE=models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16/auto_quantize/iq1_iq2_gptq_nvfp4_fp8_experts_at_2p0625bits
ARGS="--pyt_ckpt_path nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 --trust_remote_code --recipe $RECIPE \
  --calib_size 16 --calib_seq 512 --batch_size 8"
for i in 0 1 2 3 4 5; do  # one shard per GPU; with fewer GPUs, run the rest after these
  CUDA_VISIBLE_DEVICES=$i python examples/hf_ptq/hf_ptq.py $ARGS \
    --auto_quantize_checkpoint bolt-shard$i.pth --auto_quantize_format_shard $i/6 &
done
wait
python examples/hf_ptq/hf_ptq.py $ARGS \
  --auto_quantize_merge_shards bolt-shard0.pth,bolt-shard1.pth,bolt-shard2.pth,bolt-shard3.pth,bolt-shard4.pth,bolt-shard5.pth \
  --auto_quantize_checkpoint bolt-search.pth --export_path bolt-out
```

On GB200s, the shards took under two hours and the merge run about half an hour. The same search
also runs as one process, run once to search and again to finalize and export, but it scores the
six formats one after another: with the calibration above it needed about five hours.

```bash
python examples/hf_ptq/hf_ptq.py $ARGS --auto_quantize_checkpoint bolt-search.pth --export_path bolt-out
```

The recipe sets no `kv_cache`, so the finalizing run also quantizes the attention layers' KV cache
with `hf_ptq.py`'s `--kv_cache_qformat`, FP8 (`fp8_cast`) by default; the search scores without
it. Pass `--kv_cache_qformat none` to keep it BF16, as in the results below.

The language model loads only through the checkpoint's own code, hence `--trust_remote_code`,
and needs transformers 5.9 or later, which scopes a sub-model's weight renames to that sub-model;
tested with 5.18. `hf_ptq.py` holds this model on one GPU: on a GB200, one format shard of the
search peaked at about 180 GB and the finalize at 195 GB.

## Results

Held-out KL of the language model against BF16, over 128 documents of Nemotron post-training data
(62.2k tokens; SE 0.002–0.008). Every row keeps the release's formats outside the experts.

These numbers come from a script, not from the commands above. The script loads the checkpoint with
its own key renaming and calls `mtq.auto_quantize` with this recipe's search and finalize settings,
on 512 calibration documents of up to 512 tokens (128 scored). One setting differs from the recipe:
the script's scoring context held Mamba `conv1d` at FP8, while the recipe keeps it BF16. Both
finalizes keep it BF16.

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
