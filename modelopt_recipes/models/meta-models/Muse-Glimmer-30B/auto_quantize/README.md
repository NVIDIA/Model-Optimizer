# Muse Glimmer AutoQuantize recipes

`w4a16_nvfp4_4o6_mixed` searches the Muse Glimmer
language-model MLP projections, self-attention projections, and `lm_head` over
W4A16 NVFP4 Four-Over-Six, FP8, and BF16 fallback at 5.5 effective bits. The
vision tower and unmatched modules remain BF16.

Use unquantized KV cache and representative text calibration:

```bash
python examples/hf_ptq/hf_ptq.py \
  --pyt_ckpt_path <muse-glimmer-checkpoint> \
  --recipe models/meta-models/Muse-Glimmer-30B/auto_quantize/w4a16_nvfp4_4o6_mixed \
  --kv_cache_qformat none \
  --dataset nemotron-post-training-v3 \
  --calib_size 512 \
  --calib_seq 2048 \
  --export_path <output-path>
```
