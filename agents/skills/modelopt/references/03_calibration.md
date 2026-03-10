# Calibration Reference

## What Calibration Does

Calibration runs representative data through the quantized model to collect activation statistics (min, max, histogram). These statistics are used to compute optimal quantization scales that minimize information loss.

Without calibration, quantization uses arbitrary ranges and produces poor accuracy.

## Calibration Flow

```
1. Insert Q/DQ (quantize/dequantize) nodes into the model
2. Set model to calibration mode
3. Run forward passes with calibration data (forward_loop)
4. Each quantizer collects activation statistics
5. Compute optimal scales using chosen algorithm
6. Freeze scales, switch to inference mode
```

## Calibration Algorithms

| Algorithm | Flag | Speed | Accuracy | When to Use |
|-----------|------|-------|----------|-------------|
| `max` | `--calib_algo max` | Fastest | Good | Default, works well for most models |
| `mse` | `--calib_algo mse` | Slower | Better | When max gives poor accuracy |
| `percentile` | `--calib_algo percentile` | Medium | Good | Outlier-heavy activations |
| `histogram` | `--calib_algo histogram` | Slowest | Best | When other methods fail |

Default in hf_ptq.py: depends on the format config (FP8 uses `max`, AWQ uses its own algorithm).

## Calibration Size (`--calib_size`)

Number of calibration samples to run through the model.

| Size | Time | Accuracy | Use Case |
|------|------|----------|----------|
| 64 | Very fast | Lower | Quick smoke test |
| 128 | Fast | Acceptable | Debugging, iteration |
| 512 | Default | Good | Production default |
| 1024 | Slow | Best | Final production run |
| 2048+ | Very slow | Marginal gain | Diminishing returns |

**Rule of thumb:** Use 128 for quick tests, 512 for real runs, 1024 if accuracy matters.

## Default Dataset

hf_ptq.py uses CNN/DailyMail by default (loaded via `example_utils.get_calib_dataloader()`). The dataset is automatically tokenized to match the model's tokenizer.

## Impact on Accuracy

- Larger `calib_size` → better scale estimation → better accuracy
- Diminishing returns past ~1024 samples
- Dataset quality matters more than quantity — the calibration data should be representative of real inference workloads
- If accuracy is poor after quantization, try increasing `calib_size` before changing format

## Memory Considerations

- Calibration runs the full model with calibration data on GPU
- Memory usage ≈ model size + batch of calibration data + quantizer state
- If OOM during calibration: reduce `calib_size` or use gradient checkpointing
- Calibration is done once; the resulting scales are saved in the checkpoint
