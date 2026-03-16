# QDQ Placement Optimization Example

This example demonstrates automated Q/DQ (Quantize/Dequantize) node placement optimization for ONNX models using TensorRT performance measurements.

## Branch: Subgraph Mode (This Branch)

On branch **dev-wahao-autotune-subgraph-profile**, the autotuner adds a second workflow and related features not present on **dev-willg-add-auto-qdq-placement-tool**:

| Feature | Description |
|--------|-------------|
| **Subgraph mode** | `--mode subgraph`: fusion-aware grouping from TensorRT `graph.json`; profiles isolated subgraphs instead of full-model region patterns (~30 min vs ~25 h for large models). |
| **Fusion grouping** | Uses TRT layer/fusion boundaries to form quantizable groups and infers shapes for extracted subgraphs. |
| **Per-layer comparison** | When trtexec supports it, compares compute-layer timing (e.g. MatMul) instead of whole-subgraph latency to reduce reformat noise. Falls back to total latency if profiling flags are unsupported. |
| **Incremental validation** | Optional per-group full-model check: apply QDQ groups one-by-one and keep only those that improve latency; saves both a "raw" (all qualifying QDQ) and a validated final model. |
| **Subgraph cache/resume** | Phase 2 (subgraph profiling) and Phase 3 (incremental validation) write progress to `autotune_cache.json` so runs can resume after interruption. |
| **trtexec benchmarking** | `--use-trtexec` plus `--trtexec-args` for dynamic shapes and custom options (e.g. `--optShapes`, `--useCudaGraph`). |

The examples below include both the original **region** mode and the **subgraph** mode with recommended options.

## Prerequisites

### Get the Model

Download the ResNet50 model from the ONNX Model Zoo:

```bash
# Download ResNet50 from ONNX Model Zoo
curl -L -o resnet50_Opset17.onnx https://github.com/onnx/models/raw/main/Computer_Vision/resnet50_Opset17_torch_hub/resnet50_Opset17.onnx
```

### Set Fixed Batch Size (Recommended)

The downloaded model has a dynamic batch size. For best performance with TensorRT benchmarking, set a fixed batch size:

```bash
# Set batch size to 128 using the provided script
python3 set_batch_size.py resnet50_Opset17.onnx --batch-size 128 --output resnet50.bs128.onnx

# Or for other batch sizes
python3 set_batch_size.py resnet50_Opset17.onnx --batch-size 1 --output resnet50.bs1.onnx
```

This creates `resnet50.bs128.onnx` with a fixed batch size of 128, which is optimal for TensorRT performance benchmarking.

**Note:** The script requires the `onnx` package. If you have modelopt installed, this dependency should already be available.

### What's in This Directory

- `set_batch_size.py` - Script to convert dynamic batch size models to fixed batch size
- `README.md` - This guide

**Note:** ONNX model files are not included in the repository (excluded via `.gitignore`). Download and prepare them using the instructions above.

## Quick Start

### Basic Usage

Optimize the ResNet50 model with INT8 quantization:

```bash
# Using the fixed batch size model (recommended)
python3 -m modelopt.onnx.quantization.autotune \
    --model resnet50.bs128.onnx \
    --output ./resnet50_results \
    --quant-type int8 \
    --schemes-per-region 30

# Or use the original dynamic batch size model
python3 -m modelopt.onnx.quantization.autotune \
    --model resnet50_Opset17.onnx \
    --output ./resnet50_results \
    --quant-type int8 \
    --schemes-per-region 30
```

This will:
1. Automatically discover optimization regions in your model
2. Test 30 different Q/DQ placement schemes per region pattern
3. Measure TensorRT performance for each scheme
4. Export the best optimized model to `./resnet50_results/optimized_final.onnx`

### FP8 Quantization

For FP8 quantization (faster on modern GPUs):

```bash
python3 -m modelopt.onnx.quantization.autotune \
    --model resnet50.bs128.onnx \
    --output ./resnet50_fp8_results \
    --quant-type fp8 \
    --schemes-per-region 50
```

### Faster Exploration

For quick experiments, reduce the number of schemes:

```bash
python3 -m modelopt.onnx.quantization.autotune \
    --model resnet50.bs128.onnx \
    --output ./resnet50_quick \
    --schemes-per-region 15
```

### Subgraph Mode (Recommended for Large or Dynamic-Shape Models)

Subgraph mode is fusion-aware and much faster than region mode on large models. Use **trtexec** for benchmarking and pass dynamic input shapes via `--trtexec-args` when needed.

**Best-practice example** (mirrors production usage with dynamic shapes and incremental validation):

```bash
# Set output dir (e.g. <model_prefix>_autotune_subgraph) and optional graph.json
MODEL="path/to/your_model.fp16.opt.onnx"
GRAPH_JSON="path/to/your_model.fp16.opt.graph.json"   # optional; omit to auto-generate
OUTPUT_DIR="${MODEL%.onnx}_autotune_subgraph"
mkdir -p "$OUTPUT_DIR"

# Optional: optShapes for dynamic inputs (comma-separated name:shape)
OPT_SHAPES='--optShapes=agents_feature:21x33x10x16,agents_mask:21x33x10'

python3 -m modelopt.onnx.quantization.autotune \
    --model "$MODEL" \
    --output "$OUTPUT_DIR" \
    --mode subgraph \
    --graph-json "$GRAPH_JSON" \
    --quant-type fp8 \
    --use-trtexec \
    --warmup-runs 20 \
    --timing-runs 100 \
    --incremental-validation \
    --trtexec-args "--stronglyTyped --noDataTransfers --useCudaGraph --maxAuxStreams=0 $OPT_SHAPES" \
    --schemes-per-region 100

# Logs are written to stdout; redirect to a file if desired:
#   ... > "${OUTPUT_DIR}/$(basename ${MODEL%.onnx}).log.txt" 2>&1
```

**What this does:**

- **`--mode subgraph`**: Use fusion-aware subgraph grouping and per-subgraph profiling.
- **`--graph-json`**: Supply a pre-dumped TensorRT graph (e.g. from a prior FP16 build). If omitted, one is generated.
- **`--use-trtexec`**: Benchmark with trtexec so you can pass `--trtexec-args` (e.g. `--optShapes` for dynamic inputs).
- **`--incremental-validation`** (default: on): After selecting schemes per group, validate on the full model group-by-group; save `optimized_raw.onnx` (all qualifying QDQ) and `optimized_final.onnx` (incrementally validated). Use `--no-incremental-validation` to skip and use the raw model as final.
- **`--trtexec-args`**: Extra flags for trtexec (stronglyTyped, shapes, CUDA graph, etc.). Required for dynamic-shape models.

**Resume:** If the run is interrupted, run the same command again; progress is read from `output_dir/autotune_cache.json` and Phase 2/3 resume where they left off.

## Output Structure

**Region mode** (default):

```text
resnet50_results/
├── optimized_final.onnx              # Optimized model
├── baseline.onnx                     # Baseline for comparison
├── autotuner_state.yaml              # Resume checkpoint
├── autotuner_state_pattern_cache.yaml # Reusable patterns
└── logs/
    ├── baseline.log
    ├── region_*_scheme_*.log
    └── final.log
```

**Subgraph mode** (`--mode subgraph`):

```text
<output_dir>/
├── optimized_final.onnx              # Incrementally validated model (if --incremental-validation)
├── optimized_raw.onnx                # All qualifying QDQ applied (always saved)
├── autotune_cache.json               # Resume cache for Phase 2 & 3
├── subgraphs/                        # Extracted subgraph ONNX files
└── logs/
    ├── group_*_scheme_*.log          # trtexec log per group/scheme
    └── group_*_scheme_*.profile.json # Per-layer profile (when supported)
```

## Using the Optimized Model

Deploy with TensorRT. In **subgraph mode**, prefer `optimized_final.onnx` (incrementally validated); use `optimized_raw.onnx` if you disabled incremental validation or want the full QDQ set.

```bash
trtexec --onnx=resnet50_results/optimized_final.onnx \
        --saveEngine=resnet50.engine \
        --stronglyTyped
```

## Pattern Cache (Transfer Learning)

Reuse learned patterns on similar models:

```bash
# First optimization on ResNet50
python3 -m modelopt.onnx.quantization.autotune \
    --model resnet50.bs128.onnx \
    --output ./resnet50_run

# Download and prepare ResNet101 (or any similar model)
curl -L -o resnet101_Opset17.onnx https://github.com/onnx/models/raw/main/Computer_Vision/resnet101-v2-7.onnx
python3 set_batch_size.py resnet101_Opset17.onnx --batch-size 128 --output resnet101.bs128.onnx

# Reuse patterns from ResNet50 on ResNet101 (much faster!)
python3 -m modelopt.onnx.quantization.autotune \
    --model resnet101.bs128.onnx \
    --output ./resnet101_run \
    --pattern-cache ./resnet50_run/autotuner_state_pattern_cache.yaml
```

## Optimize from Existing QDQ Model

If you already have a quantized model (e.g., from manual quantization or another tool), you can use it as a starting point to potentially find even better Q/DQ placements:

```bash
# Use an existing QDQ model as baseline
python3 -m modelopt.onnx.quantization.autotune \
    --model resnet50.bs128.onnx \
    --output ./resnet50_improved \
    --qdq-baseline resnet50_quantized.onnx \
    --schemes-per-region 40
```

This will:
1. Extract Q/DQ insertion points from the baseline model
2. Use them as seed schemes during optimization
3. Generate and test variations to find better placements
4. Compare against the baseline performance

**Use cases:**
- **Improve existing quantization**: Fine-tune manually quantized models
- **Compare tools**: Test if autotuner can beat other quantization methods
- **Bootstrap optimization**: Start from expert-tuned schemes

**Example workflow:**

```bash
# Step 1: Create initial quantized model with any quantization tool
# For example, using modelopt's quantize function:
python3 -c "
import numpy as np
from modelopt.onnx.quantization import quantize

# Create dummy calibration data (replace with real data for production)
dummy_input = np.random.randn(128, 3, 224, 224).astype(np.float32)
quantize(
    'resnet50.bs128.onnx',
    calibration_data=dummy_input,
    calibration_method='entropy',
    output_path='resnet50_quantized.onnx'
)
"

# Step 2: Use the quantized baseline for autotuning
python3 -m modelopt.onnx.quantization.autotune \
    --model resnet50.bs128.onnx \
    --output ./resnet50_autotuned \
    --qdq-baseline resnet50_quantized.onnx \
    --schemes-per-region 50

# The autotuner will try to find better Q/DQ placements than the initial quantization
```

**Note:** This example uses dummy calibration data. For production use, provide real calibration data representative of your inference workload.

## Programmatic API Usage

All examples above use the command-line interface. For **low-level programmatic control** in your Python code, use the Python API directly. This allows you to:
- Integrate autotuning into custom pipelines
- Implement custom evaluation functions
- Control state management and checkpointing
- Build custom optimization workflows

**See the API Reference documentation for low-level usage:**
- [`docs/source/reference/2_qdq_placement.rst`](../../docs/source/reference/2_qdq_placement.rst)

The API docs include detailed examples of:
- Using the `Autotuner` class directly
- Customizing region discovery and scheme generation
- Managing optimization state programmatically
- Implementing custom performance evaluators

## Documentation

For comprehensive documentation on QDQ placement optimization, see:

- **User Guide**: [`docs/source/guides/9_qdq_placement.rst`](../../docs/source/guides/9_qdq_placement.rst)
  - Detailed explanations of how the autotuner works
  - Advanced usage patterns and best practices
  - Configuration options and performance tuning
  - Troubleshooting common issues

- **API Reference**: [`docs/source/reference/2_qdq_placement.rst`](../../docs/source/reference/2_qdq_placement.rst)
  - Complete API documentation for all classes and functions
  - Low-level usage examples
  - State management and pattern cache details

For command-line help:

```bash
python3 -m modelopt.onnx.quantization.autotune --help
```
