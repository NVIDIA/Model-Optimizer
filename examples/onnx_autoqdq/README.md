# QDQ Placement Optimization Example

This example demonstrates automated Q/DQ (Quantize/Dequantize) node placement optimization for ONNX models using TensorRT performance measurements.

## Table of Contents

- [Prerequisites](#prerequisites)
  - [Get the Model](#get-the-model)
  - [Set Fixed Batch Size](#set-fixed-batch-size)
  - [What's in This Directory](#whats-in-this-directory)
- [Quick Start](#quick-start)
  - [Basic Usage](#basic-usage)
  - [FP8 Quantization](#fp8-quantization)
  - [Faster Exploration](#faster-exploration)
- [Output Structure](#output-structure)
- [Region Inspection](#region-inspection)
- [Using the Optimized Model](#using-the-optimized-model)
- [Pattern Cache](#pattern-cache)
- [Optimize from Existing QDQ Model](#optimize-from-existing-qdq-model)
- [Remote Autotuning with TensorRT](#remote-autotuning-with-tensorrt)
- [Programmatic API Usage](#programmatic-api-usage)
- [Documentation](#documentation)

## Prerequisites

### Get the Model

Download the ResNet50 model from the ONNX Model Zoo:

```bash
# Download ResNet50 from ONNX Model Zoo
curl -L -o resnet50_Opset17.onnx https://github.com/onnx/models/raw/main/Computer_Vision/resnet50_Opset17_torch_hub/resnet50_Opset17.onnx
```

### Set Fixed Batch Size

The downloaded model has a dynamic batch size. For best performance with TensorRT benchmarking, set a fixed batch size:

```bash
# Set batch size to 128 using the provided script
python3 set_batch_size.py resnet50_Opset17.onnx --batch-size 128 --output resnet50.bs128.onnx

# Or for other batch sizes
python3 set_batch_size.py resnet50_Opset17.onnx --batch-size 1 --output resnet50.bs1.onnx
```

This creates `resnet50.bs128.onnx` with a fixed batch size of 128, which is optimal for TensorRT performance benchmarking.

**Note:** The script requires the `onnx` package.

### What's in This Directory

- `set_batch_size.py` - Script to convert dynamic batch size models to fixed batch size
- `README.md` - This guide

**Note:** ONNX model files are not included in the repository (excluded via `.gitignore`). Download and prepare them using the instructions above.

## Quick Start

### Basic Usage

Optimize the ResNet50 model with INT8 quantization:

```bash
# Using the fixed batch size model
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50.bs128.onnx \
    --output_dir ./resnet50_results \
    --quant_type int8 \
    --schemes_per_region 30

# Or use the original dynamic batch size model, batch is set to 1 in benchmark
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50_Opset17.onnx \
    --output_dir ./resnet50_results \
    --quant_type int8 \
    --schemes_per_region 30
```

Short options: `-m` for `--onnx_path`, `-o` for `--output_dir`, `-s` for `--schemes_per_region`. Default output directory is `./autotuner_output` if `--output_dir` is omitted.

This will:

1. Automatically discover optimization regions in the model
2. Test 30 different Q/DQ placement schemes per region pattern
3. Measure TensorRT performance for each scheme
4. Export the best optimized model to `./resnet50_results/optimized_final.onnx`

### FP8 Quantization

For FP8 quantization:

```bash
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50.bs128.onnx \
    --output_dir ./resnet50_fp8_results \
    --quant_type fp8 \
    --schemes_per_region 50
```

### Faster Exploration

For quick experiments, reduce the number of schemes:

```bash
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50.bs128.onnx \
    --output_dir ./resnet50_quick \
    --schemes_per_region 15
```

## Output Structure

After running, the output workspace will be:

```log
resnet50_results/
├── optimized_final.onnx              # Optimized model
├── baseline.onnx                     # Baseline for comparison
├── autotuner_state.yaml              # Resume checkpoint
├── autotuner_state_pattern_cache.yaml # Reusable pattern cache
├── logs/
│   ├── baseline.log                  # TensorRT baseline log
│   ├── region_*_scheme_*.log         # Per-scheme logs
│   └── final.log                     # Final model log
└── region_models/                    # Best model per region (intermediate)
    └── region_*_level_*.onnx
```

## Region Inspection

To debug how the autotuner discovers and partitions regions in your model, use the `region_inspect` tool. It runs the same region search as the autotuner and prints the region hierarchy, node counts, and summary statistics (without running benchmarks).

```bash
# Basic inspection (regions with quantizable ops only)
python3 -m modelopt.onnx.quantization.autotune.region_inspect --model resnet50.bs128.onnx

# Verbose mode for detailed debug logging
python3 -m modelopt.onnx.quantization.autotune.region_inspect --model resnet50.bs128.onnx --verbose

# Custom maximum sequence region size
python3 -m modelopt.onnx.quantization.autotune.region_inspect --model resnet50.bs128.onnx --max-sequence-size 20

# Include all regions (including those without Conv/MatMul etc.)
python3 -m modelopt.onnx.quantization.autotune.region_inspect --model resnet50.bs128.onnx --include-all-regions
```

Short option: `-m` for `--model`, `-v` for `--verbose`. Use this to verify region boundaries and counts before or during autotuning.

## Using the Optimized Model

Deploy with TensorRT:

```bash
trtexec --onnx=resnet50_results/optimized_final.onnx \
        --saveEngine=resnet50.engine \
        --stronglyTyped
```

## Pattern Cache

Reuse learned patterns on similar models (warm-start):

```bash
# First optimization on ResNet50
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50.bs128.onnx \
    --output_dir ./resnet50_run

# Download and prepare ResNet101 (or any similar model)
curl -L -o resnet101_Opset17.onnx https://github.com/onnx/models/blob/main/Computer_Vision/resnet101_Opset17_torch_hub/resnet101_Opset17.onnx
python3 set_batch_size.py resnet101_Opset17.onnx --batch-size 128 --output resnet101.bs128.onnx

# Reuse patterns from ResNet50 on ResNet101 
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet101.bs128.onnx \
    --output_dir ./resnet101_run \
    --pattern_cache ./resnet50_run/autotuner_state_pattern_cache.yaml
```

## Optimize from Existing QDQ Model

If the user already have a quantized model, he can use it as a starting point to potentially find even better Q/DQ placements:

```bash
# Use an existing QDQ model as baseline (imports quantization patterns)
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50.bs128.onnx \
    --output_dir ./resnet50_improved \
    --qdq_baseline resnet50_quantized.onnx \
    --schemes_per_region 40
```

This will:

1. Extract Q/DQ insertion points from the baseline model
2. Import them into the pattern cache as seed schemes
3. Generate and test variations to find better placements
4. Compare against the baseline performance

**Use cases:**

- **Improve existing quantization**: Fine-tune manually quantized models
- **Compare tools**: Test if autotuner can beat other quantization methods
- **Bootstrap optimization**: Start from expert-tuned schemes

**Example workflow:**

```bash
# Step 1: Create initial quantized model with modelopt 
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
# The autotuner will try to find better Q/DQ placements than the initial quantization
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50.bs128.onnx \
    --output_dir ./resnet50_autotuned \
    --qdq_baseline resnet50_quantized.onnx \
    --schemes_per_region 50
```

**Note:** This example uses dummy calibration data. For production use, provide real calibration data representative of the inference workload.

## Remote Autotuning with TensorRT

TensorRT 10.16+ supports remote autotuning, which allows TensorRT's optimization process to be offloaded to a remote hardware. This is useful when optimizing models for different target GPUs without having direct access to them.

To use remote autotuning during Q/DQ placement optimization, run with `trtexec` and pass extra args:

```bash
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50.bs128.onnx \
    --output_dir ./resnet50_remote_autotuned \
    --schemes_per_region 50 \
    --use_trtexec \
    --trtexec_benchmark_args "--remoteAutoTuningConfig=\"<remote autotuning config>\""
```

**Requirements:**

- TensorRT 10.16 or later
- Valid remote autotuning configuration
- `--use_trtexec` must be set (benchmarking uses `trtexec` instead of the TensorRT Python API)

Replace `<remote autotuning config>` with user's actual remote autotuning configuration string. Other TensorRT benchmark options (e.g. `--timing_cache`, `--warmup_runs`, `--timing_runs`, `--plugin_libraries`) are also available; run `--help` for details.

## Programmatic API Usage

All examples above use the command-line interface. For **low-level programmatic control** in Python code, use the Python API directly. This allows user to:

- Integrate autotuning into custom pipelines
- Implement custom evaluation functions
- Control state management and checkpointing
- Build custom optimization workflows

**See the API Reference documentation for low-level usage:**

- [`docs/source/reference/2_qdq_placement.rst`](../../docs/source/reference/2_qdq_placement.rst)

The API docs include detailed examples of:

- Using the `QDQAutotuner` class and `region_pattern_autotuning_workflow`
- Customizing region discovery and scheme generation
- Managing optimization state and pattern cache programmatically
- Implementing custom performance evaluators (e.g. via `init_benchmark_instance` and `benchmark_onnx_model`)

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

For command-line help and all options (e.g. `--state_file`, `--node_filter_list`, `--default_dq_dtype`, `--verbose`):

```bash
python3 -m modelopt.onnx.quantization.autotune --help
```
