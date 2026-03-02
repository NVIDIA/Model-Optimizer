===============================================
Automated Q/DQ Placement Optimization
===============================================

Overview
========

The ``modelopt.onnx.quantization.autotune`` module provides automated optimization of Quantize/Dequantize (Q/DQ) node placement in ONNX models. Instead of manually deciding where to insert Q/DQ nodes, the autotuner systematically explores different placement strategies and uses TensorRT performance measurements to find the optimal configuration that minimizes inference latency.

**Key Features:**

* **Automatic Region Discovery**: Intelligently partitions your model into optimization regions
* **Pattern-Based Optimization**: Groups structurally similar regions and optimizes them together
* **TensorRT Performance Measurement**: Uses actual inference latency (not theoretical estimates)
* **Crash Recovery**: Checkpoint/resume capability for long-running optimizations
* **Warm-Start Support**: Reuses learned patterns from previous runs
* **Multiple Quantization Types**: Supports INT8 and FP8 quantization

**When to Use This Tool:**

* You have an ONNX model you want to quantize for TensorRT deployment
* You want to optimize Q/DQ placement for best performance (not just accuracy)
* Your model has repeating structures (e.g., transformer blocks, ResNet layers)
* You need automated optimization without manual Q/DQ placement

Quick Start
===========

Command-Line Interface
-----------------------

The easiest way to use the autotuner is via the command-line interface:

.. code-block:: bash

   # Basic usage - INT8 quantization (output default: ./autotuner_output)
   python -m modelopt.onnx.quantization.autotune --onnx_path model.onnx

   # Specify output dir and FP8 with more schemes
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --output_dir ./results \
       --quant_type fp8 \
       --schemes_per_region 50

The command will:

1. Discover regions in your model automatically
2. Measure baseline performance (no quantization)
3. Test different Q/DQ placement schemes for each region pattern
4. Select the best scheme based on TensorRT latency measurements
5. Export an optimized ONNX model with Q/DQ nodes

**Output Files:**

.. code-block:: text

   results/
   ├── autotuner_state.yaml                  # Checkpoint for resuming
   ├── autotuner_state_pattern_cache.yaml    # Pattern cache for future runs
   ├── baseline.onnx                         # Unquantized baseline
   ├── optimized_final.onnx                  # Final optimized model
   ├── logs/                                 # TensorRT build logs
   │   ├── baseline.log
   │   ├── region_*_scheme_*.log
   │   └── final.log
   └── region_models/                        # Best model per region
       └── region_*_level_*.onnx

Python API
----------

For programmatic control, use the workflow function:

.. code-block:: python

   from pathlib import Path
   from modelopt.onnx.quantization.autotune.workflows import (
       region_pattern_autotuning_workflow,
       init_benchmark_instance
   )

   # When using the workflow from Python, the CLI normally calls init_benchmark_instance
   # for you. If you run the workflow directly, call it first:
   init_benchmark_instance(
       use_trtexec=False,
       timing_cache_file="timing.cache",
       warmup_runs=5,
       timing_runs=20,
   )

   # Run autotuning workflow
   autotuner = region_pattern_autotuning_workflow(
       model_path="model.onnx",
       output_dir=Path("./results"),
       num_schemes_per_region=30,
       quant_type="int8",
   )

How It Works
============

The autotuner uses a pattern-based approach that makes optimization both efficient and consistent:

1. **Region Discovery Phase**
   
   The model's computation graph is automatically partitioned into hierarchical regions. Each region is a subgraph containing related operations (e.g., a Conv-BatchNorm-ReLU block).

2. **Pattern Identification Phase**
   
   Regions with identical structural patterns are grouped together. For example, all Convolution->BatchNormalization->ReLU blocks in your model will share the same pattern.

3. **Scheme Generation Phase**
   
   For each unique pattern, multiple Q/DQ insertion schemes are generated. Each scheme specifies different locations to insert Q/DQ nodes.

4. **Performance Measurement Phase**
   
   Each scheme is evaluated by:
   
   * Exporting the ONNX model with Q/DQ nodes applied
   * Building a TensorRT engine
   * Measuring actual inference latency
   
5. **Best Scheme Selection**
   
   The scheme with the lowest latency is selected for each pattern. This scheme automatically applies to all regions matching that pattern.

6. **Model Export**
   
   The final model includes the best Q/DQ scheme for each pattern, resulting in an optimized quantized model.

**Why Pattern-Based?**

Pattern-based optimization significantly reduces the search space. Instead of optimizing each region independently (which could require thousands of benchmarks), the autotuner optimizes each unique pattern once. The time reduction depends on pattern overlap—models with many regions sharing few patterns (like transformers with repeated blocks) see the greatest speedup, while models with mostly unique patterns see less benefit.

Advanced Usage
==============

Warm-Start with Pattern Cache
------------------------------

Pattern cache files store the best Q/DQ schemes from previous optimization runs. You can reuse these patterns on similar models or model versions:

.. code-block:: bash

   # First optimization (cold start)
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model_v1.onnx \
       --output_dir ./run1

   # The pattern cache is saved to ./run1/autotuner_state_pattern_cache.yaml

   # Second optimization with warm-start
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model_v2.onnx \
       --output_dir ./run2 \
       --pattern_cache ./run1/autotuner_state_pattern_cache.yaml

By prioritizing cached schemes, the second test run has the potential to discover optimal configurations much more quickly.

**When to use pattern cache:**

* You're optimizing multiple versions of the same model
* You're optimizing models from the same family (e.g., different BERT variants)
* You want to transfer learned patterns across models

Import Patterns from Existing QDQ Models
-----------------------------------------

If you have a pre-quantized baseline model (e.g., from manual optimization or another tool), you can import its Q/DQ patterns:

.. code-block:: bash

   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --output_dir ./results \
       --qdq_baseline manually_quantized.onnx

The autotuner will:

1. Extract Q/DQ insertion points from the baseline model
2. Map these points to region patterns
3. Use them as seed schemes during optimization

This is useful for:

* Starting from expert-tuned quantization schemes
* Comparing against reference implementations
* Fine-tuning existing quantized models

Resume After Interruption
--------------------------

Long optimizations can be interrupted (Ctrl+C, cluster preemption, crashes) and automatically resumed:

.. code-block:: bash

   # Start optimization
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --output_dir ./results
   
   # ... interrupted after 2 hours ...
   
   # Resume from checkpoint (just run the same command)
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --output_dir ./results

The autotuner automatically:

* Detects the state file (``autotuner_state.yaml``)
* Loads all previous measurements and best schemes
* Continues from the next unprofiled region

Custom TensorRT Plugins
-----------------------

If your model uses custom TensorRT operations, provide the plugin libraries:

.. code-block:: bash

   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --output_dir ./results \
       --plugin_libraries /path/to/plugin1.so /path/to/plugin2.so

Low-Level API Usage
===================

For maximum control, use the autotuner classes directly:

Basic Workflow
--------------

.. code-block:: python

   import onnx
   from modelopt.onnx.quantization.autotune import QDQAutotuner, Config
   from modelopt.onnx.quantization.autotune.workflows import (
       init_benchmark_instance,
       benchmark_onnx_model,
   )

   # Initialize global benchmark (required before benchmark_onnx_model)
   init_benchmark_instance(
       use_trtexec=False,
       timing_cache_file="timing.cache",
       warmup_runs=5,
       timing_runs=20,
   )

   # Load model
   model = onnx.load("model.onnx")

   # Initialize autotuner with automatic region discovery
   autotuner = QDQAutotuner(model)
   config = Config(default_quant_type="int8", verbose=True)
   autotuner.initialize(config)

   # Measure baseline (no Q/DQ)
   autotuner.export_onnx("baseline.onnx", insert_qdq=False)
   baseline_latency = benchmark_onnx_model("baseline.onnx")
   autotuner.submit(baseline_latency)
   print(f"Baseline: {baseline_latency:.2f} ms")

   # Profile each region
   regions = autotuner.regions
   print(f"Found {len(regions)} regions to optimize")

   for region_idx, region in enumerate(regions):
       print(f"\nRegion {region_idx + 1}/{len(regions)}")
       
       # Set current profile region
       autotuner.set_profile_region(region, commit=(region_idx > 0))
       
       # After set_profile_region(), None means this region's pattern was already
       # profiled (e.g. from a loaded state file). There are no new schemes to
       # generate, so skip to the next region.
       if autotuner.current_profile_pattern_schemes is None:
           print("  Already profiled, skipping")
           continue
       
       # Generate and test schemes
       for scheme_num in range(30):  # Test 30 schemes per region
           scheme_idx = autotuner.generate()
           
           if scheme_idx == -1:
               print(f"  No more unique schemes after {scheme_num}")
               break
           
           # Export model with Q/DQ nodes
           model_bytes = autotuner.export_onnx(None, insert_qdq=True)
           
           # Measure performance
           latency = benchmark_onnx_model(model_bytes)
           success = latency != float('inf')
           autotuner.submit(latency, success=success)
           
           if success:
               speedup = baseline_latency / latency
               print(f"  Scheme {scheme_idx}: {latency:.2f} ms ({speedup:.3f}x)")
       
       # Best scheme is automatically selected
       ps = autotuner.current_profile_pattern_schemes
       if ps and ps.best_scheme:
           print(f"  Best: {ps.best_scheme.latency_ms:.2f} ms")

   # Commit final region
   autotuner.set_profile_region(None, commit=True)

   # Export optimized model
   autotuner.export_onnx("optimized_final.onnx", insert_qdq=True)
   print("\nOptimization complete!")

State Management
----------------

Save and load optimization state for crash recovery:

.. code-block:: python

   # Save state after each region
   autotuner.save_state("autotuner_state.yaml")

   # Load state to resume
   autotuner = QDQAutotuner(model)
   autotuner.initialize(config)
   autotuner.load_state("autotuner_state.yaml")
   
   # Continue optimization from last checkpoint
   # (regions already profiled will be skipped)

Pattern Cache Management
------------------------

Create and use pattern caches:

.. code-block:: python

   from modelopt.onnx.quantization.autotune import PatternCache

   # Load existing cache
   cache = PatternCache.load("autotuner_state_pattern_cache.yaml")
   print(f"Loaded {cache.num_patterns} patterns")

   # Initialize autotuner with cache
   autotuner = QDQAutotuner(model)
   autotuner.initialize(config, pattern_cache=cache)

   # After optimization, pattern cache is automatically saved
   # when you call save_state()
   autotuner.save_state("autotuner_state.yaml")
   # This also saves: autotuner_state_pattern_cache.yaml

Import from QDQ Baseline
-------------------------

Extract patterns from pre-quantized models:

.. code-block:: python

   import onnx
   from modelopt.onnx.quantization.qdq_utils import get_quantized_tensors

   # Load baseline model with Q/DQ nodes
   baseline_model = onnx.load("quantized_baseline.onnx")
   
   # Extract quantized tensor names
   quantized_tensors = get_quantized_tensors(baseline_model)
   print(f"Found {len(quantized_tensors)} quantized tensors")

   # Import into autotuner
   autotuner = QDQAutotuner(model)
   autotuner.initialize(config)
   autotuner.import_insertion_points(quantized_tensors)
   
   # These patterns will be tested first during optimization

Configuration Options
=====================

Config Class
------------

The ``Config`` class controls autotuner behavior:

.. code-block:: python

   from modelopt.onnx.quantization.autotune import Config

   config = Config(
       default_quant_type="int8",             # "int8" or "fp8"
       default_dq_dtype="float32",            # DQ output: float16, float32, bfloat16
       default_q_scale=0.1,
       default_q_zero_point=0,
       top_percent_to_mutate=0.1,
       minimum_schemes_to_mutate=10,
       maximum_mutations=3,
       maximum_generation_attempts=100,
       pattern_cache_minimum_distance=4,
       pattern_cache_max_entries_per_pattern=32,
       maximum_sequence_region_size=10,
       minimum_topdown_search_size=10,
       verbose=True,
   )

Command-Line Arguments
----------------------

Arguments use underscores. Short options: ``-m`` (onnx_path), ``-o`` (output_dir), ``-s`` (schemes_per_region), ``-v`` (verbose). Run ``python -m modelopt.onnx.quantization.autotune --help`` for full help.

.. code-block:: text

   Model and Output:
     --onnx_path, -m          Path to ONNX model file (required)
     --output_dir, -o        Output directory (default: ./autotuner_output)

   Autotuning Strategy:
     --schemes_per_region, -s Number of schemes per region (default: 30)
     --pattern_cache         Pattern cache YAML for warm-start
     --qdq_baseline          QDQ baseline model to import patterns
     --state_file            State file path for resume
     --node_filter_list      File of wildcard patterns; regions with no matching nodes are skipped

   Quantization:
     --quant_type            int8 or fp8 (default: int8)
     --default_dq_dtype      float16, float32, or bfloat16 (default: float32)

   TensorRT Benchmark:
     --use_trtexec           Use trtexec instead of TensorRT Python API
     --timing_cache          TensorRT timing cache file
     --warmup_runs           Warmup runs (default: 5)
     --timing_runs           Timing runs (default: 20)
     --plugin_libraries      TensorRT plugin .so files (optional)
     --trtexec_benchmark_args Extra trtexec args (e.g. for remote autotuning)

   Logging:
     --verbose, -v           Enable debug logging

Best Practices
==============

Choosing Scheme Count
---------------------

The ``--schemes_per_region`` parameter controls exploration depth:

* **30-50 schemes**: Fast exploration, good for quick experiments
* **50-100 schemes**: Balanced (recommended for most cases)
* **100-200+ schemes**: Thorough exploration, use with pattern cache

For models with many small regions, start with fewer schemes. For models with many big regions, start with more schemes.

Managing Optimization Time
--------------------------

Optimization time depends on:

* **Number of unique patterns** (not total regions)
* **Schemes per region**
* **TensorRT engine build time** (model complexity)

**Time Estimation Formula:**

Total time ≈ (m unique patterns) × (n schemes per region) × (t seconds per benchmark) + baseline measurement

Where:
- **m** = number of unique region patterns in your model
- **n** = schemes per region (e.g., 30)
- **t** = average benchmark time (typically 3-10 seconds, depends on model size)

**Example Calculations:**

Assuming t = 5 seconds per benchmark:

* Small model: 10 patterns × 30 schemes × 5s = **25 minutes**
* Medium model: 50 patterns × 30 schemes × 5s = **2.1 hours**
* Large model: 100 patterns × 30 schemes × 5s = **4.2 hours**

Note: Actual benchmark times may depend on TensorRT engine build complexity and GPU hardware.

**Strategies to reduce time:**

1. Use pattern cache from similar models (warm-start)
2. Reduce schemes per region for initial exploration
3. Use crash recovery to split optimization across sessions

Using Pattern Cache Effectively
--------------------------------

Pattern cache is most effective when:

* Models share architectural patterns (e.g., BERT → RoBERTa)
* You're iterating on the same model (v1 → v2 → v3)
* You're optimizing a model family

**Building a pattern library:**

.. code-block:: bash

   # Optimize first model and save patterns
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path bert_base.onnx \
       --output_dir ./bert_base_run \
       --schemes_per_region 50

   # Use patterns for similar models
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path bert_large.onnx \
       --output_dir ./bert_large_run \
       --pattern_cache ./bert_base_run/autotuner_state_pattern_cache.yaml

   python -m modelopt.onnx.quantization.autotune \
       --onnx_path roberta_base.onnx \
       --output_dir ./roberta_run \
       --pattern_cache ./bert_base_run/autotuner_state_pattern_cache.yaml

Interpreting Results
--------------------

The autotuner reports speedup ratios:

.. code-block:: text

   Baseline: 12.50 ms
   Final: 9.80 ms (1.276x speedup)

**What does the speedup ratio mean:**

The speedup ratio is the ratio of the baseline latency to the final latency. It means the final latency is 1.276x faster than the baseline latency.

**If speedup is low (<1.1x):**

* Model may already be memory-bound (not compute-bound)
* Q/DQ overhead dominates small operations
* TensorRT may not fully exploit quantization for this architecture
* Try FP8 instead of INT8

Deploying Optimized Models
===========================

The optimized ONNX model contains Q/DQ nodes and is ready for TensorRT deployment:

Using trtexec
-------------

.. code-block:: bash

   # Build TensorRT engine from optimized ONNX
   trtexec --onnx=optimized_final.onnx \
           --saveEngine=model.engine \
           --stronglyTyped

   # Run inference
   trtexec --loadEngine=model.engine

Using TensorRT Python API
--------------------------

.. code-block:: python

   import tensorrt as trt
   import numpy as np

   # Create builder and logger
   logger = trt.Logger(trt.Logger.WARNING)
   builder = trt.Builder(logger)
   network = builder.create_network(
       1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
   )
   parser = trt.OnnxParser(network, logger)

   # Parse optimized ONNX model
   with open("optimized_final.onnx", "rb") as f:
       if not parser.parse(f.read()):
           for error in range(parser.num_errors):
               print(parser.get_error(error))
           raise RuntimeError("Failed to parse ONNX")

   # Build engine
   config = builder.create_builder_config()
   engine = builder.build_serialized_network(network, config)

   # Save engine
   with open("model.engine", "wb") as f:
       f.write(engine)

   print("TensorRT engine built successfully!")

Troubleshooting
===============

Common Issues
-------------

**Issue: "Benchmark instance not initialized"**

.. code-block:: python

   # Solution: Initialize benchmark before running workflow
   from modelopt.onnx.quantization.autotune.workflows import init_benchmark_instance
   init_benchmark_instance()

**Issue: All schemes show inf latency**

Possible causes:

* TensorRT cannot parse the ONNX model
* Model contains unsupported operations
* Missing custom plugin libraries

.. code-block:: bash

   # Solution: Check TensorRT logs in output_dir/logs/
   # Add plugins if needed
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --plugin_libraries /path/to/plugin.so

**Issue: Optimization is very slow**

* Check number of unique patterns (shown at start)
* Reduce schemes per region for faster exploration
* Use pattern cache from similar model

.. code-block:: bash

   # Faster exploration with fewer schemes
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --schemes_per_region 15

**Issue: Out of GPU memory during optimization**

TensorRT engine building is GPU memory intensive:

* Close other GPU processes
* Use smaller batch size in ONNX model if applicable
* Run optimization on a GPU with more memory

**Issue: Final speedup is negative (slowdown)**

The model may not benefit from quantization:

* Try FP8 instead of INT8
* Check if model is memory-bound (not compute-bound)
* Verify TensorRT can optimize the quantized operations

**Issue: Resume doesn't work after interruption**

* Ensure output directory is the same
* Check that ``autotuner_state.yaml`` exists
* If corrupted, delete state file and restart

Debugging
---------

Enable verbose logging to see detailed information:

.. code-block:: bash

   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --verbose

Check TensorRT build logs for each scheme:

.. code-block:: bash

   # Logs are saved per scheme
   ls ./output/logs/
   # baseline.log
   # region_0_scheme_0.log
   # region_0_scheme_1.log
   # ...

   # View a specific log
   cat ./output/logs/region_0_scheme_0.log

Inspect Region Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~

To understand how the autotuner partitions your model into regions, use the region inspection tool:

.. code-block:: bash

   # Basic inspection - shows region hierarchy and statistics
   python -m modelopt.onnx.quantization.autotune.region_inspect --model model.onnx

   # Verbose mode for detailed debug information
   python -m modelopt.onnx.quantization.autotune.region_inspect --model model.onnx --verbose

   # Custom maximum sequence size (default: 10)
   python -m modelopt.onnx.quantization.autotune.region_inspect --model model.onnx --max-sequence-size 20

   # Include all regions (even without quantizable operations)
   python -m modelopt.onnx.quantization.autotune.region_inspect --model model.onnx --include-all-regions

**What this tool shows:**

* **Region hierarchy**: How your model is partitioned into LEAF and COMPOSITE regions
* **Region types**: Convergence patterns (divergence→branches→convergence) vs sequences
* **Node counts**: Number of operations in each region
* **Input/output tensors**: Data flow boundaries for each region
* **Coverage statistics**: Percentage of nodes in the model covered by regions
* **Size distribution**: Histogram showing region sizes

**When to use:**

* Before optimization: Understand how many unique patterns to expect
* Slow optimization: Check if model has too many unique patterns
* Debugging: Verify region discovery is working correctly
* Model analysis: Understand computational structure

**Example output:**

.. code-block:: text

   Phase 1 complete: 45 regions, 312/312 nodes (100.0%)
   Phase 2 complete: refined 40 regions, skipped 5
   Summary: 85 regions (80 LEAF, 5 COMPOSITE), 312/312 nodes (100.0%)
   LEAF region sizes: min=1, max=15, avg=3.9
   
   ├─ Region 0 (Level 0, Type: COMPOSITE)
   │  ├─ Direct nodes: 0
   │  ├─ Total nodes (recursive): 28
   │  ├─ Children: 4
   │  ├─ Inputs: 3 tensors
   │  └─ Outputs: 2 tensors
   │    ├─ Region 1 (Level 1, Type: LEAF)
   │    │  ├─ Direct nodes: 5
   │    │  ├─ Nodes: Conv, BatchNormalization, Relu
   │    ...

This helps you understand:

* **Number of patterns**: More regions = more unique patterns = longer optimization
* **Region sizes**: Very large regions might need adjustment via ``--max-sequence-size`` (region_inspect)
* **Model structure**: Identifies divergent/convergent patterns (skip connections, branches)

API Reference
=============

For detailed API documentation, see :doc:`../reference/2_qdq_placement`.

Key Classes:

* :class:`~modelopt.onnx.quantization.autotune.QDQAutotuner` - Main autotuner with automatic region discovery
* :class:`~modelopt.onnx.quantization.autotune.Config` - Configuration parameters
* :class:`~modelopt.onnx.quantization.autotune.PatternCache` - Pattern cache for warm-start
* :class:`~modelopt.onnx.quantization.autotune.Region` - Hierarchical subgraph representation
* :class:`~modelopt.onnx.quantization.autotune.InsertionScheme` - Q/DQ insertion point collection

Key Functions:

* :func:`~modelopt.onnx.quantization.autotune.workflows.region_pattern_autotuning_workflow` - Complete optimization workflow
* :func:`~modelopt.onnx.quantization.autotune.workflows.init_benchmark_instance` - Initialize global TensorRT benchmark (call before benchmark_onnx_model when using workflow from Python)
* :func:`~modelopt.onnx.quantization.autotune.workflows.benchmark_onnx_model` - Benchmark ONNX model with TensorRT

Frequently Asked Questions
==========================

**Q: How long does optimization take?**

A: Optimization time is: (unique patterns) × (schemes per region) × (benchmark time). For example, with 30 schemes/region and 5 seconds/benchmark: 10 patterns = 25 minutes, 50 patterns = 2.1 hours, 100 patterns = 4.2 hours. The number of unique patterns depends on your model's architectural diversity—models with repeated structures (like transformers) have fewer unique patterns. Use pattern cache to significantly reduce time for similar models.

**Q: Can I stop optimization early?**

A: Yes! Press Ctrl+C to interrupt. The progress is saved and you can resume later.

**Q: Do I need calibration data?**

A: No, the autotuner focuses on Q/DQ placement optimization, not calibration. Calibration scales are added when the Q/DQ nodes are inserted. For best accuracy, run calibration separately after optimization.

**Q: Can I use this with PyTorch models?**

A: Export your PyTorch model to ONNX first using ``torch.onnx.export()``, then run the autotuner on the ONNX model.

**Q: What's the difference from modelopt.onnx.quantization.quantize()?**

A: ``quantize()`` is a fast PTQ tool that uses heuristics for Q/DQ placement. The autotuner uses TensorRT measurements to optimize placement for best performance. Use ``quantize()`` for quick results, autotuner for maximum performance.

**Q: Can I customize region discovery?**

A: Yes, inherit from ``QDQAutotunerBase`` and provide your own regions instead of using automatic discovery:

.. code-block:: python

   from modelopt.onnx.quantization.autotune import QDQAutotunerBase, Region
   
   class CustomAutotuner(QDQAutotunerBase):
       def __init__(self, model, custom_regions):
           super().__init__(model)
           self.regions = custom_regions  # Your custom regions

**Q: Does this work with dynamic shapes?**

A: The autotuner uses TensorRT for benchmarking, which requires fixed shapes. Set fixed input shapes in your ONNX model before optimization.

**Q: Can I optimize for accuracy instead of latency?**

A: Currently, the autotuner optimizes for latency only.

Examples
========

Example 1: Basic Optimization
------------------------------

.. code-block:: bash

   # Optimize a ResNet model with INT8 quantization
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path resnet50.onnx \
       --output_dir ./resnet50_optimized \
       --quant_type int8 \
       --schemes_per_region 30

Example 2: Transfer Learning with Pattern Cache
------------------------------------------------

.. code-block:: bash

   # Optimize GPT-2 small
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path gpt2_small.onnx \
       --output_dir ./gpt2_small_run \
       --quant_type fp8 \
       --schemes_per_region 50

   # Reuse patterns for GPT-2 medium (much faster)
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path gpt2_medium.onnx \
       --output_dir ./gpt2_medium_run \
       --quant_type fp8 \
       --pattern_cache ./gpt2_small_run/autotuner_state_pattern_cache.yaml

Example 3: Import from Manual Baseline
---------------------------------------

.. code-block:: bash

   # You have a manually quantized baseline
   # Import its patterns as starting point
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --output_dir ./auto_optimized \
       --qdq_baseline manually_quantized.onnx \
       --schemes_per_region 40

Example 4: Full Python Workflow
--------------------------------

.. code-block:: python

   from pathlib import Path
   from modelopt.onnx.quantization.autotune.workflows import (
       region_pattern_autotuning_workflow,
       init_benchmark_instance
   )
   
   # Initialize TensorRT benchmark
   init_benchmark_instance(
       timing_cache_file="/tmp/trt_cache.cache",
       warmup_runs=5,
       timing_runs=20
   )
   
   # Run optimization
   autotuner = region_pattern_autotuning_workflow(
       model_path="model.onnx",
       output_dir=Path("./results"),
       num_schemes_per_region=30,
       pattern_cache_file=None,
       state_file=None,
       quant_type="int8",
       default_dq_dtype="float32",
       qdq_baseline_model=None,
       node_filter_list=None,
       verbose=False,
   )
   
   # Access results
   print(f"Baseline latency: {autotuner.baseline_latency_ms:.2f} ms")
   print(f"Number of patterns: {len(autotuner.profiled_patterns)}")
   
   # Pattern cache is automatically saved during workflow
   # Check the output directory for autotuner_state_pattern_cache.yaml
   if autotuner.pattern_cache:
       print(f"Pattern cache contains {autotuner.pattern_cache.num_patterns} patterns")

Conclusion
==========

The ``modelopt.onnx.quantization.autotune`` module provides a powerful automated approach to Q/DQ placement optimization. By combining automatic region discovery, pattern-based optimization, and TensorRT performance measurement, it finds optimal quantization strategies without manual tuning.

**Next Steps:**

* Try the quick start example on your model
* Experiment with different ``--schemes_per_region`` values
* Build a pattern cache library for your model family
* Integrate optimized models into your deployment pipeline

For architectural details and API reference, see :doc:`../reference/2_qdq_placement`.
