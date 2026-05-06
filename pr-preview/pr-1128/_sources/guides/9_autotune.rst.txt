===============================================
Autotune (ONNX)
===============================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

The ``modelopt.onnx.quantization.autotune`` module automates Q/DQ (Quantize/Dequantize) placement optimization in ONNX models. It explores placement strategies and uses TensorRT latency measurements to choose a configuration that minimizes inference time.

**Key Features:**

* **Automatic Region Discovery**: Intelligently partitions the model into optimization regions
* **Pattern-Based Optimization**: Groups structurally similar regions and optimizes them together
* **TensorRT Performance Measurement**: Uses actual inference latency (not theoretical estimates)
* **Crash Recovery**: Checkpoint/resume capability for long-running optimizations
* **Warm-Start Support**: Reuses learned patterns from previous runs
* **Multiple Quantization Types**: Supports INT8 and FP8 quantization

**When to Use This Tool:**

* Quantizing an ONNX model for TensorRT deployment
* Optimizing Q/DQ placement for best performance
* The model has repeating structures (e.g., transformer blocks, ResNet layers)

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

1. Discover regions in the model automatically
2. Measure baseline performance (no quantization)
3. Test different Q/DQ placement schemes for each region pattern
4. Select the best scheme based on TensorRT latency measurements
5. Export an optimized ONNX model with Q/DQ nodes

**Output Files:**

Files are written under the output directory (default ``./autotuner_output``, or the path given by ``--output_dir``):

.. code-block:: text

   autotuner_output/                         # default; or the path passed to --output_dir
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

   # When using the CLI, the benchmark is initialized automatically. When calling the
   # workflow from Python, call init_benchmark_instance first:
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
   
   Regions with identical structural patterns are grouped together. For example, all Convolution->BatchNormalization->ReLU blocks in the model share the same pattern.

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

**Why pattern-based?**

The autotuner optimizes each unique pattern once; the chosen scheme then applies to every region that matches that pattern. So runtime scales with the number of *patterns*, not regions. Models with repeated structure (e.g. transformers) benefit most; highly diverse graphs have more patterns and take longer.

Advanced Usage
==============

Warm-Start with Pattern Cache
------------------------------

Pattern cache files store the best Q/DQ schemes from previous optimization runs. These patterns can be reused on similar models or model versions:

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

The second run tests cached schemes first and can reach a good configuration faster.

**When to use pattern cache:**

* Optimizing multiple versions of the same model
* Optimizing models from the same family (e.g., different BERT variants)
* Transferring learned patterns across models

Import Patterns from Existing QDQ Models
-----------------------------------------

With a pre-quantized baseline model (e.g., from manual optimization or another tool), its Q/DQ patterns can be imported:

.. code-block:: bash

   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --output_dir ./results \
       --qdq_baseline manually_quantized.onnx

The workflow extracts Q/DQ insertion points from the baseline, maps them to region patterns, and uses them as seed schemes. Useful when:

* Starting from expert-tuned quantization schemes
* Comparing against reference implementations
* Fine-tuning existing quantized models

Resume After Interruption
--------------------------

A long run can be interrupted (Ctrl+C, preemption, or crash) and resumed later:

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

When rerun with the same ``--output_dir``, the autotuner detects ``autotuner_state.yaml``, restores progress, and continues from the next unprofiled region.

Custom TensorRT Plugins
-----------------------

If the model uses custom TensorRT operations, provide the plugin libraries:

.. code-block:: bash

   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --output_dir ./results \
       --plugin_libraries /path/to/plugin1.so /path/to/plugin2.so

Low-Level API Usage
===================

For fine-grained control over the autotune process (e.g. driving it step-by-step or customizing regions and schemes), use the autotuner classes directly:

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
   # when save_state() is called
   autotuner.save_state("autotuner_state.yaml")
   # This also saves: autotuner_state_pattern_cache.yaml

Import from a Q/DQ Baseline
---------------------------

To seed the autotuner from a pre-quantized model (e.g. from another tool or manual tuning), extract quantized tensor names and pass them in:

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
       default_dq_dtype="float32",            # float16, float32, bfloat16 (bfloat16 needs NumPy with np.bfloat16)
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

.. argparse::
   :module: modelopt.onnx.quantization.autotune.__main__
   :func: get_parser
   :prog: python -m modelopt.onnx.quantization.autotune
   :nodescription:
   :noepilog:

Best Practices
==============

Choosing Scheme Count
---------------------

The ``--schemes_per_region`` (or ``-s``) parameter controls exploration depth. Typical values:

* **15–30 schemes** (e.g. ``-s 30``): Quick exploration; good for trying the tool or small models
* **50 schemes** (default, ``-s 50``): Default; Recommended for most cases
* **100–200+ schemes** (e.g. ``-s 200``): Extensive search; consider using a pattern cache to avoid re-exploring

Use fewer schemes when there are many small regions or limited time; use more for large or critical regions.

.. _managing-optimization-time:

Managing Optimization Time
--------------------------

Optimization time depends on:

* **Number of unique patterns** (not total regions)
* **Schemes per region**
* **TensorRT engine build time** (model complexity)

**Time Estimation Formula:**

Total time ≈ (m unique patterns) × (n schemes per region) × (t seconds per benchmark) + baseline measurement

Where:
- **m** = number of unique region patterns in the model
- **n** = schemes per region (e.g., 30)
- **t** = average benchmark time (typically 3-10 seconds, depends on model size)

**Example Calculations:**

Assuming t = 5 seconds per benchmark:

* Small model: 10 patterns × 30 schemes × 5s = **25 minutes**
* Medium model: 50 patterns × 30 schemes × 5s = **2.1 hours**
* Large model: 100 patterns × 30 schemes × 5s = **4.2 hours**

Note: Actual benchmark times may depend on TensorRT engine build complexity and GPU hardware.

**Ways to reduce time:** Use a pattern cache from a similar model (warm-start), use fewer schemes per region for initial runs, or rely on checkpoint/resume to split work across sessions.

Using the Pattern Cache Effectively
-----------------------------------

The pattern cache helps most when models share structure (e.g. BERT → RoBERTa), when iterating on the same model (v1 → v2), or when optimizing a family of models.

**Example: building a pattern library**

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

**What the speedup ratio means:** Baseline ÷ final latency (e.g. 1.276x = final is about 22% faster than baseline).

**If speedup is low (<1.1x):**

* Model may already be memory-bound (not compute-bound)
* Q/DQ overhead dominates small operations
* TensorRT may not fully exploit quantization for this architecture
* Try FP8 instead of INT8

Deploying Optimized Models
===========================

The optimized ONNX model includes Q/DQ nodes and can be used with TensorRT as follows.

Using Trtexec
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
       | 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
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
   if engine is None:
       raise RuntimeError("TensorRT engine build failed")

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
* cuda-python package not installed when using TensorRTPyBenchmark

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

* Use the same ``--output_dir`` (and ``--onnx_path``) as the original run
* Confirm ``autotuner_state.yaml`` exists in that directory
* If the state file is corrupted, remove it and start over

Debugging
---------

Enable verbose logging to see detailed information:

.. code-block:: bash

   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --verbose

Check TensorRT build logs for each scheme (under the output directory, default ``./autotuner_output``):

.. code-block:: bash

   # Logs are saved per scheme (replace autotuner_output with your --output_dir if different)
   ls ./autotuner_output/logs/
   # baseline.log
   # region_0_scheme_0.log
   # region_0_scheme_1.log
   # ...

   # View a specific log
   cat ./autotuner_output/logs/region_0_scheme_0.log

Inspect Region Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~

To understand how the autotuner partitions the model into regions, use the region inspection tool:

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

* **Region hierarchy**: How the model is partitioned into LEAF and COMPOSITE regions
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

Use this to see how many unique patterns to expect (more patterns → longer optimization), whether region sizes need tuning (e.g. ``--max-sequence-size`` in region_inspect), and where branches or skip connections appear.

Architecture and Workflow
=========================

The autotuner partitions the ONNX graph into regions, groups regions by structural pattern, and for each pattern tests multiple Q/DQ insertion schemes via TensorRT benchmarking. The following diagram summarizes the end-to-end process:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │ 1. Model Loading & Initialization                           │
   │    • Load ONNX model                                        │
   │    • Create QDQAutotuner instance                           │
   │    • Run automatic region discovery                         │
   │    • Load pattern cache (warm-start)                        │
   │    • Import patterns from QDQ baseline (optional)           │
   └────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 2. Baseline Measurement                                     │
   │    • Export model without Q/DQ nodes                        │
   │    • Build TensorRT engine                                  │
   │    • Measure baseline latency                               │
   └────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 3. Pattern-Based Region Profiling                           │
   │    For each region: set profile region, generate schemes,   │
   │    benchmark each scheme, commit best, save state           │
   └────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 4. Finalization                                             │
   │    • Export optimized model with all best schemes           │
   │    • Save state and pattern cache                           │
   └─────────────────────────────────────────────────────────────┘

Design Rationale
----------------

* **Pattern-based**: One optimization per pattern; the chosen scheme applies to every matching region, reducing work and keeping behavior consistent.
* **Hierarchical regions**: LEAF (single ops or short sequences) and COMPOSITE (nested subgraphs) allow tuning at different granularities.
* **Incremental state**: Progress is saved after each region so runs can be resumed after interruption.

Limitations and Future Work
============================

**Current limitations:**

* Random scheme sampling may miss optimal configurations; number of schemes per region is fixed.
* Structural similarity is assumed to imply similar performance; context (input/output) can vary.
* Uniform quantization per scheme (no mixed-precision within a scheme).
* TensorRT engine build time dominates; each scheme requires a full engine build.
* Performance is measured with default/dummy inputs and may not generalize to all distributions.

**Possible future enhancements:**

* Advanced search (e.g. Bayesian optimization, evolutionary algorithms).
* Mixed-precision and per-layer bit-width.
* Accuracy constraints and multi-objective (latency + accuracy) optimization.

Glossary
========

.. glossary::

   Q/DQ Nodes
      QuantizeLinear (Q) and DequantizeLinear (DQ) nodes in ONNX that convert between
      floating-point and quantized integer representations.

   Region
      A hierarchical subgraph in an ONNX computation graph with well-defined input and
      output boundaries. Can be LEAF (atomic), COMPOSITE (containing child regions), or ROOT.

   Pattern
      A structural signature of a region. Regions with identical patterns can share insertion schemes.

   Insertion Scheme
      A collection of insertion points specifying where to insert Q/DQ nodes within a region.
      Schemes use pattern-relative addressing for portability.

   Pattern Cache
      Collection of top-performing insertion schemes for multiple patterns, used to
      warm-start optimization on similar models.

   Baseline Latency
      Inference latency of the model without any Q/DQ nodes, used as reference for speedup.

   TensorRT Timing Cache
      Persistent cache of kernel performance measurements used by TensorRT to speed up engine builds.

References
==========

* **ONNX**: https://onnx.ai/
* **ONNX Technical Details**: https://onnx.ai/onnx/technical/index.html
* **TensorRT Documentation**: https://docs.nvidia.com/deeplearning/tensorrt/
* **NVIDIA Model Optimizer (ModelOpt)**: https://github.com/NVIDIA/Model-Optimizer
* **ONNX GraphSurgeon**: https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon

Frequently Asked Questions
==========================

**Q: How long does optimization take?**

A: Time ≈ (unique patterns) × (schemes per region) × (time per benchmark). See :ref:`managing-optimization-time` for a formula and examples. Use a pattern cache when re-running on similar models to reduce time.

**Q: Can I stop optimization early?**

A: Yes. Press Ctrl+C to interrupt. Progress is saved and the run can be resumed later.

**Q: Do I need calibration data?**

A: No, the autotuner focuses on Q/DQ placement optimization, not calibration. Calibration scales are added when the Q/DQ nodes are inserted. For best accuracy, run calibration separately after optimization.

**Q: Can I use this with PyTorch models?**

A: Export the PyTorch model to ONNX first using ``torch.onnx.export()``, then run the autotuner on the ONNX model.

**Q: What's the difference from modelopt.onnx.quantization.quantize()?**

A: ``quantize()`` is a fast PTQ tool that uses heuristics for Q/DQ placement. The autotuner uses TensorRT measurements to optimize placement for best performance. Use ``quantize()`` for quick results, autotuner for maximum performance.

**Q: Can I customize region discovery?**

A: Yes. Subclass ``QDQAutotunerBase`` and supply custom regions instead of using automatic discovery:

.. code-block:: python

   from modelopt.onnx.quantization.autotune import QDQAutotunerBase, Region
   
   class CustomAutotuner(QDQAutotunerBase):
       def __init__(self, model, custom_regions):
           super().__init__(model)
           self.regions = custom_regions  # Custom regions

**Q: Does this work with dynamic shapes?**

A: The autotuner uses TensorRT for benchmarking, which requires fixed shapes. Set fixed input shapes in the ONNX model before optimization. If the model was exported with dynamic shapes, one option is to use Polygraphy to fix them to static shapes, for example:

.. code-block:: bash

   $ polygraphy surgeon sanitize --override-input-shapes x:[128,3,1024,1024] -o model_bs128.onnx model.onnx

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

   # With a manually quantized baseline
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
   
   # Run optimization (only non-defaults shown; see API for all options)
   autotuner = region_pattern_autotuning_workflow(
       model_path="model.onnx",
       output_dir=Path("./results"),
       num_schemes_per_region=30,
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

**Next steps:** Run the quick start on a model, try different ``--schemes_per_region`` values, build a pattern cache for the model family, then integrate the optimized model into the deployment pipeline.
