====================================================
Automatic ONNX Q/DQ Placement Optimizer Architecture
====================================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
========

The ``modelopt.onnx.quantization.autotune`` module provides an automatic optimization framework for Quantize/Dequantize (Q/DQ) node placement in ONNX models. The system partitions ONNX computation graphs into smaller regions and systematically searches for optimal Q/DQ insertion points to minimize TensorRT inference latency.

**Key Capabilities:**

* **Automatic Region Discovery**: Identifies optimization regions around compute-intensive operations
* **Pattern-Based Optimization**: Groups structurally similar regions and applies learned schemes across all instances
* **Performance-Driven Search**: Uses TensorRT profiling to measure actual inference latency and guide optimization
* **Incremental State Management**: Supports crash recovery and resumption of optimization sessions
* **Pattern Cache**: Enables warm-start optimization by reusing known-good schemes from previous runs
* **Baseline Import**: Transfers quantization patterns from existing QDQ models

Architecture Overview
=====================

Core Design Principles
----------------------

1. **Hierarchical Region Partitioning**: This module decomposes ONNX graphs into a hierarchical tree of regions, enabling focused optimization at different granularity levels.

2. **Pattern-Based Scheme Sharing**: Regions with identical topological structure share the same pattern signature. Optimization schemes are portable across all regions matching a pattern, reducing the search space significantly.

3. **Performance-Driven Selection**: Every insertion scheme is evaluated through actual TensorRT engine compilation and profiling, ensuring real-world performance gains.

4. **Incremental Optimization**: Regions are optimized sequentially with the best scheme committed before proceeding to the next region, allowing progressive refinement.

Module Structure
----------------

.. code-block:: text

   autotune/
   ├── Core API
   │   ├── autotuner.py           # QDQAutotuner (automatic region discovery)
   │   ├── autotuner_base.py      # QDQAutotunerBase (core optimization logic)
   │   ├── workflows.py           # High-level workflow and benchmark helpers
   │   └── common.py              # Data structures (Region, Config, PatternCache, etc.)
   │
   ├── Region Management
   │   ├── region_search.py       # CombinedRegionSearch (region discovery)
   │   ├── region_pattern.py      # RegionPattern (structural pattern matching)
   │   └── region_inspect.py      # CLI to inspect region search (debugging)
   │
   ├── Q/DQ Insertion & Export
   │   ├── insertion_points.py    # Insertion point types and resolution
   │   └── export_utils.py        # Q/DQ node creation and ONNX export
   │
   ├── Benchmarking
   │   └── benchmark.py           # TensorRTPyBenchmark, TrtExecBenchmark
   │
   └── Entry Points
   │   ├── __init__.py            # Public API exports
   │   └── __main__.py            # Command-line interface
   │
   Q/DQ analysis (used for baseline import) lives in the parent package:
   modelopt.onnx.quantization.qdq_utils (e.g. get_quantized_tensors).


Key Components
==============

1. Autotuner (autotuner.py, autotuner_base.py)
------------------------------------------------

The autotuner is the central orchestrator of the Q/DQ optimization process.

QDQAutotunerBase
~~~~~~~~~~~~~~~~

Base class (in ``autotuner_base.py``) providing core optimization functionality:

* **Scheme Generation**: Creates candidate Q/DQ insertion schemes for regions
* **Model Export**: Generates ONNX models with specified Q/DQ insertions applied
* **Performance Tracking**: Records and ranks schemes by measured latency
* **State Persistence**: Saves/loads optimization progress for crash recovery

**Key Attributes:**

* ``graph``: Clean ONNX GraphSurgeon representation of the model
* ``regions``: List of regions to optimize (populated by subclass)
* ``profiled_patterns``: Pattern-based scheme results
* ``current_profile_region``: Region currently being optimized
* ``config``: Configuration parameters
* ``pattern_cache``: Seed schemes from previous optimization runs

**Workflow Methods:**

* ``initialize(config, pattern_cache)``: Configure autotuner and prepare for profiling
* ``set_profile_region(region, commit)``: Select region to profile and commit previous results
* ``generate()``: Generate a new insertion scheme for current region
* ``export_onnx(path_or_none, insert_qdq, best=False)``: Export model with Q/DQ nodes. If path is ``None``, returns serialized model bytes (for in-memory benchmarking). When ``best=True``, exports using the current region's best scheme so far.
* ``submit(latency, success)``: Record performance measurement for current scheme
* ``save_state(path)`` / ``load_state(path)``: Persist/restore optimization state

QDQAutotuner
~~~~~~~~~~~~

Concrete implementation with automatic region discovery:

* Inherits from ``QDQAutotunerBase``
* Automatically discovers regions during initialization using ``CombinedRegionSearch``
* For custom partitioning, users can implement their own region search by subclassing ``RegionSearchBase`` and overriding ``_search_regions()`` in a subclass of ``QDQAutotuner`` to use it.

**Initialization Process:**

1. Constructs root region encompassing entire graph
2. Runs combined region search to identify optimization candidates
3. Prepares region hierarchy for sequential optimization

2. Region Management
--------------------

Region Partitioning (region_search.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The region search module implements hierarchical partitioning strategies to decompose ONNX graphs into optimization regions.

**CombinedRegionSearch**

Multi-strategy region discovery combining:

* **Pattern-Based Search**: Identifies common subgraph patterns (Conv+BN+Relu, etc.)
* **Operation-Centered Search**: Creates regions around major quantizable operations (Conv, MatMul, Gemm)
* **Sequence Merging**: Combines adjacent linear operations into single regions
* **Hierarchical Composition**: Builds multi-level region trees

**Region Discovery Algorithm:**

1. **Bottom-Up Search**: Start from individual operations
2. **Local Expansion**: Expand forward/backward from seed nodes within step limits
3. **Pattern Recognition**: Identify and merge common computational patterns
4. **Hierarchy Construction**: Build parent-child relationships between regions

**Key Classes:**

* ``RegionSearchBase``: Base class with graph traversal utilities
* ``CombinedRegionSearch``: Main region discovery implementation

**Region Types:**

* ``LEAF``: Atomic regions containing only direct nodes
* ``COMPOSITE``: Hierarchical regions containing child regions

Region Pattern Analysis (region_pattern.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provides structural pattern matching for regions, enabling scheme portability.

**RegionPattern Class**

Represents the topological signature of a region:

* **Signature Generation**: Creates deterministic hash from region structure
  
  - Node operation types
  - Connectivity patterns (inputs/outputs per node)
  - Child region structures (for composite regions)
  - Handles symmetric operations (Add, Mul) order-invariantly

* **Pattern Matching**: Groups regions by structural similarity
* **Insertion Point Resolution**: Resolves pattern-relative addresses to actual tensor names

**Signature Components:**

.. code-block:: text

   Pattern Signature = hash(
       node_types_sorted +
       connectivity_structure +
       child_region_patterns +
       symmetry_normalization
   )

**Key Methods:**

* ``from_region(region, graph)``: Generate pattern from region

Region Inspection (region_inspect.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CLI and helper for debugging region discovery without running benchmarks:

* **Entry point**: ``python -m modelopt.onnx.quantization.autotune.region_inspect --model model.onnx``
* **inspect_region_search(onnx_path, max_sequence_size=10, include_all_regions=False)**: Loads the model, runs ``CombinedRegionSearch``, and prints region hierarchy, node counts, and summary statistics. Returns the list of discovered regions.
* **Options**: ``--verbose`` / ``-v`` for debug logging; ``--max-sequence-size`` for sequence region size; ``--include-all-regions`` to include regions without major quantizable ops (Conv, MatMul, etc.).

Use this to verify how the autotuner partitions the graph before or during tuning.

3. Q/DQ Insertion Points
------------------------

Insertion Point Types (insertion_points.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines three types of Q/DQ insertion locations:

**NodeInputInsertionPoint**

Inserts Q/DQ at a specific node input:

* Pattern-relative node index
* Input tensor index (0, 1, 2, ...)
* Most common insertion type for quantizing operation inputs

**RegionOutputInsertionPoint**

Inserts Q/DQ at region output, only used for composite regions:

* Pattern-relative child region index
* Output tensor index from that region
* Used for composite regions with child boundaries

**ChildRegionInputInsertionPoint**

Inserts Q/DQ at a child region input boundary:

* Pattern-relative child region index
* Input tensor index to that region
* Enables quantization of data flowing into subregions

**InsertionScheme**

Collection of insertion points with performance metadata:

* Set of insertion points (pattern-relative)
* Measured latency (ms)
* Success/failure status
* Unique fingerprint for deduplication

**Resolution process**

Insertion points and schemes are pattern-relative (node/region indices within the pattern), so the same scheme applies to every region that matches the pattern. Before adding Q/DQ nodes to the ONNX graph, the autotuner resolves them to concrete tensor names in the current model:

1. Take pattern-relative insertion points from the scheme
2. Map node/region indices to actual graph elements for the target region
3. Resolve to concrete tensor names (producer/consumer)
4. Merge and deduplicate so each tensor gets at most one Q/DQ pair
5. Create and insert Q/DQ nodes at the resolved locations (see ``export_utils``)

Q/DQ Analysis (modelopt.onnx.quantization.qdq_utils)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The autotune package uses the parent quantization package for Q/DQ analysis:

* **get_quantized_tensors(onnx_model)** — Returns the set of tensor names that have Q/DQ nodes in the given ONNX model. Used by the workflow when ``qdq_baseline_model`` is provided to import insertion patterns from an existing quantized model for warm-start.

4. Workflows (workflows.py)
---------------------------

High-level workflow functions orchestrating the complete optimization process.

region_pattern_autotuning_workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Main workflow for pattern-based Q/DQ optimization:

**Workflow Steps:**

1. **Initialization**
   
   * Load ONNX model
   * Create autotuner with automatic region discovery
   * Load pattern cache (if provided)
   * Import patterns from QDQ baseline (if provided)

2. **Baseline Measurement**
   
   * Export model without Q/DQ nodes
   * Benchmark with TensorRT to establish baseline latency

3. **Region Profiling Loop**
   
   For each discovered region:
   
   * Set as current profile region
   * Generate N insertion schemes (default: 30)
   * For each scheme:
     
     - Export ONNX model with Q/DQ nodes applied
     - Build TensorRT engine and measure latency
     - Submit result to autotuner
   
   * Commit best scheme for region
   * Save incremental state (crash recovery)

4. **Finalization**
   
   * Export final optimized model with all best schemes
   * Measure final latency and compute speedup
   * Save complete state and pattern cache

**Key Features:**

* **Automatic Resume**: Detects existing state file and continues from last checkpoint
* **Pattern Cache Warm-Start**: Seeds scheme generation with known-good patterns
* **Baseline Import**: Extracts quantization patterns from existing QDQ models
* **Progressive Saving**: State saved after each region for crash recovery

Benchmarking Functions (workflows.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``benchmark_onnx_model(model_path, log_file=None, flush_timing_cache=False)``: Run global benchmark on ONNX model (path or bytes); returns median latency in ms or ``float('inf')`` on failure.
* ``init_benchmark_instance(use_trtexec=False, plugin_libraries=None, timing_cache_file=None, warmup_runs=5, timing_runs=20, trtexec_args=None)``: Initialize the global TensorRT benchmark used by the workflow (must be called before ``benchmark_onnx_model``).

5. Benchmarking (benchmark.py)
-------------------------------

TensorRT integration for performance measurement.

Benchmark Classes
~~~~~~~~~~~~~~~~~

**Abstract interface (Benchmark):**

* ``run(model_path, log_file=None, flush_timing_cache=False)``: Benchmark model (file path or bytes) and return median latency (ms).

**TensorRTPyBenchmark** (default)

Uses TensorRT Python API:

* Direct Python bindings to TensorRT
* Persistent Builder/Runtime/Logger instances
* Efficient for repeated benchmarking
* Timing cache support for faster engine builds
* Optional plugin library paths (list of ``.so`` paths)

**TrtExecBenchmark** (optional, ``--use_trtexec``)

Uses ``trtexec`` command-line tool:

* Spawns subprocess per benchmark
* Useful when Python API is unavailable or for remote autotuning (e.g. ``--trtexec_benchmark_args "--remoteAutoTuningConfig=..."``)
* Supports same timing cache, warmup/timing runs, and plugin libraries
* ``trtexec_args``: optional list of extra arguments passed to trtexec

**Benchmarking process:**

1. Parse ONNX model (from path or bytes)
2. Build TensorRT engine with optimization
3. Load timing cache (if available)
4. Warmup iterations (default: 5)
5. Timing iterations (default: 20)
6. Median latency reported; timing cache updated

**Configuration:**

* ``timing_cache_file``: Path to TensorRT timing cache (default: system temp ``trtexec_timing.cache``)
* ``warmup_runs``: Warmup iterations (default: 5)
* ``timing_runs``: Timed iterations (default: 20)
* ``plugin_libraries``: List of TensorRT plugin ``.so`` paths (optional)
* ``trtexec_args``: Extra arguments for trtexec (optional; only when ``use_trtexec=True``)

6. Configuration (common.py)
-----------------------------

Config Class
~~~~~~~~~~~~

Central configuration for autotuning behavior. Controls the autotuning process including 
performance requirements, quantization parameters, region building, scheme generation, and 
pattern cache behavior.

**Logging:**

* ``verbose`` (bool): Enable detailed logging of autotuning progress (default: False)

**Performance:**

* ``performance_threshold`` (float): Minimum speedup ratio to accept a scheme; 1.0 = no improvement required, 1.02 = 2% (default: 1.02)

**Quantization Parameters:**

* ``default_q_scale`` (float): Default scale parameter for Q/DQ nodes. Typical range: 0.01-0.1 (default: 0.1)
* ``default_q_zero_point`` (int): Default zero-point for Q/DQ nodes; 0 for signed int8, 128 for uint8 (default: 0)
* ``default_quant_type`` (str): Quantization type for Q/DQ nodes: "int8" (default), "fp8"
* ``default_dq_dtype`` (str): Dtype for DequantizeLinear output when not inferred: "float32" (default), "float16", "bfloat16"

**Region Builder Settings:**

* ``maximum_sequence_region_size`` (int): Maximum number of nodes in a sequence region during 
  top-down refinement. Prevents overly large merged regions (default: 10)
* ``minimum_topdown_search_size`` (int): Minimum number of nodes in a region to trigger 
  top-down search during region building (default: 10)

**Scheme Generation Settings:**

* ``top_percent_to_mutate`` (float): Top percentage of best schemes to use as mutation seeds 
  during scheme generation. Range: 0.0-1.0 (default: 0.1 = top 10%)
* ``minimum_schemes_to_mutate`` (int): Minimum number of schemes to keep as mutation seeds, 
  even if top_percent_to_mutate results in fewer (default: 10)
* ``maximum_mutations`` (int): Maximum number of mutations to apply to a single scheme 
  during generation (default: 3)
* ``maximum_generation_attempts`` (int): Maximum attempts to generate a unique new scheme 
  before giving up (default: 100)

**Pattern Cache Settings:**

* ``pattern_cache_minimum_distance`` (int): Minimum edit distance required between schemes in cache.
  When adding schemes, if a scheme is too similar (distance < minimum_distance) to an existing 
  scheme, only the better-performing one is kept (default: 4)
* ``pattern_cache_max_entries_per_pattern`` (int): Maximum number of schemes to keep per pattern 
  in pattern cache. Only the top N best-performing schemes are kept for each pattern. 
  Use 0 to keep all schemes (default: 32)

**Example:**

.. code-block:: python

   from modelopt.onnx.quantization.autotune import Config

   config = Config(
       default_quant_type="fp8",
       default_dq_dtype="float16",
       default_q_scale=0.05,
       top_percent_to_mutate=0.2,
       maximum_mutations=5,
       pattern_cache_minimum_distance=2,
       pattern_cache_max_entries_per_pattern=64,
       verbose=True,
   )

PatternCache Class
~~~~~~~~~~~~~~~~~~

Stores top-performing schemes for pattern-based warm-start:

* Maps pattern signatures to ``PatternSchemes``
* Maintains diversity through distance-based filtering
* Limits entries per pattern to avoid bloat
* Serializable to YAML for persistence

**Cache Operations:**

* ``add_pattern_schemes(pattern_schemes)``: Add a ``PatternSchemes`` instance (with diversity filtering)
* ``get_pattern_schemes(pattern_signature)``: Return ``PatternSchemes`` for a pattern signature, or ``None``
* ``save(path)`` / ``load(path)``: Persist cache to YAML / load from YAML

Region Class
~~~~~~~~~~~~

Hierarchical subgraph representation:

**Attributes:**

* ``id``: Unique identifier
* ``level``: Hierarchical level (0=leaf, higher=composite)
* ``type``: RegionType (LEAF/COMPOSITE)
* ``parent``: Parent region reference
* ``children``: List of child regions
* ``nodes``: Set of direct node indices
* ``inputs``: Input tensor names
* ``outputs``: Output tensor names

**Methods:**

* Hierarchy navigation (parent/children access)
* Node management (direct vs recursive nodes)
* Boundary computation (inputs/outputs)
* Metadata storage


Autotuning Workflow
===================

Complete Optimization Process
------------------------------

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
   │    • Submit to autotuner                                    │
   └────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 3. Pattern-Based Region Profiling                           │
   │    ┌───────────────────────────────────────────┐            │
   │    │ For each region:                          │            │
   │    │   • Set as current profile region         │            │
   │    │   • Check if pattern already profiled     │            │
   │    │   • Generate N insertion schemes          │            │
   │    │     ┌─────────────────────────────┐       │            │
   │    │     │ For each scheme:            │       │            │
   │    │     │   • Generate unique scheme  │       │            │
   │    │     │   • Export model with Q/DQ  │       │            │
   │    │     │   • Build TRT engine        │       │            │
   │    │     │   • Measure latency         │       │            │
   │    │     │   • Submit result           │       │            │
   │    │     └─────────────────────────────┘       │            │
   │    │   • Select best scheme for pattern        │            │
   │    │   • Commit scheme (applies to all         │            │
   │    │     regions with this pattern)            │            │
   │    │   • Save incremental state                │            │
   │    └───────────────────────────────────────────┘            │
   └────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 4. Finalization                                             │
   │    • Commit final region                                    │
   │    • Export optimized model with all best schemes           │
   │    • Measure final latency                                  │
   │    • Compute speedup ratio                                  │
   │    • Save complete state file                               │
   │    • Save pattern cache for future runs                     │
   └─────────────────────────────────────────────────────────────┘

Scheme Generation Process
--------------------------

For each region being profiled:

1. **Pattern Identification**: Compute structural pattern signature
2. **Pattern Schemes Initialization**: Create or retrieve ``PatternSchemes`` for pattern
3. **Cache Seeding**: Add schemes from pattern cache (warm-start)
4. **Iterative Generation**: Generate new schemes up to configured limit
   
   * Random selection of insertion points
   * Diversity filtering (avoid duplicates)
   * Pattern-relative addressing

5. **Evaluation**: Each scheme is exported, benchmarked, and ranked
6. **Best Selection**: Scheme with lowest latency becomes pattern's best scheme

Pattern-Relative Addressing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Schemes are defined using pattern-relative indices:

.. code-block:: python

   # Pattern-relative insertion point
   NodeInputInsertionPoint(node_index=2, input_index=0)
   
   # Resolved to actual tensor for Region A
   "conv1_output"  # Node 2 in Region A's pattern
   
   # Resolved to actual tensor for Region B (same pattern)
   "conv5_output"  # Node 2 in Region B's pattern

This portability enables:

* One optimization per pattern instead of per region
* Transfer learning across similar models
* Significant reduction in search space

State Management
----------------

Incremental State Saving
~~~~~~~~~~~~~~~~~~~~~~~~~

State is saved after each region optimization:

**State File Contents (YAML):**

State is saved to ``autotuner_state.yaml`` (or ``--state_file``). The pattern cache is saved alongside as ``<state_base>_pattern_cache.yaml`` (e.g. ``autotuner_state_pattern_cache.yaml``).

.. code-block:: yaml

   baseline_latency_ms: 12.5
   current_profile_pattern_schemes_signature: null   # or pattern sig if interrupted mid-region
   config: { ... }
   patterns:
     - pattern_signature: "abc123..."
       schemes: [...]
       best_scheme_index: 0
     - pattern_signature: "def456..."
       schemes: [...]
       best_scheme_index: 1

**Crash Recovery:**

If optimization is interrupted:

1. Rerun workflow with same output directory
2. State file is automatically detected and loaded
3. Already-profiled patterns are skipped
4. Optimization continues from next unprofiled region

Pattern Cache
-------------

Warm-Start Optimization
~~~~~~~~~~~~~~~~~~~~~~~

Pattern cache files store top-performing schemes:

**Cache File Structure (YAML):**

.. code-block:: yaml

   patterns:
     pattern_def456:
       signature: "def456..."
       schemes:
         - insertion_points: [...]
           latency_ms: 9.8
           distance: 5
         - insertion_points: [...]
           latency_ms: 10.1
           distance: 7
       max_entries: 16

**Usage:**

1. After first optimization, pattern cache saved automatically
2. For similar models, load cache at initialization
3. Cache schemes tested first before random generation
4. Enables faster convergence to optimal solutions

**Diversity Filtering:**

* Schemes are filtered by minimum Hamming distance
* Ensures cache contains diverse candidates
* Prevents redundant similar schemes

Region Discovery Details
========================

Hierarchical Partitioning Strategy
-----------------------------------

The region search algorithm builds a hierarchical tree of regions:

Level 0: Leaf Regions
~~~~~~~~~~~~~~~~~~~~~~

* Individual operations or small operation sequences
* Conv, MatMul, Gemm, Add, etc.
* Forward/backward expansion around seed nodes
* Direct boundary computation

Level 1+: Composite Regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Merging of related leaf regions
* Pattern-based combination (Conv+BN+Relu)
* Sequence merging (Linear→Linear→Linear)
* Hierarchical boundaries (child inputs/outputs)

Region Boundaries
-----------------

Input Tensors
~~~~~~~~~~~~~

Tensors consumed by region nodes but produced outside:

* From model inputs
* From nodes in other regions
* Used to determine Q/DQ insertion at region entry

Output Tensors
~~~~~~~~~~~~~~

Tensors produced by region nodes and consumed outside:

* By nodes in other regions
* As model outputs
* Used to determine Q/DQ insertion at region exit

Boundary Computation Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Collect all tensors consumed by region nodes
2. Filter out tensors produced within region
3. Remaining = input boundary tensors
4. Collect all tensors produced by region nodes
5. Filter out tensors only consumed within region
6. Remaining = output boundary tensors

Insertion Point Selection
=========================

The autotuner uses the same three insertion point types described in **3. Q/DQ Insertion Points** (``NodeInputInsertionPoint``, ``RegionOutputInsertionPoint``, ``ChildRegionInputInsertionPoint``). In practice: **node input** quantization is the most common (e.g. at Conv/MatMul inputs); **region output** and **child region input** quantization apply at composite-region boundaries (e.g. residual connections) and enable hierarchical strategies.

Scheme Generation Strategies
-----------------------------

Random Sampling
~~~~~~~~~~~~~~~

* Randomly select subset of available insertion points
* Probability-based selection (configurable)
* Generates diverse candidate schemes
* Default strategy for exploration

Cache-Guided Sampling
~~~~~~~~~~~~~~~~~~~~~

* When pattern cache available, test cached schemes first
* Provides warm-start for faster convergence
* Falls back to random sampling after cache exhausted

Diversity Filtering
~~~~~~~~~~~~~~~~~~~

* Compute Hamming distance between schemes
* Reject schemes too similar to already-tested ones
* Ensures exploration of diverse configurations
* Minimum distance threshold configurable

Performance Evaluation
======================

TensorRT Engine Building
-------------------------

For each scheme:

1. **ONNX Export**: Generate model with Q/DQ nodes applied
2. **Parser**: TensorRT parses ONNX graph
3. **Optimization**: TensorRT layer fusion, kernel selection
4. **Timing Cache**: Reuse measured kernel timings
5. **Engine Build**: Generate optimized engine binary

Latency Measurement
-------------------

Benchmarking Protocol:

1. **Engine Loading**: Load built engine to GPU
2. **Warmup Phase**: Run N iterations (default: 5)
   
   * Eliminate cold-start effects
   * Prime GPU caches

3. **Timing Phase**: Run M iterations (default: 10-100)
   
   * Measure end-to-end latency per iteration
   * Synchronize GPU after each iteration

4. **Aggregation**: Compute median latency (robust to outliers)

Best Scheme Selection
----------------------

For each pattern:

* Track all successfully-benchmarked schemes
* Rank by measured latency (lower is better)
* Select scheme with minimum latency
* Apply to all regions matching pattern

Usage Patterns
==============

Command-Line Interface
----------------------

**Prog:** ``modelopt.onnx.quantization.autotune``. Arguments use underscores. Short options: ``-m`` (onnx_path), ``-o`` (output_dir), ``-s`` (schemes_per_region), ``-v`` (verbose). Run ``python -m modelopt.onnx.quantization.autotune --help`` for full help.

Command-line arguments
^^^^^^^^^^^^^^^^^^^^^^

**Model and output**

* ``--onnx_path``, ``-m`` (required) — Path to ONNX model file.
* ``--output_dir``, ``-o`` — Output directory for results. Default: ``./autotuner_output``.

**Autotuning strategy**

* ``--schemes_per_region``, ``-s`` — Number of schemes to test per region. Default: ``30``.
* ``--pattern_cache`` — Path to pattern cache YAML for warm-start. Default: ``None``.
* ``--qdq_baseline`` — Path to QDQ baseline ONNX model to import quantization patterns. Default: ``None``.
* ``--state_file`` — State file path for resume. Default: ``<output_dir>/autotuner_state.yaml``.
* ``--node_filter_list`` — Path to file of wildcard patterns (one per line); regions with no matching nodes are skipped. Default: ``None``.

**Quantization**

* ``--quant_type`` — Quantization data type. Choices: ``int8``, ``fp8``. Default: ``int8``.
* ``--default_dq_dtype`` — Default DQ output dtype when not deduced. Choices: ``float16``, ``float32``, ``bfloat16``. Default: ``float32``.

**TensorRT benchmark**

* ``--use_trtexec`` — Use trtexec for benchmarking instead of TensorRT Python API. Default: ``False``.
* ``--timing_cache`` — TensorRT timing cache file path. Default: system temp ``trtexec_timing.cache``.
* ``--warmup_runs`` — Number of warmup runs. Default: ``5``.
* ``--timing_runs`` — Number of timing runs. Default: ``20``.
* ``--plugin_libraries``, ``--plugins`` — TensorRT plugin libraries (``.so``), space-separated. Default: ``None``.
* ``--trtexec_benchmark_args`` — Extra arguments to trtexec as a single quoted string (e.g. ``'--fp16 --workspace=4096'`` or ``'--remoteAutoTuningConfig=...'``). Default: ``None``.

**Logging**

* ``--verbose``, ``-v`` — Enable verbose DEBUG logging.

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   # Default INT8 quantization (output dir default: ./autotuner_output)
   python -m modelopt.onnx.quantization.autotune --onnx_path model.onnx

   # Specify output and FP8 with more schemes
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --output_dir ./output \
       --quant_type fp8 \
       --schemes_per_region 50

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: bash

   # Pattern cache warm-start
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --output_dir ./output \
       --pattern_cache ./previous_run/autotuner_state_pattern_cache.yaml

   # Import patterns from existing QDQ model
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --output_dir ./output \
       --qdq_baseline quantized_baseline.onnx

   # Custom state file and node filter (skip regions with no matching nodes)
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --output_dir ./output \
       --state_file ./output/custom_state.yaml \
       --node_filter_list nodes_to_include.txt

   # Resume after interruption: rerun with same output_dir; state is auto-loaded
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --output_dir ./output

   # Use trtexec and pass extra args (e.g. remote autotuning)
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --output_dir ./output \
       --use_trtexec \
       --trtexec_benchmark_args "--remoteAutoTuningConfig=..."

   # Custom timing cache and DQ dtype
   python -m modelopt.onnx.quantization.autotune \
       --onnx_path model.onnx \
       --output_dir ./output \
       --timing_cache /path/to/cache \
       --default_dq_dtype float16

Python API
----------

High-Level Workflow
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from modelopt.onnx.quantization.autotune.workflows import (
       region_pattern_autotuning_workflow,
   )

   # Pattern-based optimization (recommended). Call init_benchmark_instance first if not using CLI.
   autotuner = region_pattern_autotuning_workflow(
       model_path="model.onnx",
       output_dir=Path("./output"),
       num_schemes_per_region=30,
       pattern_cache_file=None,
       state_file=None,
       quant_type="int8",
       default_dq_dtype="float32",
       qdq_baseline_model=None,
       node_filter_list=None,
       verbose=False,
   )

Low-Level API
~~~~~~~~~~~~~

Initialize the global benchmark with ``init_benchmark_instance``, then use ``benchmark_onnx_model`` for measurements. The workflow uses this same global; when calling the workflow from Python you do not need to call ``init_benchmark_instance`` yourself (the CLI does it).

.. code-block:: python

   import onnx
   from modelopt.onnx.quantization.autotune import QDQAutotuner, Config
   from modelopt.onnx.quantization.autotune.workflows import (
       init_benchmark_instance,
       benchmark_onnx_model,
   )

   # Initialize benchmark (required before benchmark_onnx_model)
   init_benchmark_instance(
       use_trtexec=False,
       timing_cache_file="/tmp/timing.cache",
       warmup_runs=5,
       timing_runs=20,
   )

   # Load and initialize autotuner
   model = onnx.load("model.onnx")
   autotuner = QDQAutotuner(model)
   config = Config(default_quant_type="fp8")
   autotuner.initialize(config)

   # Measure baseline
   autotuner.export_onnx("baseline.onnx", insert_qdq=False)
   baseline_latency = benchmark_onnx_model("baseline.onnx")
   autotuner.submit(baseline_latency)

   # Profile each region
   for region in autotuner.regions:
       autotuner.set_profile_region(region, commit=True)
       for _ in range(30):
           scheme_idx = autotuner.generate()
           if scheme_idx == -1:
               break
           model_bytes = autotuner.export_onnx(None, insert_qdq=True)
           latency = benchmark_onnx_model(model_bytes)
           autotuner.submit(latency, success=(latency != float("inf")))

   # Finalize and export
   autotuner.set_profile_region(None, commit=True)
   autotuner.export_onnx("optimized.onnx", insert_qdq=True)

Design Rationale
================

Pattern-Based Optimization
--------------------------

The autotuner uses a pattern-based optimization approach:

**How It Works:**

* Regions with identical structural patterns are grouped together
* Each unique pattern is optimized once with N schemes tested
* The best scheme for a pattern is automatically applied to all regions matching that pattern
* This dramatically reduces the number of benchmarks required

**Benefits:**

* **Efficiency**: Optimize each unique pattern once instead of every region independently
* **Consistency**: All structurally similar regions use the same quantization strategy
* **Scalability**: Time scales with number of unique patterns, not total regions
* **Transfer Learning**: Pattern cache enables warm-start on similar models

**Trade-offs:**

* Assumes structural similarity implies performance similarity
* May not capture performance variations due to different input/output contexts
* Models with many unique patterns see less benefit

**Best For:**

* Models with repeated structures (transformers, ResNets, etc.)
* Most production models where consistent quantization is desirable
* Scenarios where optimization time is constrained

Forward-Only Region Search
---------------------------

The current implementation focuses on forward (downstream) region expansion:

* Simpler boundary computation
* Aligns with typical dataflow (inputs → outputs)
* Sufficient for most optimization scenarios
* Backward expansion can be added if needed

Hierarchical vs Flat Regions
-----------------------------

Hierarchical region structure provides:

* **Multi-Granularity Optimization**: Can optimize at different abstraction levels
* **Composability**: Child regions can be optimized independently
* **Scalability**: Handles large models by partitioning into manageable pieces
* **Pattern Reuse**: Patterns can be defined at multiple levels

Incremental State Saving
-------------------------

State is saved after each region instead of at the end:

* **Crash Recovery**: Long optimizations (hours/days) can be resumed
* **Early Access**: Partial results available before completion
* **Debugging**: Can inspect intermediate state
* **Resource Management**: Can pause/resume optimization as needed

Limitations and Future Work
============================

Current Limitations
-------------------

1. **Search Space Exploration**
   
   * Random sampling may miss optimal configurations
   * No gradient-based or learned search strategies
   * Number of schemes per region is fixed

2. **Pattern Matching**
   
   * Assumes structural similarity implies performance similarity
   * May miss performance variations due to input data or context

3. **Quantization Types**
   
   * Uniform quantization for all Q/DQ nodes in a scheme
   * No mixed-precision within schemes

4. **Benchmarking Overhead**
   
   * TensorRT engine build time dominates (even with timing cache)
   * Each scheme requires full engine rebuild

5. **Input Sensitivity**
   
   * Performance measured on default/dummy inputs
   * May not generalize to all input distributions

Future Enhancements
-------------------

1. **Advanced Search Strategies**
   
   * Reinforcement learning-based exploration
   * Bayesian optimization for scheme selection
   * Evolutionary algorithms for population-based search

2. **Mixed-Precision Support**
   
   * Different quantization types per insertion point
   * Learnable precision selection
   * Per-layer quantization bit-width

3. **Accuracy Constraint**
   
   * Optimize for latency while maintaining accuracy threshold
   * Multi-objective optimization (latency + accuracy)
   * Accuracy-aware scheme selection and evaluation
   * Integration with calibration and validation datasets
   * Pareto frontier exploration for latency-accuracy trade-offs

Glossary
========

.. glossary::

   Q/DQ Nodes
      QuantizeLinear (Q) and DequantizeLinear (DQ) nodes in ONNX that convert between
      floating-point and quantized integer representations.

   Region
      A hierarchical subgraph in an ONNX computation graph with well-defined input and
      output boundaries. Can be LEAF (atomic), COMPOSITE (containing child regions), or
      ROOT (entire graph).

   Pattern
      A structural signature representing the topology and operation types in a region.
      Regions with identical patterns can share insertion schemes.

   Insertion Scheme
      A collection of insertion points specifying where to insert Q/DQ nodes within a
      region. Schemes use pattern-relative addressing for portability.

   Insertion Point
      A specific location where Q/DQ nodes can be inserted: at a node input, region
      output, or region boundary.

   Pattern-Relative Addressing
      Addressing scheme using indices relative to pattern structure rather than absolute
      graph positions, enabling scheme portability across regions with matching patterns.

   Pattern Cache
      Collection of top-performing insertion schemes for multiple patterns, used to
      warm-start optimization on similar models.

   Baseline Latency
      Inference latency of the original model without any Q/DQ nodes inserted, used as
      reference for measuring optimization improvement.

   TensorRT Timing Cache
      Persistent cache of kernel performance measurements maintained by TensorRT to
      accelerate engine building by reusing previously measured timings.

   Scheme Diversity
      Measure of how different two insertion schemes are, typically computed as Hamming
      distance between their insertion point sets.

References
==========

* **ONNX**: https://onnx.ai/
* **ONNX Technical Details** (numeric types, quantization-related): https://onnx.ai/onnx/technical/index.html
* **TensorRT Documentation**: https://docs.nvidia.com/deeplearning/tensorrt/
* **NVIDIA Model Optimizer (ModelOpt)**: https://github.com/NVIDIA/Model-Optimizer
* **ONNX GraphSurgeon**: https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon
