# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pattern-Based Q/DQ Autotuning for ONNX Models.

This package provides automated optimization of Quantize/Dequantize (Q/DQ) node placement
in ONNX computation graphs to minimize TensorRT inference latency. It uses pattern-based
region analysis to efficiently explore and optimize Q/DQ insertion strategies.

**Key Features:**

- **Automated Region Discovery**: Hierarchical decomposition of computation graphs into
  LEAF and COMPOSITE regions with automatic pattern identification

- **Pattern-Based Optimization**: Groups structurally-similar regions and optimizes them
  together, making the process efficient and consistent

- **TensorRT Performance Measurement**: Direct integration with TensorRT Python API for
  accurate latency profiling of each Q/DQ configuration

- **State Management**: Checkpoint/resume capability for long-running optimizations with
  incremental state saving after each region

- **Pattern Cache**: Warm-start optimization using learned schemes from previous runs,
  enabling transfer learning across models

**Core Components:**

Autotuner Classes:
    - QDQAutotuner: Main autotuner with automatic hierarchical region discovery
    - QDQAutotunerBase: Base class for custom region identification strategies

Region Management:
    - Region: Hierarchical subgraph representation (nodes + children)
    - RegionType: Enumeration (LEAF, COMPOSITE, ROOT)
    - CombinedRegionSearch: Two-phase region discovery (partitioning + refinement)
    - RegionPattern: Structural pattern analysis and matching for region grouping

Q/DQ Insertion Points:
    - InsertionScheme: Collection of Q/DQ insertion points for a region pattern
    - NodeInputInsertionPoint: Q/DQ insertion at specific node inputs
    - ChildRegionInputInsertionPoint: Q/DQ insertion at child region input boundaries
    - RegionOutputInsertionPoint: Q/DQ insertion at region output boundaries

Configuration & State:
    - Config: Autotuning parameters (quant type, thresholds, verbosity)
    - PatternCache: Top-performing schemes indexed by pattern (warm-start)
    - PatternSchemes: Scheme collection and measurement results for a pattern

Benchmarking:
    - Benchmark: Abstract base class for model benchmarking
    - TensorRTPyBenchmark: Benchmark using TensorRT Python API (recommended)
    - TrtExecBenchmark: Benchmark using trtexec command-line tool (legacy)

**Quick Start:**

    >>> from modelopt.onnx.quantization.autotune import QDQAutotuner, Config
    >>> import onnx
    >>> # Load model and initialize autotuner
    >>> model = onnx.load("model.onnx")
    >>> autotuner = QDQAutotuner(model)
    >>> # Configure autotuning parameters
    >>> config = Config(default_quant_type="int8")
    >>> autotuner.initialize(config)
    >>> # Generate and test Q/DQ schemes
    >>> # (see workflows.region_pattern_autotuning_workflow for complete example)

**Command-Line Interface:**

    The package can be run directly as a module:

    $ python -m modelopt.onnx.quantization.autotune --model model.onnx --output ./output
    $ python -m modelopt.onnx.quantization.autotune --model model.onnx --quant-type fp8

**See Also:**

    - workflows.region_pattern_autotuning_workflow: Complete end-to-end optimization
    - QDQAutotuner: Main autotuner class documentation
    - RegionPattern: Pattern matching and signature computation
"""

# Autotuner classes
from .autotuner import QDQAutotuner, QDQAutotunerBase

# Benchmark classes
from .benchmark import Benchmark, TensorRTPyBenchmark, TrtExecBenchmark

# Core data structures
from .common import (
    AutotunerError,
    AutotunerNotInitializedError,
    Config,
    InsertionScheme,
    InvalidSchemeError,
    PatternCache,
    PatternSchemes,
    Region,
    RegionError,
    RegionType,
)

# Insertion points (from dedicated module)
from .insertion_points import (
    ChildRegionInputInsertionPoint,
    NodeInputInsertionPoint,
    RegionOutputInsertionPoint,
    ResolvedInsertionPoint,
)

# Pattern analysis
from .region_pattern import RegionPattern

# Region search
from .region_search import CombinedRegionSearch

# Public API
__all__ = [
    # Exceptions
    "AutotunerError",
    "AutotunerNotInitializedError",
    # Benchmark classes
    "Benchmark",
    "TensorRTPyBenchmark",
    "TrtExecBenchmark",
    # Configuration and state
    "Config",
    # Q/DQ insertion
    "InsertionScheme",
    "InvalidSchemeError",
    "NodeInputInsertionPoint",
    "ChildRegionInputInsertionPoint",
    "RegionOutputInsertionPoint",
    "ResolvedInsertionPoint",
    # Main autotuner classes
    "QDQAutotuner",
    "QDQAutotunerBase",
    # Region classes
    "Region",
    "RegionError",
    "RegionPattern",
    "RegionType",
    "PatternCache",
    "PatternSchemes",
    "CombinedRegionSearch",
]
