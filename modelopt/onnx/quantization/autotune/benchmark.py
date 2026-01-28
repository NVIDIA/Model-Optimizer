#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""TensorRT Utilities and Benchmark Module.

This module provides comprehensive TensorRT utilities including:
- Benchmark framework for measuring TensorRT engine performance
- Graph utilities for tensor analysis

**Benchmark Classes:**
- Benchmark: Abstract base class defining the benchmarking interface
- TrtExecBenchmark: Uses trtexec command-line tool for benchmarking
- TensorRTPyBenchmark: Uses TensorRT Python API for direct engine profiling

**Features:**
- Timing cache management for faster subsequent builds
- File path or raw bytes as model input
- Configurable warmup and timing iterations
- Custom TensorRT plugin library loading
- Automatic cleanup of temporary resources
"""

import ctypes
import os
import re
import shutil
import subprocess  # nosec B404
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

# Optional dependencies - gracefully handle missing packages
try:
    import tensorrt as trt

    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import pycuda.autoinit  # noqa: F401  # Automatically initializes CUDA (side-effect import)
    import pycuda.driver as cuda

    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False

from modelopt.onnx.logging_config import logger

# =============================================================================
# Benchmark Framework
# =============================================================================


class Benchmark(ABC):
    """Abstract base class for TensorRT model benchmarking.

    This class defines the interface that all benchmark implementations must follow.
    It provides a consistent API for measuring inference latency of ONNX models
    when converted to TensorRT engines.

    Attributes:
        timing_cache_file: Path to the TensorRT timing cache file.
        logger: Logger instance for this benchmark.

    Subclasses must implement:
        run(): Execute the benchmark and return latency in milliseconds.
    """

    def __init__(self, timing_cache_file: str | None = None):
        """Initialize the benchmark.

        Args:
            timing_cache_file: Path to timing cache file to accelerate engine builds.
                             If None, uses '/tmp/trtexec_timing.cache' as default.
        """
        global logger
        self.timing_cache_file = timing_cache_file or "/tmp/trtexec_timing.cache"  # nosec B108
        self.logger = logger

    @abstractmethod
    def run(self, path_or_bytes: str | bytes, log_file: str | None = None) -> float:
        """Run benchmark on the given ONNX model.

        Args:
            path_or_bytes: Path to the ONNX model (str) or raw model data (bytes)
            log_file: Optional path to save benchmark logs

        Returns:
            Measured latency in milliseconds, or float("inf") on failure
        """

    def __call__(self, path_or_bytes: str | bytes, log_file: str | None = None) -> float:
        """Convenience method to call benchmark as a function.

        Args:
            path_or_bytes: Path to the ONNX model (str) or raw model data (bytes)
            log_file: Optional path to save benchmark logs

        Returns:
            Measured latency in milliseconds
        """
        return self.run(path_or_bytes, log_file)


class TrtExecBenchmark(Benchmark):
    """TensorRT benchmark using trtexec command-line tool.

    This implementation uses the trtexec binary to build engines and measure
    inference latency. It is the most straightforward method and closely
    mirrors standard TensorRT workflows.

    Features:
        - Uses subprocess to call trtexec binary
        - Supports all trtexec command-line arguments
        - Custom TensorRT plugin library loading
        - Automatic temporary directory management for engines
        - Timing cache persistence across benchmarks
        - Supports both file paths and raw bytes as input

    Attributes:
        trtexec_path: Path to the trtexec binary.
        trtexec_args: Additional command-line arguments for trtexec.
        warmup_runs: Number of warmup iterations before timing.
        timing_runs: Number of iterations for latency measurement.
        timeout: Maximum time in seconds for trtexec execution.
        plugin_libraries: List of paths to plugin libraries.
        engine_dir: Directory for storing temporary engine files.
        engine_path: Path to the engine file.
        temp_model_path: Path for temporary ONNX model (when using bytes).
    """

    def __init__(
        self,
        trtexec_path: str = "trtexec",
        trtexec_args: list | None = None,
        timing_cache_file: str | None = None,
        warmup_runs: int = 5,
        timing_runs: int = 10,
        timeout: int = 300,
        plugin_libraries: list[str] | None = None,
    ):
        """Initialize the trtexec benchmark.

        Args:
            trtexec_path: Path to trtexec binary. Defaults to 'trtexec' which
                         looks for the binary in PATH.
            trtexec_args: Additional command-line arguments to pass to trtexec.
                         These are appended after the standard arguments.
                         Example: ['--fp16', '--workspace=4096', '--verbose']
            timing_cache_file: Path to TensorRT timing cache file for faster
                              subsequent builds. Defaults to '/tmp/trtexec_timing.cache'.
            warmup_runs: Number of warmup iterations before timing measurements.
            timing_runs: Number of iterations for latency measurement. Results
                        are averaged across these runs.
            timeout: Maximum time in seconds for trtexec execution before timeout.
            plugin_libraries: List of paths to TensorRT plugin shared libraries (.so files).
                             These plugins will be loaded by trtexec during engine building.
                             If None, no custom plugins are loaded.
        """
        super().__init__(timing_cache_file)

        # Store configuration
        self.trtexec_path = trtexec_path
        self.trtexec_args = trtexec_args or []
        self.warmup_runs = warmup_runs
        self.timing_runs = timing_runs
        self.timeout = timeout
        self.plugin_libraries = plugin_libraries or []

        # Create persistent temporary directory for engine and model files
        # This directory persists for the lifetime of this benchmark object
        self._temp_dir = tempfile.mkdtemp(prefix="trtexec_benchmark_")
        self.engine_dir = self._temp_dir
        self.engine_path = os.path.join(self.engine_dir, "engine.trt")
        self.temp_model_path = os.path.join(self.engine_dir, "temp_model.onnx")
        self.logger.debug(f"Created temporary engine directory: {self.engine_dir}")
        self.logger.debug(f"Temporary model path: {self.temp_model_path}")

        # Construct base trtexec command template
        # The '--onnx' argument will be added dynamically in run()
        self._base_cmd = [
            self.trtexec_path,
            f"--avgRuns={self.timing_runs}",
            f"--iterations={self.timing_runs}",
            f"--warmUp={self.warmup_runs}",
            "--stronglyTyped",  # Enable strongly typed mode for Q/DQ ops
            f"--saveEngine={self.engine_path}",
            f"--timingCacheFile={self.timing_cache_file}",
        ]

        # Add plugin libraries
        for plugin_lib in self.plugin_libraries:
            plugin_path = Path(plugin_lib).resolve()
            if not plugin_path.exists():
                self.logger.warning(f"Plugin library not found: {plugin_path}")
            else:
                self._base_cmd.append(f"--staticPlugins={plugin_path}")
                self.logger.debug(f"Added plugin library: {plugin_path}")

        # Append user-provided custom arguments
        if self.trtexec_args:
            self._base_cmd.extend(self.trtexec_args)

        self.logger.debug(f"Base command template: {' '.join(self._base_cmd)}")

    def __del__(self):
        """Cleanup temporary directory."""
        if hasattr(self, "_temp_dir"):
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
                self.logger.debug(f"Cleaned up temporary directory: {self._temp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temporary directory: {e}")

    def run(
        self,
        path_or_bytes: str | bytes,
        log_file: str | None = None,
        flush_timing_cache: bool = False,
    ) -> float:
        """Run benchmark using trtexec.

        Args:
            path_or_bytes: Path to the ONNX model (str) or raw model data (bytes)
            log_file: Optional path to save trtexec logs

        Returns:
            Measured median latency in milliseconds
        """
        cache_exists = os.path.exists(self.timing_cache_file)
        if cache_exists:
            self.logger.debug(f"Using existing timing cache: {self.timing_cache_file}")
        else:
            self.logger.debug(f"Will create timing cache: {self.timing_cache_file}")

        try:
            # If bytes provided, write to temporary model path
            if isinstance(path_or_bytes, bytes):
                with open(self.temp_model_path, "wb") as f:
                    f.write(path_or_bytes)
                model_path = self.temp_model_path
                self.logger.debug(f"Wrote model bytes to temporary file: {model_path}")
            else:
                model_path = path_or_bytes

            # Build complete command from base template
            cmd = [self._base_cmd[0], f"--onnx={model_path}", *self._base_cmd[1:]]

            self.logger.debug(f"Running: {' '.join(cmd)}")

            # Run trtexec and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)  # nosec B603

            # Save logs if requested
            if log_file is not None:
                try:
                    log_path = Path(log_file)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, "w") as f:
                        f.write(f"Command: {' '.join(cmd)}\n")
                        f.write(f"Return code: {result.returncode}\n")
                        f.write("=" * 80 + "\n")
                        f.write("STDOUT:\n")
                        f.write("=" * 80 + "\n")
                        f.write(result.stdout)
                        f.write("\n" + "=" * 80 + "\n")
                        f.write("STDERR:\n")
                        f.write("=" * 80 + "\n")
                        f.write(result.stderr)
                    self.logger.debug(f"Saved trtexec logs to: {log_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to save logs to {log_file}: {e}")

            if result.returncode != 0:
                self.logger.error(f"trtexec failed with return code {result.returncode}")
                self.logger.error(f"stderr: {result.stderr}")
                return float("inf")

            # Parse output to extract latency
            # trtexec outputs lines like:
            # "[I] Latency: min = X ms, max = Y ms, mean = Z ms, median = W ms, ..."
            output = result.stdout

            # Look for median latency in the main "[I] Latency:" line
            pattern = r"\[I\]\s+Latency:.*?median\s*=\s*([\d.]+)\s*ms"

            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                latency = float(match.group(1))
                self.logger.info(f"TrtExec benchmark (median): {latency:.2f} ms")
                return latency

            self.logger.warning("Could not parse median latency from trtexec output")
            self.logger.debug(f"trtexec stdout:\n{output}")
            return float("inf")

        except subprocess.TimeoutExpired:
            self.logger.error(f"trtexec timed out after {self.timeout} seconds")
            return float("inf")
        except FileNotFoundError:
            self.logger.error(f"trtexec binary not found: {self.trtexec_path}")
            self.logger.error("Please ensure TensorRT is installed and trtexec path is correct")
            return float("inf")
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            return float("inf")


class TensorRTPyBenchmark(Benchmark):
    """TensorRT benchmark using Python API with plugin support.

    This implementation directly uses the TensorRT Python API to build engines
    and measure inference latency. It provides more control than trtexec and
    can be faster for certain workflows as it avoids subprocess overhead.

    Features:
        - Direct TensorRT Python API usage (no subprocess)
        - Persistent Builder, Logger, and Runtime objects
        - Custom TensorRT plugin library loading
        - Automatic dynamic shape handling
        - In-memory timing cache management
        - CUDA memory management via PyCUDA
        - Detailed latency statistics (min, max, mean, median)

    Requirements:
        - tensorrt package
        - pycuda package
        - CUDA-capable GPU

    Attributes:
        trt_logger: TensorRT Logger instance (persistent).
        builder: TensorRT Builder instance (persistent).
        runtime: TensorRT Runtime instance (persistent).
        config: Builder configuration (recreated per run).
        warmup_runs: Number of warmup iterations.
        timing_runs: Number of timing iterations.
        plugin_libraries: List of loaded plugin library paths.
        _shape_configs: Dictionary storing custom shape configurations.
        _plugin_registry: TensorRT PluginRegistry instance.

    Methods:
        set_shapes(): Configure min/opt/max shapes for dynamic inputs.
        run(): Execute the benchmark and return latency.
    """

    def __init__(
        self,
        timing_cache_file: str | None = None,
        warmup_runs: int = 5,
        timing_runs: int = 20,
        plugin_libraries: list[str] | None = None,
    ):
        """Initialize the TensorRT Python API benchmark.

        Creates persistent TensorRT objects (Logger, Builder, Runtime) and
        loads the timing cache from disk if available. Optionally loads custom
        TensorRT plugin libraries for models with custom operations.

        Args:
            timing_cache_file: Path to TensorRT timing cache file. If None,
                              defaults to '/tmp/trtexec_timing.cache'.
            warmup_runs: Number of warmup iterations before timing measurements.
            timing_runs: Number of iterations for latency measurement.
            plugin_libraries: List of paths to TensorRT plugin shared libraries (.so files).
                             These plugins will be loaded and registered for use during
                             engine building. If None, no custom plugins are loaded.

        Raises:
            ImportError: If tensorrt or pycuda packages are not available.
            FileNotFoundError: If a specified plugin library file does not exist.
            RuntimeError: If plugin library loading fails.
        """
        super().__init__(timing_cache_file)
        self.warmup_runs = warmup_runs
        self.timing_runs = timing_runs
        self.plugin_libraries = plugin_libraries or []

        if not TRT_AVAILABLE:
            raise ImportError("TensorRT Python API not available. Please install tensorrt package.")
        if not PYCUDA_AVAILABLE:
            raise ImportError("PyCUDA not available. Please install pycuda package.")

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.trt_logger)
        self.runtime = trt.Runtime(self.trt_logger)
        # Load custom plugin libraries before initializing TensorRT plugins
        self._loaded_plugin_handles = []
        if self.plugin_libraries:
            self._load_plugin_libraries()
        # Get plugin registry (must be done after loading plugin libraries)
        trt.init_libnvinfer_plugins(self.trt_logger, "")

        self._plugin_registry = trt.get_plugin_registry()

        # Set network flag
        self.network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

        # Load timing cache from disk or create new one
        self._timing_cache = None
        self._load_timing_cache()

        # Storage for user-defined shape configurations
        # Format: {input_name: (min_shape, opt_shape, max_shape)}
        self._shape_configs = {}

    def _load_plugin_libraries(self):
        """Load custom TensorRT plugin libraries from shared object files.

        This method loads plugin libraries using ctypes and initializes them
        with the TensorRT plugin registry. Plugins must export the
        initLibNvInferPlugins function to register their implementations.

        The loaded library handles are stored to prevent them from being
        garbage collected during the benchmark's lifetime.

        Raises:
            FileNotFoundError: If a plugin library file does not exist.
            RuntimeError: If plugin initialization fails.
        """
        for plugin_lib in self.plugin_libraries:
            plugin_path = Path(plugin_lib).resolve()

            if not plugin_path.exists():
                raise FileNotFoundError(f"Plugin library not found: {plugin_path}")

            self.logger.info(f"Loading TensorRT plugin: {plugin_path}")

            try:
                if hasattr(os, "RTLD_LAZY") and hasattr(os, "RTLD_GLOBAL"):
                    plugin_handle = ctypes.CDLL(
                        str(plugin_path), mode=os.RTLD_LAZY | os.RTLD_GLOBAL
                    )
                else:
                    # Fallback for platforms without RTLD flags (e.g., Windows)
                    plugin_handle = ctypes.CDLL(str(plugin_path))

                # Store handle to prevent garbage collection
                self._loaded_plugin_handles.append(plugin_handle)

                # Try to initialize plugin with TensorRT registry
                # Most TensorRT plugins export initLibNvInferPlugins function
                if hasattr(plugin_handle, "initLibNvInferPlugins"):
                    init_func = plugin_handle.initLibNvInferPlugins
                    # Function signature: bool initLibNvInferPlugins(void* logger, const char* namespace)
                    init_func.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
                    init_func.restype = ctypes.c_bool

                    # Initialize with the TensorRT logger and default namespace
                    success = init_func(None, b"")
                    if not success:
                        self.logger.warning(
                            f"Plugin initialization returned false for: {plugin_path}"
                        )
                    else:
                        self.logger.info(f"Successfully initialized plugin: {plugin_path.name}")
                else:
                    self.logger.info(
                        f"Plugin loaded (no initLibNvInferPlugins function): {plugin_path.name}"
                    )

            except Exception as e:
                raise RuntimeError(f"Failed to load plugin library {plugin_path}: {e}") from e

    def set_shapes(self, input_name: str, min_shape: list, opt_shape: list, max_shape: list):
        """Set custom min/opt/max shapes for a dynamic input.

        This method allows you to specify custom shape ranges for dynamic inputs
        (inputs with -1 dimensions). If not specified, the benchmark will use
        default shapes (all -1 dimensions become 1).

        Args:
            input_name: Name of the input tensor to configure.
            min_shape: Minimum shape for this input. List of integers.
            opt_shape: Optimal/default shape for this input. List of integers.
            max_shape: Maximum shape for this input. List of integers.
        """
        if len(min_shape) != len(opt_shape) or len(opt_shape) != len(max_shape):
            raise ValueError("min_shape, opt_shape, and max_shape must have the same length")

        for i, (min_dim, opt_dim, max_dim) in enumerate(zip(min_shape, opt_shape, max_shape)):
            if not (min_dim <= opt_dim <= max_dim):
                raise ValueError(
                    f"Invalid shape range at dimension {i}: "
                    f"min={min_dim}, opt={opt_dim}, max={max_dim}. "
                    f"Must satisfy min <= opt <= max"
                )

        self._shape_configs[input_name] = (min_shape, opt_shape, max_shape)
        self.logger.debug(
            f"Set shapes for input '{input_name}': "
            f"min={min_shape}, opt={opt_shape}, max={max_shape}"
        )

    def run(
        self,
        path_or_bytes: str | bytes,
        log_file: str | None = None,
        flush_timing_cache: bool = False,
    ) -> float:
        """Run benchmark using TensorRT Python API.

        Args:
            path_or_bytes: Path to the ONNX model (str) or raw model data (bytes)
            log_file: Optional path to save benchmark logs

        Returns:
            Measured median latency in milliseconds
        """
        config = None
        network = None
        parser = None
        serialized_engine = None
        engine = None
        context = None
        inputs = []
        outputs = []
        stream = None

        try:
            self.logger.debug("Creating TensorRT builder...")
            config = self.builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.DIRECT_IO)
            if not config.set_timing_cache(self._timing_cache, ignore_mismatch=True):
                self.logger.warning("Failed to set timing cache to builder config")
            network = self.builder.create_network(self.network_flags)
            # Create network and parser using the shared builder and logger
            parser = trt.OnnxParser(network, self.trt_logger)

            # Parse ONNX model
            if isinstance(path_or_bytes, bytes):
                self.logger.debug(f"Parsing ONNX model from bytes (size: {len(path_or_bytes)})")
                model_data = path_or_bytes
            else:
                self.logger.debug(f"Parsing ONNX model: {path_or_bytes}")
                with open(path_or_bytes, "rb") as f:
                    model_data = f.read()

            if not parser.parse(model_data):
                self.logger.error("Failed to parse ONNX model")
                for error_idx in range(parser.num_errors):
                    self.logger.error(f"  {parser.get_error(error_idx)}")
                return float("inf")

            has_dynamic_shapes = False
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                shape = input_tensor.shape
                if any(dim == -1 for dim in shape):
                    has_dynamic_shapes = True
                    break

            if has_dynamic_shapes:
                profile = self.builder.create_optimization_profile()
                for i in range(network.num_inputs):
                    input_tensor = network.get_input(i)
                    input_name = input_tensor.name
                    shape = list(input_tensor.shape)

                    # Check if user provided custom shape configuration
                    if input_name in self._shape_configs:
                        min_shape, opt_shape, max_shape = self._shape_configs[input_name]
                        self.logger.debug(
                            f"Using custom shapes for input '{input_name}': "
                            f"min={min_shape}, opt={opt_shape}, max={max_shape}"
                        )
                    else:
                        # Use default: replace -1 with concrete values (1)
                        min_shape = [1 if dim == -1 else dim for dim in shape]
                        opt_shape = [1 if dim == -1 else dim for dim in shape]
                        max_shape = [1 if dim == -1 else dim for dim in shape]
                        self.logger.debug(
                            f"Using default shapes for input '{input_name}': {opt_shape}"
                        )

                    profile.set_shape(input_name, min_shape, opt_shape, max_shape)

                config.add_optimization_profile(profile)

            self.logger.debug("Building TensorRT engine...")
            build_start = time.perf_counter()
            serialized_engine = self.builder.build_serialized_network(network, config)
            build_time = time.perf_counter() - build_start

            if serialized_engine is None:
                self.logger.error("Failed to build TensorRT engine")
                return float("inf")

            self.logger.debug(f"Engine built successfully in {build_time:.2f}s")

            if flush_timing_cache:
                self._save_timing_cache()

            engine = self.runtime.deserialize_cuda_engine(serialized_engine)

            if engine is None:
                self.logger.error("Failed to deserialize engine")
                return float("inf")

            context = engine.create_execution_context()

            inputs = []
            outputs = []

            for i in range(engine.num_io_tensors):
                tensor_name = engine.get_tensor_name(i)
                dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
                shape = context.get_tensor_shape(tensor_name)

                size = trt.volume(shape)
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    np.copyto(host_mem, np.random.randn(size).astype(dtype))
                    inputs.append({"host": host_mem, "device": device_mem, "name": tensor_name})
                else:
                    outputs.append({"host": host_mem, "device": device_mem, "name": tensor_name})

                context.set_tensor_address(tensor_name, int(device_mem))

            stream = cuda.Stream()

            self.logger.debug(f"Running {self.warmup_runs} warmup iterations...")
            for _ in range(self.warmup_runs):
                for inp in inputs:
                    cuda.memcpy_htod_async(inp["device"], inp["host"], stream)
                context.execute_async_v3(stream_handle=stream.handle)
                for out in outputs:
                    cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
                stream.synchronize()

            self.logger.debug(f"Running {self.timing_runs} timing iterations...")
            latencies = []

            for _ in range(self.timing_runs):
                for inp in inputs:
                    cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

                stream.synchronize()
                start = time.perf_counter()
                context.execute_async_v3(stream_handle=stream.handle)
                stream.synchronize()
                end = time.perf_counter()

                latency_ms = (end - start) * 1000.0
                latencies.append(latency_ms)

                for out in outputs:
                    cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

            latencies = np.array(latencies)
            median_latency = float(np.median(latencies))
            mean_latency = float(np.mean(latencies))
            std_latency = float(np.std(latencies))
            min_latency = float(np.min(latencies))
            max_latency = float(np.max(latencies))

            self.logger.info("TensorRT Python API benchmark:")
            self.logger.info(
                f"  min={min_latency:.3f}ms, max={max_latency:.3f}ms, "
                f"mean={mean_latency:.3f}ms, std={std_latency:.3f}ms, median={median_latency:.3f}ms"
            )

            if log_file is not None:
                try:
                    log_path = Path(log_file)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    model_info = (
                        f"<bytes, size={len(path_or_bytes)}>"
                        if isinstance(path_or_bytes, bytes)
                        else path_or_bytes
                    )
                    with open(log_path, "w") as f:
                        f.write(f"""TensorRT Python API Benchmark
Model: {model_info}
Build time: {build_time:.2f}s
Warmup runs: {self.warmup_runs}
Timing runs: {self.timing_runs}

Latency Statistics:
  Min:    {min_latency:.3f} ms
  Max:    {max_latency:.3f} ms
  Mean:   {mean_latency:.3f} ms
  Std:    {std_latency:.3f} ms
  Median: {median_latency:.3f} ms

All latencies: {latencies.tolist()}
""")
                    self.logger.debug(f"Saved benchmark logs to: {log_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to save logs to {log_file}: {e}")
            return median_latency
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}", exc_info=True)
            return float("inf")
        finally:
            try:
                for inp in inputs:
                    if "device" in inp:
                        inp["device"].free()
                    if "host" in inp:
                        del inp["host"]
                for out in outputs:
                    if "device" in out:
                        out["device"].free()
                    if "host" in out:
                        del out["host"]
                inputs.clear()
                outputs.clear()

                if context is not None:
                    del context
                if stream is not None:
                    del stream
                if engine is not None:
                    del engine
                if serialized_engine is not None:
                    del serialized_engine
                if parser is not None:
                    del parser
                if network is not None:
                    del network
                if config is not None:
                    del config
            except Exception as cleanup_error:
                self.logger.warning(f"Error during cleanup: {cleanup_error}")

    def _load_timing_cache(self):
        """Load timing cache from file or create a new one."""
        config = self.builder.create_builder_config()
        if os.path.exists(self.timing_cache_file):
            try:
                with open(self.timing_cache_file, "rb") as f:
                    timing_cache_data = f.read()
                    self._timing_cache = config.create_timing_cache(timing_cache_data)
                    self.logger.debug(f"Loaded timing cache from: {self.timing_cache_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load timing cache: {e}")
                self.logger.debug("Creating new timing cache")
                self._timing_cache = None

        if self._timing_cache is None:
            self._timing_cache = config.create_timing_cache(b"")
            self.logger.debug("Created new timing cache")
        del config

    def _save_timing_cache(self):
        """Save timing cache to file."""
        try:
            if self._timing_cache is not None:
                config = self.builder.create_builder_config()
                output_cache = config.create_timing_cache(b"")
                if self._timing_cache is None:
                    output_cache.combline(self._timing_cache, ignore_errors=True)
                timing_cache_data = output_cache.serialize()
                with open(self.timing_cache_file, "wb") as f:
                    f.write(timing_cache_data)
                self.logger.debug(f"Saved timing cache to: {self.timing_cache_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save timing cache: {e}")
        finally:
            del config
