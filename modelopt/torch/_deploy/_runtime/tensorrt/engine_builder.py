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

import logging
import shutil
import subprocess  # nosec
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory, gettempdir

from ..._runtime.common import read_bytes, timeit, write_bytes, write_string
from ..._runtime.tensorrt.layerwise_profiling import process_layerwise_result
from ...utils import OnnxBytes
from .constants import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_NUM_INFERENCE_PER_RUN,
    SHA_256_HASH_LENGTH,
    TRT_MODE_FLAGS,
    TRTEXEC_PATH,
    WARMUP_TIME_MS,
    TRTMode,
)
from .tensorrt_utils import convert_shape_to_string, prepend_hash_to_bytes

logging.basicConfig(
    level=logging.INFO,
    format="[modelopt][_deploy] - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


# TODO: Get rid of this function or get approval for `# nosec` usage if we want to include this
#   as a non-compiled python file in the release.
def _run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int, bytes]:
    """Util function to execute a command.

    This util will not direct stdout and stderr to console if the cmd succeeds.

    Args:
        cmd: the command line list
        cwd: current working directory

    Returns:
        return code: 0 means successful, otherwise means failed
        log_string: the stdout and stderr output as a string

    """
    logging.info(" ".join(cmd))
    with NamedTemporaryFile("w+b") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=str(cwd) if cwd else None)  # nosec
        p.wait()
        log.seek(0)
        output = log.read()
        if p.returncode != 0:
            logging.error(output.decode(errors="ignore"))
        return p.returncode, output


def _get_profiling_params(profiling_runs: int) -> list[str]:
    return [
        f"--warmUp={WARMUP_TIME_MS}",
        f"--avgRuns={DEFAULT_NUM_INFERENCE_PER_RUN}",
        f"--iterations={profiling_runs * DEFAULT_NUM_INFERENCE_PER_RUN}",
        "--noDataTransfers",
        "--useCudaGraph",
        "--useSpinWait",
    ]


def _get_trtexec_params(
    engine_path: Path,
    builder_optimization_level: str,
    timing_cache_file: Path | None = None,
    verbose: bool = False,
) -> list[str]:
    cmd = [
        f"--saveEngine={engine_path}",
        f"--builderOptimizationLevel={builder_optimization_level}",
        "--skipInference",
    ]

    if timing_cache_file:
        cmd.append(f"--timingCacheFile={timing_cache_file}")

    if verbose:
        cmd.append("--verbose")

    return cmd


def _is_low_bit_mode(trt_mode: str) -> bool:
    return trt_mode in [
        TRTMode.INT8,
        TRTMode.INT4,
        TRTMode.FLOAT8,
        TRTMode.BEST,
        TRTMode.STRONGLY_TYPED,
    ]


def _update_dynamic_shapes(dynamic_shapes: dict, cmd: list[str]) -> None:
    """Update the dynamic shapes in the command list."""
    shapes = ["minShapes", "optShapes", "maxShapes"]
    for shape in shapes:
        if shape in dynamic_shapes:
            cmd.extend([f"--{shape}={convert_shape_to_string(dynamic_shapes[shape])}"])


@timeit
def build_engine(
    onnx_bytes: OnnxBytes,
    trt_mode: str = TRTMode.FLOAT32,
    engine_path: Path | None = None,
    timing_cache_path: Path | None = None,
    calib_cache: str | None = None,
    dynamic_shapes: dict | None = None,
    plugin_config: dict | None = None,
    builder_optimization_level: str = "3",
    output_dir: Path | None = None,
    verbose: bool = False,
) -> tuple[bytes | None, bytes]:
    """This method produces serialized TensorRT engine from an ONNX model.

    Args:
        onnx_bytes: Data of the ONNX model stored as an OnnxBytes object.
        engine_path: Path to save the TensorRT engine.
        timing_cache_path: Path to save/load the TensorRT timing cache.
        trt_mode: The precision with which the TensorRT engine will be built. Supported modes are:
            - TRTMode.FLOAT32
            - TRTMode.FLOAT16
            - TRTMode.BFLOAT16
            - TRTMode.INT8
            - TRTMode.FLOAT8
            - TRTMode.INT4
            - TRTMode.STRONGLY_TYPED
            - TRTMode.BEST
            For more details about TensorRT modes, please refer to:
            https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-flags
        calib_cache: Calibration data cache.
        dynamic_shapes: Dictionary of dynamic shapes for the input tensors. Example is as follows:
            {
                "minShapes": {"input": [1,3,244,244]},
                "optShapes": {"input": [16,3,244,244]},
                "maxShapes": {"input": [32,3,244,244]}
            }
        plugin_config: Dictionary of plugin configurations. Example is as follows:
            {
                "staticPlugins": ["staticPluginOne.so", "staticPluginTwo.so", ...],
                "dynamicPlugins": ["dynamicPluginOne.so", "dynamicPluginTwo.so", ...],
                "setPluginsToSerialize": ["dynamicPluginOne.so", "pluginSerializeOne.so", ...],
                "ignoreParsedPluginLibs": False
            }
        builder_optimization_level: Optimization level for the TensorRT builder.
            For more details, please refer to:
            https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Core/BuilderConfig.html
        output_dir: Directory to save the engine file and trtexec artifacts.
            If not provided, the artifacts will be saved in a temporary directory.
        verbose: Set trtexec command to verbose or not.

    Returns:
        The generated engine file data.
        The stdout log produced by trtexec tool. \
            [If there is subprocess.CalledProcessError, this byte variable is transferred to str]
    """

    def _build_command(
        onnx_path: Path,
        engine_path: Path,
        calib_cache_path: Path | None = None,
        timing_cache_path: Path | None = None,
    ) -> list[str]:
        cmd = [TRTEXEC_PATH, f"--onnx={onnx_path}"]
        cmd.extend(TRT_MODE_FLAGS[trt_mode])

        if trt_mode == TRTMode.INT8 and calib_cache and calib_cache_path:
            write_string(calib_cache, calib_cache_path)
            cmd.append(f"--calib={calib_cache_path}")

        if dynamic_shapes:
            _update_dynamic_shapes(dynamic_shapes, cmd)

        if plugin_config:
            for key, plugins in plugin_config.items():
                if key == "ignoreParsedPluginLibs":
                    if plugins:
                        cmd.append("--ignoreParsedPluginLibs")
                elif isinstance(plugins, list):
                    cmd.extend([f"--{key}={plugin}" for plugin in plugins])

        opt_level = "4" if _is_low_bit_mode(trt_mode) else builder_optimization_level
        cmd += _get_trtexec_params(engine_path, opt_level, timing_cache_path, verbose=verbose)

        return cmd

    def _setup_files_and_paths(
        tmp_dir_path: Path,
        engine_path: Path | None,
        timing_cache_path: Path | None,
    ) -> tuple[Path, Path, Path | None, Path | None, Path]:
        tmp_onnx_dir = tmp_dir_path / "onnx"
        onnx_bytes.write_to_disk(str(tmp_onnx_dir))
        onnx_path = tmp_onnx_dir / f"{onnx_bytes.model_name}.onnx"

        final_output_dir = Path(output_dir or Path(gettempdir()) / DEFAULT_ARTIFACT_DIR)
        final_output_dir.mkdir(parents=True, exist_ok=True)
        engine_path = (
            Path(engine_path)
            if engine_path
            else final_output_dir / f"{onnx_bytes.model_name}.engine"
        )
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        calib_cache_path = final_output_dir / "calib_cache" if calib_cache else None
        timing_cache_path = (
            Path(timing_cache_path) if timing_cache_path else final_output_dir / "timing.cache"
        )

        return onnx_path, engine_path, calib_cache_path, timing_cache_path, final_output_dir

    with TemporaryDirectory() as tmp_dir:
        onnx_path, engine_path, calib_cache_path, timing_cache_path, final_output_dir = (
            _setup_files_and_paths(Path(tmp_dir), engine_path, timing_cache_path)
        )
        cmd = _build_command(onnx_path, engine_path, calib_cache_path, timing_cache_path)

        try:
            ret_code, out = _run_command(cmd)
            if ret_code != 0:
                return None, out

            engine_bytes = prepend_hash_to_bytes(read_bytes(engine_path))
            engine_hash = engine_bytes[:SHA_256_HASH_LENGTH].hex()[:8]
            engine_path_with_hash = (
                final_output_dir / f"{engine_hash}-{onnx_bytes.model_name}.engine"
            )
            shutil.move(engine_path, engine_path_with_hash)
            logging.info("Engine saved to %s", engine_path_with_hash)

            return engine_bytes, out
        except Exception as e:
            logging.exception(str(e))
            return None, str(e).encode()


@timeit
def profile_engine(
    engine_bytes: bytes,
    profiling_runs: int = 1,
    onnx_node_names: list[str] | None = None,
    enable_layerwise_profiling: bool = False,
    dynamic_shapes: dict | None = None,
    output_dir: Path | None = None,
) -> tuple[dict[str, float] | None, bytes]:
    """This method profiles a TensorRT engine and returns the detailed results.

    Args:
        engine_bytes: Bytes of the serialized TensorRT engine, prepended with a SHA256 hash.
        onnx_node_names: List of node names in the onnx model.
        profiling_runs: number of profiling runs. Each run runs `DEFAULT_NUM_INFERENCE_PER_RUN` inferences.
        enable_layerwise_profiling:
            True or False based on whether layerwise profiling is required or not.
        dynamic_shapes: Dictionary of dynamic shapes for the input tensors. Example is as follows:
            {
                "minShapes": {"input": [1,3,244,244]},
                "optShapes": {"input": [16,3,244,244]},
                "maxShapes": {"input": [32,3,244,244]}
            }
        output_dir: Directory to save the engine file and trtexec artifacts.
            If not provided, the artifacts will be saved in a temporary directory.
    Returns:
        Layerwise profiling output as a json string.
        Stdout log produced by trtexec tool.
    """

    def _build_command(engine_path: Path, profile_path: Path, layer_info_path: Path) -> list[str]:
        cmd = [TRTEXEC_PATH, f"--loadEngine={engine_path}"]
        cmd += _get_profiling_params(profiling_runs)

        if enable_layerwise_profiling:
            cmd += [
                "--dumpProfile",
                "--separateProfileRun",
                f"--exportProfile={profile_path}",
                "--profilingVerbosity=detailed",
                f"--exportLayerInfo={layer_info_path}",
            ]

        if dynamic_shapes:
            _update_dynamic_shapes(dynamic_shapes, cmd)

        return cmd

    def _setup_files_and_paths(tmp_dir_path: Path, engine_hash: str) -> tuple[Path, Path, Path]:
        engine_path = tmp_dir_path / f"{engine_hash}-model.engine"
        write_bytes(engine_bytes[SHA_256_HASH_LENGTH:], engine_path)

        final_output_dir = Path(output_dir or Path(gettempdir()) / DEFAULT_ARTIFACT_DIR)
        final_output_dir.mkdir(parents=True, exist_ok=True)
        profile_path = final_output_dir / f"{engine_hash}-profile.json"
        layer_info_path = final_output_dir / f"{engine_hash}-layerInfo.json"

        return engine_path, profile_path, layer_info_path

    with TemporaryDirectory() as tmp_dir:
        engine_hash = engine_bytes[:SHA_256_HASH_LENGTH].hex()[:8]
        engine_path, profile_path, layer_info_path = _setup_files_and_paths(
            Path(tmp_dir), engine_hash
        )
        cmd = _build_command(engine_path, profile_path, layer_info_path)

        try:
            ret_code, out = _run_command(cmd)
            if ret_code != 0:
                return None, out

            layerwise_results = {}
            if enable_layerwise_profiling and profile_path.exists():
                layerwise_results = process_layerwise_result(profile_path, onnx_node_names)

            logging.info("Profile saved to %s", profile_path)
            logging.info("LayerInfo saved to %s", layer_info_path)
            return layerwise_results, out

        except Exception as e:
            logging.exception(str(e))
            return None, str(e).encode()
