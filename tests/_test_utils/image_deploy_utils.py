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
"""Deploy published image checkpoints without importing their engines during collection."""

import copy
import fnmatch
import importlib.metadata
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

__all__ = ["ImageModelDeployer"]

_GPU_SKIP_EXIT_CODE = 77


class _UnavailableGPUError(Exception):
    """The visible allocation cannot run this image recipe."""


class ImageModelDeployer:
    """Run a pinned image-generation checkpoint in an isolated engine process."""

    def __init__(
        self,
        model_id,
        backend,
        revision,
        pipeline_class,
        engine_version,
        tensor_parallel_size=1,
        mini_sm=100,
        sm_versions=None,
        engine_kwargs=None,
        sampling_kwargs=None,
        chat_kwargs=None,
        input_config=None,
        task="image-generation",
    ):
        if backend not in {"trtllm", "sglang"} or task != "image-generation":
            raise ValueError(
                "Image deployment requires the TRT-LLM or SGLang image-generation task"
            )
        if pipeline_class != "QwenImagePipeline":
            raise ValueError(f"Unsupported image pipeline: {pipeline_class}")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Image deployment requires a complete pinned Hugging Face revision")
        if chat_kwargs:
            raise ValueError("Image deployment does not accept chat_kwargs")
        if not input_config or set(input_config) != {"prompt"}:
            raise ValueError("Image deployment input_config must contain only a text prompt")
        if not isinstance(input_config["prompt"], str) or not input_config["prompt"].strip():
            raise ValueError("Image deployment requires a nonempty text prompt")
        options = copy.deepcopy(engine_kwargs or {})
        if {"quant_config", "quantization", "component_quantizations"} & options.keys():
            raise ValueError("Load quantization metadata from the published checkpoint")
        if {
            "model",
            "model_path",
            "revision",
            "transformer_weights_path",
            "component_paths",
            "component_weights_paths",
            "server_args",
        } & options.keys():
            raise ValueError("Set the published model and revision on the deployment case")
        if backend == "trtllm":
            parallel = options.setdefault("parallel_config", {})
            if "tp_size" in parallel and parallel["tp_size"] != tensor_parallel_size:
                raise ValueError("parallel_config.tp_size conflicts with tensor_parallel_size")
            parallel["tp_size"] = tensor_parallel_size
        else:
            if options.get("backend", "sglang") != "sglang":
                raise ValueError("SGLang image deployment requires its native diffusion backend")
            if options.setdefault("tp_size", tensor_parallel_size) != tensor_parallel_size:
                raise ValueError("tp_size conflicts with tensor_parallel_size")
            num_gpus = options.setdefault("num_gpus", tensor_parallel_size)
            if type(num_gpus) is not int or num_gpus < 1 or num_gpus % tensor_parallel_size:
                raise ValueError("num_gpus must be a positive multiple of tensor_parallel_size")
        sampling = copy.deepcopy(sampling_kwargs or {})
        count_key = "num_images_per_prompt" if backend == "trtllm" else "num_outputs_per_prompt"
        if sampling.get(count_key) != 1:
            raise ValueError(f"Image deployment requires {count_key}: 1")
        for dimension in ("width", "height"):
            if not isinstance(sampling.get(dimension), int) or sampling[dimension] < 1:
                raise ValueError(f"Image deployment requires a positive integer {dimension}")
        self.config = {
            "model_id": model_id,
            "backend": backend,
            "task": task,
            "revision": revision,
            "pipeline_class": pipeline_class,
            "engine_version": engine_version,
            "tensor_parallel_size": tensor_parallel_size,
            "mini_sm": mini_sm,
            "sm_versions": sm_versions,
            "engine_kwargs": options,
            "sampling_kwargs": sampling,
            "input_config": copy.deepcopy(input_config),
        }

    def run(self, output_dir):
        """Generate and validate an RGB image, retaining configuration, logs, and evidence."""
        # Pytest and the shared runner are needed only in the parent process.
        import pytest

        from .examples.run_command import ProcessGroupCleanupError, _run_capturing

        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        config_path = directory / "deployment-config.json"
        _write_json(
            config_path,
            {
                key: self.config[key]
                for key in (
                    "model_id",
                    "revision",
                    "backend",
                    "pipeline_class",
                    "engine_version",
                    "tensor_parallel_size",
                    "mini_sm",
                    "sm_versions",
                    "task",
                )
            },
        )
        evidence_path = directory / "deployment-result.json"
        _record_result(evidence_path, {"stage": "starting", "passed": False})
        print(f"Image deployment artifacts: {directory}", flush=True)
        try:
            assert os.name == "posix", "Image deployment requires Linux"
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json") as payload:
                json.dump(self.config, payload)
                payload.flush()
                command = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    payload.name,
                    str(directory),
                ]
                # Engines and validation errors may echo private overrides to either stream.
                returncode, _ = _run_capturing(
                    command,
                    Path.cwd(),
                    os.environ.copy(),
                    cleanup_process_group=True,
                    timeout=6900,
                    discard_output=True,
                )
            if returncode == _GPU_SKIP_EXIT_CODE:
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                if evidence.get("stage") == "skipped":
                    pytest.skip(evidence["reason"])
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        except ProcessGroupCleanupError as error:
            try:
                _record_result(
                    evidence_path,
                    {"stage": "cleanup", "passed": False, "subprocess_error": str(error)},
                )
            finally:
                pytest.exit(f"Cannot confirm deployment workers stopped: {error}", returncode=1)
        except (subprocess.SubprocessError, OSError) as error:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence.update(passed=False, subprocess_error=str(error))
            _record_result(evidence_path, evidence)
            print(f"Image deployment failed: {json.dumps(evidence)}", flush=True)
            raise
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        assert evidence.get("passed") is True, f"Image deployment did not pass: {evidence}"


def _record_request(directory, sampling):
    fields = (
        "width",
        "height",
        "num_inference_steps",
        "guidance_scale",
        "seed",
        "num_images_per_prompt",
        "num_outputs_per_prompt",
    )
    public = {name: sampling[name] for name in fields if type(sampling.get(name)) in (int, float)}
    _write_json(directory / "deployment-request.json", public)


def _record_result(path, evidence):
    _write_json(path, evidence)
    mode = "w" if evidence["stage"] == "starting" else "a"
    with path.with_name("deployment.log").open(mode, encoding="utf-8") as log:
        print(json.dumps(evidence), file=log)


def _write_json(path, value):
    # A killed worker must leave its last complete evidence available to the parent.
    pending = path.with_suffix(f".{os.getpid()}.tmp")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def _check_static_nvfp4(snapshot, directory):
    """Check published Linear weights and calibrated scales using headers, not tensor payloads."""
    component = snapshot / "transformer"
    config = json.loads((component / "config.json").read_text(encoding="utf-8"))
    _write_json(directory / "transformer-config.json", config)
    quant = config.get("quantization_config", {})
    assert quant.get("quant_algo") == "NVFP4", f"Expected published NVFP4 metadata, got {quant}"
    groups = quant.get("config_groups", {})
    assert groups and all(
        group.get("targets") == ["Linear"]
        and group.get("weights", {}).get("dynamic") is False
        and group.get("input_activations", {}).get("dynamic") is False
        for group in groups.values()
    ), "Published NVFP4 must contain static Linear weights and input activations"
    # The engine opens only the shards its weight index names; an unreferenced shard
    # cannot satisfy the scale inventory. Without an index every shard is active.
    indexes = sorted(component.glob("*.safetensors.index.json"))
    assert len(indexes) <= 1, f"Multiple weight indexes under {component}"
    shards = set(component.glob("*.safetensors"))
    if indexes:
        weight_map = json.loads(indexes[0].read_text(encoding="utf-8"))["weight_map"]
        named = {component / name for name in weight_map.values()}
    else:
        named = shards
    entries = {}
    for path in sorted(named):
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            assert header_size <= path.stat().st_size - 8, f"Invalid safetensors header: {path}"
            header = json.loads(stream.read(header_size))
        entries.update(
            {key: {**value, "file": path} for key, value in header.items() if key != "__metadata__"}
        )
    packed = sorted(
        name.removesuffix(".weight")
        for name, tensor in entries.items()
        if name.endswith(".weight")
        and tensor["dtype"] == "U8"
        and len(tensor["shape"]) == 2
        and not any(
            fnmatch.fnmatchcase(name.removesuffix(".weight"), pattern)
            for pattern in quant.get("ignore", [])
        )
    )
    missing = {
        suffix: [prefix for prefix in packed if f"{prefix}.{suffix}" not in entries]
        for suffix in ("weight_scale", "weight_scale_2", "input_scale")
    }
    invalid = []
    if not any(missing.values()):
        # Scales are small tensors: read their values and require finite, positive data.
        import torch
        from safetensors import safe_open

        for prefix in packed:
            dims = len(entries[f"{prefix}.weight"]["shape"])
            for suffix in ("weight_scale", "weight_scale_2", "input_scale"):
                key = f"{prefix}.{suffix}"
                with safe_open(str(entries[key]["file"]), framework="pt") as shard:
                    value = shard.get_tensor(key)
                if (
                    (suffix == "weight_scale" and value.ndim != dims)
                    or (suffix != "weight_scale" and value.numel() != 1)
                    or not (torch.isfinite(value).all() and (value > 0).all())
                ):
                    invalid.append(key)
    _write_json(
        directory / "static-activation-scales.json",
        {
            "packed_modules": packed,
            "missing_scales": missing,
            "invalid_scales": invalid,
            "unreferenced_shards": sorted(path.name for path in shards - named),
        },
    )
    assert packed, "Published checkpoint contains no packed NVFP4 Linear modules"
    assert not any(missing.values()), f"Published NVFP4 checkpoint is missing scales: {missing}"
    assert not invalid, f"Published NVFP4 checkpoint has invalid scales: {invalid}"


def _generate_sglang(config, snapshot, directory):
    # SGLang is optional and must not be imported in a TRT-LLM image.
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

    request = {
        **config["sampling_kwargs"],
        "prompt": config["input_config"]["prompt"],
        "output_path": str(directory),
        "output_file_name": "output.png",
        "save_output": True,
    }
    _record_request(directory, request)
    with DiffGenerator.from_pretrained(
        model_path=str(snapshot), **config["engine_kwargs"]
    ) as engine:
        runtime = {
            name: getattr(engine.server_args, name)
            for name in ("num_gpus", "tp_size", "sp_degree")
            if type(getattr(engine.server_args, name)) is int
        }
        runtime["pipeline_config_class"] = type(engine.server_args.pipeline_config).__name__
        _write_json(directory / "deployment-runtime.json", runtime)
        result = engine.generate(sampling_params_kwargs=request)
        assert result is not None and not isinstance(result, list), (
            "SGLang must return exactly one generated image"
        )
        assert result.output_file_path, "SGLang generated no image file"
        return Path(result.output_file_path), request["width"], request["height"]


def _check_gpu_allocation(cuda, config, required_gpus, gpus):
    visible = cuda.device_count()
    if visible < required_gpus:
        raise _UnavailableGPUError(
            f"Configured image parallelism needs {required_gpus} GPUs, got {visible}"
        )
    for index in range(len(gpus), required_gpus):
        properties = cuda.get_device_properties(index)
        sm = properties.major * 10 + properties.minor
        gpus.append({"name": properties.name, "sm": sm, "memory_bytes": properties.total_memory})
        if sm < config["mini_sm"]:
            raise _UnavailableGPUError(f"GPU {index} requires SM{config['mini_sm']}, got SM{sm}")
        if config["sm_versions"] and sm not in config["sm_versions"]:
            raise _UnavailableGPUError(
                f"GPU {index} SM{sm} is outside configured candidates {config['sm_versions']}"
            )


def _deploy(config, directory, evidence):
    # Engine and CUDA imports are confined to the child, keeping collection lightweight.
    import torch

    assert os.name == "posix", "Image deployment requires Linux"
    if not torch.cuda.is_available():
        raise _UnavailableGPUError("Image deployment requires a CUDA GPU")
    evidence["gpus"] = []
    backend = config["backend"]
    options = config["engine_kwargs"]
    required_gpus = options["num_gpus"] if backend == "sglang" else config["tensor_parallel_size"]
    _check_gpu_allocation(torch.cuda, config, required_gpus, evidence["gpus"])
    package = "tensorrt_llm" if backend == "trtllm" else "sglang"
    version = importlib.metadata.version(package)
    evidence["engine_version"] = version
    assert version.split("+")[0] == config["engine_version"], (
        f"Expected {backend} {config['engine_version']}, got {version}"
    )
    if backend == "trtllm":
        # TRT-LLM is optional and must not be imported in a SGLang image.
        from tensorrt_llm import VisualGen, VisualGenArgs

        unknown = options.keys() - VisualGenArgs.model_fields.keys()
        if unknown:
            raise ValueError(f"Unknown VisualGen engine parameters: {sorted(unknown)}")
        runtime_args = VisualGenArgs(**options)
        # VisualGen can allocate additional workers for CFG and Ulysses parallelism.
        _check_gpu_allocation(
            torch.cuda, config, runtime_args.parallel_config.n_workers, evidence["gpus"]
        )

    from huggingface_hub import snapshot_download
    from PIL import Image

    evidence["stage"] = "download"
    _record_result(directory / "deployment-result.json", evidence)
    snapshot = Path(snapshot_download(repo_id=config["model_id"], revision=config["revision"]))
    evidence.update(stage="checkpoint", snapshot=str(snapshot))
    model_index = json.loads((snapshot / "model_index.json").read_text(encoding="utf-8"))
    _write_json(directory / "model_index.json", model_index)
    assert model_index.get("_class_name") == config["pipeline_class"], model_index
    _check_static_nvfp4(snapshot, directory)
    evidence["stage"] = "generation"
    _record_result(directory / "deployment-result.json", evidence)
    if backend == "sglang":
        output_path, width, height = _generate_sglang(config, snapshot, directory)
    else:
        runtime_args.model = str(snapshot)
        _write_json(
            directory / "deployment-runtime.json",
            runtime_args.model_dump(
                mode="json", include={"parallel_config": {"tp_size", "cfg_size", "ulysses_size"}}
            ),
        )
        # No quant_config override: VisualGen loads the checkpoint's static NVFP4 metadata.
        with VisualGen(model=str(snapshot), args=runtime_args) as engine:
            params = engine.default_params
            unknown = config["sampling_kwargs"].keys() - type(params).model_fields.keys()
            if unknown:
                raise ValueError(f"Unknown VisualGen sampling parameters: {sorted(unknown)}")
            validated = type(params).model_validate(config["sampling_kwargs"])
            for name in config["sampling_kwargs"]:
                setattr(params, name, getattr(validated, name))
            _record_request(directory, params.model_dump(mode="json"))
            result = engine.generate(inputs=config["input_config"]["prompt"], params=params)
            assert result.error is None, f"TRT-LLM image generation failed: {result.error}"
            assert result.image is not None, "TRT-LLM generated no image"
            expected_shape = (1, params.height, params.width, 3)
            assert tuple(result.image.shape) == expected_shape, (
                f"Expected image shape {expected_shape}, got {result.image.shape}"
            )
            output_path = result.save(directory / "output.png")
            width, height = params.width, params.height
    assert output_path.is_file(), f"{backend} did not produce {output_path}"
    with Image.open(output_path) as image:
        image.load()
        assert image.mode == "RGB", f"Expected RGB image, got {image.mode}"
        assert image.size == (width, height), image.size
    evidence.update(stage="passed", passed=True, output=str(output_path))


if __name__ == "__main__":
    config_path = Path(sys.argv[1])
    directory = Path(sys.argv[2])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    evidence = {
        "model_id": config["model_id"],
        "backend": config["backend"],
        "revision": config["revision"],
        "stage": "environment",
        "passed": False,
    }
    try:
        _deploy(config, directory, evidence)
    except _UnavailableGPUError as error:
        evidence.update(stage="skipped", reason=str(error))
        sys.exit(_GPU_SKIP_EXIT_CODE)
    except Exception as error:
        evidence["error_type"] = type(error).__name__
        raise
    finally:
        _record_result(directory / "deployment-result.json", evidence)
