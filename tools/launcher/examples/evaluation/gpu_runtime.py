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

"""Check the serving image's GPU runtime before loading checkpoints or clients."""

import argparse
import importlib
import importlib.metadata
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


def main():
    """Retain driver/runtime details and real FP32/BF16 linear checks on one GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--suite", choices=("basic", "serving"), default="basic")
    args = parser.parse_args()
    result: dict[str, Any] = {
        "status": "failed",
        "node": socket.gethostname(),
        "checks": {},
        "suite": args.suite,
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING", "0"),
        "compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
    }
    started = time.monotonic()
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        result["driver_versions"] = sorted(set(driver.stdout.splitlines()))
        # Torch is provided by the serving image; keep --help usable without it.
        import torch

        result.update(torch_version=torch.__version__, torch_cuda_version=torch.version.cuda)
        count = torch.cuda.device_count()
        result["gpu_count"] = count
        if count != 1:
            raise ValueError(f"Expected one visible GPU, found {count}")
        result.update(
            gpu_name=torch.cuda.get_device_name(0),
            capability=list(torch.cuda.get_device_capability(0)),
        )
        cases = [
            ("torch.float32", torch.float32, 512, 4096),
            ("torch.bfloat16", torch.bfloat16, 512, 4096),
        ]
        if args.suite == "serving":
            cases += [
                ("lm_head_8", torch.bfloat16, 8, 151936),
                ("lm_head_8192", torch.bfloat16, 8192, 151936),
                ("mlp_8192", torch.bfloat16, 8192, 28672),
            ]
        for name, dtype, rows, columns in cases:
            result["checks"][name] = "running"
            x = torch.ones((rows, 4096), device="cuda", dtype=dtype)
            weight = torch.ones((columns, 4096), device="cuda", dtype=dtype)
            output = torch.nn.functional.linear(x, weight)
            torch.cuda.synchronize()
            minimum, maximum = torch.aminmax(output)
            if minimum.item() != 4096 or maximum.item() != 4096:
                raise ValueError(f"Incorrect linear result for {name}")
            result["checks"][name] = "passed"
            del x, weight, output
        if args.suite == "serving":
            result["vllm_version"] = importlib.metadata.version("vllm")
            result["checks"]["vllm_rms_norm"] = "running"
            ops = importlib.import_module("vllm._custom_ops")
            x = torch.ones((8192, 4096), device="cuda", dtype=torch.bfloat16)
            weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
            output = torch.empty_like(x)
            ops.rms_norm(output, x, weight, 1e-6)
            torch.cuda.synchronize()
            if not torch.all(output == 1).item():
                raise ValueError("Incorrect vLLM RMS normalization")
            result["checks"]["vllm_rms_norm"] = "passed"
            result["checks"]["compiled_rms_norm"] = "running"
            normalized = torch.compile(torch.nn.functional.rms_norm, fullgraph=True)
            output = normalized(x, (4096,), weight, 1e-6)
            torch.cuda.synchronize()
            if not torch.all(output == 1).item():
                raise ValueError("Incorrect compiled RMS normalization")
            result["checks"]["compiled_rms_norm"] = "passed"
        result["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
        result["status"] = "passed"
    except Exception as error:
        result["checks"] = {
            name: "failed" if state == "running" else state
            for name, state in result["checks"].items()
        }
        result["error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if os.environ.get("MLFLOW_RUN_ID"):
            # The internal tracking profile supplies this optional dependency.
            from mlflow.tracking import MlflowClient

            MlflowClient().log_artifact(
                os.environ["MLFLOW_RUN_ID"], str(args.output), artifact_path="runtime"
            )


if __name__ == "__main__":
    main()
