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

"""Check isolated PTQ package provenance, imports, and CUDA libraries."""

import argparse
import contextlib
import ctypes
import importlib
import importlib.metadata
import json
import pathlib
import platform
import shutil
import struct
import subprocess  # nosec B404: fixed argument lists for local provenance tools.
import sys


def validate_cusparselt_sbsa(distribution, machine):
    """Recognize the ARM64 0.8.1 wheel's invalid SBSA platform tag."""
    tags = [
        line
        for line in (distribution.read_text("WHEEL") or "").splitlines()
        if line.startswith("Tag:")
    ]
    library = pathlib.Path(distribution.locate_file("nvidia/cusparselt/lib/libcusparseLt.so.0"))
    with library.open("rb") as stream:
        header = stream.read(64)
    if (
        machine != "aarch64"
        or distribution.version != "0.8.1"
        or tags != ["Tag: py3-none-manylinux2014_sbsa"]
        or header[:6] != b"\x7fELF\x02\x01"
        or struct.unpack_from("<H", header, 18)[0] != 183
    ):
        raise RuntimeError("cuSPARSELt does not match the known ARM64 SBSA tag defect")
    return {
        "package": "nvidia-cusparselt-cu13",
        "version": distribution.version,
        "wheel_tags": tags,
        "elf_machine": 183,
        "library": str(library),
        "reason": "ARM64 binary, nonstandard SBSA wheel tag; native loading and GPU gates still required",
    }


def check_packages(allow_cusparselt_sbsa=False):
    """Check dependencies, with an explicit opt-in for one verified vendor tag defect."""
    result = subprocess.run(  # nosec B603: current interpreter, fixed pip check command.
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, shell=False
    )
    if result.returncode == 0:
        return {"status": "passed", "returncode": 0, "vendor_exceptions": []}
    known = "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform"
    if not allow_cusparselt_sbsa or result.stdout.strip() != known or result.stderr.strip():
        raise RuntimeError(f"pip check failed: {result.stdout}{result.stderr}")
    exception = validate_cusparselt_sbsa(
        importlib.metadata.distribution("nvidia-cusparselt-cu13"), platform.machine()
    )
    return {
        "status": "vendor_tag_exception",
        "returncode": result.returncode,
        "raw_output": result.stdout.strip(),
        "vendor_exceptions": [exception],
    }


def nvfp4_quantizer():
    """Expand preset overrides into the complete quantizer attribute schema."""
    from modelopt.torch.quantization.config import NVFP4_DEFAULT_CFG, QuantizerAttributeConfig
    from modelopt.torch.quantization.nn import TensorQuantizer

    weight_cfg = next(
        entry["cfg"]
        for entry in NVFP4_DEFAULT_CFG["quant_cfg"]
        if entry["quantizer_name"] == "*weight_quantizer"
    )
    return TensorQuantizer(QuantizerAttributeConfig(**weight_cfg))


def verify(
    source, ref, require_cuda=False, model_class="AutoModelForCausalLM", allow_cusparselt_sbsa=False
):
    """Return runtime evidence; raise on inconsistent provenance or imports."""
    source = pathlib.Path(source).resolve()
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required to verify source provenance")
    actual_ref = subprocess.check_output(  # nosec B603: trusted Git, fixed read-only arguments.
        [git, "-c", f"safe.directory={source}", "rev-parse", "HEAD"],
        cwd=source,
        text=True,
        shell=False,
    ).strip()
    if actual_ref != ref:
        raise RuntimeError(f"Source revision {actual_ref} differs from {ref}")
    prefix = pathlib.Path(sys.prefix).resolve()
    if (
        sys.prefix == sys.base_prefix
        or "include-system-site-packages = false" not in (prefix / "pyvenv.cfg").read_text()
    ):
        raise RuntimeError("PTQ requires a venv without system-site-packages")
    package_check = check_packages(allow_cusparselt_sbsa)
    distributions = sorted(
        (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()
    )
    modelopt_distributions = [
        item for item in distributions if "modelopt" in item[0].lower().replace("_", "-")
    ]
    if len(modelopt_distributions) != 1 or modelopt_distributions[0][0] != "nvidia-modelopt":
        raise RuntimeError(f"Ambiguous ModelOpt distributions: {modelopt_distributions}")
    imports = {}
    for name in (
        "torch",
        "transformers",
        "modelopt",
        "modelopt.torch.utils.plugins.hf_checkpoint_utils",
        "torchvision",
        "accelerate",
        "datasets",
        "deepspeed",
        "diffusers",
        "peft",
        "compressed_tensors",
        "fire",
        "mlflow",
        "zstandard",
    ):
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise RuntimeError(f"Missing import provenance: {name}")
        path = pathlib.Path(module.__file__).resolve()
        if not path.is_relative_to(prefix):
            raise RuntimeError(f"{name} imported outside the environment: {path}")
        imports[name] = str(path)
    # Compare installed Python source with the exact checkout, including the prior failing module.
    for name, path in imports.items():
        if name.startswith("modelopt"):
            local = source / pathlib.Path(path).relative_to(
                prefix
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            )
            if pathlib.Path(path).read_bytes() != local.read_bytes():
                raise RuntimeError(f"Installed source differs: {name}")
    import torch  # Deferred until provenance validation.

    # Resolve the requested lazy model import after validating package provenance.
    model_type = getattr(importlib.import_module("transformers"), model_class)

    libraries = {}
    for filename in ("libcudart.so.13", "libnvrtc.so.13", "libcublas.so.13", "libcusparseLt.so.0"):
        candidates = list(prefix.rglob(filename)) + list(
            pathlib.Path("/usr/local/cuda").rglob(filename)
        )
        if not candidates:
            raise RuntimeError(f"Missing native development/runtime library: {filename}")
        path = candidates[0].resolve()
        ctypes.CDLL(str(path))
        libraries[filename] = str(path)
    quantizer = nvfp4_quantizer()
    # Check a defaulted attribute on CPU so incomplete preset dictionaries fail before CUDA.
    if not isinstance(quantizer.rotate_back_is_enabled, bool):
        raise RuntimeError("Invalid quantizer rotation defaults")
    gpu = None
    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable on target hardware")
        tensor = torch.randn((64, 64), dtype=torch.bfloat16, device="cuda")
        result = tensor @ tensor.T
        torch.cuda.synchronize()
        if not torch.isfinite(result).all().item():
            raise RuntimeError("CUDA BF16 matmul produced non-finite output")
        quantized = quantizer.cuda()(tensor)
        torch.cuda.synchronize()
        if not torch.isfinite(quantized).all().item():
            raise RuntimeError("NVFP4 quantization produced non-finite output")
        gpu = {
            "name": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "compiled_architectures": torch.cuda.get_arch_list(),
            "bf16_matmul": "passed",
            "nvfp4_quantizer": "passed",
        }
    return {
        "source_ref": actual_ref,
        "python": sys.executable,
        "python_version": sys.version,
        "prefix": str(prefix),
        "torch": torch.__version__,
        "torch_distribution_version": importlib.metadata.version("torch"),
        "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "distributions": distributions,
        "imports": imports,
        "native_libraries": libraries,
        "gpu": gpu,
        "model_class": model_type.__name__,
        "pip_check": package_check,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--ref")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--model-class", default="AutoModelForCausalLM")
    parser.add_argument("--allow-cusparselt-sbsa", action="store_true")
    parser.add_argument("--check-packages-only", action="store_true")
    args = parser.parse_args()
    if not args.check_packages_only and (not args.source or not args.ref):
        parser.error("--source and --ref are required for environment verification")
    with contextlib.redirect_stdout(sys.stderr):
        result = (
            check_packages(args.allow_cusparselt_sbsa)
            if args.check_packages_only
            else verify(
                args.source,
                args.ref,
                args.require_cuda,
                args.model_class,
                args.allow_cusparselt_sbsa,
            )
        )
    print(json.dumps(result, indent=2))
