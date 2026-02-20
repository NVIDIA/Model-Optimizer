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

"""The package setup script for modelopt customizing certain aspects of the installation process."""

import setuptools
from setuptools_scm import get_version

# TODO: Set fallback_version to X.Y.Z release version when creating the release branch
version = get_version(root=".", fallback_version="0.0.0")

# Required and optional dependencies ###############################################################
required_deps = [
    # Common
    "ninja",  # for faster building of C++ / CUDA extensions
    "numpy",
    "packaging",
    "pydantic>=2.0",
    "nvidia-ml-py>=12",
    "rich",
    "scipy",
    "tqdm",
    # modelopt.torch
    "pulp",
    "regex",
    "safetensors",
    "torch>=2.6",
]

optional_deps = {
    "onnx": [
        "cppimport",
        "cupy-cuda12x; platform_machine != 'aarch64' and platform_system != 'Darwin'",
        "lief",
        "ml_dtypes",  # for bfloat16 conversion
        "onnx-graphsurgeon",
        "onnx~=1.19.0",
        "onnxconverter-common~=1.16.0",
        "onnxruntime~=1.22.0 ; platform_machine == 'aarch64' or platform_system == 'Darwin'",
        "onnxruntime-gpu~=1.22.0 ; platform_machine != 'aarch64' and platform_system != 'Darwin'",
        "onnxscript",  # For autocast opset conversion and test_onnx_dynamo_export unit test
        "onnxslim>=0.1.76",
        "polygraphy>=0.49.22",
    ],
    "hf": [
        "accelerate>=1.0.0",
        "datasets>=3.0.0",
        "deepspeed>=0.9.6 ; platform_system != 'Darwin' and platform_system != 'Windows'",
        "diffusers>=0.32.2",
        "huggingface_hub>=0.24.0",
        "peft>=0.17.0",
        "transformers>=4.53,<5.0",  # Should match modelopt/torch/__init__.py and tox.ini
        "nltk",
        "wonderwords",
    ],
    # linter tools
    "dev-lint": [
        "bandit[toml]==1.7.9",  # security/compliance checks
        "mypy==1.17.1",
        "pre-commit==4.3.0",
        "ruff==0.12.11",
    ],
    # testing
    "dev-test": [
        "coverage",
        "pytest",
        "pytest-cov",
        "pytest-instafail",
        "pytest-timeout",
        "sentencepiece",  # For test_unified_export_megatron.py, test_vllm_fakequant_megatron_export.py
        "timm",
        "torchprofile>=0.0.4",  # For computing flops of CV models
        "torchvision",
        "torch-geometric",
        "tox>4.18",
        "tox-current-env>=0.0.12",
    ],
    # docs
    "dev-docs": [
        "autodoc_pydantic>=2.1.0",
        "sphinx~=8.1.0",
        "sphinx-argparse>=0.5.2",
        "sphinx-autobuild>=2024.10.3",
        "sphinx-copybutton>=0.5.2",
        "sphinx-inline-tabs>=2023.4.21",
        "sphinx-rtd-theme~=3.0.0",  # 3.0 does not show version, which we want as Linux & Windows have separate releases
        "sphinx-togglebutton>=0.3.2",
    ],
    # build/packaging tools
    "dev-build": [
        "cython",
        "setuptools>=80",
        "setuptools-scm>=8",
    ],
}

# create "compound" optional dependencies
optional_deps["all"] = [
    deps for k in optional_deps if not k.startswith("dev") for deps in optional_deps[k]
]
optional_deps["dev"] = [deps for k in optional_deps for deps in optional_deps[k]]


if __name__ == "__main__":
    setuptools.setup(
        name="nvidia-modelopt",
        version=version,
        description="Nvidia Model Optimizer: a unified model optimization and deployment toolkit.",
        long_description="Checkout https://github.com/nvidia/Model-Optimizer for more information.",
        long_description_content_type="text/markdown",
        author="NVIDIA Corporation",
        url="https://github.com/NVIDIA/Model-Optimizer",
        license="Apache 2.0",
        license_files=("LICENSE_HEADER",),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        python_requires=">=3.10,<3.13",
        install_requires=required_deps,
        extras_require=optional_deps,
        packages=setuptools.find_namespace_packages(include=["modelopt*"]),
        package_dir={"": "."},
        package_data={"modelopt": ["**/*.h", "**/*.cpp", "**/*.cu"]},
    )
