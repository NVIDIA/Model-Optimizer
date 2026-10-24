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

import subprocess
import sys

# Top-level modules of the optional ``[hf]`` extra, plus ``requests`` which it only pulls in
# transitively. A base ``pip install nvidia-modelopt`` provides none of them.
_HF_EXTRA_MODULES = [
    "accelerate",
    "datasets",
    "deepspeed",
    "diffusers",
    "httpx",
    "huggingface_hub",
    "nltk",
    "peft",
    "requests",
    "sentencepiece",
    "tiktoken",
    "transformers",
    "wonderwords",
]


def test_modelopt_torch_imports_without_hf_extra():
    # A None entry in sys.modules makes importing that module raise ModuleNotFoundError.
    code = f"import sys; sys.modules.update(dict.fromkeys({_HF_EXTRA_MODULES!r})); import modelopt.torch"
    subprocess.run([sys.executable, "-c", code], check=True)
