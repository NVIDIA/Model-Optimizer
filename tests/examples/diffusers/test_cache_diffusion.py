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

import subprocess
import sys

import pytest
import torch
from _test_utils.examples.models import PIXART_PATH, SXDL_PATH
from _test_utils.examples.run_command import MODELOPT_ROOT
from diffusers import DiffusionPipeline, PixArtAlphaPipeline

sys.path.append(str(MODELOPT_ROOT / "examples/diffusers/cache_diffusion"))
from cache_diffusion import cachify
from cache_diffusion.utils import PIXART_DEFAULT_CONFIG, SDXL_DEFAULT_CONFIG


def test_sdxl_cachify():
    pipe = DiffusionPipeline.from_pretrained(
        SXDL_PATH,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    cachify.prepare(pipe, SDXL_DEFAULT_CONFIG)

    prompt = "A random person with a head that is made of flowers, photo by James C. Leyendecker, \
            Afrofuturism, studio portrait, dynamic pose, national geographic photo, retrofuturism, biomorphicy"
    generator = torch.Generator(device="cuda").manual_seed(2946901)
    pipe(prompt=prompt, generator=generator, num_inference_steps=30).images[0]
    # Clear cuda memory as pytest doesnt clear it between tests
    del pipe
    torch.cuda.empty_cache()


def test_pixart_cachify():
    # Fail test if apex is installed
    if "apex" in subprocess.check_output(["pip", "list"]).decode("utf-8"):
        pytest.xfail("Apex is installed, test is expected to fail")

    pipe = PixArtAlphaPipeline.from_pretrained(PIXART_PATH, torch_dtype=torch.float16).to("cuda")
    cachify.prepare(pipe, PIXART_DEFAULT_CONFIG)

    prompt = "a small cactus with a happy face in the Sahara desert"
    generator = torch.Generator(device="cuda").manual_seed(2946901)
    pipe(prompt=prompt, generator=generator, num_inference_steps=30).images[0]
    # Clear cuda memory as pytest doesnt clear it between tests
    del pipe
    torch.cuda.empty_cache()


def test_sdxl_benchmarks(tmp_path):
    # fmt: off
    subprocess.run(
        [
            "python", "benchmarks.py",
            "--model-id", "sdxl",
            "--batch-size", "1",
            "--num-iter", "2",
            "--output-dir", tmp_path,
        ],
        cwd=MODELOPT_ROOT / "examples/diffusers/cache_diffusion",
        check=True,
    )
    # fmt: on
