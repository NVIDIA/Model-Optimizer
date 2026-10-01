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
from unittest.mock import patch

import pytest
import torch
from _test_utils.examples.models import TINY_SDXL_PATH
from _test_utils.examples.run_command import MODELOPT_ROOT
from _test_utils.torch.diffusers_models import get_tiny_pixart_pipeline
from diffusers import DiffusionPipeline

sys.path.append(str(MODELOPT_ROOT / "examples/diffusers/cache_diffusion"))
from cache_diffusion import cachify
from cache_diffusion.module import CachedModule
from cache_diffusion.utils import PIXART_DEFAULT_CONFIG, SDXL_DEFAULT_CONFIG


def test_sdxl_cachify():
    pipe = DiffusionPipeline.from_pretrained(
        TINY_SDXL_PATH,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    cachify.prepare(pipe, SDXL_DEFAULT_CONFIG)

    prompt = "A random person with a head that is made of flowers, photo by James C. Leyendecker, \
            Afrofuturism, studio portrait, dynamic pose, national geographic photo, retrofuturism, biomorphicy"
    generator = torch.Generator(device="cuda").manual_seed(2946901)
    # 8 steps still exercises the step-modulo cache pattern; this is a runs-without-error smoke test.
    pipe(prompt=prompt, generator=generator, num_inference_steps=8).images[0]
    # Clear cuda memory as pytest doesnt clear it between tests
    del pipe
    torch.cuda.empty_cache()


def test_pixart_cachify():
    # Fail test if apex is installed
    if "apex" in subprocess.check_output(["pip", "list"]).decode("utf-8"):
        pytest.xfail("Apex is installed, test is expected to fail")

    pipe = get_tiny_pixart_pipeline().to(device="cuda", dtype=torch.float16)
    cachify.prepare(pipe, PIXART_DEFAULT_CONFIG)

    prompt = "a small cactus with a happy face in the Sahara desert"
    generator = torch.Generator(device="cuda").manual_seed(2946901)
    cached = pipe.transformer.transformer_blocks[0]
    assert isinstance(cached, CachedModule)
    with patch.object(cached.block, "forward", wraps=cached.block.forward) as compute:
        image = pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=8,
            height=16,
            width=16,
            use_resolution_binning=False,
        ).images[0]
    assert image.size == (16, 16)
    assert cached.cur_step == 8
    assert compute.call_count == 3  # Recompute on steps 0, 3, and 6; reuse the other five.
    # Clear cuda memory as pytest doesnt clear it between tests
    del pipe
    torch.cuda.empty_cache()
