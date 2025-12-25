# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import shutil

import pytest
from _test_utils.deploy_utils import ModelDeployerList

pytestmark = pytest.mark.release


def idfn(val):
    if hasattr(val, "test_id"):
        return val.test_id
    return str(val)


# clean up hf cache
HF_CACHE_PATH = os.getenv("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))


def clear_hf_cache():
    """Clear Hugging Face cache directory."""
    try:
        if os.path.exists(HF_CACHE_PATH):
            print(f"Clearing HF cache at: {HF_CACHE_PATH}")
            for item in os.listdir(HF_CACHE_PATH):
                item_path = os.path.join(HF_CACHE_PATH, item)
                if os.path.isdir(item_path) and "nvidia" in item:
                    shutil.rmtree(item_path, ignore_errors=True)
                    print(f"✓ Removed: {item}")
            print("✓ HF cache cleared successfully")
        else:
            print(f"HF cache path does not exist: {HF_CACHE_PATH}")
    except Exception as e:
        print(f"⚠ Warning: Failed to clear HF cache: {e}")


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatically clean up after each test."""
    yield  # Run the test
    clear_hf_cache()  # Clean up after test completes


@pytest.mark.parametrize(
    "command",
    [
        *ModelDeployerList(
            model_id="nvidia/DeepSeek-R1-FP4",
            backend=("vllm", "trtllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/DeepSeek-R1-FP4-v2",
            backend=("vllm", "trtllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/DeepSeek-R1-0528-FP4",
            backend=("vllm", "trtllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/DeepSeek-R1-0528-FP4-v2",
            backend=("vllm", "trtllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/DeepSeek-V3-0324-FP4",
            backend=("vllm", "trtllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=100,
        ),
    ],
    ids=idfn,
)
def test_deepseek(command):
    command.run()


@pytest.mark.parametrize(
    "command",
    [
        # Llama-3.1
        *ModelDeployerList(
            model_id="nvidia/Llama-3.1-8B-Instruct-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=89,
        ),
        *ModelDeployerList(
            model_id="nvidia/Llama-3.1-8B-Instruct-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=100,
        ),
        # ModelDeployer(model_id="nvidia/Llama-3.1-8B-Medusa-FP8", backend="vllm"),
        *ModelDeployerList(
            model_id="nvidia/Llama-3.1-70B-Instruct-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=4,
        ),
        *ModelDeployerList(
            model_id="nvidia/Llama-3.3-70B-Instruct-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=4,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/Llama-3.3-70B-Instruct-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=4,
            mini_sm=89,
        ),
        *ModelDeployerList(
            model_id="nvidia/Llama-3.1-405B-Instruct-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=8,
        ),
        *ModelDeployerList(
            model_id="nvidia/Llama-3.1-405B-Instruct-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=100,
        ),
        # Llama-4
        *ModelDeployerList(
            model_id="nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=8,
        ),
        *ModelDeployerList(
            model_id="nvidia/Llama-4-Maverick-17B-128E-Instruct-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=89,
        ),
        *ModelDeployerList(
            model_id="nvidia/Llama-4-Scout-17B-16E-Instruct-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=100,
        ),
    ],
    ids=idfn,
)
def test_llama(command):
    command.run()


@pytest.mark.parametrize(
    "command",
    [
        *ModelDeployerList(
            model_id="nvidia/Qwen3-8B-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/Qwen3-8B-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=89,
        ),
        *ModelDeployerList(
            model_id="nvidia/Qwen3-14B-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/Qwen3-14B-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=89,
        ),
        *ModelDeployerList(
            model_id="nvidia/Qwen3-235B-A22B-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=2,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/Qwen3-235B-A22B-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=4,
            mini_sm=89,
        ),
        *ModelDeployerList(
            model_id="nvidia/QwQ-32B-FP4", backend=("trtllm", "vllm", "sglang"), mini_sm=100
        ),
        *ModelDeployerList(
            model_id="nvidia/Qwen3-32B-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=4,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/Qwen2.5-VL-7B-Instruct-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=4,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/Qwen2.5-VL-7B-Instruct-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=4,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/Qwen3-30B-A3B-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=4,
            mini_sm=100,
        ),
    ],
    ids=idfn,
)
def test_qwen(command):
    command.run()


@pytest.mark.parametrize(
    "command",
    [
        *ModelDeployerList(
            model_id="nvidia/Mixtral-8x7B-Instruct-v0.1-FP8", backend=("trtllm", "vllm", "sglang")
        ),
        *ModelDeployerList(
            model_id="nvidia/Mixtral-8x7B-Instruct-v0.1-FP4",
            backend=("trtllm", "vllm", "sglang"),
            mini_sm=100,
        ),
        #   ModelDeployer(model_id="nvidia/Mixtral-8x7B-Instruct-v0.1-FP8", backend="sglang"), unsupported
    ],
    ids=idfn,
)
def test_mixtral(command):
    command.run()


@pytest.mark.parametrize(
    "command",
    [  # TRTLLM bug: https://nvbugs/5451286
        *ModelDeployerList(
            model_id="nvidia/gemma-3-12b-it-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=100,
            attn_backend="FLASHINFER",
        ),
        *ModelDeployerList(
            model_id="nvidia/gemma-3-12b-it-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=89,
            attn_backend="FLASHINFER",
        ),
        *ModelDeployerList(
            model_id="nvidia/gemma-3-27b-it-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=100,
            attn_backend="FLASHINFER",
        ),
        *ModelDeployerList(
            model_id="nvidia/gemma-3-27b-it-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=89,
            attn_backend="FLASHINFER",
        ),
    ],
    ids=idfn,
)
def test_gemma(command):
    command.run()


# test phi
@pytest.mark.parametrize(
    "command",
    [
        *ModelDeployerList(
            model_id="nvidia/Phi-4-multimodal-instruct-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/Phi-4-multimodal-instruct-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=89,
        ),
        *ModelDeployerList(
            model_id="nvidia/Phi-4-reasoning-plus-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/Phi-4-reasoning-plus-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=89,
        ),
    ],
    ids=idfn,
)
def test_phi(command):
    command.run()


@pytest.mark.parametrize(
    "command",
    [
        *ModelDeployerList(
            model_id="nvidia/Kimi-K2-Instruct-FP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/Kimi-K2-Thinking-NVFP4",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=100,
        ),
    ],
    ids=idfn,
)
def test_kimi(command):
    command.run()


@pytest.mark.parametrize(
    "command",
    [
        *ModelDeployerList(
            model_id="nvidia/Llama-3_3-Nemotron-Super-49B-v1-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=89,
        ),
        *ModelDeployerList(
            model_id="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=89,
        ),
        *ModelDeployerList(
            model_id="nvidia/Llama-3_1-Nemotron-Ultra-253B-v1-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=4,
            mini_sm=89,
        ),
        *ModelDeployerList(
            model_id="nvidia/Llama-3_1-Nemotron-Ultra-253B-v1-FP8",
            backend=("vllm",),
            tensor_parallel_size=8,
            mini_sm=89,
        ),
        *ModelDeployerList(
            model_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
            backend=("trtllm", "vllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=89,
            attn_backend="FLASHINFER",
        ),
    ],
    ids=idfn,
)
def test_llama_nemotron(command):
    command.run()


@pytest.mark.parametrize(
    "command",
    [
        *ModelDeployerList(
            model_id="nvidia/Llama-3.1-8B-Medusa-FP8",
            backend=("trtllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=89,
        ),
        *ModelDeployerList(
            model_id="nvidia/Llama-3.1-70B-Medusa-FP8",
            backend=("trtllm", "sglang"),
            tensor_parallel_size=2,
            mini_sm=100,
        ),
        *ModelDeployerList(
            model_id="nvidia/Llama-3.1-405B-Medusa-FP8",
            backend=("trtllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=100,
        ),
    ],
    ids=idfn,
)
@pytest.mark.skip(reason="Medusa is not supported yet")
def test_medusa(command):
    command.run()


@pytest.mark.parametrize(
    "command",
    [
        *ModelDeployerList(
            base_model="nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8",
            model_id="nvidia/Llama-4-Maverick-17B-128E-Eagle3",
            backend=("trtllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=89,
        ),
        *ModelDeployerList(
            base_model="Qwen/Qwen3-235B-A22B",
            model_id="nvidia/Qwen3-235B-A22B-Eagle3",
            backend=("trtllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=89,
        ),
        *ModelDeployerList(
            base_model="Qwen/Qwen3-235B-A22B-Thinking-2507",
            model_id="nvidia/Qwen3-235B-A22B-Thinking-2507-Eagle3",
            backend=("trtllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=89,
            eagle3_one_model=False,
        ),
        *ModelDeployerList(
            base_model="Qwen/Qwen3-30B-A3B",
            model_id="nvidia/Qwen3-30B-A3B-Eagle3",
            backend=("trtllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=89,
        ),
        *ModelDeployerList(
            base_model="Qwen/Qwen3-30B-A3B-Thinking-2507",
            model_id="nvidia/Qwen3-30B-A3B-Thinking-2507-Eagle3",
            backend=("trtllm", "sglang"),
            tensor_parallel_size=1,
            mini_sm=89,
            eagle3_one_model=False,
        ),
        *ModelDeployerList(
            base_model="openai/gpt-oss-120b",
            model_id="nvidia/gpt-oss-120b-Eagle3-long-context",
            backend=("trtllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=89,
        ),
        *ModelDeployerList(
            base_model="openai/gpt-oss-120b",
            model_id="nvidia/gpt-oss-120b-Eagle3-short-context",
            backend=("trtllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=89,
        ),
        *ModelDeployerList(
            base_model="openai/gpt-oss-120b",
            model_id="nvidia/gpt-oss-120b-Eagle3-throughput",
            backend=("trtllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=89,
        ),
        *ModelDeployerList(
            base_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            model_id="nvidia/EAGLE3-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            backend=("trtllm", "vllm", "sglang"),
            eagle3_one_model=False,
            tensor_parallel_size=8,
            mini_sm=89,
        ),
        *ModelDeployerList(
            base_model="nvidia/Llama-3.3-70B-Instruct-FP8",
            model_id="nvidia/Llama-3.3-70B-Instruct-Eagle3",
            backend=("trtllm", "sglang"),
            tensor_parallel_size=8,
            mini_sm=89,
        ),
    ],
    ids=idfn,
)
def test_eagle(command):
    """Skip test if MODELOPT_LOCAL_MODEL_ROOT is set but model doesn't exist locally.
    speculative models shoule be loaded by local path"""
    local_root = os.getenv("MODELOPT_LOCAL_MODEL_ROOT")
    if not local_root:
        return

    local_path = os.path.join(local_root, command.model_id)
    if os.path.isdir(local_path):
        # Update model_id to use local path
        command.model_id = local_path
        command.run()
    else:
        pytest.skip(f"Local model not found: {local_path}")
