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

import itertools
import subprocess

import pytest
import torch

# Common test prompts for all backends
COMMON_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


class ModelDeployer:
    def __init__(
        self,
        backend: str = "trtllm",
        model_id: str = "",
        tensor_parallel_size: int = 1,
        mini_sm: int = 89,
        attn_backend: str = "TRTLLM",
        base_model: str = "",
        eagle3_one_model: bool = True,
    ):
        """
        Initialize the ModelDeployer.

        Args:
            backend: The backend to use ('vllm', 'trtllm', or 'sglang')
            model_id: Path to the model
            tensor_parallel_size: Tensor parallel size for distributed inference
            mini_sm: Minimum SM (Streaming Multiprocessor) requirement for the model
            attn_backend: is for TRT LLM deployment
        """
        self.backend = backend
        self.model_id = model_id
        self.tensor_parallel_size = tensor_parallel_size
        self.mini_sm = mini_sm
        self.attn_backend = attn_backend
        self.base_model = base_model
        self.eagle3_one_model = eagle3_one_model

    def run(self):
        """Run the deployment based on the specified backend."""
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            pytest.skip("CUDA is not available")
            return
        if torch.cuda.get_device_capability() < (
            self.mini_sm // 10,
            self.mini_sm % 10,
        ):
            pytest.skip(reason=f"Requires sm{self.mini_sm} or higher")
            return

        if torch.cuda.device_count() < self.tensor_parallel_size:
            pytest.skip(reason=f"Requires at least {self.tensor_parallel_size} GPUs")
            return
        if self.backend == "vllm":
            self._deploy_vllm()
        elif self.backend == "trtllm":
            self._deploy_trtllm()
        elif self.backend == "sglang":
            self._deploy_sglang()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        # check gpu status
        gpu_status = subprocess.run(
            "nvidia-smi || true", shell=True, capture_output=True, text=True, check=True
        )
        print("\n=== GPU Status Before Test ===")
        print(gpu_status.stdout)
        print("=============================\n")

    def _deploy_trtllm(self):
        """Deploy a model using TensorRT-LLM."""
        try:
            from tensorrt_llm import LLM, SamplingParams
            from tensorrt_llm.llmapi import CudaGraphConfig, EagleDecodingConfig, KvCacheConfig
        except ImportError:
            pytest.skip("tensorrt_llm package not available")

        sampling_params = SamplingParams(max_tokens=32)
        spec_config = None
        llm = None
        kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.8)

        if self.model_id == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8":
            llm = LLM(
                model=self.model_id,
                tensor_parallel_size=self.tensor_parallel_size,
                enable_attention_dp=False,
                attn_backend=self.attn_backend,
                trust_remote_code=True,
                max_batch_size=8,
                kv_cache_config=KvCacheConfig(
                    enable_block_reuse=False,
                    mamba_ssm_cache_dtype="float32",
                ),
            )
        elif self.model_id == "nvidia/EAGLE3-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16":
            spec_config = EagleDecodingConfig(
                max_draft_len=3,
                speculative_model_dir=self.model_id,
                eagle3_one_model=self.eagle3_one_model,
            )
            llm = LLM(
                model=self.model_id,
                tensor_parallel_size=self.tensor_parallel_size,
                enable_attention_dp=False,
                attn_backend=self.attn_backend,
                trust_remote_code=True,
                max_batch_size=8,
                speculative_config=spec_config,
                kv_cache_config=KvCacheConfig(
                    enable_block_reuse=False,
                    mamba_ssm_cache_dtype="float32",
                ),
            )
        elif "eagle" in self.model_id.lower():
            spec_config = EagleDecodingConfig(
                max_draft_len=3,
                speculative_model_dir=self.model_id,
                eagle3_one_model=self.eagle3_one_model,
            )
            cuda_graph = CudaGraphConfig(
                max_batch_size=1,
            )
            llm = LLM(
                model=self.base_model,
                tensor_parallel_size=self.tensor_parallel_size,
                enable_attention_dp=False,
                disable_overlap_scheduler=True,
                enable_autotuner=False,
                speculative_config=spec_config,
                cuda_graph_config=cuda_graph,
                kv_cache_config=kv_cache_config,
            )
        else:
            llm = LLM(
                model=self.model_id,
                tensor_parallel_size=self.tensor_parallel_size,
                enable_attention_dp=False,
                attn_backend=self.attn_backend,
                trust_remote_code=True,
                max_batch_size=8,
                kv_cache_config=kv_cache_config,
            )

        outputs = llm.generate(COMMON_PROMPTS, sampling_params)
        # Print outputs
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        del llm

    def _deploy_vllm(self):
        """Deploy a model using vLLM."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            pytest.skip("vllm package not available")

        quantization_method = "modelopt"
        if "fp4" in self.model_id.lower():
            quantization_method = "modelopt_fp4"
        llm = LLM(
            model=self.model_id,
            quantization=quantization_method,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(temperature=0.8, top_p=0.9)
        outputs = llm.generate(COMMON_PROMPTS, sampling_params)

        # Assertions and output
        assert len(outputs) == len(COMMON_PROMPTS), (
            f"Expected {len(COMMON_PROMPTS)} outputs, got {len(outputs)}"
        )

        for i, output in enumerate(outputs):
            assert output.prompt == COMMON_PROMPTS[i], f"Prompt mismatch at index {i}"
            assert hasattr(output, "outputs"), f"Output {i} missing 'outputs' attribute"
            assert len(output.outputs) > 0, f"Output {i} has no generated text"
            assert hasattr(output.outputs[0], "text"), f"Output {i} missing 'text' attribute"
            assert isinstance(output.outputs[0].text, str), f"Output {i} text is not a string"
            assert len(output.outputs[0].text) > 0, f"Output {i} generated empty text"

            print(f"Model: {self.model_id}")
            print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
            print("-" * 50)
        del llm

    def _deploy_sglang(self):
        """Deploy a model using SGLang."""
        try:
            import sglang as sgl
        except ImportError:
            pytest.skip("sglang package not available")
        quantization_method = "modelopt"
        if "fp4" in self.model_id.lower():
            quantization_method = "modelopt_fp4"
        if "eagle" in self.model_id.lower():
            llm = sgl.Engine(
                model_path=self.base_model,
                speculative_algorithm="EAGLE3",
                speculative_num_steps=3,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=4,
                speculative_draft_model_path=self.model_id,
                tp_size=self.tensor_parallel_size,
                trust_remote_code=True,
                mem_fraction_static=0.7,
                context_length=1024,
            )
        elif self.model_id == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8":
            llm = sgl.Engine(
                model_path=self.model_id,
                quantization=quantization_method,
                tp_size=self.tensor_parallel_size,
                trust_remote_code=True,
                attention_backend="flashinfer",
            )
        else:
            llm = sgl.Engine(
                model_path=self.model_id,
                quantization=quantization_method,
                tp_size=self.tensor_parallel_size,
                trust_remote_code=True,
            )
        print(llm.generate(["What's the age of the earth? "]))
        llm.shutdown()
        del llm


class ModelDeployerList:
    def __init__(self, **params):
        self.params = {}
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                self.params[key] = list(value)
            else:
                self.params[key] = [value]

        # Pre-generate all deployers for pytest compatibility
        self._deployers = list(self._generate_deployers())

    def _generate_deployers(self):
        for values in itertools.product(*self.params.values()):
            deployer = ModelDeployer(**dict(zip(self.params.keys(), values)))
            # Set test case ID in format "model_id_backend"
            deployer.test_id = f"{deployer.model_id}_{deployer.backend}"
            yield deployer

    def __iter__(self):
        return iter(self._deployers)

    def __len__(self):
        return len(self._deployers)

    def __getitem__(self, index):
        return self._deployers[index]

    def __str__(self):
        return f"ModelDeployerList({len(self._deployers)} items)"

    def __repr__(self):
        return f"ModelDeployerList({len(self._deployers)} items)"
