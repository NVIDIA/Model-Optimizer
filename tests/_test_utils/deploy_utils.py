# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import base64
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from .deploy_config import engine_tp_kwargs
from .examples.run_command import ProcessGroupCleanupError, _run_capturing

# Cache for available backends detection (computed once at import time)
_AVAILABLE_BACKENDS = None


def get_available_backends():
    """Detect which backends are available in the current environment.

    Returns:
        set: A set of available backend names ('trtllm', 'vllm', 'sglang')
    """
    global _AVAILABLE_BACKENDS
    if _AVAILABLE_BACKENDS is not None:
        return _AVAILABLE_BACKENDS

    available = set()

    try:
        import tensorrt_llm  # noqa: F401

        available.add("trtllm")
    except ImportError:
        pass

    try:
        import vllm  # noqa: F401

        available.add("vllm")
    except ImportError:
        pass

    try:
        import sglang  # noqa: F401

        available.add("sglang")
    except ImportError:
        pass

    _AVAILABLE_BACKENDS = available
    print(f"[deploy_utils] Detected available backends: {available}")
    return _AVAILABLE_BACKENDS


def _run_deploy_via_subprocess(backend: str, **parameters) -> None:
    """Keep engine settings out of the child process arguments and failure tracebacks."""
    tests_dir = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(tests_dir), str(tests_dir.parent), os.environ.get("PYTHONPATH", "")]
        ),
    }
    launcher = []
    if backend == "trtllm":
        launcher = [
            os.environ.get("MPIRUN", "mpirun"),
            "--allow-run-as-root",
            "--oversubscribe",
            "-n",
            "1",
        ]
    # Native backend errors can echo arbitrary override values; checked-in recipe values
    # are public, so only caller-supplied overrides mark the output private. Kept out of
    # the worker config: it controls the parent's capture, not the deployment.
    private_output = parameters.pop("private_output", False)
    fd, config_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as config:
            json.dump({"backend": backend, **parameters}, config)
        cmd = [*launcher, sys.executable, "-m", "_test_utils.deploy_utils", config_path]
        returncode, output = _run_capturing(
            cmd, cwd=tests_dir, env=env, cleanup_process_group=True, discard_output=private_output
        )
        if returncode:
            raise subprocess.CalledProcessError(returncode, cmd, output=output)
    finally:
        os.unlink(config_path)


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
        base_model: str = "",
        engine_kwargs: dict | None = None,
        sm_family: int | None = None,
        sampling_kwargs: dict | None = None,
        input_config: dict | None = None,
        chat_kwargs: dict | None = None,
        require_speculation: bool = False,
        sm_versions: list[int] | None = None,
        private_output: bool = False,
    ):
        """
        Initialize the ModelDeployer.

        Args:
            backend: The backend to use ('vllm', 'trtllm', or 'sglang')
            model_id: Path to the model
            tensor_parallel_size: Tensor parallel size for distributed inference
            mini_sm: Minimum SM (Streaming Multiprocessor) requirement for the model
            engine_kwargs: resolved native engine parameters from the case and run YAML.
                Set model and TP through model_id and tensor_parallel_size only.
            sm_family: if set, skip unless the GPU belongs to this SM family (100 for SM100 to
                SM103), for a backend path that checks the family rather than a minimum SM
            sampling_kwargs: resolved native backend sampling parameters.
            input_config: optional prompt and local image for a configured request.
            chat_kwargs: chat-template settings, or native chat-call options for vLLM.
            require_speculation: require evidence of actual draft execution during generation.
            sm_versions: exact CUDA capabilities allowed by the selected backend recipe.
            private_output: drain backend output without retaining or printing it, for runs
                whose overrides may carry credentials.
        """
        self.backend = backend
        self.model_id = model_id
        self.tensor_parallel_size = tensor_parallel_size
        self.mini_sm = mini_sm
        self.base_model = base_model
        self.engine_kwargs = dict(engine_kwargs or {})
        self.sm_family = sm_family
        self.trust_remote_code = self.engine_kwargs.get("trust_remote_code", False)
        if not isinstance(self.trust_remote_code, bool):
            raise ValueError("trust_remote_code must be a boolean")
        self.sampling_kwargs = dict(sampling_kwargs or {})
        self.input_config = dict(input_config or {})
        self.chat_kwargs = dict(chat_kwargs or {})
        self.require_speculation = require_speculation
        self.sm_versions = sm_versions
        self.private_output = private_output
        if backend != "vllm" and self.chat_kwargs.keys() - {"chat_template_kwargs"}:
            raise ValueError("Only vLLM accepts chat-call options beyond chat_template_kwargs")
        if backend != "vllm":
            template_kwargs = self.chat_kwargs.get("chat_template_kwargs", {})
            if not isinstance(template_kwargs, dict):
                raise ValueError("chat_template_kwargs must be a mapping")
            if reserved := {"tokenize", "add_generation_prompt"} & template_kwargs.keys():
                raise ValueError(
                    f"The deployment runner sets {sorted(reserved)} in the chat template"
                )
        if {"messages", "sampling_params"} & self.chat_kwargs.keys():
            raise ValueError("Set input_config and sampling_kwargs instead of chat-call inputs")
        if require_speculation:
            if not base_model or backend not in ("vllm", "sglang", "trtllm"):
                raise ValueError(
                    "Speculative execution checks require a target and supported backend"
                )
            if backend == "vllm" and self.engine_kwargs.get("disable_log_stats") is not False:
                raise ValueError("Speculative execution checks require disable_log_stats: false")
            if backend == "trtllm" and self.sampling_kwargs.get("return_perf_metrics") is not True:
                raise ValueError("Speculative execution checks require return_perf_metrics: true")
        if backend == "sglang" and self.engine_kwargs.get("enable_layernorm_sp"):
            if self.engine_kwargs.get("ep_size", tensor_parallel_size) != tensor_parallel_size:
                raise ValueError("Layernorm sequence parallelism requires EP=TP; override both")
        reserved_keys = sorted(
            {"model", "model_path", "tensor_parallel_size", "tp_size"} & self.engine_kwargs.keys()
        )
        if reserved_keys:
            raise ValueError(
                f"Set model_id and tensor_parallel_size instead of {reserved_keys} in "
                "engine_kwargs so resource checks and engine arguments stay consistent."
            )
        # Resource checks cover tensor_parallel_size ranks only; rank-multiplying fields
        # (pipeline/context/data parallelism) would bypass them on this single-node harness.
        topology_fields = {
            "vllm": {"pipeline_parallel_size", "data_parallel_size"},
            "sglang": {"pp_size", "dp_size", "nnodes"},
            "trtllm": {
                "pipeline_parallel_size",
                "context_parallel_size",
                "moe_cluster_parallel_size",
                "gpus_per_node",
            },
        }.get(backend, set())
        if unsupported := {
            key: self.engine_kwargs[key]
            for key in topology_fields & self.engine_kwargs.keys()
            if self.engine_kwargs[key] != 1
        }:
            raise ValueError(
                f"Rank-multiplying engine option(s) {sorted(unsupported)} need resource checks "
                "this harness cannot perform"
            )

    def _engine_arguments(self):
        """Bind runtime model paths and TP without supplying test parameter defaults."""
        model_key = "model_path" if self.backend == "sglang" else "model"
        kwargs = {
            model_key: self.base_model or self.model_id,
            **engine_tp_kwargs(self.backend, self.tensor_parallel_size),
            **copy.deepcopy(self.engine_kwargs),
        }
        if self.base_model:
            # test_eagle may replace the checkpoint ID with its local path after collection.
            if self.backend == "sglang":
                if self.require_speculation and kwargs.get("speculative_algorithm") != "DFLASH":
                    raise ValueError(
                        "This draft execution check requires speculative_algorithm: DFLASH"
                    )
                if "speculative_draft_model_path" in kwargs:
                    raise ValueError("Set the draft checkpoint through model_id, not engine_kwargs")
                if not kwargs.get("speculative_algorithm"):
                    raise ValueError(
                        "Draft cases require speculative_algorithm to load the drafter"
                    )
                kwargs["speculative_draft_model_path"] = self.model_id
            else:
                spec_config = kwargs.get("speculative_config")
                if not isinstance(spec_config, dict):
                    raise ValueError("Draft cases require speculative_config to load the drafter")
                method_key, method = (
                    ("method", "dflash") if self.backend == "vllm" else ("decoding_type", "DFlash")
                )
                if self.require_speculation and spec_config.get(method_key) != method:
                    raise ValueError(
                        f"This draft execution check requires speculative_config.{method_key}: {method}"
                    )
                model_field = (
                    "model"
                    if self.backend == "vllm"
                    else "speculative_model"
                    if spec_config.get("decoding_type") == "DFlash"
                    else "speculative_model_dir"
                )
                model_fields = (
                    {"model"}
                    if self.backend == "vllm"
                    else {"speculative_model_dir", "speculative_model"}
                )
                if model_fields & spec_config.keys():
                    raise ValueError("Set the draft checkpoint through model_id, not engine_kwargs")
                spec_config[model_field] = self.model_id
        # Native engine options can contain credentials, including inside nested mappings.
        print(f"Engine option names ({self.backend}): {sorted(kwargs)}", flush=True)
        return kwargs

    def run(self):
        """Run the deployment based on the specified backend."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        if torch.cuda.device_count() < self.tensor_parallel_size:
            pytest.skip(f"Requires at least {self.tensor_parallel_size} GPUs")
        for device in range(self.tensor_parallel_size):
            major, minor = torch.cuda.get_device_capability(device)
            sm = major * 10 + minor
            if sm < self.mini_sm:
                pytest.skip(f"GPU {device} is SM{sm}, requires SM{self.mini_sm} or higher")
            if self.sm_family is not None and major != self.sm_family // 10:
                pytest.skip(f"GPU {device} is SM{sm}, requires an SM{self.sm_family // 10}x GPU")
            if self.sm_versions is not None and sm not in self.sm_versions:
                pytest.skip(f"GPU {device} is SM{sm}, selected recipe requires {self.sm_versions}")

        print(f"Deploying model: {self.model_id} with backend: {self.backend}")
        print(f"Tensor parallel size: {self.tensor_parallel_size}")
        print(f"Sampling option names: {sorted(self.sampling_kwargs)}", flush=True)
        if self.input_config:
            print(f"Input configuration: {self.input_config!r}", flush=True)
        if self.chat_kwargs:
            print(f"Chat option names: {sorted(self.chat_kwargs)}", flush=True)
        # Use subprocess + capture so pytest capture=tee-sys (and DB plugins) see deploy output.
        if self.backend in ("vllm", "trtllm", "sglang"):
            try:
                _run_deploy_via_subprocess(
                    backend=self.backend,
                    model_id=self.model_id,
                    tensor_parallel_size=self.tensor_parallel_size,
                    mini_sm=self.mini_sm,
                    base_model=self.base_model,
                    engine_kwargs=self.engine_kwargs,
                    sampling_kwargs=self.sampling_kwargs,
                    input_config=self.input_config,
                    chat_kwargs=self.chat_kwargs,
                    require_speculation=self.require_speculation,
                    private_output=self.private_output,
                )
            except ProcessGroupCleanupError as exc:
                pytest.exit(
                    f"Deployment workers did not exit; stopping GPU reuse: {exc}", returncode=1
                )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _deploy_trtllm_impl(self):
        """Run TensorRT-LLM deploy (used by subprocess in run())."""
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import CudaGraphConfig, EagleDecodingConfig, KvCacheConfig

        sampling_params = SamplingParams(**self.sampling_kwargs)
        llm_kwargs = self._engine_arguments()
        # Preserve the typed configs used by this suite; YAML supplies their field values.
        for field, config_type in (
            ("kv_cache_config", KvCacheConfig),
            ("cuda_graph_config", CudaGraphConfig),
        ):
            if isinstance(llm_kwargs.get(field), dict):
                llm_kwargs[field] = config_type(**llm_kwargs[field])
        spec_config = llm_kwargs.get("speculative_config")
        if isinstance(spec_config, dict) and spec_config.get("decoding_type") == "DFlash":
            # DFlash is optional in older engine images used by other cases.
            from tensorrt_llm.llmapi import DFlashDecodingConfig

            llm_kwargs["speculative_config"] = DFlashDecodingConfig(**spec_config)
        elif (
            isinstance(spec_config, dict)
            and "speculative_model_dir" in spec_config
            and spec_config.get("decoding_type", "Eagle") in ("Eagle", "Eagle3")
        ):
            if "eagle3_one_model" not in EagleDecodingConfig.model_fields:
                if spec_config.pop("eagle3_one_model", None) is False:
                    print("This TensorRT-LLM version only supports one-model EAGLE3 execution.")
            llm_kwargs["speculative_config"] = EagleDecodingConfig(**spec_config)
        llm = LLM(**llm_kwargs)
        try:
            # Transformers and multimodal helpers are optional deployment dependencies.
            from transformers import AutoTokenizer

            tokenizer_model = self.base_model or self.model_id
            raw_prompts = [self.input_config["prompt"]] if self.input_config else COMMON_PROMPTS
            template_kwargs = self.chat_kwargs.get("chat_template_kwargs", {})
            if "image" in self.input_config:
                from tensorrt_llm.inputs import async_build_multimodal_prompt
                from transformers import AutoConfig, AutoProcessor

                config = AutoConfig.from_pretrained(
                    tokenizer_model, trust_remote_code=self.trust_remote_code
                )
                processor = AutoProcessor.from_pretrained(
                    tokenizer_model, trust_remote_code=self.trust_remote_code
                )
                prompts = [
                    asyncio.run(
                        async_build_multimodal_prompt(
                            model_type=config.model_type,
                            tokenizer=llm.tokenizer,
                            processor=processor,
                            prompt=raw_prompts[0],
                            media=[self.input_config["image"]],
                            chat_template_kwargs=template_kwargs,
                        )
                    )
                ]
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_model, trust_remote_code=self.trust_remote_code
                )
                prompts = (
                    [
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                            **template_kwargs,
                        )
                        for prompt in raw_prompts
                    ]
                    if tokenizer.chat_template is not None
                    else raw_prompts
                )
            outputs = llm.generate(prompts, sampling_params)
            assert len(outputs) == len(raw_prompts), (
                f"Expected {len(raw_prompts)} outputs, got {len(outputs)}"
            )
            if self.require_speculation:
                draft_tokens = 0
                for output in outputs:
                    metrics = output.outputs[0].request_perf_metrics
                    assert metrics is not None, "Missing speculative execution metrics"
                    draft_tokens += metrics.speculative_decoding.total_draft_tokens
                print(f"Speculative execution counts: drafted_tokens={draft_tokens}", flush=True)
                assert draft_tokens > 0, "Generation did not exercise the draft checkpoint"
            for i, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                assert isinstance(generated_text, str), f"Output {i} text is not a string"
                assert generated_text.strip(), f"Output {i} generated empty text"
                print(f"Model: {self.model_id}")
                print(f"Input prompt: {raw_prompts[i]!r}")
                print(f"Rendered prompt: {output.prompt!r}")
                print(f"Generated text: {generated_text!r}")
                print("-" * 50)
        finally:
            llm.shutdown()

    def _deploy_vllm_impl(self):
        """Run vLLM deploy (used by subprocess in run())."""
        from vllm import LLM, SamplingParams

        llm = LLM(**self._engine_arguments())
        sampling_params = SamplingParams(**self.sampling_kwargs)
        prompts = [self.input_config["prompt"]] if self.input_config else COMMON_PROMPTS
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        if "image" in self.input_config:
            # Pillow is needed only for the optional vision-language cases.
            from PIL import Image

            with Image.open(self.input_config["image"]) as image, io.BytesIO() as buffer:
                image.convert("RGB").save(buffer, format="PNG")
                data = base64.b64encode(buffer.getvalue()).decode("ascii")
            conversations = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{data}"},
                            },
                            {"type": "text", "text": prompts[0]},
                        ],
                    }
                ]
            ]
        before = self._vllm_draft_counts(llm) if self.require_speculation else {}
        outputs = llm.chat(conversations, sampling_params, **self.chat_kwargs)
        if self.require_speculation:
            after = self._vllm_draft_counts(llm)
            counts = {key: after[key] - before[key] for key in before}
            print(f"Speculative execution counts (generation delta): {counts!r}", flush=True)
            assert all(value > 0 for value in counts.values()), (
                f"Generation did not exercise the draft checkpoint: {counts}"
            )

        assert len(outputs) == len(prompts), f"Expected {len(prompts)} outputs, got {len(outputs)}"

        for i, output in enumerate(outputs):
            assert hasattr(output, "outputs"), f"Output {i} missing 'outputs' attribute"
            assert len(output.outputs) > 0, f"Output {i} has no generated text"
            assert hasattr(output.outputs[0], "text"), f"Output {i} missing 'text' attribute"
            assert isinstance(output.outputs[0].text, str), f"Output {i} text is not a string"
            generated_text = output.outputs[0].text
            assert generated_text.strip(), f"Output {i} generated empty text"

            print(f"Model: {self.model_id}")
            print(f"Input prompt: {prompts[i]!r}")
            print(f"Rendered prompt: {getattr(output, 'prompt', None)!r}")
            print(f"Generated text: {generated_text!r}")
            print("-" * 50)
        del llm

    @staticmethod
    def _vllm_draft_counts(llm):
        names = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens")
        counts = dict.fromkeys(names, 0)
        for metric in llm.get_metrics():
            if metric.name in counts:
                counts[metric.name] += metric.value
        return counts

    def _deploy_sglang_impl(self):
        """Run SGLang deploy (used by subprocess in run())."""
        import sglang as sgl

        if "eagle" in self.model_id.lower() and "nemotron" in self.base_model.lower():
            # Nemotron hybrid (Mamba+attention) requires extra_buffer scheduler
            # strategy and SGLANG_ENABLE_SPEC_V2 for radix cache compatibility
            # with speculative decoding
            os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"
        llm = sgl.Engine(**self._engine_arguments())
        try:
            # Deferred import: transformers is a heavy optional dependency not always present.
            from transformers import AutoTokenizer

            tokenizer_model = self.base_model or self.model_id
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model, trust_remote_code=self.trust_remote_code
            )
            raw_prompts = [self.input_config["prompt"]] if self.input_config else COMMON_PROMPTS
            conversations = [[{"role": "user", "content": prompt}] for prompt in raw_prompts]
            generation_kwargs = (
                {"sampling_params": self.sampling_kwargs} if self.sampling_kwargs else {}
            )
            if "image" in self.input_config:
                conversations[0][0]["content"] = [
                    {"type": "image"},
                    {"type": "text", "text": raw_prompts[0]},
                ]
                generation_kwargs["image_data"] = [self.input_config["image"]]
                if tokenizer.chat_template is None:
                    raise ValueError(
                        "Image deployment requires the model's multimodal chat template"
                    )
            if tokenizer.chat_template is not None:
                prompts = [
                    tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                        **self.chat_kwargs.get("chat_template_kwargs", {}),
                    )
                    for conversation in conversations
                ]
            else:
                prompts = raw_prompts

            outputs = llm.generate(prompts, **generation_kwargs)
            if self.require_speculation:
                names = ("spec_verify_ct", "spec_num_proposed_drafts")
                counts = {
                    name: sum(output.get("meta_info", {}).get(name, 0) for output in outputs)
                    for name in names
                }
                print(f"Speculative execution counts: {counts!r}", flush=True)
                assert all(value > 0 for value in counts.values()), (
                    f"Generation did not exercise the draft checkpoint: {counts}"
                )
            assert len(outputs) == len(raw_prompts), (
                f"Expected {len(raw_prompts)} outputs, got {len(outputs)}"
            )
            for i, output in enumerate(outputs):
                if isinstance(output, dict):
                    generated_text = output["text"]
                elif isinstance(output, str):
                    generated_text = output
                else:
                    raise TypeError(f"Output {i} has unexpected type {type(output)}: {output!r}")
                assert isinstance(generated_text, str), f"Output {i} text is not a string"
                assert generated_text.strip(), f"Output {i} generated empty text"
                print(f"Model: {self.model_id}")
                print(f"Input prompt: {raw_prompts[i]!r}")
                print(f"Rendered prompt: {prompts[i]!r}")
                print(f"Generated text: {generated_text!r}")
                print("-" * 50)
        finally:
            llm.shutdown()
        del llm


if __name__ == "__main__":
    deployer = ModelDeployer(**json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    deploy = {
        "trtllm": deployer._deploy_trtllm_impl,
        "vllm": deployer._deploy_vllm_impl,
        "sglang": deployer._deploy_sglang_impl,
    }
    deploy[deployer.backend]()
