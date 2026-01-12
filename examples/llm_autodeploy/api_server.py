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

import argparse
import sys
import time
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from tensorrt_llm._torch.auto_deploy import LLM, AutoDeployConfig
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    UsageInfo,
)

import modelopt.torch.opt as mto

# global vars
app = FastAPI()
model_runner = None
args = None
sampling_params = None
model = "autodeploy_demo"


def build_runner_from_config(args) -> LLM:
    """Builds a model runner from our config."""
    mto.enable_huggingface_checkpointing()
    model_kwargs = {"max_position_embeddings": args.max_seq_len, "use_cache": False}
    build_config = BuildConfig(max_seq_len=args.max_seq_len, max_batch_size=args.max_batch_size)
    build_config.plugin_config.tokens_per_block = args.max_seq_len

    # setup AD config
    ad_config = AutoDeployConfig(
        model=args.ckpt_path,
        compile_backend=args.compile_backend,
        device=args.device,
        world_size=args.world_size,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        max_num_tokens=args.max_num_tokens,
        model_kwargs=model_kwargs,
        attn_backend="triton",
    )
    llm = LLM(**ad_config.to_llm_kwargs())

    return llm


def apply_stop_tokens(text: str, stop_words: list[str] | None) -> str:
    """Truncate text at the first occurrence of any stop token."""
    if not stop_words:
        return text  # No stop tokens provided, return as is

    for stop in stop_words:
        stop_idx = text.find(stop)
        if stop_idx != -1:
            return text[:stop_idx]  # Truncate at the first stop token

    return text  # No stop token found, return original text


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """Endpoint to handle completion requests."""
    global model_runner, model, sampling_params

    if model_runner is None:
        raise HTTPException(status_code=500, detail="Runner is not initialized")

    # Run inference using the model_runner
    if isinstance(request.prompt, str):
        prompts = [request.prompt]  # Single string becomes a list with one element
    elif isinstance(request.prompt, list):
        if all(isinstance(p, str) for p in request.prompt):  # List of strings
            prompts = request.prompt
    else:
        raise HTTPException(status_code=400, detail="Invalid prompt type")
    sampling_params.temperature = request.temperature
    outs = model_runner.generate(prompts, sampling_params)

    # formatting outputs
    outputs = []
    if isinstance(outs, RequestOutput):
        outs = [outs]
    for i, out in enumerate(outs):
        outputs.append({"prompt": out.prompt, "text": out.outputs[0].text})

    # Generate unique ID
    unique_id = str(uuid.uuid4())

    # Generate timestamp
    created_timestamp = int(time.time())

    # Construct response
    response = CompletionResponse(
        id=unique_id,
        object="text_completion",
        created=created_timestamp,
        model=model,
        choices=[
            CompletionResponseChoice(
                index=i,
                text=apply_stop_tokens(output["text"], request.stop),
                stop_reason="stop"
                if any(st in output["text"] for st in request.stop or [])
                else "length",
            )
            for i, output in enumerate(outputs)
        ],
        usage=UsageInfo(),
    )
    return response


def get_server_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--ckpt_path",
        help="Specify where the HF checkpoint path is.",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=("Target device to host the model."),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="torch-opt",
        help=("backend to compile to model."),
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=0,
        help=("target world size for hosting the model."),
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help=("max dimension for statically allocated kv cache"),
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help=("max sequence length for inference/cache"),
    )
    parser.add_argument(
        "--max_num_tokens",
        type=int,
        default=128,
        help=("max tokens to generate."),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help=("top_k for output sampling."),
    )
    parser.add_argument(
        "--compile_backend",
        type=str,
        default="torch-opt",
        help=("backend to compile the torch graph."),
    )
    return parser.parse_args()


def run_server():
    try:
        global model_runner, args, sampling_params
        args = get_server_args()
        model_runner = build_runner_from_config(args)
        sampling_params = SamplingParams(
            max_tokens=args.max_num_tokens,
            top_k=args.top_k,
            temperature=1.0,  # default value, we will take temperature from requests
        )

        uvicorn.run(app, host=args.host, port=args.port)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_server()
