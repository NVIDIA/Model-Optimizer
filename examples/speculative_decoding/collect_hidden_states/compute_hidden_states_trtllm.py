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

"""Extract hidden states from an HF-compatible LLM."""

import os

os.environ["TLLM_LOG_LEVEL"] = "error"
import argparse
import asyncio
from pathlib import Path

import torch
from datasets import load_dataset
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, SaveHiddenStatesDecodingConfig
from tqdm import tqdm as tqdm
from transformers import AutoConfig, AutoTokenizer

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Collect hidden states from conversations
        by running full conversations through a Hugging Face model."""
    )

    ## Model & Generation Parameters ##
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the served model.",
    )

    ## Client Parameters ##
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=3072,
        help="""Maximum number of tokens in a conversation. Longer conversations will be skipped.
        Defaults to 3072 tokens.""",
    )

    ## I/O Parameters ##
    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="""Path to the `jsonl` file or directory containing `jsonl` files.""",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="""Root directory in which to save the hidden states.
        The data will be saved as a torch (`.pt`) dump file for each conversation.""",
    )
    parser.add_argument(
        "--debug-max-num-conversations",
        type=int,
        default=None,
        help="""For debugging purposes, limit the number of conversations processed.
        Default is None, meaning no limit.""",
    )
    parser.add_argument(
        "--dp-rank",
        type=int,
        default=0,
        help="""Data parallel rank. TASK_ID on SLURM.""",
    )
    parser.add_argument(
        "--dp-world-size",
        type=int,
        default=1,
        help="""Data parallel world size. Number of tasks on SLURM.""",
    )
    parser.add_argument(
        "--use-cuda-graph",
        type=bool,
        default=True,
        help="""Whether to use CUDA graph.""",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="""tensor_parallel_size for TRTLLM.""",
    )
    # moe_ep * moe_tp * moe_cp should be equal to tp
    # REF: https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html
    parser.add_argument(
        "--moe-ep",
        type=int,
        default=None,
        help="""moe_expert_parallel_size for TRTLLM.""",
    )
    parser.add_argument(
        "--moe-tp",
        type=int,
        default=None,
        help="""moe_tensor_parallel_size for TRTLLM.""",
    )
    parser.add_argument(
        "--moe-cp",
        type=int,
        default=None,
        help="""moe_cluster_parallel_size for TRTLLM.""",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Load conversations
    if args.input_data.is_file() and str(args.input_data).endswith(".jsonl"):
        dataset = load_dataset("json", data_files=str(args.input_data), split="train")
    elif args.input_data.is_dir():
        dataset = load_dataset(
            "json", data_files={"train": f"{args.input_data}/*.jsonl"}, split="train"
        )
    else:
        raise ValueError(
            f"input_data must be a .jsonl file or directory containing .jsonl files, got: {args.input_data}"
        )
    print(f"Loaded {len(dataset)} conversations from {args.input_data}")

    # Shard data
    if args.dp_world_size > 1:
        dataset = dataset.shard(num_shards=args.dp_world_size, index=args.dp_rank)
    print(
        f"Sharded dataset to {len(dataset)} conversations for DP#{args.dp_rank}/{args.dp_world_size}"
    )

    # Remove already dumped conversations
    def keep_conversation(entry):
        conversation_id = entry.get("conversation_id", entry.get("uuid", None))
        assert conversation_id is not None, "conversation_id is required"
        output_file = args.output_dir / f"{conversation_id}.pt"
        return not output_file.exists()

    original_num = len(dataset)
    dataset = dataset.filter(keep_conversation)
    print(
        "Removed",
        original_num - len(dataset),
        "conversations due to existing output files",
    )

    # For debugging
    if args.debug_max_num_conversations is not None:
        dataset = dataset.select(range(args.debug_max_num_conversations))

    # Get model config and tokenizer
    model_config = AutoConfig.from_pretrained(args.model)
    num_hidden_layers = getattr(model_config, "num_hidden_layers", None)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = tokenizer.chat_template.replace(REMOVE_THINK_CHAT_TEMPLATE, "")

    # Set up LLM
    llm_common_config = {
        "model": args.model,
        "attn_backend": "TRTLLM",
        "disable_overlap_scheduler": False,
        "cuda_graph_config": CudaGraphConfig(batch_sizes=[1, 2, 4])
        if args.use_cuda_graph
        else None,
        "max_batch_size": 16,
        "kv_cache_config": KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.5),
        "enable_chunked_prefill": False,
        "tensor_parallel_size": args.tp,
        "moe_expert_parallel_size": args.moe_ep,
        "moe_tensor_parallel_size": args.moe_tp,
        "moe_cluster_parallel_size": args.moe_cp,
    }
    spec_config = {
        "output_directory": str(args.output_dir),
        "write_interval": 1,
        "file_prefix": f"dp_{args.dp_rank}",
        "eagle3_layers_to_capture": {1, num_hidden_layers // 2 - 1, num_hidden_layers - 4},
    }
    sampling_params = SamplingParams(max_tokens=32, temperature=0)

    llm_spec = LLM(
        **llm_common_config, speculative_config=SaveHiddenStatesDecodingConfig(**spec_config)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_skipped_too_long = 0
    num_invalid = 0
    num_success = 0
    pbar = tqdm(total=len(dataset), desc=f"DP#{args.dp_rank} Processing conversations")

    async def _post_process_trtllm_dumped(trtllm_dumped_file: str, conversation_id: int):
        """
        Post-process the TRTLLM dumped file to same format as HF dumped:
        1. Remove id field, replace it with conversation_id
        2. Rename hidden_state field to hidden_states
        3. From list of length 1 to dict
        4. Rename file to conversation_id.pt
        """
        if not trtllm_dumped_file.exists():
            return False
        with open(trtllm_dumped_file, "rb") as f:
            trtllm_dumped = torch.load(f)
        assert isinstance(trtllm_dumped, list) and len(trtllm_dumped) == 1, (
            "TRTLLM dumped should be a list with one element"
        )
        assert (
            isinstance(trtllm_dumped[0], dict)
            and "id" in trtllm_dumped[0]
            and "hidden_state" in trtllm_dumped[0]
        ), "TRTLLM dumped should have an 'id' and 'hidden_states' field"
        trtllm_dumped = trtllm_dumped[0]
        trtllm_dumped.pop("id")
        trtllm_dumped["conversation_id"] = conversation_id
        trtllm_dumped["hidden_states"] = trtllm_dumped.pop("hidden_state")
        output_file = args.output_dir / f"{conversation_id}.pt"
        with open(output_file, "wb") as f:
            torch.save(trtllm_dumped, f)
        trtllm_dumped_file.unlink()
        return True

    async def dump_hidden_states(idx: int, conversation_id: int, input_ids: list[int]):
        nonlocal num_success
        await llm_spec.generate_async(input_ids, sampling_params)
        # TRTLLM API name files starts from 1
        # ref:https://github.com/NVIDIA/TensorRT-LLM/pull/7012
        trtllm_dumped_file = args.output_dir / f"{spec_config['file_prefix']}_{idx + 1}.pt"
        dump_success = await _post_process_trtllm_dumped(trtllm_dumped_file, conversation_id)
        num_success += int(dump_success)
        pbar.update(1)

    async def submit_generates():
        nonlocal num_skipped_too_long
        nonlocal num_invalid
        tasks = []
        idx = 0
        for entry in dataset:
            conversation_id = entry.get("conversation_id", entry.get("uuid"))

            conversations = entry["conversations"]
            if not conversations or not isinstance(conversations, list):
                num_invalid += 1
                continue

            input_ids = tokenizer.apply_chat_template(conversations, add_generation_template=False)
            num_input_tokens = (
                input_ids.shape[1] if isinstance(input_ids, torch.Tensor) else len(input_ids)
            )
            if num_input_tokens <= 10 or num_input_tokens > args.max_seq_len:
                num_skipped_too_long += 1
                continue

            tasks.append(dump_hidden_states(idx, conversation_id, input_ids))
            # Increment only for valid conversations to match dump file index
            idx += 1
        await asyncio.gather(*tasks)

    asyncio.run(submit_generates())
    llm_spec.shutdown()
    print("LLM shutdown")

    if num_skipped_too_long > 0:
        print(f"Skipped {num_skipped_too_long} conversations due to length constraints.")
    if num_invalid > 0:
        print(f"Skipped {num_invalid} invalid conversations without proper fields.")

    if num_success == len(dataset):
        print(f"Successfully processed all {num_success} conversations.")
    else:
        print(f"Successfully processed {num_success} out of {len(dataset)} conversations.")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
