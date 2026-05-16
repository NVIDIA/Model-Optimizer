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

"""Extract hidden states from an LLM using vLLM's ExampleHiddenStatesConnector."""

import argparse
import tempfile
from pathlib import Path

import torch
from common import (
    add_answer_only_loss_args,
    add_aux_layers_args,
    load_chat_template,
    resolve_aux_layers,
    tokenize_with_loss_mask,
    verify_generation_tags,
)
from datasets import load_dataset
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect hidden states from conversations using vLLM."
    )
    parser.add_argument("--model", type=str, required=True, help="Name or path of the model.")
    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="Path to a .jsonl file or directory containing .jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save hidden states as .pt files.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size. Defaults to 1.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=3072,
        help="Maximum number of tokens per conversation. Longer ones are skipped. Defaults to 3072.",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=10,
        help="Minimum number of tokens per conversation. Shorter ones are skipped. Defaults to 10.",
    )
    parser.add_argument(
        "--debug-max-num-conversations",
        type=int,
        default=None,
        help="For debugging: limit total conversations processed.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Set trust_remote_code for Huggingface models and tokenizers",
    )
    add_aux_layers_args(parser)
    add_answer_only_loss_args(parser)
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
            f"input_data must be a .jsonl file or directory of .jsonl files, got: {args.input_data}"
        )
    print(f"Loaded {len(dataset)} conversations from {args.input_data}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Skip already processed conversations
    def keep_conversation(entry):
        conversation_id = entry.get("conversation_id", entry.get("uuid", None))
        assert conversation_id is not None, "Each entry must have a conversation_id or uuid field"
        return not (args.output_dir / f"{conversation_id}.pt").exists()

    original_num = len(dataset)
    dataset = dataset.filter(keep_conversation)
    print(f"Removed {original_num - len(dataset)} already-processed conversations")

    if args.debug_max_num_conversations is not None:
        dataset = dataset.select(range(args.debug_max_num_conversations))

    # Determine aux layer indices per --aux-layers flag.
    # Convention bridge: resolve_aux_layers returns 0-based transformer layer
    # IDs (HF: outputs.hidden_states[lid + 1] = output of layer lid). vLLM's
    # `aux_hidden_state_layers` is checked against `idx + 1` after layer idx,
    # so the index there is also "lid + 1" — i.e. shift HF's lid by +1.
    # Last-layer capture: HF puts the post-final-norm result at hidden_states[N];
    # vLLM exposes the same position (idx+1 == N after layer N-1) but stores the
    # *pre-norm* residual stream there, which is fine for our consumer below.
    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    num_hidden_layers = hf_config.num_hidden_layers
    aux_layer_ids = resolve_aux_layers(args, num_hidden_layers)
    aux_capture_ids_vllm = [lid + 1 for lid in aux_layer_ids]
    # All layers to capture: shifted aux layers + final-layer position N
    all_capture_ids = sorted({*aux_capture_ids_vllm, num_hidden_layers})
    print(
        f"Model has {num_hidden_layers} hidden layers; "
        f"aux layer ids (HF 0-based): {aux_layer_ids}, "
        f"vLLM capture ids: {all_capture_ids}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    override_template = load_chat_template(args.chat_template)
    if override_template is not None:
        tokenizer.chat_template = override_template
    if tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(REMOVE_THINK_CHAT_TEMPLATE, "")
    if args.answer_only_loss:
        verify_generation_tags(tokenizer.chat_template)

    # Tokenize and filter conversations
    token_id_list = []
    conversation_ids = []
    loss_masks_by_id: dict[str, torch.Tensor] = {}
    num_skipped_too_long = 0
    num_invalid = 0

    for entry in dataset:
        conversation_id = entry.get("conversation_id", entry.get("uuid"))
        conversations = entry.get("messages") or entry.get("conversations")
        if not conversations or not isinstance(conversations, list):
            num_invalid += 1
            continue

        # Single apply_chat_template call produces both input_ids and loss_mask,
        # guaranteeing they come from the same tokenization.
        input_ids, loss_mask = tokenize_with_loss_mask(
            tokenizer, conversations, args.answer_only_loss
        )
        num_tokens = input_ids.shape[-1]
        if num_tokens < args.min_seq_len or num_tokens > args.max_seq_len:
            num_skipped_too_long += 1
            continue

        token_id_list.append(input_ids.squeeze(0).tolist())
        conversation_ids.append(conversation_id)
        loss_masks_by_id[conversation_id] = loss_mask

    print(
        f"Tokenized {len(token_id_list)} conversations "
        f"(skipped {num_skipped_too_long} by length, {num_invalid} invalid)"
    )

    if not token_id_list:
        print("No conversations to process.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        llm = LLM(
            model=args.model,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": all_capture_ids,
                    }
                },
            },
            kv_transfer_config={
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": tmpdir,
                },
            },
            tensor_parallel_size=args.tp,
            trust_remote_code=args.trust_remote_code,
        )

        sampling_params = SamplingParams(max_tokens=1)
        outputs = llm.generate(
            [{"prompt_token_ids": ids} for ids in token_id_list],
            sampling_params,
        )

        num_success = 0
        for output, conversation_id in tqdm(
            zip(outputs, conversation_ids),
            total=len(outputs),
            desc="Saving hidden states",
        ):
            hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
            if hidden_states_path is None:
                print(
                    f"Warning: no hidden_states_path for conversation {conversation_id}, skipping"
                )
                continue

            with safe_open(hidden_states_path, framework="pt") as f:
                token_ids_tensor = f.get_tensor("token_ids")
                # Shape from vLLM: [seq_len, num_captured_layers, hidden_dim]
                hidden_states_tensor = f.get_tensor("hidden_states")

            # Last captured layer (= last model layer N-1) -> output hidden states
            # Earlier captured layers -> aux hidden states, concatenated along hidden dim
            output_hidden_states = hidden_states_tensor[:, -1, :]  # [seq_len, hidden_dim]
            aux_hidden_states = hidden_states_tensor[:, :-1, :].reshape(
                hidden_states_tensor.shape[0], -1
            )  # [seq_len, hidden_dim * num_aux_layers]

            # Align loss_mask with the token length returned by vLLM: if vLLM
            # truncated, truncate; if it somehow grew (shouldn't happen), pad with 1s
            # so that tail positions remain trainable under non-answer-only runs.
            vllm_seq_len = token_ids_tensor.shape[0]
            loss_mask = loss_masks_by_id[conversation_id]
            if loss_mask.shape[0] > vllm_seq_len:
                loss_mask = loss_mask[:vllm_seq_len]
            elif loss_mask.shape[0] < vllm_seq_len:
                pad = torch.ones(vllm_seq_len - loss_mask.shape[0], dtype=loss_mask.dtype)
                loss_mask = torch.cat([loss_mask, pad], dim=0)

            output_file = args.output_dir / f"{conversation_id}.pt"
            torch.save(
                {
                    "input_ids": token_ids_tensor.to(torch.int64),
                    "hidden_states": output_hidden_states,
                    "aux_hidden_states": aux_hidden_states,
                    "loss_mask": loss_mask,
                    "conversation_id": conversation_id,
                },
                output_file,
            )
            num_success += 1

    print(f"Successfully saved {num_success} / {len(token_id_list)} conversations.")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
