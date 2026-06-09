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

"""Extract hidden states from an LLM using vLLM's native hidden-state extractor.

This uses vLLM's built-in ``extract_hidden_states`` speculative method together with
the ``ExampleHiddenStatesConnector`` KV connector, so no third-party data-generation
dependency (e.g. ``speculators``) is required. Because the same ``eagle_aux_hidden_state_layer_ids``
convention is used at EAGLE3 deployment time in vLLM, the captured aux layers match
deployment by construction.

See https://docs.vllm.ai/en/stable/features/speculative_decoding/extract_hidden_states/
"""

import argparse
from pathlib import Path

import torch
from common import (
    add_answer_only_loss_args,
    add_aux_layers_args,
    load_chat_template,
    tokenize_with_loss_mask,
    verify_generation_tags,
)
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)


def _resolve_aux_layers_standalone(aux_layers: str, num_hidden_layers: int) -> list[int]:
    """Resolve aux-layer ids without importing modelopt.

    This dump runs in a stock vLLM container. ``common.resolve_aux_layers`` resolves the
    'dflash'/'eagle' presets by importing ``modelopt.torch.speculative.plugins`` — which
    pulls in the full ``modelopt.torch`` init chain (omegaconf, etc.) that the vLLM
    container does not have, so the import fails. Resolve the 'dflash' preset inline
    (mirroring ``modeling_dflash.build_target_layer_ids`` with num_draft=5, the recipe
    default) and accept an explicit comma-separated int list. Keep in sync with modelopt.
    """
    spec = aux_layers.strip().lower()
    if spec == "dflash":
        num_draft = 5
        if num_draft == 1:
            return [num_hidden_layers // 2]
        start = min(1, num_hidden_layers - 1)
        end = max(start, num_hidden_layers - 3)
        span = end - start
        return sorted({round(start + (i * span) / (num_draft - 1)) for i in range(num_draft)})
    ids = sorted({int(t) for t in aux_layers.split(",") if t.strip()})
    if not ids:
        raise ValueError(
            f"--aux-layers={aux_layers!r}: in the stock vLLM container (no modelopt) only the "
            "'dflash' preset or an explicit comma-separated layer-id list are supported."
        )
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Collect hidden states from conversations using vLLM's native extractor."""
    )

    parser.add_argument("--model", type=str, required=True, help="HF model path.")
    parser.add_argument(
        "--max-seq-len", type=int, default=3072, help="Max tokens per conversation."
    )
    parser.add_argument(
        "--input-data", type=Path, required=True, help="Path to jsonl file or directory."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory to save hidden states."
    )
    parser.add_argument("--dp-rank", type=int, default=0, help="Data parallel rank.")
    parser.add_argument("--dp-world-size", type=int, default=1, help="Data parallel world size.")
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="Trust remote code for HF models."
    )
    parser.add_argument("--tp", type=int, default=None, help="Tensor parallel size.")
    parser.add_argument(
        "--debug-max-num-conversations", type=int, default=None, help="Limit conversations."
    )
    add_aux_layers_args(parser)
    add_answer_only_loss_args(parser)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Import lazily so --help and arg parsing work without vLLM installed.
    from vllm import LLM, SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector
    from vllm.inputs import TokensPrompt

    # Load conversations
    if args.input_data.is_file() and str(args.input_data).endswith(".jsonl"):
        dataset = load_dataset("json", data_files=str(args.input_data), split="train")
    elif args.input_data.is_dir():
        dataset = load_dataset(
            "json", data_files={"train": f"{args.input_data}/*.jsonl"}, split="train"
        )
    else:
        raise ValueError(f"input_data must be a .jsonl file or directory, got: {args.input_data}")
    print(f"Loaded {len(dataset)} conversations from {args.input_data}")

    # Shard data
    if args.dp_world_size > 1:
        dataset = dataset.shard(num_shards=args.dp_world_size, index=args.dp_rank)
    print(f"Sharded to {len(dataset)} conversations for DP#{args.dp_rank}/{args.dp_world_size}")

    # Remove already dumped conversations
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def keep_conversation(entry):
        conversation_id = entry.get("conversation_id", entry.get("uuid", None))
        assert conversation_id is not None, "conversation_id is required"
        return not (output_dir / f"{conversation_id}.pt").exists()

    original_num = len(dataset)
    dataset = dataset.filter(keep_conversation)
    print(f"Removed {original_num - len(dataset)} conversations due to existing output files")

    if args.debug_max_num_conversations is not None:
        dataset = dataset.select(range(args.debug_max_num_conversations))

    # Resolve the aux-layer indices and append the final-layer output. vLLM saves the
    # final (un-normed) hidden state when ``num_hidden_layers`` is passed as a layer id.
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        raise ValueError(f"model config has no 'num_hidden_layers' attribute: {config}")
    aux_layer_ids = _resolve_aux_layers_standalone(args.aux_layers, num_hidden_layers)
    # The trailing entry is the final output hidden state; the rest are aux layers.
    extract_layer_ids = [*aux_layer_ids, num_hidden_layers]
    print(f"Extracting hidden states from layers {extract_layer_ids} (last = final output)")

    # Tokenize conversations
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    override_template = load_chat_template(args.chat_template)
    if override_template is not None:
        tokenizer.chat_template = override_template
    if tokenizer.chat_template is not None:
        tokenizer.chat_template = tokenizer.chat_template.replace(REMOVE_THINK_CHAT_TEMPLATE, "")
    if args.answer_only_loss:
        verify_generation_tags(tokenizer.chat_template)

    # Prepare prompts for vLLM
    prompts = []
    conversation_ids = []
    loss_masks = []
    num_skipped_too_long = 0
    num_invalid = 0

    for entry in dataset:
        conversation_id = entry.get("conversation_id", entry.get("uuid"))
        conversations = entry["conversations"]
        if not conversations or not isinstance(conversations, list):
            num_invalid += 1
            continue

        # One apply_chat_template call yields aligned input_ids + loss_mask. With
        # --answer-only-loss the mask comes from the template's {% generation %} tags;
        # otherwise it is all-ones. Same tokens are sent to vLLM, so the dumped hidden
        # states line up with this loss_mask 1:1 (prefix caching is disabled below).
        input_ids, loss_mask = tokenize_with_loss_mask(
            tokenizer, conversations, args.answer_only_loss
        )
        input_ids = input_ids.squeeze(0)
        num_tokens = input_ids.shape[0]
        if num_tokens <= 10 or num_tokens > args.max_seq_len:
            num_skipped_too_long += 1
            continue

        prompts.append(TokensPrompt(prompt_token_ids=input_ids.tolist()))
        conversation_ids.append(conversation_id)
        loss_masks.append(loss_mask)

    print(
        f"Prepared {len(prompts)} prompts ({num_skipped_too_long} skipped too long, {num_invalid} invalid)"
    )

    if len(prompts) == 0:
        print("No prompts to process.")
        return

    # Initialize vLLM with the native hidden-state extractor.
    tp = args.tp if args.tp is not None else torch.cuda.device_count()
    storage_path = output_dir / ".vllm_hidden_states"
    storage_path.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp,
        max_model_len=args.max_seq_len,
        trust_remote_code=args.trust_remote_code,
        enable_chunked_prefill=False,  # required by extract_hidden_states
        # With prefix caching on, vLLM serves shared prefixes from cache in block-sized
        # chunks and the hidden-state connector only emits the freshly-computed suffix, so
        # the dumped hidden_states come out short by N*block_size vs the full input_ids /
        # loss_mask. Disabling it forces a full prefill so every token's state is dumped.
        enable_prefix_caching=False,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": extract_layer_ids},
            },
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "shared_storage_path": str(storage_path),
                "use_synchronization_lock": False,  # batch generation, no concurrent readers
            },
        ),
    )

    # max_tokens=1: we only need a single forward pass over the prompt tokens.
    outputs = llm.generate(prompts, SamplingParams(max_tokens=1))

    # Save in the same format as compute_hidden_states_hf.py, including loss_mask.
    num_success = 0
    for conv_id, loss_mask, output in tqdm(
        zip(conversation_ids, loss_masks, outputs), total=len(outputs), desc="Saving"
    ):
        hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
        if hidden_states_path is None:
            print(f"WARNING: no hidden_states_path for conversation {conv_id}; skipping")
            continue

        obj = example_hidden_states_connector.load_hidden_states(hidden_states_path)
        token_ids = obj["token_ids"]
        # hidden_states: [num_tokens, num_extracted_layers, hidden_size], ordered to match
        # extract_layer_ids. Last layer = final output; the rest = aux layers.
        hidden_states = obj["hidden_states"]

        output_hidden_states = hidden_states[:, -1, :].cpu()
        if hidden_states.shape[1] > 1:
            # Concatenate aux layers along the hidden dim, matching the HF dump format.
            aux = hidden_states[:, :-1, :].cpu()
            aux_hidden_states = aux.reshape(aux.shape[0], -1)
        else:
            aux_hidden_states = torch.empty(0)

        output_file = output_dir / f"{conv_id}.pt"
        with open(output_file, "wb") as f:
            torch.save(
                {
                    "input_ids": token_ids.cpu(),
                    "hidden_states": output_hidden_states,
                    "aux_hidden_states": aux_hidden_states,
                    "loss_mask": loss_mask[: output_hidden_states.shape[0]].cpu(),
                    "conversation_id": conv_id,
                },
                f,
            )
        example_hidden_states_connector.cleanup_hidden_states(hidden_states_path)
        num_success += 1

    print(f"Successfully processed {num_success} out of {len(prompts)} conversations.")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
