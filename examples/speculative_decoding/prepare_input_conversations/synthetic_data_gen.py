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

"""Generate synthetic conversational data by querying OpenAI-compatible endpoints."""

import argparse
import asyncio
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import transformers
from openai import AsyncOpenAI
from tqdm import tqdm as tqdm_sync

import asyncio

DEBUG_MODE = False

def debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)

class AsyncPrioritySemaphore:
    """An asyncio Semaphore that respects task priority.
    Waiters acquire the semaphore in priority order. A lower number
    represents a higher priority. Waiters with the same priority are
    served in a first-in, first-out (FIFO) order.
    This implementation is not a subclass of asyncio.Semaphore but provides a
    compatible interface.
    """

    def __init__(self, value: int = 1) -> None:
        """Initialize the semaphore with a given initial value."""
        if value < 0:
            msg = "Semaphore initial value must be >= 0"
            raise ValueError(msg)
        self._count = value
        self._lock = asyncio.Lock()
        self._waiters = asyncio.PriorityQueue()
        self._fifo_counter = 0

    async def acquire(self, priority: int = 0) -> None:
        """Acquire the semaphore, blocking if necessary, respecting priority.
        Args:
            priority (int): The priority of the acquiring task.
                           Lower numbers are higher priority. Defaults to 0.
        """
        async with self._lock:
            if self._count > 0:
                self._count -= 1
                return

            self._fifo_counter += 1
            future = asyncio.get_running_loop().create_future()
            await self._waiters.put((priority, self._fifo_counter, future))

        try:
            await future
        except asyncio.CancelledError:
            # If the waiting task is cancelled, we must try to remove its
            # future from the queue to prevent memory leaks. This is a
            # best-effort attempt. The release() method also handles
            # cancelled futures.
            if not future.done():
                future.cancel()
            raise

    async def release(self) -> None:
        """Release the semaphore, waking up the highest-priority waiter if any."""
        async with self._lock:
            self._count += 1
            while not self._waiters.empty():
                _, _, future = self._waiters.get_nowait()

                if future.cancelled():
                    continue

                future.set_result(True)
                self._count -= 1
                break

    def locked(self) -> bool:
        """Returns True if the semaphore cannot be acquired immediately."""
        return self._count == 0

    @property
    def value(self) -> int:
        """The current semaphore count."""
        return self._count

    async def __aenter__(self) -> "AsyncPrioritySemaphore":
        """Acquire the semaphore when entering the context manager."""
        await self.acquire(priority=0)
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        """Release the semaphore when exiting the context manager."""
        await self.release()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Generate synthetic conversational data
        by sending prompts to one or more OpenAI-compatible HTTP endpoints"""
    )

    ## Model & Generation Parameters ##
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the served model.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="""Maximum number of tokens to allow in the prompt.
        If provided, the model's tokenizer is used to count tokens and
        skip prompts that exceed this limit.""",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to allow in the completion.",
    )

    ## Client Parameters ##
    parser.add_argument(
        "--base-urls",
        type=str,
        default="http://localhost:8000/v1",
        help="""Comma-separated list of base URLs for OpenAI-compatible endpoints
        (e.g., http://localhost:8000/v1,http://localhost:8001/v1).
        If more than one URL is provided,
        the script will round-robin requests across them.""",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2048,
        help="Maximum concurrent requests allowed per client.",
    )
    parser.add_argument(
        "--openai-api-key",
        default="EMPTY",
        help="""Access key required by the OpenAI Python client
        (not required for local serving engines like vLLM).""",
    )

    ## I/O Parameters ##
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("input_conversations/train.jsonl"),
        help="""Path to the input `jsonl` file containing input conversations.
        Default is 'input_conversations/train.jsonl'.
        Alternatively, you can specify a directory and all `jsonl` files will be processed.
        Only the first turn will be used as a prompt.""",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("synthetic_conversations/"),
        help="""Directory to save the output `jsonl` files.
        Default is 'synthetic_conversations/'.
        This directory will be created if it does not exist, and will also contain checkpoints
        for partial results and resumed runs in the event of a crash.
        Output files will have the same names as the inputs.
        """,
    )
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=10,
        help="""Number of checkpoints to save during processing.
        Each checkpoint will contain a portion of the processed conversations.
        Default number of checkpoints is 10.""",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="""If set, checkpoint files will be retained after successful processing. Otherwise,
        they will be deleted to save space as long as fewer than 1%% of requests failed.""",
    )
    parser.add_argument(
        "--debug-max-num-conversations-per-file",
        type=int,
        default=None,
        help="""For debugging purposes,
        limit the number of conversations processed per file.
        If set, only this many conversations will be processed from each file.
        Default is None, meaning no limit.""",
    )

    return parser.parse_args()


## Checkpointing Logic ##


@dataclass
class CheckpointRecord:
    """One output entry for a processed conversation."""

    conversation_id: str
    conversations: list[dict]
    temperature: float
    reasoning_effort: str | None
    enable_reasoning: bool | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class CheckpointState:
    """State for managing checkpointing during conversation processing."""

    # Index of this checkpoint relative to total number of checkpoints
    index: int

    # Number of requests expected for this checkpoint
    num_requests_for_checkpoint: int

    # Path to save the checkpoint file
    checkpoint_output_file: Path

    # List of processed conversation outputs
    outputs: list[CheckpointRecord | None] = field(default_factory=list)


def load_existing_conversation_ids(checkpoint_file: Path) -> list[str]:
    existing_ids = []
    if not checkpoint_file.exists():
        return existing_ids
    with checkpoint_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if not entry or "conversation_id" not in entry:
                continue
            existing_ids.append(entry["conversation_id"])

    return existing_ids


def write_checkpoint(
    checkpoint_state: CheckpointState,
) -> None:
    output_entries = []
    for record in checkpoint_state.outputs:
        if record is None:
            continue
        output_entries.append(
            {"conversation_id": record.conversation_id, "conversations": record.conversations, "temperature": record.temperature, "reasoning_effort": record.reasoning_effort, "enable_reasoning": record.enable_reasoning, "metadata": record.metadata}
        )

    if not output_entries:
        print(
            "No valid responses to write. "
            f"Skipping checkpoint at {checkpoint_state.checkpoint_output_file!s}"
        )
        return
    with checkpoint_state.checkpoint_output_file.open("a", encoding="utf-8") as f:
        for entry in output_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def add_result_to_checkpoint(
    conversation_obj: CheckpointRecord | None,
    checkpoint_state: CheckpointState,
) -> None:
    checkpoint_state.outputs.append(conversation_obj)
    if conversation_obj is not None:
        debug_print(f"Checkpoint {checkpoint_state.index}: now at {len(checkpoint_state.outputs)} / {checkpoint_state.num_requests_for_checkpoint}. Added conversation {conversation_obj.conversation_id}")
    if len(checkpoint_state.outputs) == checkpoint_state.num_requests_for_checkpoint:
        debug_print(f"Checkpoint {checkpoint_state.index} reached capacity ({checkpoint_state.num_requests_for_checkpoint}). Writing to disk...")
        write_checkpoint(checkpoint_state)


@dataclass
class InputPromptData:
    """Data structure for an input prompt to be sent to the model."""

    prompt_conversation: list[dict]
    conversation_id: str
    source_filename: str
    temperature: float
    reasoning_effort: str | None = None
    enable_reasoning: bool | None = None
    metadata: dict = field(default_factory=dict)


async def query_and_process_response(
    client: AsyncOpenAI,
    semaphore: AsyncPrioritySemaphore,
    prompt_data: InputPromptData,
    out_checkpoint_state: CheckpointState,
    args: argparse.Namespace,
    pbar: tqdm_sync,
) -> int:
    """Send a completion request while respecting per-client concurrency."""
    await semaphore.acquire(priority=out_checkpoint_state.index)
    if not hasattr(pbar, "_success_metadata"):
        pbar._success_metadata = {"successful_responses": 0, "total_responses": 0}
    success_rate = 100
    if pbar._success_metadata["total_responses"] > 0:
        success_rate = (
            pbar._success_metadata["successful_responses"]
            / pbar._success_metadata["total_responses"]
            * 100
        )
    success_rate = f"{success_rate:.1f}%"
    pbar.set_postfix({"success_rate": success_rate})
    try:
        kwargs = {}
        if prompt_data.reasoning_effort is not None:
            kwargs["reasoning_effort"] = prompt_data.reasoning_effort
        if prompt_data.enable_reasoning is not None:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": prompt_data.enable_reasoning}}
        debug_print(f"Checkpoint {out_checkpoint_state.index}: Sending request for conversation {prompt_data.conversation_id}...")
        response = await client.chat.completions.create(
            model=args.model,
            messages=prompt_data.prompt_conversation,
            temperature=prompt_data.temperature,
            max_tokens=args.max_completion_tokens,
            **kwargs
        )
    except Exception as e:
        debug_print(f"Error during request: {e}")
        response = None
    finally:
        await semaphore.release()

    if response is None or not response.choices:
        # Request failed or returned empty response
        debug_print("Response failed: ", response)
        add_result_to_checkpoint(None, out_checkpoint_state)
        pbar._success_metadata["total_responses"] += 1
        pbar.update(1)
        return 0
    
    reasoning_content = response.choices[0].message.reasoning
    content = response.choices[0].message.content

    response_text = ""
    if reasoning_content:
        response_text = "<think>" + reasoning_content
        if content: 
            response_text += "</think>"
    elif prompt_data.enable_reasoning and content:
        response_text += "<think>"
    if content:
        response_text += content
    if not response_text.strip():
        debug_print("Empty response content.")
        add_result_to_checkpoint(None, out_checkpoint_state)
        pbar._success_metadata["total_responses"] += 1
        pbar.update(1)
        return 0

    updated_conversation = [
        *prompt_data.prompt_conversation,
        {"role": "assistant", "content": response_text},
    ]
    output_obj = CheckpointRecord(
        conversation_id=prompt_data.conversation_id,
        conversations=updated_conversation,
        temperature=prompt_data.temperature,
        reasoning_effort=prompt_data.reasoning_effort,
        enable_reasoning=prompt_data.enable_reasoning,
        metadata=prompt_data.metadata,
    )
    add_result_to_checkpoint(output_obj, out_checkpoint_state)
    pbar._success_metadata["total_responses"] += 1
    pbar._success_metadata["successful_responses"] += 1
    pbar.update(1)

    return 1  # Indicate that a valid response was processed


async def main(args: argparse.Namespace) -> None:
    client_base_paths = args.base_urls.split(",")
    if not client_base_paths:
        msg = "No base URLs provided for OpenAI clients."
        raise ValueError(msg)

    # Initialize clients and synchronization primitives (for concurrency control)
    clients: list[AsyncOpenAI] = []
    semaphores: list[asyncio.Semaphore] = []
    for i in range(len(client_base_paths)):
        base_url = client_base_paths[i]
        custom_http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=2048, max_keepalive_connections=1024),
            timeout=httpx.Timeout(timeout=None),
        )
        clients.append(
            AsyncOpenAI(
                base_url=base_url,
                api_key=args.openai_api_key,
                http_client=custom_http_client,
            )
        )
        semaphores.append(AsyncPrioritySemaphore(args.max_concurrent))

    # Load tokenizer to count prompt tokens
    if args.max_prompt_tokens is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    # Load input data
    input_files: list[Path] = []
    if args.input_path.is_dir():
        input_files.extend(args.input_path.glob("*.jsonl"))
    else:
        if args.input_path.suffix != ".jsonl" or not args.input_path.exists():
            msg = f"Input path {args.input_path} is not a valid .jsonl file or directory."
            raise ValueError(msg)
        input_files.append(args.input_path)

    all_prompts: list[InputPromptData] = []
    num_removed_too_long = 0
    num_first_turn_not_user = 0
    num_invalid_conversations = 0
    global_conversation_id = 0
    for input_file in input_files:
        num_requests_per_split = 0
        with input_file.open("r", encoding="utf-8") as f:
            for line in tqdm_sync(f, desc=f"Loading {input_file.name}", dynamic_ncols=True):
                if not line.strip():
                    continue
                global_conversation_id += 1
                entry = json.loads(line)
                if not entry:
                    num_invalid_conversations += 1
                    continue
                # First, try for "prompt" field
                if "prompt" not in entry and not "conversations" in entry:
                    print("Neither 'prompt' nor 'conversations' field found in entry.")
                    num_invalid_conversations += 1
                    continue
                if "prompt" in entry:
                    prompt = entry["prompt"]
                    if not prompt or not isinstance(prompt, str):
                        num_invalid_conversations += 1
                        continue
                    msgs = [{"role": "user", "content": prompt}]
                else:
                    conversation = entry["conversations"]
                    if not conversation or not isinstance(conversation, list):
                        num_invalid_conversations += 1
                        continue
                    if conversation[0].get("role") != "user":
                        num_first_turn_not_user += 1
                        continue
                    msgs = conversation
                if not msgs:
                    num_invalid_conversations += 1
                    continue
                # 50/50 add system prompt
                if random.random() < 0.5:
                    if msgs[0]["role"] != "system":
                        msgs.insert(0, {"role": "system", "content": "You are a helpful assistant."})
                metadata = {"source_split": entry.get("source_split", None), "source_dataset": entry.get("source_dataset", None)}
                conversation_id = entry.get("conversation_id")
                if not conversation_id:
                    conversation_id = str(global_conversation_id)
                    if metadata["source_split"] is not None:
                        conversation_id = f"{metadata['source_split']}-{conversation_id}"
                    if metadata["source_dataset"] is not None:
                        conversation_id = f"{metadata['source_dataset']}-{conversation_id}"
                    entry["conversation_id"] = conversation_id
                if args.max_prompt_tokens is not None:
                    assert tokenizer is not None, (
                        "Tokenizer should be initialized if max_prompt_tokens is set."
                    )
                    if len(tokenizer.apply_chat_template(msgs)) > args.max_prompt_tokens:
                        num_removed_too_long += 1
                        continue

                # sample temperature to use
                # first sample from three cases: 0.0, 1.0, and a random value between 0.2 and 1.0
                temperature = random.choices([0.0, 1.0, random.uniform(0.2, 1.0)], weights=[0.15, 0.7, 0.15], k=1)[0]
                # sample reasoning effort to use
                # 25% "low", 50% "medium", 25% "high"
                # reasoning_effort = random.choices(
                #     ["low", "medium", "high"], weights=[0.25, 0.5, 0.25], k=1
                # )[0]
                reasoning_effort = None
                # 75% chance to enable reasoning
                enable_reasoning = random.random() < 0.75
                all_prompts.append(
                    InputPromptData(
                        prompt_conversation=msgs,
                        conversation_id=conversation_id,
                        source_filename=input_file.name,
                        temperature=temperature,
                        reasoning_effort=reasoning_effort,
                        enable_reasoning=enable_reasoning,
                        metadata=metadata,
                    )
                )
                num_requests_per_split += 1
                if (
                    args.debug_max_num_conversations_per_file
                    and num_requests_per_split >= args.debug_max_num_conversations_per_file
                ):
                    break

    # Shuffle requests
    random.seed(42)
    random.shuffle(all_prompts)

    print(f"Loaded {len(all_prompts)} total prompts from {len(input_files)} input files.")
    if num_removed_too_long > 0:
        print(f"Skipped {num_removed_too_long} prompts exceeding max token limit.")
    if num_first_turn_not_user > 0:
        print(f"Skipped {num_first_turn_not_user} prompts whose first turn was not 'user'.")
    if num_invalid_conversations > 0:
        print(f"Skipped {num_invalid_conversations} invalid conversation prompts.")

    # Initialize checkpointing state:
    # First, initialize the checkpoints and load the ids of requests already saved
    checkpoint_states = []
    num_requests_per_checkpoint = len(all_prompts) // args.num_checkpoints
    if args.num_checkpoints <= 0:
        msg = "Number of checkpoints must be greater than 0"
        raise ValueError(msg)

    all_processed_ids = set()
    for i in range(args.num_checkpoints):
        checkpoint_save_path = args.output_dir / f"checkpoints/checkpoint_{i}.jsonl"
        checkpoint_save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_states.append(
            CheckpointState(
                index=i,
                num_requests_for_checkpoint=0,
                checkpoint_output_file=checkpoint_save_path,
            )
        )
        all_processed_ids.update(load_existing_conversation_ids(checkpoint_save_path))

    print(f"Found {len(all_processed_ids)} already-processed conversations from checkpoints.")
    new_prompts = [p for p in all_prompts if p.conversation_id not in all_processed_ids]
    print(f"{len(new_prompts)} conversations remain to be processed after filtering.")

    for i in range(len(new_prompts)):
        checkpoint_idx_for_request = min(i // num_requests_per_checkpoint, args.num_checkpoints - 1)
        checkpoint_states[checkpoint_idx_for_request].num_requests_for_checkpoint += 1

    # Now, process the new requests
    pbar = tqdm_sync(
        total=len(new_prompts),
        desc="Processing prompts",
        dynamic_ncols=True,
        smoothing=0.0,
    )
    promises = []
    for i, req_obj in enumerate(new_prompts):
        checkpoint_idx_for_request = min(i // num_requests_per_checkpoint, args.num_checkpoints - 1)
        promises.append(
            query_and_process_response(
                clients[i % len(clients)],
                semaphores[i % len(semaphores)],
                req_obj,
                checkpoint_states[checkpoint_idx_for_request],
                args,
                pbar,
            )
        )

    await asyncio.gather(*promises)
    pbar.close()

    print("All requests processed. Collating checkpoints...")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load all checkpoints and collate conversations by ID
    all_conversation_dict = {}
    for checkpoint_state in checkpoint_states:
        assert len(checkpoint_state.outputs) == checkpoint_state.num_requests_for_checkpoint, (
            "Mismatch in number of processed requests for checkpoint."
        )
        if checkpoint_state.checkpoint_output_file.exists():
            with checkpoint_state.checkpoint_output_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if not entry or "conversation_id" not in entry:
                        continue
                    all_conversation_dict[entry["conversation_id"]] = entry

    # Group loaded conversations by their original input file
    results_by_file = {}
    num_skipped = 0
    num_successes = 0
    for prompt in all_prompts:
        if prompt.conversation_id not in all_conversation_dict:
            num_skipped += 1
        else:
            num_successes += 1
            if prompt.source_filename not in results_by_file:
                results_by_file[prompt.source_filename] = []
            results_by_file[prompt.source_filename].append(
                all_conversation_dict[prompt.conversation_id]
            )

    # Save all results to their respective output files
    for filename, entries in results_by_file.items():
        output_file_path = output_dir / ("generated_" + filename)
        with output_file_path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {num_successes} conversations to {output_dir!s}")
    print(f"Failed to process {num_skipped} conversations.")

    # Remove checkpoint files if the failure rate was low enough
    if not args.keep_checkpoints and (num_skipped / max(1, len(new_prompts))) < 0.05:
        print("Removing checkpoint files...")
        for checkpoint_state in checkpoint_states:
            if checkpoint_state.checkpoint_output_file.exists():
                checkpoint_state.checkpoint_output_file.unlink()
        checkpoint_dir = args.output_dir / "checkpoints"
        if checkpoint_dir.exists() and checkpoint_dir.is_dir():
            checkpoint_dir.rmdir()
        print("Checkpoint files removed.")
    elif not args.keep_checkpoints:
        print(
            "Not removing checkpoint files due to high failure rate. "
            "You can manually delete them if desired, "
            "or rerun the script to retry the failed requests."
        )

if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(main(cli_args))