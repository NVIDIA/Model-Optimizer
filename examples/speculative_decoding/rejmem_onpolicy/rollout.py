# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Rollout side of on-policy rejection-memory training: config, prompts, vLLM servers, requests.

The experiment YAML is a ModelOpt DFlash recipe plus an ``onpolicy`` section (``OnPolicyConfig``), the
single source of the deployment regime: the trainer sends its sampling parameters with every request,
and this script starts a vLLM server with its speculative config and memory switch::

    python rollout.py serve --config CONFIG --port PORT --capture_dir DIR [--draft EXPORT] [key=value ...]
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, model_validator

from modelopt.recipe.config import ModelOptDFlashRecipe
from modelopt.torch.speculative.plugins.rejmem.records import RolloutRecord, load_record

WORKER_EXTENSION = "modelopt.torch.speculative.plugins.rejmem.vllm_worker.RejMemWorkerExtension"
# Request fields the controller owns.
_REQUEST_KEYS = {
    "model",
    "prompt",
    "max_tokens",
    "n",
    "stream",
    "echo",
    "return_token_ids",
    "vllm_xargs",
}
# Request fields that end a request outside the engine or fork it, so that it gets no single record.
_UNCAPTURED_KEYS = {"stop", "use_beam_search"}
# Rows of this many shards are shuffled together: a corpus shard may hold a single subset, and fewer
# shards would skew a chunk's subset mix.
_SHARDS_PER_CHUNK = 64


class SamplingConfig(BaseModel):
    """Target sampling of every rollout request, sent verbatim in the ``/v1/completions`` body.

    All of vLLM's sampling defaults are spelled out so the target's ``generation_config.json`` cannot
    replace them; other request fields (e.g. ``seed``) pass through.
    """

    model_config = ConfigDict(extra="allow")

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # disabled
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0

    @model_validator(mode="after")
    def _no_controller_fields(self):
        if owned := _REQUEST_KEYS.intersection(self.model_extra or {}):
            raise ValueError(f"onpolicy.sampling must not set {sorted(owned)}")
        if uncaptured := _UNCAPTURED_KEYS.intersection(self.model_extra or {}):
            raise ValueError(
                f"onpolicy.sampling must not set {sorted(uncaptured)}: the servers record only "
                "requests that the engine stops itself (use stop_token_ids)"
            )
        return self


class OnPolicyConfig(BaseModel):
    """The ``onpolicy`` section: deployment regime, prompts and the node's servers."""

    model_config = ConfigDict(extra="forbid")

    # Deployment regime; the servers take num_speculative_tokens and draft_sample_method from here.
    # memory: the drafter reads the rejected rows (MODELOPT_REJMEM_CONSUMER and rejmem_mem_count).
    memory: bool = True
    num_speculative_tokens: int = 15
    draft_sample_method: Literal["greedy", "probabilistic"] = "probabilistic"
    sampling: SamplingConfig = SamplingConfig()
    # Prompts: jsonl shards of {"conversations": [...]}; a reply with "</think>" marks a think row.
    data_dir: str
    max_prompt_len: int
    prompts_per_step: int  # per trainer rank, i.e. per node
    # This node's servers and their capture directory (node-local), set by the launch script.
    server_urls: list[str] = []
    capture_dir: str | None = None
    server_args: list[str] = []  # appended to `vllm serve`
    # Together below training.ddp_timeout, so a stuck server fails its rank before the others time out.
    request_timeout: float = 1200.0
    record_timeout: float = 300.0  # from a response to its record file


def load_config(path: str, overrides: list[str]) -> tuple[ModelOptDFlashRecipe, OnPolicyConfig]:
    """Load the experiment YAML with ``key.path=value`` overrides; returns (recipe, onpolicy)."""
    if bad := [o for o in overrides if "=" not in o]:
        raise ValueError(f"overrides must be key=value, got {bad}")
    data = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(overrides)), resolve=True
    )
    if not isinstance(data, dict) or "onpolicy" not in data:
        raise ValueError(f"{path}: expected a DFlash recipe with an onpolicy section")
    onpolicy = OnPolicyConfig.model_validate(data.pop("onpolicy"))
    recipe = ModelOptDFlashRecipe.model_validate(data)
    # The rollout features stand in for the target's forward, so the draft trains offline.
    recipe.dflash.dflash_offline = True
    if onpolicy.num_speculative_tokens != recipe.dflash.dflash_block_size - 1:
        raise ValueError(
            "onpolicy.num_speculative_tokens must be dflash_block_size - 1: a round is a block"
        )
    if onpolicy.max_prompt_len >= recipe.training.training_seq_len:
        raise ValueError(
            "onpolicy.max_prompt_len must leave room for a response in training_seq_len"
        )
    return recipe, onpolicy


@dataclass
class Prompt:
    """A rendered single-turn prompt."""

    token_ids: list[int]
    enable_thinking: bool


class PromptSource:
    """This rank's share of the corpus's single-turn prompts, in a seeded order, each used once.

    Shards are shuffled with ``seed`` and dealt to the ranks round-robin. A rank reads its shards in
    chunks of ``_SHARDS_PER_CHUNK`` and shuffles each chunk's rows with ``seed``, so that every step
    mixes the subsets. ``state_dict`` is the cursor, so a resumed run continues the order.
    """

    def __init__(self, data_dir, tokenizer, max_prompt_len, seed, rank, world_size):
        files = sorted(Path(data_dir).glob("*.jsonl"))
        random.Random(seed).shuffle(files)
        self.files = files[rank::world_size]
        if not self.files:
            raise ValueError(f"{data_dir}: no *.jsonl shard for rank {rank} of {world_size}")
        self.tokenizer, self.max_prompt_len, self.seed = tokenizer, max_prompt_len, seed
        self.chunk_idx = self.row_idx = self.num_skipped = 0
        self._rows: list[tuple[list[dict], bool]] | None = None

    def take(self, n: int) -> list[Prompt]:
        """The next ``n`` prompts that fit ``max_prompt_len``."""
        prompts: list[Prompt] = []
        while len(prompts) < n:
            prompt = self._render(*self._next_row())
            if prompt is None:
                self.num_skipped += 1
            else:
                prompts.append(prompt)
        return prompts

    def _next_row(self) -> tuple[list[dict], bool]:
        while self.chunk_idx * _SHARDS_PER_CHUNK < len(self.files):
            if self._rows is None:
                start = self.chunk_idx * _SHARDS_PER_CHUNK
                chunk = self.files[start : start + _SHARDS_PER_CHUNK]
                self._rows = [row for path in chunk for row in _read_prompts(path)]
                random.Random(f"{self.seed}:{chunk[0].name}").shuffle(self._rows)
            if self.row_idx < len(self._rows):
                self.row_idx += 1
                return self._rows[self.row_idx - 1]
            self.chunk_idx, self.row_idx, self._rows = self.chunk_idx + 1, 0, None
        raise RuntimeError("the prompt corpus is exhausted; lower training.max_steps")

    def _render(self, messages: list[dict], think: bool) -> Prompt | None:
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=think,
            return_dict=False,
        )
        return Prompt(ids, think) if len(ids) <= self.max_prompt_len else None

    def state_dict(self) -> dict:
        """The cursor, with the shard list it indexes."""
        return {
            "files": [p.name for p in self.files],
            "chunk_idx": self.chunk_idx,
            "row_idx": self.row_idx,
            "num_skipped": self.num_skipped,
        }

    def load_state_dict(self, state: dict) -> None:
        """Continue from a cursor saved by a run over the same shards, seed and world size."""
        if state["files"] != [p.name for p in self.files]:
            raise ValueError(
                "the checkpoint's prompt shards differ: corpus, seed or world size changed"
            )
        self.chunk_idx, self.row_idx = state["chunk_idx"], state["row_idx"]
        self.num_skipped, self._rows = state["num_skipped"], None


def _read_prompts(path: Path) -> list[tuple[list[dict], bool]]:
    """Each row's system and first user turn, and whether its reference reply thinks."""
    prompts = []
    with path.open() as f:
        for line in f:
            turns = json.loads(line)["conversations"]
            kept = [t for t in turns[:1] if t["role"] == "system"]
            kept.append(next(t for t in turns if t["role"] == "user"))
            reply = next((t["content"] for t in turns if t["role"] == "assistant"), "")
            messages = [{"role": t["role"], "content": t["content"]} for t in kept]
            prompts.append((messages, "</think>" in reply))
    return prompts


@dataclass
class Rollout:
    """One response: ``token_ids`` is the API's prompt + output."""

    tag: str
    token_ids: list[int]
    prompt_len: int
    finish_reason: str


def generate(
    cfg: OnPolicyConfig, model: str, prompts: list[Prompt], max_tokens: list[int], tags: list[str]
) -> list[Rollout]:
    """Generate every prompt once, spread round-robin over this node's servers."""
    return asyncio.run(_generate(cfg, model, prompts, max_tokens, tags))


async def _generate(cfg, model, prompts, max_tokens, tags):
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=cfg.request_timeout, limits=limits) as client:
        return await asyncio.gather(
            *(
                _complete(client, cfg.server_urls[i % len(cfg.server_urls)], cfg, model, *args)
                for i, args in enumerate(zip(prompts, max_tokens, tags))
            )
        )


async def _complete(client, url, cfg, model, prompt, max_tokens, tag):
    body = {
        **cfg.sampling.model_dump(),
        "model": model,
        "prompt": prompt.token_ids,
        "max_tokens": max_tokens,
        "return_token_ids": True,
        "vllm_xargs": {"rejmem_tag": tag},
    }
    resp = await client.post(f"{url}/v1/completions", json=body)
    if resp.status_code != 200:
        raise RuntimeError(f"{url} request {tag}: HTTP {resp.status_code}: {resp.text[:2000]}")
    choice = resp.json()["choices"][0]
    if choice["prompt_token_ids"] != prompt.token_ids:
        raise RuntimeError(f"{url} request {tag}: the server changed the prompt tokens")
    return Rollout(
        tag, prompt.token_ids + choice["token_ids"], len(prompt.token_ids), choice["finish_reason"]
    )


def load_records(
    capture_dir: str, tags: list[str], timeout: float
) -> tuple[list[RolloutRecord], int]:
    """Wait for, load (validated) and delete the record of every tag; also returns their bytes."""
    paths = [Path(capture_dir) / f"{tag}.safetensors" for tag in tags]
    deadline = time.monotonic() + timeout
    for path in paths:
        while not path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"no record {path} {timeout}s after its response")
            time.sleep(0.05)
    nbytes = sum(p.stat().st_size for p in paths)
    records = [load_record(str(p)) for p in paths]
    for tag, rec, path in zip(tags, records, paths):
        if rec.meta["tag"] != tag:
            raise ValueError(f"{path} holds the record of tag {rec.meta['tag']}")
        path.unlink()
    return records, nbytes


def serve_command(
    recipe: ModelOptDFlashRecipe,
    cfg: OnPolicyConfig,
    port: int,
    capture_dir: str,
    draft: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """``vllm serve`` argv and environment of one rollout server; ``draft`` defaults to the init export."""
    target, draft = recipe.model.model_name_or_path, draft or recipe.dflash.dflash_init_checkpoint
    if target is None or draft is None:
        raise ValueError(
            "set model.model_name_or_path, and --draft or dflash.dflash_init_checkpoint"
        )
    ns = cfg.num_speculative_tokens
    spec = {
        "method": "dflash",
        "model": draft,
        "num_speculative_tokens": ns,
        "draft_sample_method": cfg.draft_sample_method,
    }
    argv = [
        *(sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", target),
        # Node-local: the trainer is the only client, and the weight-update endpoints unpickle bodies.
        *("--host", "127.0.0.1", "--port", str(port)),
        # The consumer writes memory slots past the committed tokens, also in the one extra step
        # that async scheduling may run after the last token: up to 3 NS (CONTRACT section 2, C1).
        *("--max-model-len", str(recipe.training.training_seq_len + 3 * ns)),
        *("--speculative-config", json.dumps(spec)),
        *("--worker-extension-cls", WORKER_EXTENSION),
        # A cached prefix would skip the rows whose features the records need.
        "--no-enable-prefix-caching",
        # The trainer pushes the drafter through CUDA IPC handles (weight_push.DraftWeightPusher).
        *("--weight-transfer-config", json.dumps({"backend": "ipc"})),
        *cfg.server_args,
    ]
    env = {k: v for k, v in os.environ.items() if k != "MODELOPT_REJMEM_CONSUMER"}
    env["MODELOPT_REJMEM_CAPTURE_DIR"] = capture_dir
    # The weight-update endpoints, which take the IPC handles pickled.
    env["VLLM_SERVER_DEV_MODE"] = env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    if cfg.memory:
        env["MODELOPT_REJMEM_CONSUMER"] = "1"
    return argv, env


def main():
    """``serve``: replace this process with a rollout server configured by the experiment YAML."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--capture_dir", required=True)
    parser.add_argument("--draft", help="drafter export to serve (default: the init checkpoint)")
    args, overrides = parser.parse_known_args()
    recipe, cfg = load_config(args.config, overrides)
    argv, env = serve_command(recipe, cfg, args.port, args.capture_dir, args.draft)
    print(" ".join(argv), flush=True)
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
