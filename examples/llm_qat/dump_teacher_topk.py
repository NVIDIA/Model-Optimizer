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

"""Dump a teacher's top-k next-token log-probs for offline QAD (``--offline_teacher_dir``).

Reads the exact tokenized train/eval rows that ``train.py`` uses (same blend config, cache and
``model_max_length``), runs the teacher once over each unpadded row, and stores the top-k
log-probs at every position whose next token carries loss. The layout is documented in
``modelopt.torch.distill.plugins.offline_teacher``.

The ``vllm`` backend uses vLLM's ``extract_hidden_states`` to capture the last layer's output
(before the final norm), then applies the teacher's own HF final norm and lm_head, matching the
online KD path. The ``hf`` backend runs the HF teacher instead, to separate engine numerics from
anything else when comparing with online QAD.

Pass the same ``--model_name_or_path`` (whose tokenizer ``train.py`` uses), ``--dataset_config`` and
``--dataset_cache_dir`` as ``train.py`` so the rows match; run one process per ``--tp`` GPUs, splitting the
shards with ``--dp_rank`` / ``--dp_world_size``::

    CUDA_VISIBLE_DEVICES=0 python dump_teacher_topk.py --model_name_or_path qwen3-8b-nvfp4-ptq \\
        --teacher_model Qwen/Qwen3-8B --output_dir teacher_topk --dp_rank 0 --dp_world_size 4
"""

import json
import os
import time
from dataclasses import dataclass, field

import torch
import transformers
from arguments import DataArguments, ModelArguments
from safetensors import safe_open
from safetensors.torch import save_file
from utils import make_supervised_data_module

from modelopt.torch.distill.plugins.offline_teacher import input_checksum

IGNORE_INDEX = -100


@dataclass
class DumpArguments:
    teacher_model: str = field(metadata={"help": "Teacher model path."})
    output_dir: str = field(metadata={"help": "Where to write meta.json and <split>/shard-*."})
    top_k: int = field(default=4096, metadata={"help": "Teacher log-probs kept per position."})
    backend: str = field(default="vllm", metadata={"help": "'vllm' or 'hf'."})
    splits: str = field(default="train,eval", metadata={"help": "Comma-separated splits."})
    shard_size: int = field(default=32, metadata={"help": "Dataset rows per shard file."})
    tp: int = field(default=1, metadata={"help": "vLLM tensor parallel size per dump process."})
    dp_rank: int = field(default=0)
    dp_world_size: int = field(default=1)
    gpu_memory_utilization: float = field(
        default=0.6, metadata={"help": "vLLM share of the GPU; the rest holds lm_head and logits."}
    )
    max_shards: int = field(default=-1, metadata={"help": "Debug: stop after this many shards."})


def _load_modules(path, names, config):
    """Load the named HF submodules of ``path`` (e.g. final norm, lm_head) with their real weights."""
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
    else:  # a single-file checkpoint
        with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as f:
            weight_map = dict.fromkeys(f.keys(), "model.safetensors")
    modules = {}
    for key, getter in names.items():
        module = getter(model)
        prefix = next(n for n, m in model.named_modules() if m is module)
        module = module.to_empty(device="cuda")
        state = {}
        for pname, _ in module.named_parameters():
            full = f"{prefix}.{pname}"
            if full not in weight_map and key == "lm_head" and config.tie_word_embeddings:
                full = f"{next(n for n, m in model.named_modules() if m is model.get_input_embeddings())}.{pname}"
            with safe_open(os.path.join(path, weight_map[full]), framework="pt") as f:
                state[pname] = f.get_tensor(full)
        module.load_state_dict(state)
        modules[key] = module.eval()
    return modules


class _VllmTeacher:
    """Last-layer hidden states (pre final norm) from vLLM's native hidden-state extractor."""

    def __init__(self, args, config, max_len, staging):
        from vllm import LLM
        from vllm.config.kv_transfer import KVTransferConfig

        self.llm = LLM(
            model=args.teacher_model,
            dtype="bfloat16",
            tensor_parallel_size=args.tp,
            max_model_len=max_len + 1,  # a full-length row plus the one sampled token
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
            enable_chunked_prefill=False,  # required by extract_hidden_states
            enable_prefix_caching=False,  # cached prefixes would be missing from the dump
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {"eagle_aux_hidden_state_layer_ids": [config.num_hidden_layers]}
                },
            },
            kv_transfer_config=KVTransferConfig(
                kv_connector="ExampleHiddenStatesConnector",
                kv_role="kv_producer",
                kv_connector_extra_config={
                    "shared_storage_path": staging,
                    "use_synchronization_lock": True,
                },
            ),
        )
        self.needs_final_norm = True

    def hidden_states(self, rows):
        from vllm import SamplingParams
        from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector
        from vllm.inputs import TokensPrompt

        prompts = [TokensPrompt(prompt_token_ids=r.tolist()) for r in rows]
        outputs = self.llm.generate(prompts, SamplingParams(max_tokens=1), use_tqdm=False)
        assert len(outputs) == len(rows)
        for row, out in zip(rows, outputs):
            path = out.kv_transfer_params["hidden_states_path"]
            try:
                obj = example_hidden_states_connector.load_hidden_states(path)
                if not torch.equal(obj["token_ids"].to(torch.int64), row.to(torch.int64)):
                    raise RuntimeError("vLLM hidden states belong to different token ids.")
                yield obj["hidden_states"][:, -1, :]
            finally:
                example_hidden_states_connector.cleanup_hidden_states(path)


class _HFTeacher:
    """Final-norm hidden states from the HF teacher, as the online KD path computes them."""

    def __init__(self, args, attn_implementation):
        kwargs = {"attn_implementation": attn_implementation} if attn_implementation else {}
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            args.teacher_model, dtype=torch.bfloat16, **kwargs
        ).cuda()
        self.model.eval()
        self.needs_final_norm = False

    @torch.no_grad()
    def hidden_states(self, rows):
        for row in rows:
            out = self.model.model(input_ids=row[None].cuda())
            yield out.last_hidden_state[0]


@torch.no_grad()
def _topk(hidden, positions, norm, lm_head, top_k, chunk=1024):
    """Top-k log-probs of the teacher at ``positions``; logits in bf16 then fp32, as online."""
    ids, logprobs = [], []
    for start in range(0, positions.numel(), chunk):
        h = hidden[positions[start : start + chunk]].cuda()
        if norm is not None:
            h = norm(h)
        logits = (h @ lm_head.weight.t()).float()
        values, index = logits.log_softmax(-1).topk(top_k, dim=-1)
        ids.append(index.to(torch.int32).cpu())
        logprobs.append(values.to(torch.float16).cpu())
    return torch.cat(ids), torch.cat(logprobs)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, DumpArguments))
    model_args, data_args, args = parser.parse_args_into_dataclasses()

    # Tokenize exactly as train.py does (tokenizer of --model_name_or_path), so the cached rows match.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, model_max_length=model_args.model_max_length
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    data = make_supervised_data_module(data_args, tokenizer)
    datasets = {"train": data["train_dataset"], "eval": data["eval_dataset"]}
    splits = [s for s in args.splits.split(",") if s]

    os.makedirs(args.output_dir, exist_ok=True)
    meta = {
        "teacher_model": args.teacher_model,
        "top_k": args.top_k,
        "temperature": 1.0,
        "backend": args.backend,
        "model_max_length": model_args.model_max_length,
        "dataset_config": data_args.dataset_config,
        "shard_size": args.shard_size,
        "num_rows": {s: len(d) for s, d in datasets.items()},
    }
    meta_path = os.path.join(args.output_dir, "meta.json")
    if os.path.exists(meta_path):
        # Shards from different runs are only mixable when they were made the same way.
        with open(meta_path) as f:
            existing = json.load(f)
        if existing != meta:
            raise ValueError(f"{meta_path} differs from this run: {existing} vs {meta}")
    elif args.dp_rank == 0:
        with open(meta_path + ".tmp", "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(meta_path + ".tmp", meta_path)

    config = transformers.AutoConfig.from_pretrained(args.teacher_model)
    modules = _load_modules(
        args.teacher_model,
        {"norm": lambda m: m.get_decoder().norm, "lm_head": lambda m: m.get_output_embeddings()},
        config,
    )
    if args.backend == "vllm":
        staging = f"/dev/shm/teacher_topk_dp{args.dp_rank}"
        os.makedirs(staging, exist_ok=True)
        teacher = _VllmTeacher(args, config, model_args.model_max_length, staging)
    elif args.backend == "hf":
        teacher = _HFTeacher(args, model_args.attn_implementation)
    else:
        raise ValueError(f"Unknown backend {args.backend!r}")
    norm = modules["norm"] if teacher.needs_final_norm else None

    done = 0
    for split in splits:
        ds = datasets[split]
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        num_shards = (len(ds) + args.shard_size - 1) // args.shard_size
        for shard in range(args.dp_rank, num_shards, args.dp_world_size):
            path = os.path.join(split_dir, f"shard-{shard:05d}.safetensors")
            if os.path.exists(path):
                continue
            if 0 <= args.max_shards <= done:
                return
            t0 = time.time()
            rows = list(range(shard * args.shard_size, min((shard + 1) * args.shard_size, len(ds))))
            inputs, positions, sizes, sums = [], [], [], []
            for r in rows:
                item = ds[r]
                n = int(sum(item["attention_mask"]))
                ids = torch.tensor(item["input_ids"][:n], dtype=torch.int64)
                labels = torch.tensor(item["labels"][:n])
                # Position t predicts token t + 1, the causal shift of the KD loss.
                positions.append((labels[1:] != IGNORE_INDEX).nonzero().flatten())
                inputs.append(ids)
                sizes.append(n)
                sums.append(input_checksum(ids))
            top_ids, top_lp = [], []
            for pos, hidden in zip(positions, teacher.hidden_states(inputs)):
                i, lp = _topk(hidden, pos, norm, modules["lm_head"], args.top_k)
                top_ids.append(i)
                top_lp.append(lp)
            offsets = torch.zeros(len(rows) + 1, dtype=torch.int64)
            offsets[1:] = torch.tensor([p.numel() for p in positions]).cumsum(0)
            tensors = {
                "row": torch.tensor(rows, dtype=torch.int64),
                "num_tokens": torch.tensor(sizes, dtype=torch.int64),
                "checksum": torch.tensor(sums, dtype=torch.int64),
                "offsets": offsets,
                "positions": torch.cat(positions).to(torch.int64),
                "ids": torch.cat(top_ids),
                "logprobs": torch.cat(top_lp),
            }
            # Write-then-rename: an existing shard is treated as finished on resume.
            tmp = path + ".tmp"
            save_file(tensors, tmp)
            os.replace(tmp, path)
            done += 1
            print(
                f"[dp{args.dp_rank}] {split} shard {shard}/{num_shards}: {len(rows)} rows, "
                f"{sum(sizes)} tokens, {int(offsets[-1])} positions, {time.time() - t0:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
