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

"""Dump teacher top-k log-probs as Megatron-LM offline KD V3 corpus sources.

Reads the tokenized rows ``train.py`` uses, runs the teacher once over each unpadded row, and
writes each split as a V3 corpus source (``modelopt.torch.distill.plugins.offline_kd_v3``) that
Megatron-LM's offline KD readers consume: ``<output_dir>/<split>/chunk-<start>-<end>/``, one chunk
per ``--shard_size`` rows. Rows without loss are kept as empty samples, so a source covers its whole
split. A chunk with an ``index.json`` is finished; rerunning skips it.

The ``vllm`` backend uses vLLM's ``extract_hidden_states`` to capture the last layer's output
(before the final norm), then applies the teacher's own HF final norm and lm_head, matching the
online KD path. The ``hf`` backend runs the HF teacher instead, to separate engine numerics from
anything else when comparing with online KD.

Pass the same ``--model_name_or_path`` (whose tokenizer ``train.py`` uses), ``--dataset_config``
and ``--dataset_cache_dir`` as ``train.py`` so the rows match; run one process per ``--tp`` GPUs,
splitting the chunks with ``--dp_rank`` / ``--dp_world_size``::

    CUDA_VISIBLE_DEVICES=0 python dump_teacher_v3.py --model_name_or_path Qwen/Qwen3-8B \\
        --teacher_model Qwen/Qwen3-8B --output_dir teacher_v3 --dp_rank 0 --dp_world_size 4

A Megatron-LM student then reads, e.g., ``{"corpus": "teacher_v3", "sources": {"train": 1.0}}``
with ``--logits-load-corpus-blend``.
"""

import glob
import json
import os
import time
from dataclasses import dataclass, field

import torch
import transformers
from arguments import DataArguments, ModelArguments
from dataset_utils import IGNORE_TOKEN_ID
from safetensors import safe_open
from utils import make_supervised_data_module

from modelopt.torch.distill.plugins.offline_kd_v3 import (
    chunk_dir,
    corpus_settings,
    publish_chunk,
    strip_row,
    topk_targets,
)


@dataclass
class V3DumpArguments:
    teacher_model: str = field(metadata={"help": "Teacher model path."})
    output_dir: str = field(
        metadata={"help": "Corpus directory; split <split> becomes source <output_dir>/<split>."}
    )
    top_k: int = field(default=128, metadata={"help": "Teacher log-probs kept per position."})
    backend: str = field(default="vllm", metadata={"help": "'vllm' or 'hf'."})
    splits: str = field(default="train,eval", metadata={"help": "Comma-separated splits."})
    shard_size: int = field(default=256, metadata={"help": "Dataset rows per corpus chunk."})
    tp: int = field(default=1, metadata={"help": "vLLM tensor parallel size per dump process."})
    dp_rank: int = field(default=0)
    dp_world_size: int = field(default=1)
    gpu_memory_utilization: float = field(
        default=0.6, metadata={"help": "vLLM share of the GPU; the rest holds lm_head and logits."}
    )
    max_shards: int = field(default=-1, metadata={"help": "Debug: stop after this many chunks."})


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
def _targets(hidden, positions, norm, lm_head, top_k, vocab_size, chunk=1024):
    """Top-k targets at ``positions``; logits in bf16 then fp32, as the online KD path."""
    values, indices, tail = [], [], []
    for start in range(0, positions.numel(), chunk):
        h = hidden[positions[start : start + chunk]].cuda()
        if norm is not None:
            h = norm(h)
        v, i, t = topk_targets(h @ lm_head.weight.t(), top_k, vocab_size)
        values.append(v.half().cpu())
        indices.append(i.int().cpu())
        tail.append(t.cpu())
    return {"values": torch.cat(values), "indices": torch.cat(indices), "tail": torch.cat(tail)}


def _check_resume(source, chunks, settings):
    """Refuse to extend a source that a run with other chunk bounds or settings started."""
    for path in glob.glob(os.path.join(source, "chunk-*", "index.json")):
        with open(path) as f:
            written = json.load(f)["settings"]
        if os.path.dirname(path) not in chunks or written != settings:
            raise ValueError(f"{path} comes from a run with another --shard_size or settings")


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, V3DumpArguments))
    model_args, data_args, args = parser.parse_args_into_dataclasses()

    # Tokenize exactly as train.py does (tokenizer of --model_name_or_path), so the rows match.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, model_max_length=model_args.model_max_length
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    data = make_supervised_data_module(data_args, tokenizer)
    datasets = {"train": data["train_dataset"], "eval": data["eval_dataset"]}

    if args.backend not in ("vllm", "hf"):
        raise ValueError(f"Unknown backend {args.backend!r}")
    config = transformers.AutoConfig.from_pretrained(args.teacher_model)
    quant = (getattr(config, "quantization_config", None) or {}).get("quant_method")
    provenance = {
        "producer": "modelopt",
        "backend": args.backend,
        "teacher_precision": f"bf16+{quant}" if quant else "bf16",
        "top_k": args.top_k,
        "save_dtype": "fp16",
    }
    # Plan every split's chunks first, so a mismatched rerun fails before loading the teacher.
    jobs = []
    for split in [s for s in args.splits.split(",") if s]:
        ds, source = datasets[split], os.path.join(args.output_dir, split)
        recorded = {
            "teacher": args.teacher_model,
            "seq_length": model_args.model_max_length,
            "padding_divisor": 1,  # Unpacked rows need no headroom for a student's segment padding.
            "data": {
                "kind": "llm_qat",
                "config": os.path.realpath(data_args.dataset_config),
                "split": split,
                "shuffle": data_args.shuffle,
                "shuffle_buffer": data_args.shuffle_buffer,
            },
            "seed": data_args.dataset_seed,
            "builder": "make_supervised_data_module",
        }
        settings = corpus_settings(
            tokenizer, config, len(ds), recorded=recorded, provenance=provenance
        )
        bounds = [
            (s, min(s + args.shard_size, len(ds))) for s in range(0, len(ds), args.shard_size)
        ]
        dirs = [chunk_dir(source, start, end) for start, end in bounds]
        _check_resume(source, set(dirs), settings)
        for (start, end), directory in list(zip(bounds, dirs))[args.dp_rank :: args.dp_world_size]:
            if not os.path.exists(os.path.join(directory, "index.json")):
                jobs.append((ds, settings, source, start, end))
    if args.max_shards >= 0:
        jobs = jobs[: args.max_shards]
    if not jobs:
        return

    modules = _load_modules(
        args.teacher_model,
        {"norm": lambda m: m.get_decoder().norm, "lm_head": lambda m: m.get_output_embeddings()},
        config,
    )
    if args.backend == "vllm":
        staging = f"/dev/shm/teacher_v3_dp{args.dp_rank}"
        os.makedirs(staging, exist_ok=True)
        teacher = _VllmTeacher(args, config, model_args.model_max_length, staging)
    else:
        teacher = _HFTeacher(args, model_args.attn_implementation)
    norm = modules["norm"] if teacher.needs_final_norm else None
    vocab_size = len(tokenizer)
    no_targets = {
        "values": torch.zeros(0, args.top_k, dtype=torch.float16),
        "indices": torch.zeros(0, args.top_k, dtype=torch.int32),
        "tail": torch.zeros(0),
    }
    for ds, settings, source, start, end in jobs:
        t0 = time.time()
        samples = []
        for row in range(start, end):
            item = ds[row]
            n = int(sum(item["attention_mask"]))
            ids, labels = torch.tensor(item["input_ids"][:n]), torch.tensor(item["labels"][:n])
            samples.append({**no_targets, **strip_row(ids, labels, IGNORE_TOKEN_ID)})
        # vLLM needs a nonempty prompt; rows without loss keep no tokens and no targets.
        scored = [s for s in samples if s["segments"]]
        for sample, hidden in zip(scored, teacher.hidden_states([s["tokens"] for s in scored])):
            sample.update(
                _targets(
                    hidden, sample["positions"], norm, modules["lm_head"], args.top_k, vocab_size
                )
            )
        directory = publish_chunk(source, samples, settings, start=start, vocab_size=vocab_size)
        print(
            f"[dp{args.dp_rank}] {directory}: {sum(s['tokens'].numel() for s in samples)} tokens, "
            f"{sum(len(s['tail']) for s in samples)} targets, {time.time() - t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
