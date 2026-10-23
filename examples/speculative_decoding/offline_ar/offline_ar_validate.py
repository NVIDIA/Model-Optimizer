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

"""Offline acceptance-rate validation of a DFlash drafter on dumped target hidden states.

``scripts/ar_validate.py`` runs the target at every speculative step. This script instead
replays greedy speculative decoding on conversations dumped by ``collect_hidden_states/``:
the dumped assistant replies are the reference and their aux hidden states feed the draft,
so the target never runs and one dump evaluates any number of draft checkpoints.

The reported AR is vLLM's acceptance length, 1 + accepted / drafts pooled over all steps.
It is exact for greedy decoding when the dumped replies are the target's own greedy outputs;
the reported greedy agreement checks that from the dumped final hidden states.
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import modelopt.torch.opt as mto
from modelopt.torch.speculative.eagle.utils import OfflineSupervisedDataset
from modelopt.torch.speculative.plugins.hf_dflash import DFlashBaseModelOutput, HFDFlashModel
from modelopt.torch.speculative.utils import load_vlm_or_llm

mto.enable_huggingface_checkpointing()


@torch.no_grad()
def draft_block(
    model: HFDFlashModel, anchor: torch.Tensor, context_hidden: torch.Tensor
) -> torch.Tensor:
    """Greedy draft tokens after ``anchor``, given the aux hidden states of every token before it.

    The draft half of ``HFDFlashModel.pseudo_speculative_generate``, fed from a dump instead of a
    target forward: ``anchor`` is a token id, ``context_hidden`` is ``[ctx_len, num_aux * hidden]``.
    """
    device = model._base_device()
    block_size = model.dflash_block_size
    block_ids = torch.full((1, block_size), model.mask_token_id, dtype=torch.long, device=device)
    block_ids[0, 0] = anchor
    ctx_len = context_hidden.shape[0]
    target_hidden = context_hidden[None]
    draft_hidden = model.dflash_module(
        noise_embedding=model._base_model_embeddings(block_ids).to(device),
        target_hidden=target_hidden,
        position_ids=torch.arange(ctx_len + block_size, device=device)[None],
        attention_mask=model._build_generate_swa_mask(ctx_len, 1, target_hidden.dtype, device),
    )
    return model._base_model_lm_head(draft_hidden[:, 1:]).argmax(dim=-1)[0]


def replay(
    model: HFDFlashModel,
    input_ids: torch.Tensor,
    aux_hidden_states: torch.Tensor,
    start: int,
    end: int,
    num_speculative_tokens: int,
) -> tuple[int, int]:
    """Greedy speculative decoding of the reply ``input_ids[start:end]``.

    The target produces ``input_ids[start]`` at prefill. Each step drafts from the last produced
    token, accepts the drafts that match the reply up to the first mismatch, and the target adds
    one token. Returns the tokens the steps produced and the number of steps.
    """
    produced, steps, pos = 0, 0, start
    while pos < end - 1:
        drafts = draft_block(model, input_ids[pos], aux_hidden_states[:pos])
        drafts = drafts[:num_speculative_tokens]
        reference = input_ids[pos + 1 : end][: len(drafts)]
        accepted = int((drafts[: len(reference)] == reference).int().cumprod(0).sum())
        produced += accepted + 1
        steps += 1
        pos += accepted + 1
    return produced, steps


@torch.no_grad()
def greedy_agreement(
    model: HFDFlashModel,
    base_outputs: DFlashBaseModelOutput,
    input_ids: torch.Tensor,
    start: int,
    end: int,
) -> int:
    """How many tokens of the reply ``input_ids[start:end]`` are the target's own greedy pick."""
    hits = 0
    # Chunked: each row of the target's logits spans the whole vocabulary.
    for positions in torch.arange(start - 1, end - 1, device=input_ids.device).split(1024):
        greedy = model._teacher_logits(base_outputs, positions[None]).argmax(dim=-1)[0]
        hits += int((greedy == input_ids[positions + 1]).sum())
    return hits


def assistant_spans(loss_mask: torch.Tensor) -> list[tuple[int, int]]:
    """``[start, end)`` of each contiguous run of assistant tokens."""
    edges = torch.nn.functional.pad(loss_mask.int(), (1, 1)).diff().nonzero().flatten().tolist()
    return list(zip(edges[::2], edges[1::2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model_path", required=True, help="DFlash checkpoint saved by main.py (HF format)."
    )
    parser.add_argument(
        "--dump_dir",
        type=Path,
        required=True,
        help="Directory of .pt dumps from collect_hidden_states/compute_hidden_states_*.py, made "
        "with --answer-only-loss and --aux-layers set to the draft's target_layer_ids.",
    )
    parser.add_argument(
        "--num_speculative_tokens",
        type=int,
        default=None,
        help="Draft tokens verified per step. Defaults to block_size - 1.",
    )
    parser.add_argument("--num_samples", type=int, default=None, help="Use the first N dumps.")
    parser.add_argument("--output", type=Path, default=None, help="Write results as JSON here.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code.")
    args = parser.parse_args()

    model = load_vlm_or_llm(
        args.model_path, device_map="auto", trust_remote_code=args.trust_remote_code
    )
    if not isinstance(model, HFDFlashModel):
        raise TypeError(f"{args.model_path} is not a DFlash checkpoint: {type(model).__name__}.")
    if type(model).pseudo_speculative_generate is not HFDFlashModel.pseudo_speculative_generate:
        raise NotImplementedError(
            f"{type(model).__name__} drafts beyond the DFlash block, which this replay covers."
        )
    model.eval()
    num_spec = args.num_speculative_tokens or model.dflash_block_size - 1
    if not 0 < num_spec < model.dflash_block_size:
        raise ValueError(f"--num_speculative_tokens must be in [1, {model.dflash_block_size - 1}].")

    files = sorted(args.dump_dir.glob("*.pt"))[: args.num_samples]
    if not files:
        raise FileNotFoundError(f"No .pt dumps in {args.dump_dir}.")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    dataset = OfflineSupervisedDataset(files, answer_only_loss=True, tokenizer=tokenizer)
    device = model._base_device()
    fc = model.dflash_module.fc

    replies, produced, steps, greedy, tokens = [], 0, 0, 0, 0
    for path, sample in zip(files, tqdm(dataset, desc="Replaying")):
        aux = sample["aux_hidden_states"]
        if aux.shape[-1] != fc.in_features:
            raise ValueError(
                f"{path.name}: aux hidden width {aux.shape[-1]} != {fc.in_features}; dump with "
                f"--aux-layers {','.join(map(str, model.target_layer_ids))}."
            )
        aux, input_ids = aux.to(device, fc.weight.dtype), sample["input_ids"].to(device)
        base_outputs = DFlashBaseModelOutput(
            target_hidden=aux[None],
            base_hidden=sample["base_model_hidden_states"][None].to(device),
            base_hidden_prenorm=sample["base_hidden_prenorm"],
        )
        for start, end in assistant_spans(sample["loss_mask"]):
            reply_produced, reply_steps = replay(model, input_ids, aux, start, end, num_spec)
            reply_greedy = greedy_agreement(model, base_outputs, input_ids, start, end)
            produced, steps = produced + reply_produced, steps + reply_steps
            greedy, tokens = greedy + reply_greedy, tokens + end - start
            replies.append(
                {
                    "dump": path.name,
                    "start": start,
                    "end": end,
                    "steps": reply_steps,
                    "greedy_tokens": reply_greedy,
                }
            )

    ar, agreement = produced / steps, greedy / tokens
    print(
        f"AR {ar:.4f}: {steps} steps over {len(replies)} replies, {num_spec} drafts per step. "
        f"Target greedy agreement {agreement:.4f}."
    )
    if agreement < 0.98:
        print(
            "WARNING: the replies are not the target's greedy outputs, so the AR is biased. "
            "Generate them at temperature 0 and keep special tokens (see README.md)."
        )
    if args.output:
        result = {
            "ar": ar,
            "greedy_agreement": agreement,
            "num_speculative_tokens": num_spec,
            "steps": steps,
            "replies": replies,
        }
        args.output.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
