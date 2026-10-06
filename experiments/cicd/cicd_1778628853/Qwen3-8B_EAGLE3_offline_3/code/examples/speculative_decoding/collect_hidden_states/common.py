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

"""Shared helpers for ``compute_hidden_states_*`` dump scripts.

Groups two concerns used by both the HF and vLLM dump entry points:

- Aux-layer selection via the ``--aux-layers`` flag (``"eagle"`` / ``"dflash"``
  / explicit comma-separated list). Returned values are **0-based transformer
  layer IDs**; callers indexing into HuggingFace's ``outputs.hidden_states``
  tuple must add ``+1`` because ``hidden_states[0]`` is the embedding output.
- Answer-only-loss support: registering ``--answer-only-loss`` /
  ``--chat-template`` flags, loading a chat template file, verifying the
  template contains ``{% generation %}`` tags, and computing per-conversation
  ``loss_mask`` from the tokenizer's ``assistant_masks``.
"""

import argparse
from pathlib import Path

import torch

_DFLASH_DEFAULT_NUM_DRAFT_LAYERS = 5


def add_aux_layers_args(parser: argparse.ArgumentParser) -> None:
    """Register the ``--aux-layers`` flag on ``parser``."""
    parser.add_argument(
        "--aux-layers",
        type=str,
        default="eagle",
        help=(
            "Aux layer indices to capture. One of: "
            "'eagle' (EAGLE-3 default from modelopt), "
            f"'dflash' ({_DFLASH_DEFAULT_NUM_DRAFT_LAYERS}-layer DFlash default from modelopt), "
            "or a comma-separated list like '2,5,8' to override. Default: eagle."
        ),
    )


def resolve_aux_layers(args: argparse.Namespace, num_hidden_layers: int) -> list[int]:
    """Resolve ``args.aux_layers`` to a sorted, de-duped list of 0-based layer IDs."""
    value = args.aux_layers.strip().lower()
    if value == "eagle":
        from modelopt.torch.speculative.plugins.hf_eagle import default_eagle_aux_layer_ids

        return default_eagle_aux_layer_ids(num_hidden_layers)
    if value == "dflash":
        from modelopt.torch.speculative.plugins.modeling_dflash import build_target_layer_ids

        return sorted(
            set(build_target_layer_ids(num_hidden_layers, _DFLASH_DEFAULT_NUM_DRAFT_LAYERS))
        )
    try:
        indices = [int(tok) for tok in args.aux_layers.split(",") if tok.strip()]
    except ValueError as e:
        raise ValueError(
            f"--aux-layers must be 'eagle', 'dflash', or a comma-separated int list, "
            f"got: {args.aux_layers!r}"
        ) from e
    if not indices:
        raise ValueError(f"--aux-layers int list is empty: {args.aux_layers!r}")
    for i in indices:
        if not 0 <= i < num_hidden_layers:
            raise ValueError(f"--aux-layers index {i} out of range [0, {num_hidden_layers})")
    return sorted(set(indices))


def add_answer_only_loss_args(parser: argparse.ArgumentParser) -> None:
    """Register ``--answer-only-loss`` and ``--chat-template`` flags on ``parser``."""
    parser.add_argument(
        "--answer-only-loss",
        action="store_true",
        help=(
            "If set, compute an assistant-token mask via the tokenizer's "
            "{% generation %} tags and save it as 'loss_mask' in each .pt file. "
            "Downstream offline training uses this to apply loss only on "
            "assistant-produced tokens."
        ),
    )
    parser.add_argument(
        "--chat-template",
        type=Path,
        default=None,
        help=(
            "Path to a Jinja chat template file that overrides tokenizer.chat_template. "
            "Required with --answer-only-loss if the model's default template lacks "
            "{% generation %} / {% endgeneration %} tags."
        ),
    )


def load_chat_template(path: Path | None) -> str | None:
    """Read a Jinja chat template from ``path``, or return ``None`` if not provided."""
    if path is None:
        return None
    with open(path) as f:
        return f.read()


def verify_generation_tags(chat_template: str | None) -> None:
    """Raise if ``chat_template`` lacks ``{% generation %}`` / ``{% endgeneration %}`` tags.

    These tags are required for ``apply_chat_template(..., return_assistant_tokens_mask=True)``
    to return the assistant-token mask needed for answer-only-loss training.
    """
    if chat_template and "generation" in chat_template and "endgeneration" in chat_template:
        return
    raise ValueError(
        "--answer-only-loss requires {% generation %} / {% endgeneration %} tags in the "
        "chat template, but the current template does not have them.\n\n"
        "To fix, pass --chat-template pointing to a template with generation tags:\n"
        "  1. Copy the model's chat_template from tokenizer_config.json\n"
        "  2. Wrap assistant content with {% generation %} / {% endgeneration %}\n"
        "See https://huggingface.co/docs/transformers/en/chat_templating"
        "#train-on-completions-only for details."
    )


def tokenize_with_loss_mask(
    tokenizer,
    conversations: list,
    answer_only_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize one conversation and derive its loss mask from the same call.

    Uses a single ``apply_chat_template`` invocation so ``input_ids`` and
    ``loss_mask`` are guaranteed to come from the same tokenization — this
    eliminates the risk of argument drift between two separate calls.

    Returns:
        input_ids: ``LongTensor`` of shape ``(1, seq_len)``.
        loss_mask: ``LongTensor`` of shape ``(seq_len,)``. All-ones when
            ``answer_only_loss=False``; the assistant-token mask from the
            tokenizer when ``answer_only_loss=True`` (requires ``{% generation %}``
            tags in the chat template — verify beforehand).
    """
    out = tokenizer.apply_chat_template(
        conversations,
        return_tensors="pt",
        return_dict=True,
        return_assistant_tokens_mask=answer_only_loss,
        add_generation_prompt=False,
    )
    input_ids = out["input_ids"]
    seq_len = input_ids.shape[-1]
    if answer_only_loss:
        mask = out["assistant_masks"]
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long)
        loss_mask = mask.squeeze(0).to(torch.long)
        if loss_mask.shape[0] != seq_len:
            raise RuntimeError(
                f"assistant_masks length {loss_mask.shape[0]} does not match "
                f"input_ids length {seq_len}"
            )
    else:
        loss_mask = torch.ones(seq_len, dtype=torch.long)
    return input_ids, loss_mask
