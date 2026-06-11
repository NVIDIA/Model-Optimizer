# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Generate the CFG negative-prompt embedding for DMD2 training.

The canonical config sets ``data.dataloader.negative_prompt_embedding_path``; the dataloader
loads that single file once and broadcasts it across the batch for classifier-free guidance on
the teacher. This script makes the example self-contained: it encodes the negative prompt
(default the empty string ``""``, the standard CFG unconditional) with the **same Qwen text
encoder** the preprocessing uses (via ``QwenImageProcessor``), and saves it in the loader's
payload format ``{"embed": [seq, dim], "mask": [seq]}``.

Run it once after building the cache, pointing ``--output`` at the cache directory:

    python examples/diffusers/fastgen/make_negative_prompt_embedding.py \\
        --output <cache_dir>/negative_prompt_embedding.pt

Then set the config's ``negative_prompt_embedding_path`` to that file.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Make the ``preprocess`` package importable as a top-level package (same seam as the launcher).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def main() -> None:
    import torch

    from preprocess.processors.qwen_image import QwenImageProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, help="Output path, e.g. <cache_dir>/negative_prompt_embedding.pt"
    )
    parser.add_argument("--model", default="Qwen/Qwen-Image", help="Qwen-Image model id or local path")
    parser.add_argument(
        "--negative_prompt", default="", help='Negative prompt to encode (default "" = unconditional)'
    )
    parser.add_argument("--device", default="cuda", help="Device for the text encoder")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Reuse the vendored Qwen-Image processor's model loading + encoder so the negative embedding
    # matches the cached prompt embeddings exactly (same chat template / tokenizer / dtype).
    processor = QwenImageProcessor()
    models = processor.load_models(args.model, args.device)
    pipeline = models["pipeline"]

    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            prompt=args.negative_prompt, device=args.device
        )

    embed = prompt_embeds.detach().cpu().to(torch.bfloat16).squeeze(0)  # [seq, dim]
    if prompt_embeds_mask is not None:
        mask = prompt_embeds_mask.detach().cpu().to(torch.long).squeeze(0)  # [seq]
    else:
        mask = torch.ones(embed.shape[0], dtype=torch.long)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Payload format consumed by fastgen_data.build_text_to_image_multiresolution_dataloader's
    # negative_prompt_embedding_path loader: a dict with an "embed" tensor and optional "mask".
    torch.save({"embed": embed, "mask": mask}, args.output)
    logging.info(
        "[fastgen] saved negative prompt embedding: embed=%s mask=%s -> %s",
        tuple(embed.shape),
        tuple(mask.shape),
        args.output,
    )


if __name__ == "__main__":
    main()
