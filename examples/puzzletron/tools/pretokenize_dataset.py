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

"""Offline, parallel pre-tokenization of the puzzletron dataset.

The per-run data path renders each conversation's chat template (Jinja) and
tokenizes it lazily, single-threaded on the master rank, on every run. That is
the slow part of "creating the dataset" at long block sizes.

This helper does that work ONCE, across all CPU cores, and writes a copy of the
dataset with a ``token_ids`` column. ``ConstantLengthDataset`` auto-detects that
column and skips both the chat-template rendering and tokenization (see
``modelopt/torch/puzzletron/utils/data/dataset.py``), leaving only the cheap
packing. It reuses the SAME rendering helper (``render_messages_to_text``) and
reads the dataset path / tokenizer / content field from the SAME Hydra config the
run uses, so the token ids are identical to the on-the-fly path and can't drift.

Token ids are independent of ``block_size`` and ``eval_samples`` (packing happens
later, at load time), so a single pre-tokenized copy serves every stage (pruning
activation scoring, single-block scoring, realize-model eval, and bypass) at any
block size.

Usage (CPU-only, no GPU; run on a many-core node):
    python examples/puzzletron/tools/pretokenize_dataset.py \
        --config examples/puzzletron/configs/qwen3_6-27b_pruneffn_runtime/qwen3_6_27b_pruneffn_runtime.yaml \
        --output /lustre/.../Nemotron-Post-Training-Dataset-v2-qwen36-tokenized \
        --num-proc 64

Then set ``dataset_path: <output>`` in the config. The tokenizer is read from
``teacher_dir``, so run this after the teacher checkpoint exists (i.e. after the
convert step), or pass --tokenizer to point at the HF model directly.
"""

import argparse
import os
from pathlib import Path

# Parallelism comes from --num-proc (one tokenizer per worker process); disable
# the fast tokenizer's own threads to avoid contention/fork warnings in workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import datasets
from transformers import AutoTokenizer

import modelopt.torch.puzzletron as mtpz
from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptorFactory
from modelopt.torch.puzzletron.utils.data.dataset import render_messages_to_text

# Mirrors ConstantLengthDataset(max_sample_length=...): per-sample char cap applied
# before tokenization on the not-yet-tokenized path.
MAX_SAMPLE_CHARS = 200_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the run's Hydra config YAML.")
    parser.add_argument("--output", required=True, help="Output dataset directory (save_to_disk).")
    parser.add_argument(
        "--num-proc", type=int, default=os.cpu_count(), help="Parallel worker processes (default: all cores)."
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="datasets.map batch size.")
    parser.add_argument(
        "--content-field", default=None, help="Override the messages/content column (default: from config)."
    )
    parser.add_argument(
        "--tokenizer", default=None, help="Override the tokenizer path (default: teacher_dir from config)."
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing --output.")
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"--output already exists: {out} (pass --overwrite to replace).")

    # Load the run's config exactly like main.py does, so paths/tokenizer/field match.
    mtpz.tools.register_hydra_resolvers()
    cfg_path = Path(args.config).resolve()
    cfg = mtpz.tools.initialize_hydra_config_for_dir(
        config_dir=str(cfg_path.parent), config_name=cfg_path.stem, overrides=[]
    )

    dataset_path = str(cfg.dataset_path)
    tokenizer_path = args.tokenizer or str(cfg.teacher_dir)
    content_field = args.content_field or cfg.scoring.get("data_column", "messages")

    trust_remote_code = False
    try:
        trust_remote_code = ModelDescriptorFactory.get(cfg.descriptor).requires_trust_remote_code()
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    if not tokenizer.is_fast:
        print("WARNING: tokenizer is not a fast (Rust) tokenizer; pre-tokenization will be slow.")

    print(f"dataset_path  = {dataset_path}")
    print(f"tokenizer     = {tokenizer_path} (is_fast={tokenizer.is_fast})")
    print(f"content_field = {content_field}")
    print(f"num_proc      = {args.num_proc}")
    print(f"output        = {out}")

    ds = datasets.load_from_disk(dataset_path)
    columns = ds.column_names
    # DatasetDict -> {split: [cols]}; Dataset -> [cols].
    sample_cols = next(iter(columns.values())) if isinstance(columns, dict) else columns
    if "token_ids" in sample_cols:
        print("NOTE: dataset already has a 'token_ids' column; it will be regenerated.")

    def fn(batch):
        # Same rendering + tokenization (and char cap) as ConstantLengthDataset.__iter__.
        texts = [render_messages_to_text(c, tokenizer)[:MAX_SAMPLE_CHARS] for c in batch[content_field]]
        return {"token_ids": tokenizer(texts, truncation=False)["input_ids"]}

    ds = ds.map(
        fn,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=[content_field],
        desc="pretokenize",
    )
    ds.save_to_disk(str(out))
    print(f"\nDone. Set `dataset_path: {out}` in your config; the slow chat-template + tokenize")
    print("steps will be skipped (only packing remains).")


if __name__ == "__main__":
    main()
