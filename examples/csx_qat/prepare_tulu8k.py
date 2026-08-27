#!/usr/bin/env python
"""Build the tulu@8k SFT corpus used by the CSX QAT samples.

8,000 training + 500 held-out examples from allenai/tulu-3-sft-mixture
(~5.7M tokens at max_length 4096; 98.7% of examples fit without truncation).

Evaluate in-domain on this corpus's own test split. A recovery number is only
meaningful against a ceiling (no quantization) and a floor (PTQ applied to the
*finetuned* model) measured on the same split -- see README.md.
"""
import argparse

from datasets import DatasetDict, load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = load_dataset("allenai/tulu-3-sft-mixture")["train"].shuffle(seed=args.seed)
    n = args.n_train + args.n_test
    sub = d.select(range(n)).select_columns(["messages"])
    DatasetDict(
        {
            "train": sub.select(range(args.n_train)),
            "test": sub.select(range(args.n_train, n)),
        }
    ).save_to_disk(args.out)
    print(f"saved {args.n_train} train / {args.n_test} test -> {args.out}")


if __name__ == "__main__":
    main()
