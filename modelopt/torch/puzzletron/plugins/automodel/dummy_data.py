"""Tiny setup-only dataset for AutoModel-backed Puzzletron stages.

The NeMo fine-tuning recipe always builds a training dataloader during
``setup()``, even for Puzzletron's forward-only scoring recipes. Puzzletron
replaces that dataloader with its own calibration loader immediately after
``super().setup()`` returns, so the recipe dataset only needs to be valid enough
for NeMo to construct samplers, schedulers, and optional PP masks.
"""

from __future__ import annotations


def make_dummy_dataset(
    tokenizer=None,
    num_samples: int = 64,
    seq_length: int = 16,
    split: str | None = None,
):
    """Return a small deterministic causal-LM dataset for NeMo recipe setup."""
    del split  # AutoModel requires this selector when recipe-side packing is enabled.
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    pad_id = 0 if pad_id is None else int(pad_id)
    eos_id = pad_id if eos_id is None else int(eos_id)
    bos_id = eos_id if bos_id is None else int(bos_id)

    seq_length = max(int(seq_length), 2)
    num_samples = max(int(num_samples), 1)
    middle_len = seq_length - 2
    dataset = []
    for sample_idx in range(num_samples):
        # Keep ids in a tiny safe range; the data is never used for real scoring.
        middle = [3 + ((sample_idx + offset) % 97) for offset in range(middle_len)]
        input_ids = [bos_id, *middle, eos_id]
        dataset.append(
            {
                "input_ids": input_ids,
                "labels": input_ids[1:] + [eos_id],
                "attention_mask": [1] * len(input_ids),
                "___PAD_TOKEN_IDS___": {
                    "input_ids": pad_id,
                    "labels": -100,
                    "attention_mask": 0,
                },
            }
        )
    return dataset
