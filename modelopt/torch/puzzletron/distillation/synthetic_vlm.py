"""Small deterministic image/text dataset for global-KD integration runs."""

from __future__ import annotations

from PIL import Image

__all__ = ["make_synthetic_llm_dataset", "make_synthetic_vlm_dataset"]


def make_synthetic_llm_dataset(
    tokenizer,
    count: int = 16,
    seq_length: int = 128,
    split: str | None = None,
    **_: object,
) -> list[dict]:
    del split
    texts = [
        f"Question: What is {index} plus one? Answer: {index + 1}."
        for index in range(int(count))
    ]
    encoded = tokenizer(
        texts,
        max_length=int(seq_length),
        padding="max_length",
        truncation=True,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    examples = []
    for index in range(len(texts)):
        input_ids = list(encoded["input_ids"][index])
        attention_mask = list(encoded["attention_mask"][index])
        labels = [token if mask else -100 for token, mask in zip(input_ids, attention_mask)]
        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )
    return examples


def make_synthetic_vlm_dataset(count: int = 16, split: str | None = None, **_: object) -> list[dict]:
    del split
    examples = []
    for index in range(int(count)):
        color = ((37 * index) % 256, (67 * index) % 256, (97 * index) % 256)
        image = Image.new("RGB", (64, 64), color=color)
        examples.append(
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Name the dominant RGB color."},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"The RGB color is {color[0]}, {color[1]}, {color[2]}.",
                            }
                        ],
                    },
                ]
            }
        )
    return examples
