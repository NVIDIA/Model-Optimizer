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

"""A simple MMMU evaluation for Megatron VLM models."""

import ast
import contextlib
import re
from typing import Any

import requests
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, ProcessorMixin

from .megatron_generate import megatron_generate

__all__ = ["megatron_mmmu"]


def _resize_for_vlm(image: Image.Image, max_edge: int = 768) -> Image.Image:
    """Resize a PIL image to cap its longer edge for stable VLM token lengths."""
    w, h = image.size
    if max(w, h) <= max_edge:
        return image
    scale = float(max_edge) / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), resample=Image.BICUBIC)


def _get_all_subjects():
    """All subjects can be acquired from querying all subsets and splits."""
    response = requests.get(
        "https://datasets-server.huggingface.co/splits?dataset=MMMU/MMMU", timeout=10
    )
    data = response.json()
    all_subjects = {split["config"] for split in data["splits"]}
    return sorted(all_subjects)


def _parse_options(options) -> list[str]:
    """Parse options from MMMU's string-serialized list format."""
    if isinstance(options, list):
        return [str(opt) for opt in options]
    if not isinstance(options, str):
        return []
    with contextlib.suppress(Exception):
        parsed = ast.literal_eval(options)
        if isinstance(parsed, list):
            return [str(opt) for opt in parsed]
    return []


def _collect_images(example: dict) -> list[Image.Image]:
    """Collect non-empty image_1..image_7 entries from one MMMU example."""
    images: list[Image.Image] = []
    for idx in range(1, 8):
        value = example.get(f"image_{idx}")
        if value is None:
            continue
        if isinstance(value, Image.Image):
            images.append(_resize_for_vlm(value.convert("RGB")))
            continue
        # HF `datasets` can also return dict-like image payloads.
        if isinstance(value, dict):
            if isinstance(value.get("bytes"), (bytes, bytearray)):
                with contextlib.suppress(Exception):
                    from io import BytesIO

                    decoded = Image.open(BytesIO(value["bytes"])).convert("RGB")
                    images.append(_resize_for_vlm(decoded))
                    continue
            if isinstance(value.get("path"), str):
                with contextlib.suppress(Exception):
                    decoded = Image.open(value["path"]).convert("RGB")
                    images.append(_resize_for_vlm(decoded))
                    continue
    return images


def _apply_chat_template(processor: ProcessorMixin, messages: list[dict]) -> str | None:
    """Apply processor/tokenizer chat template when available."""
    proc_template_fn = getattr(processor, "apply_chat_template", None)
    if callable(proc_template_fn):
        with contextlib.suppress(Exception):
            return proc_template_fn(messages, tokenize=False, add_generation_prompt=True)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "apply_chat_template"):
        with contextlib.suppress(Exception):
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return None


def _extract_choice_letter(text: str, labels: list[str]) -> str | None:
    """Extract the first predicted option letter from generated text."""
    if not text:
        return None
    normalized = text.strip().upper()
    for label in labels:
        if normalized.startswith(label):
            return label
    match = re.search(r"\b([A-Z])\b", normalized)
    if match is not None and match.group(1) in labels:
        return match.group(1)
    return None


def _resolve_gold_label(answer: Any, labels: list[str]) -> str | None:
    """Normalize answer value to one of the option labels."""
    if isinstance(answer, str):
        normalized = answer.strip().upper()
        if normalized in labels:
            return normalized
    if isinstance(answer, int) and 0 <= answer < len(labels):
        return labels[answer]
    if isinstance(answer, float) and answer.is_integer():
        as_int = int(answer)
        if 0 <= as_int < len(labels):
            return labels[as_int]
    return None


def _build_prompt(
    question: str,
    options: list[str],
    *,
    include_answer: bool,
    answer: str | None = None,
) -> str:
    """Build a multiple-choice prompt block."""
    lines = [question.strip()]
    for idx, option in enumerate(options):
        lines.append(f"{chr(ord('A') + idx)}. {option}")
    if include_answer:
        if answer is None:
            raise ValueError("`answer` must be provided when include_answer=True.")
        lines.append(f"Answer: {answer}")
        lines.append("")
    else:
        lines.append("")
        lines.append("Answer with the option letter only.")
        lines.append("Answer:")
    return "\n".join(lines)


def _generate_prompt(
    test_example: dict[str, Any],
    dev_examples: list[dict[str, Any]],
    *,
    subject: str,
    few_shots: int,
) -> str | None:
    """Generate a MMLU-style prompt with optional few-shot exemplars."""
    options = _parse_options(test_example.get("options"))
    if not options:
        return None
    test_block = _build_prompt(
        str(test_example.get("question", "")),
        options,
        include_answer=False,
    )
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        " ".join(str(subject).split("_"))
    )
    if few_shots > 0:
        num_added = 0
        for dev_example in dev_examples:
            if str(dev_example.get("question_type", "")).lower() != "multiple-choice":
                continue
            dev_options = _parse_options(dev_example.get("options"))
            if not dev_options:
                continue
            dev_labels = [chr(ord("A") + i) for i in range(len(dev_options))]
            dev_gold = _resolve_gold_label(dev_example.get("answer"), dev_labels)
            if dev_gold is None:
                continue
            prompt += _build_prompt(
                str(dev_example.get("question", "")),
                dev_options,
                include_answer=True,
                answer=dev_gold,
            )
            num_added += 1
            if num_added >= few_shots:
                break
    prompt += test_block
    return prompt


def _prepare_encoded_example(
    example: dict[str, Any],
    processor: ProcessorMixin,
    use_images: bool,
    prompt_override: str | None = None,
) -> dict[str, Any] | None:
    """Prepare one MMMU example as encoded model inputs + answer metadata."""
    if str(example.get("question_type", "")).lower() != "multiple-choice":
        return None

    options = _parse_options(example.get("options"))
    if not options:
        return None
    labels = [chr(ord("A") + i) for i in range(len(options))]

    gold = _resolve_gold_label(example.get("answer"), labels)
    if gold is None:
        return None

    prompt = prompt_override or _build_prompt(
        str(example.get("question", "")),
        options,
        include_answer=False,
    )
    if use_images:
        images = _collect_images(example)
        if not images:
            # Keep rank behavior deterministic if an image decode fails on some workers.
            images = [Image.new("RGB", (16, 16), color="white")]

        messages = [
            {
                "role": "user",
                "content": ([{"type": "image", "image": ""} for _ in images])
                + [{"type": "text", "text": prompt}],
            }
        ]
        templated_prompt = _apply_chat_template(processor, messages) or prompt

        enc = processor(
            text=[templated_prompt],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        if hasattr(enc, "data"):
            enc = enc.data
        enc_out = {
            k: (v.cpu() if torch.is_tensor(v) else v)
            for k, v in dict(enc).items()
            if k in {"input_ids", "pixel_values", "image_grid_thw", "image_sizes"}
        }
    else:
        # Robust scoring path: text-only encoding from MMMU question/options.
        enc = processor.tokenizer(prompt, return_tensors="pt")
        if hasattr(enc, "data"):
            enc = enc.data
        enc_out = {"input_ids": dict(enc)["input_ids"].cpu()}

    if "input_ids" not in enc_out:
        return None
    return {
        "enc": enc_out,
        "labels": labels,
        "gold": gold,
    }


def megatron_mmmu(
    model,
    processor: ProcessorMixin | None = None,
    hf_model_name_or_path: str | None = None,
    trust_remote_code: bool = False,
    few_shots: int = 0,
    percentage: float = 0.05,
    enable_kv_cache: bool = False,
    use_images: bool = False,
) -> float:
    """Evaluate the model on MMMU test split with optional dev few-shots.

    Args:
        model: The model to evaluate.
        processor: Optional HF processor used for multimodal encoding.
        hf_model_name_or_path: HF model path used to lazily build processor if `processor` is None.
        trust_remote_code: Whether to trust remote code for AutoProcessor.
        few_shots: The number of dev examples to prepend as few-shot context.
        percentage: The percentage of each subject test split to evaluate on.
        enable_kv_cache: Whether to enable KV-cache decoding.
        use_images: If True, include image inputs during scoring. Disabled by default for robust PP scoring.
    """
    if processor is None:
        if hf_model_name_or_path is None:
            raise ValueError("Either `processor` or `hf_model_name_or_path` must be provided.")
        processor = AutoProcessor.from_pretrained(
            hf_model_name_or_path, trust_remote_code=trust_remote_code
        )

    if not hasattr(processor, "tokenizer"):
        raise ValueError("MMMU evaluation requires a processor that includes a tokenizer.")

    all_correct: dict[str, list[bool]] = {}
    all_subjects = _get_all_subjects()
    local_device = next(model.parameters()).device
    model_cfg = getattr(model, "config", None)
    max_seq_length = int(getattr(model_cfg, "seq_length", 0) or 0)
    if max_seq_length <= 0:
        lm_cfg = getattr(getattr(model, "language_model", None), "config", None)
        max_seq_length = int(getattr(lm_cfg, "seq_length", 0) or 0)
    if max_seq_length <= 0:
        max_seq_length = 4096
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    rank0 = rank == 0

    if rank0:
        print(f"\nMMMU ({percentage * 100}%, {few_shots}-shot) evaluation started...\n", flush=True)
        print(
            f"MMMU evaluator backend: datasets.load_dataset (use_images={use_images})",
            flush=True,
        )
        print("{:48} | (ACC) | Count/Total".format("Subject"), flush=True)
        print("{:48} | {:5} | {:11}".format("-" * 48, "-" * 5, "-" * 11), flush=True)

    for subject in all_subjects:
        test_data = load_dataset("MMMU/MMMU", subject, split="test")
        dev_data = load_dataset("MMMU/MMMU", subject, split="dev")
        max_examples = int(percentage * len(test_data)) + 1

        correct: list[bool] = []
        skipped_long = 0
        for idx, example in enumerate(test_data):
            if idx >= max_examples:
                break
            prompt = _generate_prompt(
                example,
                dev_data,
                subject=subject,
                few_shots=few_shots,
            )
            if prompt is None:
                continue
            payload = _prepare_encoded_example(
                example,
                processor,
                use_images=use_images,
                prompt_override=prompt,
            )
            if payload is None:
                continue

            enc = payload["enc"]
            seq_len = int(enc["input_ids"].shape[-1])
            if seq_len > max_seq_length:
                skipped_long += 1
                if rank0 and skipped_long <= 3:
                    print(
                        f"[MMMU] skipping over-length sample in {subject} "
                        f"(seq_len={seq_len} > seq_length={max_seq_length})",
                        flush=True,
                    )
                continue
            generated_ids = megatron_generate(
                model=model,
                input_ids=enc["input_ids"].to(local_device),
                pixel_values=enc.get("pixel_values", None).to(local_device)
                if enc.get("pixel_values", None) is not None
                else None,
                image_grid_thw=enc.get("image_grid_thw", None).to(local_device)
                if enc.get("image_grid_thw", None) is not None
                else None,
                image_sizes=enc.get("image_sizes", None).to(local_device)
                if enc.get("image_sizes", None) is not None
                else None,
                # For VLM wrapper scoring, a single-step decode is more robust than multi-step
                # decoding under PP during NAS candidate scoring.
                osl=1,
                disable_tqdm=True,
                enable_kv_cache=enable_kv_cache,
            )
            predict_text = processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            pred = _extract_choice_letter(predict_text, payload["labels"])
            correct.append(pred == payload["gold"])

        all_correct[subject] = correct

        if rank0:
            if skipped_long > 0:
                print(
                    f"[MMMU] {subject}: skipped {skipped_long} over-length samples",
                    flush=True,
                )
            if len(correct) == 0:
                print(f"{subject:48} | {'nan':>5} | {0:5}/{0:5}", flush=True)
            else:
                print(
                    f"{subject:48} | {sum(correct) / len(correct):.3f} | {sum(correct):5}/{len(correct):5}",
                    flush=True,
                )

    avg_correct: list[bool] = []
    for _, subject_correct in all_correct.items():
        avg_correct += subject_correct

    if rank0:
        print("{:48} | {:5} | {:11}".format("-" * 48, "-" * 5, "-" * 11), flush=True)
        if len(avg_correct) == 0:
            print("{:48} | {:>5} | {:5}/{:5}".format("average", "nan", 0, 0), flush=True)
        else:
            print(
                "{:48} | {:.3f} | {:5}/{:5}".format(
                    "average",
                    sum(avg_correct) / len(avg_correct),
                    sum(avg_correct),
                    len(avg_correct),
                ),
                flush=True,
            )

    if len(avg_correct) == 0:
        raise RuntimeError(
            "MMMU evaluation did not produce any valid multiple-choice samples. "
            "Please check dataset availability and processor compatibility."
        )

    return sum(avg_correct) / len(avg_correct)
