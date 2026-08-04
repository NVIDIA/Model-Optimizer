# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for offline speculative decoding PTQ support."""

import argparse
import importlib.util
import os

# ---------------------------------------------------------------------------
# Load eagle_utils from examples/ via importlib (not a package, so no import).
# eagle_utils has a top-level `from scripts.ar_validate import validate_ar` that
# only resolves when run from examples/speculative_decoding/. We stub it out here.
# ---------------------------------------------------------------------------
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama, get_tiny_tokenizer

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.eagle.default_config import default_eagle_config
from modelopt.torch.speculative.eagle.utils import (
    EagleOfflineDataCollator,
    OfflineSupervisedDataset,
)
from modelopt.torch.utils.plugins import transformers_dataset

_mock_scripts = types.ModuleType("scripts")
_mock_ar = types.ModuleType("scripts.ar_validate")
_mock_ar.validate_ar = lambda *args, **kwargs: None  # type: ignore[attr-defined]
sys.modules.setdefault("scripts", _mock_scripts)
sys.modules.setdefault("scripts.ar_validate", _mock_ar)

_EAGLE_UTILS_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../../..",
    "examples/speculative_decoding/eagle_utils.py",
)
_spec = importlib.util.spec_from_file_location("eagle_utils", _EAGLE_UTILS_PATH)
assert _spec is not None and _spec.loader is not None
_eagle_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eagle_utils)
make_speculative_data_module = _eagle_utils.make_speculative_data_module


# ---------------------------------------------------------------------------
# online VLM data-module wiring
# ---------------------------------------------------------------------------


def test_vlm_data_module_passes_dflash_label_mode(monkeypatch):
    """VLM batches must use unshifted labels for DFlash and preserve OSL settings."""
    data_args = argparse.Namespace(
        mode="online",
        data_path="unused.jsonl",
        vlm_processor="dummy-vlm-processor",
        vlm_img_dir="/images",
        chat_template=None,
    )
    collator = MagicMock()
    monkeypatch.setattr(_eagle_utils, "ShardedDataset", MagicMock())
    monkeypatch.setattr(_eagle_utils, "VisionLanguageDataCollator", collator)

    module = make_speculative_data_module(
        MagicMock(), data_args, train_len=16, answer_only_loss=True, shift_labels=False
    )

    collator.assert_called_once_with(
        processor="dummy-vlm-processor",
        train_len=16,
        local_image_path="/images",
        return_labels=True,
        answer_only_loss=True,
        shift_labels=False,
        chat_template=None,
    )
    assert module["data_collator"] is collator.return_value


def test_vlm_data_collator_accepts_unshifted_labels(monkeypatch):
    """The real VLM collator must support DFlash's unshifted labels."""
    processor = types.SimpleNamespace(tokenizer=get_tiny_tokenizer())
    monkeypatch.setattr(
        transformers_dataset.transformers.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: processor,
    )

    collator = transformers_dataset.VisionLanguageDataCollator(
        processor="dummy-vlm-processor",
        chat_template="{{ messages }}",
        shift_labels=False,
    )

    assert collator.shift_labels is False


class _WhitespaceTokenizer:
    """Minimal tokenizer for testing VLM content truncation."""

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return types.SimpleNamespace(input_ids=list(range(len(text.split()))))

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(f"token-{token_id}" for token_id in token_ids)


def _bare_vlm_collator():
    """Create a collator for helper-method tests without loading a processor."""

    collator = object.__new__(transformers_dataset.VisionLanguageDataCollator)
    collator.tokenizer = _WhitespaceTokenizer()
    collator.max_prompt_tokens = None
    collator.max_assistant_tokens = None
    collator._prompt_content_truncation_warned = False
    collator._assistant_content_truncation_warned = False
    collator._conversations_warned = False
    return collator


def test_vlm_collator_configures_visual_limits(monkeypatch):
    """VLM environment limits configure image, video, and text processing."""

    tokenizer = get_tiny_tokenizer()
    video_processor = types.SimpleNamespace(size={"shortest_edge": 32, "longest_edge": 256})
    processor = types.SimpleNamespace(tokenizer=tokenizer, video_processor=video_processor)
    from_pretrained = MagicMock(return_value=processor)
    monkeypatch.setattr(
        transformers_dataset.transformers.AutoProcessor, "from_pretrained", from_pretrained
    )
    monkeypatch.setenv("VLM_MIN_PIXELS", "64")
    monkeypatch.setenv("VLM_MAX_PIXELS", "512")
    monkeypatch.setenv("VLM_VIDEO_MIN_PIXELS", "128")
    monkeypatch.setenv("VLM_VIDEO_MAX_PIXELS", "1024")
    monkeypatch.setenv("VLM_MAX_PROMPT_TOKENS", "32")
    monkeypatch.setenv("VLM_MAX_ASSISTANT_TOKENS", "16")

    collator = transformers_dataset.VisionLanguageDataCollator(
        processor="dummy-vlm-processor", chat_template="{{ messages }}"
    )

    from_pretrained.assert_called_once_with("dummy-vlm-processor", min_pixels=64, max_pixels=512)
    assert video_processor.size == {"shortest_edge": 128, "longest_edge": 1024}
    assert collator.max_prompt_tokens == 32
    assert collator.max_assistant_tokens == 16
    assert processor.chat_template == collator.tokenizer.chat_template


def test_vlm_collator_rejects_video_limits_without_video_processor(monkeypatch):
    """Explicit video limits require a processor that supports video input."""

    monkeypatch.setenv("VLM_VIDEO_MIN_PIXELS", "128")
    processor = types.SimpleNamespace(video_processor=None)

    with pytest.raises(ValueError, match="has no video_processor"):
        transformers_dataset.VisionLanguageDataCollator._configure_video_processor(processor, {})


def test_vlm_collator_applies_image_limits_to_video_processor():
    """Image pixel limits are used for video when no video-specific limits are set."""

    video_processor = types.SimpleNamespace(size={"shortest_edge": 32, "longest_edge": 256})
    processor = types.SimpleNamespace(video_processor=video_processor)

    transformers_dataset.VisionLanguageDataCollator._configure_video_processor(
        processor, {"min_pixels": 64, "max_pixels": 512}
    )

    assert video_processor.size == {"shortest_edge": 64, "longest_edge": 512}


@pytest.mark.parametrize(
    ("env_name", "value", "message"),
    [
        ("VLM_MIN_PIXELS", "not-an-integer", "VLM_MIN_PIXELS must be an integer"),
        ("VLM_MAX_PROMPT_TOKENS", "0", "VLM_MAX_PROMPT_TOKENS must be a positive integer"),
        (
            "VLM_MAX_PROMPT_TOKENS",
            "not-an-integer",
            "VLM_MAX_PROMPT_TOKENS must be a positive integer",
        ),
        (
            "VLM_MAX_ASSISTANT_TOKENS",
            "not-an-integer",
            "VLM_MAX_ASSISTANT_TOKENS must be a positive integer",
        ),
    ],
)
def test_vlm_collator_rejects_invalid_environment_limits(monkeypatch, env_name, value, message):
    """Configuration errors identify the invalid VLM environment variable."""

    processor = types.SimpleNamespace(tokenizer=get_tiny_tokenizer())
    monkeypatch.setattr(
        transformers_dataset.transformers.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: processor,
    )
    monkeypatch.setenv(env_name, value)

    with pytest.raises(ValueError, match=message):
        transformers_dataset.VisionLanguageDataCollator(
            processor="dummy-vlm-processor", chat_template="{{ messages }}"
        )


def test_vlm_collator_rejects_non_integer_video_limit(monkeypatch):
    """Video bounds must be integer pixel counts."""

    monkeypatch.setenv("VLM_VIDEO_MIN_PIXELS", "not-an-integer")
    processor = types.SimpleNamespace(video_processor=types.SimpleNamespace(size={}))

    with pytest.raises(ValueError, match="VLM_VIDEO_MIN_PIXELS must be an integer"):
        transformers_dataset.VisionLanguageDataCollator._configure_video_processor(processor, {})


def test_vlm_collator_pads_template_output_and_builds_unshifted_labels():
    """DFlash batches preserve current-token labels after fixed-length padding."""

    collator = _bare_vlm_collator()
    collator.train_len = 5
    collator.answer_only_loss = True
    collator.return_labels = True
    collator.shift_labels = False
    collator.add_generation_prompt = False
    collator._assistant_marker_specs = list
    collator.processor = types.SimpleNamespace(
        apply_chat_template=MagicMock(
            return_value={
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "assistant_masks": torch.tensor([[0, 1, 1]]),
                "pixel_values": torch.tensor([1.0]),
            }
        )
    )

    output = collator._process_multimodal_sample([[{"role": "user", "content": "prompt"}]])

    assert output["input_ids"].tolist() == [[1, 2, 3, 0, 0]]
    assert output["attention_mask"].tolist() == [[1, 1, 1, 0, 0]]
    assert output["assistant_masks"].tolist() == [[0, 1, 1, 0, 0]]
    assert output["labels"].tolist() == [
        [
            transformers_dataset.IGNORE_TOKEN_ID,
            2,
            3,
            transformers_dataset.IGNORE_TOKEN_ID,
            transformers_dataset.IGNORE_TOKEN_ID,
        ]
    ]
    assert output["pixel_values"].tolist() == [1.0]
    collator.processor.apply_chat_template.assert_called_once_with(
        [[{"role": "user", "content": "prompt"}]],
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
    )


def test_vlm_collator_preserves_processor_length_when_it_matches_training_length():
    """No padding is added when the processor already returns train_len tokens."""

    collator = _bare_vlm_collator()
    collator.train_len = 3
    tokenized = {"input_ids": torch.tensor([[1, 2, 3]])}

    assert collator._pad_sequence_tensors(tokenized) is tokenized


@pytest.mark.parametrize(
    ("shift_labels", "assistant_mask", "expected_labels"),
    [
        (
            True,
            torch.tensor([[0, 1, 0]]),
            [[2, transformers_dataset.IGNORE_TOKEN_ID, transformers_dataset.IGNORE_TOKEN_ID]],
        ),
        (
            False,
            torch.zeros((1, 3), dtype=torch.long),
            [
                [
                    transformers_dataset.IGNORE_TOKEN_ID,
                    transformers_dataset.IGNORE_TOKEN_ID,
                    transformers_dataset.IGNORE_TOKEN_ID,
                ]
            ],
        ),
    ],
)
def test_vlm_collator_masks_shifted_and_empty_assistant_labels(
    shift_labels, assistant_mask, expected_labels
):
    """Assistant masks align shifted labels and mask empty assistant spans."""

    collator = _bare_vlm_collator()
    collator.train_len = None
    collator.answer_only_loss = True
    collator.return_labels = True
    collator.shift_labels = shift_labels
    collator.add_generation_prompt = False
    collator._assistant_marker_specs = list
    collator.processor = types.SimpleNamespace(
        apply_chat_template=MagicMock(
            return_value={
                "input_ids": torch.tensor([[1, 2, 3]]),
                "assistant_masks": assistant_mask,
            }
        )
    )

    output = collator._process_multimodal_sample([[{"role": "assistant", "content": "answer"}]])

    assert output["labels"].tolist() == expected_labels


def test_vlm_collator_requires_assistant_masks_for_answer_only_loss():
    """A processor without assistant masks cannot support answer-only loss."""

    collator = _bare_vlm_collator()
    collator.train_len = None
    collator.answer_only_loss = True
    collator.return_labels = True
    collator.shift_labels = True
    collator.add_generation_prompt = False
    collator._assistant_marker_specs = list
    collator.processor = types.SimpleNamespace(
        apply_chat_template=MagicMock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})
    )

    with pytest.raises(ValueError, match="requires assistant_masks"):
        collator._process_multimodal_sample([[{"role": "assistant", "content": "answer"}]])


def test_vlm_collator_derives_masks_from_chatml_markers():
    """ChatML fallback derives masks after the processor tokenizes the batch."""

    collator = _bare_vlm_collator()
    collator.train_len = None
    collator.answer_only_loss = True
    collator.add_generation_prompt = False
    collator._assistant_marker_specs = lambda: [([10, 11], [[99]])]
    collator.processor = types.SimpleNamespace(
        apply_chat_template=MagicMock(
            return_value={
                "input_ids": torch.tensor([[10, 11, 7, 99]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            }
        )
    )

    tokenized = collator._apply_chat_template([[{"role": "assistant", "content": "answer"}]])

    assert tokenized["assistant_masks"].tolist() == [[0, 0, 1, 0]]
    assert (
        collator.processor.apply_chat_template.call_args.kwargs["return_assistant_tokens_mask"]
        is False
    )


def test_vlm_collator_normalizes_and_truncates_structured_messages(tmp_path):
    """Structured contents become processor-compatible text, image, and video parts."""

    collator = _bare_vlm_collator()
    collator.local_image_path = str(tmp_path)
    collator.max_prompt_tokens = 3
    collator.max_assistant_tokens = 3
    captured = {}
    collator._process_multimodal_sample = lambda batch: captured.setdefault("batch", batch)

    result = collator(
        [
            {
                "messages": [
                    {"role": "system", "content": {"format": "json"}},
                    {
                        "role": "user",
                        "content": [
                            "one two three four",
                            {"text": "five six seven eight"},
                            {"image": "image.png", "text": "", "video": "", "fps": 0},
                        ],
                    },
                    {"role": "assistant", "content": 42},
                    {"role": "assistant", "content": "nine ten eleven twelve"},
                ]
            }
        ]
    )

    assert result == captured["batch"]
    messages = captured["batch"][0]
    assert messages[0]["content"] == [{"type": "text", "text": '{"format": "json"}'}]
    assert messages[1]["content"][0]["text"] == "token-0 token-1 token-3"
    assert messages[1]["content"][1]["text"] == "token-0 token-1 token-3"
    assert messages[1]["content"][2] == {"type": "image", "image": str(tmp_path / "image.png")}
    assert messages[2]["content"] == [{"type": "text", "text": "42"}]
    assert messages[3]["content"] == [{"type": "text", "text": "token-0 token-1 token-3"}]


def test_vlm_collator_normalizes_conversations_and_malformed_content():
    """Deprecated conversations and malformed content remain processor-compatible."""

    collator = _bare_vlm_collator()
    captured = {}
    collator._process_multimodal_sample = lambda batch: captured.setdefault("batch", batch)

    result = collator(
        [
            {
                "conversations": [
                    {"role": "system", "content": {"type": "text", "text": "declared"}},
                    {
                        "role": "user",
                        "content": [
                            7,
                            {"unrecognized": "data"},
                            {"type": "text", "text": "kept", "image": "", "video": None},
                        ],
                    },
                ]
            }
        ]
    )

    assert result == captured["batch"]
    messages = captured["batch"][0]
    assert collator._conversations_warned is True
    assert messages[0]["content"] == [{"type": "text", "text": "declared"}]
    assert messages[1]["content"] == [
        {"type": "text", "text": "7"},
        {"type": "text", "text": '{"unrecognized": "data"}'},
        {"type": "text", "text": "kept"},
    ]


def test_vlm_collator_builds_assistant_masks_from_markers():
    """ChatML boundaries provide assistant-only masks without generation tags."""

    collator = _bare_vlm_collator()
    collator._assistant_marker_specs = lambda: [([10, 11], [[99]])]
    tokenized_messages = {
        "input_ids": torch.tensor([[0, 10, 11, 7, 8, 99, 0]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 0]]),
    }

    assistant_masks = collator._build_assistant_masks(tokenized_messages)

    assert assistant_masks.tolist() == [[0, 0, 0, 1, 1, 0, 0]]


def test_vlm_collator_recognizes_chatml_markers_without_generation_tags():
    """ChatML templates do not need Hugging Face generation tags."""

    collator = _bare_vlm_collator()
    collator.tokenizer.chat_template = "<|im_start|>{{ message }}<|im_end|>"
    markers = {
        "<|im_start|>assistant\n": [10, 11],
        "<|im_end|>\n": [99],
        "<|im_end|>": [99],
    }
    collator.tokenizer = MagicMock(
        chat_template=collator.tokenizer.chat_template,
        side_effect=lambda text, **_kwargs: {"input_ids": markers[text]},
    )

    collator._verify_generation_tags()

    assert collator._assistant_marker_specs() == [([10, 11], [[99], [99]])]


def test_vlm_collator_rejects_templates_without_generation_or_chatml_markers():
    """Answer-only loss still rejects templates with no supported boundaries."""

    collator = _bare_vlm_collator()
    collator.tokenizer.chat_template = "{{ messages }}"

    with pytest.raises(ValueError, match=r"requires \{\% generation \%\}"):
        collator._verify_generation_tags()


def test_vlm_collator_skips_unlimited_and_non_text_content_truncation():
    """Disabled limits and media parts are left unchanged."""

    collator = _bare_vlm_collator()
    messages = [{"role": "assistant", "content": [{"type": "image", "image": "image.png"}]}]

    collator._truncate_assistant_content(messages)
    collator._truncate_prompt_content(messages)

    collator.max_assistant_tokens = 1
    collator._truncate_assistant_content(messages)

    assert messages == [{"role": "assistant", "content": [{"type": "image", "image": "image.png"}]}]


def test_vlm_collator_rejects_processor_output_above_training_length():
    """Visual expansion must not silently truncate DFlash sequences."""

    collator = _bare_vlm_collator()
    collator.train_len = 2

    with pytest.raises(ValueError, match="above training_seq_len"):
        collator._pad_sequence_tensors({"input_ids": torch.tensor([[1, 2, 3]])})


# ---------------------------------------------------------------------------
# sample_size truncation tests
# ---------------------------------------------------------------------------


def _make_data_args(sample_size, tmp_path, n_files=5):
    """Create a temp dir with n_files dummy .pt files and an argparse.Namespace."""
    for i in range(n_files):
        torch.save({}, tmp_path / f"sample_{i}.pt")
    return argparse.Namespace(
        mode="offline",
        vlm_processor=None,
        vlm_img_dir=None,
        offline_data_path=str(tmp_path),
        lazy_preprocess=True,
        sample_size=sample_size,
    )


def test_sample_size_positive_truncates(tmp_path):
    """sample_size > 0 should truncate the dataset to that many samples."""
    data_args = _make_data_args(sample_size=3, tmp_path=tmp_path, n_files=5)
    tokenizer = MagicMock()
    module = make_speculative_data_module(tokenizer, data_args, train_len=8)
    assert len(module["train_dataset"]) == 3


def test_sample_size_minus_one_uses_all(tmp_path):
    """sample_size=-1 should use all samples."""
    data_args = _make_data_args(sample_size=-1, tmp_path=tmp_path, n_files=5)
    tokenizer = MagicMock()
    module = make_speculative_data_module(tokenizer, data_args, train_len=8)
    assert len(module["train_dataset"]) == 5


def test_sample_size_zero_raises(tmp_path):
    """sample_size=0 should raise ValueError."""
    data_args = _make_data_args(sample_size=0, tmp_path=tmp_path, n_files=5)
    tokenizer = MagicMock()
    with pytest.raises(ValueError, match="sample_size must be -1"):
        make_speculative_data_module(tokenizer, data_args, train_len=8)


def test_sample_size_larger_than_dataset_uses_all(tmp_path):
    """sample_size > number of files should use all samples without error."""
    data_args = _make_data_args(sample_size=100, tmp_path=tmp_path, n_files=5)
    tokenizer = MagicMock()
    module = make_speculative_data_module(tokenizer, data_args, train_len=8)
    assert len(module["train_dataset"]) == 5


def test_sample_size_no_pt_files_raises(tmp_path):
    """Empty directory should raise ValueError."""
    data_args = argparse.Namespace(
        mode="offline",
        vlm_processor=None,
        vlm_img_dir=None,
        offline_data_path=str(tmp_path),
        lazy_preprocess=True,
        sample_size=-1,
    )
    tokenizer = MagicMock()
    with pytest.raises(ValueError, match=r"No .pt files found"):
        make_speculative_data_module(tokenizer, data_args, train_len=8)


# ---------------------------------------------------------------------------
# get_dummy_inputs() for export forward pass
# ---------------------------------------------------------------------------

TINY_EAGLE_ARCH_CFG = {
    "num_hidden_layers": 1,
    "intermediate_size": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 2,
    "use_last_layernorm": True,
    "use_aux_hidden_state": False,
    "eagle_aux_hidden_state_layer_ids": [],
}

TINY_EAGLE_MODE_CFG = {
    "eagle_architecture_config": {**default_eagle_config, **TINY_EAGLE_ARCH_CFG},
}


@pytest.fixture
def eagle_model():
    model = get_tiny_llama(num_hidden_layers=4)
    mtsp.convert(model, mode=[("eagle", TINY_EAGLE_MODE_CFG)])
    return model


def test_get_dummy_inputs_online(eagle_model):
    """Online EAGLE model returns input_ids only (no base_model_outputs)."""
    eagle_model.eagle_offline = False
    dummy = eagle_model.get_dummy_inputs()
    assert "input_ids" in dummy
    assert "base_model_outputs" not in dummy


def test_get_dummy_inputs_offline(eagle_model):
    """Offline EAGLE model returns input_ids and base_model_outputs with correct shapes."""
    eagle_model.eagle_offline = True
    dummy = eagle_model.get_dummy_inputs()
    assert "input_ids" in dummy
    assert "base_model_outputs" in dummy
    hidden_size = eagle_model.config.hidden_size
    assert dummy["base_model_outputs"]["base_model_hidden_states"].shape[-1] == hidden_size
    assert dummy["base_model_outputs"]["base_model_input_embeds"].shape[-1] == hidden_size


# ---------------------------------------------------------------------------
# OfflineSupervisedDataset tests
# ---------------------------------------------------------------------------

SEQ_LEN = 16
HIDDEN_SIZE = 8


def _make_offline_pt(path, seq_len=SEQ_LEN, hidden_size=HIDDEN_SIZE):
    """Write a realistic .pt file matching the format expected by OfflineSupervisedDataset."""
    data = {
        "input_ids": torch.randint(0, 100, (seq_len,)),
        "hidden_states": torch.randn(seq_len, hidden_size),
        "aux_hidden_states": torch.randn(seq_len, hidden_size),
        "base_model_input_embeds": torch.randn(seq_len, hidden_size),
    }
    torch.save(data, path)
    return data


def test_offline_dataset_len_and_getitem(tmp_path):
    """OfflineSupervisedDataset should load .pt files and return proper keys."""
    n = 3
    files = []
    for i in range(n):
        p = tmp_path / f"sample_{i}.pt"
        _make_offline_pt(p)
        files.append(str(p))

    ds = OfflineSupervisedDataset(files)
    assert len(ds) == n

    item = ds[0]
    assert set(item.keys()) == {
        "input_ids",
        "base_model_hidden_states",
        "aux_hidden_states",
        "attention_mask",
        "loss_mask",
        "labels",
        "base_hidden_prenorm",
    }
    assert item["input_ids"].shape == (SEQ_LEN,)
    assert item["attention_mask"].shape == (SEQ_LEN,)
    assert item["labels"].shape == (SEQ_LEN,)


def test_offline_dataset_labels_shift(tmp_path):
    """Labels should be input_ids shifted left by 1."""
    p = tmp_path / "sample.pt"
    orig = _make_offline_pt(p)
    ds = OfflineSupervisedDataset([str(p)])
    item = ds[0]
    # labels[:-1] should equal input_ids[1:]
    assert torch.equal(item["labels"][:-1], orig["input_ids"][1:])


# ---------------------------------------------------------------------------
# EagleOfflineDataCollator tests
# ---------------------------------------------------------------------------


def test_collator_truncates(tmp_path):
    """Collator should truncate sequences longer than train_len."""
    train_len = 8
    p = tmp_path / "sample.pt"
    _make_offline_pt(p, seq_len=SEQ_LEN)  # SEQ_LEN > train_len
    ds = OfflineSupervisedDataset([str(p)])
    collator = EagleOfflineDataCollator(train_len=train_len)
    batch = collator([ds[0]])
    assert batch["input_ids"].shape == (1, train_len)
    assert batch["base_model_outputs"]["base_model_hidden_states"].shape[1] == train_len


def test_collator_pads(tmp_path):
    """Collator should pad sequences shorter than train_len."""
    train_len = 32
    p = tmp_path / "sample.pt"
    _make_offline_pt(p, seq_len=SEQ_LEN)  # SEQ_LEN < train_len
    ds = OfflineSupervisedDataset([str(p)])
    collator = EagleOfflineDataCollator(train_len=train_len)
    batch = collator([ds[0]])
    assert batch["input_ids"].shape == (1, train_len)
    # Padded region should be zeros
    assert (batch["input_ids"][0, SEQ_LEN:] == 0).all()


def test_collator_batches_multiple(tmp_path):
    """Collator should stack multiple samples into a batch."""
    train_len = SEQ_LEN
    files = []
    for i in range(4):
        p = tmp_path / f"sample_{i}.pt"
        _make_offline_pt(p)
        files.append(str(p))
    ds = OfflineSupervisedDataset(files)
    collator = EagleOfflineDataCollator(train_len=train_len)
    batch = collator([ds[i] for i in range(4)])
    assert batch["input_ids"].shape == (4, train_len)
    assert batch["base_model_outputs"]["base_model_hidden_states"].shape == (
        4,
        train_len,
        HIDDEN_SIZE,
    )
