# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DSV4 native-MTP data preparation and parity validation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COLLECT_DIR = _REPO_ROOT / "examples/speculative_decoding/collect_hidden_states"
sys.path.insert(0, str(_COLLECT_DIR))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _COLLECT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = _load_script("prepare_dsv4_mtp_data")
validate = _load_script("validate_dsv4_mtp_features")


def _valid_row() -> dict:
    reasoning = "We need to reason carefully about the problem."
    answer = "The answer is 42."
    return {
        "uuid": "sample-1",
        "category": "math",
        "messages": [
            {"role": "system", "content": "Be accurate."},
            {"role": "user", "content": "What is the answer?"},
        ],
        "reasoning_content": reasoning,
        "generation": answer,
        "finish_reason": "stop",
        "serialized_output": [
            {
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": answer,
            }
        ],
    }


def test_normalize_generation_row_preserves_reasoning_and_answer():
    record, reason = prepare.normalize_generation_row(_valid_row())

    assert reason is None
    assert record["conversation_id"] == "sample-1"
    assert [message["role"] for message in record["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert record["messages"][-1]["reasoning_content"].startswith("We need to reason")
    assert record["messages"][-1]["content"] == "The answer is 42."


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ({"finish_reason": "length"}, "incomplete_finish_reason"),
        ({"reasoning_content": ""}, "empty_reasoning"),
        ({"generation": ""}, "empty_final_answer"),
        ({"generation": "bad </think> marker"}, "embedded_control_marker"),
        ({"uuid": "../unsafe"}, "invalid_conversation_id"),
    ],
)
def test_normalize_generation_row_rejects_unfit_rows(mutation, expected_reason):
    row = _valid_row()
    row.update(mutation)

    _, reason = prepare.normalize_generation_row(row)

    assert reason == expected_reason


def test_category_budget_is_equal_by_default_and_exact():
    budgets = prepare._budgets(1_000_003, prepare.DEFAULT_CATEGORY_WEIGHTS)

    assert sum(budgets.values()) == 1_000_003
    assert max(budgets.values()) - min(budgets.values()) == 1


def test_tensor_error_metrics_identical_bf16():
    tensor = torch.randn(17, 4, 8, dtype=torch.bfloat16)

    metrics = validate.tensor_error_metrics(tensor, tensor.clone(), chunk_elements=23)

    assert metrics["cosine"] == pytest.approx(1.0)
    assert metrics["relative_l2"] == 0.0
    assert metrics["max_absolute"] == 0.0


def test_tensor_error_metrics_rejects_precision_mismatch():
    with pytest.raises(TypeError, match="dtype mismatch"):
        validate.tensor_error_metrics(
            torch.ones(2, dtype=torch.bfloat16),
            torch.ones(2, dtype=torch.float32),
        )


def test_dsv4_template_masks_complete_assistant_span():
    template = (
        _REPO_ROOT / "examples/speculative_decoding/input_conversations/dsv4_thinking.jinja"
    ).read_text()

    generation_start = template.index("{% generation %}")
    reasoning = template.index("reasoning_content")
    final_answer = template.index("message['content']", reasoning)
    generation_end = template.index("{% endgeneration %}")
    assert generation_start < reasoning < final_answer < generation_end
