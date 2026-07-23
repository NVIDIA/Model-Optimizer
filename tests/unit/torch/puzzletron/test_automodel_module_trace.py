# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for opt-in AutoModel synchronized module tracing."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modelopt.torch.puzzletron.plugins.automodel import module_trace, solution_launch
from modelopt.torch.puzzletron.plugins.automodel.module_trace import synchronized_module_trace


class _PassThrough(nn.Module):
    def forward(self, value):
        return value


class _SharedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = _PassThrough()
        self.up_proj = _PassThrough()
        self.down_proj = _PassThrough()

    def forward(self, value):
        value = self.gate_proj(value)
        value = self.up_proj(value)
        return self.down_proj(value)


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_experts = _SharedExperts()
        self.experts = _PassThrough()

    def forward(self, value):
        return self.shared_experts(value) + self.experts(value)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = _PassThrough()
        self.linear_attn = _PassThrough()
        self.mlp = _MoE()

    def forward(self, value):
        return self.mlp(self.linear_attn(self.input_layernorm(value)))


@pytest.fixture
def recipe():
    layers = {3: _Layer(), 4: _Layer()}
    return SimpleNamespace(
        layers=layers,
        _find_decoder_layer=lambda layer_idx: layers.get(layer_idx),
    )


@pytest.fixture
def fake_cuda(monkeypatch):
    synchronized = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)
    return synchronized


def test_disabled_trace_is_a_noop(monkeypatch, capsys, recipe, fake_cuda):
    monkeypatch.delenv("PUZZLETRON_TRACE_MODULE_SYNCS", raising=False)
    monkeypatch.delenv("PUZZLETRON_TRACE_MODULE_LAYER", raising=False)

    with synchronized_module_trace(recipe):
        recipe.layers[4](torch.ones(1))

    assert fake_cuda == []
    assert capsys.readouterr().out == ""


def test_trace_filters_layer_and_emits_synchronized_projection_phases(
    monkeypatch, capsys, recipe, fake_cuda
):
    monkeypatch.setenv("PUZZLETRON_TRACE_MODULE_SYNCS", "1")
    monkeypatch.setenv("PUZZLETRON_TRACE_MODULE_LAYER", "4")

    with synchronized_module_trace(recipe):
        recipe.layers[3](torch.ones(1))
        recipe.layers[4](torch.ones(1))

    lines = capsys.readouterr().out.splitlines()
    gate_lines = [line for line in lines if "module=shared_experts.gate_proj " in line]
    assert [line.rsplit("phase=", 1)[1] for line in gate_lines] == [
        "enter",
        "inputs_synchronized",
        "returned",
        "output_synchronized",
    ]
    assert all("layer=4" in line for line in lines)
    assert fake_cuda
    assert set(fake_cuda) == {torch.device("cuda", 7)}

    sync_count = len(fake_cuda)
    recipe.layers[4](torch.ones(1))
    assert len(fake_cuda) == sync_count
    assert capsys.readouterr().out == ""


def test_trace_filters_modules(monkeypatch, capsys, recipe, fake_cuda):
    monkeypatch.setenv("PUZZLETRON_TRACE_MODULE_SYNCS", "1")
    monkeypatch.setenv("PUZZLETRON_TRACE_MODULE_LAYER", "4")
    monkeypatch.setenv("PUZZLETRON_TRACE_MODULE_FILTER", "shared_experts.down_proj")

    with synchronized_module_trace(recipe):
        recipe.layers[4](torch.ones(1))

    lines = capsys.readouterr().out.splitlines()
    assert lines
    assert all("module=shared_experts.down_proj " in line for line in lines)
    assert len(fake_cuda) == 2


@pytest.mark.parametrize("value", [None, "bad", "-1"])
def test_enabled_trace_rejects_invalid_layer(monkeypatch, recipe, value):
    monkeypatch.setenv("PUZZLETRON_TRACE_MODULE_SYNCS", "1")
    if value is None:
        monkeypatch.delenv("PUZZLETRON_TRACE_MODULE_LAYER", raising=False)
    else:
        monkeypatch.setenv("PUZZLETRON_TRACE_MODULE_LAYER", value)

    with (
        pytest.raises(ValueError, match="PUZZLETRON_TRACE_MODULE_LAYER"),
        synchronized_module_trace(recipe),
    ):
        pass


def test_trace_rejects_missing_layer(monkeypatch, recipe):
    monkeypatch.setenv("PUZZLETRON_TRACE_MODULE_SYNCS", "1")
    monkeypatch.setenv("PUZZLETRON_TRACE_MODULE_LAYER", "9")

    with (
        pytest.raises(ValueError, match="requested trace layer 9"),
        synchronized_module_trace(recipe),
    ):
        pass


def test_trace_removes_hooks_after_exception(monkeypatch, capsys, recipe, fake_cuda):
    monkeypatch.setenv("PUZZLETRON_TRACE_MODULE_SYNCS", "1")
    monkeypatch.setenv("PUZZLETRON_TRACE_MODULE_LAYER", "4")

    with pytest.raises(RuntimeError, match="stop"), synchronized_module_trace(recipe):
        raise RuntimeError("stop")

    capsys.readouterr()
    fake_cuda.clear()
    recipe.layers[4](torch.ones(1))
    assert fake_cuda == []
    assert capsys.readouterr().out == ""


def test_missing_optional_modules_are_reported(monkeypatch, capsys):
    layer = nn.Module()
    layer.mlp = _PassThrough()
    recipe = SimpleNamespace(_find_decoder_layer=lambda layer_idx: layer)
    monkeypatch.setenv("PUZZLETRON_TRACE_MODULE_SYNCS", "1")
    monkeypatch.setenv("PUZZLETRON_TRACE_MODULE_LAYER", "4")

    with synchronized_module_trace(recipe):
        pass

    output = capsys.readouterr().out
    assert "module=shared_experts phase=unavailable" in output
    assert "module=routed_experts phase=unavailable" in output


def test_non_owning_pipeline_rank_reports_without_error(monkeypatch, capsys):
    recipe = SimpleNamespace(_find_decoder_layer=lambda layer_idx: None)
    monkeypatch.setenv("PUZZLETRON_TRACE_MODULE_SYNCS", "1")
    monkeypatch.setenv("PUZZLETRON_TRACE_MODULE_LAYER", "4")
    monkeypatch.setattr(module_trace, "_layer_exists_on_any_rank", lambda layer: True)

    with synchronized_module_trace(recipe):
        pass

    assert "module=decoder phase=not_owned" in capsys.readouterr().out


def test_candidate_scoring_scopes_trace_inside_candidate_context(monkeypatch, tmp_path):
    events = []

    @contextmanager
    def candidate_context(*args, **kwargs):
        events.append("candidate_enter")
        try:
            yield
        finally:
            events.append("candidate_exit")

    @contextmanager
    def trace_context(recipe):
        events.append("trace_enter")
        try:
            yield
        finally:
            events.append("trace_exit")

    def captures():
        events.append("iterate")
        yield None, None

    recipe = SimpleNamespace(
        has_outputs=False,
        _groups=None,
        tensor_parallel_group=lambda: None,
        iterate_captures=captures,
        observability_metadata=dict,
    )
    monkeypatch.setattr(solution_launch, "_is_output_writer", lambda recipe: False)
    monkeypatch.setattr(solution_launch, "_candidate_execution_context", candidate_context)
    monkeypatch.setattr(solution_launch, "synchronized_module_trace", trace_context, raising=False)

    solution_launch._score_candidate(
        recipe,
        cache=None,
        params={},
        output_dir=tmp_path,
        scoring={},
        name="realized_0008",
        payload={},
        prune_target={"layer_idx": 4},
    )

    assert events == [
        "candidate_enter",
        "trace_enter",
        "iterate",
        "trace_exit",
        "candidate_exit",
    ]
