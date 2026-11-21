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

"""AutoQuantize invariants for modules opted into per-weight search, across methods and flows."""

import copy
import os

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.algorithms import (
    QuantRecipe,
    _iter_tensor_quantizers,
    _resolve_best_recipe,
    _tensor_quantizer_format_signature,
)
from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry, TensorQuantizer
from modelopt.torch.utils import safe_load, safe_save


class _Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(2, 16, 32) * 0.1)
        self.w2 = torch.nn.Parameter(torch.randn(2, 32, 16) * 0.1)

    def forward(self, x):
        return torch.bmm(torch.relu(torch.bmm(x, self.w1)), self.w2)


class _QuantExperts(QuantModule):
    _auto_quantize_per_weight_quantizers = True

    def _setup(self):
        for name in ("w1", "w2"):
            setattr(self, f"{name}_input_quantizer", TensorQuantizer())
            setattr(self, f"{name}_weight_quantizer", TensorQuantizer())
        # Only one weight has an output quantizer: unquantized, its weights still match.
        self.w1_output_quantizer = TensorQuantizer()

    def forward(self, x):
        hidden = torch.bmm(self.w1_input_quantizer(x), self.w1_weight_quantizer(self.w1))
        hidden = self.w2_input_quantizer(torch.relu(self.w1_output_quantizer(hidden)))
        return torch.bmm(hidden, self.w2_weight_quantizer(self.w2))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.dense = torch.nn.Linear(16, 16)
        self.experts = _Experts()
        self.out = torch.nn.Linear(16, 16)

    def forward(self, x):
        return self.out(self.experts(self.dense(x)))


def _deny(*entries):
    return {"quant_cfg": [{"quantizer_name": "*", "enable": False}, *entries], "algorithm": "max"}


_INT4_W = {"num_bits": 4, "block_sizes": {-1: 16}}
_CANDIDATES = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int4_weight_only": _deny({"quantizer_name": "*weight_quantizer", "cfg": _INT4_W}),
    "w4a8": _deny(
        {"quantizer_name": "*weight_quantizer", "cfg": [_INT4_W, {"num_bits": 8, "axis": 0}]},
        {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
    ),
    # Full-path glob: matches no module-relative name, so the experts stay BF16 under it.
    "experts_path_no_op": _deny({"quantizer_name": "*.experts.*weight_quantizer", "cfg": _INT4_W}),
}
_METHODS = {
    "gradient": "gradient",
    "kl_div": "kl_div",
    "kl_div_lowest": {"method": "kl_div", "context_quant_level": "lowest"},
    "aumann_shapley": "aumann_shapley",
}


@pytest.fixture(autouse=True)
def _registered():
    QuantModuleRegistry.register({_Experts: "test.PerWeightExperts"})(_QuantExperts)
    yield
    QuantModuleRegistry.unregister(_Experts)


def _search(method, formats, bits, **kwargs):
    model = _Model()
    data = [torch.randn(2, 4, 16) for _ in range(2)]
    method_name = method if isinstance(method, str) else method["method"]
    model, state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": bits},
        quantization_formats=[(copy.deepcopy(_CANDIDATES[f]), f) for f in formats],
        data_loader=data,
        forward_step=lambda model, batch: model(batch),
        loss_func=(lambda output, batch: output.sum()) if method_name == "gradient" else None,
        num_calib_steps=2,
        num_score_steps=2,
        method=method,
        **kwargs,
    )
    return model, state


def _formats(model, names=("experts", "dense", "out")):
    """Each searched module's quantizer formats, as the model holds them."""
    return {
        f"{name}.{attr}": tuple(
            _tensor_quantizer_format_signature(q) if q.is_enabled else "disabled"
            for q in _iter_tensor_quantizers(quantizer)
        )
        for name in names
        for attr, quantizer in vars(model.get_submodule(name))["_modules"].items()
        if attr.endswith("quantizer")
    }


def _replayed(config):
    """The frozen config applied as it is used: quantize and calibrate a fresh model."""
    data = [torch.randn(2, 4, 16) for _ in range(2)]

    def forward_loop(model):
        for batch in data:
            model(batch)

    return mtq.quantize(_Model(), config, forward_loop)


@pytest.mark.parametrize("method", list(_METHODS))
@pytest.mark.parametrize("bits", [5.0, 9.0, 16.0])
def test_per_weight_search_invariants(method, bits, tmp_path):
    formats = list(_CANDIDATES)
    checkpoint = str(tmp_path / "search.pth")
    model, state = _search(_METHODS[method], formats, bits, checkpoint=checkpoint)
    stats = state["candidate_stats"]["experts.quant_recipe"]
    [hparam] = model.experts._hparams_for_scoring

    # I1: replaying the frozen config gives every searched module the formats it was scored with.
    # Dense layers that selected a full-path candidate are excluded: their replay is #3508.
    checked = ["experts"] + [
        name
        for name in ("dense", "out")
        if not str(state["best"]["recipe"][f"{name}.quant_recipe"]).startswith("experts_path")
    ]
    config = mtq.get_auto_quantize_config(state)
    assert _formats(_replayed(config), checked) == _formats(model, checked)

    # I2: a candidate is costed at BF16 exactly when it leaves the experts' weights unquantized.
    for recipe, cost in zip(stats["formats"], stats["costs"]):
        quantizers = hparam._all_quantizer_choices[recipe][model.experts]
        quantized = any(
            q.is_enabled
            for attr, value in quantizers.items()
            if "weight_quantizer" in attr
            for q in _iter_tensor_quantizers(value)
        )
        assert (cost < stats["uncompressed_cost"]) == quantized, recipe

    # I3: the ladder, and so KL selection, the KL context and the corner, ascend in resolved cost.
    assert stats["costs"] == sorted(stats["costs"])
    assert [hparam.get_cost(r, cost_weight=1.0) for r in hparam.ladder] == sorted(
        hparam.get_cost(r, cost_weight=1.0) for r in hparam.ladder
    )

    # I4: an offline re-solve, and a resumed search, select what the live search selected.
    saved = safe_load(checkpoint)
    assert _resolve_best_recipe(saved, {"effective_bits": bits}) == state["best"]["recipe"]
    resumed, resumed_state = _search(_METHODS[method], formats, bits, checkpoint=checkpoint)
    assert resumed_state["best"]["recipe"] == state["best"]["recipe"]
    assert _formats(resumed) == _formats(model)


def test_replay_follows_formats_a_searcher_prunes(monkeypatch):
    searcher = mtq.algorithms.AUTO_QUANTIZE_SEARCHERS["aumann_shapley"]
    exclude = searcher._exclude_non_finite_candidates

    def int4_diverges(self, no_quant):
        stats = self.candidate_stats["experts.quant_recipe"]
        names = [str(recipe).split("(")[0] for recipe in stats["formats"]]
        stats["raw_scores"][names.index("int4_weight_only")] = float("inf")
        return exclude(self, no_quant)

    monkeypatch.setattr(searcher, "_exclude_non_finite_candidates", int4_diverges)
    model, state = _search("aumann_shapley", ["int4_weight_only", "int8"], 9.0)
    formats = state["candidate_stats"]["experts.quant_recipe"]["formats"]
    assert not any(str(recipe).startswith("int4_weight_only") for recipe in formats)
    assert _formats(_replayed(mtq.get_auto_quantize_config(state)), ("experts",)) == _formats(
        model, ("experts",)
    )


@pytest.mark.parametrize("method", ["gradient", "kl_div"])
def test_per_weight_format_shards_merge_into_the_live_search(method, tmp_path):
    formats = list(_CANDIDATES)
    _, live = _search(method, formats, 9.0)
    shards = []
    for index in range(2):
        shard = str(tmp_path / f"shard{index}.pth")
        _search(method, formats, 9.0, checkpoint=shard, format_shard=(index, 2))
        shards.append(shard)
    merged = str(tmp_path / "merged.pth")
    mtq.merge_auto_quantize_checkpoints(shards, merged)
    _, merged_state = _search(method, formats, 9.0, checkpoint=merged)
    assert merged_state["best"]["recipe"] == live["best"]["recipe"]


def test_per_weight_modules_replay_every_rule_form_as_scored():
    named = _deny(
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
        # An enable-only rule keeps the format before it; a rule may target a stage a list config
        # created.
        {"quantizer_name": "*w[12]*weight_quantizer", "cfg": [_INT4_W, {"num_bits": 8, "axis": 0}]},
        {"quantizer_name": "*w1_weight_quantizer", "enable": False},
        {"quantizer_name": "*w1_weight_quantizer", "enable": True},
        {"quantizer_name": "*weight_quantizer.0", "enable": False},
        # A parent_class rule applies to the module's own quantizers, here a list config over a
        # list config without reaching its stages.
        {"quantizer_name": "*input_quantizer", "cfg": [{"num_bits": 8, "axis": None}] * 2},
        {
            "quantizer_name": "*input_quantizer",
            "parent_class": "test.PerWeightExperts",
            "cfg": [{"num_bits": 8, "axis": None}] * 2,
        },
    )
    _CANDIDATES["named"] = named
    try:
        model, state = _search("gradient", ["named"], 8.0)
    finally:
        del _CANDIDATES["named"]
    replayed = _replayed(mtq.get_auto_quantize_config(state))
    # Dense replay of such patterns is tracked in #3508.
    assert _formats(replayed, ("experts",)) == _formats(model, ("experts",))
    assert [q.is_enabled for q in model.experts.w1_weight_quantizer] == [False, True]
    assert model.experts.w2_input_quantizer.is_enabled
    assert isinstance(state["best"]["recipe"]["experts.quant_recipe"], QuantRecipe)


def test_path_rules_of_other_modules_leave_unquantized_per_weight_modules_alone():
    path_rule = _deny(
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
        {"quantizer_name": "experts.w[12]_*_quantizer", "cfg": _INT4_W},
    )
    _CANDIDATES["path_rule"] = path_rule
    try:
        _, state = _search("gradient", ["path_rule"], 8.0)
    finally:
        del _CANDIDATES["path_rule"]
    best = state["best"]["recipe"]
    best["experts.quant_recipe"] = QuantRecipe(quant_cfg=None)
    replayed = _replayed(mtq.get_auto_quantize_config(state))
    assert set(_formats(replayed, ("experts",)).values()) == {("disabled",)}
    assert replayed.dense.weight_quantizer.is_enabled


def test_a_target_only_resolved_costs_meet_is_searched():
    # Nominally 16 bits, as its INT4 rule is disabled, but the toggle quantizes the experts.
    _CANDIDATES["toggled_int4"] = _deny(
        {"quantizer_name": "*w[12]_weight_quantizer", "cfg": _INT4_W, "enable": False},
        {"quantizer_name": "*w[12]_weight_quantizer", "enable": True},
    )
    try:
        model, _ = _search("gradient", ["toggled_int4"], 8.0)
    finally:
        del _CANDIDATES["toggled_int4"]
    assert model.experts.w1_weight_quantizer.num_bits == 4


def test_per_weight_modules_are_priced_at_their_resolved_weight_format():
    # The named rule overrides the recipe's 4-bit default: the experts hold, and cost, 8 bits.
    int4_with_int8_experts = _deny(
        {"quantizer_name": "*weight_quantizer", "cfg": _INT4_W},
        {"quantizer_name": "*w[12]_weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
    )
    _CANDIDATES["int4_int8_experts"] = int4_with_int8_experts
    try:
        _, state = _search("gradient", ["int4_int8_experts"], 16.0)
    finally:
        del _CANDIDATES["int4_int8_experts"]
    stats = state["candidate_stats"]["experts.quant_recipe"]
    [cost] = [c for f, c in zip(stats["formats"], stats["costs"]) if not f.is_no_quant]
    assert cost == stats["uncompressed_cost"] / 2


def test_a_candidate_leaving_per_weight_modules_unquantized_warns():
    with pytest.warns(UserWarning, match="leaves the weights of .*_Experts unquantized"):
        _search("gradient", ["experts_path_no_op"], 16.0)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes")
@pytest.mark.parametrize("saved_before_cost_versions", [False, True])
def test_a_search_resumes_from_a_read_only_checkpoint(tmp_path, saved_before_cost_versions):
    checkpoint = tmp_path / "scores" / "search.pth"
    _, state = _search("gradient", list(_CANDIDATES), 9.0, checkpoint=str(checkpoint))
    if saved_before_cost_versions:  # its resume refreshes the costs, which it cannot save here
        saved = safe_load(checkpoint)
        del saved["cost_version"]
        safe_save(saved, checkpoint)
    checkpoint.chmod(0o444)
    checkpoint.parent.chmod(0o555)
    try:
        _, resumed = _search("gradient", list(_CANDIDATES), 9.0, checkpoint=str(checkpoint))
    finally:
        checkpoint.parent.chmod(0o755)
        checkpoint.chmod(0o644)
    assert resumed["best"]["recipe"] == state["best"]["recipe"]
