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

"""Tests specific to the aumann_shapley AutoQuantize method.

Shared behavior (search across models/formats, checkpoint resume) is covered by the method
parametrizations in test_autoquant.py; this module pins the method's own guarantees: config
parity with the standard builder, the damage model, the path-integral completeness property,
damage-bound search, and use of the shared linear-program solver.
"""

import copy
import functools
import itertools
import math

import numpy as np
import pytest
import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.model_quant as _model_quant
from modelopt.torch.quantization._auto_quantize_shapley import (
    AutoQuantizeAumannShapleySearcher,
    _anchor_ceiling,
    _as_seed_coverage,
    _predict_damage,
)
from modelopt.torch.quantization.algorithms import QuantRecipe
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.utils.distributed import DistributedProcessGroup

SEARCH_FORMATS = [mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG]


class _Attention(torch.nn.Module):
    """Minimal attention block."""

    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(32, 32)
        self.k_proj = torch.nn.Linear(32, 32)
        self.v_proj = torch.nn.Linear(32, 32)
        self.o_proj = torch.nn.Linear(32, 32)

    def forward(self, x):
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            x = layer(x)
        return x


class _Block(torch.nn.Module):
    """Minimal transformer block."""

    def __init__(self, seed=0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.attn = _Attention()
            self.mlp = torch.nn.Linear(32, 32)
        self._input_seed = seed + 1000

    def forward(self, x):
        return self.mlp(self.attn(x))

    def get_input(self):
        generator = torch.Generator().manual_seed(self._input_seed)
        return torch.randn(1, 4, 32, generator=generator)


class _OneLinear(torch.nn.Module):
    """Single-linear model."""

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.fc = torch.nn.Linear(32, 32)

    def forward(self, x):
        return self.fc(x)

    def get_input(self):
        generator = torch.Generator().manual_seed(1000)
        return torch.randn(1, 4, 32, generator=generator)


def _search(model, method="aumann_shapley", effective_bits=6.0, method_options=None, **kwargs):
    """Run auto_quantize with the given method."""
    return mtq.auto_quantize(
        model,
        constraints={"effective_bits": effective_bits} if effective_bits is not None else None,
        quantization_formats=list(SEARCH_FORMATS),
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=(lambda output, data: output.sum()) if method == "gradient" else None,
        num_calib_steps=2,
        num_score_steps=2,
        method=method,
        method_options=method_options,
        **kwargs,
    )


@pytest.fixture(scope="module")
def shapley_state():
    """Shared aumann_shapley search state."""
    _model, state = _search(_Block(), method_options={"num_path_nodes": 2})
    return state


def test_damage_model_and_score_monotonicity(shapley_state):
    """Damage model and score monotonicity."""
    assert shapley_state["method"] == "aumann_shapley"
    assert shapley_state["best"]["is_satisfied"]
    assert shapley_state["best"]["predicted_damage"] >= 0
    assert shapley_state["best"]["predicted_damage_valid"] is True

    damage_model = shapley_state["damage_model"]
    assert damage_model["link"] == "coverage"
    assert damage_model["c"] >= damage_model["f_corner"] > 0
    assert damage_model["completeness"] > 0
    assert damage_model["valid"] is True
    assert isinstance(damage_model["approximation_flags"], list)
    assert damage_model["damage_reference"] == {"type": "unquantized"}
    assert shapley_state["scoring_signature"]["num_path_nodes"] == 2

    for stat in shapley_state["candidate_stats"].values():
        assert len(stat["formats"]) == len(stat["scores"]) == len(stat["costs"])
        assert all(
            stat["scores"][i] >= stat["scores"][i + 1] - 1e-12
            for i in range(len(stat["scores"]) - 1)
        )


def test_identical_scores_emit_identical_configs(shapley_state):
    """With identical stats and scores, the emitted config must match the gradient method's
    dict-for-dict: config emission and solving are shared, only scoring differs."""
    _model, gradient_state = _search(_Block(), method="gradient")

    shapley = copy.deepcopy(shapley_state)
    gradient = copy.deepcopy(gradient_state)
    assert list(shapley["candidate_stats"]) == list(gradient["candidate_stats"])
    for i, name in enumerate(gradient["candidate_stats"]):
        num_choices = len(gradient["candidate_stats"][name]["scores"])
        synthetic = [float(num_choices - j) * (1.0 + 0.1 * i) for j in range(num_choices)]
        gradient["candidate_stats"][name]["scores"] = list(synthetic)
        shapley["candidate_stats"][name]["scores"] = list(synthetic)

    for bits in (14.0, 9.0, 6.0):
        config_gradient = mtq.get_auto_quantize_config(gradient, {"effective_bits": bits})
        config_shapley = mtq.get_auto_quantize_config(shapley, {"effective_bits": bits})
        assert config_shapley == config_gradient, f"configs diverge at effective_bits={bits}"


def test_config_applies_and_resolve_tightens(shapley_state):
    """Config applies and resolve tightens."""
    config = mtq.get_auto_quantize_config(shapley_state)
    assert config["algorithm"] == "max"
    assert config["quant_cfg"][0] == {"quantizer_name": "*", "enable": False}

    model = _Block(seed=1)
    mtq.quantize(model, config, lambda m: m(m.get_input()))
    with torch.no_grad():
        model(model.get_input())

    def enabled_entries(bits):
        config = mtq.get_auto_quantize_config(shapley_state, {"effective_bits": bits})
        return sum(1 for entry in config["quant_cfg"] if entry.get("enable"))

    assert enabled_entries(5.0) >= enabled_entries(14.0)


def test_completeness_one_group():
    """With a single group the path integral must recover the measured corner damage."""
    _model, state = _search(
        _OneLinear(), effective_bits=16.0, method_options={"num_path_nodes": 32}
    )
    assert state["damage_model"]["completeness"] == pytest.approx(1.0, rel=0.05)


def test_damage_bound_mode_respects_the_quote(shapley_state):
    """Damage-bound mode keeps the selected recipe within the requested estimate."""
    epsilon = 0.5 * shapley_state["damage_model"]["f_corner"]

    _model, sla_state = _search(
        _Block(), effective_bits=None, method_options={"max_predicted_damage": epsilon}
    )
    assert sla_state["best"]["is_satisfied"]
    assert sla_state["best"]["predicted_damage"] <= epsilon + 1e-12


def _synthetic_searcher(n_groups=5, seed=0):
    """Searcher preloaded with synthetic candidate stats."""
    rng = np.random.default_rng(seed)
    aggressive = QuantRecipe("INT4_BLOCKWISE_WEIGHT_ONLY_CFG")
    moderate = QuantRecipe("INT8_DEFAULT_CFG")
    no_quant = QuantRecipe(quant_cfg=None)

    searcher = AutoQuantizeAumannShapleySearcher()
    searcher.candidate_stats = {}
    b_total = 0.0
    for i in range(n_groups):
        numel = float(rng.integers(100, 1000))
        b8 = float(rng.uniform(0.001, 0.05))
        b4 = b8 + float(rng.uniform(0.001, 0.1))
        b_total += b4
        searcher.candidate_stats[f"g{i}.quant_recipe"] = {
            "formats": [aggressive, moderate, no_quant],
            "scores": [b4, b8, 0.0],
            "costs": [numel * aggressive.compression, numel * moderate.compression, numel],
            "module_names": [f"g{i}"],
            "quantizer_attrs": {f"g{i}": ("input_quantizer", "weight_quantizer")},
            "cost_weight": 1.0,
            "allow_no_quant": True,
            "is_fixed": False,
            "uncompressed_cost": numel,
        }
    searcher.damage_model = {
        "link": "coverage",
        "c": 0.5,
        "f_corner": 0.5 * (1 - np.exp(-b_total)),
        "valid": True,
    }
    searcher.cost_model = "weight"
    searcher.config = {**searcher.default_search_config}
    return searcher


def _selected(searcher, best):
    """Total score and cost summed over the selected entries."""
    score = cost = 0.0
    for info in best.values():
        score += info["scores"]
        cost += info["costs"]
    return score, cost


def test_damage_bound_search_is_exact():
    """Damage-bound LPS matches exhaustive allocation."""
    for seed in range(3):
        searcher = _synthetic_searcher(seed=seed)
        c = searcher.damage_model["c"]
        for eps_frac in (0.9, 0.1, 0.02):
            epsilon = eps_frac * searcher.damage_model["f_corner"]
            searcher.config["max_predicted_damage"] = epsilon
            best, is_satisfied = searcher.run_search_with_stats(max_weight_size=np.inf)
            score, cost = _selected(searcher, best)
            assert _predict_damage(c, score) <= epsilon + 1e-12
            assert is_satisfied

            budget = -np.log(1.0 - epsilon / c)
            stats = searcher.candidate_stats
            names = list(stats)
            optimal = min(
                (
                    sum(stats[n]["costs"][k] for n, k in zip(names, combo, strict=True))
                    for combo in itertools.product(
                        *[range(len(stats[n]["formats"])) for n in names]
                    )
                    if sum(stats[n]["scores"][k] for n, k in zip(names, combo, strict=True))
                    <= budget + 1e-12
                ),
                default=np.inf,
            )
            assert cost == pytest.approx(optimal, rel=1e-9)


def _brute_force_min_score(stats, budget):
    """Exhaustive minimum-score allocation."""
    names = list(stats)
    return min(
        (
            sum(stats[n]["scores"][k] for n, k in zip(names, combo, strict=True))
            for combo in itertools.product(*[range(len(stats[n]["formats"])) for n in names])
            if sum(stats[n]["costs"][k] for n, k in zip(names, combo, strict=True)) <= budget + 1e-9
        ),
        default=np.inf,
    )


def test_linear_program_search_is_exact():
    """The shared linear-program search matches exhaustive allocation."""
    for seed in range(3):
        searcher = _synthetic_searcher(seed=seed)
        stats = searcher.candidate_stats
        total = sum(stat["uncompressed_cost"] for stat in stats.values())
        for fraction in (0.9, 0.5):
            budget = total * fraction
            best, is_satisfied = searcher.run_search_with_stats(budget)
            assert is_satisfied
            score, cost = _selected(searcher, best)
            assert cost <= budget + 1e-6
            assert score == pytest.approx(_brute_force_min_score(stats, budget), rel=1e-9)


def test_coverage_inversion_recovers_forward_model():
    """Coverage inversion recovers coefficients integrated over the full path."""
    c = 0.9
    a_true = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    attributions = []
    for index, a_i in enumerate(a_true):
        coefficients = np.array([1.0])
        for a_j in np.delete(a_true, index):
            coefficients = np.convolve(coefficients, [1.0, -a_j])
        integral = sum(
            coefficient / (degree + 1) for degree, coefficient in enumerate(coefficients)
        )
        attributions.append(c * a_i * integral)
    attributions = np.asarray(attributions)

    a, b, converged = _as_seed_coverage(attributions, c=c)
    assert converged, a
    assert np.allclose(a, a_true, rtol=1e-5)
    assert np.allclose(b, -np.log1p(-a_true), rtol=1e-5)


def test_both_targets_rejected_before_model_conversion():
    """Both targets rejected before model conversion."""

    model = _Block()
    with pytest.raises(ValueError, match="not both"):
        _search(model, effective_bits=8.0, method_options={"max_predicted_damage": 1e-3})
    assert type(model.mlp) is torch.nn.Linear
    assert not any(isinstance(m, TensorQuantizer) for m in model.modules())


def _inject_scores_and_corner(monkeypatch, injected, corner):
    def inject(self, is_param_grad_enabled):
        no_quant = QuantRecipe(quant_cfg=None)
        self._corner_kl_sum = torch.tensor(float(corner))
        self._score_tokens = 1
        for hparam in self._configurable_hparams():
            for recipe in hparam.choices:
                if recipe == no_quant:
                    continue
                value = injected[str(recipe).split("(")[0]]
                for module in hparam.score_modules:
                    hparam._importance_dict[recipe][module] = torch.tensor(value)

    monkeypatch.setattr(AutoQuantizeAumannShapleySearcher, "_estimate_auto_quantize_scores", inject)


def test_zero_corner_with_mixed_sign_attributions_is_invalid(monkeypatch):
    """A zero measured corner with remaining positive attribution mass must invalidate the
    fit: a signed cancellation is not a zero-damage model."""
    _inject_scores_and_corner(
        monkeypatch,
        {"INT4_BLOCKWISE_WEIGHT_ONLY_CFG": -0.01, "INT8_DEFAULT_CFG": 0.02},
        corner=0.0,
    )
    _model, state = _search(_OneLinear(), effective_bits=16.0)
    damage_model = state["damage_model"]
    assert "zero_corner_with_attribution_mass" in damage_model["approximation_flags"]
    assert damage_model["valid"] is False
    assert state["best"]["predicted_damage_valid"] is False
    # An invalidated fit leaves c = 0.0; the quote must not read as "zero damage".
    assert math.isnan(state["best"]["predicted_damage"])


def test_nonconvergent_coverage_inversion_is_flagged(monkeypatch):
    """Persist the reason a finite coverage fit could not be inverted."""
    _inject_scores_and_corner(
        monkeypatch,
        {"INT4_BLOCKWISE_WEIGHT_ONLY_CFG": 0.2, "INT8_DEFAULT_CFG": 0.2},
        corner=0.01,
    )
    with pytest.warns(UserWarning, match="inversion_not_converged"):
        _model, state = _search(_OneLinear(), effective_bits=16.0)

    damage_model = state["damage_model"]
    assert damage_model["inversion_converged"] is False
    assert damage_model["valid"] is False
    assert "inversion_not_converged" in damage_model["approximation_flags"], damage_model


def test_projected_damage_model_matches_solver_scores(monkeypatch):
    """The persisted link values must be the quote-operative (projected) ones, with the
    projection recorded and the unprojected corner anchor kept alongside."""
    _inject_scores_and_corner(
        monkeypatch,
        {"INT4_BLOCKWISE_WEIGHT_ONLY_CFG": 0.01, "INT8_DEFAULT_CFG": 0.02},
        corner=0.01,
    )
    _model, state = _search(_OneLinear(), effective_bits=16.0)
    damage_model = state["damage_model"]
    assert "monotonicity_projection" in damage_model["approximation_flags"]
    assert damage_model["monotonicity_adjustment"] > 0

    (name,) = next(iter(damage_model["b"].values())).keys()
    stat = state["candidate_stats"][name]
    format_labels = [str(recipe) for recipe in stat["formats"]]
    for label, values in damage_model["b"].items():
        assert values[name] == pytest.approx(stat["scores"][format_labels.index(label)])

    # The unprojected link stays exactly anchored; the projected corner may exceed it.
    unprojected_corner = sum(
        values[name]
        for label, values in damage_model["b_unprojected"].items()
        if label.startswith("INT4")
    )
    assert _predict_damage(damage_model["c"], unprojected_corner) == pytest.approx(
        damage_model["f_corner"], rel=0.02
    )
    assert damage_model["projected_corner_damage"] >= damage_model["f_corner"] * 0.99


def test_custom_format_identity_across_search_spaces():
    """Identical custom formats under different auto-generated names are one format."""
    custom = {
        "quant_cfg": [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}}],
        "algorithm": "max",
    }
    model = _Block()
    with pytest.warns(UserWarning, match="custom quantization formats"):
        _model, state = mtq.auto_quantize(
            model,
            constraints={"effective_bits": 12.0},
            module_search_spaces=[
                {"module_name_patterns": ["*attn*"], "quantization_formats": [dict(custom)]},
                {"module_name_patterns": ["*mlp*"], "quantization_formats": [dict(custom)]},
            ],
            fixed_quantization_config="INT8_DEFAULT_CFG",
            data_loader=[model.get_input() for _ in range(2)],
            forward_step=lambda model, batch: model(batch),
            num_calib_steps=1,
            num_score_steps=1,
            method="aumann_shapley",
        )
    assert len(state["damage_model"]["as_scores"]) == 1
    assert state["damage_model"]["damage_reference"]["type"] == "quantized_baseline"


def test_heterogeneous_ladders_flagged():
    """Heterogeneous ladders flagged."""
    model = _Block()
    with pytest.warns(UserWarning, match="differing candidate ladders"):
        _model, state = mtq.auto_quantize(
            model,
            constraints={"effective_bits": 12.0},
            quantization_formats=list(SEARCH_FORMATS),
            module_search_spaces=[
                {
                    "module_name_patterns": ["*mlp*"],
                    "quantization_formats": [mtq.INT8_DEFAULT_CFG],
                }
            ],
            data_loader=[model.get_input() for _ in range(2)],
            forward_step=lambda model, batch: model(batch),
            num_calib_steps=1,
            num_score_steps=1,
            method="aumann_shapley",
        )
    assert "heterogeneous_ladders" in state["damage_model"]["approximation_flags"]


def test_corner_is_anchored_even_when_attributions_are_incomplete():
    """The damage link must reproduce the measured corner regardless of attribution mass."""
    rng = np.random.default_rng(0)
    f_corner = 0.4
    for scale in (1.0, 0.1, 1e-6):  # complete, incomplete, and nearly-vanished attributions
        attributions = rng.uniform(0.001, 0.01, size=32) * scale
        mask = {"fmt": np.ones(32, dtype=bool)}
        c, b_by_key, _kappa, _inflation, converged = _anchor_ceiling(
            {"fmt": attributions}, f_corner, mask
        )
        assert converged
        corner_prediction = _predict_damage(c, float(b_by_key["fmt"].sum()))
        assert corner_prediction == pytest.approx(f_corner, rel=0.02)


def test_tiny_attributions_do_not_inflate():
    """Arbitrarily small positive attributions must not be floored into phantom damage."""
    a, b, converged = _as_seed_coverage(np.full(5000, 1e-15), c=0.4)
    assert converged
    assert float(b.sum()) < 1e-9


def test_raw_scores_survive_the_base_monotonicity_clamp(monkeypatch):
    """A negative attribution for one format must never overwrite its neighbor's positive
    one: fitting and diagnostics read the unclamped values; only solver scores are
    monotonized. Uses injected scores so the negative/positive case is deterministic."""
    injected = {"INT4_BLOCKWISE_WEIGHT_ONLY_CFG": -2.793747e-06, "INT8_DEFAULT_CFG": 2.738088e-07}
    _inject_scores_and_corner(monkeypatch, injected, corner=0.4)
    _model, state = _search(_OneLinear(), effective_bits=16.0)

    as_scores = state["damage_model"]["as_scores"]
    (name,) = next(iter(as_scores.values())).keys()
    for label, values in as_scores.items():
        expected = injected[label.split("(")[0]]
        assert values[name] == pytest.approx(expected, rel=1e-9)

    # The positive INT8 damage must reach the solver: the monotone projection may raise
    # the more aggressive neighbor but must never erase a real score.
    stat = state["candidate_stats"][name]
    int8_index = [str(r).split("(")[0] for r in stat["formats"]].index("INT8_DEFAULT_CFG")
    assert stat["scores"][int8_index] > 0
    assert state["damage_model"]["negative_attribution_mass"] > 0.5


@pytest.mark.parametrize(
    ("method", "options", "exception"),
    [
        ("aumann_shapley", [("num_path_nodes", 2)], TypeError),
        ("aumann_shapley", {"unknown_option": 1}, ValueError),
        ("aumann_shapley", {"num_score_steps": 999}, ValueError),
        ("aumann_shapley", {"num_path_nodes": True}, ValueError),
        ("aumann_shapley", {"num_path_nodes": 1.5}, ValueError),
        ("aumann_shapley", {"num_path_nodes": 0}, ValueError),
        ("aumann_shapley", {"damage_link": "unsupported"}, ValueError),
        ("aumann_shapley", {"solver": "lp"}, ValueError),
        ("aumann_shapley", {"max_predicted_damage": float("inf")}, ValueError),
        ("aumann_shapley", {"max_predicted_damage": -1.0}, ValueError),
        ("kl_div", {"num_path_nodes": 2}, ValueError),
    ],
)
def test_invalid_method_options_leave_model_untouched(method, options, exception):
    """Reject invalid method options before converting the model."""
    model = _Block()
    with pytest.raises(exception):
        _search(model, method=method, method_options=options)
    assert type(model.mlp) is torch.nn.Linear
    assert not any(isinstance(module, TensorQuantizer) for module in model.modules())


def test_forced_single_format_group_recorded_as_baseline():
    """A single-candidate allow_no_quant=False group stays quantized in every reference
    pass, so the damage reference must name it (quotes are incremental to it)."""
    model = _Block()
    _model, state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 12.0},
        quantization_formats=list(SEARCH_FORMATS),
        module_search_spaces=[
            {
                "module_name_patterns": ["*mlp*"],
                "quantization_formats": [mtq.INT8_DEFAULT_CFG],
                "allow_no_quant": False,
            }
        ],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        num_calib_steps=1,
        num_score_steps=1,
        method="aumann_shapley",
    )
    reference = state["damage_model"]["damage_reference"]
    assert reference["type"] == "quantized_baseline"
    assert any("mlp" in name for name in reference["forced_groups"])
    assert state["best"]["predicted_damage"] >= 0


def test_zero_attributions_invert_to_exact_zero():
    """Zero attributions invert to exact zero."""
    attributions = np.array([0.0, 0.02, 0.0, 0.05])
    a, b, converged = _as_seed_coverage(attributions, c=0.4)
    assert converged
    assert a[0] == a[2] == b[0] == b[2] == 0.0
    assert (a[[1, 3]] > 0).all()

    a, b, converged = _as_seed_coverage(np.zeros(4), c=0.4)
    assert converged and (a == 0).all() and (b == 0).all()


def test_scoring_signature_guards_resume(tmp_path):
    """Scoring signature guards resume."""
    checkpoint = str(tmp_path / "state.pth")
    _search(_Block(), checkpoint=checkpoint, method_options={"num_path_nodes": 2})

    # Changing what the stored scores mean must be rejected.
    with pytest.raises(ValueError, match="scoring signature"):
        _search(_Block(), checkpoint=checkpoint, method_options={"num_path_nodes": 3})

    # Changing only the solve target reuses the stored scores.
    _model, state = _search(
        _Block(),
        effective_bits=None,
        checkpoint=checkpoint,
        method_options={
            "num_path_nodes": 2,
            "max_predicted_damage": 1.0,
        },
    )
    assert state["best"]["is_satisfied"]


def _shapley_data_parallel(rank, size, baseline):

    _model, state = _search(_Block(seed=0), method_options={"num_path_nodes": 2})
    state_rank0 = DistributedProcessGroup.get_dist_syncd_obj(
        state if rank == 0 else None, DistributedProcessGroup(None), lambda a: a[0]
    )
    local = {k: v for k, v in state.items() if k != "quantizer_states"}
    rank0 = {k: v for k, v in state_rank0.items() if k != "quantizer_states"}
    assert local == rank0
    assert state["best"]["is_satisfied"]

    # Every rank scores the same batches, so correct DP reductions multiply the token count
    # by the world size while leaving all per-token quantities equal to the single-process
    # baseline; a dropped reduction shows up as a factor of the world size. Tolerances
    # absorb float32 backward jitter (amplified by the coverage inversion), nothing more.
    damage_model = state["damage_model"]
    assert damage_model["n_score_tokens"] == size * baseline["n_score_tokens"]
    assert damage_model["f_corner"] == pytest.approx(baseline["f_corner"], rel=1e-6)
    for name, scores in baseline["scores"].items():
        got = state["candidate_stats"][name]["scores"]
        assert got == pytest.approx(scores, rel=1e-3, abs=1e-9), f"{name}: {got} vs {scores}"


def test_data_parallel_aumann_shapley(skip_on_windows):
    """Data parallel aumann shapley."""
    _model, single = _search(_Block(seed=0), method_options={"num_path_nodes": 2})
    baseline = {
        "n_score_tokens": single["damage_model"]["n_score_tokens"],
        "f_corner": single["damage_model"]["f_corner"],
        "scores": {name: stat["scores"] for name, stat in single["candidate_stats"].items()},
    }
    spawn_multiprocess_job(
        2, functools.partial(_shapley_data_parallel, baseline=baseline), backend="gloo"
    )


def test_anchor_ceiling_rejects_non_finite_measurements():
    """Anchor ceiling rejects non finite measurements."""
    attributions = np.array([0.01, 0.02])
    mask = {"fmt": np.ones(2, dtype=bool)}
    for f_corner in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            _anchor_ceiling({"fmt": attributions}, f_corner, mask)
    with pytest.raises(ValueError, match="finite"):
        _anchor_ceiling({"fmt": np.array([0.01, float("nan")])}, 0.4, mask)


@pytest.mark.parametrize("corner", [float("nan"), float("inf")])
def test_non_finite_corner_invalidates_damage_model(monkeypatch, corner):
    """A non-finite corner KL must invalidate the fit (not hang) and keep scores finite."""
    _inject_scores_and_corner(
        monkeypatch,
        {"INT4_BLOCKWISE_WEIGHT_ONLY_CFG": 2e-6, "INT8_DEFAULT_CFG": 1e-7},
        corner,
    )
    _model, state = _search(_OneLinear(), effective_bits=16.0)

    damage_model = state["damage_model"]
    assert damage_model["valid"] is False
    assert "non_finite_measurements" in damage_model["approximation_flags"]
    for stat in state["candidate_stats"].values():
        assert all(math.isfinite(score) for score in stat["scores"])


def test_non_finite_attribution_excluded_and_anchor_invalidated(monkeypatch):
    """A broken candidate leaves the search space; because it was the group's most
    aggressive format, the measured corner no longer describes the pruned candidate space
    and the fit must not certify quotes against it."""
    _inject_scores_and_corner(
        monkeypatch,
        {"INT4_BLOCKWISE_WEIGHT_ONLY_CFG": float("nan"), "INT8_DEFAULT_CFG": 1e-7},
        0.4,
    )
    _model, state = _search(_OneLinear(), effective_bits=16.0)

    (name,) = state["candidate_stats"]
    stat = state["candidate_stats"][name]
    assert all("INT4_BLOCKWISE" not in str(recipe) for recipe in stat["formats"])
    # The solver objective must be the normalized additive attributions, NOT an inversion
    # anchored to the removed format's corner measurement.
    assert stat["scores"] == [pytest.approx(1e-7), 0.0]
    damage_model = state["damage_model"]
    assert "non_finite_scores_excluded" in damage_model["approximation_flags"]
    assert "corner_format_excluded" in damage_model["approximation_flags"]
    (dropped,) = damage_model["excluded_candidates"][name]
    assert "INT4_BLOCKWISE" in dropped
    assert damage_model["valid"] is False
    assert state["best"]["predicted_damage_valid"] is False


def _search_no_bf16(model, effective_bits):
    """Search where the only candidates are INT4/INT8 (no no-quant fallback)."""
    return mtq.auto_quantize(
        model,
        constraints={"effective_bits": effective_bits},
        module_search_spaces=[
            {
                "module_name_patterns": ["*"],
                "quantization_formats": list(SEARCH_FORMATS),
                "allow_no_quant": False,
            }
        ],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        num_calib_steps=2,
        num_score_steps=2,
        method="aumann_shapley",
    )


def test_pruned_singleton_group_stays_fitted_with_unquantized_reference(monkeypatch):
    """Pruning down to one candidate must not demote the group to a fixed-baseline one:
    it was unquantized during the reference pass and its survivor still needs a
    token-normalized fitted score."""
    _inject_scores_and_corner(
        monkeypatch,
        {"INT4_BLOCKWISE_WEIGHT_ONLY_CFG": float("nan"), "INT8_DEFAULT_CFG": 1e-7},
        0.4,
    )
    _model, state = _search_no_bf16(_OneLinear(), effective_bits=8.0)

    (name,) = state["candidate_stats"]
    stat = state["candidate_stats"][name]
    assert [str(recipe).split("(")[0] for recipe in stat["formats"]] == ["INT8_DEFAULT_CFG"]
    damage_model = state["damage_model"]
    assert damage_model["damage_reference"] == {"type": "unquantized"}
    (label,) = damage_model["as_scores"]
    assert damage_model["as_scores"][label] == {name: pytest.approx(1e-7)}
    assert stat["scores"] == [pytest.approx(1e-7)]
    assert "corner_format_excluded" in damage_model["approximation_flags"]
    assert damage_model["valid"] is False


def test_offline_resolve_preserves_forced_invalid_state(monkeypatch):
    """get_auto_quantize_config re-solves on a bare searcher; the forced-candidate state
    must survive the round trip so the re-solve cannot silently report a clean solution."""
    _inject_scores_and_corner(
        monkeypatch,
        {"INT4_BLOCKWISE_WEIGHT_ONLY_CFG": float("inf"), "INT8_DEFAULT_CFG": float("nan")},
        0.4,
    )
    _model, state = _search_no_bf16(_OneLinear(), effective_bits=8.0)
    assert not state["best"]["is_satisfied"]

    with pytest.warns(UserWarning, match="non-finite"):
        config = mtq.get_auto_quantize_config(state, {"effective_bits": 8.0})
    assert config["algorithm"] == "max"


def test_all_non_finite_without_no_quant_reports_unsatisfied(monkeypatch):
    """With every candidate non-finite and no no-quant fallback, the retained forced
    choice must not report success, and the failed measurement must stay visible."""
    _inject_scores_and_corner(
        monkeypatch,
        {"INT4_BLOCKWISE_WEIGHT_ONLY_CFG": float("inf"), "INT8_DEFAULT_CFG": float("nan")},
        0.4,
    )
    _model, state = _search_no_bf16(_OneLinear(), effective_bits=8.0)

    (name,) = state["candidate_stats"]
    stat = state["candidate_stats"][name]
    assert [str(recipe).split("(")[0] for recipe in stat["formats"]] == ["INT8_DEFAULT_CFG"]
    assert not math.isfinite(stat["raw_scores"][0])
    assert not state["best"]["is_satisfied"]
    damage_model = state["damage_model"]
    assert damage_model["valid"] is False
    assert "non_finite_candidate_forced" in damage_model["approximation_flags"]
    (forced_format,) = damage_model["forced_candidates"].values()
    assert "INT8_DEFAULT" in forced_format
    assert state["best"]["predicted_damage_valid"] is False


def test_infinite_candidate_never_wins_the_allocation(monkeypatch):
    """A non-finite measurement must not be zeroed into a free candidate: with finite INT4
    damage, infinite INT8 damage, and an 8-bit target, the solver must pick INT4."""
    _inject_scores_and_corner(
        monkeypatch,
        {"INT4_BLOCKWISE_WEIGHT_ONLY_CFG": 2e-6, "INT8_DEFAULT_CFG": float("inf")},
        0.4,
    )
    _model, state = _search(_OneLinear(), effective_bits=8.0)

    (name,) = state["candidate_stats"]
    assert all(
        "INT8_DEFAULT" not in str(recipe) for recipe in state["candidate_stats"][name]["formats"]
    )
    assert "INT4_BLOCKWISE" in str(state["best"]["recipe"][name])
    assert state["best"]["is_satisfied"]
    assert state["damage_model"]["valid"] is True


def test_all_candidates_non_finite_falls_back_to_no_quant(monkeypatch):
    """All candidates non finite falls back to no quant."""
    _inject_scores_and_corner(
        monkeypatch,
        {"INT4_BLOCKWISE_WEIGHT_ONLY_CFG": float("inf"), "INT8_DEFAULT_CFG": float("nan")},
        0.4,
    )
    _model, state = _search(_OneLinear(), effective_bits=6.0)

    (name,) = state["candidate_stats"]
    stat = state["candidate_stats"][name]
    assert [str(recipe).split("(")[0] for recipe in stat["formats"]] == ["NONE"]
    assert str(state["best"]["recipe"][name]).split("(")[0] == "NONE"
    assert not state["best"]["is_satisfied"]


def test_nested_score_modules_are_scored():
    """A score module nested inside another must not be zeroed by the outer replay.

    Routed experts score at ``...mlp`` while shared experts inside that same mlp score at
    themselves. The outer module's replay loop re-enters the inner forward under
    ``no_grad``; if that clears the inner's cached diffs, the shared experts silently
    score zero and the solver treats them as free to quantize.
    """

    class _Expert(torch.nn.Module):
        """Minimal expert block."""

        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(32, 32)
            self.up_proj = torch.nn.Linear(32, 32)
            self.down_proj = torch.nn.Linear(32, 32)

        def forward(self, x):
            return self.down_proj(self.gate_proj(x) * self.up_proj(x))

    class _MLP(torch.nn.Module):
        """Minimal MoE MLP container."""

        def __init__(self):
            super().__init__()
            self.experts = torch.nn.ModuleList([_Expert() for _ in range(2)])
            self.shared_experts = _Expert()

        def forward(self, hidden_states):
            out = self.shared_experts(hidden_states)
            for expert in self.experts:
                out = out + expert(hidden_states)
            return out

    class _Layer(torch.nn.Module):
        """Minimal decoder layer."""

        def __init__(self):
            super().__init__()
            self.mlp = _MLP()

        def forward(self, x):
            return self.mlp(hidden_states=x)

    class _Model(torch.nn.Module):
        """Minimal model wrapper."""

        def __init__(self):
            super().__init__()
            self.layer = _Layer()

        def forward(self, x):
            return self.layer(x)

        def get_input(self):
            return torch.randn(1, 4, 32)

    torch.manual_seed(0)
    model = _Model()
    mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=[mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        num_calib_steps=2,
        num_score_steps=2,
        method="aumann_shapley",
    )

    def _quant_score(module):
        hparam = module.get_hparam("quant_recipe")
        return max(
            hparam.get_score(recipe) for recipe in hparam.choices if "NONE" not in str(recipe)
        )

    # Assert the routing, not just that scores exist: this test kept passing when the
    # score-module rules were dropped, because positive scores say nothing about where the
    # attribution was measured. Routed experts share the MLP container; the shared expert's
    # projections score at themselves, which is what creates the nesting.
    routed = model.layer.mlp.experts[0].gate_proj.get_hparam("quant_recipe")
    shared = model.layer.mlp.shared_experts.gate_proj.get_hparam("quant_recipe")
    assert routed.score_modules == [model.layer.mlp]
    assert set(shared.score_modules) == {
        model.layer.mlp.shared_experts.gate_proj,
        model.layer.mlp.shared_experts.up_proj,
    }

    # The nested (shared-expert) group must carry real attribution, like the routed group.
    assert _quant_score(model.layer.mlp.shared_experts.gate_proj) > 0.0
    assert _quant_score(model.layer.mlp.experts[0].gate_proj) > 0.0


def test_reused_score_module_accumulates_every_invocation():
    """Used calls of a reused score module must be attributed independently."""

    class _ReusedLinear(_OneLinear):
        def forward(self, x):
            first = self.fc(x)
            second = self.fc(x)
            self.fc(torch.zeros_like(x))
            return first + second

    class _ScaledLinear(_OneLinear):
        def forward(self, x):
            return 2 * self.fc(x)

    repeated = _ReusedLinear()
    scaled = _ScaledLinear()
    scaled.load_state_dict(repeated.state_dict())
    data = [torch.randn(1, 4, 32) for _ in range(2)]

    def run(model):
        _model, state = mtq.auto_quantize(
            model,
            constraints={"effective_bits": 12.0},
            quantization_formats=[mtq.INT8_DEFAULT_CFG],
            data_loader=data,
            forward_step=lambda model, batch: model(batch),
            num_calib_steps=2,
            num_score_steps=2,
            method="aumann_shapley",
        )
        (stat,) = state["candidate_stats"].values()
        return next(
            score
            for recipe, score in zip(stat["formats"], stat["raw_scores"], strict=True)
            if not recipe.is_no_quant
        )

    assert run(repeated) == pytest.approx(run(scaled), rel=1e-5, abs=1e-8)


def test_model_specific_backward_support_is_used(monkeypatch):
    """Aumann-Shapley should use the same model-specific backward setup as gradient scoring."""
    calls = {"context": 0, "parameters": 0}

    class _TrackingContext:
        def __enter__(self):
            calls["context"] += 1

        def __exit__(self, *_exc_info):
            calls["context"] += 1

    def is_param_grad_enabled(_name, _model):
        calls["parameters"] += 1
        return True

    monkeypatch.setattr(
        AutoQuantizeAumannShapleySearcher,
        "_custom_support",
        [(lambda _model: True, lambda _model: _TrackingContext(), is_param_grad_enabled)],
    )

    _search(_OneLinear())

    assert calls["context"] == 2
    assert calls["parameters"] > 0


def test_negative_inf_candidates_do_not_leak_into_solver_scores(monkeypatch):
    """A -inf attribution must not survive candidate exclusion.

    Attributions here are signed and unclamped, so a candidate can measure -inf. The base
    searcher's running-min chain then propagates it into every less aggressive entry
    including no-quant, and a group left with no quantized candidate is dropped from the
    solver tables, so the coverage projection never rewrites it. Unlike +inf and nan, which
    collapse to 0.0 through ``min``, -inf would otherwise reach the LP objective.
    """
    _inject_scores_and_corner(
        monkeypatch,
        {
            "INT4_BLOCKWISE_WEIGHT_ONLY_CFG": float("-inf"),
            "INT8_DEFAULT_CFG": float("-inf"),
        },
        0.4,
    )
    _model, state = _search(_OneLinear(), effective_bits=6.0)

    (name,) = state["candidate_stats"]
    stat = state["candidate_stats"][name]
    assert [str(recipe).split("(")[0] for recipe in stat["formats"]] == ["NONE"]
    assert all(math.isfinite(score) for score in stat["scores"])
    assert str(state["best"]["recipe"][name]).split("(")[0] == "NONE"
    assert not state["best"]["is_satisfied"]


def test_vocab_sharded_loss_is_rejected_before_calibration(monkeypatch):
    """The unsupported-parallelism error must fire before the calibration passes run."""
    monkeypatch.setattr(
        AutoQuantizeAumannShapleySearcher, "_loss_is_vocab_sharded", lambda self: True
    )

    calibrated = []

    real_calibrate = _model_quant.calibrate

    def _spy(*args, **kwargs):
        """Record that calibration ran."""
        calibrated.append(True)
        return real_calibrate(*args, **kwargs)

    monkeypatch.setattr(_model_quant, "calibrate", _spy)

    with pytest.raises(NotImplementedError, match="vocab-sharded"):
        _search(_OneLinear(), effective_bits=6.0)
    assert not calibrated, "calibration ran before the unsupported-method check"


def test_no_quant_sorts_last_against_a_compression_tie():
    """The searcher reads formats[0] as the most aggressive candidate and treats the last
    entry as unquantized. no_quant's compression is 1.0, which a config that leaves weights
    at 16 bits ties exactly, so the ordering must pin no_quant last rather than let the
    config-JSON tiebreak decide.
    """
    no_quant = QuantRecipe(quant_cfg=None)
    # Enabling a quantizer without a cfg leaves estimate_quant_compression at 1.0.
    tied = QuantRecipe(
        {"quant_cfg": [{"quantizer_name": "*input_quantizer", "enable": True}]},
        name="TIED_16BIT",
    )
    assert tied.compression == no_quant.compression
    assert not tied.is_no_quant and no_quant.is_no_quant

    ladder = sorted([QuantRecipe("NVFP4_DEFAULT_CFG"), no_quant, tied])
    assert not ladder[0].is_no_quant, "most aggressive entry must be a quantized format"
    assert ladder[-1].is_no_quant, "no_quant must terminate the ladder"

    # Formats that compress weights are unaffected by the tiebreak.
    standard = [QuantRecipe(c) for c in ("INT8_DEFAULT_CFG", "NVFP4_DEFAULT_CFG")] + [no_quant]
    assert sorted(standard) == sorted(
        standard, key=lambda r: (r.compression, r.checkpoint_signature)
    )


def test_no_scored_tokens_still_reports_an_invalid_quote():
    """Every failure path must signal through predicted_damage, never omit it.

    With no scored tokens the fit cannot be built at all. Returning without a damage model
    would leave predicted_damage unset, so a consumer reading the documented key would get a
    KeyError rather than the nan/valid=False signal the other failure paths produce.
    """
    model = _OneLinear()
    _m, state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=list(SEARCH_FORMATS),
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda m, batch: m(batch),
        num_calib_steps=2,
        num_score_steps=0,
        method="aumann_shapley",
    )

    assert math.isnan(state["best"]["predicted_damage"])
    assert state["best"]["predicted_damage_valid"] is False
    damage_model = state["damage_model"]
    assert "no_scored_tokens" in damage_model["approximation_flags"]
    # Same key shape as a normal run, so the documented contract does not KeyError.
    assert math.isnan(damage_model["completeness"])
    for key in ("link", "f_corner", "n_score_tokens", "damage_reference", "as_scores"):
        assert key in damage_model


def test_additive_link_bound_is_certified_when_the_fit_is_valid(monkeypatch):
    """A valid additive-link fit still certifies its bound."""
    _inject_scores_and_corner(
        monkeypatch,
        {"INT4_BLOCKWISE_WEIGHT_ONLY_CFG": 0.01, "INT8_DEFAULT_CFG": 0.02},
        corner=0.4,
    )
    _model, state = _search(
        _OneLinear(),
        effective_bits=None,
        method_options={"damage_link": "additive", "max_predicted_damage": 1.0},
    )

    assert state["damage_model"]["valid"] is True
    assert state["best"]["is_satisfied"] is True
    assert not math.isnan(state["best"]["predicted_damage"])


def test_additive_link_does_not_certify_an_invalid_fit(monkeypatch):
    """An invalidated fit cannot certify a bound under EITHER link.

    The validity gate used to sit inside the coverage branch, so an additive-link search
    reported is_satisfied=True while the quote was NaN -- a self-contradictory result.
    """
    _inject_scores_and_corner(
        monkeypatch,
        {"INT4_BLOCKWISE_WEIGHT_ONLY_CFG": 0.01, "INT8_DEFAULT_CFG": 0.02},
        corner=float("inf"),
    )
    _model, state = _search(
        _OneLinear(),
        effective_bits=None,
        method_options={"damage_link": "additive", "max_predicted_damage": 1e-3},
    )

    assert state["damage_model"]["valid"] is False
    assert state["best"]["is_satisfied"] is False
    assert math.isnan(state["best"]["predicted_damage"])
