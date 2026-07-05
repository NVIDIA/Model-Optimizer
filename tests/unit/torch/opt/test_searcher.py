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

"""Unit tests for the ``BaseSearcher`` contract in ``modelopt.torch.opt.searcher``."""

import os

import pytest
import torch
import torch.nn as nn

from modelopt.torch.opt.searcher import BaseSearcher, SearchStateDict

# -----------------------------------------------------------------------------------------------
# Minimal concrete searchers to exercise the base machinery
# -----------------------------------------------------------------------------------------------


class _TrackingSearcher(BaseSearcher):
    """Minimal searcher recording the order of the search workflow hooks."""

    @property
    def default_state_dict(self) -> SearchStateDict:
        return {"history": [], "best_score": float("-inf")}

    def reset_search(self) -> None:
        super().reset_search()
        self.call_order = ["reset_search"]

    def before_search(self) -> None:
        self.call_order.append("before_search")

    def run_search(self) -> None:
        self.call_order.append("run_search")
        # flip the training mode to check that ``search()`` restores it
        self.model.train(not self.model.training)
        self.history.append(len(self.history))
        self.best_score = 1.0
        self.best = {"best_score": self.best_score}

    def after_search(self) -> None:
        self.call_order.append("after_search")


class _CheckpointSearcher(_TrackingSearcher):
    """Searcher that saves its state after every search for resumption."""

    def after_search(self) -> None:
        super().after_search()
        self.save_search_checkpoint()


class _OtherStateSearcher(BaseSearcher):
    """Searcher with a state dict incompatible with ``_TrackingSearcher``."""

    @property
    def default_state_dict(self) -> SearchStateDict:
        return {"foo": 123}

    def run_search(self) -> None:
        pass


_SHARED_DEFAULT_STATE = {"items": [0]}


class _SharedDefaultSearcher(BaseSearcher):
    """Searcher whose default state dict is a shared mutable object."""

    @property
    def default_state_dict(self) -> SearchStateDict:
        return _SHARED_DEFAULT_STATE

    def run_search(self) -> None:
        pass


class _CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.num_forwards = 0

    def forward(self, x):
        self.num_forwards += 1
        return self.linear(x)


def _searcher_with_config(config=None) -> _TrackingSearcher:
    """Get a searcher with a manually initialized config (bypassing full ``search()``)."""
    searcher = _TrackingSearcher()
    searcher.config = searcher.sanitize_search_config(config)
    return searcher


# -----------------------------------------------------------------------------------------------
# Construction / config sanitization
# -----------------------------------------------------------------------------------------------


def test_base_searcher_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        BaseSearcher()


def test_default_search_config_keys():
    assert _TrackingSearcher().default_search_config.keys() == {
        "checkpoint",
        "verbose",
        "forward_loop",
        "data_loader",
        "collect_func",
        "max_iter_data_loader",
        "score_func",
        "loss_func",
        "deployment",
    }


def test_sanitize_search_config_none_gives_defaults():
    searcher = _TrackingSearcher()
    config = searcher.sanitize_search_config(None)
    assert config == searcher.default_search_config
    assert config["verbose"] is True  # single process is always master


def test_sanitize_search_config_merges_partial_config():
    searcher = _TrackingSearcher()
    config = searcher.sanitize_search_config({"max_iter_data_loader": 5})
    assert config["max_iter_data_loader"] == 5
    assert config["checkpoint"] is None
    assert config.keys() == searcher.default_search_config.keys()


def test_sanitize_search_config_verbose_false_is_kept():
    searcher = _TrackingSearcher()
    assert searcher.sanitize_search_config({"verbose": False})["verbose"] is False


def test_sanitize_search_config_rejects_unexpected_keys():
    searcher = _TrackingSearcher()
    with pytest.raises(AssertionError, match=r"Unexpected config keys: \{'typo_key'\}"):
        searcher.sanitize_search_config({"typo_key": 1})


# -----------------------------------------------------------------------------------------------
# Search workflow (hooks, state, training mode)
# -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("is_training", [True, False])
def test_search_runs_hooks_in_order_and_restores_train_state(is_training):
    model = nn.Linear(4, 2).train(is_training)
    searcher = _TrackingSearcher()
    constraints = {"params": 1000}

    best = searcher.search(model, constraints)

    assert searcher.call_order == ["reset_search", "before_search", "run_search", "after_search"]
    assert best == {"best_score": 1.0}
    assert searcher.model is model
    assert searcher.constraints is constraints
    assert searcher.dummy_input is None
    assert searcher.deployment is None
    assert searcher.forward_loop is None  # no data_loader / forward_loop configured
    assert model.training == is_training  # restored even though run_search flipped it


def test_search_resets_state_between_calls():
    model = nn.Linear(4, 2)
    searcher = _TrackingSearcher()
    searcher.search(model, {})
    searcher.search(model, {})
    assert searcher.history == [0]  # not [0, 1]: state is reset per search
    assert searcher.best == {"best_score": 1.0}


def test_search_constructs_forward_loop_from_config():
    searcher = _TrackingSearcher()
    searcher.search(nn.Linear(4, 2), {}, config={"forward_loop": lambda m: None})
    assert callable(searcher.forward_loop)


def test_reset_search_deepcopies_default_state():
    searcher = _SharedDefaultSearcher()
    searcher.reset_search()
    searcher.items.append(1)
    assert searcher.items == [0, 1]
    assert _SHARED_DEFAULT_STATE == {"items": [0]}  # class default must not be mutated
    assert searcher.best == {}


def test_state_dict_reflects_current_attributes():
    searcher = _TrackingSearcher()
    searcher.reset_search()
    assert searcher.state_dict() == {"history": [], "best_score": float("-inf")}
    searcher.history.append(7)
    searcher.best_score = 0.5
    assert searcher.state_dict() == {"history": [7], "best_score": 0.5}


# -----------------------------------------------------------------------------------------------
# Checkpoint path resolution
# -----------------------------------------------------------------------------------------------


def test_get_checkpoint_path_none():
    assert _searcher_with_config()._get_checkpoint_path() is None


def test_get_checkpoint_path_file_with_extension(tmp_path):
    ckpt = str(tmp_path / "search_state.pth")
    searcher = _searcher_with_config({"checkpoint": ckpt})
    assert searcher._get_checkpoint_path() == ckpt


def test_get_checkpoint_path_existing_dir_appends_rank_file(tmp_path):
    searcher = _searcher_with_config({"checkpoint": str(tmp_path)})
    assert searcher._get_checkpoint_path() == os.path.join(str(tmp_path), "rank0.pth")


def test_get_checkpoint_path_extensionless_is_treated_as_dir(tmp_path):
    ckpt = str(tmp_path / "does_not_exist_yet")
    searcher = _searcher_with_config({"checkpoint": ckpt})
    assert searcher._get_checkpoint_path() == os.path.join(ckpt, "rank0.pth")


def test_get_checkpoint_path_accepts_pathlib(tmp_path):
    searcher = _searcher_with_config({"checkpoint": tmp_path / "state.pth"})
    assert searcher._get_checkpoint_path() == str(tmp_path / "state.pth")


# -----------------------------------------------------------------------------------------------
# Checkpoint save / load
# -----------------------------------------------------------------------------------------------


def test_load_checkpoint_without_checkpoint_config():
    searcher = _searcher_with_config()
    assert searcher.load_search_checkpoint() is False


def test_load_checkpoint_missing_file_returns_false(tmp_path):
    searcher = _searcher_with_config({"checkpoint": str(tmp_path / "missing.pth")})
    searcher.reset_search()
    with pytest.warns(UserWarning, match="does not exist"):
        assert searcher.load_search_checkpoint() is False
    assert searcher.state_dict() == {"history": [], "best_score": float("-inf")}


def test_save_without_checkpoint_config_is_noop(tmp_path):
    searcher = _searcher_with_config()
    searcher.reset_search()
    searcher.save_search_checkpoint()  # must not raise nor write anything
    assert list(tmp_path.iterdir()) == []


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    ckpt = str(tmp_path / "state.pth")
    searcher = _searcher_with_config({"checkpoint": ckpt})
    searcher.reset_search()
    searcher.history = [1, 2, 3]
    searcher.best_score = 3.5
    searcher.save_search_checkpoint()
    assert os.path.isfile(ckpt)

    restored = _searcher_with_config({"checkpoint": ckpt})
    restored.reset_search()
    assert restored.load_search_checkpoint() is True
    assert restored.history == [1, 2, 3]
    assert restored.best_score == 3.5


def test_save_checkpoint_creates_missing_directories(tmp_path):
    ckpt = str(tmp_path / "nested" / "dir" / "state.pth")
    searcher = _searcher_with_config({"checkpoint": ckpt})
    searcher.reset_search()
    searcher.save_search_checkpoint()
    assert os.path.isfile(ckpt)


def test_load_checkpoint_strict_rejects_mismatched_keys(tmp_path):
    ckpt = str(tmp_path / "state.pth")
    searcher = _searcher_with_config({"checkpoint": ckpt})
    searcher.reset_search()
    searcher.save_search_checkpoint()

    other = _OtherStateSearcher()
    other.config = other.sanitize_search_config({"checkpoint": ckpt})
    other.reset_search()
    with pytest.raises(AssertionError, match="Keys in checkpoint don't match!"):
        other.load_search_checkpoint()


def test_load_checkpoint_non_strict_falls_back_to_defaults(tmp_path):
    ckpt = str(tmp_path / "state.pth")
    searcher = _searcher_with_config({"checkpoint": ckpt})
    searcher.reset_search()
    searcher.save_search_checkpoint()

    other = _OtherStateSearcher()
    other.config = other.sanitize_search_config({"checkpoint": ckpt})
    other.reset_search()
    assert other.load_search_checkpoint(strict=False) is True
    assert other.foo == 123  # missing key filled from default state dict


def test_search_resumes_from_checkpoint(tmp_path):
    ckpt = str(tmp_path / "resume.pth")
    model = nn.Linear(4, 2)
    config = {"checkpoint": ckpt}

    _CheckpointSearcher().search(model, {}, config=config)
    searcher = _CheckpointSearcher()
    searcher.search(model, {}, config=config)

    # second search resumed from the first one's saved history instead of starting fresh
    assert searcher.history == [0, 1]


# -----------------------------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------------------------


def test_has_score():
    assert not _searcher_with_config().has_score
    assert _searcher_with_config({"score_func": lambda m: 0.0}).has_score


def test_eval_score_requires_score_func():
    searcher = _searcher_with_config()
    with pytest.raises(AssertionError, match="Please provide `score_func`!"):
        searcher.eval_score()


def test_eval_score_returns_float(capsys):
    def noisy_score(model):
        print("SCORING NOISE")
        return 42  # int on purpose: eval_score must cast to float

    searcher = _searcher_with_config({"score_func": noisy_score})
    searcher.model = nn.Linear(4, 2)

    score = searcher.eval_score()  # silent by default
    assert score == 42.0
    assert isinstance(score, float)
    assert "SCORING NOISE" not in capsys.readouterr().out

    searcher.eval_score(silent=False)
    assert "SCORING NOISE" in capsys.readouterr().out


# -----------------------------------------------------------------------------------------------
# Forward loop construction
# -----------------------------------------------------------------------------------------------


def test_construct_forward_loop_trivial_case_is_none():
    assert _searcher_with_config().construct_forward_loop() is None


def test_construct_forward_loop_rejects_both_sources():
    searcher = _searcher_with_config(
        {"data_loader": [(torch.ones(1, 1),)], "forward_loop": lambda m: None}
    )
    with pytest.raises(AssertionError, match=r"Only provide `data_loader` or `forward_loop`!"):
        searcher.construct_forward_loop()


def test_construct_forward_loop_wraps_forward_loop(capsys):
    called_with = []
    searcher = _searcher_with_config({"forward_loop": called_with.append})
    model = nn.Linear(1, 1)

    loop = searcher.construct_forward_loop(silent=True)
    loop(model)
    assert called_with == [model]
    assert "Running forward loop..." in capsys.readouterr().out


def test_construct_forward_loop_with_data_loader():
    data_loader = [(torch.ones(1, 1),)] * 4
    searcher = _searcher_with_config({"data_loader": data_loader})
    model = _CountingModel()

    searcher.construct_forward_loop()(model)
    assert model.num_forwards == 4


def test_construct_forward_loop_respects_config_max_iter():
    data_loader = [(torch.ones(1, 1),)] * 4
    searcher = _searcher_with_config({"data_loader": data_loader, "max_iter_data_loader": 2})
    model = _CountingModel()

    searcher.construct_forward_loop()(model)
    assert model.num_forwards == 2


def test_construct_forward_loop_arg_overrides_config_max_iter():
    data_loader = [(torch.ones(1, 1),)] * 4
    searcher = _searcher_with_config({"data_loader": data_loader, "max_iter_data_loader": 2})
    model = _CountingModel()

    searcher.construct_forward_loop(max_iter_data_loader=1)(model)
    assert model.num_forwards == 1
