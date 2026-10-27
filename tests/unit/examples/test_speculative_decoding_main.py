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

"""``setup_external_draft`` wiring in the speculative-decoding entrypoint.

Lives under ``tests/unit`` rather than ``tests/examples/`` so CI runs it on every
change: the example lanes cover a fixed allowlist.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

import modelopt.torch.speculative as mtsp
from modelopt.recipe.config import ModelOptExternalDraftRecipe

_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "speculative_decoding"

VOCAB = 64
HIDDEN = 32


@pytest.fixture(scope="module")
def main_mod():
    """Import the entrypoint by path; its sibling modules need the example dir on sys.path."""
    sys.path.insert(0, str(_EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location("sd_main", _EXAMPLE_DIR / "main.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as e:  # optional example-only deps
        pytest.skip(f"speculative_decoding/main.py not importable: {e}")
    finally:
        sys.path.pop(0)
    return module


def _tiny_causal_lm(vocab_size=VOCAB, hidden=HIDDEN):
    cfg = transformers.LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return transformers.LlamaForCausalLM(cfg)


class _FakeTok:
    """Enough of a tokenizer for the vocabulary checks; both sides share one vocab."""

    def __init__(self, vocab=None):
        self._vocab = {chr(i): i for i in range(VOCAB)} if vocab is None else dict(vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def __len__(self):
        return len(self._vocab)


def _recipe(**data):
    return ModelOptExternalDraftRecipe.model_validate(
        {
            "metadata": {"description": "d"},
            "model": {"model_name_or_path": "base"},
            "draft_model_name_or_path": "draft",
            "data": data,
        }
    )


@pytest.fixture
def stub_tokenizers(monkeypatch, main_mod):
    monkeypatch.setattr(
        main_mod.transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _FakeTok()),
    )


def _training_args():
    return types.SimpleNamespace(training_seq_len=128, device=torch.device("cpu"), cp_size=1)


def test_resume_of_a_converted_checkpoint_does_not_reconvert(main_mod, stub_tokenizers, tmp_path):
    """A checkpoint this mode writes carries modelopt_state, so it restores already converted.

    Converting it a second time raises ``TypeError: Cannot create a consistent method
    resolution order``, so any HF-format resume died at startup.
    """
    draft = _tiny_causal_lm()
    mtsp.convert(draft, [("external", {})])
    draft.save_pretrained(tmp_path / "ckpt")

    restored = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "ckpt")
    assert main_mod.ModeloptStateManager.is_converted(restored), (
        "checkpoint did not restore as converted; this test no longer covers the bug"
    )

    base = _tiny_causal_lm(hidden=2 * HIDDEN)
    out, _ = main_mod.setup_external_draft(
        _recipe(offline_data_path="/tmp/hidden"),
        _training_args(),
        base_model=base,
        draft=restored,
    )
    assert out is restored
    # The head is not part of the draft's own state, so a resume must re-attach it.
    assert out._base_lm_head is not None


def test_exported_architecture_survives_a_resume(main_mod, stub_tokenizers, tmp_path):
    """A second convert would nest the dynamic class and export it as ``architectures``."""
    draft = _tiny_causal_lm()
    mtsp.convert(draft, [("external", {})])
    draft.save_pretrained(tmp_path / "ckpt")
    restored = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "ckpt")

    main_mod.setup_external_draft(
        _recipe(offline_data_path="/tmp/hidden"),
        _training_args(),
        base_model=_tiny_causal_lm(hidden=2 * HIDDEN),
        draft=restored,
    )
    restored.save_pretrained(tmp_path / "out")
    config = transformers.AutoConfig.from_pretrained(tmp_path / "out")
    assert config.architectures == ["LlamaForCausalLM"], (
        f"draft exported as {config.architectures}; serving frameworks key off this name"
    )


def test_dry_run_is_exempt_from_the_data_source_requirement(main_mod, stub_tokenizers):
    """The shipped recipe carries no data path, so it derives mode="online".

    --dry_run only validates the recipe and writes an untrained checkpoint, so it must
    not require a dataset -- matching the data_path check in train().
    """
    recipe = _recipe()
    assert recipe.data.mode == "online"
    base = _tiny_causal_lm(hidden=2 * HIDDEN)

    with pytest.raises(ValueError, match="no online path"):
        main_mod.setup_external_draft(
            recipe, _training_args(), base_model=base, draft=_tiny_causal_lm()
        )

    draft, _ = main_mod.setup_external_draft(
        recipe, _training_args(), base_model=base, draft=_tiny_causal_lm(), dry_run=True
    )
    assert draft is not None


def test_resume_honours_an_edited_external_section(main_mod, stub_tokenizers, tmp_path):
    """An edited `external:` section must take effect on resume, not be silently dropped.

    The restore replays the config saved in the checkpoint, so skipping convert also
    skips modify(). Resuming with a changed objective would then train the old one while
    the printed recipe reports the new -- a plausible loss curve and the wrong draft.
    """
    draft = _tiny_causal_lm()
    mtsp.convert(draft, [("external", {"external_loss": "soft_ce"})])
    draft.save_pretrained(tmp_path / "ckpt")
    restored = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "ckpt")
    assert restored.external_loss == "soft_ce", "checkpoint did not carry the objective"

    recipe = _recipe(offline_data_path="/tmp/hidden")
    recipe.external.external_loss = "tvd_deploy"
    out, _ = main_mod.setup_external_draft(
        recipe,
        _training_args(),
        base_model=_tiny_causal_lm(hidden=2 * HIDDEN),
        draft=restored,
    )
    assert out.external_loss == "tvd_deploy", (
        f"resume trained {out.external_loss!r} while the recipe asked for 'tvd_deploy'"
    )

    # modify() writes only the live attributes, so the stored state has to be updated too:
    # otherwise the next checkpoint records the objective the draft was never trained under
    # and a restore outside main.py hands that back.
    out.save_pretrained(tmp_path / "out")
    again = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "out")
    assert again.external_loss == "tvd_deploy", (
        f"checkpoint recorded {again.external_loss!r}; the draft was trained with 'tvd_deploy'"
    )


def test_context_parallelism_is_rejected(main_mod, stub_tokenizers):
    """Nothing in this mode shards the sequence, so cp_size > 1 must refuse, not run.

    main.py sets model.eagle_cp_size unconditionally further down, so without this the
    draft trains with CP configured and silently ignored.
    """
    args = _training_args()
    args.cp_size = 2
    with pytest.raises(ValueError, match="context parallelism"):
        main_mod.setup_external_draft(
            _recipe(offline_data_path="/tmp/hidden"),
            args,
            base_model=_tiny_causal_lm(hidden=2 * HIDDEN),
            draft=_tiny_causal_lm(),
        )


def test_caller_supplied_base_tokenizer_is_used(main_mod, monkeypatch):
    """The fresh path already holds the base tokenizer; it must not be re-fetched."""
    calls = []

    def _from_pretrained(cls, *a, **k):
        calls.append(a[0] if a else None)
        return _FakeTok()

    monkeypatch.setattr(
        main_mod.transformers.AutoTokenizer, "from_pretrained", classmethod(_from_pretrained)
    )
    main_mod.setup_external_draft(
        _recipe(offline_data_path="/tmp/hidden"),
        _training_args(),
        base_model=_tiny_causal_lm(hidden=2 * HIDDEN),
        draft=_tiny_causal_lm(),
        base_tokenizer=_FakeTok(),
    )
    assert calls == ["draft"], f"base tokenizer re-loaded: {calls}"


# None: a fresh draft; otherwise the swap flag the checkpoint was trained with
@pytest.mark.parametrize("saved_swap", [None, False, True])
def test_vocab_swap_skips_only_an_already_swapped_draft(main_mod, monkeypatch, saved_swap):
    """Swapping a base-indexed draft again scrambles its trained rows; equal widths hide it."""
    draft_tok = _FakeTok({chr(i): VOCAB - 1 - i for i in range(VOCAB)})  # reversed ids
    monkeypatch.setattr(
        main_mod.transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, name, *a, **k: draft_tok if name == "draft" else _FakeTok()),
    )
    recipe = _recipe(offline_data_path="/tmp/hidden")
    recipe.external.external_vocab_swap = True
    draft = _tiny_causal_lm()
    if saved_swap is not None:
        mtsp.convert(draft, [("external", {"external_vocab_swap": saved_swap})])
    before = draft.get_input_embeddings().weight.data.clone()

    main_mod.setup_external_draft(
        recipe, _training_args(), base_model=_tiny_causal_lm(hidden=2 * HIDDEN), draft=draft
    )
    after = draft.get_input_embeddings().weight.data
    assert torch.equal(after, before if saved_swap else before.flip(0))


def test_resuming_a_different_speculative_mode_is_refused(main_mod, stub_tokenizers, tmp_path):
    """A checkpoint from another speculative mode must fail with a clear message.

    The skip-convert guard only asks whether *some* speculative mode is present, so without
    this an eagle/dflash checkpoint would skip the convert, never become an external draft,
    and die later on an unrelated attribute.
    """
    draft = _tiny_causal_lm()
    mtsp.convert(draft, [("eagle", {})])
    draft.save_pretrained(tmp_path / "ckpt")
    restored = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "ckpt")

    with pytest.raises(ValueError, match="not 'external'"):
        main_mod.setup_external_draft(
            _recipe(offline_data_path="/tmp/hidden"),
            _training_args(),
            base_model=_tiny_causal_lm(hidden=2 * HIDDEN),
            draft=restored,
        )


def test_sparse_requires_a_tvd_objective(main_mod, stub_tokenizers):
    """soft_ce needs full teacher logits; a sparse dump only stores the truncated policy.

    Without the gate the run trains against a zero-padded distribution and the loss curve
    looks ordinary, so the refusal has to happen before training starts.
    """
    recipe = _recipe(sparse_data_path="/tmp/policy")
    recipe.external.external_loss = "soft_ce"
    with pytest.raises(ValueError, match="external_loss"):
        main_mod.setup_external_draft(
            recipe,
            _training_args(),
            base_model=_tiny_causal_lm(hidden=2 * HIDDEN),
            draft=_tiny_causal_lm(),
        )


def test_sparse_path_skips_the_base_head_attach(main_mod, stub_tokenizers):
    """The sparse dump carries the teacher's policy, so the base lm_head is never needed."""
    draft, _ = main_mod.setup_external_draft(
        _recipe(sparse_data_path="/tmp/policy"),
        _training_args(),
        base_model=_tiny_causal_lm(hidden=2 * HIDDEN),
        draft=_tiny_causal_lm(),
    )
    assert getattr(draft, "_base_lm_head", None) is None, (
        "the sparse path attached the base head it does not use"
    )
