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

"""hf_ptq's generation previews of a model its distribution plan shards.

Each rank previews the first sample of its own calibration shard, so the ranks finish decoding at
different steps; a sharded model's previews keep every rank decoding until all have finished, and
keep ``generate`` from moving a MoE's experts off the backend quantization hooks.
"""

import contextlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from _test_utils.examples.hf_ptq_example_utils import example_utils
from _test_utils.torch.transformers_models import get_tiny_tokenizer
from PIL import Image

from modelopt.torch.utils.distributed import DistributionPlan, set_distribution_plan

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"
_CKPT = "/ckpt/model"


@pytest.fixture
def hf_ptq(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("hf_ptq")


class _PreviewModel(torch.nn.Module):
    """A model whose ``generate`` records the kwargs each preview passes it, and how often it entered
    the decode hook that Transformers' ``generate`` enters to switch a MoE's experts."""

    def __init__(self, plan=None):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False, model_type="qwen3_moe")
        self.calls = []
        self.switches = 0
        if plan is not None:
            set_distribution_plan(self, plan)

    @contextlib.contextmanager
    def _optimize_model_for_decode(self):
        self.switches += 1
        yield

    def generate(self, input_ids, **kwargs):
        self.calls.append(kwargs)
        with self._optimize_model_for_decode():
            return torch.cat([input_ids, input_ids[:, :1]], dim=1)


_PREVIEW_PLANS = pytest.mark.parametrize(
    ("plan", "lockstep"),
    [
        (None, False),
        (DistributionPlan(4), False),
        (DistributionPlan(4, ep_size=4), True),
        (DistributionPlan(4, dense="fsdp"), True),
        (DistributionPlan(1, dense="fsdp"), False),
    ],
    ids=["single_process", "replicated", "expert_parallel", "fsdp2", "one_rank_fsdp2"],
)


@_PREVIEW_PLANS
def test_a_sharded_model_previews_in_lockstep(hf_ptq, monkeypatch, plan, lockstep):
    """Each rank previews its own calibration shard's first sample, so the ranks finish decoding at
    different steps. A sharded model runs a collective in every forward, and a rank that stopped
    early left the others blocked in it: Qwen3-30B-A3B NVFP4 at EP4 hung in the post-PTQ preview.
    Both previews keep every rank of a sharded model decoding until all have finished."""
    model = _PreviewModel(plan)
    loader = [{"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3)}]
    args = SimpleNamespace(
        specdec_offline_dataset=None,
        skip_generate=False,
        verbose=False,
        local_checkpoint_path=_CKPT,
        trust_remote_code=False,
    )
    monkeypatch.setattr(hf_ptq, "export_quantized", lambda *args, **kwargs: None)

    ids, mask, before = hf_ptq.pre_quantize(args, model, "qwen3_moe", None, loader, False)
    hf_ptq.post_quantize(
        args,
        model,
        model,
        "qwen3_moe",
        get_tiny_tokenizer(),
        None,
        ids,
        mask,
        before,
        False,
        None,
        loader,
    )

    synced = [call.get("synced_gpus", False) for call in model.calls]
    assert synced == [lockstep, lockstep]  # before and after quantization
    assert model.switches == 0  # neither preview lets generate move the experts off their backend


@_PREVIEW_PLANS
def test_a_sharded_nemotron_vl_model_previews_in_lockstep(monkeypatch, plan, lockstep):
    """The Nemotron-VL preview's text, fallback and image generations keep a sharded model's ranks
    decoding together too."""
    recorded = {}
    keeps = lambda model: "_optimize_model_for_decode" in vars(model)  # noqa: E731
    vlm_utils = SimpleNamespace(
        run_text_only_generation=lambda model, tokenizer, question, config, *args: recorded.update(
            text=config, text_keeps_the_experts=keeps(model)
        ),
        run_vl_preview_generation=lambda model, tokenizer, path, stage, trust, kwargs: (
            recorded.update(image=kwargs, image_keeps_the_experts=keeps(model))
        ),
    )
    monkeypatch.setitem(sys.modules, "vlm_utils", vlm_utils)
    model = _PreviewModel(plan)

    example_utils.run_nemotron_vl_preview(
        model, get_tiny_tokenizer(), torch.tensor([[1, 2, 3]]), _CKPT, "stage", allow_fallback=True
    )

    expected = {"synced_gpus": True} if lockstep else {}
    assert recorded["text"].get("synced_gpus", False) == lockstep
    assert recorded["image"] == expected
    assert recorded["text_keeps_the_experts"] and recorded["image_keeps_the_experts"]
    assert "_optimize_model_for_decode" not in vars(model)
    assert model.calls[0].get("synced_gpus", False) == lockstep  # the fallback generate


@pytest.mark.parametrize(("lockstep", "raises"), [({}, False), ({"synced_gpus": True}, True)])
def test_a_failed_text_only_vl_preview_stops_a_sharded_run(monkeypatch, lockstep, raises):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    vlm_utils = importlib.import_module("vlm_utils")

    class _ChatModel:
        def chat(self, *args, **kwargs):
            raise RuntimeError("the preview failed on this rank")

    run = lambda: vlm_utils.run_text_only_generation(  # noqa: E731
        _ChatModel(), None, "question", dict(lockstep), _CKPT
    )

    if raises:
        with pytest.raises(RuntimeError, match="failed on this rank"):
            run()
    else:
        assert run() is None


@pytest.mark.parametrize("generation_kwargs", [None, {"synced_gpus": True}])
def test_the_vl_image_preview_generates_with_the_kwargs_it_is_handed(
    monkeypatch, tmp_path, generation_kwargs
):
    """run_vl_preview_generation hands the kwargs it is given to the model's chat(), in its config."""
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    vlm_utils = importlib.import_module("vlm_utils")
    (tmp_path / "images").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "images" / "example1a.jpeg")
    monkeypatch.setattr(vlm_utils, "__file__", str(tmp_path / "vlm_utils.py"))
    monkeypatch.setattr(
        vlm_utils.AutoImageProcessor, "from_pretrained", lambda *args, **kwargs: lambda images: {}
    )
    received = []

    class _ChatModel:
        device = torch.device("cpu")

        def chat(self, tokenizer, question, generation_config, **image_features):
            received.append(generation_config)
            return "a picture"

    vlm_utils.run_vl_preview_generation(
        _ChatModel(), SimpleNamespace(eos_token_id=0), _CKPT, "stage", False, generation_kwargs
    )

    assert len(received) == 1
    assert received[0].get("synced_gpus", False) == (generation_kwargs is not None)


@pytest.mark.parametrize("has_image", [True, False], ids=["a_peer_has_none", "no_rank_has_one"])
def test_the_ranks_of_a_sharded_model_skip_the_image_preview_together(
    monkeypatch, tmp_path, has_image
):
    """Each node reads the sample images from its own checkout, which may have none: the repository
    ships none. A rank that previewed alone would wait for its peers in lockstep, so a rank with an
    image skips when a peer has none, and when no rank has one, they all skip without stopping."""
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    vlm_utils = importlib.import_module("vlm_utils")
    if has_image:
        (tmp_path / "images").mkdir()
        Image.new("RGB", (4, 4)).save(tmp_path / "images" / "example1a.jpeg")
    monkeypatch.setattr(vlm_utils, "__file__", str(tmp_path / "vlm_utils.py"))
    monkeypatch.setattr(
        vlm_utils.AutoImageProcessor, "from_pretrained", lambda *args, **kwargs: lambda images: {}
    )
    # This rank's answer, and a peer's that found no image.
    monkeypatch.setattr(vlm_utils, "all_reduce", lambda label, fn, combine: combine([fn(), False]))

    class _ChatModel:
        device = torch.device("cpu")

        def chat(self, *args, **kwargs):
            pytest.fail("this rank previewed without its peers")

    preview = vlm_utils.run_vl_preview_generation(
        _ChatModel(), SimpleNamespace(eos_token_id=0), _CKPT, "stage", False, {"synced_gpus": True}
    )

    assert preview is None
