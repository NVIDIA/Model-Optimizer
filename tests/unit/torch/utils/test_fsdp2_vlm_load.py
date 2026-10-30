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

"""``parallel_load_and_prepare_fsdp2`` on a VLM checkpoint in the published (pre-v5) key layout.

Transformers registers the renamings for that layout (``language_model.model.*`` ->
``model.language_model.*``, ...) on the top-level class, so they must be looked up before
``fully_shard`` swaps the root's class. A 1-rank gloo group on CPU is enough: the swap happens at
any world size.
"""

import json

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("accelerate")
pytest.importorskip("transformers")

from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

from modelopt.torch.utils.plugins.model_load_utils import parallel_load_and_prepare_fsdp2

# Current Llava parameter prefix -> the prefix the published llava-hf checkpoints store.
_PUBLISHED_LAYOUT = {
    "model.language_model.": "language_model.model.",
    "model.vision_tower.": "vision_tower.",
    "model.multi_modal_projector.": "multi_modal_projector.",
    "lm_head.": "language_model.lm_head.",
}


def _tiny_llava_checkpoint(path, drop=()):
    text = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    vision = CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
    )
    config = LlavaConfig(text_config=text, vision_config=vision, image_token_index=63)
    config.architectures = ["LlavaForConditionalGeneration"]
    config.dtype = torch.float32  # the meta build defaults to bfloat16 otherwise
    torch.manual_seed(0)
    model = LlavaForConditionalGeneration(config).eval()

    def published(name):
        for new, old in _PUBLISHED_LAYOUT.items():
            if name.startswith(new):
                return old + name[len(new) :]
        return name

    state = {
        published(n): t.clone().contiguous()
        for n, t in model.state_dict().items()
        if not any(n.startswith(d) for d in drop)
    }
    save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})
    (path / "config.json").write_text(json.dumps(config.to_dict()))
    return model


@pytest.fixture
def gloo_group(tmp_path):
    """1-rank group, kept for the whole test: gathering a loaded DTensor needs it."""
    owns_group = not torch.distributed.is_initialized()
    if owns_group:
        torch.distributed.init_process_group(
            "gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'group'}"
        )
    yield
    if owns_group:
        torch.distributed.destroy_process_group()


def _fsdp2_load(path):
    return parallel_load_and_prepare_fsdp2(
        str(path), torch.device("cpu"), 0, 1, attn_implementation="eager"
    )


def test_published_layout_vlm_loads_every_weight(tmp_path, gloo_group):
    ref = _tiny_llava_checkpoint(tmp_path)
    model = _fsdp2_load(tmp_path)

    ref_params = dict(ref.named_parameters())
    for name, param in model.named_parameters():
        loaded = param.full_tensor() if isinstance(param, DTensor) else param
        assert torch.equal(loaded.detach(), ref_params[name].detach()), name

    # Vision tower and projector are their own FSDP2 units, not part of the root's.
    assert isinstance(model.model.vision_tower, FSDPModule)
    assert isinstance(model.model.multi_modal_projector, FSDPModule)
    # The export reverses exactly the renamings the load ran, as after from_pretrained.
    assert model._weight_conversions, "renamings the published layout needs were not recorded"


def test_unsourced_parameter_raises(tmp_path, gloo_group):
    _tiny_llava_checkpoint(tmp_path, drop=("model.multi_modal_projector.",))
    with pytest.raises(RuntimeError, match="get no tensor from the checkpoint"):
        _fsdp2_load(tmp_path)


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="with a GPU, FSDP2 gathers the CPU-loaded root onto it"
)
def test_nemotron_vl_dummy_forward_gathers_the_root(tmp_path, gloo_group):
    """The Nemotron-VL export forward calls the language model submodule, which skips the FSDP2
    root's pre-forward unshard; the root's own parameters (embed_tokens) must be gathered for it."""
    from modelopt.torch.export.unified_export_hf import _llm_dummy_forward

    _tiny_llava_checkpoint(tmp_path)
    model = _fsdp2_load(tmp_path)
    model.config.architectures = ["NemotronTestVL"]  # takes the Nemotron-VL branch
    _llm_dummy_forward(model)
    assert all(isinstance(p, DTensor) for p in model.model.language_model.embed_tokens.parameters())
