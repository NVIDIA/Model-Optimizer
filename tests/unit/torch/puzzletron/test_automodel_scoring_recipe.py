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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import UserDict
from types import SimpleNamespace

import torch

from modelopt.torch.puzzletron.dataset import DataLayout, PuzzletronBatch
from modelopt.torch.puzzletron.plugins.automodel.dummy_data import make_dummy_dataset
from modelopt.torch.puzzletron.plugins.automodel.scoring_recipe import (
    ActivationScoringRecipe,
    _ensure_packed_qkv_format,
    _HiddenStatePassthrough,
)


def test_dummy_dataset_accepts_packed_recipe_split_selector():
    dataset = make_dummy_dataset(num_samples=2, seq_length=4, split="train")

    assert len(dataset) == 2


def test_packed_canonical_payload_declares_thd_qkv_format():
    cu_seqlens = torch.tensor([0, 3, 9], dtype=torch.int32)
    payload = {"input_ids": torch.ones(1, 9, dtype=torch.long), "cu_seqlens": cu_seqlens}

    actual = _ensure_packed_qkv_format(payload)

    assert actual is payload
    assert actual["qkv_format"] == "thd"


def test_variable_length_pp_update_restores_hidden_state_output_metadata():
    class StageModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
            self.lm_head = _HiddenStatePassthrough()
            self._puzzletron_lm_head_passthrough = True

    stage = SimpleNamespace(
        submod=StageModule(),
        inputs_meta=(torch.empty(1, 8, 1024, device="meta", dtype=torch.bfloat16),),
        _outputs_meta=(torch.empty(1, 8, 1024, device="meta", dtype=torch.bfloat16),),
        is_first=False,
        is_last=True,
    )

    class FakePipeline:
        def __init__(self):
            self.info = SimpleNamespace(stages=[stage])
            self.pp_microbatch_size = 2
            self._pp_current_seq_len = 8
            self.updated = []

        def update_seq_len(self, seq_len):
            self.updated.append((seq_len, self.pp_microbatch_size, self._pp_current_seq_len))
            stage.inputs_meta = (
                torch.empty(1, seq_len, 1024, device="meta", dtype=torch.bfloat16),
            )
            # Reproduce the native metadata hook's passthrough-as-LM-head reset.
            stage._outputs_meta = (
                torch.empty(1, seq_len, 248320, device="meta", dtype=torch.bfloat16),
            )

    recipe = ActivationScoringRecipe.__new__(ActivationScoringRecipe)
    recipe.pp = FakePipeline()

    recipe._update_pp_seq_len_for_scoring(8, envelope_batch_size=1)

    assert recipe.pp.updated == [(8, 1, None)]
    assert stage._outputs_meta[0].shape == (1, 8, 1024)


def test_pp_envelope_batch_size_accounts_for_forward_microbatches():
    recipe = ActivationScoringRecipe.__new__(ActivationScoringRecipe)
    recipe.pp = SimpleNamespace(info=SimpleNamespace(schedule=SimpleNamespace(_n_microbatches=2)))

    assert recipe._pp_envelope_batch_size(local_batch_size=4) == 2


def test_pp_envelope_batch_size_rounds_up_partial_final_batch():
    recipe = ActivationScoringRecipe.__new__(ActivationScoringRecipe)
    recipe.pp = SimpleNamespace(info=SimpleNamespace(schedule=SimpleNamespace(_n_microbatches=2)))

    assert recipe._pp_envelope_batch_size(local_batch_size=3) == 2


def test_data_parallel_slice_uses_the_reduction_group_not_the_combined_ep_mesh():
    ep_only = SimpleNamespace(dp_size=4)
    dp_ep = SimpleNamespace(dp_size=8)
    dp_cp = SimpleNamespace(dp_size=2)

    assert ActivationScoringRecipe._resolve_data_parallel_slice(
        ep_only, token_size=1, token_rank=0, cp_size=1
    ) == (1, 0)
    assert ActivationScoringRecipe._resolve_data_parallel_slice(
        dp_ep, token_size=2, token_rank=1, cp_size=1
    ) == (2, 1)
    assert ActivationScoringRecipe._resolve_data_parallel_slice(
        dp_cp, token_size=4, token_rank=3, cp_size=2
    ) == (2, 1)


def test_data_parallel_slice_accepts_mapping_collator_outputs():
    recipe = ActivationScoringRecipe.__new__(ActivationScoringRecipe)
    recipe._dp_size = 2
    recipe._dp_rank = 1
    batch = UserDict(
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 3, 3]]),
            "processor_metadata": "preserved",
        }
    )

    actual = recipe._dp_slice_batch(batch)

    assert torch.equal(actual["input_ids"], torch.tensor([[3, 4]]))
    assert torch.equal(actual["image_grid_thw"], torch.tensor([[1, 3, 3]]))
    assert actual["processor_metadata"] == "preserved"


def test_canonicalize_batch_accepts_mapping_collator_outputs():
    recipe = ActivationScoringRecipe.__new__(ActivationScoringRecipe)
    recipe._data_spec = SimpleNamespace(layout=DataLayout.PADDED_VARLEN)
    recipe._data_cfg = {
        "path": "/materialized/vlm",
        "revision": "dataset-commit",
        "processor_identity": "qwen-vlm-processor",
    }
    recipe._dp_size = 1
    recipe._use_puzzletron_dataloader = True
    recipe._cp_info = lambda: (0, 1)
    collated = UserDict(
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "pixel_values": torch.randn(1, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
    )

    actual = recipe._canonicalize_batch(collated, step=7)

    assert isinstance(actual, PuzzletronBatch)
    assert actual.sample_ids == ("batch-00000007-row-0",)
    assert torch.equal(actual.model_kwargs["pixel_values"], collated["pixel_values"])
    assert actual.source_metadata["processor"] == "qwen-vlm-processor"


def test_resumed_observability_merges_local_forward_evidence_without_duplication():
    recipe = ActivationScoringRecipe.__new__(ActivationScoringRecipe)
    recipe._resumed_observability_local = {
        "vision_forward_count": 3,
        "vision_output_checksums": ["old", "shared"],
        "batch_fingerprints": ["batch-a"],
    }
    recipe._vision_monitors = [SimpleNamespace(forward_count=2, output_checksums=["shared", "new"])]
    recipe._canonical_batch_fingerprints = ["batch-a", "batch-b"]

    actual = recipe._local_observability_metadata()

    assert actual == {
        "vision_forward_count": 5,
        "vision_output_checksums": ["old", "shared", "new"],
        "batch_fingerprints": ["batch-a", "batch-b"],
    }


def test_load_balanced_cp_metric_shards_preserve_exact_token_partition():
    values = torch.arange(16).reshape(2, 8)

    rank0 = ActivationScoringRecipe._load_balanced_cp_sequence_shard(values, 0, 2)
    rank1 = ActivationScoringRecipe._load_balanced_cp_sequence_shard(values, 1, 2)

    assert rank0.shape == rank1.shape == (2, 4)
    combined = torch.cat((rank0, rank1), dim=1)
    assert torch.equal(combined.sort(dim=1).values, values)
    assert not torch.equal(rank0, values[:, :4])


def test_nonfirst_pp_stage_restores_replicated_tp_layout_before_sequence_parallel(
    monkeypatch,
):
    class DecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = torch.nn.Identity()
            self.mlp = torch.nn.Identity()

        def forward(self, x=None):
            return x

    class StageModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = DecoderLayer()

        def forward(self, hidden_states=None, inputs_embeds=None):
            value = hidden_states if hidden_states is not None else inputs_embeds
            return self.layer(value)

    module = StageModule()
    model_part = StageModule()
    recipe = ActivationScoringRecipe.__new__(ActivationScoringRecipe)
    recipe.pp = SimpleNamespace(
        info=SimpleNamespace(
            has_first_stage=True,
            stages=[SimpleNamespace(submod=module, is_first=False)],
        )
    )
    recipe.model_parts = [model_part]
    recipe.device_mesh = {"tp": "tp-mesh"}
    recipe._groups = SimpleNamespace(tp_size=2)
    recipe.cfg = SimpleNamespace(distributed=SimpleNamespace(sequence_parallel=True))
    seen = []

    def fake_replicate(hidden_states, tp_mesh):
        seen.append((hidden_states.clone(), tp_mesh))
        return hidden_states + 1

    monkeypatch.setattr(
        recipe,
        "_module_expects_sequence_parallel_input",
        lambda _module, _tp_mesh: True,
    )
    monkeypatch.setattr(recipe, "_replicate_plain_pp_input", fake_replicate)

    assert recipe._install_pp_sequence_parallel_input_restorer() == 2
    positional_result = module(torch.tensor([3.0]))
    keyword_result = model_part(inputs_embeds=torch.tensor([5.0]))
    native_layer_result = module.layer(x=torch.tensor([7.0]))

    assert seen[0][1] == "tp-mesh"
    assert seen[1][1] == "tp-mesh"
    assert torch.equal(positional_result, torch.tensor([4.0]))
    assert torch.equal(keyword_result, torch.tensor([6.0]))
    assert torch.equal(native_layer_result, torch.tensor([8.0]))


def test_nonfirst_pp_stage_keeps_plain_input_when_norm_is_not_sequence_parallel(
    monkeypatch,
):
    class DecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = torch.nn.LayerNorm(1)
            self.mlp = torch.nn.Identity()

        def forward(self, x=None):
            return self.input_layernorm(x)

    class StageModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = DecoderLayer()

        def forward(self, hidden_states=None):
            return self.layer(hidden_states)

    module = StageModule()
    recipe = ActivationScoringRecipe.__new__(ActivationScoringRecipe)
    recipe.pp = SimpleNamespace(
        info=SimpleNamespace(
            has_first_stage=False,
            stages=[SimpleNamespace(submod=module, is_first=False)],
        )
    )
    recipe.model_parts = [module]
    recipe.device_mesh = {"tp": "tp-mesh"}
    recipe._groups = SimpleNamespace(tp_size=2)
    recipe.cfg = SimpleNamespace(distributed=SimpleNamespace(sequence_parallel=True))
    seen = []

    def fake_replicate(hidden_states, tp_mesh):
        seen.append((hidden_states.clone(), tp_mesh))
        return hidden_states + 1

    monkeypatch.setattr(recipe, "_replicate_plain_pp_input", fake_replicate)

    assert recipe._install_pp_sequence_parallel_input_restorer() == 1
    result = module(hidden_states=torch.tensor([[3.0]]))

    assert seen == []
    assert torch.equal(result, torch.tensor([[0.0]]))


def test_tp_replicated_placement_is_distinct_from_fsdp_sharding():
    from torch.distributed.tensor import Replicate, Shard

    tp_mesh = SimpleNamespace(mesh_dim_names=("tp",))
    fsdp_weight = SimpleNamespace(
        device_mesh=SimpleNamespace(mesh_dim_names=("dp_shard",)),
        placements=(Shard(0),),
    )
    composable_tp_weight = SimpleNamespace(
        device_mesh=SimpleNamespace(mesh_dim_names=("dp_shard", "tp")),
        placements=(Shard(0), Replicate()),
    )

    assert not ActivationScoringRecipe._has_tp_replicated_placement(
        fsdp_weight,
        tp_mesh,
    )
    assert ActivationScoringRecipe._has_tp_replicated_placement(
        composable_tp_weight,
        tp_mesh,
    )
