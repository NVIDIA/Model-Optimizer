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

from types import SimpleNamespace

import pytest
import torch
import yaml
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.utils.instantiate_utils import instantiate
from megatron.bridge.utils.yaml_utils import dump_dataclass_to_yaml
from megatron.core.transformer.module import Float16Module
from torch.distributed.checkpoint import save
from transformers import Qwen3MoeConfig

from modelopt.torch.utils.plugins.mbridge import (
    _restore_checkpoint_dtypes,
    set_moe_expert_layout,
    use_moe_grouped_gemm,
)


@pytest.mark.parametrize(
    ("architecture", "backend", "expected"),
    [
        ("Qwen3MoeForCausalLM", "modelopt", False),
        ("NemotronHForCausalLM", "modelopt", True),
        ("UnregisteredMoeForCausalLM", "modelopt", None),
        ("Qwen3MoeForCausalLM", "bridge", None),
    ],
)
def test_expert_layout_selection_preserves_native_defaults(
    tmp_path, architecture, backend, expected
):
    Qwen3MoeConfig(architectures=[architecture], num_experts=2).save_pretrained(tmp_path)
    kwargs = {} if backend == "modelopt" else {"export_backend": backend}
    layout = use_moe_grouped_gemm(str(tmp_path), **kwargs)
    assert layout is expected
    provider = HybridModelProvider(
        num_layers=2, hidden_size=64, num_attention_heads=4, moe_grouped_gemm=True
    )
    original_spec = provider.hybrid_stack_spec
    set_moe_expert_layout(provider, layout)
    if expected is None:
        assert provider.moe_grouped_gemm is True
        assert provider.hybrid_stack_spec is original_spec
    else:
        assert provider.moe_grouped_gemm is expected
    assert use_moe_grouped_gemm(str(tmp_path), force_sequential=True, **kwargs) is False


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("with_config", [False, True])
def test_saved_expert_layout_wins_over_hf_defaults(tmp_path, grouped, with_config):
    source, checkpoint = tmp_path / "source", tmp_path / "checkpoint"
    architecture = "Qwen3MoeForCausalLM" if grouped else "NemotronHForCausalLM"
    Qwen3MoeConfig(architectures=[architecture], num_experts=2).save_pretrained(source)
    key = (
        "decoder.layers.0.mlp.experts.linear_fc1.weight0"
        if grouped
        else "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight"
    )
    save({key: torch.ones(2)}, checkpoint_id=checkpoint)
    if with_config:
        (checkpoint / "run_config.yaml").write_text(
            yaml.safe_dump({"model": {"moe_grouped_gemm": grouped}})
        )
    for backend in ("modelopt", "bridge"):
        assert (
            use_moe_grouped_gemm(str(source), megatron_path=str(checkpoint), export_backend=backend)
            is grouped
        )
    if grouped:
        with pytest.raises(ValueError):
            use_moe_grouped_gemm(str(source), megatron_path=str(checkpoint), force_sequential=True)
    else:
        assert (
            use_moe_grouped_gemm(str(source), megatron_path=str(checkpoint), force_sequential=True)
            is False
        )


def test_checkpoint_storage_precision_is_restored_before_copy():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
            self.register_buffer("state", torch.ones(2, dtype=torch.float32))
            self.register_buffer("indices", torch.ones(2, dtype=torch.int32))

        def sharded_state_dict(self):
            return {
                "nested": [
                    SimpleNamespace(key="weights", data=self.weight),
                    SimpleNamespace(key="state", data=self.state),
                    SimpleNamespace(key="indices", data=self.indices),
                ]
            }

    model = Model()
    metadata = {
        name: SimpleNamespace(dtype=dtype)
        for name, dtype in {
            "weights": torch.float32,
            "state": torch.bfloat16,
            "indices": torch.int64,
        }.items()
    }
    _restore_checkpoint_dtypes([model], metadata)
    source = torch.tensor([1 + 2**-14, 1 + 2**-13])
    model.weight.data.copy_(source)
    assert torch.equal(model.weight, source)
    assert model.state.dtype == torch.bfloat16
    assert model.indices.dtype == torch.int32
    # Later mixed-precision wrapping must honor checkpoint storage, including when
    # the destination was already FP32 and no dtype conversion was needed above.
    _restore_checkpoint_dtypes([model], metadata)
    Float16Module(
        SimpleNamespace(fp16=False, bf16=True, virtual_pipeline_model_parallel_size=None), model
    )
    assert torch.equal(model.weight, source)
    assert model.state.dtype == torch.bfloat16


def _round_trip(value):
    """Serialize through the writer used for run_config.yaml, then reload."""
    node = yaml.safe_load(dump_dataclass_to_yaml({"spec": value}))["spec"]
    return node["_target_"], instantiate(node)


@pytest.mark.parametrize(
    ("moe_grouped_gemm", "expected_experts", "expected_target"),
    [
        (
            True,
            "TEGroupedMLP",
            "megatron.bridge.models.hybrid.hybrid_provider.transformer_engine_hybrid_stack_spec",
        ),
        (
            False,
            "SequentialMLP",
            "modelopt.torch.utils.plugins.megatron_layer_specs.te_hybrid_stack_spec_sequential_mlp",
        ),
    ],
)
def test_set_moe_expert_layout_survives_run_config_round_trip(
    moe_grouped_gemm, expected_experts, expected_target
):
    """A provider's stack spec must still build real submodules after a run_config round trip.

    A built ``ModuleSpec`` loses its ``MLPSubmodules`` / ``MoESubmodules`` when written to
    ``run_config.yaml``, so ``set_moe_expert_layout`` stores a factory function instead.
    """
    provider = HybridModelProvider(num_layers=2, hidden_size=64, num_attention_heads=4)
    set_moe_expert_layout(provider, moe_grouped_gemm=moe_grouped_gemm)
    assert provider.moe_grouped_gemm == moe_grouped_gemm

    assert callable(provider.hybrid_stack_spec)

    target, factory = _round_trip(provider.hybrid_stack_spec)
    # The target is an on-disk contract: renaming or moving the factory breaks saved checkpoints.
    assert target == expected_target

    provider.hybrid_stack_spec = factory
    spec = provider._resolve_hybrid_stack_spec()

    mlp = spec.submodules.mlp_layer.submodules.mlp.keywords["submodules"]
    assert mlp.linear_fc1 is not None
    assert mlp.linear_fc2 is not None

    moe = spec.submodules.moe_layer.submodules.mlp.keywords["submodules"]
    assert moe.experts is not None
    # Experts are built through a partial for the grouped-GEMM layout.
    assert getattr(moe.experts, "func", moe.experts).__name__ == expected_experts
