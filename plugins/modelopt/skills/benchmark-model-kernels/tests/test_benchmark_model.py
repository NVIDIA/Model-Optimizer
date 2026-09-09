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

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("accelerate")
transformers = pytest.importorskip("transformers")

from torch import nn

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_model.py"
SPEC = importlib.util.spec_from_file_location("benchmark_model", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark_model = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark_model
SPEC.loader.exec_module(benchmark_model)


def _save(tmp_path, config):
    config.save_pretrained(tmp_path)
    return tmp_path


def _preview(model_ref, monkeypatch, capsys, *, tp=1, ep=1):
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), str(model_ref), "--tp", str(tp), "--ep", str(ep), "--print_only"],
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    benchmark_model.main()
    return capsys.readouterr().out


def _llama_config():
    return transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )


def _nemotron_h_config(*, n_groups=2):
    return transformers.NemotronHConfig(
        vocab_size=128,
        hidden_size=32,
        layers_block_type=["mamba", "moe"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=40,
        use_mamba_kernels=False,
        ssm_state_size=4,
        mamba_num_heads=4,
        mamba_head_dim=8,
        n_groups=n_groups,
        conv_kernel=4,
        expand=1,
        n_routed_experts=4,
        n_shared_experts=1,
        moe_intermediate_size=48,
        moe_shared_expert_intermediate_size=40,
        num_experts_per_tok=2,
    )


def test_llama_meta_walk_fuses_common_projections(tmp_path, monkeypatch, capsys):
    model_dir = _save(tmp_path, _llama_config())

    output = _preview(model_dir, monkeypatch, capsys, tp=2)

    assert "layout: Transformers meta model; fused QKV and gate/up" in output
    assert (
        "32x32 <- model.layers.*.self_attn.q_proj|model.layers.*.self_attn.k_proj|model.layers.*.self_attn.v_proj"
        in output
    )
    assert "32x16 <- model.layers.*.self_attn.o_proj" in output
    assert "64x32 <- model.layers.*.mlp.gate_proj|model.layers.*.mlp.up_proj" in output
    # Same-shape kernels keep separate N,K,NAME arguments.
    assert "--nks " in output
    assert output.count("32,32,model.layers.") == 2
    assert "128x32" not in output  # The output head is outside this benchmark.


def test_gqa_kv_heads_are_replicated_when_tp_exceeds_kv_heads(tmp_path):
    config = _llama_config()
    model_dir = _save(tmp_path, config)
    _, model = benchmark_model._load_meta_model(str(model_dir), False, None)

    kernels, _, problems = benchmark_model._inspect_model(model, config, tp=4, ep=1)

    assert (
        benchmark_model._Kernel(
            24,
            32,
            "model.layers.*.self_attn.q_proj|model.layers.*.self_attn.k_proj|model.layers.*.self_attn.v_proj",
        )
        in kernels
    )
    assert problems == []


def test_meta_loader_never_materializes_model_tensors(tmp_path):
    model_dir = _save(tmp_path, _llama_config())

    _, model = benchmark_model._load_meta_model(str(model_dir / "config.json"), False, None)

    tensors = list(model.named_parameters()) + list(model.named_buffers())
    assert tensors and all(tensor.is_meta for _, tensor in tensors)


def test_revision_does_not_reach_registered_model_constructor(tmp_path):
    model_dir = _save(tmp_path, _llama_config())

    _, model = benchmark_model._load_meta_model(str(model_dir), False, "main")

    assert type(model).__name__ == "LlamaForCausalLM"


def test_mixtral_modulelist_experts_use_ep(tmp_path):
    config = transformers.MixtralConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    model_dir = _save(tmp_path, config)
    _, model = benchmark_model._load_meta_model(str(model_dir), False, None)

    _, moe, _ = benchmark_model._inspect_model(model, config, tp=2, ep=2)

    assert moe == benchmark_model._MoeShape(32, 48, 2, 2, "Swiglu")


def test_gpt_oss_direct_expert_tensors_are_inspected(tmp_path):
    config = transformers.GptOssConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    model_dir = _save(tmp_path, config)
    _, model = benchmark_model._load_meta_model(str(model_dir), False, None)

    _, moe, _ = benchmark_model._inspect_model(model, config, tp=2, ep=2)

    assert moe == benchmark_model._MoeShape(32, 48, 2, 2, "Swiglu")


def test_nemotron_h_mamba_and_stacked_experts_are_inspected(tmp_path):
    config = _nemotron_h_config()
    model_dir = _save(tmp_path, config)
    _, model = benchmark_model._load_meta_model(str(model_dir), False, None)

    experts = next(module for name, module in model.named_modules() if name.endswith(".experts"))
    kernels, moe, problems = benchmark_model._inspect_model(model, config, tp=2, ep=1)

    assert experts.up_proj.ndim == experts.down_proj.ndim == 3
    assert benchmark_model._Kernel(42, 32, "model.layers.*.mixer.in_proj") in kernels
    assert benchmark_model._Kernel(32, 16, "model.layers.*.mixer.out_proj") in kernels
    assert benchmark_model._Kernel(20, 32, "model.layers.*.mixer.shared_experts.up_proj") in kernels
    assert (
        benchmark_model._Kernel(32, 20, "model.layers.*.mixer.shared_experts.down_proj") in kernels
    )
    assert moe == benchmark_model._MoeShape(32, 24, 4, 2, "Relu2")
    assert problems == []
    command = benchmark_model._command(kernels, moe, [])
    assert command[command.index("--moe_activation_type") + 1] == "Relu2"

    with pytest.raises(benchmark_model.ShapeError, match=r"n_groups=2.*TP=4"):
        benchmark_model._inspect_model(model, config, tp=4, ep=1)

    config.n_routed_experts = 8
    _, _, problems = benchmark_model._inspect_model(model, config, tp=2, ep=1)
    assert any("declares 8" in problem and "[4]" in problem for problem in problems)


@pytest.mark.parametrize(
    ("config_cls_name", "model_cls_name"),
    [
        ("Qwen3NextConfig", "Qwen3NextForCausalLM"),
        ("Qwen3_5MoeTextConfig", "Qwen3_5MoeForCausalLM"),
    ],
)
def test_gated_delta_net_kernels_are_derived(config_cls_name, model_cls_name):
    config_cls = getattr(transformers, config_cls_name, None)
    model_cls = getattr(transformers, model_cls_name, None)
    if config_cls is None or model_cls is None:
        pytest.skip(f"transformers does not provide {config_cls_name}")
    assert config_cls is not None and model_cls is not None
    from accelerate import init_empty_weights

    config = config_cls(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        decoder_sparse_step=1,
        max_position_embeddings=64,
    )
    with init_empty_weights(include_buffers=True):
        model = model_cls(config)

    kernels, moe, problems = benchmark_model._inspect_model(model, config, tp=2, ep=1)

    # key_dim = 2 heads x 8 = 16, value_dim = 4 heads x 8 = 32; vLLM's shared
    # GDN mixer runs qkv+z and b+a as two fused per-rank GEMMs (both the
    # pre-fused Qwen3-Next and split Qwen3.5 checkpoint layouts).
    prefix = "model.layers.*.linear_attn"
    if config_cls_name == "Qwen3NextConfig":
        qkvz_label, ba_label = f"{prefix}.in_proj_qkvz", f"{prefix}.in_proj_ba"
    else:
        qkvz_label = f"{prefix}.in_proj_qkv|{prefix}.in_proj_z"
        ba_label = f"{prefix}.in_proj_b|{prefix}.in_proj_a"
    assert benchmark_model._Kernel(48, 32, qkvz_label) in kernels
    assert benchmark_model._Kernel(4, 32, ba_label) in kernels
    assert benchmark_model._Kernel(32, 16, f"{prefix}.out_proj") in kernels
    assert problems == []
    assert moe == benchmark_model._MoeShape(32, 8, 4, 2, "Swiglu")

    with pytest.raises(
        benchmark_model.ShapeError, match=r"linear_num_key_heads=2 is not divisible by 4"
    ):
        benchmark_model._inspect_model(model, config, tp=4, ep=1)


def test_expert_audit_problem_is_not_masked_by_per_rank_validation(tmp_path):
    config = _nemotron_h_config()
    model_dir = _save(tmp_path, config)
    _, model = benchmark_model._load_meta_model(str(model_dir), False, None)
    config.n_routed_experts = 8

    # EP=3 does not divide the instantiated expert count; the audit mismatch
    # must still be reported instead of a masking divisibility error.
    kernels, moe, problems = benchmark_model._inspect_model(model, config, tp=1, ep=3)

    assert kernels
    assert moe is None
    assert any("declares 8" in problem for problem in problems)


def test_moe_only_model_benchmarks_without_dense_kernels():
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(32, 48, bias=False)
            self.down_proj = nn.Linear(48, 32, bias=False)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([Expert() for _ in range(4)])

    model = nn.Module()
    model.layers = nn.ModuleList([Block()])
    config = SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        num_experts_per_tok=2,
        mlp_hidden_act="relu2",
    )

    kernels, moe, problems = benchmark_model._inspect_model(model, config, tp=1, ep=1)

    assert kernels == []
    assert problems == []
    assert moe == benchmark_model._MoeShape(32, 48, 4, 2, "Relu2")
    command = benchmark_model._command(kernels, moe, [])
    assert "--nks" not in command
    assert command[:2] == ["--moe_hidden_size", "32"]


def test_legacy_nongated_modulelist_experts_are_inspected():
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(32, 48, bias=False)
            self.down_proj = nn.Linear(48, 32, bias=False)

    model = nn.Module()
    model.experts = nn.ModuleList([Expert() for _ in range(4)])
    config = SimpleNamespace(num_experts_per_tok=2, mlp_hidden_act="relu2")

    assert benchmark_model._moe_shapes(model, config) == {
        benchmark_model._MoeShape(32, 48, 4, 2, "Relu2")
    }


def test_gate_and_up_projections_must_shard_individually():
    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(32, 6, bias=False)
            self.up_proj = nn.Linear(32, 6, bias=False)
            self.down_proj = nn.Linear(6, 32, bias=False)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Mlp()

    model = nn.Module()
    model.layers = nn.ModuleList([Block()])
    config = SimpleNamespace(hidden_size=32, num_attention_heads=4)

    # The summed width (12) divides by TP=4, but vLLM shards gate and up
    # individually, so the per-projection width (6) must divide too.
    with pytest.raises(benchmark_model.ShapeError, match=r"gate_proj=6 is not divisible by 4"):
        benchmark_model._inspect_model(model, config, tp=4, ep=1)


def test_top_k_fallback_matches_the_modelopt_list():
    auto_quantize_cost = pytest.importorskip("modelopt.torch.quantization._auto_quantize_cost")

    assert benchmark_model._MOE_TOP_K_ATTRS_FALLBACK == auto_quantize_cost._ACTIVE_MOE_TOP_K_ATTRS


def test_top_k_covers_the_modelopt_attribute_aliases():
    assert benchmark_model._top_k(SimpleNamespace(num_selected_experts=2)) == 2
    assert benchmark_model._top_k(SimpleNamespace(num_experts_per_token=4)) == 4
    assert benchmark_model._top_k(SimpleNamespace(top_k=6)) == 6
    assert benchmark_model._top_k(SimpleNamespace(num_experts_per_tok=8, top_k=50)) == 8
    assert benchmark_model._top_k(SimpleNamespace()) is None


def test_gated_moe_activation_is_derived_or_rejected():
    assert (
        benchmark_model._moe_activation(SimpleNamespace(hidden_act="gelu_pytorch_tanh"), True)
        == "Geglu"
    )
    assert benchmark_model._moe_activation(SimpleNamespace(hidden_act="gelu_new"), True) == "Geglu"
    # Exact gelu and quick_gelu are not served by vLLM's FlashInfer MoE path,
    # so they must be rejected instead of timed via the tanh-GELU kernel.
    for activation in ("gelu", "quick_gelu", "relu"):
        with pytest.raises(benchmark_model.ShapeError, match="unsupported gated MoE activation"):
            benchmark_model._moe_activation(SimpleNamespace(hidden_act=activation), True)


def test_mamba_single_group_is_replicated_across_tp():
    class Mixer(nn.Module):
        intermediate_size = 32
        num_heads = 4
        n_groups = 1
        ssm_state_size = 4

        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(32, 76, bias=False)
            self.out_proj = nn.Linear(32, 32, bias=False)

    model = nn.Module()
    model.mixer = Mixer()

    assert benchmark_model._mamba_kernels(model, tp=2) == [
        benchmark_model._Kernel(42, 32, "mixer.in_proj"),
        benchmark_model._Kernel(32, 16, "mixer.out_proj"),
    ]


def test_unrecognized_decoder_linear_is_reported():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(32, 48, bias=False)
            self.down_proj = nn.Linear(48, 32, bias=False)
            self.unknown_proj = nn.Linear(32, 48, bias=False)

    model = nn.Module()
    model.layers = nn.ModuleList([Block()])

    assert benchmark_model._unsupported_decoder_linears(model) == [
        ("layers.0.unknown_proj", 48, 32)
    ]
    config = SimpleNamespace(hidden_size=32, num_attention_heads=4)
    kernels, _, problems = benchmark_model._inspect_model(model, config, tp=1, ep=1)

    assert benchmark_model._Kernel(48, 32, "layers.*.up_proj") in kernels
    assert problems == ["unsupported decoder Linear GEMM layout(s): layers.0.unknown_proj (48x32)"]


def test_partial_inventory_is_printed_when_the_audit_fails(monkeypatch, capsys):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(32, 48, bias=False)
            self.down_proj = nn.Linear(48, 32, bias=False)
            self.unknown_proj = nn.Linear(32, 48, bias=False)

    model = nn.Module()
    model.layers = nn.ModuleList([Block()])
    config = SimpleNamespace(hidden_size=32, num_attention_heads=4, model_type="test")
    monkeypatch.setattr(benchmark_model, "_load_meta_model", lambda *_: (config, model))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "unused/model", "--print_only"])

    with pytest.raises(SystemExit, match="2"):
        benchmark_model.main()

    captured = capsys.readouterr()
    assert "# 48x32 <- layers.*.up_proj" in captured.out
    assert "# unsupported: unsupported decoder Linear GEMM layout(s)" in captured.out
    assert "unknown_proj (48x32)" in captured.out
    assert "benchmark_via_builtin.py" in captured.err


def test_declared_moe_without_supported_experts_is_reported():
    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(32, 48, bias=False)
            self.down_proj = nn.Linear(48, 32, bias=False)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Mlp()

    model = nn.Module()
    model.layers = nn.ModuleList([Block()])
    config = SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        num_local_experts=4,
        num_experts_per_tok=2,
    )

    _, moe, problems = benchmark_model._inspect_model(model, config, tp=1, ep=1)

    assert moe is None
    assert problems == [
        "model declares 4 routed experts but no supported expert GEMM layout was found"
    ]


def test_command_keeps_same_shape_kernels_as_separate_named_pairs():
    kernels = [
        benchmark_model._Kernel(64, 32, "a_proj|b_proj"),
        benchmark_model._Kernel(64, 32, "c_proj"),
        benchmark_model._Kernel(32, 64, "down_proj"),
    ]

    command = benchmark_model._command(kernels, None, [])

    assert command == [
        "--nks",
        "64,32,a_proj|b_proj",
        "64,32,c_proj",
        "32,64,down_proj",
    ]


def test_moe_name_carries_the_expert_container_path(tmp_path):
    config = _nemotron_h_config()
    model_dir = _save(tmp_path, config)
    _, model = benchmark_model._load_meta_model(str(model_dir), False, None)

    _, moe, problems = benchmark_model._inspect_model(model, config, tp=2, ep=1)

    assert problems == []
    assert moe is not None
    # The MoE row is labeled by its real container path, not a generic name,
    # and the path survives per-rank sharding.
    assert moe.name == "model.layers.*.mixer.experts"
    command = benchmark_model._command([], moe, [])
    assert command[command.index("--moe_name") + 1] == "model.layers.*.mixer.experts"
    # The path is metadata: two layouts differing only by it stay one shape.
    assert benchmark_model._MoeShape(8, 4, 2, 1, None, "a") == benchmark_model._MoeShape(
        8, 4, 2, 1, None, "b"
    )


def test_moe_sharding_interpretation_is_printed(monkeypatch, capsys):
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(32, 48, bias=False)
            self.down_proj = nn.Linear(48, 32, bias=False)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([Expert() for _ in range(6)])

    model = nn.Module()
    model.layers = nn.ModuleList([Block()])
    config = SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        num_experts_per_tok=2,
        mlp_hidden_act="relu2",
        model_type="test",
    )
    monkeypatch.setattr(benchmark_model, "_load_meta_model", lambda *_: (config, model))
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "unused/model", "--tp", "2", "--ep", "2", "--print_only"]
    )

    benchmark_model.main()

    output = capsys.readouterr().out
    assert "# MoE sharding: EP=2 partitions whole experts" in output

    # EP that is not a multiple of TP matches no modeled serving layout.
    with pytest.raises(
        benchmark_model.ShapeError, match=r"EP=2 is not a multiple of TP=4.*benchmark_via_builtin"
    ):
        benchmark_model._inspect_model(model, config, tp=4, ep=2)


def test_runner_is_invoked_in_process(tmp_path, monkeypatch):
    model_dir = _save(tmp_path, _llama_config())
    launched = []
    runner = SimpleNamespace(main=launched.append)
    monkeypatch.setattr(benchmark_model, "_load_runner", lambda: runner)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(model_dir), "--ms", "1", "16"])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    benchmark_model.main()

    assert launched == [
        [
            "--nks",
            "64,32,model.layers.*.self_attn.q_proj|model.layers.*.self_attn.k_proj"
            "|model.layers.*.self_attn.v_proj",
            "32,32,model.layers.*.self_attn.o_proj",
            "128,32,model.layers.*.mlp.gate_proj|model.layers.*.mlp.up_proj",
            "32,64,model.layers.*.mlp.down_proj",
            "--ms",
            "1",
            "16",
        ]
    ]


def test_router_gate_projection_is_not_treated_as_an_mlp():
    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(32, 48, bias=False)
            self.down_proj = nn.Linear(48, 32, bias=False)

    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(32, 4, bias=False)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Mlp()
            self.router = Router()

    model = nn.Module()
    model.layers = nn.ModuleList([Block()])
    config = SimpleNamespace(hidden_size=32, num_attention_heads=4)

    kernels, moe, problems = benchmark_model._inspect_model(model, config, tp=1, ep=1)

    assert moe is None
    assert problems == []
    assert kernels == [
        benchmark_model._Kernel(48, 32, "layers.*.mlp.up_proj"),
        benchmark_model._Kernel(32, 48, "layers.*.mlp.down_proj"),
    ]


@pytest.mark.parametrize(
    ("option", "value"), [("--nks", "1,1"), ("--moe_activation_type", "Relu2")]
)
def test_derived_shapes_cannot_be_overridden(monkeypatch, capsys, option, value):
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "unused/model", option, value, "--print_only"],
    )

    with pytest.raises(SystemExit, match="2"):
        benchmark_model.main()
    assert "cannot be overridden" in capsys.readouterr().err


@pytest.mark.parametrize(("option", "value"), [("--tp", "0"), ("--ep", "-1")])
def test_parallel_sizes_must_be_positive(monkeypatch, capsys, option, value):
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "unused/model", option, value, "--print_only"],
    )

    with pytest.raises(SystemExit, match="2"):
        benchmark_model.main()
    assert "expected a positive integer" in capsys.readouterr().err
