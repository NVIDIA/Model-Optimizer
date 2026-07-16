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
    assert "32x32 <- fused_qkv" in output
    assert "32x16 <- attention_out" in output
    assert "64x32 <- fused_gate_up" in output
    assert "--nks 32,32 32,16 64,32" in output
    assert "128x32" not in output  # The output head is outside this benchmark.


def test_gqa_kv_heads_are_replicated_when_tp_exceeds_kv_heads(tmp_path):
    config = _llama_config()
    model_dir = _save(tmp_path, config)
    _, model = benchmark_model._load_meta_model(str(model_dir), False, None)

    kernels, _, _, problems = benchmark_model._inspect_model(model, config, tp=4, ep=1)

    assert (24, 32, "fused_qkv") in kernels
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

    _, moe, routing, _ = benchmark_model._inspect_model(model, config, tp=2, ep=2)

    assert moe == benchmark_model._MoeShape(32, 48, 2, 2, "Swiglu")
    assert routing == benchmark_model._MoeRouting("topk")


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

    _, moe, _, _ = benchmark_model._inspect_model(model, config, tp=2, ep=2)

    assert moe == benchmark_model._MoeShape(32, 48, 2, 2, "Swiglu")


def test_nemotron_h_mamba_and_stacked_experts_are_inspected(tmp_path):
    config = _nemotron_h_config()
    model_dir = _save(tmp_path, config)
    _, model = benchmark_model._load_meta_model(str(model_dir), False, None)

    experts = next(module for name, module in model.named_modules() if name.endswith(".experts"))
    kernels, moe, routing, problems = benchmark_model._inspect_model(model, config, tp=2, ep=1)

    assert experts.up_proj.ndim == experts.down_proj.ndim == 3
    assert (42, 32, "mamba_in") in kernels
    assert (32, 16, "mamba_out") in kernels
    assert (20, 32, "up") in kernels
    assert (32, 20, "down") in kernels
    assert moe == benchmark_model._MoeShape(32, 24, 4, 2, "Relu2")
    assert problems == []
    # NemotronH declares DeepSeek-style routing fields and a score-correction
    # bias buffer on its router.
    assert routing == benchmark_model._MoeRouting("deepseek_v3", 1, 1, 1.0, True)
    command = benchmark_model._command(kernels, moe, routing, [])
    assert command[command.index("--moe_activation_type") + 1] == "Relu2"
    assert command[command.index("--moe_routing_method") + 1] == "deepseek_v3"
    assert command[command.index("--moe_num_expert_group") + 1] == "1"
    assert command[command.index("--moe_topk_group") + 1] == "1"
    assert command[command.index("--moe_routed_scaling_factor") + 1] == "1.0"
    assert "--moe_use_routing_bias" in command

    with pytest.raises(benchmark_model.ShapeError, match=r"n_groups=2.*TP=4"):
        benchmark_model._inspect_model(model, config, tp=4, ep=1)

    config.n_routed_experts = 8
    _, _, _, problems = benchmark_model._inspect_model(model, config, tp=2, ep=1)
    assert any("declares 8" in problem and "[4]" in problem for problem in problems)


def test_expert_audit_problem_is_not_masked_by_per_rank_validation(tmp_path):
    config = _nemotron_h_config()
    model_dir = _save(tmp_path, config)
    _, model = benchmark_model._load_meta_model(str(model_dir), False, None)
    config.n_routed_experts = 8

    # EP=3 does not divide the instantiated expert count; the audit mismatch
    # must still be reported instead of a masking divisibility error.
    kernels, moe, routing, problems = benchmark_model._inspect_model(model, config, tp=1, ep=3)

    assert kernels
    assert moe is None
    assert routing is None
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

    kernels, moe, routing, problems = benchmark_model._inspect_model(model, config, tp=1, ep=1)

    assert kernels == []
    assert problems == []
    assert moe == benchmark_model._MoeShape(32, 48, 4, 2, "Relu2")
    command = benchmark_model._command(kernels, moe, routing, [])
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


def test_gated_moe_activation_is_derived_or_rejected():
    assert (
        benchmark_model._moe_activation(SimpleNamespace(hidden_act="gelu_pytorch_tanh"), True)
        == "Geglu"
    )
    with pytest.raises(benchmark_model.ShapeError, match="unsupported gated MoE activation"):
        benchmark_model._moe_activation(SimpleNamespace(hidden_act="relu"), True)


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
        (42, 32, "mamba_in"),
        (32, 16, "mamba_out"),
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
    kernels, _, _, problems = benchmark_model._inspect_model(model, config, tp=1, ep=1)

    assert (48, 32, "up") in kernels
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
    assert "# 48x32 <- up" in captured.out
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

    _, moe, _, problems = benchmark_model._inspect_model(model, config, tp=1, ep=1)

    assert moe is None
    assert problems == [
        "model declares 4 routed experts but no supported expert GEMM layout was found"
    ]


def test_moe_routing_is_derived_from_config_fields():
    model = nn.Module()
    deepseek = SimpleNamespace(n_group=2, topk_group=1, routed_scaling_factor=2.5)
    renormalize = SimpleNamespace(norm_topk_prob=True)

    assert benchmark_model._moe_routing(model, deepseek) == benchmark_model._MoeRouting(
        "deepseek_v3", 2, 1, 2.5, False
    )
    assert benchmark_model._moe_routing(model, renormalize) == benchmark_model._MoeRouting(
        "renormalize"
    )
    assert benchmark_model._moe_routing(model, SimpleNamespace()) == benchmark_model._MoeRouting(
        "topk"
    )


def test_command_names_gemm_shapes_and_merges_duplicates():
    kernels = [
        (64, 32, "fused_qkv"),
        (64, 32, "fused_gate_up"),
        (32, 64, "down"),
    ]

    command = benchmark_model._command(kernels, None, None, [])

    assert command == [
        "--nks",
        "64,32",
        "32,64",
        "--nk_names",
        "fused_qkv/fused_gate_up",
        "down",
    ]


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
            "64,32",
            "32,32",
            "128,32",
            "32,64",
            "--nk_names",
            "fused_qkv",
            "attention_out",
            "fused_gate_up",
            "down",
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

    kernels, moe, routing, problems = benchmark_model._inspect_model(model, config, tp=1, ep=1)

    assert moe is None
    assert routing is None
    assert problems == []
    assert kernels == [(48, 32, "up"), (32, 48, "down")]


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
