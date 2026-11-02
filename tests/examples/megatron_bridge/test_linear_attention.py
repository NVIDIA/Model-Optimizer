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

import json
import runpy
import sys

import pytest
import torch
import yaml
from _test_utils.examples.megatron_example_runner import reset_megatron_global_state
from _test_utils.examples.run_command import MODELOPT_ROOT, extend_cmd_parts, run_example_command
from _test_utils.torch.transformers_models import get_tiny_tokenizer
from megatron.bridge import AutoBridge
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    FinetuningDatasetConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.post_training.checkpointing import has_modelopt_state
from megatron.core.utils import unwrap_model
from transformers import AutoModelForCausalLM, Qwen3_5ForCausalLM, Qwen3_5TextConfig

import modelopt.torch.opt as mto
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.linear_attention import linear_attention_training_phase
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils import is_quantized
from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils.plugins.mbridge import (
    load_mbridge_model_from_hf,
    load_modelopt_megatron_checkpoint,
)

sys.path.insert(0, str(MODELOPT_ROOT / "examples/llm_qat/linear_attention"))
run_training = runpy.run_path(str(MODELOPT_ROOT / "examples/llm_qat/linear_attention/train.py"))[
    "run_training"
]
sys.path.pop(0)


def _train(
    qad,
    recipe,
    hf_model,
    checkpoint=None,
    train_iters=1,
    prompt_data=None,
    teacher_serving_arithmetic=False,
):
    captured = {}

    def provider(name):
        model = AutoBridge.from_hf_pretrained(str(hf_model)).to_megatron_provider(
            load_weights=False
        )
        model.seq_length = 16
        model.calculate_per_token_loss = True
        model.gradient_accumulation_fusion = False
        model.cross_entropy_loss_fusion = False

        def capture(models):
            module = unwrap_model(models[0])
            captured[name] = module
            if name == "student":
                captured["before"] = (
                    module.decoder.layers[0].self_attention.out_proj.weight.detach().clone()
                )

                def capture_loaded(attention, args):
                    # Bridge loads checkpoint weights after the model's wrapping hooks.
                    if "loaded" not in captured:
                        captured["phase"] = (
                            attention._linear_attention_prefill_lengths,
                            attention._linear_attention_sequence_lengths,
                        )
                        captured["loaded"] = attention.out_proj.weight.detach().clone()

                module.decoder.layers[0].self_attention.register_forward_pre_hook(capture_loaded)
            elif teacher_serving_arithmetic:

                def capture_teacher_phase(attention, args):
                    captured.setdefault(
                        "teacher_phase",
                        (
                            attention._linear_attention_prefill_lengths,
                            attention._linear_attention_sequence_lengths,
                        ),
                    )

                module.decoder.layers[0].self_attention.register_forward_pre_hook(
                    capture_teacher_phase
                )
            return models

        model.register_post_wrap_hook(capture)
        return model

    config = ConfigContainer(
        model=provider("student"),
        train=TrainingConfig(train_iters=train_iters, global_batch_size=1, micro_batch_size=1),
        validation=ValidationConfig(eval_iters=0, eval_interval=1),
        optimizer=OptimizerConfig(
            optimizer="adam", lr=1e-2, min_lr=0, weight_decay=0, use_distributed_optimizer=True
        ),
        scheduler=SchedulerConfig(
            lr_decay_style="constant",
            lr_warmup_iters=0,
            start_weight_decay=0,
            end_weight_decay=0,
            use_checkpoint_opt_param_scheduler=True,
        ),
        ddp=DistributedDataParallelConfig(
            average_in_collective=False, use_distributed_optimizer=True
        ),
        dataset=MockGPTDatasetConfig(
            seq_length=16,
            random_seed=123,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            dataloader_type="single",
            num_workers=0,
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=128),
        checkpoint=CheckpointConfig(
            save=str(checkpoint) if checkpoint else None,
            load=str(checkpoint) if checkpoint else None,
            save_interval=1,
            async_save=False,
            ckpt_format="torch_dist",
        ),
        logger=LoggerConfig(log_interval=1),
        rng=RNGConfig(seed=123),
        mixed_precision="bf16_mixed",
    )
    if prompt_data:
        config.dataset = FinetuningDatasetConfig(
            seq_length=16,
            dataset_root=str(prompt_data),
            seed=123,
            dataloader_type="batch",
            do_validation=False,
            do_test=False,
            dataset_kwargs={
                "prompt_template": "{input}{output}",
                "label_key": "output",
                "answer_only_loss": True,
                "add_bos": False,
                "add_eos": True,
            },
        )
        config.tokenizer = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer", tokenizer_model=str(hf_model)
        )
    run_training(
        config,
        recipe,
        None if prompt_data else 8,
        provider("teacher") if qad else None,
        teacher_serving_arithmetic=teacher_serving_arithmetic,
    )
    return captured


def _state_logits_and_gradients(model):
    model.eval()
    tokens = torch.arange(16, device="cuda").unsqueeze(0)
    parameters = tuple(model.decoder.layers[0].self_attention.parameters())
    with linear_attention_training_phase(model, [8]), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(tokens, tokens, None).float()
        gradients = torch.autograd.grad(logits[:, 8:].square().mean(), parameters)
    return logits.detach(), gradients


@pytest.fixture(scope="module")
def linear_attention_hf_model(tmp_path_factory):
    """Build a local model shared by state and prefill training checks."""
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    hf_model = tmp_path_factory.mktemp("gdn_hf")
    tokenizer = get_tiny_tokenizer()
    tokenizer.save_pretrained(hf_model)
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        layer_types=["linear_attention"],
        max_position_embeddings=128,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    Qwen3_5ForCausalLM(config).to(torch.bfloat16).save_pretrained(hf_model)
    return hf_model


@pytest.fixture(scope="module")
def compiled_state_training(linear_attention_hf_model):
    """Compile one tiny GDN shape before timing the single-GPU training checks."""
    pytest.importorskip("vllm", exc_type=ModuleNotFoundError)
    hf_model = linear_attention_hf_model
    recipe = load_recipe("general/ptq/linear_attention_state_int8_block32_dynamic").quantize
    try:
        _train(True, recipe, hf_model)
    finally:
        reset_megatron_global_state()
    return recipe, hf_model


def test_operand_qad_teacher_uses_full_prefill(linear_attention_hf_model):
    recipe = load_recipe("general/ptq/gdn_prefill_fp8_dynamic").quantize
    try:
        captured = _train(True, recipe, linear_attention_hf_model, teacher_serving_arithmetic=True)
        assert captured["phase"] == captured["teacher_phase"] == ((16,), None)
        teacher = captured["teacher"]
        assert not any(parameter.requires_grad for parameter in teacher.parameters())
        assert not any(
            quantizer.is_enabled
            for quantizer in teacher.modules()
            if isinstance(quantizer, TensorQuantizer)
        )
        weight = captured["student"].decoder.layers[0].self_attention.out_proj.weight
        assert torch.isfinite(weight).all()
        assert not torch.equal(weight, captured["before"])
    finally:
        reset_megatron_global_state()


@pytest.mark.parametrize("qad", [False, True], ids=["qat", "qad"])
def test_state_training_checkpoint_roundtrip(compiled_state_training, qad, tmp_path, request):
    recipe, hf_model = compiled_state_training
    checkpoint = tmp_path / "checkpoints"
    captured = _train(qad, recipe, hf_model, checkpoint)
    attention = captured["student"].decoder.layers[0].self_attention
    assert attention.gdn_state_quantizer.is_enabled
    assert torch.isfinite(attention.out_proj.weight).all()
    assert not torch.equal(attention.out_proj.weight, captured["before"])
    if qad:
        teacher = captured["teacher"]
        assert not is_quantized(teacher)
        assert not any(parameter.requires_grad for parameter in teacher.parameters())
    assert has_modelopt_state(str(checkpoint))
    trained_weight = attention.out_proj.weight.detach().clone()
    policy = attention.linear_attention_config.model_copy(deep=True)
    quantizer_config = attention.gdn_state_quantizer.get_modelopt_state()
    reset_megatron_global_state()
    resumed = _train(qad, recipe, hf_model, checkpoint, train_iters=2)
    torch.testing.assert_close(resumed["loaded"], trained_weight, rtol=0, atol=0)
    attention = resumed["student"].decoder.layers[0].self_attention
    assert attention.linear_attention_config == policy
    assert attention.gdn_state_quantizer.get_modelopt_state() == quantizer_config
    assert not torch.equal(attention.out_proj.weight, trained_weight)
    assert (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip() == "2"

    # Export starts from disk, as in export_quantized_megatron_to_hf.py.
    reset_megatron_global_state()
    torch.distributed.init_process_group(
        "cpu:gloo,cuda:nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    request.addfinalizer(dist.cleanup)
    request.addfinalizer(reset_megatron_global_state)
    bridge, _, models, student, _ = load_mbridge_model_from_hf(
        hf_model_name_or_path=str(hf_model),
        provider_overrides={"gradient_accumulation_fusion": False},
        init_model_parallel=True,
        load_weights=False,
    )
    # Bridge caches the latest tracker within a process; select the newly saved iteration.
    load_modelopt_megatron_checkpoint(models, str(checkpoint / "iter_0000002"))
    restored_attention = student.decoder.layers[0].self_attention
    assert restored_attention.linear_attention_config == policy
    assert restored_attention.gdn_state_quantizer.get_modelopt_state() == quantizer_config
    torch.testing.assert_close(
        restored_attention.out_proj.weight, attention.out_proj.weight, rtol=0, atol=0
    )
    expected_weights = {name: p.detach().clone() for name, p in student.named_parameters()}
    expected_numerics = _state_logits_and_gradients(student)
    export_dir = tmp_path / "hf_export"
    bridge.hf_pretrained.save_artifacts(export_dir)
    # Shift zero-centered norm weights in FP32 so BF16 does not erase small updates.
    student.float()
    bridge.save_hf_weights([student], export_dir)
    # Keep Megatron metadata separate: Transformers has no GDN/KDA state-QAT adapter.
    metadata_path = export_dir / "megatron_modelopt_state.pt"
    torch.save(mto.modelopt_state(student), metadata_path)
    exported, loading = AutoModelForCausalLM.from_pretrained(
        export_dir, dtype=torch.float32, output_loading_info=True
    )
    assert not loading["missing_keys"] and not loading["unexpected_keys"]
    torch.testing.assert_close(
        exported.model.layers[0].linear_attn.out_proj.weight,
        attention.out_proj.weight.float().cpu(),
        rtol=0,
        atol=0,
    )
    _, _, _, reimported, _ = load_mbridge_model_from_hf(
        hf_model_name_or_path=str(export_dir),
        provider_overrides={"gradient_accumulation_fusion": False},
        init_model_parallel=False,
        load_weights=True,
    )
    mto.restore_from_modelopt_state(reimported, modelopt_state_path=metadata_path)
    reimported_attention = reimported.decoder.layers[0].self_attention
    assert reimported_attention.linear_attention_config == policy
    assert reimported_attention.gdn_state_quantizer.get_modelopt_state() == quantizer_config
    torch.testing.assert_close(
        reimported_attention.out_proj.weight, attention.out_proj.weight, rtol=0, atol=0
    )
    for name, parameter in reimported.named_parameters():
        expected = expected_weights[name]
        if name.endswith(".self_attention.out_norm.weight"):
            # HF stores gamma+1; compare the effective FP32 RMSNorm multiplier.
            parameter, expected = parameter.float() + 1, expected.float() + 1
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        _state_logits_and_gradients(reimported), expected_numerics, rtol=0, atol=0
    )


def test_prompt_metadata_drives_qad_phase(compiled_state_training, tmp_path):
    recipe, hf_model = compiled_state_training
    (tmp_path / "training.jsonl").write_text(
        json.dumps({"input": "a b c d ", "output": "e f g h"}) + "\n"
    )
    try:
        captured = _train(True, recipe, hf_model, prompt_data=tmp_path)
        prefix, valid = captured["phase"]
        assert 0 < prefix[0] < valid[0] <= 16
        attention = captured["student"].decoder.layers[0].self_attention
        assert attention.gdn_state_quantizer.is_enabled
        assert not torch.equal(attention.out_proj.weight, captured["before"])
    finally:
        reset_megatron_global_state()


@pytest.fixture(scope="module")
def replay_hf_model(linear_attention_hf_model, tmp_path_factory):
    """Replay kernels require K=128, unlike the ordinary state-QDQ fixture."""
    pytest.importorskip("vllm", exc_type=ModuleNotFoundError)
    config = Qwen3_5TextConfig.from_pretrained(linear_attention_hf_model)
    config.linear_key_head_dim = 128
    config.linear_value_head_dim = 128
    hf_model = tmp_path_factory.mktemp("gdn_replay_hf")
    get_tiny_tokenizer().save_pretrained(hf_model)
    Qwen3_5ForCausalLM(config).to(torch.bfloat16).save_pretrained(hf_model)
    return hf_model


@pytest.mark.parametrize(
    ("prompt_data", "state_quantization"),
    [(False, False), (True, False), (True, True)],
    ids=["replay-fixed", "replay-sft", "nvfp4-state-sft"],
)
def test_standard_distill_linear_attention_resume(
    replay_hf_model, tmp_path, prompt_data, state_quantization
):
    """The public QAD CLI trains and restores full-state and replay-read policies."""
    hf_model = replay_hf_model
    recipe_name = (
        "linear_attention_state_nvfp4_dynamic" if state_quantization else "gdn_replayssm_nvfp4_read"
    )
    recipe = load_recipe(f"general/ptq/{recipe_name}").model_dump(
        exclude={"metadata": {"recipe_type"}}
    )
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        "# modelopt-schema: modelopt.recipe.config.ModelOptPTQRecipe\n" + yaml.safe_dump(recipe)
    )
    output = tmp_path / "qad"
    data_args = ["--use_mock_data", "--prefill_tokens", "8"]
    if prompt_data:
        (tmp_path / "training.jsonl").write_text(
            json.dumps({"input": "a b c d ", "output": "e f g h"}) + "\n"
        )
        data_args = ["--sft", "--sft_dataset_root", str(tmp_path)]
    command = extend_cmd_parts(
        [
            "torchrun",
            "--nproc_per_node=1",
            "distill.py",
            "--no_async_save",
            "--teacher_serving_arithmetic",
            "--recompute_gdn",
            *data_args,
        ],
        student_hf_path=hf_model,
        teacher_hf_path=hf_model,
        linear_attention_recipe=recipe_path,
        output_dir=output,
        seq_length=64,
        mbs=1,
        gbs=1,
        train_iters=1,
        lr=1e-3,
        min_lr=1e-3,
        lr_warmup_iters=0,
        logit_kl_top_k=8,
        eval_iters=0,
        eval_interval=1,
        save_interval=1,
        log_interval=1,
    )
    run_example_command(command, example_path="megatron_bridge")
    checkpoint = output / "checkpoints"
    assert has_modelopt_state(str(checkpoint))
    assert (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip() == "1"
    command[command.index("--train_iters") + 1] = "2"
    run_example_command(command, example_path="megatron_bridge")
    assert (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip() == "2"
    if not prompt_data:
        # The usual PTQ -> QAD entry initializes a new run from a ModelOpt checkpoint.
        initialized = command.copy()
        initialized[initialized.index("--output_dir") + 1] = str(tmp_path / "initialized")
        initialized[initialized.index("--train_iters") + 1] = "1"
        initialized.extend(["--student_megatron_path", str(checkpoint / "iter_0000002")])
        run_example_command(initialized, example_path="megatron_bridge")
        assert has_modelopt_state(str(tmp_path / "initialized/checkpoints"))
    reset_megatron_global_state()
    try:
        _, _, models, student, _ = load_mbridge_model_from_hf(
            hf_model_name_or_path=str(hf_model),
            provider_overrides={"gradient_accumulation_fusion": False},
            init_model_parallel=True,
            load_weights=True,
        )
        before_mlp = student.decoder.layers[0].mlp.linear_fc2.weight.detach().clone()
        load_modelopt_megatron_checkpoint(models, str(checkpoint / "iter_0000002"))
        attention = student.decoder.layers[0].self_attention
        if state_quantization:
            assert attention.linear_attention_config.precision == "vllm"
            assert attention.gdn_state_quantizer.is_enabled
            assert attention.gdn_state_quantizer.is_nvfp4_dynamic
            assert not attention.gdn_read_quantizer.is_enabled
        else:
            assert attention.linear_attention_config.replay_window == 16
            assert attention.gdn_read_quantizer.is_enabled
            assert not attention.gdn_state_quantizer.is_enabled
        assert torch.isfinite(attention.out_proj.weight).all()
        # The standard QAD default trains the whole student, including its MLP.
        assert not torch.equal(student.decoder.layers[0].mlp.linear_fc2.weight, before_mlp)
    finally:
        reset_megatron_global_state()
