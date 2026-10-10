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

import runpy

import pytest
import torch
from _test_utils.examples.megatron_example_runner import reset_megatron_global_state
from _test_utils.examples.run_command import MODELOPT_ROOT
from _test_utils.torch.transformers_models import get_tiny_tokenizer
from megatron.bridge import AutoBridge
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
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
from modelopt.torch.quantization.utils import is_quantized
from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils.plugins.mbridge import (
    load_mbridge_model_from_hf,
    load_modelopt_megatron_checkpoint,
)

run_training = runpy.run_path(str(MODELOPT_ROOT / "examples/llm_qat/linear_attention/train.py"))[
    "run_training"
]


def _train(qad, recipe, hf_model, checkpoint=None, train_iters=1):
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
                        captured["loaded"] = attention.out_proj.weight.detach().clone()

                module.decoder.layers[0].self_attention.register_forward_pre_hook(capture_loaded)
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
    run_training(config, recipe, 8, provider("teacher") if qad else None)
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
def compiled_state_training(tmp_path_factory):
    """Compile one tiny GDN shape before timing the single-GPU training checks."""
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    pytest.importorskip("vllm", exc_type=ModuleNotFoundError)
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
    recipe = load_recipe("general/ptq/linear_attention_state_int8_block32_dynamic").quantize
    try:
        _train(True, recipe, hf_model)
    finally:
        reset_megatron_global_state()
    return recipe, hf_model


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
    dist.setup()
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
