# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-rank differential for the Qwen MR210 PDD recipe under FSDP2."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import shutil
import sys
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.func import functional_call

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
for path in (_REPO_ROOT, _REPO_ROOT / "tests", _FASTGEN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _test_utils.torch.diffusers_models import create_tiny_qwen_image_pipeline_dir
from diffusers import QwenImageTransformer2DModel
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from pdd.recipe import (
    build_pdd_setup,
    build_pdd_training_artifacts,
    initialize_pdd_distributed,
    resolve_pdd_recipe_config,
)
from pdd.training import PDDValidationAssignment, PreparedPDDBatch, run_pdd_validation
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
)

from modelopt.torch.fastgen import PDDPipeline
from modelopt.torch.fastgen.plugins.qwen_image import build_img_shapes, pack_latents
from modelopt.torch.fastgen.plugins.qwen_image_pdd import (
    QwenImagePDDAdapter,
    adopt_qwen_image_mr210_forward,
    convert_qwen_image_to_pdd,
)


class _FSDPBoundaryReferenceAdapter(QwenImagePDDAdapter):
    """Emulate FSDP's BF16 parameter views backed by FP32 masters."""

    def _call_packed(
        self,
        model,
        state,
        time,
        condition,
        model_kwargs,
        *,
        condition_name,
    ):
        encoder_hidden_states, attention_mask = self._prepare_call(
            model,
            state,
            time,
            condition,
            model_kwargs,
            condition_name=condition_name,
        )
        batch_size, _, height, width = state.shape
        max_txt_seq_len = int(attention_mask.sum(dim=1).max().to(torch.int32).item())
        parameters = {
            name: parameter.to(torch.bfloat16) if parameter.dtype.is_floating_point else parameter
            for name, parameter in model.named_parameters()
        }
        output = functional_call(
            model,
            parameters,
            (),
            {
                "hidden_states": pack_latents(state).to(torch.bfloat16),
                "timestep": time,
                "encoder_hidden_states": encoder_hidden_states.to(torch.bfloat16),
                "encoder_hidden_states_mask": attention_mask,
                "img_shapes": build_img_shapes(batch_size, height, width),
                "max_txt_seq_len": max_txt_seq_len,
                "return_dict": False,
                **model_kwargs,
            },
            strict=False,
        )
        return self._extract_packed_output(output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-checkpointing", action="store_true")
    return parser.parse_args()


def _raw_config(
    model_dir: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    *,
    activation_checkpointing: bool,
) -> dict:
    return {
        "model": {
            "pretrained_model_name_or_path": str(model_dir),
            "torch_dtype": "bfloat16",
            "device": "cuda",
            "transformer_engine_linear": False,
            "peft": None,
            "guidance_embeds": False,
            "fuse_qkv_projections": False,
        },
        "pdd": {
            "pred_type": "flow",
            "num_train_timesteps": None,
            "guidance_scale": 4.0,
            "student_sample_steps": 2,
            "student_sample_type": "ode",
            "grid_size": 4,
            "grid_max_t": 0.999,
            "flow_shift": 5.0,
            "block_size_min": 1,
            "block_size_max": 4,
            "teacher_integrator": "euler",
            "inference_blocks": [2, 2],
            "data_free": False,
        },
        "seed": 42,
        "optim": {
            "learning_rate": 2.0e-5,
            "optimizer": {"_target_": "torch.optim.AdamW", "weight_decay": 0.0},
        },
        "lr_scheduler": {
            "lr_decay_style": "constant",
            "lr_warmup_steps": 0,
            "min_lr": 2.0e-5,
        },
        "step_scheduler": {
            "max_steps": 1,
            "num_epochs": 1,
            "log_every": 1,
            "ckpt_every_steps": 1,
            "local_batch_size": 1,
            "global_batch_size": 2,
            "save_checkpoint_every_epoch": False,
        },
        "training_health": {"max_grad_norm": 1.0, "zero_grad_warmup_steps": 0},
        "validation": {"count": 1, "seed": 11, "split_seed": 7, "every_steps": 1},
        "data": {
            "dataloader": {
                "_target_": "fastgen_data.build_text_to_image_multiresolution_dataloader",
                "batch_size": 1,
                "drop_last": True,
                "shuffle": True,
                "dynamic_batch_size": False,
            }
        },
        "fsdp": {
            "dp_size": 2,
            "tp_size": 1,
            "cp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "activation_checkpointing": activation_checkpointing,
        },
        "checkpoint": {
            "enabled": True,
            "checkpoint_dir": str(checkpoint_dir),
            "model_save_format": "torch_save",
            "save_consolidated": False,
        },
    }


def _full_fp32_gradient(parameter: torch.nn.Parameter) -> torch.Tensor:
    gradient = parameter.grad
    if gradient is None:
        raise RuntimeError("expected a materialized gradient")
    if gradient.dtype != torch.float32:
        raise RuntimeError(f"expected an FP32 gradient, got {gradient.dtype}")
    if isinstance(gradient, DTensor):
        gradient = gradient.full_tensor()
    if gradient.dtype != torch.float32:
        raise RuntimeError(f"expected a gathered FP32 gradient, got {gradient.dtype}")
    return gradient.detach()


def _assert_checkpointing(model: torch.nn.Module, *, enabled: bool) -> None:
    if model.gradient_checkpointing:
        raise RuntimeError("native Qwen gradient checkpointing remained enabled")
    for index, block in enumerate(model.transformer_blocks):
        if enabled:
            if not isinstance(block, CheckpointWrapper):
                raise RuntimeError(f"Qwen block {index} was not checkpoint-wrapped")
            if block.checkpoint_impl is not CheckpointImpl.NO_REENTRANT:
                raise RuntimeError(f"Qwen block {index} uses the wrong checkpoint implementation")
            if not isinstance(block._checkpoint_wrapped_module, QwenImageTransformerBlock):
                raise RuntimeError(f"Qwen block {index} wrapped an unexpected module")
        elif not isinstance(block, QwenImageTransformerBlock):
            raise RuntimeError(f"Qwen block {index} changed while checkpointing was disabled")


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen MR210 FSDP2 regression requires CUDA")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    initialize_pdd_distributed(backend="nccl", timeout_minutes=5)
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    payload = [tempfile.mkdtemp(prefix="modelopt-pdd-mr210-fsdp-") if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0, device=device)
    root = pathlib.Path(payload[0])
    model_root = root / "model"
    model_dir = model_root / "tiny_qwen_image"
    completed = False
    try:
        if rank == 0:
            assert create_tiny_qwen_image_pipeline_dir(model_root) == model_dir
        dist.barrier()

        config = resolve_pdd_recipe_config(
            _raw_config(
                model_dir,
                root / "checkpoints",
                activation_checkpointing=args.activation_checkpointing,
            )
        )
        setup = build_pdd_setup(config)
        _assert_checkpointing(setup.student, enabled=args.activation_checkpointing)
        _assert_checkpointing(setup.teacher, enabled=args.activation_checkpointing)
        actual_pipeline = PDDPipeline(
            setup.student,
            setup.teacher,
            config.pdd,
            QwenImagePDDAdapter(config.pdd, compute_dtype=torch.bfloat16),
        )

        reference_student = QwenImageTransformer2DModel.from_pretrained(
            model_dir,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        reference_student = adopt_qwen_image_mr210_forward(reference_student)
        reference_teacher = copy.deepcopy(reference_student).eval().requires_grad_(False)
        reference_projection = convert_qwen_image_to_pdd(reference_student, config.pdd)
        reference_student.to(device=device, dtype=torch.float32)
        reference_teacher.to(device=device, dtype=torch.float32)
        reference_pipeline = PDDPipeline(
            reference_student,
            reference_teacher,
            config.pdd,
            _FSDPBoundaryReferenceAdapter(config.pdd, compute_dtype=torch.bfloat16),
        )

        root_calls = {"student": 0, "teacher": 0}
        student_times: list[torch.Tensor] = []
        teacher_times: list[torch.Tensor] = []
        inner_student_calls = 0

        def student_root_hook(_module, _args, _kwargs):
            root_calls["student"] += 1

        def teacher_root_hook(_module, _args, _kwargs):
            root_calls["teacher"] += 1

        def student_time_hook(_module, args_for_module):
            student_times.append(args_for_module[0].detach().clone())

        def teacher_time_hook(_module, args_for_module):
            teacher_times.append(args_for_module[0].detach().clone())

        def inner_student_hook(_module, _args, _kwargs):
            nonlocal inner_student_calls
            inner_student_calls += 1

        student_block = setup.student.transformer_blocks[0]
        if isinstance(student_block, CheckpointWrapper):
            student_block = student_block._checkpoint_wrapped_module
        hooks = [
            setup.student.register_forward_pre_hook(student_root_hook, with_kwargs=True),
            setup.teacher.register_forward_pre_hook(teacher_root_hook, with_kwargs=True),
            setup.student.time_text_embed.register_forward_pre_hook(student_time_hook),
            setup.teacher.time_text_embed.register_forward_pre_hook(teacher_time_hook),
            student_block.register_forward_pre_hook(inner_student_hook, with_kwargs=True),
        ]

        generator = torch.Generator().manual_seed(20260716)
        data = torch.randn(1, 4, 4, 4, generator=generator).to(device)
        noise = torch.randn(1, 4, 4, 4, generator=generator).to(device)
        condition = (
            torch.randn(1, 3, 16, generator=generator).to(
                device=device,
                dtype=torch.bfloat16,
            ),
            torch.tensor([[1, 1, 1]], device=device, dtype=torch.long),
        )
        negative_condition = (
            torch.randn(1, 2, 16, generator=generator).to(
                device=device,
                dtype=torch.bfloat16,
            ),
            torch.tensor([[1, 1]], device=device, dtype=torch.long),
        )
        n = torch.tensor([0], device=device, dtype=torch.long)
        k = torch.tensor([2], device=device, dtype=torch.long)

        actual_loss, _ = actual_pipeline.compute_loss(
            data,
            noise=noise,
            condition=condition,
            negative_condition=negative_condition,
            n=n,
            k=k,
        )
        actual_loss.backward()
        reference_loss, _ = reference_pipeline.compute_loss(
            data,
            noise=noise,
            condition=condition,
            negative_condition=negative_condition,
            n=n,
            k=k,
        )
        reference_loss.backward()
        for hook in hooks:
            hook.remove()

        if actual_loss.dtype != torch.float32 or reference_loss.dtype != torch.float32:
            raise RuntimeError(
                f"PDD losses must be FP32, got {actual_loss.dtype} and {reference_loss.dtype}"
            )
        if root_calls != {"student": 1, "teacher": 2}:
            raise RuntimeError(f"unexpected adopted-root call counts: {root_calls}")
        if len(student_times) != 1 or len(teacher_times) != 2:
            raise RuntimeError(
                f"unexpected time capture counts: {len(student_times)}, {len(teacher_times)}"
            )
        if any(value.dtype != torch.float32 for value in (*student_times, *teacher_times)):
            raise RuntimeError("FSDP rounded an MR210 timestep before time_text_embed")
        expected_time = actual_pipeline.time_grid(device)[n]
        if not torch.equal(student_times[0], expected_time):
            raise RuntimeError("student time does not equal the exact first grid value")
        if student_times[0].item() == student_times[0].to(torch.bfloat16).float().item():
            raise RuntimeError("the 0.999 discriminator did not distinguish BF16 rounding")
        expected_inner_calls = 2 if args.activation_checkpointing else 1
        if inner_student_calls != expected_inner_calls:
            raise RuntimeError(
                "unexpected student block call count: "
                f"expected {expected_inner_calls}, got {inner_student_calls}"
            )

        torch.testing.assert_close(actual_loss, reference_loss, rtol=2e-3, atol=2e-4)
        gradient_pairs = (
            (setup.projection.weight, reference_projection.weight),
            (setup.student.img_in.weight, reference_student.img_in.weight),
        )
        for actual_parameter, reference_parameter in gradient_pairs:
            actual_gradient = _full_fp32_gradient(actual_parameter)
            reference_gradient = _full_fp32_gradient(reference_parameter)
            if (
                not torch.isfinite(actual_gradient).all()
                or not torch.isfinite(reference_gradient).all()
            ):
                raise FloatingPointError("MR210 FSDP gradient comparison is non-finite")
            torch.testing.assert_close(
                actual_gradient,
                reference_gradient,
                rtol=1e-2,
                atol=2e-3,
            )

        gathered_losses = [torch.zeros_like(actual_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_losses, actual_loss.detach())
        for value in gathered_losses[1:]:
            torch.testing.assert_close(value, gathered_losses[0], rtol=0, atol=0)

        setup.optimizer.zero_grad(set_to_none=True)
        training = build_pdd_training_artifacts(setup, config)
        validation_batch = PreparedPDDBatch(
            data,
            condition,
            negative_condition,
            (f"post-validation-update-{rank}",),
        )
        run_pdd_validation(
            training.pipeline,
            [validation_batch],
            [
                PDDValidationAssignment(index, f"post-validation-update-{index}", 0, 2)
                for index in range(dist.get_world_size())
            ],
            validation_seed=11,
        )
        post_validation = training.trainer.train_step(
            validation_batch,
            noise=noise,
            n=n,
            k=k,
        )
        if (
            post_validation.pdd_projection_update_ratio is None
            or post_validation.pdd_projection_update_ratio <= 0
        ):
            raise RuntimeError("post-validation FSDP projection update was not measured")
        if rank == 0:
            print(
                json.dumps(
                    {
                        "activation_checkpointing": args.activation_checkpointing,
                        "actual_loss": actual_loss.item(),
                        "reference_loss": reference_loss.item(),
                        "post_validation_projection_update_ratio": (
                            post_validation.pdd_projection_update_ratio
                        ),
                        "student_block_calls": inner_student_calls,
                        "time_0": student_times[0].item(),
                        "world_size": dist.get_world_size(),
                    },
                    sort_keys=True,
                )
            )
        dist.barrier()
        completed = True
    finally:
        if completed and rank == 0:
            shutil.rmtree(root, ignore_errors=True)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
