# Adapted from https://github.com/tatsu-lab/stanford_alpaca/blob/3783d18/train.py

#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

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

import argparse
import dataclasses
import os

import fsdp2_buffer_patch
import torch
import transformers
from eagle_utils import (
    DFlashFSDP2ShardedSDExportCallback,
    EagleTrainerWithAccLog,
    EagleTrainingPlot,
    LoRAWarmupCallback,
    make_mtp_boost_data_module,
    make_speculative_data_module,
    patch_ring_attention_for_ttt,
)
from rich.pretty import pprint
from transformers.trainer_utils import get_last_checkpoint

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.recipe import load_recipe
from modelopt.recipe.config import (
    ModelOptDFlashRecipe,
    ModelOptEagleRecipe,
    ModelOptMedusaRecipe,
    ModelOptMTPBoostRecipe,
    ModelOptSpeculativeRecipeBase,
)
from modelopt.torch.speculative.plugins.hf_domino import DominoLambdaCallback
from modelopt.torch.speculative.plugins.hf_training_args import (
    TrainingArguments as SpecTrainingArgs,
)
from modelopt.torch.speculative.utils import load_vlm_or_llm, patch_transformers5_params_loading
from modelopt.torch.utils import print_rank_0
from modelopt.torch.utils.distributed import is_master, local_rank

torch.manual_seed(0)
mto.enable_huggingface_checkpointing()

if os.environ.get("PATCH_FSDP2_BUFFERS_TF457") == "1":
    fsdp2_buffer_patch.apply()


# HF-compatible TrainingArguments with our speculative-decoding extensions, auto-derived
# from :class:`SpecTrainingArgs` so its field set can't drift from the Pydantic recipe schema.
# Used at runtime as ``HfTrainingArguments(**recipe.training.model_dump())`` to obtain a
# ``transformers.Trainer``-compatible dataclass.
HfTrainingArguments = dataclasses.make_dataclass(
    "HfTrainingArguments",
    [
        (name, fi.annotation, dataclasses.field(default=fi.default))
        for name, fi in SpecTrainingArgs.model_fields.items()
    ],
    bases=(transformers.TrainingArguments,),
)


def _mtp_boost_training_config(
    recipe: ModelOptMTPBoostRecipe,
) -> dict:
    """Return Trainer arguments with memory-safe FSDP2 loading enabled.

    ``ParallelismConfig`` describes the device mesh but does not construct an
    FSDP plugin. Native MTP masters are too large to tolerate Trainer's normal
    pre-wrap ``model.to(device)`` path, so distributed MTP boosting explicitly
    enables FSDP2 and lets Accelerate shard from a rank-0 CPU state dictionary.
    """
    training_config = recipe.training.model_dump()
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    cp_size = int(training_config.get("cp_size", 1))
    dp_shard_size = training_config.get("dp_shard_size")
    if dp_shard_size is None:
        dp_shard_size = world_size // cp_size
    if dp_shard_size <= 1:
        return training_config

    configured_fsdp = training_config.get("fsdp")
    if configured_fsdp in (None, False, "", []):
        training_config["fsdp"] = "full_shard"

    fsdp_config = dict(training_config.get("fsdp_config") or {})
    version = int(fsdp_config.setdefault("version", 2))
    if version != 2:
        raise ValueError("speculative_mtp_boost requires training.fsdp_config.version=2.")
    if fsdp_config.get("cpu_ram_efficient_loading") is False:
        raise ValueError(
            "speculative_mtp_boost requires "
            "training.fsdp_config.cpu_ram_efficient_loading=true to avoid materializing "
            "the complete BF16 MTP on every GPU before sharding."
        )
    fsdp_config["cpu_ram_efficient_loading"] = True
    training_config["fsdp_config"] = fsdp_config
    return training_config


def _parse_cli() -> tuple[str, bool, list[str]]:
    """Parse --config (required) and --dry_run from argv; return remaining args as dotlist overrides.

    Extra positional args use dotlist syntax, e.g.
    ``model.model_name_or_path=meta-llama/Llama-3.2-1B training.output_dir=ckpts/test``.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--config",
        required=True,
        help=(
            "Path to a modelopt speculative-decoding recipe YAML "
            "(speculative_eagle / speculative_dflash / speculative_medusa / "
            "speculative_mtp_boost)."
        ),
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip training: load base + mtsp.convert + save_pretrained, then exit. "
        "Produces a ModelOpt HF checkpoint with untrained draft-head weights, suitable "
        "for end-to-end plumbing tests (e.g. running scripts/export_hf_checkpoint.py).",
    )
    args, overrides = p.parse_known_args()
    return args.config, args.dry_run, overrides


def init_distributed_env(training_args: transformers.TrainingArguments) -> None:
    """Resolve dp_shard_size from the live env and attach a ParallelismConfig in-place.

    Reads ``WORLD_SIZE`` / ``torch.cuda.device_count()`` and (when actually distributed)
    builds an ``accelerate.ParallelismConfig`` on ``training_args``. Kept out of the
    Pydantic schema so the recipe stays a pure declarative spec.
    """
    if training_args.cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {training_args.cp_size}.")
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    if training_args.dp_shard_size is None:
        training_args.dp_shard_size = world_size // training_args.cp_size
    if training_args.dp_shard_size < 1:
        raise ValueError(
            f"dp_shard_size resolved to {training_args.dp_shard_size}; "
            f"WORLD_SIZE ({world_size}) must be >= cp_size ({training_args.cp_size})."
        )

    if training_args.cp_size > 1 or training_args.dp_shard_size > 1:
        parallel_size = training_args.dp_shard_size * training_args.cp_size
        if world_size % parallel_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by "
                f"dp_shard_size ({training_args.dp_shard_size}) * "
                f"cp_size ({training_args.cp_size}) = {parallel_size}"
            )
        try:
            from accelerate import ParallelismConfig
        except ImportError as e:
            raise ImportError(
                "cp_size>1 or dp_shard_size>1 requires `accelerate` for ParallelismConfig. "
                "Install it via `pip install accelerate`."
            ) from e
        training_args.parallelism_config = ParallelismConfig(
            cp_size=training_args.cp_size,
            dp_shard_size=training_args.dp_shard_size,
            dp_replicate_size=world_size // parallel_size,
        )


def _is_hf_format_checkpoint(checkpoint: str | None) -> bool:
    """True if the checkpoint dir holds consolidated HF weights (from_pretrained-loadable).

    FSDP2 SHARDED_STATE_DICT checkpoints contain only distributed shards
    (``pytorch_model_fsdp_*/``), no ``model.safetensors`` — those return False, signalling
    the caller to load the base model and resume via the Trainer instead. This inspects the
    on-disk format of the *resume* checkpoint, which is a property of the existing bytes and
    is independent of the current run's save mode (the two can differ across runs), so it's
    intentionally separate from the save-time FSDP state-dict-type gate used for the export
    callback.
    """
    if not checkpoint:
        return False
    hf_files = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
    return any(os.path.isfile(os.path.join(checkpoint, f)) for f in hf_files)


def train():
    config_path, dry_run, overrides = _parse_cli()
    recipe = load_recipe(config_path, overrides=overrides)
    if not isinstance(recipe, ModelOptSpeculativeRecipeBase):
        raise ValueError(
            f"main.py expects a speculative-decoding recipe (eagle / dflash / medusa); "
            f"got {type(recipe).__name__} from {config_path!r}."
        )

    is_mtp_boost = isinstance(recipe, ModelOptMTPBoostRecipe)

    # Pydantic-typed sections flow straight through as *_args; only TrainingArguments is
    # reconstructed as an HF dataclass so it can be handed to transformers.Trainer.
    training_config = (
        _mtp_boost_training_config(recipe) if is_mtp_boost else recipe.training.model_dump()
    )
    training_args = HfTrainingArguments(**training_config)
    init_distributed_env(training_args)

    if not dry_run and recipe.data.mode in ("online", "streaming") and not recipe.data.data_path:
        raise ValueError(f"data.mode={recipe.data.mode!r} requires data.data_path.")
    if training_args.cp_size > 1:
        if is_mtp_boost:
            raise ValueError("speculative_mtp_boost does not support context parallelism.")
        patch_ring_attention_for_ttt()
        # accelerate requires an fsdp_plugin when cp_size > 1; the --fsdp launcher flags that
        # used to provide one were dropped from launch_train.sh.
        if not training_args.fsdp_plugin_args:
            training_args.fsdp = "full_shard"
            training_args.fsdp_config = {"fsdp_version": 2}
            training_args.fsdp_plugin_args = training_args._process_fsdp_args()
    if is_master():
        pprint(recipe)

    # Detect checkpoint to resume from
    last_checkpoint = (
        get_last_checkpoint(training_args.output_dir)
        if os.path.isdir(training_args.output_dir)
        else None
    )
    if last_checkpoint:
        print_rank_0(f"Last checkpoint detected: {last_checkpoint}")

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint

    use_offline_training = recipe.data.mode != "online"

    # Resume path depends on the existing checkpoint's on-disk format: consolidated HF
    # weights load via from_pretrained; FSDP sharded checkpoints load the base model and
    # resume through the Trainer.
    checkpoint_is_hf = _is_hf_format_checkpoint(checkpoint)

    if is_mtp_boost:
        model_name_or_path = recipe.model.model_name_or_path
        if model_name_or_path is None:
            raise ValueError("model.model_name_or_path must be set for speculative_mtp_boost.")
        # Native adapters are optional/heavy model-specific integrations, so defer their import
        # until this recipe is selected and keep the existing EAGLE/DFlash startup path unchanged.
        from modelopt.torch.speculative.mtp import (
            attach_native_mtp_online_target,
            create_native_mtp_boost_model,
        )

        model = create_native_mtp_boost_model(
            model_name_or_path,
            dtype=torch.bfloat16,
            adapter=recipe.mtp_boost.adapter,
            rollout_steps=recipe.mtp_boost.rollout_steps,
            hsm_mode=recipe.mtp_boost.hsm_mode,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=training_args.training_seq_len,
            trust_remote_code=recipe.model.trust_remote_code,
        )
        if recipe.data.mode == "online" and not dry_run:
            target_checkpoint = recipe.mtp_boost.target_checkpoint
            assert target_checkpoint is not None
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if training_args.dp_shard_size != world_size:
                raise ValueError(
                    "DSV4 online training uses all ranks for target model parallelism and "
                    "requires training.dp_shard_size=WORLD_SIZE."
                )
            attach_native_mtp_online_target(
                model,
                model_name_or_path,
                target_checkpoint,
                device=torch.device("cuda", local_rank()),
                max_batch_size=training_args.per_device_train_batch_size * world_size,
                max_seq_len=training_args.training_seq_len,
                adapter=recipe.mtp_boost.adapter,
            )
    elif checkpoint_is_hf:
        assert checkpoint is not None  # guaranteed by checkpoint_is_hf
        with patch_transformers5_params_loading():
            model = load_vlm_or_llm(
                checkpoint, dtype="auto", trust_remote_code=recipe.model.trust_remote_code
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=recipe.model.trust_remote_code
        )
    else:
        if checkpoint:
            print_rank_0(
                f"Checkpoint {checkpoint} is not in HF format (FSDP distributed checkpoint). "
                f"Loading base model and resuming via Trainer."
            )
        model_name_or_path = recipe.model.model_name_or_path
        if model_name_or_path is None:
            raise ValueError(
                "model.model_name_or_path must be set in the recipe YAML or via a dotlist override."
            )
        # To avoid OOM for large models, we load and convert model on CPU first.
        # Model will be moved to GPU during HF trainer.init().
        model = load_vlm_or_llm(
            model_name_or_path,
            use_fake_base=recipe.model.use_fake_base_for_offline,
            use_offline_training=use_offline_training,
            dtype="auto",
            device_map="cpu",
            trust_remote_code=recipe.model.trust_remote_code,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=training_args.training_seq_len,
            trust_remote_code=recipe.model.trust_remote_code,
        )
        if isinstance(recipe, ModelOptMedusaRecipe):
            medusa_cfg: dict = recipe.medusa.model_dump()
            mtsp.convert(model, [("medusa", medusa_cfg)])
        elif isinstance(recipe, ModelOptEagleRecipe):
            eagle_cfg: dict = recipe.eagle.model_dump()
            mtsp.convert(model, [("eagle", eagle_cfg)])
            # Load draft vocab cache
            mtsp.plugins.HFEagleModel.load_draft_vocab_cache(model, recipe.data.draft_vocab_cache)
        elif isinstance(recipe, ModelOptDFlashRecipe):
            # Fall back to tokenizer.mask_token_id when not set in the recipe; require one of the two.
            if recipe.dflash.dflash_mask_token_id is None:
                recipe.dflash.dflash_mask_token_id = getattr(tokenizer, "mask_token_id", None)
            if recipe.dflash.dflash_mask_token_id is None:
                raise ValueError(
                    "dflash.dflash_mask_token_id is required: set it in the recipe YAML "
                    "or use a tokenizer that defines mask_token_id."
                )
            dflash_cfg: dict = recipe.dflash.model_dump()
            mtsp.convert(model, [("dflash", dflash_cfg)])
        else:
            raise ValueError(f"Unsupported speculative recipe type: {type(recipe).__name__}")

    if dry_run:
        # is_master() is unreliable here: we return before the HF Trainer inits torch.distributed,
        # so use local_rank() (env-based) to keep a single writer to output_dir.
        if local_rank() == 0:
            os.makedirs(training_args.output_dir, exist_ok=True)
            model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            print_rank_0(
                f"[dry-run] saved ModelOpt HF checkpoint (untrained draft head) to "
                f"{training_args.output_dir}"
            )
        return

    # Move any remaining CPU buffers to CUDA so DDP (NCCL-only) can broadcast
    # them.  We iterate named_buffers and reassign via the owning module to
    # keep the module tree consistent.  Parameters are left on CPU — the HF
    # Trainer will move them during init.
    if torch.cuda.is_available():
        _target_dev = torch.device("cuda", local_rank())
        for name, buf in list(model.named_buffers()):
            if buf.device.type == "cpu":
                parts = name.split(".")
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                setattr(mod, parts[-1], buf.to(_target_dev))

    print_rank_0("Loading dataset...")
    if is_mtp_boost:
        data_module = make_mtp_boost_data_module(
            tokenizer,
            recipe.data,
            train_len=training_args.training_seq_len,
            answer_only_loss=training_args.answer_only_loss,
            seed=training_args.seed,
        )
    else:
        is_dflash = isinstance(recipe, ModelOptDFlashRecipe)
        data_module = make_speculative_data_module(
            tokenizer,
            recipe.data,
            train_len=training_args.training_seq_len,
            answer_only_loss=training_args.answer_only_loss,
            shift_labels=not is_dflash,
            final_aux_is_base_hidden=recipe.data.final_aux_is_base_hidden,
        )

    callbacks = [EagleTrainingPlot(training_args.ar_validate_steps, training_args.estimate_ar)]
    if (
        isinstance(recipe, ModelOptEagleRecipe)
        and recipe.eagle.eagle_base_lora
        and recipe.eagle.eagle_base_lora_warmup_steps > 0
    ):
        callbacks.append(LoRAWarmupCallback(recipe.eagle.eagle_base_lora_warmup_steps))
    # Domino (dflash recipe with projector_type=domino) needs the lambda_base
    # curriculum schedule driven by the trainer's global step.
    if (
        isinstance(recipe, ModelOptDFlashRecipe)
        and recipe.dflash.dflash_architecture_config.get("projector_type") == "domino"
    ):
        callbacks.append(DominoLambdaCallback())
    # Leave training_args.ignore_data_skip at its default (False). The dataset is
    # map-style, so HF Trainer's resume skips consumed indices at the batch-sampler
    # level (accelerate.skip_first_batches) without re-fetching them, landing at the
    # exact data position. Setting it True would restart the data order from the top.

    # Tell the draft model the CP degree so it skips the dense eagle mask under CP.
    model.eagle_cp_size = training_args.cp_size

    trainer = EagleTrainerWithAccLog(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )

    if os.environ.get("PATCH_FSDP2_BUFFERS_TF457") == "1":
        fsdp2_buffer_patch.patch_accelerator(trainer.accelerator)

    # DFlash: export the draft submodule after each checkpoint save — but only under FSDP2
    # SHARDED_STATE_DICT, where checkpoints are distributed shards the post-training
    # export_hf_checkpoint.py pass can't read. Gate by reading the live FSDP state dict
    # type off the accelerator; full-state-dict runs (DDP, single-device, FSDP2
    # FULL_STATE_DICT) use the launcher's post-run export instead.
    if isinstance(recipe, ModelOptDFlashRecipe):
        fsdp_plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
        sd_type = str(getattr(fsdp_plugin, "state_dict_type", "") or "")
        if "SHARDED_STATE_DICT" in sd_type:
            trainer.add_callback(DFlashFSDP2ShardedSDExportCallback())
            print_rank_0("DFlash: FSDP2 SHARDED_STATE_DICT — enabling per-save draft export.")
        else:
            print_rank_0(
                f"DFlash: checkpoints use {sd_type or 'a full state dict'}; relying on the "
                "launcher's post-run export (no per-save export callback added)."
            )

    # Manually enable this to return loss in eval
    trainer.can_return_loss = True
    # Make sure label_smoother is None
    assert trainer.label_smoother is None, (
        "label_smoother is not supported in speculative decoding!"
    )

    # Diagnostic (no-op unless DFLASH_LOG_PARAM_DTYPES=1): verifies FSDP2 dtype sync.
    fsdp2_buffer_patch.log_param_dtypes(trainer.model)

    print_rank_0("Start training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    if not is_mtp_boost:
        return

    if recipe.mtp_boost.export_path:
        # Getting the state dict on every rank is required by FSDP before rank
        # zero writes the native deployment checkpoint.
        state_dict = trainer.accelerator.get_state_dict(trainer.model)
        if trainer.is_world_process_zero:
            native_model = trainer.accelerator.unwrap_model(trainer.model)
            base_model_path = recipe.model.model_name_or_path
            assert base_model_path is not None
            from modelopt.torch.speculative.mtp import export_native_mtp_checkpoint

            export_native_mtp_checkpoint(
                native_model,
                base_model_path,
                recipe.mtp_boost.export_path,
                state_dict=state_dict,
            )
        trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
