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
    ModelOptExternalDraftRecipe,
    ModelOptMedusaRecipe,
    ModelOptSpeculativeRecipeBase,
)
from modelopt.torch.opt.conversion import ModeloptStateManager, modelopt_state
from modelopt.torch.speculative.config import ExternalDraftConfig
from modelopt.torch.speculative.external.vocab_swap import swap_draft_vocabulary
from modelopt.torch.speculative.plugins.hf_dflash import HFDFlashModel
from modelopt.torch.speculative.plugins.hf_domino import DominoLambdaCallback
from modelopt.torch.speculative.plugins.hf_external import SPARSE_CAPABLE_LOSSES
from modelopt.torch.speculative.plugins.hf_training_args import (
    TrainingArguments as SpecTrainingArgs,
)
from modelopt.torch.speculative.plugins.master_weight_adamw import VerifyMasterWeightsCallback
from modelopt.torch.speculative.plugins.modeling_fakebase import find_final_norm_path
from modelopt.torch.speculative.utils import load_vlm_or_llm, patch_transformers5_params_loading
from modelopt.torch.utils import print_rank_0
from modelopt.torch.utils.distributed import is_master, local_rank
from modelopt.torch.utils.logging import warn_rank_0

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
            "(speculative_eagle / speculative_dflash / speculative_medusa)."
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


def setup_external_draft(
    recipe, training_args, base_model, draft=None, dry_run=False, base_tokenizer=None
):
    """Convert an external draft and wire it to the base model.

    Runs on both the fresh-load and the HF-resume path. The base lm_head is never part
    of the draft's own state, so a resumed draft must be re-attached to it either way;
    doing this only on the fresh path leaves a resumed run to fail at the first step
    with no lm_head attached.

    A checkpoint written by this mode does carry ``modelopt_state.pth``, so the patched
    ``from_pretrained`` already restored the ``external`` mode and handed back a
    converted draft. Converting it a second time raises ``TypeError: Cannot create a
    consistent method resolution order``, so the conversion is skipped when the draft is
    already converted -- the validations and the head attachment below still run.

    Args:
        base_model: the target, kept only for its lm_head and vocabulary.
        draft: the already-loaded draft on the resume path; loaded here when None.
        dry_run: recipe validation only; skips the data-source requirement, as the
            ``data.data_path`` check in :func:`train` does.
        base_tokenizer: the base's tokenizer when the caller already loaded it; the fresh
            path has it, the resume path does not and leaves this None.
    """
    if recipe.draft_model_name_or_path is None:
        raise ValueError(
            "draft_model_name_or_path must be set in the recipe YAML or via a dotlist override."
        )
    # Checked here, not on the recipe schema: the loader validates the recipe file once
    # before splicing CLI overrides, so a schema-level raise would make it impossible to
    # pass the data path on the command line -- the shipped recipe carries none.
    # Only "online" is rejected: it is the one mode that produces no cached base
    # outputs at all. "streaming" feeds the same pre-computed hidden states offline
    # does, so it is permitted but untested for this mode -- if its batches ever lack
    # base_model_outputs the forward raises rather than training on nothing.
    if not dry_run and recipe.data.mode == "online":
        raise ValueError(
            "external draft training has no online path: the draft is scored against "
            "cached base outputs, which an online run does not produce. Set "
            "data.offline_data_path."
        )
    if training_args.cp_size > 1:
        raise ValueError(
            "external draft training does not support context parallelism: the draft is a "
            "standalone model scored against cached base outputs, and nothing in this mode "
            f"shards the sequence. Got cp_size={training_args.cp_size}; set it to 1."
        )
    base_vocab_size = base_model.config.vocab_size
    if draft is None:
        draft = transformers.AutoModelForCausalLM.from_pretrained(
            recipe.draft_model_name_or_path,
            dtype="auto",
            device_map="cpu",
            trust_remote_code=recipe.model.trust_remote_code,
        )
    external_cfg: dict = recipe.external.model_dump()
    # The draft's own tokenizer, needed before conversion if its vocabulary has to be
    # re-indexed onto the base's.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        recipe.draft_model_name_or_path,
        model_max_length=training_args.training_seq_len,
        trust_remote_code=recipe.model.trust_remote_code,
    )
    if base_tokenizer is None:
        base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            recipe.model.model_name_or_path,
            model_max_length=training_args.training_seq_len,
            trust_remote_code=recipe.model.trust_remote_code,
        )
    # Gate on the flag, not on a size difference: different tokenizers are routinely padded
    # to the same width. Must precede convert so the saved ModelOpt state describes the
    # post-swap architecture.
    if external_cfg["external_vocab_swap"]:
        # The checkpoint's saved config says whether its rows are already base-indexed;
        # swapping those again would scramble them while every check below still passes.
        saved = dict(modelopt_state(draft)["modelopt_state_dict"]).get("external")
        if saved is not None and saved["config"].get("external_vocab_swap"):
            print_rank_0("[external] draft is already swapped; keeping its embeddings as saved")
        else:
            stats = swap_draft_vocabulary(draft, tokenizer, base_tokenizer, base_vocab_size)
            print_rank_0(
                f"[external] vocabulary swapped onto the base: "
                f"{stats['matched']}/{stats['total']} rows carried over, "
                f"{stats['frequent_coverage']:.1%} of the first 50k ids"
            )
        # The draft's rows are indexed by the base vocabulary from here on, so train and
        # validate it against the base tokenizer.
        tokenizer = base_tokenizer
    # Sparse data carries only the teacher's truncated policy. Objectives that need full
    # teacher logits would silently train against a zero-padded distribution, so gate on
    # the list the plugin exports.
    if recipe.data.mode == "sparse" and external_cfg["external_loss"] not in SPARSE_CAPABLE_LOSSES:
        raise ValueError(
            f"data.sparse_data_path requires external.external_loss in "
            f"{list(SPARSE_CAPABLE_LOSSES)}, got {external_cfg['external_loss']!r}."
        )
    restored_modes = {m for m, _ in modelopt_state(draft)["modelopt_state_dict"]}
    if "external" in restored_modes:
        # The restore replayed the config stored in the checkpoint, so modify() has already
        # run with the values this draft was last trained under. Re-apply the recipe's:
        # otherwise an edited external_loss / top_k / top_p is dropped while the banner and
        # the recipe saved beside the checkpoint report the new one. The manager's config is
        # updated alongside it because modify() writes only the live attributes -- without
        # this the next save_pretrained records the objective the draft was never trained
        # under, and a restore outside this script hands back that stale objective.
        external_config = ExternalDraftConfig(**external_cfg)
        draft.modify(external_config)
        ModeloptStateManager(draft)._last_config = external_config
    elif ModeloptStateManager.has_state_for_mode_type("speculative", model=draft):
        raise ValueError(
            f"the draft checkpoint carries speculative mode(s) {sorted(restored_modes)}, not "
            "'external'. Skipping the convert would leave it as that mode's model type and "
            "fail later with an unrelated error, so refuse here."
        )
    else:
        mtsp.convert(draft, [("external", external_cfg)])
    # Fail before training: a vocab mismatch yields a plausible loss curve and a draft
    # with no acceptance. Equal sizes are not enough -- different tokenizers are
    # routinely padded to the same width -- so also check that the two agree on the ids.
    draft.validate_against_base(base_vocab_size)
    draft.validate_tokenizer_against_base(tokenizer, base_tokenizer)
    # The norm is not always at base_model.model.norm -- FakeBaseModel keeps it at .norm
    # -- so search the same paths the other offline modes use. It stays optional;
    # _teacher_logits raises only if the dump declares a pre-norm hidden.
    # The sparse path carries the teacher's policy in the data, so the base lm_head is
    # dead weight on the GPU; the base is still loaded for the vocabulary check, which is
    # cheap under use_fake_base_for_offline.
    if recipe.data.mode != "sparse":
        norm_path = find_final_norm_path(base_model)
        base_final_norm = base_model.get_submodule(norm_path) if norm_path else None
        # Attach on the device training will run on: the draft is loaded with
        # device_map="cpu", so otherwise the first forward moves the copies mid-step on
        # every rank. training_args.device is the device accelerate assigned this
        # process, which is correct both when every rank sees all GPUs and when
        # CUDA_VISIBLE_DEVICES pins one GPU per rank -- local_rank arithmetic gets the
        # second case wrong.
        draft.attach_base_lm_head(
            base_model.get_output_embeddings(), base_final_norm, device=training_args.device
        )
    return draft, tokenizer


def train():
    config_path, dry_run, overrides = _parse_cli()
    recipe = load_recipe(config_path, overrides=overrides)
    if not isinstance(recipe, ModelOptSpeculativeRecipeBase):
        raise ValueError(
            f"main.py expects a speculative-decoding recipe (eagle / dflash / medusa / external); "
            f"got {type(recipe).__name__} from {config_path!r}."
        )

    # Pydantic-typed sections flow straight through as *_args; only TrainingArguments is
    # reconstructed as an HF dataclass so it can be handed to transformers.Trainer.
    training_args = HfTrainingArguments(**recipe.training.model_dump())
    init_distributed_env(training_args)

    if not dry_run and recipe.data.mode in ("online", "streaming") and not recipe.data.data_path:
        raise ValueError(f"data.mode={recipe.data.mode!r} requires data.data_path.")
    if training_args.cp_size > 1:
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
    if isinstance(recipe, ModelOptExternalDraftRecipe):
        # This mode has no online path (see setup_external_draft), and the shipped recipe
        # carries no data-source field, so mode resolves to "online" under --dry_run.
        # Deriving the flag from it there would make load_vlm_or_llm ignore
        # use_fake_base_for_offline and materialise every base decoder layer on every
        # rank, and would route a VLM base to the real multimodal model, whose parent
        # config has no vocab_size for validate_against_base to read.
        use_offline_training = True

    # Resume path depends on the existing checkpoint's on-disk format: consolidated HF
    # weights load via from_pretrained; FSDP sharded checkpoints load the base model and
    # resume through the Trainer.
    checkpoint_is_hf = _is_hf_format_checkpoint(checkpoint)

    if checkpoint_is_hf:
        assert checkpoint is not None  # guaranteed by checkpoint_is_hf
        with patch_transformers5_params_loading():
            model = load_vlm_or_llm(
                checkpoint,
                dtype="auto",
                trust_remote_code=recipe.model.trust_remote_code,
                config_overrides=recipe.model.config_overrides,
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=recipe.model.trust_remote_code
        )
        if isinstance(recipe, ModelOptExternalDraftRecipe):
            # The restored draft is already converted (the checkpoint carries
            # modelopt_state.pth), but the base lm_head it is scored against is not part
            # of its state, so it still has to be re-attached. The base is not on this
            # path at all, so load it here; use_fake_base_for_offline keeps that cheap.
            if recipe.model.model_name_or_path is None:
                raise ValueError(
                    "model.model_name_or_path must be set to resume an external-draft run: "
                    "the base model supplies the lm_head the draft is scored against."
                )
            # Same arguments as the fresh path below; use_offline_training is pinned
            # True for this recipe above.
            base_model = load_vlm_or_llm(
                recipe.model.model_name_or_path,
                use_fake_base=recipe.model.use_fake_base_for_offline,
                use_offline_training=use_offline_training,
                dtype="auto",
                device_map="cpu",
                trust_remote_code=recipe.model.trust_remote_code,
                config_overrides=recipe.model.config_overrides,
            )
            model, tokenizer = setup_external_draft(
                recipe, training_args, base_model=base_model, draft=model, dry_run=dry_run
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
            config_overrides=recipe.model.config_overrides,
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
        elif isinstance(recipe, ModelOptExternalDraftRecipe):
            # This mode inverts the usual relationship: we train a pretrained draft, and
            # `model` loaded above is the base, kept only for its lm_head and vocabulary.
            model, tokenizer = setup_external_draft(
                recipe,
                training_args,
                base_model=model,
                dry_run=dry_run,
                base_tokenizer=tokenizer,
            )
        else:
            raise ValueError(f"Unsupported speculative recipe type: {type(recipe).__name__}")

    # On the HF-format restore path above, DFlash's modify() ran with the base model still on
    # meta, so the draft has no device, dtype or rotary buffer yet. Re-apply them here, before
    # the Trainer is built: DDP's broadcast_buffers hangs on a draft whose rotary buffer is
    # missing, and the forward only avoids reconciling dtypes because the draft matches the
    # base. A no-op on a fresh convert.
    if isinstance(model, HFDFlashModel):
        model.restore_draft_precision()

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

    print_rank_0("Loading dataset...")
    is_dflash = isinstance(recipe, ModelOptDFlashRecipe)
    if recipe.data.mode == "sparse" and not isinstance(recipe, ModelOptExternalDraftRecipe):
        # Only the external draft reads a sparse policy. Any other mode would build the
        # dataset, hand the collator's teacher_topk_* to a forward that ignores them, and
        # train on an all-zero loss -- so refuse instead of running an empty objective.
        raise ValueError(
            f"data.sparse_data_path is only supported by the external draft recipe; "
            f"{type(recipe).__name__} cannot consume a sparse teacher policy."
        )
    data_module = make_speculative_data_module(
        tokenizer,
        recipe.data,
        train_len=training_args.training_seq_len,
        answer_only_loss=training_args.answer_only_loss,
        shift_labels=not is_dflash,
        final_aux_is_base_hidden=recipe.data.final_aux_is_base_hidden,
        # The external draft never reads the aux planes, so it can consume dumps
        # written with --no-aux-hidden-states. Every other mode must still fail
        # loudly on a dump that omits them.
        aux_hidden_states_optional=isinstance(recipe, ModelOptExternalDraftRecipe),
    )
    if recipe.data.mode == "sparse" and recipe.external.external_loss == "tvd_deploy":
        # tvd_deploy filters the draft side by external_top_k, while the teacher side is
        # whatever the producer baked into the dump. If the two disagree the objective
        # compares two differently-truncated policies: the loss stays well-formed and
        # still falls, so nothing surfaces except a draft that accepts worse than it
        # should. Checked here because this is the first point where both the dump's
        # stored top_k and the configured one are known. A wider draft filter can be a
        # deliberate choice, so warn rather than refuse.
        dump_top_k = getattr(data_module.get("train_dataset"), "top_k", None)
        if dump_top_k and dump_top_k != recipe.external.external_top_k:
            warn_rank_0(
                f"the dump stores the teacher's top-{dump_top_k} policy but "
                f"external.external_top_k={recipe.external.external_top_k} filters the draft; "
                f"tvd_deploy then compares two differently-truncated policies. Set "
                f"external.external_top_k={dump_top_k} to match the dump."
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
    # fp32 master weights are the optimizer's job, and wiring the optimizer is the training
    # loop's. This fails the run if that wiring is ever missed, rather than letting the flag
    # be silently inert for a whole job.
    if getattr(model, "dflash_fp32_master_weights", False):
        callbacks.append(VerifyMasterWeightsCallback())
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


if __name__ == "__main__":
    train()
