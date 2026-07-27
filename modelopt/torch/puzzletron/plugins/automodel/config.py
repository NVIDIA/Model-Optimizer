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

"""Translate stage-owned Puzzletron parallelism into a NeMo AutoModel recipe.

Operators configure the model mesh directly under each model-loading stage. Puzzletron owns
the stable AutoModel recipe boilerplate and derives ``dp_size`` from the independent FSDP
shard and replication axes. This keeps activation scoring, depth RPC, replacement scoring,
and bypass/KD free to use different topologies without maintaining duplicated recipe YAMLs.

Example config block (in the puzzletron pruning config — only scalars/paths, no ``_target_``)::

    pruning:
      backend: automodel
      model_name_or_path: ${puzzle_dir}/ckpts/teacher   # optional; defaults to this
      activations_log_dir: ${puzzle_dir}/pruning/.../scores
      activation_hooks_kwargs: {method: iterative, ...}
      automodel:
        force_hf: true
        use_puzzletron_dataloader: true
        parallel:
          tp: 1
          cp: 1
          pp: 2
          ep: 4
          dp_shard: 4
          dp_replicate: 1
"""

import logging
from pathlib import Path

from omegaconf import OmegaConf

from ...anymodel.model_descriptor import ModelDescriptorFactory
from ...dataset.config import DataLayout, Modality, PuzzletronDataSpec

logger = logging.getLogger(__name__)

__all__ = [
    "build_recipe_config",
    "build_stage_recipe_config",
    "build_solution_recipe_config",
    "inject_descriptor_model_kwargs",
    "inject_descriptor_pipeline_config",
    "scoring_params",
    "solution_scoring_params",
]

_FROM_PRETRAINED_TARGET = "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained"
_FROM_PRETRAINED_VLM_TARGET = (
    "nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained"
)
_DEFAULT_TEACHER_SUBDIR = "ckpts/teacher"


def _int_or_default(value, default: int = 1) -> int:
    if value in (None, "none", "None", ""):
        return default
    return int(value)


def _as_dict(node) -> dict:
    """Convert an OmegaConf node (or plain mapping/None) to a resolved plain dict."""
    if node is None:
        return {}
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)
    return dict(node)


def _teacher_path(hydra_cfg) -> str:
    explicit = hydra_cfg.pruning.get("model_name_or_path", None)
    if explicit:
        return str(explicit)
    return f"{hydra_cfg.puzzle_dir}/{_DEFAULT_TEACHER_SUBDIR}"


def _inject_canonical_data(
    recipe: dict,
    hydra_cfg,
    *,
    split: str = "validation",
    num_samples: int | None = None,
) -> dict:
    raw_data = _as_dict(hydra_cfg.get("data", None))
    if not raw_data or "layout" not in raw_data:
        return recipe
    spec = PuzzletronDataSpec.from_mapping(raw_data)
    model = dict(recipe.get("model") or {})
    if spec.modality is Modality.TEXT and spec.layout is DataLayout.FIXED:
        return recipe
    source_path = raw_data.get("path")
    if not source_path:
        raise ValueError("canonical data requires data.path")
    if spec.modality is Modality.MULTIMODAL:
        if bool(model.get("force_hf", True)):
            raise ValueError(
                "multimodal Puzzletron stages require native AutoModel; "
                "set model.force_hf=False"
            )
        model["_target_"] = _FROM_PRETRAINED_VLM_TARGET
        recipe["model"] = model
        dataset = dict(recipe.get("dataset") or {})
        dataset.update(
            {
                "_target_": (
                    "modelopt.torch.puzzletron.dataset."
                    "load_materialized_conversation_dataset"
                ),
                "path_or_dataset": str(source_path),
                "pretokenize": True,
                "truncate": False,
                "inject_fake_images": False,
                "max_length": int(spec.max_sample_length),
            }
        )
        if num_samples is not None and spec.layout is not DataLayout.PACKED_VARLEN:
            dataset["num_samples"] = int(num_samples)
        recipe["dataset"] = dataset
        processor = dict(recipe.get("processor") or {})
        processor.setdefault("trust_remote_code", True)
        recipe["processor"] = processor
    elif spec.layout is not DataLayout.FIXED:
        if (
            spec.layout is DataLayout.PACKED_VARLEN
            and bool(model.get("force_hf", True))
        ):
            raise ValueError(
                "packed variable-length text data require native AutoModel; "
                "set model.force_hf=False"
            )
        dataset = {
            "_target_": (
                "modelopt.torch.puzzletron.distillation.dataset."
                "make_puzzletron_chat_dataset"
            ),
            "dataset_path": str(source_path),
            "split": str(split),
            "seq_length": int(spec.max_sample_length),
            "seed": int(raw_data.get("seed", 42)),
        }
        if num_samples is not None and spec.layout is not DataLayout.PACKED_VARLEN:
            dataset["num_samples"] = int(num_samples)
        recipe["dataset"] = dataset
        recipe["dataloader"] = {
            "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
            "shuffle": False,
            "collate_fn": "nemo_automodel.components.datasets.utils.default_collater",
        }
    if spec.layout is DataLayout.PACKED_VARLEN:
        packing = spec.packing
        if spec.modality is Modality.MULTIMODAL:
            recipe["packed_sequence"] = {
                "pack_size": int(packing.pack_size),
                "max_length": int(spec.max_sample_length),
                "packing_ratio": float(packing.packing_ratio),
                "drop_long_samples": bool(packing.drop_long_samples),
                "pretokenize": True,
                # AutoModel retains indexed document IDs and reconstructs kernel boundaries.
                "attn_implementation": "flash_attention_2",
                "collate_max_length": int(packing.pack_size),
            }
            if num_samples is not None:
                recipe["packed_sequence"]["max_packs"] = int(num_samples)
        else:
            recipe["packed_sequence"] = {
                "packed_sequence_size": int(packing.pack_size),
                "packing_strategy": "neat",
                "drop_long_samples": bool(packing.drop_long_samples),
            }
            if num_samples is not None:
                recipe["packed_sequence"]["max_packs"] = int(num_samples)
    return recipe


def build_stage_recipe_config(automodel_cfg) -> dict:
    """Generate the standard AutoModel recipe from a stage's parallel mesh.

    ``dp_shard`` is the FSDP shard axis on which expert parallelism is overlaid;
    ``dp_replicate`` is the independent FSDP replication axis. Consequently AutoModel's
    combined ``dp_size`` is their product and ``dp_shard`` must be divisible by ``ep``.
    """
    config = _as_dict(automodel_cfg)
    removed = sorted({"recipe", "recipe_path"}.intersection(config))
    if removed:
        raise ValueError(
            "AutoModel recipe files and inline recipes were removed; configure "
            f"automodel.parallel instead (found {', '.join(removed)})"
        )
    parallel = _as_dict(config.get("parallel"))
    if not parallel:
        raise ValueError("AutoModel stages require an automodel.parallel mapping")

    tp = _int_or_default(parallel.get("tp"), 1)
    cp = _int_or_default(parallel.get("cp"), 1)
    pp = _int_or_default(parallel.get("pp"), 1)
    ep = _int_or_default(parallel.get("ep"), 1)
    dp_shard = _int_or_default(parallel.get("dp_shard"), 1)
    dp_replicate = _int_or_default(parallel.get("dp_replicate"), 1)
    axes = {
        "tp": tp,
        "cp": cp,
        "pp": pp,
        "ep": ep,
        "dp_shard": dp_shard,
        "dp_replicate": dp_replicate,
    }
    invalid = {name: value for name, value in axes.items() if value < 1}
    if invalid:
        raise ValueError(f"AutoModel parallel axes must be positive integers: {invalid}")
    if dp_shard % ep:
        raise ValueError(
            "automodel.parallel.dp_shard must be divisible by ep because AutoModel "
            f"overlays EP on the FSDP shard axis; got dp_shard={dp_shard}, ep={ep}"
        )
    if dp_replicate > 1 and dp_shard == 1:
        raise ValueError(
            "automodel.parallel.dp_replicate greater than one requires dp_shard "
            "greater than one because AutoModel FSDP2 does not support a pure DDP mesh; "
            f"got dp_shard={dp_shard}, dp_replicate={dp_replicate}"
        )

    distributed = {
        "dp_size": dp_shard * dp_replicate,
        "tp_size": tp,
        "cp_size": cp,
        "ep_size": ep,
        "pp_size": pp,
        "sequence_parallel": bool(parallel.get("sequence_parallel", False)),
    }
    if dp_replicate > 1:
        distributed["dp_replicate_size"] = dp_replicate
    if pp > 1:
        distributed["pipeline"] = {
            "pp_schedule": str(parallel.get("pipeline_schedule", "1f1b")),
            "pp_microbatch_size": 1,
            "pp_batch_size": pp,
        }

    return {
        "step_scheduler": {"global_batch_size": 1, "local_batch_size": 1, "max_steps": 1},
        "dist_env": {"backend": "nccl"},
        "model": {"torch_dtype": "bf16", "trust_remote_code": True},
        "checkpoint": {"enabled": False},
        "distributed": distributed,
        "distributed_config": {
            "_target_": "nemo_automodel.components.distributed.config.FSDP2Config",
            "activation_checkpointing": bool(config.get("activation_checkpointing", True)),
        },
        "packed_sequence": {"packed_sequence_size": 0},
        "dataset": {
            "_target_": (
                "modelopt.torch.puzzletron.plugins.automodel.dummy_data.make_dummy_dataset"
            ),
            "num_samples": 1,
            "seq_length": 512,
        },
        "dataloader": {
            "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
            "shuffle": False,
            "collate_fn": "nemo_automodel.components.datasets.utils.default_collater",
        },
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 0.0},
        "loss_fn": {
            "_target_": "nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy"
        },
    }


def _load_model_config_for_descriptor(model_path, *, trust_remote_code: bool):
    """Load only checkpoint metadata for descriptor-owned recipe adaptations."""
    from transformers import AutoConfig

    from ...anymodel.registry import register_native_config_aliases

    register_native_config_aliases()
    kwargs = {"trust_remote_code": trust_remote_code}
    if Path(str(model_path)).exists():
        kwargs["local_files_only"] = True
    return AutoConfig.from_pretrained(str(model_path), **kwargs)


def _registered_descriptor(descriptor_name):
    descriptor = ModelDescriptorFactory.get(descriptor_name) if descriptor_name else None
    if isinstance(descriptor, str):
        # Some callers (notably global KD recipe construction) reach this helper before
        # ``apply_patch()`` imports the top-level AnyModel package. Importing the model package
        # triggers descriptor registration without constructing any model weights.
        from ...anymodel import models as _registered_models  # noqa: F401

        descriptor = ModelDescriptorFactory.get(descriptor_name)
    return None if isinstance(descriptor, str) else descriptor


def _merge_required_mapping(target: dict, required: dict, *, path: str) -> None:
    for key, value in required.items():
        key_path = f"{path}.{key}" if path else key
        if key not in target:
            target[key] = value
        elif isinstance(target[key], dict) and isinstance(value, dict):
            _merge_required_mapping(target[key], value, path=key_path)
        elif target[key] != value:
            raise ValueError(
                f"Descriptor requires {key_path}={value!r}, got {target[key]!r}"
            )


def inject_descriptor_model_kwargs(
    recipe: dict,
    *,
    model_path,
    descriptor_name,
    trust_remote_code: bool,
    model_key: str = "model",
) -> dict:
    descriptor = _registered_descriptor(descriptor_name)
    if descriptor is None:
        return recipe
    required_fn = getattr(descriptor, "automodel_model_kwargs", None)
    if not callable(required_fn):
        return recipe
    model_config = _load_model_config_for_descriptor(
        model_path, trust_remote_code=trust_remote_code
    )
    required = dict(
        required_fn(model_config, distributed=dict(recipe.get("distributed") or {})) or {}
    )
    distributed = dict(recipe.get("distributed") or {})
    model = dict(recipe.get(model_key) or {})
    capabilities_fn = getattr(descriptor, "puzzletron_capabilities", None)
    capabilities = capabilities_fn(model_config) if callable(capabilities_fn) else None
    native_tp = (
        int(distributed.get("tp_size", 1) or 1) > 1
        and not bool(model.get("force_hf", True))
        and bool(getattr(capabilities, "native_automodel_supported", False))
    )
    if native_tp:
        backend_fn = getattr(descriptor, "automodel_tp_linear_backend", None)
        linear_backend = backend_fn(model_config) if callable(backend_fn) else "torch"
        if linear_backend is not None:
            backend = dict(required.get("backend") or {})
            configured = backend.get("linear")
            if configured is not None and configured != linear_backend:
                raise ValueError(
                    f"Descriptor requires native TP linear backend {linear_backend!r}, "
                    f"got {configured!r}"
                )
            backend["linear"] = linear_backend
            required["backend"] = backend
    if required:
        _merge_required_mapping(model, required, path=model_key)
        recipe[model_key] = model
    return recipe


# Backward-compatible private name for in-tree callers and downstream imports.
_inject_descriptor_model_kwargs = inject_descriptor_model_kwargs


def inject_descriptor_pipeline_config(
    recipe: dict,
    *,
    model_path,
    descriptor_name,
    trust_remote_code: bool = True,
) -> dict:
    """Let the model descriptor customize AutoModel PP splitting for this recipe.

    The NeMo HF splitter is intentionally generic and assumes common module names.  Puzzletron
    descriptors know the exact remote-code names, so PP FQNs are injected here before NeMo builds
    the pipeline.  Existing explicit ``distributed.pipeline.module_fqns_per_model_part`` entries
    are respected.
    """
    descriptor = _registered_descriptor(descriptor_name)
    if descriptor is None or not hasattr(descriptor, "pipeline_module_fqns_per_model_part"):
        return recipe

    distributed = dict(recipe.get("distributed", {}))
    pp_size = _int_or_default(distributed.get("pp_size"), 1)
    if pp_size <= 1:
        return recipe

    pipeline = dict(distributed.get("pipeline") or {})
    if pipeline.get("module_fqns_per_model_part") is not None:
        return recipe

    model_config = _load_model_config_for_descriptor(
        model_path,
        trust_remote_code=bool(trust_remote_code),
    )
    descriptor_pipeline = dict(pipeline)
    descriptor_pipeline["_puzzletron_force_hf"] = bool(
        recipe.get("model", {}).get("force_hf", True)
    )
    module_fqns = descriptor.pipeline_module_fqns_per_model_part(
        model_config,
        pp_size=pp_size,
        pipeline_config=descriptor_pipeline,
    )
    if not module_fqns:
        return recipe

    if len(module_fqns) % pp_size != 0:
        raise ValueError(
            f"{descriptor_name} descriptor returned {len(module_fqns)} PP stage(s), "
            f"which is not divisible by pp_size={pp_size}"
        )

    pipeline["module_fqns_per_model_part"] = module_fqns
    distributed["pipeline"] = pipeline
    recipe["distributed"] = distributed
    logger.info(
        "Injected descriptor PP module FQNs for %s: %d stage(s), pp_size=%d",
        descriptor_name,
        len(module_fqns),
        pp_size,
    )
    return recipe


def _align_pipeline_seq_len(recipe: dict, *, block_size, cp_size=None) -> dict:
    """Keep static PP shape metadata aligned with Puzzletron's real dataloader.

    The matrix script writes a standalone NeMo recipe before Puzzletron resolves the final
    stage config.  When a smoke config changes ``block_size`` later, the PyTorch pipeline
    stage can be initialized with stale ``pp_seq_len`` metadata and reject the real batch
    shape.  Align it here, at the boundary where both configs are visible.
    """
    if block_size in (None, "none", "None", ""):
        return recipe

    distributed = dict(recipe.get("distributed", {}))
    if _int_or_default(distributed.get("pp_size"), 1) <= 1:
        return recipe

    cp = _int_or_default(cp_size if cp_size is not None else distributed.get("cp_size"), 1)
    pipeline = dict(distributed.get("pipeline") or {})
    pipeline["pp_seq_len"] = int(block_size) // max(cp, 1)
    distributed["pipeline"] = pipeline
    recipe["distributed"] = distributed
    return recipe


def _align_dummy_dataset(recipe: dict, *, num_samples, block_size) -> dict:
    """Keep generated setup data aligned with the stage's real validation shape."""

    dataset = dict(recipe.get("dataset") or {})
    if not str(dataset.get("_target_", "")).endswith("make_dummy_dataset"):
        return recipe
    if num_samples not in (None, "none", "None", ""):
        dataset["num_samples"] = int(num_samples)
    if block_size not in (None, "none", "None", ""):
        dataset["seq_length"] = int(block_size)
    recipe["dataset"] = dataset
    return recipe


def _align_pipeline_batch_size(recipe: dict, *, micro_batch_size) -> dict:
    """Align static PP metadata with the batch after Puzzletron's DP split."""
    if micro_batch_size in (None, "none", "None", ""):
        return recipe
    distributed = dict(recipe.get("distributed", {}))
    explicit_dp = distributed.get("dp_size")
    # AutoModel's StepScheduler counts the EP mesh as its data mesh when no
    # explicit FSDP/DP axis exists. EP ranks still consume the same packed
    # batch, so this factor affects scheduler divisibility only—not batch
    # slicing or the PP microbatch shape.
    scheduler_dp = max(
        _int_or_default(
            explicit_dp if explicit_dp not in (None, "none", "None", "") else distributed.get("ep_size"),
            1,
        ),
        1,
    )
    # AutoModel overlays EP on the configured DP shard axis. Only the residual
    # DP degree owns distinct samples; EP peers consume the same sample slice.
    if explicit_dp in (None, "none", "None", ""):
        batch_dp = 1
    else:
        ep_size = max(_int_or_default(distributed.get("ep_size"), 1), 1)
        if scheduler_dp % ep_size:
            raise ValueError(
                "AutoModel dp_size must be divisible by ep_size for pipeline batch "
                f"alignment; got dp_size={scheduler_dp}, ep_size={ep_size}"
            )
        batch_dp = max(scheduler_dp // ep_size, 1)
    global_batch = int(micro_batch_size)
    local_batch = (
        global_batch // batch_dp if global_batch % batch_dp == 0 else global_batch
    )
    scheduler = dict(recipe.get("step_scheduler") or {})
    if _int_or_default(distributed.get("pp_size"), 1) <= 1:
        scheduler["local_batch_size"] = local_batch
        scheduler["global_batch_size"] = local_batch * scheduler_dp
        recipe["step_scheduler"] = scheduler
        return recipe
    pipeline = dict(distributed.get("pipeline") or {})
    pipeline["pp_microbatch_size"] = local_batch
    pipeline["pp_batch_size"] = local_batch * _int_or_default(distributed.get("pp_size"), 1)
    distributed["pipeline"] = pipeline
    recipe["distributed"] = distributed
    scheduler["local_batch_size"] = pipeline["pp_batch_size"]
    scheduler["global_batch_size"] = pipeline["pp_batch_size"] * scheduler_dp
    recipe["step_scheduler"] = scheduler
    return recipe


def build_recipe_config(hydra_cfg) -> dict:
    """Build the NeMo recipe config dict for AutoModel activation scoring.

    Generates the standard recipe from ``pruning.automodel.parallel`` and injects the teacher
    checkpoint path, AnyModel descriptor, and ``force_hf`` into its ``model`` block.
    """
    automodel_cfg = hydra_cfg.pruning.get("automodel", None)
    recipe = build_stage_recipe_config(automodel_cfg)

    model = dict(recipe.get("model", {}))
    model.setdefault("_target_", _FROM_PRETRAINED_TARGET)
    model["pretrained_model_name_or_path"] = _teacher_path(hydra_cfg)
    model.setdefault("anymodel_descriptor", hydra_cfg.get("descriptor", None))
    model.setdefault("force_hf", bool(_as_dict(automodel_cfg).get("force_hf", True)))
    model.setdefault("trust_remote_code", True)
    recipe["model"] = model

    if model["anymodel_descriptor"] is None:
        raise ValueError(
            "AutoModel scoring needs an AnyModel descriptor. Set "
            "pruning.automodel.recipe.model.anymodel_descriptor or the top-level 'descriptor'."
        )

    _inject_canonical_data(
        recipe,
        hydra_cfg,
        split=str(hydra_cfg.pruning.get("val_dataset_name", "validation")),
        num_samples=hydra_cfg.pruning.get("eval_samples", None),
    )
    _align_dummy_dataset(
        recipe,
        num_samples=hydra_cfg.pruning.get("eval_samples", None),
        block_size=hydra_cfg.pruning.get("block_size", None),
    )

    _inject_descriptor_model_kwargs(
        recipe,
        model_path=model["pretrained_model_name_or_path"],
        descriptor_name=model["anymodel_descriptor"],
        trust_remote_code=bool(model.get("trust_remote_code", True)),
    )

    inject_descriptor_pipeline_config(
        recipe,
        model_path=model["pretrained_model_name_or_path"],
        descriptor_name=model["anymodel_descriptor"],
        trust_remote_code=bool(model.get("trust_remote_code", True)),
    )
    _align_pipeline_seq_len(recipe, block_size=hydra_cfg.pruning.get("block_size", None))
    _align_pipeline_batch_size(
        recipe, micro_batch_size=hydra_cfg.pruning.get("micro_batch_size", None)
    )

    # Extend the distributed timeout for model build / vLLM-free startup if the
    # operator set nccl_timeout_minutes and the recipe did not set a timeout.
    if hydra_cfg.get("nccl_timeout_minutes", None) is not None:
        dist_env = dict(recipe.get("dist_env", {}))
        dist_env.setdefault("timeout_minutes", int(hydra_cfg.nccl_timeout_minutes))
        recipe["dist_env"] = dist_env

    return recipe


def _inject_model(recipe: dict, model_path, descriptor, force_hf: bool) -> dict:
    """Set the recipe's ``model`` block to load ``model_path`` via the patched from_pretrained."""
    model = dict(recipe.get("model", {}))
    model.setdefault("_target_", _FROM_PRETRAINED_TARGET)
    model["pretrained_model_name_or_path"] = str(model_path)
    model.setdefault("anymodel_descriptor", descriptor)
    model.setdefault("force_hf", bool(force_hf))
    model.setdefault("trust_remote_code", True)
    recipe["model"] = model
    if model["anymodel_descriptor"] is None:
        raise ValueError(
            "AutoModel scoring needs an AnyModel descriptor (set the top-level 'descriptor')."
        )
    _inject_descriptor_model_kwargs(
        recipe,
        model_path=model["pretrained_model_name_or_path"],
        descriptor_name=model["anymodel_descriptor"],
        trust_remote_code=bool(model.get("trust_remote_code", True)),
    )
    inject_descriptor_pipeline_config(
        recipe,
        model_path=model["pretrained_model_name_or_path"],
        descriptor_name=model["anymodel_descriptor"],
        trust_remote_code=bool(model.get("trust_remote_code", True)),
    )
    return recipe


def build_solution_recipe_config(hydra_cfg, model_path) -> dict:
    """NeMo recipe dict for AutoModel replace-1-block scoring, pointed at ``model_path``.

    ``model_path`` is the teacher dir for the target-extraction phase and a per-solution
    candidate checkpoint dir during the candidate loop. The scoring stage owns its mesh through
    ``scoring.automodel.parallel``.
    """
    automodel_cfg = hydra_cfg.scoring.get("automodel", None)
    recipe = build_stage_recipe_config(automodel_cfg)
    force_hf = bool(_as_dict(automodel_cfg).get("force_hf", False))
    runtime_cfg = hydra_cfg.get("_runtime", {}) or {}
    descriptor = hydra_cfg.get("descriptor", None) or runtime_cfg.get("descriptor", None)
    recipe = _inject_model(recipe, model_path, descriptor, force_hf)
    _inject_canonical_data(
        recipe,
        hydra_cfg,
        split=str(hydra_cfg.scoring.get("val_dataset_name", "validation")),
        num_samples=hydra_cfg.scoring.get("eval_samples", None),
    )
    _align_dummy_dataset(
        recipe,
        num_samples=hydra_cfg.scoring.get("eval_samples", None),
        block_size=hydra_cfg.scoring.get(
            "block_size", hydra_cfg.pruning.get("block_size", None)
        ),
    )
    _align_pipeline_seq_len(
        recipe,
        block_size=hydra_cfg.scoring.get(
            "block_size", hydra_cfg.pruning.get("block_size", None)
        ),
    )
    _align_pipeline_batch_size(
        recipe, micro_batch_size=hydra_cfg.scoring.get("micro_batch_size", None)
    )
    if hydra_cfg.get("nccl_timeout_minutes", None) is not None:
        dist_env = dict(recipe.get("dist_env", {}))
        dist_env.setdefault("timeout_minutes", int(hydra_cfg.nccl_timeout_minutes))
        recipe["dist_env"] = dist_env
    return recipe


def solution_scoring_params(hydra_cfg) -> dict:
    """Scoring-specific knobs for replace-1-block scoring (read from ``scoring``)."""
    scoring = hydra_cfg.scoring
    automodel_cfg = _as_dict(scoring.get("automodel", None))
    teacher_cache_device = str(automodel_cfg.get("teacher_cache_device", "cuda")).lower()
    if teacher_cache_device not in {"cpu", "cuda"}:
        raise ValueError(
            "scoring.automodel.teacher_cache_device must be 'cpu' or 'cuda', "
            f"got {teacher_cache_device!r}"
        )
    data_cfg = _as_dict(hydra_cfg.get("data", None))
    eval_samples = scoring.get("eval_samples", None)
    micro_batch_size = scoring.get("micro_batch_size", 1) or 1
    # The validation loader keeps a partial final batch.  Ceiling division is
    # therefore required here: floor division silently turns a valid smoke run
    # such as 8 samples at micro-batch 32 into zero capture iterations.
    eval_iters = (
        (int(eval_samples) + int(micro_batch_size) - 1) // int(micro_batch_size)
        if eval_samples is not None
        else None
    )
    return {
        "eval_iters": eval_iters,
        "force_hf": bool(automodel_cfg.get("force_hf", True)),
        "use_puzzletron_dataloader": (
            False
            if data_cfg.get("modality") == "multimodal"
            else bool(automodel_cfg.get("use_puzzletron_dataloader", True))
        ),
        "data_cfg": data_cfg,
        "embedding_pruning_cfg": _as_dict(hydra_cfg.get("embedding_pruning", None)),
        "temperature": float(automodel_cfg.get("temperature", 1.0)),
        "chunk_size": int(automodel_cfg.get("chunk_size", 16384)),
        "lm_head_backend": str(automodel_cfg.get("lm_head_backend", "flash_kld")),
        "teacher_cache_device": teacher_cache_device,
        "flash_kld_token_chunk_size": automodel_cfg.get("flash_kld_token_chunk_size", None),
        "flash_kld_reduction_backend": str(
            automodel_cfg.get("flash_kld_reduction_backend", "fla")
        ),
    }


def scoring_params(hydra_cfg) -> dict:
    """Extract the scoring-specific knobs the recipe needs (not part of the NeMo schema).

    ``hook_kwargs`` mirrors the legacy ``activation_hooks_kwargs`` (with
    ``validation_full_iters`` injected, matching ``validate_model``) and is handed to the
    target resolver. ``eval_iters`` is the number of calibration batches: for the iterative
    method it must equal ``validation_full_iters`` (one iteration per batch).
    """
    pruning = hydra_cfg.pruning
    automodel_cfg = _as_dict(pruning.get("automodel", None))
    data_cfg = _as_dict(hydra_cfg.get("data", None))
    hook_kwargs = _as_dict(pruning.get("activation_hooks_kwargs", None))

    recipe = build_stage_recipe_config(pruning.get("automodel", None))
    distributed = _as_dict(recipe.get("distributed", None))
    ep_size = distributed.get("ep_size", 1) or 1

    method = hook_kwargs.get("method")
    eval_samples = pruning.get("eval_samples", None)
    micro_batch_size = pruning.get("micro_batch_size", 1) or 1
    # validate_model derives this as eval_samples // micro_batch_size (single data rank).
    validation_full_iters = (
        int(eval_samples) // int(micro_batch_size) if eval_samples is not None else None
    )
    hook_kwargs["validation_full_iters"] = validation_full_iters

    # Stateful greedy methods use one batch per pruning iteration, so the calibration length is
    # fixed by validation_full_iters. For additive methods honor an explicit automodel.eval_iters,
    # else default to the same validation_full_iters budget (eval_samples // micro_batch_size) so
    # the calibration length matches the legacy scorer instead of silently sweeping the whole
    # dataloader.
    if method in (
        "iterative",
        "ple_channel_contribution",
        "grouped_attention_contribution",
        "moe_channel",
        "moe_cett",
        "expert_intermediate_contribution",
        "shared_expert_intermediate_contribution",
        "moe_shared_channel",
    ):
        eval_iters = validation_full_iters
    else:
        eval_iters = int(automodel_cfg.get("eval_iters", 0)) or validation_full_iters

    return {
        "method": method,
        "hook_kwargs": hook_kwargs,
        "activations_log_dir": pruning.get("activations_log_dir", None),
        "eval_samples": None if eval_samples is None else int(eval_samples),
        "micro_batch_size": int(micro_batch_size),
        "eval_iters": eval_iters,
        "force_hf": bool(automodel_cfg.get("force_hf", True)),
        "ep_size": int(ep_size),
        "use_puzzletron_dataloader": (
            False
            if data_cfg.get("modality") == "multimodal"
            else bool(automodel_cfg.get("use_puzzletron_dataloader", True))
        ),
        "data_cfg": data_cfg,
        "embedding_pruning_cfg": _as_dict(hydra_cfg.get("embedding_pruning", None)),
    }
