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

"""Utility functions for bypass distillation."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils.robust_json import json_dump, json_load

__all__ = [
    "BYPASS_STATE_FILENAME",
    "BYPASS_SUBBLOCK_KEYS_TO_LEARN",
    "bypass_run_is_complete",
    "expected_bypass_runs",
    "get_bypass_config_fingerprint",
    "get_bypass_experiment_fingerprint",
    "get_bypass_run_identity",
    "get_bypass_state_path",
    "get_distributed_modules_ownership",
    "get_pipeline_ownership_context",
    "learned_subblocks_from_keys_to_learn",
    "load_bypass_state",
    "mark_bypass_run_completed",
    "normalize_keys_to_learn",
    "set_experiment_dir",
    "set_experiment_id",
    "update_bypass_checkpoint_state",
    "write_bypass_state",
]

BYPASS_STATE_FILENAME = "bypass_state.json"
BYPASS_SUBBLOCK_KEYS_TO_LEARN = frozenset(
    {"subblock_ffn", "subblock_attention", "subblock_mamba", "entire_block"}
)


def _to_plain_container(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def normalize_keys_to_learn(keys_to_learn: Any) -> dict[str, Any]:
    """Normalize bypass ``keys_to_learn`` into v1 subblock semantics."""
    keys_to_learn = _to_plain_container(keys_to_learn)
    if isinstance(keys_to_learn, str):
        if keys_to_learn in BYPASS_SUBBLOCK_KEYS_TO_LEARN:
            return {"mode": "subblocks", "subblocks": (keys_to_learn,)}
        raise ValueError(
            "keys_to_learn must be one of "
            f"{sorted(BYPASS_SUBBLOCK_KEYS_TO_LEARN)}, got {keys_to_learn!r}"
        )

    if isinstance(keys_to_learn, Sequence):
        values = tuple(keys_to_learn)
        if not all(isinstance(value, str) for value in values):
            raise TypeError(f"keys_to_learn entries must be strings, got {keys_to_learn!r}")
        if not values:
            raise ValueError("keys_to_learn cannot be empty")
        invalid = [value for value in values if value not in BYPASS_SUBBLOCK_KEYS_TO_LEARN]
        if invalid:
            raise ValueError(
                f"keys_to_learn supports only subblock keys in v1; invalid entries: {invalid!r}"
            )
        if "entire_block" in values and len(set(values)) > 1:
            raise ValueError("keys_to_learn cannot mix 'entire_block' with other subblock keys")
        return {"mode": "subblocks", "subblocks": tuple(dict.fromkeys(values))}

    raise TypeError(f"Unsupported keys_to_learn={keys_to_learn!r}")


def learned_subblocks_from_keys_to_learn(keys_to_learn: Any) -> list[str]:
    """Return replacement-library subblocks represented by ``keys_to_learn``."""
    normalized = normalize_keys_to_learn(keys_to_learn)
    subblocks = set(normalized["subblocks"])
    if subblocks == {"entire_block"}:
        return ["block"]

    out: list[str] = []
    if "subblock_attention" in subblocks or "subblock_mamba" in subblocks:
        out.append("attention")
    if "subblock_ffn" in subblocks:
        out.append("ffn")
    return out


def _slug(value: Any) -> str:
    text = str(value).strip().lower().replace("subblock_", "")
    keep = [ch if ch.isalnum() else "_" for ch in text]
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "custom"


def get_bypass_run_identity(cfg: DictConfig) -> dict[str, Any]:
    """Return the config subset that defines a bypass output.

    The full Hydra config carries mutable runtime counters, checkpoint paths and
    logging fields.  Those should not decide whether a completed bypass run can
    be reused.  This identity intentionally keeps architecture, training budget,
    data shape and learning-target fields, because changing any of them changes
    the produced checkpoint.
    """
    bypass = _to_plain_container(cfg.bypass)
    training = bypass.get("training", {})
    data = bypass.get("data", {})
    model = bypass.get("model", {})
    model_factory = bypass.get("model_factory", {})
    return {
        "model": {
            "student_weights_dtype": model.get("student_weights_dtype"),
            "model_config_overrides": model.get("model_config_overrides"),
        },
        "model_factory": {
            "factory": model_factory.get("factory"),
            "block_loss_func": model_factory.get("block_loss_func"),
            "gqa_init_mode": model_factory.get("gqa_init_mode"),
            "mlp_init_mode": model_factory.get("mlp_init_mode"),
            "mlp_init_config": model_factory.get("mlp_init_config"),
            "linear_init_mode": model_factory.get("linear_init_mode"),
            "submodule_for_loss_calculation": model_factory.get("submodule_for_loss_calculation"),
            "keys_to_learn": model_factory.get("keys_to_learn"),
        },
        "training": {
            "learning_rate": training.get("learning_rate"),
            "training_tokens": training.get("training_tokens"),
            "micro_batch_size": training.get("micro_batch_size"),
            "grad_accumulation_steps": training.get("grad_accumulation_steps"),
            "weight_decay": training.get("weight_decay"),
            "decay_lr": training.get("decay_lr"),
            "beta1": training.get("beta1"),
            "beta2": training.get("beta2"),
            "grad_clip": training.get("grad_clip"),
            "grad_clip_type": training.get("grad_clip_type"),
            "warmup_ratio": training.get("warmup_ratio"),
            "min_lr_factor": training.get("min_lr_factor"),
        },
        "data": {
            "dataset_path": cfg.get("dataset_path", None),
            "block_size": data.get("block_size"),
            "data_column": data.get("data_column"),
            "fim_rate": data.get("fim_rate"),
            "fim_spm_rate": data.get("fim_spm_rate"),
            "bos_rate": data.get("bos_rate"),
            "source_datasets_to_discard": data.get("source_datasets_to_discard"),
            "load_from_disk": data.get("load_from_disk"),
            "keep_in_memory": data.get("keep_in_memory"),
            "shuffle_train_data_seed": data.get("shuffle_train_data_seed"),
            "val_dataset_name": data.get("val_dataset_name"),
            "max_eval_samples": data.get("max_eval_samples"),
            "eval_samples_per_process": data.get("eval_samples_per_process"),
        },
        "validation": {
            "disable_validation": bypass.get("disable_validation"),
            "save_best_ckpt": bypass.get("save_best_ckpt"),
            "realize_best_or_latest": bypass.get("realize_best_or_latest"),
            "eval_interval": training.get("eval_interval"),
            "val_micro_batch_size": training.get("val_micro_batch_size"),
        },
        "seed": bypass.get("seed"),
        "dtype": bypass.get("dtype"),
    }


def get_bypass_config_fingerprint(cfg: DictConfig) -> str:
    identity = get_bypass_run_identity(cfg)
    payload = json.dumps(identity, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_bypass_experiment_fingerprint(cfg: DictConfig) -> str:
    """Return a stable ID fingerprint for the architecture and learning target.

    Training budget and data settings are deliberately excluded so a longer
    rerun can resume the same architecture from its previous final checkpoint.
    The full config fingerprint is still recorded in bypass_state.json and used
    for skip-if-complete decisions.
    """
    identity = get_bypass_run_identity(cfg)
    experiment_identity = {
        "model": identity["model"],
        "model_factory": {
            "factory": identity["model_factory"]["factory"],
            "block_loss_func": identity["model_factory"]["block_loss_func"],
            "keys_to_learn": identity["model_factory"]["keys_to_learn"],
            "gqa_init_mode": identity["model_factory"]["gqa_init_mode"],
            "mlp_init_mode": identity["model_factory"]["mlp_init_mode"],
            "mlp_init_config": identity["model_factory"]["mlp_init_config"],
            "linear_init_mode": identity["model_factory"]["linear_init_mode"],
            "submodule_for_loss_calculation": identity["model_factory"][
                "submodule_for_loss_calculation"
            ],
        },
    }
    payload = json.dumps(experiment_identity, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def set_experiment_id(cfg: DictConfig) -> None:
    """Set the experiment ID based on the model config overrides.

    The ID encodes every override that affects the produced student so that
    sweeps over (FFN size × KV heads) or (num_experts × KV heads) get distinct
    directories instead of clobbering each other.
    """
    if cfg.bypass.experiment_id is not None:
        return

    overrides = cfg.bypass.model.model_config_overrides
    parts: list[str] = []

    if "ffn" in overrides:
        ffn_override = overrides.ffn[0]
        if "intermediate_size" in ffn_override and ffn_override["intermediate_size"] is not None:
            parts.append(f"ffn_{ffn_override['intermediate_size']}")
        elif "moe" in ffn_override and ffn_override["moe"] is not None:
            parts.append(f"experts_{ffn_override['moe']['num_local_experts']}")

    if "attention" in overrides:
        attn_override = overrides.attention[0]
        if (
            "num_key_value_heads" in attn_override
            and attn_override["num_key_value_heads"] is not None
        ):
            parts.append(f"heads_{attn_override['num_key_value_heads']}")

    keys_to_learn = cfg.bypass.model_factory.get("keys_to_learn", None)
    if keys_to_learn not in (None, "entire_block"):
        parts.append(_slug(keys_to_learn))

    if not parts:
        parts.append("custom")

    # Keep the readable architecture prefix, but suffix it with the config
    # fingerprint so two runs with the same architecture but different learning
    # target or training budget cannot collide in the same experiment_dir.
    cfg.bypass.experiment_id = "bypass_" + "_".join(parts)
    cfg.bypass.experiment_id += f"_{get_bypass_experiment_fingerprint(cfg)[:8]}"


def set_experiment_dir(cfg: DictConfig) -> None:
    """Set the experiment directory for the bypass run.

    Stores the path as a string in the OmegaConf node (OmegaConf only supports
    primitive types natively). Use sites should reconstruct ``Path(...)`` as needed.
    """
    experiment_dir = Path(cfg.puzzle_dir) / "bypass" / "bypass_runs" / cfg.bypass.experiment_id
    cfg.bypass.experiment_dir = str(experiment_dir)
    if dist.is_master():
        experiment_dir.mkdir(parents=True, exist_ok=True)


def get_bypass_state_path(experiment_dir: str | Path) -> Path:
    return Path(experiment_dir) / BYPASS_STATE_FILENAME


def load_bypass_state(experiment_dir: str | Path) -> dict[str, Any] | None:
    state_path = get_bypass_state_path(experiment_dir)
    if not state_path.exists():
        return None
    return json_load(state_path)


def write_bypass_state(cfg: DictConfig, state: dict[str, Any]) -> None:
    if not dist.is_master():
        return
    json_dump(state, get_bypass_state_path(cfg.bypass.experiment_dir))


def _base_bypass_state(cfg: DictConfig) -> dict[str, Any]:
    return {
        "version": 1,
        "experiment_id": cfg.bypass.get("experiment_id", None),
        "config_fingerprint": get_bypass_config_fingerprint(cfg),
        "identity": get_bypass_run_identity(cfg),
        "status": "running",
        "checkpoints": {},
        "realized_checkpoint": None,
        "ckpts_symlink": None,
    }


def update_bypass_checkpoint_state(
    cfg: DictConfig, checkpoint_dir: str | Path, checkpoint_role: str
) -> None:
    if not dist.is_master():
        return
    state = load_bypass_state(cfg.bypass.experiment_dir) or _base_bypass_state(cfg)
    state["status"] = "running"
    state["config_fingerprint"] = get_bypass_config_fingerprint(cfg)
    state["identity"] = get_bypass_run_identity(cfg)
    state.setdefault("checkpoints", {})[checkpoint_role] = str(Path(checkpoint_dir))
    write_bypass_state(cfg, state)


def mark_bypass_run_completed(
    cfg: DictConfig, realized_checkpoint: str | Path, ckpts_symlink: str | Path
) -> None:
    state = load_bypass_state(cfg.bypass.experiment_dir) or _base_bypass_state(cfg)
    state["status"] = "completed"
    state["config_fingerprint"] = get_bypass_config_fingerprint(cfg)
    state["identity"] = get_bypass_run_identity(cfg)
    state["realized_checkpoint"] = str(realized_checkpoint)
    state["ckpts_symlink"] = str(ckpts_symlink)
    write_bypass_state(cfg, state)
    if dist.is_master():
        (Path(cfg.bypass.experiment_dir) / "_DONE").touch()


def bypass_run_is_complete(cfg: DictConfig) -> bool:
    state = load_bypass_state(cfg.bypass.experiment_dir)
    if state is None:
        return False
    if state.get("status") != "completed":
        return False
    if state.get("config_fingerprint") != get_bypass_config_fingerprint(cfg):
        return False
    realized = state.get("realized_checkpoint")
    symlink = state.get("ckpts_symlink")
    if not realized or not Path(realized).exists():
        return False
    if not symlink or not Path(symlink).exists():
        return False
    return True


def expected_bypass_runs(cfg: DictConfig) -> list[dict[str, Any]]:
    """Return expected run metadata for the current bypass config or sweep."""
    runs: list[dict[str, Any]] = []
    configs_list = cfg.bypass.get("configs", None)
    overrides = configs_list if configs_list else [None]

    for override in overrides:
        run_cfg = OmegaConf.create(
            {
                "puzzle_dir": cfg.puzzle_dir,
                "dataset_path": cfg.get("dataset_path", None),
                "descriptor": cfg.get("descriptor", None),
                "bypass": OmegaConf.to_container(cfg.bypass, resolve=True),
            }
        )
        OmegaConf.set_struct(run_cfg, False)
        if override:
            run_cfg.bypass.experiment_id = None
            if "model_config_overrides" in override:
                run_cfg.bypass.model.model_config_overrides = override.model_config_overrides
            if "keys_to_learn" in override:
                run_cfg.bypass.model_factory.keys_to_learn = override.keys_to_learn
        set_experiment_id(run_cfg)
        experiment_dir = (
            Path(run_cfg.puzzle_dir) / "bypass" / "bypass_runs" / run_cfg.bypass.experiment_id
        )
        runs.append(
            {
                "experiment_id": run_cfg.bypass.experiment_id,
                "experiment_dir": str(experiment_dir),
                "config_fingerprint": get_bypass_config_fingerprint(run_cfg),
            }
        )
    return runs


def get_distributed_modules_ownership(module_count: int, world_size: int) -> list[int]:
    """Map module (block) indices to GPU ranks for pipeline-parallel distribution."""
    modules_process_ownership: list[int] = []

    for i in range(world_size):
        num_modules_for_process = module_count // world_size
        if i < module_count % world_size:
            num_modules_for_process += 1

        modules_process_ownership.extend([i] * num_modules_for_process)

    return modules_process_ownership


def get_pipeline_ownership_context(
    module_ownership: Sequence[int], rank: int | None = None
) -> dict[str, Any]:
    """Return local module indices and neighboring pipeline ranks for ``rank``."""
    if rank is None:
        rank = dist.rank()
    owned_indices = [i for i, owner in enumerate(module_ownership) if owner == rank]
    if not owned_indices:
        raise RuntimeError(
            f"rank {rank} owns no modules in pipeline ownership map {list(module_ownership)}"
        )

    min_owned_index = min(owned_indices)
    max_owned_index = max(owned_indices)
    prev_rank = None if min_owned_index == 0 else module_ownership[min_owned_index - 1]
    next_rank = (
        None
        if max_owned_index + 1 >= len(module_ownership)
        else module_ownership[max_owned_index + 1]
    )
    return {
        "owned_indices": owned_indices,
        "owned_index_set": set(owned_indices),
        "prev_rank": prev_rank,
        "next_rank": next_rank,
    }
