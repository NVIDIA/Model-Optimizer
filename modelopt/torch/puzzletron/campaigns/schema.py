"""Public contracts for stage-by-stage, cross-model Puzzletron campaigns."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class DatasetKind(str, Enum):
    """Canonical datasets whose concrete settings are owned by campaign configs."""

    PINNED_INTERSYN = "pinned_intersyn"
    PUZZLE_KD_TEXT = "puzzle_kd_text"


class ModelKind(str, Enum):
    DENSE = "dense"
    MOE = "moe"


@dataclass(frozen=True)
class ParallelTopology:
    """Logical model and data parallel dimensions for one model stage."""

    tp: int
    cp: int
    pp: int
    fsdp: int
    ep: int

    @property
    def world_size(self) -> int:
        return self.tp * self.cp * self.pp * self.fsdp * self.ep

    def validate(self) -> None:
        dimensions = dataclasses.asdict(self)
        invalid = {name: value for name, value in dimensions.items() if value < 1}
        if invalid:
            raise ValueError(f"Parallel dimensions must be positive: {invalid}")


DENSE_TOPOLOGY = ParallelTopology(tp=2, cp=2, pp=2, fsdp=2, ep=1)
MOE_TOPOLOGY = ParallelTopology(tp=2, cp=2, pp=2, fsdp=1, ep=2)


@dataclass(frozen=True)
class CampaignModel:
    """One immutable model entry in the cross-family campaign."""

    model_id: str
    hf_id: str
    hf_revision: str
    model_kind: ModelKind
    is_multimodal: bool
    dataset: DatasetKind
    topology: ParallelTopology
    topology_exception: str | None = None
    force_hf: bool = False
    expect_native_automodel: bool = True
    mtp_policy: str = "if_present"
    elastic_no_op_subblocks: tuple[str, ...] = ()

    def validate(self) -> None:
        self.topology.validate()
        if self.force_hf:
            raise ValueError(f"{self.model_id}: cross-model campaigns require force_hf=False")
        if self.model_kind is ModelKind.MOE:
            if self.topology.ep != 2 or self.topology.fsdp != 1:
                raise ValueError(f"{self.model_id}: MoE campaigns require EP=2 and FSDP=1")
        elif self.topology.ep != 1 or self.topology.fsdp != 2:
            raise ValueError(f"{self.model_id}: dense campaigns require EP=1 and FSDP=2")
        if self.topology_exception is None and self.topology.world_size != 16:
            raise ValueError(
                f"{self.model_id}: the two-node campaign requires 16 ranks, got "
                f"{self.topology.world_size}"
            )
        if self.topology_exception is not None:
            if not self.topology_exception.strip():
                raise ValueError(f"{self.model_id}: topology_exception must explain the exception")
            if self.topology.world_size > 16:
                raise ValueError(f"{self.model_id}: exception topology cannot exceed 16 ranks")
        if self.is_multimodal and self.dataset is not DatasetKind.PINNED_INTERSYN:
            raise ValueError(f"{self.model_id}: multimodal models require the pinned InterSyn dataset")
        if not self.is_multimodal and self.dataset is not DatasetKind.PUZZLE_KD_TEXT:
            raise ValueError(f"{self.model_id}: text-only models require the Puzzle-KD text dataset")
        if self.mtp_policy != "if_present":
            raise ValueError(f"{self.model_id}: MTP policy must be 'if_present'")
        if len(self.elastic_no_op_subblocks) != len(set(self.elastic_no_op_subblocks)):
            raise ValueError(
                f"{self.model_id}: elastic_no_op_subblocks must not contain duplicates"
            )
        invalid_no_ops = set(self.elastic_no_op_subblocks) - {
            "attention",
            "ffn",
            "gdn",
            "mamba",
            "moe",
        }
        if invalid_no_ops:
            raise ValueError(
                f"{self.model_id}: unsupported elastic no-op subblocks: "
                f"{sorted(invalid_no_ops)}"
            )

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["model_kind"] = self.model_kind.value
        value["dataset"] = self.dataset.value
        return value


@dataclass(frozen=True)
class CrossModelCampaign:
    """An ordered set of models evaluated one stage at a time."""

    models: tuple[CampaignModel, ...]
    sequence_length: int = 2048
    activation_samples: int = 16
    kd_steps: int = 8
    data_layout: str = "packed_varlen"

    def validate(self) -> None:
        if not self.models:
            raise ValueError("A cross-model campaign must contain at least one model")
        model_ids = [model.model_id for model in self.models]
        hf_ids = [model.hf_id for model in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("Campaign model_id values must be unique")
        if len(hf_ids) != len(set(hf_ids)):
            raise ValueError("Campaign hf_id values must be unique")
        if self.sequence_length < 1 or self.activation_samples < 1 or self.kd_steps < 1:
            raise ValueError("Sequence length, activation samples, and KD steps must be positive")
        if self.data_layout not in {"fixed", "padded_varlen", "packed_varlen"}:
            raise ValueError(f"Unsupported campaign data layout: {self.data_layout}")
        for model in self.models:
            model.validate()

    @property
    def fingerprint(self) -> str:
        payload = {
            "models": [model.to_dict() for model in self.models],
            "sequence_length": self.sequence_length,
            "activation_samples": self.activation_samples,
            "kd_steps": self.kd_steps,
            "data_layout": self.data_layout,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CampaignStageIdentity:
    """Content identity for one model/stage execution and its dependencies."""

    campaign_fingerprint: str
    model_id: str
    stage: str
    upstream_identities: tuple[str, ...]

    @classmethod
    def create(
        cls,
        campaign: CrossModelCampaign,
        *,
        model_id: str,
        stage: str,
        upstream_identities: tuple[str, ...] = (),
    ) -> "CampaignStageIdentity":
        if model_id not in {model.model_id for model in campaign.models}:
            raise ValueError(f"Unknown campaign model_id: {model_id}")
        if not stage.strip():
            raise ValueError("Campaign stage must not be empty")
        return cls(
            campaign_fingerprint=campaign.fingerprint,
            model_id=model_id,
            stage=stage,
            upstream_identities=tuple(upstream_identities),
        )

    @property
    def fingerprint(self) -> str:
        payload = dataclasses.asdict(self)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_campaign(path: str | Path) -> CrossModelCampaign:
    """Load and validate a campaign YAML file without accepting implicit defaults."""

    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Campaign config must be a mapping")
    required_top = {"models", "sequence_length", "activation_samples", "kd_steps"}
    missing = required_top - raw.keys()
    if missing:
        raise ValueError(f"Campaign config is missing required keys: {sorted(missing)}")

    models = []
    for index, item in enumerate(raw["models"]):
        try:
            topology = ParallelTopology(**item["topology"])
            models.append(
                CampaignModel(
                    model_id=item["model_id"],
                    hf_id=item["hf_id"],
                    hf_revision=item["hf_revision"],
                    model_kind=ModelKind(item["model_kind"]),
                    is_multimodal=bool(item["is_multimodal"]),
                    dataset=DatasetKind(item["dataset"]),
                    topology=topology,
                    topology_exception=item.get("topology_exception"),
                    force_hf=bool(item["force_hf"]),
                    expect_native_automodel=bool(item["expect_native_automodel"]),
                    mtp_policy=item["mtp_policy"],
                    elastic_no_op_subblocks=tuple(
                        str(kind) for kind in item.get("elastic_no_op_subblocks", ())
                    ),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid campaign model at index {index}: {error}") from error

    campaign = CrossModelCampaign(
        models=tuple(models),
        sequence_length=int(raw["sequence_length"]),
        activation_samples=int(raw["activation_samples"]),
        kd_steps=int(raw["kd_steps"]),
        data_layout=str(raw.get("data_layout", "packed_varlen")),
    )
    campaign.validate()
    return campaign


def _model(
    model_id: str,
    hf_id: str,
    *,
    kind: ModelKind,
    multimodal: bool,
    expect_native_automodel: bool = True,
    topology: ParallelTopology | None = None,
    topology_exception: str | None = None,
    elastic_no_op_subblocks: tuple[str, ...] = (),
) -> CampaignModel:
    return CampaignModel(
        model_id=model_id,
        hf_id=hf_id,
        # Preflight resolves and records the immutable Hub commit before conversion.
        hf_revision="main",
        model_kind=kind,
        is_multimodal=multimodal,
        dataset=DatasetKind.PINNED_INTERSYN if multimodal else DatasetKind.PUZZLE_KD_TEXT,
        topology=topology or (MOE_TOPOLOGY if kind is ModelKind.MOE else DENSE_TOPOLOGY),
        topology_exception=topology_exception,
        expect_native_automodel=expect_native_automodel,
        elastic_no_op_subblocks=elastic_no_op_subblocks,
    )


def default_cross_model_campaign() -> CrossModelCampaign:
    """Return the approved five-model acceptance matrix."""

    campaign = CrossModelCampaign(
        models=(
            _model(
                "qwen35_dense",
                "Qwen/Qwen3.5-0.8B",
                kind=ModelKind.DENSE,
                multimodal=True,
            ),
            _model(
                "nemotron3_nano_30b_a3b",
                "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                kind=ModelKind.MOE,
                multimodal=False,
                topology=ParallelTopology(tp=1, cp=1, pp=2, fsdp=1, ep=2),
                topology_exception=(
                    "Native AutoModel NemotronH has no TP plan and its non-TE CP path is unsupported"
                ),
            ),
            _model(
                "gpt_oss_20b",
                "openai/gpt-oss-20b",
                kind=ModelKind.MOE,
                multimodal=False,
                topology=ParallelTopology(tp=1, cp=1, pp=2, fsdp=1, ep=2),
                topology_exception=(
                    "Native AutoModel GPT-OSS has no TP plan and non-TE CP is unsupported"
                ),
                elastic_no_op_subblocks=("attention", "moe"),
            ),
            _model(
                "qwen36_35b_a3b",
                "Qwen/Qwen3.6-35B-A3B",
                kind=ModelKind.MOE,
                multimodal=True,
                topology=ParallelTopology(tp=1, cp=1, pp=2, fsdp=1, ep=2),
                topology_exception=(
                    "Native AutoModel custom Qwen3.6 MoE has no TP support and its CP path requires Transformer Engine"
                ),
            ),
            _model(
                "llama31_8b",
                "meta-llama/Llama-3.1-8B-Instruct",
                kind=ModelKind.DENSE,
                multimodal=False,
                topology=ParallelTopology(tp=2, cp=1, pp=2, fsdp=2, ep=1),
                topology_exception="AutoModel Llama context parallelism requires Transformer Engine",
            ),
        )
    )
    campaign.validate()
    return campaign
