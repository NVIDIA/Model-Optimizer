"""Metadata-only preflight contracts for cross-model campaigns."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import CampaignModel, CrossModelCampaign


@dataclass(frozen=True)
class ModelMetadata:
    """Resolved metadata; obtaining it must never instantiate model weights."""

    immutable_revision: str
    architectures: tuple[str, ...]
    model_type: str
    selected_model_class: str
    native_automodel: bool
    descriptor_name: str | None
    tokenizer_available: bool
    processor_available: bool
    nested_text_config: dict[str, Any] | None = None
    mtp_fields: tuple[str, ...] = ()
    probe_errors: tuple[str, ...] = ()
    parallel_support: dict[str, bool] | None = None
    axis_score_methods: dict[str, str] | None = None


@dataclass(frozen=True)
class ModelPreflight:
    model_id: str
    hf_id: str
    immutable_revision: str | None
    architectures: tuple[str, ...]
    model_type: str | None
    selected_model_class: str | None
    native_automodel: bool | None
    descriptor_name: str | None
    tokenizer_available: bool
    processor_available: bool
    nested_text_config: dict[str, Any] | None
    mtp_fields: tuple[str, ...]
    parallel_support: dict[str, bool] | None
    axis_score_methods: dict[str, str] | None
    topology: dict[str, int]
    errors: tuple[str, ...]

    @property
    def success(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class CampaignPreflight:
    campaign_fingerprint: str
    models: tuple[ModelPreflight, ...]

    @property
    def success(self) -> bool:
        return all(model.success for model in self.models)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_fingerprint": self.campaign_fingerprint,
            "success": self.success,
            "models": [model.to_dict() for model in self.models],
        }


MetadataLoader = Callable[[CampaignModel], ModelMetadata]


def _config_architectures(config: Any) -> tuple[str, ...]:
    architectures = getattr(config, "architectures", None)
    if not architectures:
        architectures = getattr(getattr(config, "text_config", None), "architectures", None)
    if isinstance(architectures, str):
        return (architectures,)
    return tuple(architectures or ())


def _find_mtp_fields(config_dict: dict[str, Any]) -> tuple[str, ...]:
    fields: list[str] = []

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                path = f"{prefix}.{key_text}" if prefix else key_text
                key_lower = key_text.lower()
                if "mtp" in key_lower or "nextn" in key_lower or "multi_token" in key_lower:
                    fields.append(path)
                visit(child, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{prefix}[{index}]")

    visit(config_dict, "")
    return tuple(sorted(set(fields)))


def _hf_model_class_name(config: Any, *, multimodal: bool) -> str:
    from transformers import AutoModelForCausalLM

    factories: list[Any] = []
    if multimodal:
        try:
            from transformers import AutoModelForImageTextToText

            factories.append(AutoModelForImageTextToText)
        except ImportError:
            pass
        try:
            from transformers import AutoModelForVision2Seq

            factories.append(AutoModelForVision2Seq)
        except ImportError:
            pass
    factories.append(AutoModelForCausalLM)
    errors: list[str] = []
    for factory in factories:
        try:
            model_class = factory._model_mapping[type(config)]
            return f"{model_class.__module__}.{model_class.__name__}"
        except Exception as error:
            errors.append(f"{factory.__name__}: {error}")
    raise ValueError("No HF fallback model class supports the config: " + "; ".join(errors))


def probe_huggingface_metadata(model: CampaignModel) -> ModelMetadata:
    """Resolve Hub, AutoModel, processor, and descriptor metadata without loading weights."""

    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    from ..anymodel.registry import resolve_descriptor

    # Importing the native registry eagerly installs compatibility AutoConfig
    # names before Transformers reads config.json.
    try:
        from nemo_automodel._transformers.registry import ModelRegistry
    except ImportError:
        ModelRegistry = None

    info = HfApi().model_info(model.hf_id, revision=model.hf_revision)
    immutable_revision = str(info.sha)
    try:
        config = AutoConfig.from_pretrained(
            model.hf_id,
            revision=immutable_revision,
            trust_remote_code=True,
        )
    except Exception as error:
        return ModelMetadata(
            immutable_revision=immutable_revision,
            architectures=(),
            model_type="",
            selected_model_class="",
            native_automodel=False,
            descriptor_name=None,
            tokenizer_available=False,
            processor_available=False,
            probe_errors=(f"config: {type(error).__name__}: {error}",),
        )
    config_dict = config.to_dict()
    architectures = _config_architectures(config)
    probe_errors: list[str] = []

    native_model_class = None
    try:
        if ModelRegistry is None:
            raise ImportError("nemo_automodel registry is unavailable")
        for architecture in architectures:
            native_model_class = ModelRegistry.resolve_custom_model_cls(architecture, config)
            if native_model_class is not None:
                break
    except ImportError as error:
        probe_errors.append(f"native registry: {type(error).__name__}: {error}")
        native_model_class = None

    if native_model_class is not None:
        selected_model_class = f"{native_model_class.__module__}.{native_model_class.__name__}"
        native_automodel = True
    else:
        try:
            selected_model_class = _hf_model_class_name(config, multimodal=model.is_multimodal)
        except Exception as error:
            selected_model_class = ""
            probe_errors.append(f"model class: {type(error).__name__}: {error}")
        native_automodel = False

    parallel_support = None
    if native_model_class is not None and hasattr(native_model_class, "get_capabilities"):
        try:
            model_capabilities = native_model_class.get_capabilities(config)
            parallel_support = {
                name: bool(getattr(model_capabilities, f"supports_{name}"))
                for name in ("tp", "cp", "pp", "ep")
                if hasattr(model_capabilities, f"supports_{name}")
            }
        except Exception as error:
            probe_errors.append(f"parallel capabilities: {type(error).__name__}: {error}")

    # Loading tokenizer/processor assets is intentional: it validates gated access and
    # custom processor code, while never constructing or downloading model weights.
    tokenizer_available = True
    try:
        AutoTokenizer.from_pretrained(
            model.hf_id,
            revision=immutable_revision,
            trust_remote_code=True,
        )
    except Exception as error:
        tokenizer_available = False
        probe_errors.append(f"tokenizer: {type(error).__name__}: {error}")
    processor_available = False
    if model.is_multimodal:
        try:
            AutoProcessor.from_pretrained(
                model.hf_id,
                revision=immutable_revision,
                trust_remote_code=True,
            )
            processor_available = True
        except Exception as error:
            probe_errors.append(f"processor: {type(error).__name__}: {error}")

    descriptor_name = None
    axis_score_methods = None
    try:
        from ..anymodel.capabilities import resolve_score_method

        descriptor_resolution = resolve_descriptor(config)
        descriptor_name = descriptor_resolution.name
        axis_score_methods = {
            axis_id: resolve_score_method(axis)
            for axis_id, axis in descriptor_resolution.capabilities.axes.items()
            if axis.sortable and not axis.variant_only
        }
    except Exception as error:
        probe_errors.append(f"descriptor: {type(error).__name__}: {error}")
    text_config = getattr(config, "text_config", None)
    nested_text_config = text_config.to_dict() if hasattr(text_config, "to_dict") else None
    return ModelMetadata(
        immutable_revision=immutable_revision,
        architectures=architectures,
        model_type=str(getattr(config, "model_type", "")),
        selected_model_class=selected_model_class,
        native_automodel=native_automodel,
        descriptor_name=descriptor_name,
        tokenizer_available=tokenizer_available,
        processor_available=processor_available,
        nested_text_config=nested_text_config,
        mtp_fields=_find_mtp_fields(config_dict),
        probe_errors=tuple(probe_errors),
        parallel_support=parallel_support,
        axis_score_methods=axis_score_methods,
    )


def _validate_metadata(model: CampaignModel, metadata: ModelMetadata) -> tuple[str, ...]:
    errors: list[str] = list(metadata.probe_errors)
    if not metadata.immutable_revision or metadata.immutable_revision == "main":
        errors.append("metadata probe did not resolve an immutable revision")
    if not metadata.architectures:
        errors.append("checkpoint config contains no architecture")
    if not metadata.selected_model_class:
        errors.append("AutoModel/HF fallback model class was not resolved")
    if not metadata.descriptor_name:
        errors.append("Puzzletron descriptor could not be resolved")
    if not metadata.tokenizer_available:
        errors.append("tokenizer metadata is unavailable")
    if model.is_multimodal and not metadata.processor_available:
        errors.append("multimodal processor metadata is unavailable")
    if model.expect_native_automodel and not metadata.native_automodel:
        errors.append("expected a native AutoModel implementation but selected the HF fallback")
    if not model.expect_native_automodel and metadata.native_automodel:
        errors.append("expected the documented HF fallback but selected a native AutoModel class")
    if metadata.parallel_support is not None:
        for name in ("tp", "cp", "pp", "ep"):
            requested = int(getattr(model.topology, name))
            if requested > 1 and metadata.parallel_support.get(name) is False:
                errors.append(
                    f"requested {name.upper()}={requested}, but the selected model class "
                    f"declares {name.upper()} unsupported"
                )
    return tuple(errors)


def run_preflight(campaign: CrossModelCampaign, load_metadata: MetadataLoader) -> CampaignPreflight:
    """Resolve every model independently and preserve all failures in one report."""

    campaign.validate()
    records: list[ModelPreflight] = []
    for model in campaign.models:
        metadata: ModelMetadata | None = None
        try:
            metadata = load_metadata(model)
            errors = _validate_metadata(model, metadata)
        except Exception as error:  # Preflight is an audit: retain all access/resolution failures.
            errors = (f"{type(error).__name__}: {error}",)

        records.append(
            ModelPreflight(
                model_id=model.model_id,
                hf_id=model.hf_id,
                immutable_revision=metadata.immutable_revision if metadata else None,
                architectures=metadata.architectures if metadata else (),
                model_type=metadata.model_type if metadata else None,
                selected_model_class=metadata.selected_model_class if metadata else None,
                native_automodel=metadata.native_automodel if metadata else None,
                descriptor_name=metadata.descriptor_name if metadata else None,
                tokenizer_available=metadata.tokenizer_available if metadata else False,
                processor_available=metadata.processor_available if metadata else False,
                nested_text_config=metadata.nested_text_config if metadata else None,
                mtp_fields=metadata.mtp_fields if metadata else (),
                parallel_support=metadata.parallel_support if metadata else None,
                axis_score_methods=metadata.axis_score_methods if metadata else None,
                topology=dataclasses.asdict(model.topology),
                errors=errors,
            )
        )
    return CampaignPreflight(campaign_fingerprint=campaign.fingerprint, models=tuple(records))


def write_preflight(result: CampaignPreflight, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")


def load_preflight(path: str | Path) -> CampaignPreflight:
    raw = json.loads(Path(path).read_text())
    records = []
    for item in raw.get("models", []):
        item = dict(item)
        item["architectures"] = tuple(item.get("architectures") or ())
        item["mtp_fields"] = tuple(item.get("mtp_fields") or ())
        item["errors"] = tuple(item.get("errors") or ())
        item.setdefault("parallel_support", None)
        item.setdefault("axis_score_methods", None)
        records.append(ModelPreflight(**item))
    return CampaignPreflight(
        campaign_fingerprint=raw["campaign_fingerprint"],
        models=tuple(records),
    )
