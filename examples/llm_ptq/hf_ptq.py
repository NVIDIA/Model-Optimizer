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
import copy
import random
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate.hooks import remove_hook_from_module
from cast_mxfp4_to_nvfp4 import apply_to_model as apply_cast_mxfp4_to_nvfp4
from cast_mxfp4_to_nvfp4 import force_weight_quantizers_static
from example_utils import (
    build_quant_cfg,
    copy_custom_model_files,
    create_vlm_calibration_loop,
    get_model,
    get_processor,
    get_tokenizer,
    is_enc_dec,
    is_nemotron_vl,
    load_mtp_weights,
    needs_checkpoint_path_update,
    resolve_checkpoint_dir,
    run_nemotron_vl_preview,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    ProcessorMixin,
    WhisperProcessor,
)

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import modelopt.torch.sparsity as mts
from modelopt.recipe import ModelOptPTQRecipe, load_recipe
from modelopt.torch.export import (
    export_hf_checkpoint,
    export_hf_vllm_fq_checkpoint,
    export_speculative_decoding,
    export_tensorrt_llm_checkpoint,
    get_model_type,
    has_spec_opt,
    save_expert_token_count_table,
)
from modelopt.torch.export.model_utils import get_language_model_from_vl, is_multimodal_model
from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg, need_calibration
from modelopt.torch.quantization.plugins.accelerate import init_quantized_weights
from modelopt.torch.quantization.utils import is_quantized
from modelopt.torch.speculative.eagle.utils import (
    EagleOfflineDataCollator,
    OfflineSupervisedDataset,
)
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
    get_max_batch_size,
    get_supported_datasets,
)
from modelopt.torch.utils.memory_monitor import launch_memory_monitor
from modelopt.torch.utils.speech_dataset_utils import get_speech_dataset_dataloader
from modelopt.torch.utils.vlm_dataset_utils import get_vlm_dataset_dataloader

RAND_SEED = 1234


def _set_kv_cache_constant_amax(quant_cfg: list) -> None:
    """Set use_constant_amax on KV cache quantizers.

    Creates a new dict for the KV bmm quantizer config to avoid mutating shared references.
    """
    for i, entry in enumerate(quant_cfg):
        if entry.get("quantizer_name") != "*[kv]_bmm_quantizer":
            continue
        cfg = entry.get("cfg") or {}
        assert isinstance(cfg, dict)
        quant_cfg[i] = {**entry, "cfg": {**cfg, "use_constant_amax": True}}
        break


QUANT_CFG_CHOICES: dict[str, dict[str, Any]] = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "int8_wo": mtq.INT8_WEIGHT_ONLY_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,
    "nvfp4_mse": mtq.NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG,
    "fp8_pb_wo": mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    "fp8_pc_pt": mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG,
    "w4a8_nvfp4_fp8": mtq.W4A8_NVFP4_FP8_CFG,
    "w4a8_mxfp4_fp8": mtq.W4A8_MXFP4_FP8_CFG,
    "nvfp4_mlp_only": mtq.NVFP4_MLP_ONLY_CFG,
    "nvfp4_experts_only": mtq.NVFP4_EXPERTS_ONLY_CFG,
    "nvfp4_omlp_only": mtq.NVFP4_OMLP_ONLY_CFG,
    "nvfp4_svdquant": mtq.NVFP4_SVDQUANT_DEFAULT_CFG,
    "mxfp8": mtq.MXFP8_DEFAULT_CFG,
    "nvfp4_local_hessian": mtq.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG,
}

KV_QUANT_CFG_CHOICES = {
    "none": "none",
    "fp8_cast": "FP8_KV_CFG",
    "fp8": "FP8_KV_CFG",
    "fp8_affine": "FP8_AFFINE_KV_CFG",
    "nvfp4_cast": "NVFP4_KV_CFG",
    "nvfp4": "NVFP4_KV_CFG",
    "nvfp4_affine": "NVFP4_AFFINE_KV_CFG",
    "nvfp4_rotate": "NVFP4_KV_ROTATE_CFG",
}

# Formats that use use_constant_amax (no calibration needed).
_KV_CAST_FORMATS = {"fp8_cast", "nvfp4_cast"}

mto.enable_huggingface_checkpointing()


def extract_and_prepare_language_model_from_vl(full_model):
    """Extract language model from VL model and disable quantization for non-language components.

    Args:
        full_model: The full VLM model

    Returns:
        tuple: (language_model, model_type) or (None, None) if not a VLM
    """
    language_model_lineage = get_language_model_from_vl(full_model)
    if language_model_lineage is not None:
        language_model = language_model_lineage.pop(-1)
        ancestors = language_model_lineage
        # Apply disabled quant to all modules that are not part of language_model
        # This excludes them during HF export
        disabled_quant_cfg = {
            "quant_cfg": [{"quantizer_name": "*", "enable": False}],
            "algorithm": "max",
        }

        memo = set(ancestors) | {language_model}
        for ancestor in ancestors:
            for _, module in ancestor.named_children():
                if module not in memo:
                    mtq.quantize(module, disabled_quant_cfg, forward_loop=None)
                    memo.add(module)

        model_type = get_model_type(language_model)
        return language_model, model_type

    return None, None


class _DeviceDataLoader:
    """Wrapper around a DataLoader that moves each batch to a target device."""

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            yield _move_batch_to_device(batch, self.device)

    def __len__(self):
        return len(self.dataloader)


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Recursively move all tensors in a batch dict to the given device."""

    def _to_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: _to_device(v) for k, v in value.items()}
        return value

    return {k: _to_device(v) for k, v in batch.items()}


def make_calib_dataloader(
    args: argparse.Namespace,
    language_model: torch.nn.Module,
    processor: ProcessorMixin | None,
    tokenizer: PreTrainedTokenizerBase | None,
    device: torch.device,
    model_type: str | None,
) -> tuple[DataLoader | _DeviceDataLoader, str | None]:
    calib_dataloader = None
    first_text_speech_dataset = None
    if args.specdec_offline_dataset is not None:
        offline_data_path = Path(args.specdec_offline_dataset)
        dumped_files = sorted(str(p) for p in offline_data_path.glob("*.pt"))
        if not dumped_files:
            raise ValueError(f"No .pt files found in {args.specdec_offline_dataset}")
        if args.calib_size[0] > 0:
            dumped_files = dumped_files[: args.calib_size[0]]
        dataset = OfflineSupervisedDataset(dumped_files)
        collator = EagleOfflineDataCollator(train_len=args.calib_seq)
        raw_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        # Wrap to move batches to the target device; device-transfer logic is kept
        # out of the data collator to avoid interference with dataloader prefetching.
        calib_dataloader = _DeviceDataLoader(raw_loader, device)
    elif args.calib_with_images:
        # VLM image-text calibration path: assume Nemotron VLM dataset by default.
        assert processor is not None, (
            "Please provide a processor (e.g., AutoProcessor) for image calibration."
        )
        assert len(args.calib_size) == 1, (
            "Image calibration currently supports a single dataset. "
            "Please pass --calib_size with one value (e.g., --calib_size 256)."
        )
        calib_dataloader = get_vlm_dataset_dataloader(
            dataset_name="nemotron_vlm_dataset_v2",
            processor=processor,
            batch_size=args.batch_size,
            num_samples=args.calib_size[0],
            device=device,
            max_length=args.calib_seq,
            require_image=True,
            subsets=["sparsetables", "plotqa_cot", "wiki_en"],
            shuffle_buffer_size=10_000,
            seed=42,
            use_media_shards=True,
            max_shards=1,
        )
    elif model_type == "whisper":
        assert processor is not None and isinstance(processor, WhisperProcessor), (
            "The AutoProcessor must be set."
        )
        assert len(args.calib_size) == 1, (
            "whisper only supports one dataset for calibration, can extend this in the future"
        )
        calib_dataloader, first_text_speech_dataset = get_speech_dataset_dataloader(
            dataset_name=args.dataset[0] if args.dataset else "peoples_speech",
            processor=processor,
            batch_size=args.batch_size,
            num_samples=args.calib_size[0],
            device=device,
            dtype=language_model.dtype,
        )
    else:
        assert tokenizer is not None and isinstance(
            tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)
        ), "The PreTrainedTokenizer must be set"
        # Labels are only needed for gradient-based auto_quantize
        include_labels = (
            args.auto_quantize_bits is not None and args.auto_quantize_method == "gradient"
        )

        calib_dataloader = get_dataset_dataloader(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_samples=args.calib_size,
            max_sample_length=args.calib_seq,
            device=device,
            include_labels=include_labels,
        )
    return calib_dataloader, first_text_speech_dataset


def auto_quantize(
    args: argparse.Namespace,
    language_model: torch.nn.Module,
    calib_dataloader: DataLoader,
    auto_quantize_method="gradient",
    auto_quantize_score_size=128,
    auto_quantize_checkpoint=None,
):
    """Auto search quantization of multiple formats."""

    if args.calib_with_images:
        raise NotImplementedError(
            "AutoQuantize with image-text calibration is not supported yet. "
            "Please run plain PTQ (e.g., --qformat nvfp4) with --calib_with_images."
        )

    assert not (args.auto_quantize_bits and args.inference_pipeline_parallel > 1), (
        "Auto Quantization is not supported for pipeline parallel size > 1"
    )

    qformat_list = args.qformat.split(",")
    assert qformat_list, "No quantization formats provided"
    # Check if all provided quantization formats are supported
    assert all(
        qformat
        in [
            "fp8",
            "int8_sq",
            "int8_wo",
            "int4_awq",
            "nvfp4",
            "nvfp4_awq",
            "nvfp4_mse",
            "w4a8_awq",
            "fp8_pb_wo",
            "w4a8_mxfp4_fp8",
            "nvfp4_mlp_only",
            "nvfp4_experts_only",
            "nvfp4_omlp_only",
            "nvfp4_local_hessian",
            "mxfp8",
        ]
        for qformat in qformat_list
    ), "One or more quantization formats provided are not supported for unified checkpoint export"

    def loss_func(output, data):
        # For transformers AutoModelForCausalLM models, the outputs are wrapped in `CausalLMOutputWithPast`
        # which contains the loss attribute.
        return output.loss

    if auto_quantize_method == "gradient":
        # For gradient-based method, return full output with loss
        def forward_step(model, batch):
            return model(**batch)
    elif auto_quantize_method == "kl_div":
        # For KL divergence method, return only logits
        def forward_step(model, batch):
            return model(**batch).logits
    else:
        raise ValueError(
            f"Invalid auto_quantize_method: {auto_quantize_method}. Must be 'gradient' or 'kl_div'"
        )

    language_model, _ = mtq.auto_quantize(
        language_model,
        constraints={"effective_bits": args.auto_quantize_bits},
        data_loader=calib_dataloader,
        forward_step=forward_step,
        loss_func=loss_func,  # Only used for gradient-based method
        # TRTLLM only support one quantization format or None (do not quantize, internally supported)
        quantization_formats=[QUANT_CFG_CHOICES[format] for format in qformat_list],
        num_calib_steps=len(calib_dataloader),
        # AutoQuantize scoring is the costly phase; allow smaller sample counts than calibration.
        num_score_steps=min(
            len(calib_dataloader), max(auto_quantize_score_size // args.batch_size, 1)
        ),
        verbose=True,
        # Disable all default disabled layers such as lm_head, mlp.gate, router etc.
        disabled_layers=[
            entry["quantizer_name"]
            for entry in _default_disabled_quantizer_cfg
            if "parent_class" not in entry
        ],
        method=auto_quantize_method,
        checkpoint=auto_quantize_checkpoint,
    )

    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)
    # We need to explicitly set up KV cache quantization after auto_quantize
    enable_quant_kv_cache = args.kv_cache_qformat != "none"
    print(f"{'Enable' if enable_quant_kv_cache else 'Disable'} KV cache quantization")
    if enable_quant_kv_cache:
        kv_cache_quant_cfg = copy.deepcopy(
            getattr(mtq, KV_QUANT_CFG_CHOICES[args.kv_cache_qformat])["quant_cfg"]
        )
        kv_cache_quant_cfg = [
            e for e in kv_cache_quant_cfg if e["quantizer_name"] != "*"
        ]  # keep other quantizers from auto_quantize

        if args.kv_cache_qformat in _KV_CAST_FORMATS:
            _set_kv_cache_constant_amax(kv_cache_quant_cfg)

        mtq.set_quantizer_by_cfg(language_model, quant_cfg=kv_cache_quant_cfg)
        if args.kv_cache_qformat not in _KV_CAST_FORMATS:
            # Calibrate only the KV cache quantizers; disable all others.
            with mtq.set_quantizer_by_cfg_context(
                language_model,
                [{"quantizer_name": "*", "enable": False}, *kv_cache_quant_cfg],
            ):
                mtq.calibrate(language_model, algorithm="max", forward_loop=calibrate_loop)
    return language_model


def load_model(args: argparse.Namespace):
    # If low memory mode is enabled, we compress the model while loading the HF checkpoint.
    calibration_only = False
    if args.specdec_offline_dataset is not None or not args.low_memory_mode:
        full_model = get_model(
            args.pyt_ckpt_path,
            args.device,
            gpu_mem_percentage=args.gpu_max_mem_percentage,
            trust_remote_code=args.trust_remote_code,
            use_seq_device_map=args.use_seq_device_map,
            attn_implementation=args.attn_implementation,
        )
    else:
        assert args.qformat in QUANT_CFG_CHOICES, (
            f"Quantization format is not supported for low memory mode. Supported formats: {QUANT_CFG_CHOICES.keys()}"
        )
        quant_cfg = QUANT_CFG_CHOICES[args.qformat]
        if args.kv_cache_qformat != "none":
            quant_cfg = mtq.utils.update_quant_cfg_with_kv_cache_quant(
                quant_cfg,
                getattr(mtq, KV_QUANT_CFG_CHOICES[args.kv_cache_qformat])["quant_cfg"],
            )
            # Mirror the use_constant_amax logic from quantize_main so that init_quantized_weights
            # builds the KV quantizers with use_constant_amax already set. In calibration_only mode
            # mtq.calibrate() does not re-apply quant_cfg, so this must happen before
            # init_quantized_weights runs.
            if args.kv_cache_qformat in _KV_CAST_FORMATS:
                quant_cfg = copy.deepcopy(quant_cfg)
                _set_kv_cache_constant_amax(quant_cfg["quant_cfg"])

        # Do not use real quant GEMM so the calibration can be more accurate.
        with init_quantized_weights(
            quant_cfg, gpu_mem_percentage=args.gpu_max_mem_percentage, quant_gemm=False
        ):
            model_kwargs = {"trust_remote_code": args.trust_remote_code}
            if args.attn_implementation is not None:
                model_kwargs["attn_implementation"] = args.attn_implementation
            full_model = AutoModelForCausalLM.from_pretrained(
                args.pyt_ckpt_path,
                **model_kwargs,
            )
        calibration_only = True

    model_type = get_model_type(full_model)

    device = full_model.device
    if hasattr(full_model, "model"):
        device = full_model.model.device
    processor = None
    tokenizer = None
    language_model = full_model
    default_padding_side = None
    default_pad_token = None

    is_nemotron_vl_model = is_nemotron_vl(full_model)

    # Default to image-text calibration for VLM models
    if is_nemotron_vl_model and not args.calib_with_images:
        print("Nemotron VL model detected. Enabling image-text calibration by default.")
        args.calib_with_images = True

    if model_type == "whisper":
        processor = get_processor(
            args.pyt_ckpt_path,
            model_type,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.calib_with_images:
        # For VLM image calibration, we need an AutoProcessor to build multimodal inputs.
        processor = AutoProcessor.from_pretrained(
            args.pyt_ckpt_path,
            trust_remote_code=args.trust_remote_code,
            padding_side="left",
        )

        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            tokenizer = processor.tokenizer
        else:
            tokenizer = get_tokenizer(args.pyt_ckpt_path, trust_remote_code=args.trust_remote_code)

        default_pad_token = tokenizer.pad_token
        # Some Nemotron tokenizers may not define pad_token by default; but we use padding=True during calibration.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        assert tokenizer.pad_token is not None, f"Pad token for {args.pyt_ckpt_path} cannot be set!"

        default_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        # Quantize only the language model, but keep the full_model for calibration forward.
        extracted_lm, extracted_model_type = extract_and_prepare_language_model_from_vl(full_model)
        if extracted_lm is not None:
            language_model = extracted_lm
            model_type = extracted_model_type
    else:
        if args.specdec_offline_dataset is not None:
            language_model = full_model
        else:
            if args.dataset is None:
                args.dataset = ["cnn_dailymail", "nemotron-post-training-dataset-v2"]
                warnings.warn(
                    "No dataset specified. Defaulting to cnn_dailymail and nemotron-post-training-dataset-v2."
                )
            # Adjust calib_size to match dataset length by extending or truncating as needed
            args.calib_size = (args.calib_size + [args.calib_size[-1]] * len(args.dataset))[
                : len(args.dataset)
            ]

            # We only quantize the language model for VLMs other than the type supported above.
            extracted_lm, extracted_model_type = extract_and_prepare_language_model_from_vl(
                full_model
            )
            if extracted_lm is not None:
                language_model = extracted_lm
                model_type = extracted_model_type

        tokenizer = get_tokenizer(args.pyt_ckpt_path, trust_remote_code=args.trust_remote_code)

        default_padding_side = tokenizer.padding_side
        default_pad_token = tokenizer.pad_token
        # Left padding usually provides better calibration result.
        tokenizer.padding_side = "left"

    if model_type == "phi4mm":
        warnings.warn("Please set the default input_mode to InputMode.LANGUAGE before quantizing.")

    return (
        full_model,
        language_model,
        model_type,
        calibration_only,
        processor,
        tokenizer,
        default_padding_side,
        default_pad_token,
        device,
    )


def sparsity_main(
    args: argparse.Namespace,
    full_model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase | None,
    device: torch.device,
):
    if args.batch_size == 0:
        # Sparse algorithm takes more GPU memory so we reduce the batch_size by 4.
        args.batch_size = max(get_max_batch_size(full_model) // 4, 1)
        args.batch_size = min(args.batch_size, sum(args.calib_size))

    print(f"Use calib batch_size {args.batch_size}")

    # Different calibration datasets are also available, e.g., "pile" and "wikipedia"
    # Please also check the docstring for the datasets available
    assert tokenizer is not None and isinstance(
        tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)
    ), "The PreTrainedTokenizer must be set"
    calib_dataloader = get_dataset_dataloader(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_samples=args.calib_size,
        max_sample_length=args.calib_seq,
        device=device,
    )
    full_model = mts.sparsify(
        full_model,
        args.sparsity_fmt,
        config={"data_loader": calib_dataloader, "collect_func": lambda x: x},
    )
    mts.export(full_model)


def mono_quantize(
    args: argparse.Namespace,
    quant_cfg: dict[str, Any],
    full_model: torch.nn.Module,
    language_model: torch.nn.Module,
    model_type: str | None,
    calibration_only: bool,
    calib_dataloader: DataLoader,
    is_nemotron_vl_model: bool,
):
    """Plain quantization of the given language model to a single quantization configuration."""

    model_is_already_quantized = is_quantized(language_model)

    if "awq" in args.qformat:
        print(
            "\n####\nAWQ calibration could take longer than other calibration methods. "
            "Consider reducing calib_size to reduce calibration time.\n####\n"
        )

    # For Nemotron VL models, disable quantization of vision components
    if is_nemotron_vl_model:
        print("Disabling quantization for vision components in Nemotron VL model")
        quant_cfg["quant_cfg"].append({"quantizer_name": "*vision*", "enable": False})
        quant_cfg["quant_cfg"].append({"quantizer_name": "*image*", "enable": False})
        # Also disable radio model components specifically (for Nemotron-Parse)
        quant_cfg["quant_cfg"].append({"quantizer_name": "*radio*", "enable": False})
        quant_cfg["quant_cfg"].append({"quantizer_name": "*visual*", "enable": False})
        quant_cfg["quant_cfg"].append(
            {"quantizer_name": "*encoder*", "enable": False}
        )  # Disable encoder
        quant_cfg["quant_cfg"].append(
            {"quantizer_name": "*model_encoder*", "enable": False}
        )  # Nemotron-Parse specific
        print("Quantization will only be applied to the decoder (text generation) component")

    if not model_is_already_quantized or calibration_only:
        # quantize the model

        use_calibration = need_calibration(quant_cfg)

        if not use_calibration:
            warnings.warn("Dynamic quantization. Calibration skipped.")
        calibrate_loop = None
        if use_calibration:
            # For Nemotron VL image calibration, the dataloader yields multimodal kwargs (e.g., pixel_values).
            # Those kwargs must be consumed by the *full* VLM model, not the extracted language_model.
            if args.calib_with_images and is_nemotron_vl_model:
                calibrate_loop = create_vlm_calibration_loop(full_model, calib_dataloader)
            else:
                calibrate_loop = create_forward_loop(
                    dataloader=calib_dataloader,
                    allowed_non_tensor_keys={"base_model_outputs"}
                    if args.specdec_offline_dataset is not None
                    else None,
                )

        if calibration_only:
            language_model = mtq.calibrate(
                language_model, quant_cfg["algorithm"], forward_loop=calibrate_loop
            )
        else:
            language_model = mtq.quantize(language_model, quant_cfg, forward_loop=calibrate_loop)

        # For VL models, update full_model to use the quantized language model
        if is_nemotron_vl_model:
            language_model_lineage = get_language_model_from_vl(full_model)
            if language_model_lineage is not None:
                print("Updating full_model with quantized language_model...")
                language_model_lineage[-2].language_model = language_model

    else:
        warnings.warn("Skipping quantization: model is already quantized.")


def export_quantized(
    args: argparse.Namespace,
    full_model: torch.nn.Module,
    language_model: torch.nn.Module,
    model_type: str | None,
    tokenizer: PreTrainedTokenizerBase | None,
    default_padding_side,
    default_pad_token,
):
    with torch.inference_mode():
        if model_type is None:
            print(f"Unknown model type {type(language_model).__name__}. Continue exporting...")
            model_type = f"unknown:{type(language_model).__name__}"

        export_path = args.export_path

        # Early exit for speculative decoding checkpoints
        # No tokenizer saving needed for spec ckpts
        if has_spec_opt(full_model):
            export_speculative_decoding(full_model, export_dir=export_path)
            print(f"Quantized speculative decoding checkpoint exported to: {export_path}")
            return

        # Check if the model is a multimodal/VLM model
        is_vlm = is_multimodal_model(full_model)

        if is_vlm:
            # Save original model config and the processor config to the export path for VLMs.
            print(f"Saving original model config to {export_path}")

            config_kwargs = {"trust_remote_code": args.trust_remote_code}
            if args.attn_implementation is not None:
                config_kwargs["attn_implementation"] = args.attn_implementation
            AutoConfig.from_pretrained(args.pyt_ckpt_path, **config_kwargs).save_pretrained(
                export_path
            )

            # Try to save processor config if available
            try:
                print(f"Saving processor config to {export_path}")
                AutoProcessor.from_pretrained(
                    args.pyt_ckpt_path, trust_remote_code=args.trust_remote_code
                ).save_pretrained(export_path)
            except Exception as e:
                print(f"Warning: Could not save processor config: {e}")
                print("This is normal for some VLM architectures that don't use AutoProcessor")

        start_time = time.time()
        if (
            model_type in ["t5", "bart", "whisper"]
            or args.sparsity_fmt != "dense"
            or "int8_sq" in args.qformat
        ):
            if (
                args.inference_tensor_parallel != 1 or args.inference_pipeline_parallel != 1
            ) and args.qformat == "nvfp4_svdquant":
                raise NotImplementedError("Svdquant does not support multiple GPUs yet.")
            warnings.warn(
                "Still exporting TensorRT-LLM checkpoints for models not supported by the TensorRT-LLM torch runtime."
            )

            # Move meta tensor back to device before exporting.
            remove_hook_from_module(language_model, recurse=True)

            export_tensorrt_llm_checkpoint(
                language_model,
                model_type,
                export_dir=export_path,
                inference_tensor_parallel=args.inference_tensor_parallel,
                inference_pipeline_parallel=args.inference_pipeline_parallel,
            )

            # Copy custom model files (Python files and JSON configs) for TensorRT-LLM export
            copy_custom_model_files(args.pyt_ckpt_path, export_path, args.trust_remote_code)
        else:
            # Check arguments for unified_hf export format and set to default if unsupported arguments are provided
            assert args.sparsity_fmt == "dense", (
                f"Sparsity format {args.sparsity_fmt} not supported by unified export api."
            )

            if args.inference_tensor_parallel != 1 or args.inference_pipeline_parallel != 1:
                warnings.warn(
                    "Unified HF export format does not specify inference tensor parallel or pipeline parallel. "
                    "They will be set at deployment time."
                )

            # Load any missing weights from non-standard safetensors (handled in get_model for non-low-memory mode)
            # Store the MTP layer prefixes on the model for later exclusion from quantization
            if args.vllm_fakequant_export:
                export_hf_vllm_fq_checkpoint(
                    full_model, export_dir=export_path, inplace_mem_efficient=True
                )
            else:
                mtp_layer_prefixes, mtp_state_dict = load_mtp_weights(
                    full_model, args.pyt_ckpt_path
                )

                if mtp_layer_prefixes:
                    full_model._mtp_layer_prefixes = mtp_layer_prefixes

                export_hf_checkpoint(
                    full_model,
                    export_dir=export_path,
                    extra_state_dict=mtp_state_dict,
                )

        # Restore default padding and export the tokenizer as well.
        if tokenizer is not None:
            tokenizer.padding_side = default_padding_side
            if default_pad_token is not None:
                tokenizer.pad_token = default_pad_token
            tokenizer.save_pretrained(export_path)

        # Copy custom model files (Python files and JSON configs) if trust_remote_code is used.
        # This must run AFTER tokenizer.save_pretrained() so original tokenizer files
        # from the source checkpoint take precedence over regenerated ones (which may
        # differ in format due to newer transformers versions).
        copy_custom_model_files(args.pyt_ckpt_path, export_path, args.trust_remote_code)

        end_time = time.time()
        print(
            f"Quantized model exported to: {export_path}. Total time used {end_time - start_time}s"
        )


def pre_quantize(
    args: argparse.Namespace,
    full_model: torch.nn.Module,
    model_type: str | None,
    tokenizer: PreTrainedTokenizerBase | None,
    calib_dataloader: DataLoader | None,
    is_nemotron_vl_model: bool,
):
    """
    Processing before the quantization.

    Currently we run one round of generation for a sample prompt, to be compared with
    post-quantize generation.

    """
    # Offline specdec models skip pre-quantize preview (no tokenizer or standard dataloader)
    if args.specdec_offline_dataset is not None:
        return None, None

    # Only run single sample for preview
    assert calib_dataloader is not None, "calib_dataloader is required for pre-quantize preview"
    preview_input_ids = next(iter(calib_dataloader))[
        "input_features" if model_type == "whisper" else "input_ids"
    ][0:1]

    # Generate preview before quantization
    if args.skip_generate:
        generated_ids_before_ptq = None
    elif model_type == "deepseek":
        # DeepSeek generation may go OOM, so we skip it
        generated_ids_before_ptq = None
    elif model_type == "nemotron_h":
        # NemotronH (SSM/Mamba hybrid) modeling code does not work with accelerate's big model inference
        # when multiple GPUs are used. So we skip generation for NemotronH models. The issue presents in
        # the remote code and also in transformers library integration code from v5.3
        generated_ids_before_ptq = None
    elif is_nemotron_vl_model and tokenizer is not None:
        generated_ids_before_ptq = run_nemotron_vl_preview(
            full_model,
            tokenizer,
            preview_input_ids,
            args.pyt_ckpt_path,
            "before quantization",
            allow_fallback=False,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        generated_ids_before_ptq = full_model.generate(preview_input_ids, max_new_tokens=100)

    return preview_input_ids, generated_ids_before_ptq


def post_quantize(
    args: argparse.Namespace,
    full_model: torch.nn.Module,
    language_model: torch.nn.Module,
    model_type: str | None,
    tokenizer: PreTrainedTokenizerBase | None,
    processor: ProcessorMixin | None,
    preview_input_ids,
    generated_ids_before_ptq,
    is_nemotron_vl_model,
    first_text_speech_dataset,
    default_padding_side,
    default_pad_token,
    calib_dataloader: DataLoader,
):
    """
    Processing after the quantization, then export.

    For offline speculative decoding models, skip generation comparison and proceed
    directly to export.  For standard models, run one round of generation using the
    quantized model for a sample prompt and compare it with pre-quantize generation.

    """
    # Early exit for offline speculative decoding: skip generation comparison and export directly.
    # The model's get_dummy_inputs() provides the right input format for the export forward pass.
    if args.specdec_offline_dataset is not None:
        export_quantized(
            args,
            full_model,
            language_model,
            model_type,
            tokenizer,
            default_padding_side,
            default_pad_token,
        )
        return

    if args.verbose:
        try:
            mtq.print_quant_summary(full_model, args.export_path)
            save_expert_token_count_table(full_model, args.export_path)
        except Exception as e:
            print(f"Error saving quant summary: {e}")
            print("Continuing with generation...")

    # Run some samples
    torch.cuda.empty_cache()
    generated_ids_after_ptq = None
    if generated_ids_before_ptq is None:
        pass
    elif model_type != "llama4" and not is_nemotron_vl_model:
        # Our fake quantizer may not be fully compatible with torch.compile.
        generated_ids_after_ptq = full_model.generate(preview_input_ids, max_new_tokens=100)
    elif is_nemotron_vl_model and tokenizer is not None:
        generated_ids_after_ptq = run_nemotron_vl_preview(
            full_model,
            tokenizer,
            preview_input_ids,
            args.pyt_ckpt_path,
            "after quantization",
            allow_fallback=False,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        warnings.warn(
            "Llama4 Maverick generation after quantization has a bug. Skipping generation sample."
        )

    def input_decode(input_ids):
        if processor is not None and isinstance(processor, WhisperProcessor):
            return first_text_speech_dataset
        elif tokenizer is not None:
            return tokenizer.batch_decode(input_ids)
        else:
            raise ValueError("The processor or tokenizer must be set")

    def output_decode(generated_ids, input_shape):
        if is_enc_dec(model_type):
            if processor is not None and isinstance(processor, WhisperProcessor):
                return processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            elif tokenizer is not None:
                return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        elif tokenizer is not None:
            return tokenizer.batch_decode(generated_ids[:, input_shape:])
        else:
            raise ValueError("The processor or tokenizer must be set")

    if generated_ids_after_ptq is not None:
        print("--------")
        if is_nemotron_vl_model:
            # For Nemotron VL models, generated_ids are text strings from model.chat()
            print("Nemotron VL model text-only generation results:")
            print(f"Text response before quantization: {generated_ids_before_ptq}")
            print("--------")
            print(f"Text response after quantization: {generated_ids_after_ptq}")
            print("--------")
            print("Note: Additional VL tests with images were run separately above")
        else:
            # For regular LLMs, generated_ids are token tensors that need decoding
            print(f"example test input: {input_decode(preview_input_ids)}")
            print("--------")
            print(
                f"example outputs before ptq: {output_decode(generated_ids_before_ptq, preview_input_ids.shape[1])}"
            )
            print("--------")
            print(
                f"example outputs after ptq: {output_decode(generated_ids_after_ptq, preview_input_ids.shape[1])}"
            )

    export_quantized(
        args,
        full_model,
        language_model,
        model_type,
        tokenizer,
        default_padding_side,
        default_pad_token,
    )


def quantize_main(
    args: argparse.Namespace,
    full_model: torch.nn.Module,
    language_model: torch.nn.Module,
    model_type: str | None,
    calibration_only: bool,
    processor: ProcessorMixin | None,
    tokenizer: PreTrainedTokenizerBase | None,
    default_padding_side,
    default_pad_token,
    device: torch.device,
):
    if args.batch_size == 0:
        # For VL models with image-text calibration, skip automatic batch size detection
        # since get_max_batch_size can't handle multimodal inputs
        if args.calib_with_images:
            print("Image-text calibration enabled. Using default batch_size=1 for calibration.")
            args.batch_size = 1
        # Speculative decoding offline model dost not support get_max_batch_size() because of
        # the customized dataloader, so we set batch_size to 1 to avoid OOM.
        elif args.specdec_offline_dataset is not None:
            print(
                "Offline speculative decoding calibration enabled. Using default batch_size=1 for calibration."
            )
            args.batch_size = 1
        else:
            # Calibration/sparsification will actually take much more memory than regular inference
            # due to intermediate tensors for fake quantization. Setting sample_memory_usage_ratio
            # to 2 to avoid OOM for AWQ/SmoothQuant fake quantization as it will take more memory than inference.
            sample_memory_usage_ratio = 2 if "awq" in args.qformat or "sq" in args.qformat else 1.1
            # Whisper model expects mel-spectrogram input features of length 3000
            # Whisper model needs input of shape (batch_size, num_mel_bins, 3000)
            # As the encoder of Whisper doesn't have embedding layer, input dtype has to be float
            # For non-Whisper models (language models), sample_input will be set up inside get_max_batch_size()
            if model_type == "whisper":
                max_sample_length = 3000
                num_mel_bins = language_model.config.num_mel_bins
                sample_input_single_batch = (
                    torch.ones([1, num_mel_bins, max_sample_length], dtype=language_model.dtype).to(
                        language_model.device
                    )
                    * 100
                )
            else:
                sample_input_single_batch = None

            run_auto_quant = args.auto_quantize_bits is not None

            args.batch_size = get_max_batch_size(
                language_model,
                max_sample_length=args.calib_seq,
                sample_memory_usage_ratio=sample_memory_usage_ratio if not run_auto_quant else 1.0,
                sample_input_single_batch=sample_input_single_batch,
                enable_grad=run_auto_quant,
            )
            args.batch_size = min(args.batch_size, sum(args.calib_size))

    print(f"Use calib batch_size {args.batch_size}")

    calib_dataloader, first_text_speech_dataset = make_calib_dataloader(
        args, language_model, processor, tokenizer, device, model_type
    )

    # Detect if this is a Nemotron VL model using architecture-based detection
    is_nemotron_vl_model = is_nemotron_vl(full_model)

    preview_input_ids, generated_ids_before_ptq = pre_quantize(
        args, full_model, model_type, tokenizer, calib_dataloader, is_nemotron_vl_model
    )

    if args.auto_quantize_bits:
        assert len(args.qformat.split(",")) > 1, (
            "Auto quantization needs multiple quantization format."
        )

        auto_quantize(
            args,
            language_model,
            calib_dataloader,
        )

    else:
        # mono quantization

        if args.recipe is not None:
            print(f"Use recipe {args.recipe} for quantization")
            recipe = load_recipe(args.recipe)
            assert isinstance(recipe, ModelOptPTQRecipe), (
                f"Expected PTQ recipe, but got {type(recipe).__name__} from {args.recipe}"
            )
            quant_cfg = recipe.quantize.model_dump()

        else:
            assert len(args.qformat.split(",")) == 1, (
                "Plain quantization supports only one quantization format."
            )

            assert args.qformat in QUANT_CFG_CHOICES, (
                f"Unsupported quantization format: {args.qformat}, choices are: {list(QUANT_CFG_CHOICES.keys())}"
            )
            quant_cfg = QUANT_CFG_CHOICES[args.qformat]

            quant_cfg = build_quant_cfg(
                args.qformat,
                quant_cfg,
                args.awq_block_size,
                model_type,
                args.moe_calib_experts_ratio,
            )

            enable_quant_kv_cache = args.kv_cache_qformat != "none"
            print(f"{'Enable' if enable_quant_kv_cache else 'Disable'} KV cache quantization")

            # Check if any bmm_quantizer is in the quant_cfg. If so, we need to enable the bmm_quantizer.
            if enable_quant_kv_cache:
                quant_cfg = mtq.update_quant_cfg_with_kv_cache_quant(
                    quant_cfg,
                    getattr(mtq, KV_QUANT_CFG_CHOICES[args.kv_cache_qformat])["quant_cfg"],
                )

        # Exclude MTP layers from quantization if detected (e.g., GLM-4.7's layer 92)
        # These layers are typically speculative decoding layers that should be exported as-is
        mtp_layer_prefixes = getattr(full_model, "_mtp_layer_prefixes", None)
        if mtp_layer_prefixes:
            quant_cfg = copy.deepcopy(quant_cfg)
            for prefix in mtp_layer_prefixes:
                pattern = f"*{prefix}*"
                quant_cfg["quant_cfg"].append({"quantizer_name": pattern, "enable": False})
                print(f"Excluding MTP layer from quantization: {pattern}")

        # Use constant amax for KV quantizers when a cast format is selected.
        # Recipes are authoritative for KV cache config (including use_constant_amax),
        # so skip this post-hoc override when --recipe is used; rely on the YAML instead
        # (see modelopt_recipes/general/ptq/*_cast_kv.yaml).
        if args.recipe is None and args.kv_cache_qformat in _KV_CAST_FORMATS:
            quant_cfg = copy.deepcopy(quant_cfg)
            _set_kv_cache_constant_amax(quant_cfg["quant_cfg"])

        if needs_checkpoint_path_update(quant_cfg):
            quant_cfg = resolve_checkpoint_dir(quant_cfg, args.pyt_ckpt_path)
            print(
                f"Auto-resolved layerwise_checkpoint_dir: {quant_cfg['algorithm']['layerwise_checkpoint_dir']}"
            )

        if args.cast_mxfp4_to_nvfp4:
            quant_cfg = copy.deepcopy(quant_cfg)
            force_weight_quantizers_static(quant_cfg["quant_cfg"])

        if args.qformat in QUANT_CFG_CHOICES:
            mono_quantize(
                args,
                quant_cfg,
                full_model,
                language_model,
                model_type,
                calibration_only,
                calib_dataloader,
                is_nemotron_vl_model,
            )
        else:
            assert model_type != "dbrx", f"Does not support export {model_type} without quantizaton"
            print(f"qformat: {args.qformat}. No quantization applied, export {device} model")

    # If asked, run the closed-form MXFP4 -> NVFP4 cast: read the source MXFP4
    # *_scales tensors and pin each NVFP4 weight quantizer's scale_2 to 2^m.
    # Runs after calibration (max_calibrate has already promoted weight quantizers
    # to NVFP4StaticQuantizer with a data-derived ``_global_amax``); we just
    # override that scalar with the closed-form value before export.
    if args.cast_mxfp4_to_nvfp4:
        apply_cast_mxfp4_to_nvfp4(language_model, args.pyt_ckpt_path)

    post_quantize(
        args,
        full_model,
        language_model,
        model_type,
        tokenizer,
        processor,
        preview_input_ids,
        generated_ids_before_ptq,
        is_nemotron_vl_model,
        first_text_speech_dataset,
        default_padding_side,
        default_pad_token,
        calib_dataloader,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyt_ckpt_path",
        "--model",
        help=(
            "Model name or path to the PyTorch checkpoint to be quantized. "
            "Can be a local path or a Huggingface model name."
        ),
        required=True,
    )
    parser.add_argument(
        "--recipe",
        help=(
            "PTQ recipe YAML file or name without suffix (e.g. general/ptq/nvfp4_default-fp8_cast_kv, "
            "general/ptq/nvfp4_default-fp8_kv, general/ptq/nvfp4_default-nvfp4_cast_kv). "
            "When set, --kv_cache_qformat is ignored; the recipe fully determines KV cache config."
        ),
        default=None,
    )

    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--qformat",
        help=(
            "Quantization format. If --auto_quantize_bits is set, this argument specifies the quantization "
            "format for optimal per-layer auto_quantize search."
        ),
        default="fp8",
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size for calibration. Default to 0 as we calculate max batch size on-the-fly",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--calib_size",
        help=(
            "Number of samples for calibration. If a comma separated list of values is provided, "
            "each value will be used as the calibration size for the corresponding dataset. "
            "This argument will be parsed and converted as a list of ints."
        ),
        type=str,
        default="512",
    )
    parser.add_argument(
        "--calib_seq",
        help="Maximum sequence length for calibration.",
        type=int,
        default=512,
    )
    parser.add_argument("--export_path", default="exported_model")
    parser.add_argument(
        "--dataset",
        help=(
            f"name of a dataset, or a comma separated list of datasets. "
            f"dataset choices are {get_supported_datasets()}"
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--specdec_offline_dataset",
        help=(
            "If set, the model is a speculative decoding model,"
            "which uses offline dataset for calibration. "
        ),
        default=None,
    )
    parser.add_argument(
        "--calib_with_images",
        action="store_true",
        help=(
            "Calibrate with image-text pairs (for VLMs). "
            "This uses nemotron_vlm_dataset_v2 with default subsets (sparsetables, plotqa_cot, wiki_en)."
        ),
    )
    parser.add_argument("--inference_tensor_parallel", type=int, default=1)
    parser.add_argument("--inference_pipeline_parallel", type=int, default=1)
    parser.add_argument("--awq_block_size", default=0, type=int)
    parser.add_argument(
        "--sparsity_fmt",
        help="Sparsity format.",
        default="dense",
        choices=["dense", "sparsegpt"],
    )
    parser.add_argument(
        "--auto_quantize_bits",
        default=None,
        type=float,
        help=(
            "Effective bits constraint for auto_quantize. If not set, "
            "regular quantization without auto_quantize search will be applied."
        ),
    )
    parser.add_argument(
        "--kv_cache_qformat",
        required=False,
        default="fp8_cast",
        choices=KV_QUANT_CFG_CHOICES.keys(),
        help=(
            "Specify KV cache quantization format. Default: fp8_cast. "
            "Formats ending in '_cast' (fp8_cast, nvfp4_cast) set the amax to FP8 range "
            "without data-driven calibration. "
            "Other formats (fp8, nvfp4, etc.) use data-driven calibration. "
            "Ignored when --recipe is given: the recipe YAML is authoritative for KV "
            "cache config (use the *_cast_kv.yaml recipes for the cast variants)."
        ),
    )
    parser.add_argument(
        "--export_fmt",
        required=False,
        default="hf",
        choices=["tensorrt_llm", "hf"],
        help="Deprecated. Please avoid using this argument.",
    )
    parser.add_argument(
        "--trust_remote_code",
        help="Set trust_remote_code for Huggingface models and tokenizers",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--gpu_max_mem_percentage",
        help=(
            "Specify the percentage of available GPU memory to use for loading the model when "
            "device_map is set to sequential. "
            "By default, 80%% of the available GPU memory is used."
        ),
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--use_seq_device_map",
        help=(
            "Use device_map=sequential to load the model onto GPUs. This ensures the model is loaded "
            "utilizing the percentage of available GPU memory as specified by the value passed with gpu_max_mem flag."
            "Helpful in cases where device_map=auto loads model unevenly on GPUs causing GPU OOM during quantization."
        ),
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--verbose",
        help="Print verbose output (e.g. quantization summary). Disable by --no-verbose.",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--skip_generate",
        help=(
            "Skip pre/post-quantization preview calls that invoke model.generate(). "
            "Note: this does not skip calibration or batch-size probing. "
            "For very large models, pair with --batch_size 1 to avoid max-batch probing."
        ),
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--low_memory_mode",
        help=(
            "Use low memory mode for quantization."
            "This is an experimental feature and may not work for all quantization formats."
        ),
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--attn_implementation",
        help=(
            "Specify the attention implementation to use. "
            "This arg will be passed to the HF model loading if specified."
        ),
        default=None,
        type=str,
    )
    parser.add_argument(
        "--auto_quantize_method",
        type=str,
        default="gradient",
        choices=["gradient", "kl_div"],
        help=(
            "Method for auto_quantize sensitivity analysis. 'gradient' uses gradient-based method "
            "(requires labels in dataset). 'kl_div' uses KL divergence between original and "
            "quantized model outputs (no labels required). Default: 'gradient'"
        ),
    )
    parser.add_argument(
        "--auto_quantize_score_size",
        type=int,
        default=128,
        help=(
            "Number of samples to use for auto_quantize scoring. Most of auto_quantize time is spent on "
            "sensitivity score estimation, so reducing this speeds it up while only minimally affecting "
            "final model accuracy compared to lowering --calib_size (the number of samples used for calibration)."
        ),
    )
    parser.add_argument(
        "--auto_quantize_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to checkpoint file for saving/restoring auto_quantize search state "
            "(sensitivity scores, costs, etc.). Only used when auto_quantize_bits is specified."
        ),
    )
    parser.add_argument(
        "--moe_calib_experts_ratio",
        type=float,
        default=None,
        help=(
            "Fraction of experts to calibrate during forward pass (ratio in (0.0, 1.0]). "
            "Only used for MOE models; used to reduce the number of experts calibrated during the forward pass. "
            "Does not impact non-MOE models."
        ),
    )
    parser.add_argument(
        "--vllm_fakequant_export",
        default=False,
        action="store_true",
        help="Export as vLLM fake-quant checkpoint (produces vllm_fq_modelopt_state.pth "
        "for use with vllm_serve_fakequant.py).",
    )
    parser.add_argument(
        "--cast_mxfp4_to_nvfp4",
        action="store_true",
        default=False,
        help=(
            "After calibration, override NVFP4 weight quantizers' global_amax with "
            "the closed-form value derived from the source MXFP4 *_scales. "
            "Per-block _amax is computed from the loaded BF16 weights (data-derived). "
            "Use when --pyt_ckpt_path points at an MXFP4 HF checkpoint (e.g. "
            "openai/gpt-oss-20b) and the target qformat is NVFP4-family."
        ),
    )

    args = parser.parse_args()
    if args.moe_calib_experts_ratio is not None and not (0.0 < args.moe_calib_experts_ratio <= 1.0):
        parser.error("--moe_calib_experts_ratio must be in the range (0.0, 1.0].")

    if args.specdec_offline_dataset is not None and args.sparsity_fmt != "dense":
        parser.error("--specdec_offline_dataset is only supported with --sparsity_fmt dense (PTQ).")

    if args.specdec_offline_dataset is not None and args.low_memory_mode:
        parser.error("--specdec_offline_dataset is not compatible with --low_memory_mode.")

    return args


def main(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise OSError("GPU is required for inference.")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    # launch a memory monitor to read the currently used GPU memory.
    launch_memory_monitor()

    # Force eager execution for all model types.
    torch.compiler.set_stance("force_eager")

    (
        full_model,
        language_model,
        model_type,
        calibration_only,
        processor,
        tokenizer,
        default_padding_side,
        default_pad_token,
        device,
    ) = load_model(args)

    if args.sparsity_fmt != "dense":
        # Sparse
        sparsity_main(args, full_model, tokenizer, device)
    else:
        # Quantize
        quantize_main(
            args,
            full_model,
            language_model,
            model_type,
            calibration_only,
            processor,
            tokenizer,
            default_padding_side,
            default_pad_token,
            device,
        )


if __name__ == "__main__":
    args = parse_args()

    if args.export_fmt != "hf":
        warnings.warn("Deprecated. --export_fmt forced to hf.")

    args.dataset = args.dataset.split(",") if isinstance(args.dataset, str) else args.dataset
    args.calib_size = [int(num_sample) for num_sample in args.calib_size.split(",")]

    if args.specdec_offline_dataset is not None and len(args.calib_size) != 1:
        raise ValueError(
            "--specdec_offline_dataset expects a single --calib value, not a comma-separated list."
        )

    if args.cast_mxfp4_to_nvfp4:
        qformats = [q.strip() for q in args.qformat.split(",")]
        if not all("nvfp4" in q for q in qformats):
            raise ValueError(
                "--cast_mxfp4_to_nvfp4 requires NVFP4-family --qformat values "
                f"(got {args.qformat!r}). Use e.g. --qformat nvfp4 or nvfp4_mlp_only."
            )
        if args.auto_quantize_bits is not None:
            raise ValueError(
                "--cast_mxfp4_to_nvfp4 is not supported with --auto_quantize_bits "
                "(multi-format auto-quantize)."
            )

    main(args)
