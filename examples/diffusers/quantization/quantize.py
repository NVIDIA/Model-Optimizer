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
import logging
import sys
import time as time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from calibration import Calibrator
from config import (
    FP8_DEFAULT_CONFIG,
    INT8_DEFAULT_CONFIG,
    NVFP4_DEFAULT_CONFIG,
    NVFP4_FP8_MHA_CONFIG,
    reset_set_int8_config,
    set_quant_config_attr,
)
from diffusers import DiffusionPipeline
from models_utils import MODEL_DEFAULTS, ModelType, get_model_filter_func, parse_extra_params
from onnx_utils.export import generate_fp8_scales, modelopt_export_sd
from pipeline_manager import PipelineManager
from quantize_config import (
    CalibrationConfig,
    CollectMethod,
    DataType,
    ExportConfig,
    ModelConfig,
    QuantAlgo,
    QuantFormat,
    QuantizationConfig,
)
from utils import check_conv_and_mha, check_lora

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        verbose: Enable verbose logging

    Returns:
        Configured logger instance
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create custom formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.addHandler(console_handler)

    # Optionally reduce noise from other libraries
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    return logger


class Quantizer:
    """Handles model quantization operations."""

    def __init__(
        self, config: QuantizationConfig, model_config: ModelConfig, logger: logging.Logger
    ):
        """
        Initialize quantizer.

        Args:
            config: Quantization configuration
            model_config: Model configuration
            logger: Logger instance
        """
        self.config = config
        self.model_config = model_config
        self.logger = logger

    def get_quant_config(self, n_steps: int, backbone: torch.nn.Module) -> Any:
        """
        Build quantization configuration based on format.

        Args:
            n_steps: Number of denoising steps

        Returns:
            Quantization configuration object
        """
        self.logger.info(f"Building quantization config for {self.config.format.value}")

        if self.config.format == QuantFormat.INT8:
            if self.config.algo == QuantAlgo.SMOOTHQUANT:
                quant_config = mtq.INT8_SMOOTHQUANT_CFG
            else:
                quant_config = INT8_DEFAULT_CONFIG
            if self.config.collect_method != CollectMethod.DEFAULT:
                reset_set_int8_config(
                    quant_config,
                    self.config.percentile,
                    n_steps,
                    collect_method=self.config.collect_method.value,
                    backbone=backbone,
                )
        elif self.config.format == QuantFormat.FP8:
            quant_config = FP8_DEFAULT_CONFIG
        elif self.config.format == QuantFormat.FP4:
            if self.model_config.model_type.value.startswith("flux"):
                quant_config = NVFP4_FP8_MHA_CONFIG
            else:
                quant_config = NVFP4_DEFAULT_CONFIG
        else:
            raise NotImplementedError(f"Unknown format {self.config.format}")
        set_quant_config_attr(
            quant_config,
            self.model_config.trt_high_precision_dtype.value,
            self.config.algo.value,
            alpha=self.config.alpha,
            lowrank=self.config.lowrank,
        )

        return quant_config

    def quantize_model(
        self,
        backbone: torch.nn.Module,
        quant_config: Any,
        forward_loop: callable,  # type: ignore[valid-type]
    ) -> torch.nn.Module:
        """
        Apply quantization to the model.

        Args:
            backbone: Model backbone to quantize
            quant_config: Quantization configuration
            forward_loop: Forward pass function for calibration
        """
        self.logger.info("Checking for LoRA layers...")
        check_lora(backbone)

        self.logger.info("Starting model quantization...")
        mtq.quantize(backbone, quant_config, forward_loop)
        # Get model-specific filter function
        model_filter_func = get_model_filter_func(self.model_config.model_type)
        self.logger.info(f"Using filter function for {self.model_config.model_type.value}")

        self.logger.info("Disabling specific quantizers...")
        mtq.disable_quantizer(backbone, model_filter_func)

        self.logger.info("Quantization completed successfully")
        return backbone


class ExportManager:
    """Handles model export operations."""

    def __init__(self, config: ExportConfig, logger: logging.Logger):
        """
        Initialize export manager.

        Args:
            config: Export configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger

    def _has_conv_layers(self, model: torch.nn.Module) -> bool:
        """
        Check if the model contains any convolutional layers.

        Args:
            model: Model to check

        Returns:
            True if model contains Conv layers, False otherwise
        """
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)) and (
                module.input_quantizer.is_enabled or module.weight_quantizer.is_enabled
            ):
                return True
        return False

    def save_checkpoint(self, backbone: torch.nn.Module) -> None:
        """
        Save quantized model checkpoint.

        Args:
            backbone: Model backbone to save
        """
        if not self.config.quantized_torch_ckpt_path:
            return

        self.logger.info(f"Saving quantized checkpoint to {self.config.quantized_torch_ckpt_path}")
        mto.save(backbone, str(self.config.quantized_torch_ckpt_path))
        self.logger.info("Checkpoint saved successfully")

    def export_onnx(
        self,
        pipe: DiffusionPipeline,
        backbone: torch.nn.Module,
        model_type: ModelType,
        quant_format: QuantFormat,
    ) -> None:
        """
        Export model to ONNX format.

        Args:
            pipe: Diffusion pipeline
            backbone: Model backbone
            model_type: Type of model
            quant_format: Quantization format
        """
        if not self.config.onnx_dir:
            return

        self.logger.info(f"Starting ONNX export to {self.config.onnx_dir}")

        if quant_format == QuantFormat.FP8 and self._has_conv_layers(backbone):
            self.logger.info(
                "Detected quantizing conv layers in backbone. Generating FP8 scales..."
            )
            generate_fp8_scales(backbone)
        self.logger.info("Preparing models for export...")
        pipe.to("cpu")
        torch.cuda.empty_cache()
        backbone.to("cuda")
        # Export to ONNX
        backbone.eval()
        with torch.no_grad():
            self.logger.info("Exporting to ONNX...")
            modelopt_export_sd(
                backbone, str(self.config.onnx_dir), model_type.value, quant_format.value
            )

        self.logger.info("ONNX export completed successfully")

    def restore_checkpoint(self, backbone: nn.Module) -> None:
        """
        Restore a previously quantized model.

        Args:
            backbone: Model backbone to restore into
        """
        if not self.config.restore_from:
            return

        self.logger.info(f"Restoring model from {self.config.restore_from}")
        mto.restore(backbone, str(self.config.restore_from))
        self.logger.info("Model restored successfully")

    # TODO: should not do the any data type
    def export_hf_ckpt(self, pipe: Any) -> None:
        """
        Export quantized model to HuggingFace checkpoint format.

        Args:
            pipe: Diffusion pipeline containing the quantized model
        """
        if not self.config.hf_ckpt_dir:
            return

        self.logger.info(f"Exporting HuggingFace checkpoint to {self.config.hf_ckpt_dir}")
        export_hf_checkpoint(pipe, export_dir=self.config.hf_ckpt_dir)
        self.logger.info("HuggingFace checkpoint export completed successfully")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Enhanced Diffusion Model Quantization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Basic INT8 quantization with SmoothQuant
            %(prog)s --model flux-dev --format int8 --quant-algo smoothquant --collect-method global_min

            # FP8 quantization with ONNX export
            %(prog)s --model sd3-medium --format fp8 --onnx-dir ./onnx_models/

            # FP8 quantization with weight compression (reduces memory footprint)
            %(prog)s --model flux-dev --format fp8 --compress

            # Quantize LTX-Video model with full multi-stage pipeline
            %(prog)s --model ltx-video-dev --format fp8 --batch-size 1 --calib-size 32

            # Faster LTX-Video quantization (skip upsampler)
            %(prog)s --model ltx-video-dev --format fp8 --batch-size 1 --calib-size 32 --ltx-skip-upsampler

            # Restore and export a previously quantized model
            %(prog)s --model flux-schnell --restore-from checkpoint.pt --onnx-dir ./exports/
        """,
    )
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        default="flux-dev",
        choices=[m.value for m in ModelType],
        help="Model to load and quantize",
    )
    model_group.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="model backbone in the DiffusionPipeline to work on, if not provided use default based on model type",
    )
    model_group.add_argument(
        "--model-dtype",
        type=str,
        default="Half",
        choices=[d.value for d in DataType],
        help="Precision for loading the pipeline. If you want different dtypes for separate components, "
        "please specify using --component-dtype",
    )
    model_group.add_argument(
        "--component-dtype",
        action="append",
        default=[],
        help="Precision for loading each component of the model by format of name:dtype. "
        "You can specify multiple components. "
        "Example: --component-dtype vae:Half --component-dtype transformer:BFloat16",
    )
    model_group.add_argument(
        "--override-model-path", type=str, help="Custom path to model (overrides default)"
    )
    model_group.add_argument(
        "--cpu-offloading", action="store_true", help="Enable CPU offloading for limited VRAM"
    )
    model_group.add_argument(
        "--ltx-skip-upsampler",
        action="store_true",
        help="Skip upsampler pipeline for LTX-Video (faster calibration, only quantizes main transformer)",
    )
    model_group.add_argument(
        "--extra-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra model-specific parameters in KEY=VALUE form. Can be provided multiple times. "
            "These override model-specific CLI arguments when present."
        ),
    )
    quant_group = parser.add_argument_group("Quantization Configuration")
    quant_group.add_argument(
        "--format",
        type=str,
        default="int8",
        choices=[f.value for f in QuantFormat],
        help="Quantization format",
    )
    quant_group.add_argument(
        "--quant-algo",
        type=str,
        default="max",
        choices=[a.value for a in QuantAlgo],
        help="Quantization algorithm",
    )
    quant_group.add_argument(
        "--percentile",
        type=float,
        default=1.0,
        help="Percentile for calibration, works for INT8, not including smoothquant",
    )
    quant_group.add_argument(
        "--collect-method",
        type=str,
        default="default",
        choices=[c.value for c in CollectMethod],
        help="Calibration collection method, works for INT8, not including smoothquant",
    )
    quant_group.add_argument("--alpha", type=float, default=1.0, help="SmoothQuant alpha parameter")
    quant_group.add_argument("--lowrank", type=int, default=32, help="SVDQuant lowrank parameter")
    quant_group.add_argument(
        "--quantize-mha", action="store_true", help="Quantizing MHA into FP8 if its True"
    )
    quant_group.add_argument(
        "--compress",
        action="store_true",
        help="Compress quantized weights to reduce memory footprint (FP8/FP4 only)",
    )

    calib_group = parser.add_argument_group("Calibration Configuration")
    calib_group.add_argument("--batch-size", type=int, default=2, help="Batch size for calibration")
    calib_group.add_argument(
        "--calib-size", type=int, default=128, help="Total number of calibration samples"
    )
    calib_group.add_argument("--n-steps", type=int, default=30, help="Number of denoising steps")
    calib_group.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Calibrate using prompts in the file instead of the default dataset.",
    )

    export_group = parser.add_argument_group("Export Configuration")
    export_group.add_argument(
        "--quantized-torch-ckpt-save-path",
        type=str,
        help="Path to save quantized PyTorch checkpoint",
    )
    export_group.add_argument("--onnx-dir", type=str, help="Directory for ONNX export")
    export_group.add_argument(
        "--hf-ckpt-dir",
        type=str,
        help="Directory for HuggingFace checkpoint export",
    )
    export_group.add_argument(
        "--restore-from", type=str, help="Path to restore from previous checkpoint"
    )
    export_group.add_argument(
        "--trt-high-precision-dtype",
        type=str,
        default="Half",
        choices=[d.value for d in DataType],
        help="Precision for TensorRT high-precision layers",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser


def main() -> None:
    from diffusers.models.normalization import RMSNorm as DiffuserRMSNorm

    torch.nn.RMSNorm = DiffuserRMSNorm
    torch.nn.modules.normalization.RMSNorm = DiffuserRMSNorm

    parser = create_argument_parser()
    args, unknown_args = parser.parse_known_args()

    model_type = ModelType(args.model)
    if args.backbone is None:
        args.backbone = MODEL_DEFAULTS[model_type]["backbone"]
    s = time.time()

    model_dtype = {"default": DataType(args.model_dtype).torch_dtype}
    for component_dtype in args.component_dtype:
        component, dtype = component_dtype.split(":")
        model_dtype[component] = DataType(dtype).torch_dtype

    logger = setup_logging(args.verbose)
    logger.info("Starting Enhanced Diffusion Model Quantization")

    try:
        extra_params = parse_extra_params(args.extra_param, unknown_args, logger)
        model_config = ModelConfig(
            model_type=model_type,
            model_dtype=model_dtype,
            backbone=args.backbone,
            trt_high_precision_dtype=DataType(args.trt_high_precision_dtype),
            override_model_path=Path(args.override_model_path)
            if args.override_model_path
            else None,
            cpu_offloading=args.cpu_offloading,
            ltx_skip_upsampler=args.ltx_skip_upsampler,
            extra_params=extra_params,
        )

        quant_config = QuantizationConfig(
            format=QuantFormat(args.format),
            algo=QuantAlgo(args.quant_algo),
            percentile=args.percentile,
            collect_method=CollectMethod(args.collect_method),
            alpha=args.alpha,
            lowrank=args.lowrank,
            quantize_mha=args.quantize_mha,
            compress=args.compress,
        )

        if args.prompts_file is not None:
            prompts_file = Path(args.prompts_file)
            assert prompts_file.exists(), (
                f"User specified prompts file {prompts_file} does not exist."
            )
            prompts_dataset = prompts_file
        else:
            prompts_dataset = MODEL_DEFAULTS[model_type]["dataset"]
        calib_config = CalibrationConfig(
            prompts_dataset=prompts_dataset,
            batch_size=args.batch_size,
            calib_size=args.calib_size,
            n_steps=args.n_steps,
        )

        export_config = ExportConfig(
            quantized_torch_ckpt_path=Path(args.quantized_torch_ckpt_save_path)
            if args.quantized_torch_ckpt_save_path
            else None,
            onnx_dir=Path(args.onnx_dir) if args.onnx_dir else None,
            hf_ckpt_dir=Path(args.hf_ckpt_dir) if args.hf_ckpt_dir else None,
            restore_from=Path(args.restore_from) if args.restore_from else None,
        )

        logger.info("Validating configurations...")
        quant_config.validate()
        export_config.validate()
        if not export_config.restore_from:
            calib_config.validate()

        pipeline_manager = PipelineManager(model_config, logger)
        pipe = pipeline_manager.create_pipeline()
        pipeline_manager.setup_device()

        backbone = pipeline_manager.get_backbone()
        export_manager = ExportManager(export_config, logger)

        if export_config.restore_from and export_config.restore_from.exists():
            export_manager.restore_checkpoint(backbone)

            if export_config.quantized_torch_ckpt_path and not export_config.restore_from.samefile(
                export_config.restore_from
            ):
                export_manager.save_checkpoint(backbone)
        else:
            logger.info("Initializing calibration...")
            calibrator = Calibrator(pipeline_manager, calib_config, model_config.model_type, logger)
            batched_prompts = calibrator.load_and_batch_prompts()

            quantizer = Quantizer(quant_config, model_config, logger)
            backbone_quant_config = quantizer.get_quant_config(calib_config.n_steps, backbone)

            # Pipe loads the ckpt just before the inference.
            def forward_loop(mod):
                calibrator.run_calibration(batched_prompts)

            quantizer.quantize_model(backbone, backbone_quant_config, forward_loop)

            # Compress model weights if requested (only for FP8/FP4)
            if quant_config.compress:
                logger.info("Compressing model weights to reduce memory footprint...")
                mtq.compress(backbone)
                logger.info("Model compression completed")

            export_manager.save_checkpoint(backbone)

        check_conv_and_mha(
            backbone, quant_config.format == QuantFormat.FP4, quant_config.quantize_mha
        )
        mtq.print_quant_summary(backbone)

        export_manager.export_onnx(
            pipe,
            backbone,
            model_config.model_type,
            quant_config.format,
        )

        export_manager.export_hf_ckpt(pipe)

        logger.info(
            f"Quantization process completed successfully! Time taken = {time.time() - s} seconds"
        )

    except Exception as e:
        logger.error(f"Quantization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
