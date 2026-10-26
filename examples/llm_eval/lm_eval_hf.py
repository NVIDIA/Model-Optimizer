# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/tree/aa457edc3d64d81530159cd3a182932320c78f8c

# MIT License
#
# Copyright (c) 2020 EleutherAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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
import contextlib
import glob
import json
import os
import sys
import warnings
from importlib.metadata import version

import datasets
from lm_eval import utils
from packaging.version import Version

if Version(version("lm_eval")) < Version("0.4.12"):
    # Matches the floor pinned in requirements.txt.
    raise ImportError(f"lm_eval_hf.py requires lm-eval >= 0.4.12; found {version('lm_eval')}.")

from lm_eval._cli import HarnessCLI
from lm_eval.api.model import T
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import setup_logging
from quantization_utils import quantize_model
from sparse_attention_utils import sparsify_model

import modelopt.torch.opt as mto
from modelopt.torch.quantization.utils import is_quantized
from modelopt.torch.sparsity.attention_sparsity.conversion import is_attn_sparsified

try:
    import modelopt.torch.puzzletron as mtpz

    _ANYMODEL_AVAILABLE = True
except ImportError:
    _ANYMODEL_AVAILABLE = False


def _anymodel_patcher_context(pretrained, trust_remote_code=False):
    """Return a deci_x_patcher context if *pretrained* is a Puzzletron checkpoint, else a no-op."""
    if not _ANYMODEL_AVAILABLE or not pretrained:
        return contextlib.nullcontext()
    try:
        descriptor = mtpz.anymodel.resolve_descriptor_from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )
    except (ValueError, AttributeError):
        return contextlib.nullcontext()
    return mtpz.anymodel.deci_x_patcher(model_descriptor=descriptor)


def create_from_arg_obj(cls: type[T], arg_dict: dict, additional_config: dict | None = None) -> T:
    """Override HFLM.create_from_arg_obj to add quantization, sparsity, and Puzzletron support."""

    quant_cfg = arg_dict.pop("quant_cfg", None)
    auto_quantize_bits = arg_dict.pop("auto_quantize_bits", None)
    auto_quantize_method = arg_dict.pop("auto_quantize_method", "gradient")
    auto_quantize_score_size = arg_dict.pop("auto_quantize_score_size", 128)
    auto_quantize_checkpoint = arg_dict.pop("auto_quantize_checkpoint", None)
    calib_batch_size = arg_dict.pop("calib_batch_size", None)
    calib_size = arg_dict.pop("calib_size", 512)
    compress = arg_dict.pop("compress", False)

    # Sparse attention arguments
    sparse_cfg = arg_dict.pop("sparse_cfg", None)

    additional_config = {} if additional_config is None else additional_config
    additional_config = {k: v for k, v in additional_config.items() if v is not None}

    # Enable automatic save/load of modelopt state huggingface checkpointing
    mto.enable_huggingface_checkpointing()

    with _anymodel_patcher_context(
        arg_dict.get("pretrained"), arg_dict.get("trust_remote_code", False)
    ):
        model_obj = cls(**arg_dict, **additional_config)
    model_obj.tokenizer.padding_side = "left"
    if is_quantized(model_obj.model):
        # return if model is already quantized
        warnings.warn("Skipping quantization: model is already quantized.")
        return model_obj

    if quant_cfg:
        if not calib_batch_size:
            calib_batch_size = model_obj.batch_size

        quantize_model(
            model=model_obj,
            quant_cfg=quant_cfg.split(",") if auto_quantize_bits is not None else quant_cfg,
            tokenizer=model_obj.tokenizer,
            batch_size=calib_batch_size,
            calib_size=calib_size,
            auto_quantize_bits=auto_quantize_bits,
            auto_quantize_method=auto_quantize_method,
            auto_quantize_score_size=auto_quantize_score_size,
            test_generated=False,
            compress=compress,
            auto_quantize_checkpoint=auto_quantize_checkpoint,
        )

    if sparse_cfg:
        if is_attn_sparsified(model_obj.model):
            warnings.warn("Skipping sparse attention: model already has sparse attention applied.")
        else:
            sparsify_model(
                model=model_obj,
                sparse_cfg=sparse_cfg,
            )

    return model_obj


def create_from_arg_string(
    cls: type[T], arg_string: str, additional_config: dict | None = None
) -> T:
    """Override HFLM.create_from_arg_string to support Puzzletron checkpoints."""
    args = utils.simple_parse_args_string(arg_string)
    additional_config = {} if additional_config is None else additional_config
    args2 = {k: v for k, v in additional_config.items() if v is not None}

    mto.enable_huggingface_checkpointing()

    with _anymodel_patcher_context(args.get("pretrained"), args.get("trust_remote_code", False)):
        model_obj = cls(**args, **args2)

    return model_obj


HFLM.create_from_arg_obj = classmethod(create_from_arg_obj)
HFLM.create_from_arg_string = classmethod(create_from_arg_string)


# ModelOpt-specific args that we add to lm-eval's parser. After parsing, these are
# moved out of the argparse namespace and into args.model_args so they reach
# HFLM.create_from_arg_obj (and so lm-eval's own arg validation doesn't reject them).
_MODELOPT_ARG_KEYS = (
    "quant_cfg",
    "calib_batch_size",
    "calib_size",
    "auto_quantize_bits",
    "auto_quantize_method",
    "auto_quantize_score_size",
    "auto_quantize_checkpoint",
    "compress",
    "sparse_cfg",
)


def _add_modelopt_args(parser):
    """Extend an lm-eval argument parser with ModelOpt quantization and sparsity options."""
    parser.add_argument(
        "--quant_cfg",
        type=str,
        help=(
            "Quantization format. If `--auto_quantize_bits` is specified, this argument specifies the "
            "comma-separated list of quantization quantization formats that will be searched by `auto_quantize`"
        ),
    )
    parser.add_argument(
        "--calib_batch_size", type=int, help="Batch size for quantization calibration"
    )
    parser.add_argument(
        "--calib_size", type=int, help="Calibration size for quantization", default=512
    )
    parser.add_argument(
        "--auto_quantize_bits",
        type=float,
        help=(
            "Effective bits constraint for auto_quantize. If not set, "
            "regular quantization will be applied."
        ),
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
        help=("Path to checkpoint file for saving/restoring auto_quantize search state. "),
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress the model after quantization",
    )
    parser.add_argument(
        "--sparse_cfg",
        type=str,
        help="Sparse attention configuration (e.g., SKIP_SOFTMAX_DEFAULT, SKIP_SOFTMAX_CALIB)",
    )
    # Not a model arg (kept out of _MODELOPT_ARG_KEYS): popped separately and
    # applied after the eval as an accuracy gate.
    parser.add_argument(
        "--accuracy_lower_bound",
        type=float,
        default=None,
        help=(
            "Exit non-zero if the requested task's acc is below this. Requires exactly one "
            "--tasks and --output_path."
        ),
    )


def _inject_modelopt_args_into_model_args(args):
    """Move ModelOpt args from the argparse namespace into args.model_args.

    args.model_args is a dict (parsed by lm-eval's MergeDictAction). The ModelOpt
    keys must be removed from the namespace so EvaluatorConfig.from_cli doesn't
    reject them as unknown kwargs. They only apply to the HF backend; passing a
    quantization/sparsity arg with a non-HF backend raises an error.
    """
    model_args = dict(args.model_args) if args.model_args else {}
    # HFLM and its subclasses (registered as hf / hf-auto / huggingface / hf-multimodal).
    is_hf = getattr(args, "model", "hf") in ("hf", "hf-auto", "huggingface", "hf-multimodal")

    if getattr(args, "trust_remote_code", False):
        # Propagate the user-provided --trust_remote_code flag (not hardcoded).
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
        model_args["trust_remote_code"] = True
        args.trust_remote_code = None

    if not is_hf:
        requested = [
            k
            for k in ("quant_cfg", "auto_quantize_bits", "compress", "sparse_cfg")
            if getattr(args, k, None)
        ]
        if requested:
            raise ValueError(
                f"{', '.join('--' + k for k in requested)} request quantization/sparsity, which "
                f"only applies to the HF backend; with --model {args.model!r} the checkpoint must "
                "already be quantized/sparsified (drop the ModelOpt args)."
            )

    for key in _MODELOPT_ARG_KEYS:
        value = vars(args).pop(key, None)
        if is_hf:
            model_args[key] = value

    args.model_args = model_args


def _enforce_accuracy_gate(output_path, task, lower_bound):
    """Read lm-eval results at output_path; exit non-zero if <task>/acc < lower_bound."""
    # HarnessCLI.execute() returns None, so we recover the result dict from disk.
    files = glob.glob(os.path.join(output_path, "**", "results*.json"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No results*.json under {output_path}")
    # Sort by mtime, not path: a reused output_path nests results under a
    # <model_name>/ dir, and lexical order would pick the wrong run's file.
    with open(max(files, key=os.path.getmtime)) as f:
        scores = json.load(f)["results"].get(task, {})
    # lm-eval keys metrics by filter, e.g. "acc,none"; take acc (never acc_stderr).
    acc = next((float(v) for k, v in scores.items() if k == "acc" or k.startswith("acc,")), None)
    if acc is None:
        raise KeyError(f"acc not found for '{task}' (have: {list(scores)})")
    passed = acc >= lower_bound
    print(
        f"[accuracy_gate] {task}/acc = {acc:.4f} (lower_bound {lower_bound}) -> "
        f"{'PASS' if passed else 'FAIL'}"
    )
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    setup_logging()
    cli = HarnessCLI()
    # The `run` subcommand owns the model/task arguments; extend that parser.
    # `_subparsers` is private API; guard so a future lm-eval refactor surfaces a
    # clear error instead of an opaque AttributeError.
    try:
        run_parser = cli._subparsers.choices["run"]
    except (AttributeError, KeyError) as e:
        raise RuntimeError(
            "Cannot locate lm-eval's `run` subparser; the HarnessCLI internals may "
            f"have changed. Installed lm-eval version: {version('lm_eval')}."
        ) from e
    _add_modelopt_args(run_parser)
    args = cli.parse_args()
    _inject_modelopt_args_into_model_args(args)
    lower_bound = vars(args).pop("accuracy_lower_bound", None)
    output_path = getattr(args, "output_path", None)
    gate_task = None
    if lower_bound is not None:  # fail fast before the (expensive) eval
        if not output_path:
            raise ValueError("--accuracy_lower_bound requires --output_path.")
        raw = args.tasks if isinstance(args.tasks, list) else str(args.tasks or "").split(",")
        tasks = [t.strip() for t in raw if t and t.strip()]
        if len(tasks) != 1:
            raise ValueError(
                f"--accuracy_lower_bound needs exactly one --tasks (got {tasks or 'none'})."
            )
        gate_task = tasks[0]
    cli.execute(args)
    if lower_bound is not None:
        _enforce_accuracy_gate(output_path, gate_task, lower_bound)
