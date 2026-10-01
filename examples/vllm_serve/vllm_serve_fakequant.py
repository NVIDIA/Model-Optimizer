# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# MIT License
#
# Copyright (c) 2023 Deep Cognition and Language Research (DeCLaRe) Lab
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

"""Thin ``vllm`` CLI shim: translates ``--modelopt-*`` flags to env vars, then delegates
to vLLM's real CLI (``vllm.entrypoints.cli.main.main``) unmodified.

Without fakequant settings, the wrapper delegates to
stock vLLM without installing the fakequant worker. Install this as the ``vllm``
console script (see pyproject.toml) so ``vllm serve <model> ...`` works directly.
Add ``--modelopt-*`` flags (or set the env vars) to opt into fakequant.
The wrapper then defaults to ``--worker_cls fakequant_worker.FakeQuantWorker``;
an explicit ``--worker_cls`` still overrides it.
"""

import os
import sys
from pathlib import Path

import vllm
from packaging import version
from vllm_mlflow_utils import MLFLOW_ENV_VARS, add_mlflow_args, resolve_mlflow_args


def _is_missing_entrypoint(error: ModuleNotFoundError, entrypoint: str) -> bool:
    missing_module = error.name
    return missing_module is not None and (
        missing_module == entrypoint or entrypoint.startswith(f"{missing_module}.")
    )


vllm_version = version.parse(vllm.__version__)
if vllm_version <= version.parse("0.11.0"):
    from vllm.utils import FlexibleArgumentParser
else:
    from vllm.utils.argparse_utils import FlexibleArgumentParser


# Env vars to copy from the driver to Ray workers (must match fakequant_worker / vllm_ptq_utils).
# The MLflow ones are settled by resolve_mlflow_args() below, after this list is published:
# Ray reads the values when it creates the actors, so naming them here is enough.
_RAY_ENV_VARS = {
    "QUANT_DATASET",
    "QUANT_CALIB_SIZE",
    "QUANT_CFG",
    "QUANT_FILE_PATH",
    "KV_QUANT_CFG",
    "MODELOPT_STATE_PATH",
    "CALIB_BATCH_SIZE",
    "RECIPE_PATH",
    "TRUST_REMOTE_CODE",
    *MLFLOW_ENV_VARS,
}


def _register_ray_env_vars() -> None:
    try:
        from vllm.executor.ray_distributed_executor import RayDistributedExecutor

        RayDistributedExecutor.ADDITIONAL_ENV_VARS.update(_RAY_ENV_VARS)
    except (ImportError, AttributeError):
        # vLLM v1 Ray: vllm/ray/ray_env.py (get_env_vars_to_copy); merge with any user-set list.
        extra_env_var = "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"
        merged_env_vars = {
            t.strip() for t in os.environ.get(extra_env_var, "").split(",") if t.strip()
        } | _RAY_ENV_VARS
        os.environ[extra_env_var] = ",".join(sorted(merged_env_vars))


def _parser_has_argument(parser, dest: str) -> bool:
    return any(action.dest == dest for action in parser._actions)


def _make_vllm_serve_parser():
    try:
        from vllm.entrypoints.openai.cli_args import make_arg_parser
    except ModuleNotFoundError as error:
        # vLLM 0.29 moved the serve parser out of the OpenAI entrypoint package.
        if not _is_missing_entrypoint(error, "vllm.entrypoints.openai.cli_args"):
            raise
        from vllm.entrypoints.cli.serve import make_arg_parser

    parser = FlexibleArgumentParser(add_help=False)
    make_arg_parser(parser)
    return parser


def _vllm_supports_moe_backend() -> bool:
    return _parser_has_argument(_make_vllm_serve_parser(), "moe_backend")


def _bool_env(key: str) -> bool:
    return os.environ.get(key, "").lower() in ("1", "true", "yes")


def _has_flag(argv: list[str], *names: str) -> bool:
    """Whether argv contains a flag in either --flag or --flag=value form."""
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv for name in names)


def _add_fakequant_args(parser) -> None:
    g = parser.add_argument_group(
        "ModelOpt FakeQuant options",
        description="Each flag falls back to its corresponding env var when not set on the CLI.",
    )
    g.add_argument(
        "--modelopt-quant-cfg",
        default=os.environ.get("QUANT_CFG"),
        help="ModelOpt quantization config name (e.g. FP8_DEFAULT_CFG, INT8_DEFAULT_CFG) [env: QUANT_CFG]",
    )
    g.add_argument(
        "--modelopt-kv-quant-cfg",
        default=os.environ.get("KV_QUANT_CFG"),
        help="KV cache quantization config name [env: KV_QUANT_CFG]",
    )
    g.add_argument(
        "--modelopt-quant-file-path",
        default=os.environ.get("QUANT_FILE_PATH"),
        help=(
            "Path to quantizer_state.pth in a Megatron (MCore) vLLM fakequant export "
            "[env: QUANT_FILE_PATH]"
        ),
    )
    g.add_argument(
        "--modelopt-state-path",
        default=os.environ.get("MODELOPT_STATE_PATH"),
        help=(
            "Path to vllm_fq_modelopt_state.pth in an HF vLLM fakequant export "
            "[env: MODELOPT_STATE_PATH]"
        ),
    )
    g.add_argument(
        "--modelopt-recipe-path",
        default=os.environ.get("RECIPE_PATH"),
        help="Path to a ModelOpt PTQ recipe YAML [env: RECIPE_PATH]",
    )
    g.add_argument(
        "--modelopt-quant-dataset",
        default=os.environ.get("QUANT_DATASET", "cnn_dailymail"),
        help="Calibration dataset name (default: cnn_dailymail) [env: QUANT_DATASET]",
    )
    g.add_argument(
        "--modelopt-quant-calib-size",
        type=int,
        default=int(os.environ.get("QUANT_CALIB_SIZE", 512)),
        help="Number of calibration samples (default: 512) [env: QUANT_CALIB_SIZE]",
    )
    g.add_argument(
        "--modelopt-calib-batch-size",
        type=int,
        default=int(os.environ.get("CALIB_BATCH_SIZE", 1)),
        help="Calibration batch size (default: 1) [env: CALIB_BATCH_SIZE]",
    )


def _fakequant_requested(modelopt_args) -> bool:
    """Mirror the settings that run the fakequant worker's quantize or restore path."""
    return bool(
        modelopt_args.modelopt_quant_cfg
        or modelopt_args.modelopt_kv_quant_cfg
        or modelopt_args.modelopt_state_path
        or modelopt_args.modelopt_recipe_path
    )


def _apply_fakequant_env(args, rest_argv: list) -> None:
    """Translate parsed --modelopt-* CLI args to env vars that fakequant_worker reads."""
    env_map = {
        "QUANT_CFG": args.modelopt_quant_cfg,
        "KV_QUANT_CFG": args.modelopt_kv_quant_cfg,
        "QUANT_FILE_PATH": args.modelopt_quant_file_path,
        "MODELOPT_STATE_PATH": args.modelopt_state_path,
        "RECIPE_PATH": args.modelopt_recipe_path,
        "QUANT_DATASET": args.modelopt_quant_dataset,
        "QUANT_CALIB_SIZE": args.modelopt_quant_calib_size,
        "CALIB_BATCH_SIZE": args.modelopt_calib_batch_size,
    }
    # None means "flag not passed" (string args); skip so an already-exported env var isn't
    # clobbered with "None". The calib-size/batch-size ints always have a value here (argparse
    # defaults from the env vars), so they always pass this check.
    for key, val in env_map.items():
        if val is not None:
            os.environ[key] = str(val)

    # vllm's --trust-remote-code → TRUST_REMOTE_CODE for the tokenizer loader.
    # rest_argv still holds it unparsed (modelopt_parser only knows --modelopt-* flags).
    if "--trust-remote-code" in rest_argv or _bool_env("TRUST_REMOTE_CODE"):
        os.environ["TRUST_REMOTE_CODE"] = "true"


# vLLM's top-level CLI subcommands (vllm.entrypoints.cli.main). Kept as a plain set rather
# than introspected, since introspection would need vllm imported before we know whether this
# invocation even needs it (--help with no other args, etc.).
_VLLM_SUBCOMMANDS = {"serve", "chat", "complete", "bench", "run-batch", "collect-env", "launch"}


def _default_to_serve(rest_argv: list) -> list:
    """Prepend ``serve`` when no vLLM subcommand was given, so ``vllm_serve_fakequant.py
    <model> ...`` keeps working the way the old single-purpose launcher did."""
    if rest_argv and rest_argv[0] not in _VLLM_SUBCOMMANDS and not rest_argv[0].startswith("-"):
        return ["serve", *rest_argv]
    return rest_argv


def _find_serve_model(rest_argv: list) -> str | None:
    """Resolve the model from ``vllm serve`` arguments using vLLM's own parser.

    Options may legally precede the positional model, so scanning for the first non-option
    token would mistake values such as ``--port 8000`` for the model path.
    """
    if not rest_argv or rest_argv[0] != "serve":
        return None

    args, _ = _make_vllm_serve_parser().parse_known_args(rest_argv[1:])
    return getattr(args, "model_tag", None) or getattr(args, "model", None)


def _run_vllm_cli(argv: list[str]) -> None:
    """Delegate arguments to the stock vLLM CLI."""
    sys.argv = ["vllm", *argv]
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


def main():
    argv = sys.argv[1:]
    # Non-serve commands do not consume ModelOpt settings, including env defaults.
    if argv and argv[0] in _VLLM_SUBCOMMANDS and argv[0] != "serve":
        _run_vllm_cli(argv)
        return

    modelopt_parser = FlexibleArgumentParser(add_help=False)
    _add_fakequant_args(modelopt_parser)
    add_mlflow_args(modelopt_parser)
    modelopt_args, rest_argv = modelopt_parser.parse_known_args(sys.argv[1:])
    rest_argv = _default_to_serve(rest_argv)
    if rest_argv and rest_argv[0] != "serve":
        _run_vllm_cli(argv)
        return

    if (modelopt_args.modelopt_quant_cfg or modelopt_args.modelopt_kv_quant_cfg) and (
        modelopt_args.modelopt_recipe_path
    ):
        raise SystemExit(
            "--modelopt-quant-cfg/--modelopt-kv-quant-cfg and --modelopt-recipe-path are "
            "mutually exclusive -- the recipe file already carries the quant_cfg. Pass only one."
        )

    if modelopt_args.modelopt_quant_file_path:
        if modelopt_args.modelopt_state_path:
            raise SystemExit(
                "--modelopt-quant-file-path cannot be combined with --modelopt-state-path; "
                "the full ModelOpt state already contains quantizer state."
            )
        if not (
            modelopt_args.modelopt_quant_cfg
            or modelopt_args.modelopt_kv_quant_cfg
            or modelopt_args.modelopt_recipe_path
        ):
            raise SystemExit(
                "--modelopt-quant-file-path requires --modelopt-quant-cfg, "
                "--modelopt-kv-quant-cfg, or --modelopt-recipe-path to initialize quantizers."
            )

    # Settled before the engine starts, so an unusable tracking URI fails here rather than
    # in a worker that has already loaded the weights.
    modelopt_args.model = _find_serve_model(rest_argv) or "unknown-model"
    resolve_mlflow_args(modelopt_args, modelopt_parser)
    if _fakequant_requested(modelopt_args):
        # vLLM's compile cache is not keyed on serve-time fake quantization.
        os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
        _apply_fakequant_env(modelopt_args, rest_argv)

        # Fakequant only actually runs inside FakeQuantWorker; default to it here so
        # requesting fakequant (quant_cfg/state_path/recipe_path) is enough on its own,
        # without also requiring this flag every time. An explicit --worker_cls still wins.
        if not _has_flag(rest_argv, "--worker-cls", "--worker_cls"):
            rest_argv = [*rest_argv, "--worker_cls", "fakequant_worker.FakeQuantWorker"]

        # Workers (Ray spawn / multi-proc) must be able to import fakequant_worker.
        repo_root = str(Path(__file__).resolve().parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        python_path = os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if repo_root not in python_path:
            os.environ["PYTHONPATH"] = os.pathsep.join([*filter(None, python_path), repo_root])

        _register_ray_env_vars()

        # Match the fakequant launcher default: use the decomposed Triton MoE backend when
        # this vLLM version exposes the option. An explicit user selection still wins.
        if _vllm_supports_moe_backend():
            if not _has_flag(rest_argv, "--moe-backend", "--moe_backend"):
                rest_argv = [*rest_argv, "--moe_backend", "triton"]

    _run_vllm_cli(rest_argv)


if __name__ == "__main__":
    main()
