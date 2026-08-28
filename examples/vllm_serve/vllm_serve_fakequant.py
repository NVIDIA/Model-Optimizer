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

With no ``--modelopt-*`` flag (and none of QUANT_CFG/KV_QUANT_CFG/MODELOPT_STATE_PATH/
RECIPE_PATH set in the environment, nor their sidecar files auto-detected next to a local
model dir), this is a byte-for-byte passthrough to stock vLLM -- fakequant machinery is
never touched. Install this as the ``vllm`` console script (see pyproject.toml) so
``vllm serve <model> ...`` works directly; add ``--modelopt-*`` flags (or set the env vars)
to opt into fakequant -- ``--worker_cls fakequant_worker.FakeQuantWorker`` is then defaulted
automatically (an explicit ``--worker_cls`` still overrides it).
"""

import os
import sys
from pathlib import Path

import vllm
from packaging import version
from vllm_mlflow_utils import MLFLOW_ENV_VARS, add_mlflow_args, resolve_mlflow_args

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


def _vllm_supports_moe_backend() -> bool:
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    probe = FlexibleArgumentParser(add_help=False)
    make_arg_parser(probe)
    return _parser_has_argument(probe, "moe_backend")


def _bool_env(key: str) -> bool:
    return os.environ.get(key, "").lower() in ("1", "true", "yes")


def _find_flag_value(argv: list, *names: str) -> str | None:
    """Return the value of the first matching --flag/--flag=value in argv, else None."""
    for i, a in enumerate(argv):
        for name in names:
            if a == name:
                return argv[i + 1] if i + 1 < len(argv) else None
            if a.startswith(name + "="):
                return a.split("=", 1)[1]
    return None


def _add_fakequant_args(parser) -> None:
    g = parser.add_argument_group(
        "ModelOpt FakeQuant options",
        description="Each flag falls back to its corresponding env var when not set on the CLI.",
    )
    g.add_argument("--modelopt-quant-cfg", default=os.environ.get("QUANT_CFG"),
                   help="ModelOpt quantization config name (e.g. FP8_DEFAULT_CFG, INT8_DEFAULT_CFG) [env: QUANT_CFG]")
    g.add_argument("--modelopt-kv-quant-cfg", default=os.environ.get("KV_QUANT_CFG"),
                   help="KV cache quantization config name [env: KV_QUANT_CFG]")
    g.add_argument("--modelopt-quant-file-path", default=os.environ.get("QUANT_FILE_PATH"),
                   help="Path to amax / quantizer state file (.pt). Auto-detected as "
                        "<model_dir>/quantizer_state.pth for a local model directory if omitted "
                        "[env: QUANT_FILE_PATH]")
    g.add_argument("--modelopt-state-path", default=os.environ.get("MODELOPT_STATE_PATH"),
                   help="Path to full ModelOpt state checkpoint (.pt). Auto-detected as "
                        "<model_dir>/vllm_fq_modelopt_state.pth for a local model directory if "
                        "omitted [env: MODELOPT_STATE_PATH]")
    g.add_argument("--modelopt-recipe-path", default=os.environ.get("RECIPE_PATH"),
                   help="Path to a quantization recipe file, or a Megatron export's "
                        "per-quantizer resolved config YAML (auto-translated to vLLM naming). "
                        "Auto-detected as <model_dir>/vllm_fq_quantizer_state.yaml for a local "
                        "model directory if omitted [env: RECIPE_PATH]")
    g.add_argument("--modelopt-quant-dataset", default=os.environ.get("QUANT_DATASET", "cnn_dailymail"),
                   help="Calibration dataset name (default: cnn_dailymail) [env: QUANT_DATASET]")
    g.add_argument("--modelopt-quant-calib-size", type=int, default=int(os.environ.get("QUANT_CALIB_SIZE", 512)),
                   help="Number of calibration samples (default: 512) [env: QUANT_CALIB_SIZE]")
    g.add_argument("--modelopt-calib-batch-size", type=int, default=int(os.environ.get("CALIB_BATCH_SIZE", 1)),
                   help="Calibration batch size (default: 1) [env: CALIB_BATCH_SIZE]")
    

def _fakequant_requested(modelopt_args) -> bool:
    """Mirrors the gate in fakequant_worker.compile_or_warm_up_model: these are the
    only settings that actually cause mtq.quantize (and the structural module
    replacement it triggers) to run."""
    return bool(
        modelopt_args.modelopt_quant_cfg
        or modelopt_args.modelopt_kv_quant_cfg
        or modelopt_args.modelopt_state_path
        or modelopt_args.modelopt_recipe_path
    )


def _autodetect_fakequant_paths(args) -> None:
    """Fill in --modelopt-state-path / --modelopt-recipe-path / --modelopt-quant-file-path
    from the model dir's standard export sidecar files, when unset. state-path wins over
    recipe-path if both are present; quant-file-path (amax override) only applies when no
    state path is in play.
    """
    model = args.model
    manual_ptq_requested = bool(
        args.modelopt_quant_cfg or args.modelopt_kv_quant_cfg or args.modelopt_recipe_path
    )
    if not args.modelopt_state_path and not manual_ptq_requested and os.path.exists(f"{model}/vllm_fq_modelopt_state.pth"):
        args.modelopt_state_path = str(Path(model) / "vllm_fq_modelopt_state.pth")

    elif not args.modelopt_quant_file_path and not args.modelopt_state_path:
        if os.path.exists(f"{model}/quantizer_state.pth"):
            args.modelopt_quant_file_path = str(Path(model) / "quantizer_state.pth")
        if os.path.exists(f"{model}/vllm_fq_quantizer_state.yaml"):
            args.modelopt_recipe_path = str(Path(model) / "vllm_fq_quantizer_state.yaml")

def _apply_fakequant_env(args, rest_argv: list) -> None:
    """Translate parsed --modelopt-* CLI args to env vars that fakequant_worker reads."""
    env_map = {
        "QUANT_CFG":              args.modelopt_quant_cfg,
        "KV_QUANT_CFG":           args.modelopt_kv_quant_cfg,
        "QUANT_FILE_PATH":        args.modelopt_quant_file_path,
        "MODELOPT_STATE_PATH":    args.modelopt_state_path,
        "RECIPE_PATH":            args.modelopt_recipe_path,
        "QUANT_DATASET":          args.modelopt_quant_dataset,
        "QUANT_CALIB_SIZE":       args.modelopt_quant_calib_size,
        "CALIB_BATCH_SIZE":       args.modelopt_calib_batch_size,
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
_VLLM_SUBCOMMANDS = {"serve", "chat", "complete", "bench", "run-batch", "collect-env"}


def _default_to_serve(rest_argv: list) -> list:
    """Prepend ``serve`` when no vLLM subcommand was given, so ``vllm_serve_fakequant.py
    <model> ...`` keeps working the way the old single-purpose launcher did."""
    if rest_argv and rest_argv[0] not in _VLLM_SUBCOMMANDS and not rest_argv[0].startswith("-"):
        return ["serve", *rest_argv]
    return rest_argv


def _find_serve_model(rest_argv: list) -> str | None:
    """Best-effort recovery of ``vllm serve <model>``'s positional model, for mlflow's
    default experiment name. Only meaningful under the ``serve`` subcommand; any other
    subcommand (bench, chat, run-batch, ...) returns ``None``."""
    if not rest_argv or rest_argv[0] != "serve":
        return None
    for tok in rest_argv[1:]:
        if not tok.startswith("-"):
            return tok
    return _find_flag_value(rest_argv, "--model")


def main():
    modelopt_parser = FlexibleArgumentParser(add_help=False)
    _add_fakequant_args(modelopt_parser)
    add_mlflow_args(modelopt_parser)
    modelopt_args, rest_argv = modelopt_parser.parse_known_args(sys.argv[1:])
    rest_argv = _default_to_serve(rest_argv)

    if (modelopt_args.modelopt_quant_cfg or modelopt_args.modelopt_kv_quant_cfg) and (
        modelopt_args.modelopt_recipe_path
    ):
        raise SystemExit(
            "--modelopt-quant-cfg/--modelopt-kv-quant-cfg and --modelopt-recipe-path are "
            "mutually exclusive -- the recipe file already carries the quant_cfg. Pass only one."
        )

    # Settled before the engine starts, so an unusable tracking URI fails here rather than
    # in a worker that has already loaded the weights.
    modelopt_args.model = _find_serve_model(rest_argv) or "unknown-model"
    resolve_mlflow_args(modelopt_args, modelopt_parser)
    _autodetect_fakequant_paths(modelopt_args)
    print(f"Modelopt args: {modelopt_args}")
    print(f"Fakequant requested: {_fakequant_requested(modelopt_args)}")
    if _fakequant_requested(modelopt_args):
        print(f"Fakequant requested for model: {modelopt_args.model}")
        _apply_fakequant_env(modelopt_args, rest_argv)

        # Fakequant only actually runs inside FakeQuantWorker; default to it here so
        # requesting fakequant (quant_cfg/state_path/recipe_path) is enough on its own,
        # without also requiring this flag every time. An explicit --worker_cls still wins.
        if _find_flag_value(rest_argv, "--worker-cls", "--worker_cls") is None:
            rest_argv = [*rest_argv, "--worker_cls", "fakequant_worker.FakeQuantWorker"]

        # Workers (Ray spawn / multi-proc) must be able to import fakequant_worker.
        repo_root = str(Path(__file__).resolve().parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + repo_root

        _register_ray_env_vars()

        # ModelOpt expert fakequant needs a decomposed MoE backend so both expert
        # GEMMs are visible during calibration -- but only when weight quantization
        # (QUANT_CFG) is actually requested; a KV-cache-only quant_cfg never touches
        # expert weights, and forcing a backend choice is wrong when the user picked
        # one deliberately (e.g. a dense model, where moe_backend is moot).
        if modelopt_args.modelopt_quant_cfg and _vllm_supports_moe_backend():
            moe_backend_value = _find_flag_value(rest_argv, "--moe-backend", "--moe_backend")
            if moe_backend_value is None:
                raise SystemExit(
                    "QUANT_CFG/--modelopt-quant-cfg is set, but no --moe_backend was given. "
                    "ModelOpt expert fakequant needs a decomposed MoE backend (both expert "
                    "GEMMs must be visible during calibration) -- if your model has MoE "
                    "experts, rerun with --moe_backend triton. If it's a dense model, pass "
                    "any --moe_backend value explicitly to acknowledge and skip this check."
                )

    sys.argv = ["vllm", *rest_argv]
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
