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

"""Megatron-LM PTQ quantization task with typed configuration.

NOTE — currently NOT wired into SandboxPipeline:
    Under nemo_run/Fiddle YAML loading, dataclass `__post_init__` runs *before*
    nested fields like `config` are populated, so `self.config` is None at that
    point and `materialize_from_config()` returns early. The previous fix —
    a `materialize_from_config()` hook called explicitly by
    `SandboxPipeline.__post_init__` after Fiddle finishes building — was
    removed to keep `core.py` minimal. As a result, typed-config YAMLs of the
    form `_target_: ...MegatronLMQuantizeTask` no longer materialize. Until
    that hook is reinstated, use the raw `script`/`args`/`environment` form in
    YAMLs (see `examples/Qwen/Qwen3-8B/megatron_lm_ptq.yaml`).

    This module is intentionally retained as a reference for the eventual
    re-enablement of typed task configs.

Example YAML (typed config — currently disabled, see note above):

    task_0:
      _target_: common.megatron_lm.quantize.task.MegatronLMQuantizeTask
      config:
        model: Qwen/Qwen3-8B
        quant_cfg: NVFP4_DEFAULT_CFG
        tp: 4
        calib_dataset: abisee/cnn_dailymail
        calib_size: 32
      slurm_config:
        _factory_: "slurm_factory"
        nodes: 1

Example YAML (raw SandboxTask — the supported form today):

    task_0:
      script: common/megatron_lm/quantize/quantize.sh
      args:
        - --calib-dataset-path-or-name /hf-local/abisee/cnn_dailymail
        - --calib-size 32
      environment:
        - MLM_MODEL_CFG: Qwen/Qwen3-8B
        - QUANT_CFG: NVFP4_DEFAULT_CFG
        - TP: "4"
"""

from dataclasses import dataclass

from core import SandboxTask


@dataclass
class MegatronLMQuantizeConfig:
    """Typed configuration for Megatron-LM PTQ quantization.

    Attributes:
        model: HuggingFace model ID (e.g., Qwen/Qwen3-8B).
        quant_cfg: ModelOpt quantization config name (e.g., NVFP4_DEFAULT_CFG).
        tp: Tensor parallelism degree.
        calib_dataset: Calibration dataset path or HuggingFace repo ID.
        calib_size: Number of calibration samples.
        calib_max_sequence_length: Maximum sequence length for calibration samples.
        mmlu_dataset: MMLU evaluation dataset path or HuggingFace repo ID.
        mmlu_fraction: Fraction of MMLU to evaluate (0.0-1.0).
        mmlu_lower_bound: Minimum MMLU score to pass.
        hf_local: Path prefix for local model/dataset storage (with trailing slash).
    """

    model: str = "Qwen/Qwen3-8B"
    quant_cfg: str = "NVFP4_DEFAULT_CFG"
    tp: int = 4
    pp: int = 1
    ep: int = 1
    etp: int = 1
    extra_args: str = ""
    calib_dataset: str = "abisee/cnn_dailymail"
    calib_size: int = 32
    calib_max_sequence_length: int = 512
    mmlu_dataset: str = "cais/mmlu"
    mmlu_fraction: float = 0.01
    mmlu_lower_bound: float = 0.38
    hf_local: str = "/hf-local/"


@dataclass
class MegatronLMQuantizeTask(SandboxTask):
    """PTQ quantization task — converts typed config to args/environment.

    Set `config` to use typed fields. SandboxPipeline calls materialize_from_config()
    after Fiddle has populated all fields, which expands the typed config into the
    plain `script`, `args`, and `environment` SandboxTask fields. `slurm_config`
    is set directly on the task and is not affected.

    If both `config` and `args`/`environment` are set, `config` takes precedence.
    """

    config: MegatronLMQuantizeConfig = None

    def __post_init__(self):
        # Idempotent: also materializes for direct (non-Fiddle) Python construction,
        # where __post_init__ sees `config` already populated as an __init__ kwarg.
        # Under nemo_run/Fiddle YAML loading, `config` may still be None here; the
        # pipeline calls materialize_from_config() again once the build completes.
        self.materialize_from_config()

    def materialize_from_config(self):
        """Expand `self.config` into the plain SandboxTask `script`, `args`, `environment` fields.

        Idempotent. Called by SandboxPipeline.__post_init__ once Fiddle has populated `config`.
        """
        if self.config is None:
            return
        c = self.config
        self.script = self.script or "common/megatron_lm/quantize/quantize.sh"
        args = [
            f"--calib-dataset-path-or-name {c.hf_local}{c.calib_dataset}",
            f"--calib-size {c.calib_size}",
            f"--calib-max-sequence-length {c.calib_max_sequence_length}",
        ]
        if c.extra_args:
            args.append(c.extra_args)
        self.args = args
        self.environment = [
            {"MLM_MODEL_CFG": c.model},
            {"QUANT_CFG": c.quant_cfg},
            {"HF_MODEL_CKPT": f"{c.hf_local}{c.model}"},
            {"MMLU_DATASET": f"{c.hf_local}{c.mmlu_dataset}"},
            {"TP": str(c.tp)},
            {"PP": str(c.pp)},
            {"EP": str(c.ep)},
            {"ETP": str(c.etp)},
            {"MMLU_LOWER_BOUND": str(c.mmlu_lower_bound)},
        ]
