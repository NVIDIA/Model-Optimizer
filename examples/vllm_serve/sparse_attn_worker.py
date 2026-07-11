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

"""vLLM worker lifecycle wiring for ModelOpt attention transforms."""

from vllm.v1.worker.gpu_worker import Worker as BaseWorker

from modelopt.torch.sparsity.attention_sparsity.plugins.vllm_runtime import (
    install_vllm_nvfp4_attention,
    install_vllm_sparse_attention_from_checkpoint,
)

__all__ = ["SparseAttnWorker", "QuantSparseAttnWorker"]  # noqa: RUF022


def _unwrapped_model(worker):
    model = worker.model_runner.model
    return model.unwrap() if hasattr(model, "unwrap") else model


def _print_install_report(policy, report) -> None:
    if report.installed_count:
        if policy == "NVFP4 attention":
            print(
                "[ModelOpt] Installed NVFP4 quant+sparse attention on "
                f"{report.installed_count} layers: {dict(report.backend_counts)}"
            )
        else:
            if report.sparse_algorithm:
                print(f"[ModelOpt] Sparse attention config: algo -> {report.sparse_algorithm}")
            print(
                f"[ModelOpt] Sparse attention: replaced impl on {report.installed_count} "
                f"attention layers: {dict(report.backend_counts)}"
            )
    elif report.sparse_algorithm:
        print(
            f"[ModelOpt] Sparse attention config {report.sparse_algorithm} matched no active "
            "attention layers; vLLM remains unchanged"
        )
    else:
        print(
            "[ModelOpt] No sparse_attention_config found in the checkpoint; "
            "skipping sparse attention. Run examples/llm_sparsity/attention_sparsity/"
            "hf_sa.py to calibrate and export a checkpoint with the config embedded."
        )


class SparseAttnWorker(BaseWorker):
    """Install checkpoint-driven sparse attention after model loading."""

    def load_model(self, *args, **kwargs) -> None:
        """Load the model, then install checkpoint-configured attention."""
        super().load_model(*args, **kwargs)
        report = install_vllm_sparse_attention_from_checkpoint(self.model_runner)
        _print_install_report("Sparse attention", report)


class QuantSparseAttnWorker(BaseWorker):
    """Install fixed NVFP4 attention plus optional checkpoint sparsity."""

    def load_model(self, *args, **kwargs) -> None:
        """Load the model, then install NVFP4 attention."""
        super().load_model(*args, **kwargs)
        report = install_vllm_nvfp4_attention(self.model_runner, sparse_cfg="checkpoint")
        _print_install_report("NVFP4 attention", report)

    def determine_available_memory(self) -> int:
        """Profile memory without compiling the dynamically converted modules."""
        # Sparse-only imports must remain independent of quantization-specific APIs.
        import torch

        from modelopt.torch.quantization.plugins.vllm import disable_compilation

        with torch.inference_mode(), disable_compilation(_unwrapped_model(self)):
            return BaseWorker.determine_available_memory(self)
