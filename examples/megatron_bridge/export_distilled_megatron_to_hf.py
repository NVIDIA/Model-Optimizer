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
"""Convert a full-precision distilled Megatron checkpoint (produced by distill.py) to HuggingFace.

Two mechanisms, dispatched on the model type:

  - LLM (Homogeneous or Puzzletron Heterogeneous): the full model is on disk, so it is exported directly with
    ``AutoBridge.export_ckpt`` (which reads the checkpoint's actual per-layer shapes and therefore
    handles both homogeneous and heterogeneous students).
  - VLM: only the ``language_model`` submodule is distilled and checkpointed, so the full VLM is
    reassembled in memory -- vision tower + projector from the original HF model (--student_hf_path),
    the distilled language model from the checkpoint -- and written with ``AutoBridge.save_hf_weights``.

These two helpers (``export_llm_to_hf`` / ``save_vlm_to_hf``) are also reused by distill.py for its
final-checkpoint export.

Example LLM (Homogeneous or Puzzletron Heterogeneous):

    torchrun --nproc_per_node 1 export_distilled_megatron_to_hf.py \
        --student_hf_path Qwen/Qwen3-0.6B \
        --megatron_path /tmp/distill-out/checkpoints/iter_0000500 \
        --hf_export_path /tmp/distilled-hf_iter_0000500

Example VLM (checkpoint reshards on load, so TP/PP/EP need not match training):

    torchrun --nproc_per_node 1 export_distilled_megatron_to_hf.py \
        --student_hf_path Qwen/Qwen3-VL-4B-Instruct \
        --megatron_path /tmp/distill-out/checkpoints/iter_0000500 \
        --hf_export_path /tmp/distilled-vlm-hf_iter_0000500

See `README.md` in this directory for more details.
"""

import argparse

import torch
from megatron.bridge import AutoBridge
from transformers import AutoConfig

import modelopt.torch.utils.distributed as dist
from modelopt.torch.export import copy_hf_ckpt_remote_code
from modelopt.torch.utils import print_args, print_rank_0
from modelopt.torch.utils.plugins.mbridge import (
    load_mbridge_model_from_hf,
    load_modelopt_megatron_checkpoint,
)


def export_llm_to_hf(
    megatron_path: str,
    hf_export_path: str,
    student_hf_path: str,
    template_hf: str | None = None,
    trust_remote_code: bool = False,
) -> None:
    """Export a LLM (Homogeneous or Puzzletron Heterogeneous) Megatron checkpoint to HF.

    Args:
        megatron_path: Megatron checkpoint directory (an ``iter_*`` dir or its parent).
        hf_export_path: Directory to write the HuggingFace checkpoint to.
        student_hf_path: Student HF model used for the exported config / tokenizer.
        template_hf: Reference HF model with a homogeneous architecture, used as the export template
            for a heterogeneous (Puzzletron/NAS) student. Defaults to ``student_hf_path`` (correct for
            homogeneous students).
        trust_remote_code: Whether to trust remote code when loading the HF model.
    """
    # TODO: unify with save_vlm_to_hf's in-memory export path. This LLM path re-loads the checkpoint
    # from disk via export_ckpt (which reads the actual per-layer shapes, so it handles heterogeneous
    # Puzzletron/NAS students); an in-memory export would need to rebuild the (possibly heterogeneous)
    # student first.
    export_bridge = AutoBridge.from_hf_pretrained(
        template_hf or student_hf_path, trust_remote_code=trust_remote_code
    )
    export_bridge.export_ckpt(
        megatron_path=megatron_path, hf_path=hf_export_path, show_progress=True, strict=True
    )
    # Config / tokenizer come from the student definition (handles local paths and HF model IDs).
    AutoConfig.from_pretrained(
        student_hf_path, trust_remote_code=trust_remote_code
    ).save_pretrained(hf_export_path)


def save_vlm_to_hf(
    full_model,
    hf_export_path: str,
    student_hf_path: str,
    trust_remote_code: bool = False,
) -> None:
    """Write an in-memory full VLM (distilled LM already in place) to HF format.

    Only the language model is distilled; the vision tower / projector are the original weights, so
    the original VLM config / tokenizer / remote code are reused and only the weights are written.
    ``full_model.language_model`` must already be a plain module (any KD wrapper stripped by the
    caller). Requires the model-parallel groups to be initialized (for the weight gather).

    Args:
        full_model: The in-memory full VLM with the distilled language model in place.
        hf_export_path: Directory to write the HuggingFace checkpoint to.
        student_hf_path: Original VLM HF model providing config / tokenizer / remote code.
        trust_remote_code: Whether to trust remote code when loading the HF model.
    """
    export_bridge = AutoBridge.from_hf_pretrained(
        student_hf_path, trust_remote_code=trust_remote_code
    )
    export_bridge.hf_pretrained.save_artifacts(hf_export_path)
    export_bridge.save_hf_weights([full_model], hf_export_path)
    copy_hf_ckpt_remote_code(student_hf_path, hf_export_path)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--student_hf_path",
        type=str,
        required=True,
        help="Student HF model (used for the exported config / tokenizer, and, for VLMs, the vision "
        "tower / projector weights). Must match the model distilled by distill.py.",
    )
    parser.add_argument(
        "--megatron_path",
        type=str,
        required=True,
        help="Distilled Megatron checkpoint to convert (an iter_* directory or its parent).",
    )
    parser.add_argument(
        "--hf_export_path",
        type=str,
        required=True,
        help="Directory to write the exported HuggingFace checkpoint to.",
    )
    parser.add_argument(
        "--student_hf_model",
        type=str,
        default=None,
        help="Reference HF model with a homogeneous architecture, used as the export template for a "
        "heterogeneous (Puzzletron/NAS) student's weights. Defaults to --student_hf_path, which is "
        "correct for homogeneous students; unused for VLMs.",
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep_size", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--cp_size", type=int, default=1, help="Context parallel size")

    args = parser.parse_args()
    print_args(args)

    return args


def main(args: argparse.Namespace):
    is_vlm = hasattr(
        AutoConfig.from_pretrained(args.student_hf_path, trust_remote_code=args.trust_remote_code),
        "vision_config",
    )

    if is_vlm:
        # Build the full VLM (vision tower / projector + original LM from HF), then overwrite the LM
        # with the distilled checkpoint weights, then export the assembled VLM.
        print_rank_0(
            f"Reassembling distilled VLM and exporting to HF format at {args.hf_export_path}"
        )
        _bridge, _provider, _model, full_model, _tokenizer = load_mbridge_model_from_hf(
            hf_model_name_or_path=args.student_hf_path,
            trust_remote_code=args.trust_remote_code,
            provider_overrides={
                "tensor_model_parallel_size": args.tp_size,
                "pipeline_model_parallel_size": args.pp_size,
                "expert_model_parallel_size": args.ep_size,
                "context_parallel_size": args.cp_size,
                # VLMs run with sequence parallelism off (see distill.py).
                "sequence_parallel": False,
                "pipeline_dtype": torch.bfloat16,
            },
            init_model_parallel=True,
            load_weights=True,  # vision tower / projector + original LM; the LM is overwritten below
        )
        # Load only the distilled language-model weights (skip ModelOpt-state restore -- the kd_loss
        # mode / teacher are irrelevant for export and would otherwise require a teacher model).
        load_modelopt_megatron_checkpoint(
            [full_model.language_model], args.megatron_path, restore_modelopt_state=False
        )
        save_vlm_to_hf(
            full_model,
            args.hf_export_path,
            args.student_hf_path,
            trust_remote_code=args.trust_remote_code,
        )
        print_rank_0(f"Saved distilled VLM to {args.hf_export_path} in HF format")
    else:
        print_rank_0(f"Exporting distilled checkpoint to HF format at {args.hf_export_path}")
        # Save rank before destroying process group (dist.rank() won't work after destruction).
        is_rank_0 = dist.rank() == 0
        # export_ckpt creates its own temporary process group; destroy this one first so cleanup
        # does not hang on a barrier once rank 0 has left.
        dist.cleanup()
        if is_rank_0:
            export_llm_to_hf(
                megatron_path=args.megatron_path,
                hf_export_path=args.hf_export_path,
                student_hf_path=args.student_hf_path,
                template_hf=args.student_hf_model,
                trust_remote_code=args.trust_remote_code,
            )
        print_rank_0(f"Exported HuggingFace checkpoint to {args.hf_export_path}")


if __name__ == "__main__":
    dist.setup()
    args = get_args()
    try:
        main(args)
    finally:
        dist.cleanup()
