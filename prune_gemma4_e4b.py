# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

r"""Minitron (width) pruning of a Gemma4-E4B language model on a single GPU.

Loads a Megatron dist-checkpoint of the Gemma4-E4B language tower (produced by the Megatron-LM
``examples/gemma4/hf_to_mlm_convert.py`` converter), calibrates on a text dataset, prunes the
requested width hyperparameters (``hidden_size`` / ``ffn_hidden_size``) with ModelOpt's
``mcore_minitron`` algorithm, and saves the pruned dist-checkpoint.

Notes (validated on google/gemma-4-E4B):
  * Attention heads are NOT pruned (Gemma4 uses the hidden-size-only policy, like GDN/MLA).
  * ``ffn_hidden_size`` prunes gracefully un-distilled; ``hidden_size`` is a sharp capacity cliff
    (fine <= ~2.5%, collapses by ~10%) and needs knowledge-distillation recovery for larger cuts.
  * Depth (``num_layers``) pruning is not yet supported for Gemma4 (PLE per-layer slabs + KV bus).
  * Pipeline parallelism is not supported by the fork's Gemma4Model, so this runs on a single GPU.

Running in the NeMo container (nvcr.io/nvidia/nemo:26.06)
--------------------------------------------------------
The container already ships megatron-core, Transformer Engine, and ModelOpt's dependencies. Mount
three things:
  1. this Model-Optimizer repo, overlaying ``modelopt`` into site-packages so these local changes
     take effect (as in examples/megatron_bridge/README.md);
  2. the Megatron-LM checkout on the ``alit/gemma4-e4b-tp-sp`` branch that defines the Gemma4 model,
     mounted *over* the container's bundled Megatron-LM (installed editable from
     ``/opt/Megatron-Bridge/3rdparty/Megatron-LM``) so ``import megatron`` resolves to the fork;
  3. a workspace holding the input Megatron checkpoint and the pruned output.

    export MODELOPT_DIR=/path/to/Model-Optimizer                # branch: kmorabia/prune-gemma4-e4b
    export MEGATRON_LM_DIR=/path/to/Megatron-LM-Ali-Fork        # branch: alit/gemma4-e4b-tp-sp
    export CKPT_DIR=/path/to/checkpoints                        # holds the input MLM ckpt + outputs

    docker run --gpus all --shm-size=16GB --net=host --ulimit memlock=-1 --rm -it \
        -v ${MODELOPT_DIR}:/opt/Model-Optimizer \
        -v ${MODELOPT_DIR}/modelopt:/opt/venv/lib/python3.12/site-packages/modelopt \
        -v ${MODELOPT_DIR}/modelopt_recipes:/opt/venv/lib/python3.12/site-packages/modelopt_recipes \
        -v ${MEGATRON_LM_DIR}:/opt/Megatron-Bridge/3rdparty/Megatron-LM \
        -v ${CKPT_DIR}:/workspace/ckpts \
        -w /opt/Model-Optimizer \
        nvcr.io/nvidia/nemo:26.06 bash

Inside the container, log in to HuggingFace -- the tokenizer (``google/gemma-4-E4B``) and the default
``nemotron-post-training-dataset-v2`` calibration dataset are gated:

    hf auth login --token <your token>

Prerequisite: ``--ckpt`` is a Megatron dist-checkpoint of the Gemma4-E4B *language tower*, produced
from the HF checkpoint by the fork's ``examples/gemma4/hf_to_mlm_convert.py`` converter.

Run it (single GPU, no ``torchrun`` -- the fork's Gemma4Model has no pipeline parallelism):

    python /opt/Model-Optimizer/prune_gemma4_e4b.py \
        --ckpt /workspace/ckpts/gemma4_e4b_base_mlm_ckpt \
        --save /workspace/ckpts/gemma4_e4b_base_pruned \
        --export_config '{"ffn_hidden_size": 9216}' \
        --hf_model google/gemma-4-E4B
"""

import argparse
import json
import os
import sys

# Always run in the nemo:26.06 container with the Gemma4 Megatron-LM fork mounted over the bundled
# MLM at /opt/Megatron-Bridge/3rdparty/Megatron-LM. gemma4_common is a loose script in
# examples/gemma4 (not an installed package), so add its dir to sys.path before importing it; it
# must precede megatron/modelopt (it applies an nvrx version shim before megatron loads).
# isort: off
sys.path.insert(0, "/opt/Megatron-Bridge/3rdparty/Megatron-LM/examples/gemma4")

import gemma4_common
from megatron.core import dist_checkpointing
from megatron.core.models.gemma4.gemma4_layer_specs import (
    get_gemma4_layer_local_spec,
    get_gemma4_layer_with_transformer_engine_spec,
)
from transformers import AutoTokenizer

import modelopt.torch.opt as mto
import modelopt.torch.prune as mtp
from modelopt.torch.utils.plugins.megatron_calibration import (
    get_megatron_calibration_forward_loop,
)

# isort: on


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="Input Gemma4-E4B Megatron dist-checkpoint.")
    parser.add_argument("--save", required=True, help="Output path for the pruned dist-checkpoint.")
    parser.add_argument(
        "--export_config",
        required=True,
        help='Target width as JSON, e.g. \'{"hidden_size": 2304, "ffn_hidden_size": 9216}\'. '
        "Attention heads and num_layers are not pruned for Gemma4.",
    )
    parser.add_argument(
        "--hf_model",
        default="google/gemma-4-E4B",
        help="HF model id for the tokenizer (calibration).",
    )
    parser.add_argument(
        "--spec", choices=["local", "te"], default="local", help="Gemma4 layer spec."
    )
    parser.add_argument("--calib_dataset", default="nemotron-post-training-dataset-v2")
    parser.add_argument("--calib_samples", type=int, default=512)
    parser.add_argument("--calib_batch_size", type=int, default=16)
    parser.add_argument("--calib_seq_length", type=int, default=512)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Load the checkpoint, calibrate, width-prune, and save the pruned checkpoint."""
    export_config = json.loads(args.export_config)
    assert isinstance(export_config, dict) and export_config, (
        "--export_config must be a non-empty dict"
    )

    spec_fn = (
        get_gemma4_layer_with_transformer_engine_spec
        if args.spec == "te"
        else get_gemma4_layer_local_spec
    )

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    gemma4_common._init_distributed()  # single process (DDP=TP=PP=CP=1)
    config = gemma4_common._make_config()
    model = gemma4_common._build_model(spec_fn, config)

    # Drop TE _extra_state keys from the load request (absent in local-spec ckpts, defaulted by TE).
    sharded_sd = {k: v for k, v in model.sharded_state_dict().items() if "_extra_state" not in k}
    model.load_state_dict(dist_checkpointing.load(sharded_sd, args.ckpt), strict=False)
    model.eval()

    # Only apply the chat template when the tokenizer has one (base/PT models do not).
    use_chat_template = getattr(tokenizer, "chat_template", None) is not None
    forward_loop = get_megatron_calibration_forward_loop(
        tokenizer,
        dataset_name=args.calib_dataset,
        num_samples=args.calib_samples,
        seq_length=args.calib_seq_length,
        batch_size=args.calib_batch_size,
        pack=True,
        apply_chat_template=use_chat_template,
    )

    # Manual pruning: fine (64) search-space granularity for every dimension, so any export_config
    # value that is a multiple of 64 is a valid choice.
    ss_config = mtp.mcore_minitron.get_mcore_minitron_config(
        hidden_size_divisor=64, ffn_hidden_size_divisor=64
    )
    print(f"Pruning Gemma4-E4B ({args.spec} spec) with export_config={export_config}", flush=True)
    model, _ = mtp.prune(
        model,
        mode=[("mcore_minitron", ss_config)],  # type: ignore[arg-type]
        constraints={"export_config": export_config},
        dummy_input=None,
        config={"forward_loop": forward_loop},
    )
    # Homogeneous checkpoint -> drop the modelopt state.
    if mto.ModeloptStateManager.has_state_for_mode_type("prune", model=model):
        mto.ModeloptStateManager.remove_state(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Pruned config: hidden_size={model.config.hidden_size} "
        f"ffn_hidden_size={model.config.ffn_hidden_size} | params={n_params / 1e9:.2f}B",
        flush=True,
    )

    os.makedirs(args.save, exist_ok=True)
    dist_checkpointing.save(model.sharded_state_dict(), args.save)
    print(f"Saved pruned Gemma4-E4B checkpoint -> {args.save}", flush=True)
    print(
        "To reload, build the model with the pruned hidden_size/ffn_hidden_size above.", flush=True
    )


if __name__ == "__main__":
    main(get_args())
