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

import argparse
import os
import sys
from datetime import timedelta

import nemo_run as run
from nemo.collections import llm
from nemo.collections.llm.gpt.data import MockDataModule, PreTrainingDataModule
from nemo.collections.llm.modelopt.recipes.distillation_recipe import distillation_recipe
from nemo.collections.llm.modelopt.recipes.prune_recipe import prune_recipe
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

from modelopt.torch.export.plugins.nemo_run import export_most_recent_ckpt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "common")))
from process_climbmix import SUBSET_IDX
from utils import SlurmConfig, create_slurm_executor


def get_args():
    parser = argparse.ArgumentParser(description="NeMo 2.0 Pruning + Distillation flow.")
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name",
        default="prune_distill_flow",
    )
    parser.add_argument(
        "--model-id-or-path",
        type=str,
        help="ID or path of the HF model",
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--base-recipe",
        type=str,
        default="qwen3_8b",
        help=(
            "Choose NeMo 2.0 recipe. Recipes are named in the format of "
            "<model_name>_<model_size>(_<long_sequence_length> or other special settings)"
        ),
    )
    parser.add_argument(
        "--prune-target-num-layers",
        type=int,
        default=24,
        help="Number of model layers to remain after pruning",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help=(
            "Path to the directory to store logs. Best to pass in a non-relative path so that "
            "artifacts are stored in one location."
        ),
        default="logs",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to the preprocessed dataset",
    )
    parser.add_argument(
        "--data-prefixes",
        type=str,
        nargs="*",
        help="Prefixes of the .bin and .idx files in the data directory",
        default=[f"part_{i}_text_document" for i in SUBSET_IDX],
    )
    parser.add_argument(
        "--train-gpus",
        type=int,
        help="Number of GPUs for training",
        default=8,
    )
    parser.add_argument(
        "--nodes",
        type=int,
        help="Number of nodes",
        default=1,
    )
    parser.add_argument(
        "--use-slurm",
        action="store_true",
        help="Run on slurm using run.SlurmExecutor",
    )
    parser.add_argument(
        "--mock-run",
        action="store_true",
        help="Run in mock mode",
    )
    return parser.parse_args()


def main(args):
    # Common items
    exp_dir = os.path.join(args.log_dir, args.experiment)
    model_name = args.base_recipe

    if args.mock_run:
        data = run.Config(
            MockDataModule,
            global_batch_size=DISTILL_GBS,
            micro_batch_size=DISTILL_MBS,
            seq_length=SEQUENCE_LENGTH,
        )
    else:
        if not args.data_dir:
            raise ValueError("--data-dir must be provided unless --mock-run is enabled.")
        tokenizer = run.Config(
            get_nmt_tokenizer,
            library="huggingface",
            model_name=args.model_id_or_path,
        )
        data = run.Config(
            PreTrainingDataModule,
            paths=[f"{args.data_dir}/{prefix}" for prefix in args.data_prefixes],
            seq_length=SEQUENCE_LENGTH,
            tokenizer=tokenizer,
            global_batch_size=DISTILL_GBS,
            micro_batch_size=DISTILL_MBS,
            split="99,1,0",
        )

    # 1. Import and save initial NeMo checkpoint
    initial_model_out = f"{exp_dir}/{model_name}_initial"
    model_module = getattr(llm, model_name)
    import_model = run.Partial(
        llm.import_ckpt,
        model=model_module.model(),
        source=f"hf://{args.model_id_or_path}",
        output_path=initial_model_out,
        overwrite=True,
    )

    # 2. Prune to obtain student
    prune_data = data.clone()
    prune_data.micro_batch_size = PRUNE_MBS
    prune_data.global_batch_size = prune_data.micro_batch_size
    pruned_model_out = f"{exp_dir}/{model_name}_pruned"
    prune = prune_recipe(
        nemo_checkpoint=initial_model_out,
        save_path=pruned_model_out,
    )
    prune.tokenizer_path = args.model_id_or_path
    prune.pruning_config.target_num_layers = args.prune_target_num_layers
    prune.devices = 1
    prune.pp_size = 1
    prune.data = prune_data
    prune.num_train_samples = PRUNE_SAMPLES
    prune.legacy_ckpt = True

    # 3. Distill
    distill = distillation_recipe(
        teacher_model_path=initial_model_out,
        student_model_path=pruned_model_out,
        num_nodes=args.nodes,
        num_gpus_per_node=args.train_gpus,
    )
    distill.data = data
    distill.tokenizer = "data"
    distill.log.log_dir = f"{exp_dir}/distill_log_dir"
    distill.log.ckpt.train_time_interval = run.Config(timedelta, hours=SAVE_CKPT_EVERY_N_HOURS)
    distill.log.ckpt.save_top_k = 2
    distill.optim.config.lr = MAX_LR
    distill.optim.lr_scheduler.min_lr = MIN_LR
    distill.optim.lr_scheduler.warmup_steps = WARMUP_STEPS
    distill.trainer.max_steps = DISTILL_STEPS
    distill.trainer.val_check_interval = VAL_INTERVAL
    distill.trainer.limit_val_batches = VAL_BATCHES
    distill.trainer.strategy.tensor_model_parallel_size = args.train_gpus
    distill.trainer.strategy.ckpt_load_strictness = "log_all"

    # 4. Evaluate MMLU
    if args.use_slurm:
        mmlu_script_path = "examples/nemo_run/common/in_memory_mmlu.py"
    else:
        mmlu_script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../common/in_memory_mmlu.py")
        )
    eval_teacher = run.Script(
        mmlu_script_path,
        entrypoint="python",
        args=["--nemo_ckpt", initial_model_out, "--tensor_parallelism", f"{args.train_gpus}"],
    )
    eval_student = run.Script(
        mmlu_script_path,
        entrypoint="python",
        args=[
            "--finetuned_ckpt_dir",
            distill.log.log_dir,
            "--tensor_parallelism",
            f"{args.train_gpus}",
        ],
    )

    # 5. Export
    export_path = f"{exp_dir}/{model_name}_final"
    export_model = run.Partial(
        export_most_recent_ckpt,
        directory=distill.log.log_dir,
        output_path=export_path,
    )

    # Setup executors
    if args.use_slurm:
        gpu_executor = create_slurm_executor(
            SLURM_CONFIG,
            ntasks_per_node=1,
            num_gpus=1,
            nodes=1,
        )
        multi_gpu_executor = create_slurm_executor(
            SLURM_CONFIG,
            ntasks_per_node=args.train_gpus,
            num_gpus=args.train_gpus,
            nodes=args.nodes,
        )
    else:
        gpu_executor = run.LocalExecutor(launcher="torchrun", ntasks_per_node=1)
        multi_gpu_executor = run.LocalExecutor(launcher="torchrun", ntasks_per_node=args.train_gpus)

    # Execute
    with run.Experiment(exp_dir, log_level="INFO") as exp:
        s1 = exp.add(
            import_model,
            executor=gpu_executor,
            tail_logs=True,
            name="01_import",
        )
        _ = exp.add(
            eval_teacher,
            executor=multi_gpu_executor,
            tail_logs=True,
            name="02a_eval_teacher",
            dependencies=[s1],
        )
        s2 = exp.add(
            prune,
            executor=gpu_executor,
            tail_logs=True,
            name="02b_prune",
            dependencies=[s1],
        )
        s3 = exp.add(
            distill,
            executor=multi_gpu_executor,
            tail_logs=True,
            name="03_distill",
            dependencies=[s2],
        )
        _ = exp.add(
            eval_student,
            executor=multi_gpu_executor,
            tail_logs=True,
            name="04a_eval_student",
            dependencies=[s3],
        )
        # WAR: Export needs access to all GPUs but only 1 task due to bug in NeMo
        multi_gpu_executor.ntasks_per_node = 1  # will throw error if more than 1 task during export
        _ = exp.add(
            export_model,
            executor=multi_gpu_executor,
            tail_logs=True,
            name="04b_export",
            dependencies=[s3],
        )
        exp.run(detach=True)


if __name__ == "__main__":
    args = get_args()

    # # # # # # # # SLURM SETUP # # # # # #
    # # # # # # # MODIFY THIS # # # # # # #
    if args.use_slurm:
        SLURM_CONFIG = SlurmConfig(
            account="",
            partition_gpu="batch",
            partition_cpu="cpu",
            time="HH:MM:SS",
            container_image="nvcr.io/nvidia/nemo:25.09",
            env_vars={
                "HF_TOKEN": "",
            },
            use_local_tunnel=False,
            host="",
            user="",
            container_mounts=[],
            job_dir="/path/to/logs",
        )

    # # # # # # # # # # # # # # # # # # # # # #
    # # # # # CONFIGURABLE PARAMETERS # # # # #
    SEQUENCE_LENGTH = 4096
    PRUNE_MBS = 4
    DISTILL_MBS = 2
    VAL_BATCHES = 32
    MAX_LR = 1e-4
    MIN_LR = 1e-5
    WARMUP_STEPS = 100
    SAVE_CKPT_EVERY_N_HOURS = 3.5
    if args.mock_run:
        PRUNE_SAMPLES = 3
        DISTILL_GBS = 8
        DISTILL_STEPS = 20
        VAL_INTERVAL = 10
    else:
        PRUNE_SAMPLES = 512
        DISTILL_GBS = 768
        NUM_TOKENS = int(90e9)
        DISTILL_STEPS = int(NUM_TOKENS / DISTILL_GBS / SEQUENCE_LENGTH)
        VAL_INTERVAL = 1000
    # # # # # # # # # # # # # # # # # # # # # #

    main(args)
