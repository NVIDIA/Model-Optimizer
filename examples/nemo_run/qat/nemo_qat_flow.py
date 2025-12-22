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

import argparse
import os
import sys

import nemo_run as run
from nemo.collections import llm
from nemo.collections.llm.gpt.data.chat import ChatDataModule
from nemo.collections.llm.modelopt.quantization.quant_cfg_choices import get_quant_cfg_choices
from nemo.collections.llm.modelopt.recipes.distillation_recipe import distillation_recipe
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

from modelopt.torch.export.plugins.nemo_run import export_most_recent_ckpt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "common")))
from utils import SlurmConfig, create_slurm_executor, get_finetune_recipe, read_chat_template


def get_args():
    parser = argparse.ArgumentParser(
        description="""NeMo2.0 QAT/QAD simplified flow. Supports running model locally on 1 node with 8 GPUs
                       or on a Slurm cluster with 1 or more nodes. Runs QAT on Qwen3-8B NVFP4 with the
                       nvidia/OpenScience dataset by default."""
    )
    quant_cfg_choices_list = ["no_quant", *get_quant_cfg_choices()]

    parser.add_argument(
        "--model-name",
        type=str,
        help="Name of the HF model",
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--finetune-recipe",
        type=str,
        default="qwen3_8b",
        help=(
            "Choose NeMo 2.0 recipe. Recipes are named in the format of "
            "<model_name>_<model_size>(_<long_sequence_length> or other special settings)"
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the finetuning chat dataset. Can be either ShareGPT or HuggingFace/OpenAI chat format",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
        default=1e-5,
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Whether to do quantization aware distillation (QAD)",
    )
    parser.add_argument(
        "--hf-tokenizer",
        type=str,
        help="Name of HF model to use for tokenizer.",
        required=False,
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        help="Path to the custom chat template to replace the HF tokenizer default chat template.",
        required=False,
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="nvfp4",
        choices=quant_cfg_choices_list,
        help="Model-Optimizer quantization algorithm",
    )
    parser.add_argument(
        "--use-slurm",
        action="store_true",
        help="Run on slurm using run.SlurmExecutor",
        default=False,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name",
        default="qat_flow_ckpts",
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
        "--ptq-gpus",
        type=int,
        help="Number of GPUs for quantization. Some models require a different number of GPUs for PTQ vs training.",
        default=4,
    )
    parser.add_argument(
        "--train-gpus",
        type=int,
        help="Number of GPUs for training",
        default=8,
    )
    parser.add_argument(
        "--train-nodes",
        type=int,
        help="Number of nodes for training. Does not apply to PTQ (assumes model will fit in 1 node)",
        default=1,
    )
    parser.add_argument(
        "--kv-cache-qformat",
        type=str,
        default="fp8",
        choices=["fp8", "nvfp4"],
        help="KV-cache quantization format",
    )
    parser.add_argument(
        "--enable_kv_cache",
        help="Enables KV-cache quantization",
        action="store_true",
        default=False,
    )
    parser.add_argument("--tensor_parallelism", type=int, default=2)
    parser.add_argument("--pipeline_parallelism", type=int, default=1)
    return parser.parse_args()


def main(args):
    if not args.distill and not args.finetune_recipe:
        raise ValueError("If distillation is not used, --finetune-recipe must be specified")
    model_name = args.finetune_recipe
    model_module = getattr(llm, model_name)
    if not model_name:
        model_name = os.path.basename(args.model_name)
    exp_dir = f"{args.log_dir.rstrip('/')}/{args.experiment}"

    # 1. Process data
    # TODO figure out path
    # LOCALLY common/process.py works
    # On slurm examples/nemo_run/common/process.py works

    openscience_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../common/process_openscience.py")
    )
    openscience_data = run.Script(
        openscience_path
        if not args.use_slurm
        else "examples/nemo_run/common/process_openscience.py",
        entrypoint="python",
        args=["--output-dir", exp_dir],
    )

    # 2. Import Model
    bf16_ckpt_path = f"{exp_dir}/{model_name}-nemo"
    import_model = run.Partial(
        llm.import_ckpt,
        model=model_module.model(),
        source=f"hf://{args.model_name}",
        output_path=bf16_ckpt_path,
        overwrite=True,
    )
    # 3. PTQ
    ptq_model_out = f"{exp_dir}/{model_name}-{args.algorithm}"

    ptq = run.Script(
        "/opt/NeMo/scripts/llm/ptq.py",
        args=[
            "-nc",
            bf16_ckpt_path,
            "-out",
            ptq_model_out,
            "--export_format",
            "nemo",
            "--algorithm",
            args.algorithm,
            "--kv_cache_qformat",
            args.kv_cache_qformat,
            "--enable_kv_cache" if args.enable_kv_cache else "--disable_kv_cache",
            "-ctp",
            f"{args.ptq_gpus}",
        ],
        entrypoint="python",
    )

    # 4. Train
    if not args.hf_tokenizer:
        tokenizer_path = os.path.join(bf16_ckpt_path, "context/nemo_tokenizer")
        tokenizer = run.Config(
            get_nmt_tokenizer,
            library="huggingface",
            model_name=tokenizer_path,
            chat_template=read_chat_template(args.chat_template) if args.chat_template else None,
        )
    else:
        tokenizer = run.Config(
            get_nmt_tokenizer,
            library="huggingface",
            model_name=args.hf_tokenizer,
            chat_template=read_chat_template(args.chat_template) if args.chat_template else None,
        )

    data_path = args.data_path if args.data_path is not None else f"{exp_dir}/openscience_proc"
    data = run.Config(
        ChatDataModule,
        dataset_root=data_path,
        seq_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer,
        global_batch_size=GBS,
        micro_batch_size=MBS,
        use_hf_tokenizer_chat_template=True,
        num_workers=1,
        persistent_workers=True,
    )
    if args.distill:
        train = distillation_recipe(ptq_model_out, bf16_ckpt_path)
    else:
        train = get_finetune_recipe(args.finetune_recipe)
        train.resume.restore_config.path = ptq_model_out
        train.optim.config.lr = args.learning_rate
    train.tokenizer = "data"
    train.data = data
    train.log.log_dir = exp_dir
    train.trainer.val_check_interval = VAL_INTERVAL
    train.trainer.max_steps = TRAIN_STEPS
    train.trainer.devices = args.train_gpus
    train.trainer.num_nodes = args.train_nodes
    train.trainer.limit_val_batches = 32
    train.trainer.strategy.tensor_model_parallel_size = args.tensor_parallelism
    train.trainer.strategy.pipeline_model_parallel_size = args.pipeline_parallelism

    # 5. Export
    export = run.Partial(
        export_most_recent_ckpt, train.log.log_dir, output_path=f"{exp_dir}/{model_name}_hf"
    )
    # 6. Evaluate MMLU

    mmlu_script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../common/in_memory_mmlu.py")
    )
    if args.use_slurm:
        mmlu_script_path = "examples/nemo_run/common/in_memory_mmlu.py"
    eval_ptq = run.Script(
        mmlu_script_path,
        args=["--nemo_ckpt", ptq_model_out, "--tensor_parallelism", f"{args.ptq_gpus}"],
        entrypoint="python",
    )
    eval_bf16 = run.Script(
        mmlu_script_path,
        args=["--nemo_ckpt", bf16_ckpt_path, "--tensor_parallelism", f"{args.ptq_gpus}"],
        entrypoint="python",
    )
    eval_sft = run.Script(
        mmlu_script_path,
        args=["--finetuned_ckpt_dir", exp_dir, "--tensor_parallelism", f"{args.ptq_gpus}"],
        entrypoint="python",
    )

    if args.use_slurm:
        cpu_executor = create_slurm_executor(SLURM_CONFIG)
        ptq_gpu_executor = create_slurm_executor(
            SLURM_CONFIG, num_gpus=args.ptq_gpus, ntasks_per_node=args.ptq_gpus
        )
        train_gpu_executor = create_slurm_executor(
            SLURM_CONFIG, num_gpus=args.train_gpus, ntasks_per_node=args.train_gpus
        )
        single_gpu_executor = create_slurm_executor(SLURM_CONFIG, num_gpus=1, ntasks_per_node=1)
    else:
        cpu_executor = single_gpu_executor = run.LocalExecutor()
        ptq_gpu_executor = run.LocalExecutor(launcher="torchrun", ntasks_per_node=args.ptq_gpus)
        train_gpu_executor = run.LocalExecutor(launcher="torchrun", ntasks_per_node=args.train_gpus)

    with run.Experiment(exp_dir, log_level="INFO") as exp:
        if not args.data_path:
            s0 = exp.add(
                openscience_data, tail_logs=True, name="00_openscience_data", executor=cpu_executor
            )
        # 1. Import BF16 model and evaluate MMLU
        s1 = exp.add(
            import_model, tail_logs=True, name="01_import_model", executor=single_gpu_executor
        )
        exp.add(
            eval_bf16,
            tail_logs=True,
            name="02_mmlu_bf16",
            executor=ptq_gpu_executor,
            dependencies=[s1],
        )

        # 2. PTQ model and evaluate PTQ model
        s2 = exp.add(
            ptq, tail_logs=True, name="03_ptq", executor=ptq_gpu_executor, dependencies=[s1]
        )
        s3 = exp.add(
            eval_ptq,
            tail_logs=True,
            name="04_mmlu_ptq",
            executor=ptq_gpu_executor,
            dependencies=[s2],
        )
        # 3. Train PTQ model (QAT or QAD)
        train_dep = [s3]
        if not args.data_path:
            train_dep.append(s0)
        s4 = exp.add(
            train,
            tail_logs=True,
            name="05_train",
            executor=train_gpu_executor,
            dependencies=train_dep,
        )
        s5 = exp.add(
            eval_sft,
            tail_logs=True,
            name="06_mmlu_sft",
            executor=ptq_gpu_executor,
            dependencies=[s4],
        )
        # WAR: Export needs access to all GPUs but only 1 task due to bug in NeMo
        train_gpu_executor.ntasks_per_node = 1  # will throw error if more than 1 task during export
        exp.add(
            export,
            tail_logs=True,
            name="07_export_hf",
            executor=train_gpu_executor,
            dependencies=[s5],
        )
        exp.run(detach=True)


if __name__ == "__main__":
    args = get_args()

    # # # # # # # # SLURM SETUP # # # # # #
    # # # # # # MODIFY THIS  # # # # # # #
    if args.use_slurm:
        SLURM_CONFIG = SlurmConfig(
            account="",
            partition_gpu="batch",
            partition_cpu="cpu",
            time="04:00:00",
            container_image="nvcr.io/nvidia/nemo:25.07",
            env_vars={
                "HF_TOKEN": "",
            },
            use_local_tunnel=False,
            host="",
            user="",
            container_mounts=[],
            job_dir="/path/to/logs",
            identity=None,
        )

    # # # # # # # # # # # # # # # # # # # # # #
    # # # # # CONFIGURABLE PARAMETERS # # # # #
    SEQUENCE_LENGTH = 4096
    MBS = 1
    GBS = 512
    TRAIN_STEPS = 200
    VAL_INTERVAL = 50
    # # # # # # # # # # # # # # # # # # # # # #

    main(args)
