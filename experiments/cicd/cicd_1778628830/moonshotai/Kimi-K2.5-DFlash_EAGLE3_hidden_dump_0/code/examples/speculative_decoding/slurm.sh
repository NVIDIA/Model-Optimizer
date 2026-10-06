#!/bin/bash

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

#SBATCH -A {account}
#SBATCH --job-name={job_name}
#SBATCH --nodes={num_nodes} --ntasks-per-node=1 --gpus-per-node={num_gpus_per_node}
#SBATCH -p {partition}
#SBATCH -t {time_limit}

CONTAINER_IMAGE={container_image}
WORK_DIR={path_to_modelopt}

CONTAINER_MOUNT="${WORK_DIR}:/modelopt"

OUTPUT_DIR={path_to_output_dir}
MODEL={path_to_model_dir}
DATA={path_to_data_dir}
OFFLINE_DATA={path_to_offline_data_dir}

CMD="./launch_train.sh --model $MODEL \
            --output_dir $OUTPUT_DIR \
            --data $DATA \
            --num_epochs 1 \
            --train_bs 1 \
            --lr 1e-4 \
            --eagle_config eagle_config.json \
            --training_seq_len 4096 \
            --save_steps 1000 \
            --estimate_ar True \
            --disable_tqdm True \
            --offline-data $OFFLINE_DATA \
            --num_nodes $SLURM_NNODES \
            --head_node_ip $head_node_ip \
"

srun -l \
    --mpi=pmix \
    --output=%x_%j_$DATETIME.log \
    --container-workdir "/modelopt/examples/speculative_decoding" \
    --container-image ${CONTAINER_IMAGE} --container-mounts ${CONTAINER_MOUNT} \
    bash -lc "$CMD"

set +x
