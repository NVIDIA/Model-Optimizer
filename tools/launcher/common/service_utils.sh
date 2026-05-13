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

native_mpi_rank=$OMPI_COMM_WORLD_RANK
native_mpi_local_rank=$OMPI_COMM_WORLD_LOCAL_RANK
# Works with Slurm launching with `--mpi=pmix`
mpi_rank=${PMIX_RANK:-${native_mpi_rank:-${SLURM_PROCID:-0}}}
mpi_local_rank=${PMIX_LOCAL_RANK:-${native_mpi_local_rank:-${SLURM_LOCALID:-0}}}
mpi_rank=${PMIX_RANK:-${native_mpi_rank:-${SLURM_PROCID:-0}}}
mpi_local_rank=${PMIX_LOCAL_RANK:-${native_mpi_local_rank:-${SLURM_LOCALID:-0}}}

FAIL=0
FAIL_EXIT=0

function error_handler {
    local last_status_code=$?
    echo "[ERROR] $1:$2 failed with status $last_status_code." >&2

    if [[ "$mpi_rank" -eq 0 ]]; then
        echo "<REPORT>$1:$2</REPORT>" >&2
    fi
    FAIL=1
    FAIL_EXIT=1
}

function exit_handler {
    if [[ $FAIL_EXIT == 1 ]]; then
        exit 1
    fi
}

function report_result {
    if [[ "$mpi_rank" -eq 0 ]]; then
        echo "<REPORT>$1</REPORT>"
    fi
}

function util_install_extra_dep {
    local _marker=/tmp/.nmm_extra_dep_installed
    if [[ -f "$_marker" ]]; then
        return 0
    fi
    if [[ "$mpi_local_rank" -eq 0 ]]; then
        if ! pip install diskcache; then
            report_result "FAIL: util_install_extra_dep: pip install diskcache failed"
            exit 1
        fi
        # Megatron-LM asserts nvrx >= 0.6.0 on import. nemo containers ship
        # nvrx in two Python envs; `pip` targets the system one, `python -m
        # pip` the uv venv. Uninstall from both, install into the venv.
        pip uninstall -y nvidia-resiliency-ext 2>/dev/null || true
        python -m pip uninstall -y nvidia-resiliency-ext 2>/dev/null || true
        if ! python -m pip install --upgrade "nvidia-resiliency-ext>=0.6.0"; then
            report_result "FAIL: util_install_extra_dep: pip install nvidia-resiliency-ext failed"
            exit 1
        fi
        touch "$_marker"
    else
        local _waited=0
        while [[ ! -f "$_marker" && $_waited -lt 600 ]]; do
            sleep 1
            _waited=$((_waited + 1))
        done
        if [[ ! -f "$_marker" ]]; then
            report_result "FAIL: util_install_extra_dep: timed out waiting for rank-0 install marker"
            exit 1
        fi
    fi
}

LOCAL_NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)
printf "RANK ${mpi_rank} GPU count: ${LOCAL_NUM_GPUS}\n"

# Increase the modelopt version number manually
if [[ "$mpi_local_rank" -eq 0 ]]; then
    echo "__version__ = '1.0.0'" >> ./modules/Model-Optimizer/modelopt/__init__.py
fi
