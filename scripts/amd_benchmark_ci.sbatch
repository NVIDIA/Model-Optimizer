#!/usr/bin/env bash
#SBATCH --job-name=rmo-ci-bench
#SBATCH --partition=amd-rccl
#SBATCH --gres=gpu:gfx942-mi300x:1
#SBATCH --exclude=splinter-126-016a,splinter-126-008d,ctr-s93-mi300x-21-vm5,ctr-s93-mi300x-21-vm6,ctr-s93-mi300x-21-vm7,ctr-s93-mi300x-21-vm8,ppac-cyxtera-cx62-3,ppac-cyxtera-cx59-2,ppac-cyxtera-cx76-2,ppac-cyxtera-cx76-3
#SBATCH --time=00:30:00
#SBATCH --output=logs/ci-bench-%j.out
#SBATCH --error=logs/ci-bench-%j.err
IMAGE="aisdkshared/private:cluster2026-repro-verl-rocm7.0-20260518"
export DOCKER_CONFIG="/scratch/users/zhdu/.docker-auth"
echo "node=$(hostname)"
WORK="/scratch/users/zhdu/Project/ROCm-Model-Optimizer"
docker run --rm --entrypoint /bin/bash \
    --device=/dev/kfd --device=/dev/dri --ipc=host \
    --group-add video --group-add render \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --network=host \
    -e HIP_VISIBLE_DEVICES=0 \
    -v "${DOCKER_CONFIG}:/root/.docker:ro" \
    -v "${WORK}:/work" \
    "${IMAGE}" -c '
set -e
cd /tmp && tar -xzf /work/rmo-opt.tar.gz && mv rocm-model-optimizer rmo
cd /tmp/rmo && bash scripts/amd_benchmark_ci.sh
'
