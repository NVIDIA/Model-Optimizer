#!/bin/bash

#SBATCH -p batch -A coreai_dlalgo_llm --nodes=16 --gpus-per-node=8 --time=04:00:00

export WANDB_API_KEY=e4d035be35c7aaf604aa7bce2d5d3b75b0c47457
### CHANGE THESE ###
IMAGE=/lustre/fsw/portfolios/llmservice/users/mchochowski/images/nemo_26.02.nemotron_3_super.sqsh


options=" \
--student gptoss \
--teacher gptoss   \
--student-checkpoint /lustre/fsw/portfolios/llmservice/users/mchochowski/puzzletron/workspaces/any_model_gpt_oss_20b/mip/puzzle_solutions/stats_num_params_14914757184/solutions--checkpoints/solution_0/ \
--teacher-checkpoint /lustre/fsw/portfolios/llmservice/users/mchochowski/puzzletron/source_model_checkpoints/openai_gpt-oss-120b/   \
--config-file examples/puzzletron/distillation/kd-container-default.yaml \
--tensor-model-parallel-size 8 \
--pipeline-model-parallel-size 6 \
--expert-model-parallel-size 8 \
--expert-tensor-parallel-size 1 \
train.train_iters=1000 checkpoint.save=/lustre/fsw/portfolios/llmservice/users/mchochowski/puzzletron/workspaces/any_model_gpt_oss_20b/kd/puzzle_solutions/stats_num_params_11914757184/from-120B-mbridge-v6/ \
"

run_cmd="python -u examples/puzzletron/distillation/distill.py ${options}"

    # --container-mounts "/lustre:/lustre/,/lustre/fsw/portfolios/llmservice/users/mchochowski/git/ModelOpt/Model-Optimizer-anymodel-Sep/fused_bias_dropout.py:/opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py,/lustre/fsw/portfolios/llmservice/users/mchochowski/git/ModelOpt/Model-Optimizer-anymodel-Sep/state_dict_saver.py:/opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/dist_checkpointing/strategies/state_dict_saver.py,/lustre/fsw/portfolios/llmservice/users/mchochowski/git/ModelOpt/Model-Optimizer-anymodel-Sep/checkpointing.py:/opt/Megatron-Bridge/src/megatron/bridge/training/checkpointing.py,/lustre/fsw/portfolios/llmservice/users/mchochowski/git/ModelOpt/Model-Optimizer-anymodel-Sep/serialization.py:/opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/dist_checkpointing/serialization.py,/lustre/fsw/portfolios/llmservice/users/mchochowski/git/ModelOpt/Model-Optimizer-anymodel-Sep/state_dict_utils.py:/opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/dist_checkpointing/state_dict_utils.py,/lustre/fsw/portfolios/llmservice/users/mchochowski/git/ModelOpt/Model-Optimizer-anymodel-Sep/validation.py:/opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/dist_checkpointing/validation.py,/lustre/fsw/portfolios/llmservice/users/mchochowski/git/ModelOpt/Model-Optimizer-anymodel-Sep/train.py:/opt/Megatron-Bridge/src/megatron/bridge/training/train.py,/lustre/fsw/portfolios/llmservice/users/mchochowski/git/ModelOpt/Model-Optimizer-anymodel-Sep/gpt_oss_bridge.py:/opt/Megatron-Bridge/src/megatron/bridge/models/gpt_oss/gpt_oss_bridge.py,/lustre/fsw/portfolios/llmservice/users/mchochowski/git/ModelOpt/Model-Optimizer-anymodel-Sep/blended_megatron_dataset_builder.py:/opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets/blended_megatron_dataset_builder.py,/lustre/fsw/portfolios/llmservice/users/mchochowski/git/ModelOpt/Model-Optimizer-anymodel-Sep/optimizer_param_scheduler.py:/opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/optimizer_param_scheduler.py" \

srun -l \
    --ntasks-per-node 8 \
    --container-image ${IMAGE} \
    --container-mounts "/lustre:/lustre/,/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/users/mchochowski/git/ModelOpt/Model-Optimizer-puzzletron/_opt_Megatron-Bridge_src_megatron_bridge_models_conversion_model_bridge.py:/opt/Megatron-Bridge/src/megatron/bridge/models/conversion/model_bridge.py" \
    --export=PYTHONPATH=/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM:/lustre/fsw/portfolios/llmservice/users/mchochowski/git/ModelOpt/Model-Optimizer-puzzletron \
    bash -c "pip install fire mip; cd /lustre/fsw/portfolios/llmservice/users/mchochowski/git/ModelOpt/Model-Optimizer-puzzletron && ${run_cmd}"
