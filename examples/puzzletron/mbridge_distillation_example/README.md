## Setup Steps

**Note:** Set `$WORKSPACE` to your project root directory before running these commands:

```bash
export WORKSPACE=...
```

1. **Initialize Megatron-Bridge submodules:**

   ```bash
   cd $WORKSPACE/Megatron-Bridge
   git submodule init
   git submodule update
   ```

2. **Start docker container with mounts:**

   ```bash
   submit_job --partition interactive --time 4 \
     --image $WORKSPACE/docker/modelopt_puzzletron_nemo_25_11.sqsh \
     --mounts $WORKSPACE:/workspace,$WORKSPACE/Megatron-Bridge/3rdparty/Megatron-LM:/opt/megatron-lm \
     --interactive --gpu 1
   ```

3. **Run distillation:**

   ```bash
   cd /workspace/Model-Optimizer/examples/puzzletron/mbridge_distillation_example
   
   bash distill.sh model.tensor_model_parallel_size=8 model.teacher.tensor_model_parallel_size=8 train.global_batch_size=4 train.micro_batch_size=1 dataset.sequence_length=8192 train.train_iters=5000 logger.log_interval=1

   ```

**Note:** The mount `/opt/megatron-lm` is required because Megatron-Bridge depends on the Megatron-LM submodule.
