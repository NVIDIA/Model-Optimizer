# HF Model Distillation with Megatron-Bridge

For setup steps, see the [parent README](../README.md).

## Run Distillation

   ```bash
   cd /workspace/Model-Optimizer/examples/puzzletron/mbridge_distillation_example/hf_model_distillation
   
   bash distill.sh model.tensor_model_parallel_size=8 model.teacher.tensor_model_parallel_size=8 train.global_batch_size=4 train.micro_batch_size=1 dataset.sequence_length=8192 train.train_iters=5000 logger.log_interval=1

   ```

## Files

- **`import_hf_checkpoint.py`**: Imports HuggingFace models to Megatron-Bridge format
- **`load_megatron_model.py`**: Loads Megatron-Bridge checkpoints
- **`distill.py`**: Main distillation script
- **`distill.sh`**: Shell script to launch distillation
