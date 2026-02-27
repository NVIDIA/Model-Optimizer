
## GptOss - 20b

With this release Puzzle algorithm supports only experts removal for `Gpt-Oss-20b`.

This model comes as a quantized checkpoint i.e. MoE experts matrices are quantized with _MXFP4_ format.
In the prunning steps puzzle utilizes decompressed model (back to BF16) for statistics and scores computation.
This means, during the conversion to puzzle format we decompress the model and store it as a BF16.
Once the pruning is done i.e. experts to be removed are identified and the process is finished, user may want to get back the _MXFP4_ format of the checkpoint.
To do so, there is an additional script, that takes the original and the pruned checkpoint and outputs pruned checkpoint in _MXFP4_ format.

```bash
python -m modelopt.torch.puzzletron.anymodel.models.gpt_oss_20b.gpt_oss_pruned_to_mxfp4 --student-path /workspaces/any_model_gpt_oss_20b/mip/puzzle_solutions/stats_num_params_18014757184/solutions--checkpoints/solution_0/ --original-path /workspaces/source_model_checkpoints/openai_gpt-oss-20b/ --output-path /workspaces/any_model_gpt_oss_20b/mip/puzzle_solutions/stats_num_params_18014757184/solutions--checkpoints/mxfp4-ckpt/  --num-layers 24
```
