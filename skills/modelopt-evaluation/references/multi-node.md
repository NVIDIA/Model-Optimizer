# Multi-Node Evaluation Patterns

There are two multi-node patterns. Ask the user which applies:

## Pattern A: Multi-instance (independent instances with HAProxy)

Only if model >120B parameters or user wants more throughput. Explain: "Each node runs an independent deployment instance. HAProxy load-balances requests across all instances."

```yaml
execution:
    num_nodes: 4       # Total nodes
    num_instances: 4   # 4 independent instances → HAProxy auto-enabled
```

## Pattern B: Multi-node single instance (Ray TP/PP across nodes)

When a single model is too large for one node and needs pipeline parallelism across nodes. Use `vllm_ray` deployment config:

```yaml
defaults:
  - deployment: vllm_ray   # Built-in Ray cluster setup (replaces manual pre_cmd)

execution:
    num_nodes: 2           # Single instance spanning 2 nodes

deployment:
    tensor_parallel_size: 8
    pipeline_parallel_size: 2
```

## Pattern A+B combined: Multi-instance with multi-node instances

For very large models needing both cross-node parallelism AND multiple instances:

```yaml
defaults:
  - deployment: vllm_ray

execution:
    num_nodes: 4       # Total nodes
    num_instances: 2   # 2 instances of 2 nodes each → HAProxy auto-enabled

deployment:
    tensor_parallel_size: 8
    pipeline_parallel_size: 2
```

## Common Confusions

- **`num_instances`** controls independent deployment instances with HAProxy. **`data_parallel_size`** controls DP replicas *within* a single instance.
- Global data parallelism is `num_instances x data_parallel_size` (e.g., 2 instances x 8 DP each = 16 replicas).
- With multi-instance, `parallelism` in task config is the total concurrent requests across all instances, not per-instance.
- `num_nodes` must be divisible by `num_instances`.
