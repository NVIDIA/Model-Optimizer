# SLURM Prepare Data

For basic parallelization of synthetic data generation we provide some SLURM support.
Assuming a `$SLURM_JOB_ID` is present and nodes, n1, n2, n3, n4 are selected the following is achievable.

Example of allocating 4 nodes for 120 minutes

```sh
salloc  -N4 -A <account> -p <partition>  -J <account>-synthetic:data-gen -t 120
```

Create shards of some given size

```sh
python3 distributed_generate/sharding_utils.py --input_path /data/train.jsonl --output_dir /data/train/ --max_lines_per_shard 10000
```

Run workers on SLURM

```sh
bash distributed_generate/launch.sh $SLURM_JOB_ID vllm TinyLlama/TinyLlama-1.1B-Chat-v1.0 /data/train/ /data/output /scripts/ 0 10 n1,n2,n3,n4 "\"You are a helpful assistant.\""
```

`/scripts/` is the absolute path to `modelopt/examples/speculative_decoding` which contains `server_generate.py` and `distributed_generate`.
This will launch a vllm server (sglang is also available) on each node. Each node will work through 10 shards of data (10\*max_lines_per_shard number of samples).
In this case, the first 40 shards of data will be processed.
To process the next 40 shards

```sh
bash distributed_generate/launch.sh $SLURM_JOB_ID vllm TinyLlama/TinyLlama-1.1B-Chat-v1.0 /data/train/ /data/output /scripts/ 40 10 n1,n2,n3,n4
```

To combine the shards back

```sh
python3 distributed_generate/sharding_utils.py --input_dir /data/output/ --output_path /data/output.jsonl --combine
```
