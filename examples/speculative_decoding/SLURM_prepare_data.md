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

## Failures and resuming

Workers continue to later shards after per-conversation failures by default. Successful
conversations stay in the output shard, while failures are logged to stderr and recorded in
`<output_shard>.failures`. Failed conversations and partial answers are never written as
training rows. The journal records the conversation ID, error, and whether ordinary resume
will retry it.

Rerun the launch command for the same shard range and output directory to resume. Keep shard
names and input ordering unchanged, since conversation IDs are positions within each shard.
Completed IDs are skipped. Temporary connection, timeout, rate-limit, and server errors remain
retryable. Empty final answers are retryable at positive temperatures, but recorded as rejected
at temperature zero. Unsupported tool roles or calls, malformed conversations, and HTTP 400/422
responses are also recorded as rejected and skipped on ordinary resume. Inspect these
rejections before training; they can indicate bad input or incompatible request settings.

Set `RETRY_FAILED=1` when launching to retry rejected conversations after fixing their input
or request settings. Set `FAIL_ON_ERROR=1` to make any unresolved failures return a nonzero
status after the current shard finishes, stopping that worker before later shards. These
variables enable the generator's `--retry_failed` and `--fail_on_error` flags:

```sh
RETRY_FAILED=1 FAIL_ON_ERROR=1 bash distributed_generate/launch.sh $SLURM_JOB_ID vllm TinyLlama/TinyLlama-1.1B-Chat-v1.0 /data/train/ /data/output /scripts/ 0 10 n1,n2,n3,n4
```

Authentication failures, missing endpoints or models, and unexpected internal or output-write
errors stop workers even without strict mode. The worker enables `--log_empty_conversations`; its
`finished` marker means all inputs are saved or recorded as rejected, not that every input
succeeded. No new marker is written while retryable failures remain. A retry can append rows
after an older marker, so the generator reads the entire output when resuming.

## Combining shards

To combine the shards back

```sh
python3 distributed_generate/sharding_utils.py --input_dir /data/output/ --output_path /data/output.jsonl --combine
```

The combiner ignores the `.failures` journals and completion markers. Review the journals and
filter conversations marked `truncated: true` before using the combined data for training.
