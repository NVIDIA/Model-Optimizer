# Post-MIP pipelines

`post_mip.flows` defines campaign-specific processing after MIP. Each flow starts
from one MIP run and consists of named, single-input nodes. A node can branch from
any earlier node. Node IDs must be unique across the campaign because they are also
stable metric namespaces.

When at least one flow is configured, the campaign orchestrator replaces the
legacy fixed post-MIP stages with these dynamic nodes. Run such campaigns through
`examples/puzzletron/orchestrate.py`; the simple `main.py` stage runner does not
schedule dynamic/manual nodes.

```yaml
post_mip:
  flows:
    production:
      source:
        run: memory-050
        variants: all
        objectives: all
      nodes:
        initial_filter:
          type: filter
          mode: top_k
          metric: mip.score
          direction: minimize
          top_k: {heterogeneous: 28, homogeneous: 100}

        online_eval:
          type: evaluation
          input: initial_filter
          config:
            eval_samples: 128
            block_size: 8192

        best_kl:
          type: filter
          input: online_eval
          mode: top_k
          metric: online_eval.kl_div
          direction: minimize
          top_k: 32

        materialized:
          type: materialize
          input: best_kl

        serving:
          type: aiperf
          input: materialized
          config:
            concurrency: [4]
            input_tokens: 8192
            output_tokens: 1024

        throughput_gate:
          type: filter
          input: serving
          mode: threshold
          metric: serving.request_throughput
          min: 1000

        pareto:
          type: filter
          input: throughput_gate
          mode: pareto
          metrics:
            - {metric: online_eval.kl_div, direction: minimize}
            - {metric: serving.request_throughput, direction: maximize}

        short_kd:
          type: global_kd
          input: pareto
          config:
            max_steps: 128

        manual:
          type: manual_filter
          input: short_kd
          prompt: Pick the candidates for the long distillation run

        original_checkpoint:
          type: materialize
          input: manual
          model_source: origin

        long_kd:
          type: global_kd
          input: original_checkpoint
          config:
            max_steps: 2048
```

Evaluation nodes publish every finite metric they produce. They do not declare
metric lists or cases. Later filters reference metrics as `mip.<metric>` or
`<node_id>.<metric>`.

## Node types

- `filter`: metadata-only selection; modes are `top_k`, `threshold`, `pareto`,
  and `aggregate_rank`.
- `manual_filter`: writes a durable review, asks in an interactive controller,
  and pauses cleanly in non-interactive execution until a decision is supplied.
- `materialize`: converts a config-only candidate into a checkpoint.
- `evaluation`: evaluates either a config-only candidate or a checkpoint and
  publishes all result metrics.
- `aiperf`: benchmarks a checkpoint and publishes all result metrics.
- `downstream_evaluation`: runs `lmms-eval` against a materialized checkpoint
  and publishes task metrics.
- `global_kd`: produces a new checkpoint revision.
- `ptq`: reserved interface; configuring it currently fails plan compilation
  with a clear not-implemented error.

Nodes that require checkpoints never materialize implicitly. Add a `materialize`
node where the transition is needed.

## Lineage and model source

Architectures are deduplicated across MIP and homogeneous results, while all MIP
origins remain attached. Every transformer creates an immutable candidate revision;
every evaluator adds an observation. Earlier observations remain addressable after
later transformations.

`model_source` controls which model artifact a node uses:

- `latest` (default): the input candidate's latest revision.
- `origin`: the original MIP solution/checkpoint.
- a transformer node ID: that node's revision for the same architecture.

Selection still follows `input`; `model_source` only chooses the artifact operated
on. This supports a long KD run selected using short-KD/PTQ results but restarted
from the original candidate.

## Downstream evaluation

`downstream_evaluation` runs `python -m lmms_eval` as a subprocess through
`command_prefix`. The runner passes an argument list directly and does not invoke
a shell. Values in `command_prefix` and `extra_args` are arguments; shell syntax
is not interpreted. Install the pinned evaluator into an isolated environment
rather than the Puzzletron runtime environment, because `lmms-eval==0.7.2` pins
`wandb==0.25.0` and the pinned AutoModel build requires a newer `wandb`:

```bash
python3 -m venv /workspace/.venv-lmms-eval
source /workspace/.venv-lmms-eval/bin/activate
python -m pip install -r examples/puzzletron/requirements-lmms-eval.txt
python -c 'import importlib.metadata as m; assert m.version("lmms-eval") == "0.7.2"'
deactivate

export PUZZLETRON_LMMS_EVAL_PYTHON=/workspace/.venv-lmms-eval/bin/python
```

The runner derives the realized checkpoint path, vLLM topology arguments, task
list, and output path from the campaign config. Use `model_args` only for
non-derived model options such as dtype or maximum model length, and `extra_args`
only for non-reserved `lmms-eval` flags.

## Filters

`top_k` accepts one integer or separate homogeneous/heterogeneous quotas.
`threshold` accepts `min`, `max`, or both. `pareto` retains the nondominated set.
`aggregate_rank` computes a weighted mean rank and retains `top_k`:

```yaml
type: filter
mode: aggregate_rank
top_k: 2
metrics:
  - {metric: downstream.accuracy, direction: maximize, weight: 2}
  - {metric: serving.request_throughput, direction: maximize, weight: 1}
```

Missing or non-finite metrics exclude a candidate and are recorded as the reason.
For a non-interactive manual gate, copy the review's `execution_identity` and the
chosen `revision_ids` into `manual_decision.json`; this prevents a stale decision
from being reused after upstream candidates or node configuration changes.
Each execution publishes immutable observations, candidate sets, raw evaluator
outputs, and transformed checkpoints under
`artifacts/post_mip/nodes/<node_id>/executions/<execution_identity>/`. `current.json`
is only a pointer to the active execution; the node index retains all prior execution
identities. The central registry preserves architecture, revision, origin, and metric
lineage. The campaign report builds its post-MIP DAG and node summaries from the same
configured nodes.

Execution identities include the input candidate-set producer, referenced metric-node
executions, and the resolved `model_source` revisions. Metric and explicit model-source
dependencies are also DAG parents, even when they are not the candidate `input` edge.
