# Post-MIP pipelines

`post_mip.flows` defines campaign-specific processing after MIP. Each flow starts
from one MIP run and consists of named, single-input nodes. A node can branch from
any earlier node. Node IDs must be unique across the campaign because they are also
stable metric namespaces.

When at least one flow is configured, Puzzletron replaces the
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

Evaluation nodes publish every finite metric they produce. Downstream lmms-eval
nodes publish results only after every configured task, or every leaf of a
configured task group, has numeric metrics and positive effective sample counts.
Later filters reference metrics as `mip.<metric>` or `<node_id>.<metric>`.

## Node types

- `filter`: metadata-only selection; modes are `top_k`, `threshold`, `pareto`,
  and `aggregate_rank`.
- `manual_filter`: writes a durable review, asks through an interactive run,
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

## Add downstream evaluation to an existing campaign

Keep the campaign's `puzzle_dir` and add a `post_mip.flows` entry whose source
selects the completed MIP run. Select the candidate, materialize it, and pass
that checkpoint to `downstream_evaluation`:

```yaml
post_mip:
  flows:
    runtime-eval:
      source:
        run: runtime-075
        variants: all
        objectives: all
      nodes:
        best_mip:
          type: filter
          mode: top_k
          metric: mip.score
          direction: minimize
          top_k: 1
        materialized:
          type: materialize
          input: best_mip
        lmms_eval:
          type: downstream_evaluation
          input: materialized
          config:
            tasks: [ifeval, gsm8k]
            limit: 128
            topology:
              tensor_parallel_size: 8
              pipeline_parallel_size: 1
              data_parallel_size: 1
              prefill_context_parallel_size: 1
              decode_context_parallel_size: 1
              enable_expert_parallel: false
              gpu_group_size: 8
```

Replace `runtime-075` with a MIP run defined by the campaign and adjust the
tasks, sample limit, and topology for the worker environment. A non-empty
`post_mip.flows` mapping replaces the legacy fixed post-MIP stages. See the
[lmms-eval run configuration](../configs/families/nemotron3/nano_30b_a3b_bf16/runs/lmms_eval.yaml)
for the complete configuration, including model and runtime settings.

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

## Evaluate saved checkpoints

`downstream_evaluation` adapts the generic
[checkpoint evaluator](checkpoint_evaluation.md) to campaign checkpoints and
publishes their task metrics. Add it after any checkpoint-producing node, such
as `materialize` or `global_kd`. Use the standalone
[checkpoint evaluation](checkpoint_evaluation.md) command when campaign
lineage, filtering, and reports are not needed.

Materialization writes a reloadable Hugging Face checkpoint directory. Its
configuration records the realized per-layer block sizes, and its safetensors
contain the physically sliced weights. Evaluation passes that saved directory
unchanged to a fresh `lmms-eval` process backed by vLLM. It does not convert the
AnyModel instance back into an AutoModel instance. Modality-specific profiles
may prepare pinned task adapters and offline dataset snapshots first, but they
delegate checkpoint execution and completion validation to the same evaluator.

Global KD publishes a consolidated Hugging Face checkpoint and preserves the
realized pruning configuration and required tokenizer or processor assets. A
downstream evaluation node after KD therefore uses the same checkpoint contract
as one after materialization. The Qwen 3.5 text and VLM smoke flows evaluate the
selected checkpoint both before and after their short KD stage.

The post-MIP graph does not treat the teacher as a candidate revision. An
evaluation node can set `reference_checkpoint` to evaluate the teacher with the
same task, evaluator version, dataset revision, prompt settings, and sample
limit. Its summary then publishes `candidate.*`, `reference.*`, and `delta.*`
metrics together. Reference evaluation runs once for each candidate revision
handled by that node. Place the reference comparison after filtering when only
the selected candidates need teacher-relative metrics. Without
`reference_checkpoint`, evaluate the teacher separately under the same contract
before comparing results.

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
