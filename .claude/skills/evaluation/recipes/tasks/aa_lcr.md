# AA-LCR

## Task Details

- Run time: Long
- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/AA-LCR.html>

## Params

Recommended judge: use Qwen3 235B as an OpenAI-compatible equality-checker
judge, and keep the same judge across comparable runs.

AA-LCR needs long context: plan for roughly 120K input tokens plus 16K
generation tokens. Set deployment `--max-model-len` to at least `131072`, and
use a larger value when the model supports it.

## YAML Fragment

Use this config fragment:

```yaml
deployment:
  extra_args: --max-model-len 131072

evaluation:
  tasks:
    - name: aa_lcr
      container: nvcr.io/nvidia/eval-factory/aa-lcr:26.03
      env_vars:
        HF_TOKEN: host:HF_TOKEN
        JUDGE_API_KEY: host:JUDGE_API_KEY
      nemo_evaluator_config:
        config:
          params:
            extra:
              n_samples: 3
              judge:
                model_id: <qwen3_235b_judge_model_id>
                url: <openai_compatible_judge_chat_completions_url>
                api_key: JUDGE_API_KEY
```

## Score Extraction

AA-LCR accuracy comes from:

```text
results.groups.aa_lcr.metrics.accuracy.scores.accuracy.value
```

AA-LCR stores accuracy as a fraction. Report accuracy in percentage points by
multiplying the value by 100.

For stderr, use the same score entry's stats field and multiply by 100:

```text
results.groups.aa_lcr.metrics.accuracy.scores.accuracy.stats.stderr * 100
```
