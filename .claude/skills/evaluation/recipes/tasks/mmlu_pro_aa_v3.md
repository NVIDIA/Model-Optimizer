# MMLU-Pro AA v3

## Task Details

- Task: `simple_evals.mmlu_pro_aa_v3`
- Harness: simple-evals, chat
- Primary metric: task accuracy
- Run time: Medium
- Samples: 1
- Requires `HF_TOKEN`

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: simple_evals.mmlu_pro_aa_v3
  container: nvcr.io/nvidia/eval-factory/simple-evals:26.03
  env_vars:
    HF_TOKEN: host:HF_TOKEN
  nemo_evaluator_config:
    config:
      params:
        extra:
          n_samples: 1
```

## Score Extraction

Inspect the generated `results.yml` for the exact simple-evals group and score
key.
