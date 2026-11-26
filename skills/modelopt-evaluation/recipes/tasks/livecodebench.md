# LiveCodeBench v6

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/index.html>

## Params

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_livecodebench
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  # Pin the binary-stdio judge fix (wasiahmad/livecodebench#1).
  pre_cmd: |-
    python -m pip install --quiet --no-cache-dir --no-deps --force-reinstall "git+https://github.com/wasiahmad/livecodebench.git@7b16053825437528ce2e00a58213f0ae4f6e5f92" &&
    python -c "import json, importlib.metadata as m; c = json.loads(m.distribution('livecodebench').read_text('direct_url.json'))['vcs_info']['commit_id']; assert c == '7b16053825437528ce2e00a58213f0ae4f6e5f92', c"
  nemo_evaluator_config:
    config:
      params:
        extra:
          num_repeats: 8
          dataset_split: test_v6_2408_2505
```

Judge pin:

- Keep the `pre_cmd` until the image ships the fix (`26.05.1` still has `fddc0ea`). It needs
  `NEMO_EVALUATOR_TRUST_PRE_CMD=1` and GitHub/PyPI access from the eval node.
- A task-level `pre_cmd` **replaces** `evaluation.pre_cmd`; merge any commands from that one into this one.
- Stock-judge scores are not comparable: that judge fails correct binary-stdio solutions, and the
  pinned one also adds float/JSON output tolerance. Grade baseline and candidate with the same
  judge; re-run older LCB results before comparing.

## Score Extraction

Result (0-100): `livecodebench_pass_at_1_avg-of-N_accuracy`

N is the repeat count.  If the repeat count is unknown, use the highest available `avg-of-N`.
