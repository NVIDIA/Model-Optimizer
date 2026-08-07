# Historical Puzzletron Campaign Reports

The two checked-in reports were produced in an earlier experimental state and
have not been reproduced on current code. They preserve historical observations;
they are not model-support claims. The
[historical campaign manifest](../reports/historical_campaigns.yaml) records the
fields checked against the reports and current configuration files.

## Report summary

| Model | Historical report | Current configuration relationship | Static findings and reproduction status |
|---|---|---|---|
| Nemotron-3 Nano 30B-A3B | [Campaign report](../reports/nemotron3_nano_30b_a3b.html): sequence length 8,192; parameter, runtime, and memory MIP profiles; stages marked complete through zero-shot evaluation plus global-distillation sanity | [default.yaml](../configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml) is a current-code migration, not the executed configuration | Not reproduced on current code; the report contains 176 slicing-equivalence findings and 2 descriptor-realization-gate findings; AIPerf, full global distillation, and post-distillation evaluation remain pending |
| Qwen3.5-9B | [Campaign report](../reports/qwen3p5_9b.html): sequence length 16,384; runtime MIP profile; stages marked complete through post-distillation evaluation | [default.yaml](../configs/families/qwen3_5/qwen3p5_9b/runs/default.yaml) is a reconstruction; the report used 23 overrides and embedding widths 4,096, 3,840, and 3,584, while the current entry retains only 4,096 | Not reproduced on current code; the report contains 101 slicing-equivalence findings |

## What this proves

| Question | Answer |
|---|---|
| Do the checked-in reports contain the listed model, configuration, stage, and warning metadata? | Yes; a CPU test parses and checks those static fields. |
| Do the current configurations reproduce the executed configurations? | No; one is a migration and one is a reconstruction. |
| Have these campaigns been rerun on current code? | No. |
| Do the reports establish physical-slicing correctness or model support? | No; unresolved slicing findings remain. |

## Requirements for a future support claim

| Gate | Required evidence |
|---|---|
| Provenance | Pin the current code, model revision, configuration, and overrides. |
| Correctness | Resolve slicing findings and add focused dynamic-versus-physical coverage for every claimed axis. |
| Execution | Run the pinned campaign through its stated boundary on current code. |
| Results | Generate and verify a new report, including accuracy and any enabled downstream stages. |
