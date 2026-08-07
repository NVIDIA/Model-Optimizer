# Historical Puzzletron Campaign Results

This page catalogs campaign results retained from earlier experimental work.
The entries summarize report metadata, relationships to current configuration
files, and open reproduction or correctness gaps. These results have not been
reproduced on current code and are not model-support claims. The
[historical campaign manifest](../reports/historical_campaigns.yaml) is the
machine-readable source for the fields below.

## Historical result summary

| Model | Archived result | Current configuration relationship | Reproduction and correctness status |
|---|---|---|---|
| Nemotron-3 Nano 30B-A3B | [Campaign report](../reports/nemotron3_nano_30b_a3b.html): sequence length 8,192; parameter, runtime, and memory MIP profiles; stages marked complete through zero-shot evaluation plus global-distillation sanity | [default.yaml](../configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml) is a current-code migration, not the executed configuration | Not reproduced on current code; the report contains 176 slicing-equivalence findings and 2 descriptor-realization-gate findings; AIPerf, full global distillation, and post-distillation evaluation remain pending |
| Qwen3.5-9B | [Campaign report](../reports/qwen3p5_9b.html): sequence length 16,384; runtime MIP profile; stages marked complete through post-distillation evaluation | [default.yaml](../configs/families/qwen3_5/qwen3p5_9b/runs/default.yaml) is a reconstruction; the report used 23 overrides and embedding widths 4,096, 3,840, and 3,584, while the current entry retains only 4,096 | Not reproduced on current code; the report contains 101 slicing-equivalence findings |

## Evidence boundary

| Record | Status |
|---|---|
| Archived report metadata | Retained and checked statically against the manifest. |
| Current-configuration relationship | Recorded as a migration or reconstruction, not an executed configuration match. |
| Current-code reproduction | Not available for the listed results. |
| Model-support status | Not established; unresolved slicing findings remain. |

## Reproduction requirements

| Gate | Required evidence |
|---|---|
| Provenance | Pin the current code, model revision, configuration, and overrides. |
| Correctness | Resolve slicing findings and add focused dynamic-versus-physical coverage for every claimed axis. |
| Execution | Run the pinned campaign through its stated boundary on current code. |
| Results | Generate and verify a new report, including accuracy and any enabled downstream stages. |
