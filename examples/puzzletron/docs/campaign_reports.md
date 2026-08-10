# Puzzletron Campaign Reports

This page catalogs retained Puzzletron campaign reports and the status of their
evidence. The compact [campaign report index](../reports/campaign_report_index.yaml)
records each report's producer state, reproduction and support status, metadata
origin, current-configuration relationship, and known limitations. Detailed
run facts remain in the reports.

## Report status

| Model | Report | Producer state | Reproduction | Support | Current configuration relationship |
|---|---|---|---|---|---|
| Nemotron-3 Nano 30B-A3B | [Campaign report](../reports/nemotron3_nano_30b_a3b.html) | `development_snapshot`; revision `unknown` | `not_reproduced` | `not_established` | `migration`: [default.yaml](../configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml) is not the executed configuration |
| Qwen3.5-9B | [Campaign report](../reports/qwen3p5_9b.html) | `development_snapshot`; revision `unknown` | `not_reproduced` | `not_established` | `reconstruction`: [default.yaml](../configs/families/qwen3_5/qwen3p5_9b/runs/default.yaml) is not the executed configuration; the report records additional overrides and width values |

## Evidence boundary

| Record | Status |
|---|---|
| Retained reports | Preserve the detailed configuration, stage, result, and warning data from their producing development state. |
| Campaign report index | Records only curated status, metadata origin, current-configuration relationship, and known limitations. |
| Current configuration references | Provide migration or reconstruction starting points, not frozen executed configurations. |
| Reproduction status | No reproduction is recorded for the listed reports. |
| Support status | Not established while reproduction and unresolved correctness findings remain open. |

## Future campaign records

The current entries were curated from retained reports. Future campaigns can
use the same catalog fields, but their evidence artifacts should be generated
by the campaign pipeline from the structured data used to render the HTML
report rather than assembled after the run.

| Consideration | Future direction |
|---|---|
| Generation | Emit a versioned evidence artifact alongside the HTML report and report manifest. |
| Provenance | Record exact code, model, data, resolved configuration, and override identities. |
| Results | Reference canonical stage outcomes, metrics, warnings, and artifacts without copying them manually. |
| Validation | Make schema and semantic verification part of report publication. |
| Support | Keep model-support promotion as a separate reviewed decision after current-code reproduction and correctness gates pass. |
