# Recovery After Failed Run Validation

When excessive truncation fails validation, inspect raw provider responses to
distinguish unfinished reasoning from repetition, parser loss, or context
exhaustion. For genuine output-budget exhaustion, **increase `max_new_tokens`
and rerun** rather than declaring the model infeasible; for example,
`65536 → 131072` when task policy and model/runtime capacity permit. This is a
recovery diagnostic, not a universal default. If context is exhausted, increase
supported deployed context instead; a larger output cap alone cannot help.

Canary the increased budget, then rerun the entire affected benchmark, not only
truncated samples. Use fresh invocation/output/cache namespaces; never mix old-
and new-budget responses.
