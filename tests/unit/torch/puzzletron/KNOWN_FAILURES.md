# Quarantined Puzzletron test failures

The `puzzletron_v2` Nox session loads `quarantined_tests.json` through the local
pytest configuration. Every registered node still executes, but is marked as a
strict expected failure. An unexpected pass therefore fails CI and tells us to
remove the stale entry. The Nox session also rejects missing or duplicate node
IDs, unknown schema data, and entries without an observed mismatch, removal
condition, and side-effect safety note.

These entries are unresolved, pre-existing contract discrepancies found by
comparing this branch with its base. They are not expected behavior or
intentional skips, and inclusion here does not decide whether the test or the
implementation is correct.

| Registry group | Current nodes |
| --- | ---: |
| Campaign and granularity reports | 16 |
| MIP profiles and resume identity | 11 |
| DAG and orchestration | 8 |
| Scenario, evaluation, and resume | 7 |
| Setup wizard and candidate display | 6 |
| Runtime-stat configuration | 2 |

The JSON registry records group-level reproduced evidence because the baseline
run did not preserve a trustworthy traceback for every individual node. It does
not infer more specific causes from test names. Source inspection confirms that
the registered tests use in-memory state, temporary files, injected fakes, or
command construction only; they do not allocate a GPU, submit scheduler work,
or call an external service.

Remove a node in the same change that reconciles its contract, then run that
focused test with `--runxfail` and the complete `puzzletron_v2` session. Adding a
node requires a reproduced pre-existing failure, an honest observed mismatch,
a removal condition, and a fresh side-effect audit.
