# Launcher version — the validated `nemo-evaluator-launcher` pin

Shared reference for the **0.2.x path** (Steps 1–9, including GDPVal). nel-next
(`nemo-evaluator` 0.4.x) is a different package and pins separately — see
`references/nel-next.md`.

The validated version lives in one place, `scripts/nel-validated-version.sh`, and
is consumed by `scripts/nel-check.sh` (Step 1 gate) and `scripts/nel-gdpval.sh`
(GDPVal's hard pin). Bump that one file, not the call sites.

## Why the version is pinned

**Comparability.** This skill exists to produce baseline-vs-candidate deltas.
Running the two legs on different launchers folds a harness change into the
measured model delta — the same reasoning `nel-next.sh` already applies to its
0.4.x git SHA. Keep a comparison pair on one launcher, and record the version
with the scores.

**A known failure, on GDPVal specifically.** Launchers older than the pin emit
`export NEL_INVOCATION_ID="${NEL_INVOCATION_ID}"` into the generated `run.sub`
without assigning it first, so the job exits with `NEL_INVOCATION_ID: unbound
variable` under `set -u` before the evaluation client starts. This fires only for
configs that forward `runtime:NEL_INVOCATION_ID`, which today is GDPVal alone —
so it is a GDPVal correctness requirement, while comparability applies to every
scored task. Details and the dry-run check: `references/gym-gdpval.md`.

## Presence is not the same as version

Step 1 used to accept any `nel` already on PATH. That is how a GDPVal run picked
up a stale 0.2.4 from the base environment and failed. Note that
`nemo-evaluator-launcher-internal` ships its **own** launcher version and can
supply an older `nel` even when it is itself newer — `nel --version` prints both
rows, and only the `nemo_evaluator_launcher` row governs generated-Slurm and
schema behavior:

```text
nemo_evaluator_launcher: 0.2.4              <- this row gates
nemo_evaluator_launcher_internal: 0.3.174+20260609
```

Run the gate rather than eyeballing it:

```bash
"$SKILL_DIR/scripts/nel-check.sh"     # exits non-zero with the pip command to fix it
```

`NEL_ALLOW_UNVALIDATED=1` downgrades a mismatch to a warning and marks the output
`(UNVALIDATED)` — dev/canary only, never for scored or compared runs. GDPVal has
no escape hatch: it runs through `nel-gdpval.sh`, which hard-pins the launcher.

## Updating the pin

When a newer `nemo-evaluator-launcher` release is available:

1. Review its release notes for launcher schema, generated Slurm, resume, and
   export changes.
2. Update `NEL_VALIDATED_VERSION` in `scripts/nel-validated-version.sh`, and the
   expected version in `tests/test_nel_check.py` / `tests/test_nel_gdpval.py`.
3. Run the focused tests and pre-commit checks, and confirm `nel-check.sh
   --version` and `nel-gdpval.sh --version` both report the candidate.
4. Re-validate on a real run before adopting it for scored work — for GDPVal,
   follow the extra dry-run and canary steps in `references/gym-gdpval.md`
   (`limit_samples` is inert there, so there is no cheap reduced-sample canary).
5. Do not mix launcher versions within a baseline-versus-candidate comparison. If
   the baseline was scored on the old pin, either keep the candidate there too or
   re-run the baseline on the new one.
