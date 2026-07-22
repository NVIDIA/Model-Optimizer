# Puzzletron Orchestrator Terminal Controls Design

## Goal

Make the interactive orchestrator easier to understand and safer to stop without
changing scheduler durability or non-interactive behavior.

## Chosen design

Add a small terminal-control component instead of adding raw-terminal logic to the
campaign controller. The component is active only for an interactive TTY and owns:

- nonblocking `q` detection;
- temporarily leaving raw mode while a quit decision is shown; and
- the three quit decisions: cancel jobs and quit, keep jobs running and quit, or
  resume the controller.

Both `q` and Ctrl-C open the decision menu in an interactive terminal. SIGTERM and
non-interactive Ctrl-C keep the existing immediate-cancel behavior. A second Ctrl-C
while the menu is open selects cancel-and-quit so scheduler jobs are not orphaned by
an interrupted prompt.

## Controller behavior

Represent user exits explicitly with `ShutdownAction` values:

- `CANCEL`: run the existing durable cancellation path, mark attempts cancelled,
  and clear live handles;
- `DETACH`: exit without calling the executor cancel method or changing attempt/live
  handle state, allowing the next invocation to recover the jobs; and
- `CONTINUE`: dismiss the menu and resume polling.

The controller will use one helper to apply the chosen action, replacing repeated
shutdown branches. Lease release and signal-handler restoration remain in the common
`finally` path. The result payload distinguishes `cancelled` from `detached`.

## Presentation

The dashboard footer advertises `q / Ctrl-C: quit options`. Use a calmer palette:
blue for active work and borders, green for completion, amber for waiting/warnings,
red for failures, and dim text for pending work. Plain ANSI logs use ordinary colors
instead of the current bright variants. Existing `--color` and `NO_COLOR` behavior is
unchanged.

## Boundaries and verification

This is a surgical cleanup, not a TUI/event-loop rewrite. Scheduler polling,
submission, recovery, state files, and executor APIs stay unchanged. Verification is
limited to focused tests for cancel, detach, resume, and `q` polling plus the existing
shutdown/dashboard tests.
