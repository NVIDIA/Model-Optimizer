---
name: release-cherry-pick
description: Cherry-pick merged PRs labeled for a release branch into that branch, then open a PR and apply the cherry-pick-done label. Use when asked to "cherry-pick PRs for release/X.Y.Z", "pick PRs to release branch", or "cherry-pick labeled PRs".
---

# Cherry-pick PRs to a Release Branch

Cherry-pick all merged `main` PRs labeled `cherry-pick-<version>` (but not `cherry-pick-done`) into the corresponding `release/<version>` branch, one by one in merge order.

## Step 1 — Identify the target version

Ask the user for the release version (e.g. `0.44.0`) if not already provided.

Set `VERSION=<version>` for use in subsequent steps.

## Step 2 — Fetch pending PRs

Use the GitHub search API to list PRs that have the cherry-pick label but not cherry-pick-done, sorted by merge date ascending:

```bash
gh api "search/issues?q=repo:NVIDIA/Model-Optimizer+is:pr+is:merged+base:main+label:cherry-pick-<VERSION>+-label:cherry-pick-done&sort=updated&order=asc&per_page=50" \
  --jq '.items[] | [.number, .title, .pull_request.merged_at] | @tsv' \
  | sort -t$'\t' -k3
```

Present the list to the user before proceeding.

## Step 3 — Set up the release branch

Check out `release/<VERSION>`, creating it from the remote if it doesn't exist locally:

```bash
git fetch origin release/<VERSION>
git checkout release/<VERSION>
```

## Step 4 — Get merge commit SHAs

All PRs are squash-merged, so each has a single-parent commit. Retrieve the SHA for each PR:

```bash
gh pr view <NUM> --repo NVIDIA/Model-Optimizer --json mergeCommit --jq '.mergeCommit.oid'
```

## Step 5 — Cherry-pick in merge order

Cherry-pick each commit with `-s` (DCO sign-off). GPG signing is handled automatically by the repo's git config.

```bash
git cherry-pick -s <SHA>
```

**On conflict:** Tell the user which PR caused the conflict and ask them to fix it, then continue:

```bash
git cherry-pick --continue
```

## Step 6 — Create a PR to the release branch

Push the cherry-picks to a new branch and open a PR targeting `release/<VERSION>`. The PR title lists every cherry-picked PR number. The body uses `## Cherry-picked PRs` as the only heading with one `- #<NUM>` bullet per PR — no titles, no links, no extra text.

```bash
git checkout -B cherry-picks/release-<VERSION>
git push -u origin cherry-picks/release-<VERSION>

gh pr create \
  --title "[Cherry-pick] PRs #<NUM1> #<NUM2> ..." \
  --base release/<VERSION> \
  --head cherry-picks/release-<VERSION> \
  --body "$(cat <<'EOF'
## Cherry-picked PRs

- #<NUM1>
- #<NUM2>
...
EOF
)"
```

## Step 7 — Apply cherry-pick-done label

Add the `cherry-pick-done` label to every PR that was successfully cherry-picked:

```bash
for pr in <NUM1> <NUM2> ...; do
  gh pr edit $pr --repo NVIDIA/Model-Optimizer --add-label "cherry-pick-done"
done
```
