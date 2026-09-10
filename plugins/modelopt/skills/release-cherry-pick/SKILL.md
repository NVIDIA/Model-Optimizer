---
name: release-cherry-pick
description: Audit merged bug-fix PRs and release NVBugs for missing cherry-pick labels, then cherry-pick labeled PRs into a release branch and open a PR. Use when asked to "cherry-pick PRs for release/X.Y.Z", "pick PRs to release branch", "verify cherry-pick labels", or "cherry-pick labeled PRs".
---

# Cherry-pick PRs to a Release Branch

Cherry-pick all merged `main` PRs labeled `cherry-pick-<version>` (but not `cherry-pick-done`) into the corresponding `release/<version>` branch, one by one in merge order.

## Step 1 — Identify the target version

Ask the user for the release version (e.g. `0.44.0`) if not already provided.

Set `VERSION=<version>` for use in subsequent steps.

## Step 2 — Audit candidates for missing labels

Before fetching the labeled queue, audit release NVBugs and recent merged PRs so bug fixes are not omitted.

### Audit release NVBugs

Use NVBug IDs supplied by the user or found in their release source (for example, a test-plan document or release message). If none were supplied, ask the user for the NVBug list; do not assume the GitHub queue is complete.

For each NVBug:

1. Open `https://nvbugspro.nvidia.com/bug/<NVBUG>`.
2. Verify it has the exact release label `Committed_ModelOpt_<VERSION>`.
3. Inspect its comments for `github.com/NVIDIA/Model-Optimizer/pull/<PR>` links. Record every linked PR, not only the latest comment.
4. Verify each linked PR is merged into `main` and is a bug fix. A release-labeled NVBug is evidence for review, not by itself proof that every linked PR should be picked.

If an NVBug lacks the expected NVBug label, report it but do not edit NVBug. If browser access is unavailable, report that the NVBug portion of the audit could not be completed and ask the user for the relevant PR links or exported comments.

### Audit recent merged PRs

Treat PRs merged into `main` since the release branch diverged as "recent":

```bash
git fetch origin main release/<VERSION>
BASE=$(git merge-base origin/main origin/release/<VERSION>)
SINCE=$(git show -s --format=%cs "$BASE")

gh pr list \
  --repo NVIDIA/Model-Optimizer \
  --state merged \
  --base main \
  --limit 1000 \
  --json number,title,author,mergedAt,labels,url \
  | jq --arg since "${SINCE}T00:00:00Z" \
      '[.[] | select(.mergedAt >= $since)]'
```

Review each PR's title, body, labels, changed files, and linked issue context. Classify it as:

- **Yes** — repairs incorrect behavior, a regression, crash, compatibility problem, or documentation defect relevant to the release.
- **No** — feature, refactor, cleanup, dependency refresh, or other change not needed to correct the release.
- **Unclear** — insufficient evidence or meaningful backport risk; ask the user.

Do not classify a PR as a bug fix from the word `fix` alone. Deduplicate PRs found through both audits.

### Report and label

Present the complete audit before changing GitHub labels:

| PR | Title | Author | NVBug(s) | Bug fix? | `cherry-pick-<VERSION>` present? | Recommendation |
|---|---|---|---|---|---|---|

Use `—` when no NVBug is known. Also list NVBugs with no linked PR or a missing `Committed_ModelOpt_<VERSION>` label. Ask the user to confirm which recommended PRs should receive the missing label. After confirmation, apply it:

```bash
for pr in <APPROVED_NUMBERS>; do
  gh pr edit "$pr" --repo NVIDIA/Model-Optimizer --add-label "cherry-pick-<VERSION>"
done
```

Do not label unmerged PRs, PRs not based on `main`, or candidates classified **Unclear** without explicit approval. Re-run the audit table after edits so it reflects the final label state.

## Step 3 — Fetch pending PRs

Use the GitHub search API to list PRs that have the cherry-pick label but not cherry-pick-done, sorted by merge date ascending:

```bash
gh api "search/issues?q=repo:NVIDIA/Model-Optimizer+is:pr+is:merged+base:main+label:cherry-pick-<VERSION>+-label:cherry-pick-done&sort=updated&order=asc&per_page=50" \
  --jq '.items[] | [.number, .title, .pull_request.merged_at] | @tsv' \
  | sort -t$'\t' -k3
```

Present the list to the user before proceeding.

## Step 4 — Set up the release branch

Check out `release/<VERSION>`, creating it from the remote if it doesn't exist locally:

```bash
git fetch origin release/<VERSION>
git checkout release/<VERSION>
```

## Step 5 — Get merge commit SHAs

All PRs are squash-merged, so each has a single-parent commit. Retrieve the SHA for each PR:

```bash
gh pr view <NUM> --repo NVIDIA/Model-Optimizer --json mergeCommit --jq '.mergeCommit.oid'
```

## Step 6 — Cherry-pick in merge order

Cherry-pick each commit with `-s` (DCO sign-off). GPG signing is handled automatically by the repo's git config.

```bash
git cherry-pick -s <SHA>
```

**On conflict:** Tell the user which PR caused the conflict and ask them to fix it, then continue:

```bash
git cherry-pick --continue
```

## Step 7 — Create a PR to the release branch

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

## Step 8 — Apply cherry-pick-done label

Add the `cherry-pick-done` label to every PR that was successfully cherry-picked:

```bash
for pr in <NUM1> <NUM2> ...; do
  gh pr edit $pr --repo NVIDIA/Model-Optimizer --add-label "cherry-pick-done"
done
```
