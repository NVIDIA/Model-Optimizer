// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const MARKER = '<!-- modelopt-readme-testability -->';

/** Publish advisory text only if it still describes the PR's current head. */
async function publishProposal({github, context, proposal, headSha}) {
  if (!proposal || typeof proposal.needs_changes !== 'boolean' ||
      typeof proposal.proposal !== 'string' || proposal.proposal.length > 50000 ||
      (proposal.needs_changes && !proposal.proposal.trim()) || !/^[a-f0-9]{40}$/.test(headSha)) {
    throw new Error('Invalid README testability proposal');
  }
  const {owner, repo} = context.repo;
  const issue_number = context.payload.pull_request.number;
  const {data : pr} = await github.rest.pulls.get({owner, repo, pull_number : issue_number});
  if (pr.state !== 'open' || pr.head.sha !== headSha)
    return;

  const comments =
      await github.paginate(github.rest.issues.listComments, {owner, repo, issue_number});
  const existing = comments.find(
      (comment) => comment.user.login === 'github-actions[bot]' && comment.body.startsWith(MARKER),
  );
  if (!proposal.needs_changes && !existing)
    return;

  const advice =
      proposal.needs_changes
          ? `${proposal.proposal.trim()}\n\n` +
                `To implement, an authorized maintainer can comment:\n\n` +
                `\`@claude implement the README testability proposal for commit ${headSha}\``
          : 'No additional README testability changes are suggested at this commit. The earlier proposal is superseded.';
  const body = `${MARKER}\n### README testability proposal\n\nReviewed commit: \`${headSha}\`\n\n` +
               `${advice}`;
  if (existing) {
    if (existing.body !== body) {
      await github.rest.issues.updateComment({owner, repo, comment_id : existing.id, body});
    }
  } else {
    await github.rest.issues.createComment({owner, repo, issue_number, body});
  }
}

module.exports = {
  MARKER,
  publishProposal
};
