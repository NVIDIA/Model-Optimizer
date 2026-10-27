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

const assert = require('node:assert/strict');
const test = require('node:test');
const {MARKER, publishProposal} = require('./readme_testability.cjs');

const SHA = 'a'.repeat(40);
const PROPOSAL = {
  needs_changes : true,
  proposal : 'Add ID `train`, a smoke test, and a tiny-model fixture.'
};

function setup({sha = SHA, state = 'open', comments = []} = {}) {
  const writes = [];
  return {
    writes,
    context : {
      repo : {owner : 'NVIDIA', repo : 'Model-Optimizer'},
      payload : {pull_request : {number : 1}}
    },
    headSha : SHA,
    proposal : PROPOSAL,
    github : {
      rest : {
        pulls : {get : async () => ({data : {head : {sha}, state}})},
        issues : {
          listComments : () => {},
          createComment : async (args) => writes.push({method : 'create', ...args}),
          updateComment : async (args) => writes.push({method : 'update', ...args}),
        },
      },
      paginate : async () => comments,
    },
  };
}

test('asks for explicit implementation and posts only an advisory comment', async () => {
  const run = setup();
  await publishProposal(run);
  assert.equal(run.writes.length, 1);
  assert.equal(run.writes[0].method, 'create');
  assert.equal(run.writes[0].issue_number, 1);
  assert.match(run.writes[0].body, /tiny-model fixture/);
  assert.match(run.writes[0].body, /To implement, an authorized maintainer can comment/);
  assert.match(run.writes[0].body, new RegExp(`@claude implement.*${SHA}`));
});

test('skips proposals for stale commits and closed PRs', async () => {
  for (const options of [{sha : 'b'.repeat(40)}, {state : 'closed'}]) {
    const run = setup(options);
    await publishProposal(run);
    assert.deepEqual(run.writes, []);
  }
});

test('updates a single bot-owned comment and avoids duplicate writes on retries', async () => {
  const run = setup();
  await publishProposal(run);
  const body = run.writes[0].body;
  const comment = {id : 42, body, user : {login : 'github-actions[bot]'}};
  const retry = setup({comments : [ comment ]});
  await publishProposal(retry);
  assert.deepEqual(retry.writes, []);
  retry.proposal = {needs_changes : true, proposal : 'Updated fixture plan.'};
  await publishProposal(retry);
  assert.equal(retry.writes[0].method, 'update');
  assert.equal(retry.writes[0].comment_id, 42);
});

test('leaves human comments intact even if they contain the marker', async () => {
  const run = setup({comments : [ {id : 1, body : MARKER, user : {login : 'maintainer'}} ]});
  await publishProposal(run);
  assert.equal(run.writes[0].method, 'create');
});

test('silences no-change reports and supersedes an obsolete proposal', async () => {
  const run = setup();
  run.proposal = {needs_changes : false, proposal : ''};
  await publishProposal(run);
  assert.deepEqual(run.writes, []);
  const previous =
      setup({comments : [ {id : 3, body : MARKER, user : {login : 'github-actions[bot]'}} ]});
  previous.proposal = run.proposal;
  await publishProposal(previous);
  assert.match(previous.writes[0].body, /earlier proposal is superseded/);
  assert.doesNotMatch(previous.writes[0].body, /@claude implement/);
});

test('rejects malformed model output before any API write', async () => {
  for (const proposal of [null, {needs_changes : true, proposal : ''},
                          {needs_changes : 'yes', proposal : 'bad'}]) {
    const run = setup();
    run.proposal = proposal;
    await assert.rejects(publishProposal(run), /Invalid/);
    assert.deepEqual(run.writes, []);
  }
});
