# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sparse teacher-policy dataset for external draft training.

An alternative to caching base hidden states: the TVD objectives need only the
teacher's deployment policy (top-k then top-p at the serving temperature), which
is far smaller than a hidden state per token and is captured during generation
rather than by a second forward pass.

Records are gzipped JSONL, one conversation per line::

    {
        "id": str,
        "prompt_ids": [int],  # context, not scored
        "gen_ids": [int],  # response tokens, scored
        "topk_tok": [[int]],  # per response position, teacher support
        "topk_lp": [[float]],
    }  # matching log-probabilities

They are decoded once into a memmapped cache beside the shards, stored as int32
ids and float16 probabilities; holding them as Python objects does not fit in
memory for a corpus of any size.
"""

import gzip
import json
import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from modelopt.torch.utils.logging import warn_rank_0

__all__ = ["SparsePolicyCollator", "SparsePolicyDataset", "discover_shards"]

CACHE_VERSION = 2


def _cache_signature(files: list[str]) -> list[list]:
    """Identify the source shards so a stale cache is rebuilt rather than reused."""
    return [[str(f), os.path.getsize(f), int(os.path.getmtime(f))] for f in files]


def _cache_matches(meta: dict, files: list[str], top_k: int | None) -> bool:
    """Whether a published cache serves this request; an explicit ``top_k`` must match too."""
    return (
        meta.get("version") == CACHE_VERSION
        and meta.get("sources") == _cache_signature(files)
        and (top_k is None or meta.get("top_k") == top_k)
    )


def discover_shards(root) -> list[str]:
    """Every sparse-policy shard under ``root``, in the order the cache is keyed on.

    Shared with the training entrypoint, and named in the pre-build instructions, so a
    cache built by hand is keyed identically to the one training would build. A command
    that globbed differently -- one pattern, or non-recursively -- would produce a
    different ``_cache_signature``, and training would rebuild from scratch under
    torchrun having apparently pre-built nothing.

    Both suffixes are combined rather than preferred: an ``or`` would silently drop every
    ``.jsonl`` shard from a directory that also holds a ``.jsonl.gz``.
    """
    root = Path(root)
    return sorted(
        str(p)
        for pattern in ("*.jsonl.gz", "*.jsonl")
        for p in root.rglob(pattern)
        if not p.name.startswith(".")
    )


def _builder_progress(cache: Path) -> tuple[int, int]:
    """Bytes written and newest mtime across rank 0's in-progress cache directories.

    The builder writes into a sibling ``<cache>.tmp.*`` directory and renames it into
    place at the end, so its growth is the only signal that it is still alive.
    """
    total = newest = 0
    for tmp in cache.parent.glob(f"{cache.name}.tmp.*"):
        for f in tmp.glob("*"):
            try:
                st = f.stat()
            except OSError:
                continue
            total += st.st_size
            newest = max(newest, int(st.st_mtime))
    return total, newest


def _wait_for_cache(
    cache: Path, files: list[str], top_k: int | None = None, stall_s: float = 1800.0
) -> None:
    """Block until rank 0 publishes a cache built from ``files``, without a collective.

    Waiting on mere existence is not enough. The only way to reach this with a cache
    already on disk is that it was found stale, so ``meta.json`` is there from the
    previous dump; returning on it would leave every rank but the builder memmapped to
    the old shards -- and a mapping survives the rename, so they would train a whole run
    on the wrong corpus against a perfectly ordinary loss curve.

    ``stall_s`` bounds time *without progress*, not the whole build. A first build over a
    large corpus runs for hours, and a total deadline would fail every multi-GPU run that
    has not pre-built its cache -- the one case this function exists to serve. Rank 0's
    temporary directory grows while it works, so a wait that watches it can be patient
    with a slow build and still fail promptly on a dead builder.
    """
    last_progress = _builder_progress(cache)
    deadline = time.monotonic() + stall_s
    while True:
        try:
            meta = json.loads((cache / "meta.json").read_text())
            if _cache_matches(meta, files, top_k):
                return
        except (OSError, ValueError):
            # Absent, or caught mid-publish; either way it is not ours yet.
            pass
        progress = _builder_progress(cache)
        if progress != last_progress:
            last_progress = progress
            deadline = time.monotonic() + stall_s
        elif time.monotonic() > deadline:
            raise TimeoutError(
                f"the sparse cache at {cache} has not grown for {stall_s:.0f}s and no cache "
                "matching these shards was published. Rank 0 is not building: pre-build it "
                "in a single process before torchrun, using the same discovery training "
                "does, or the cache is keyed differently and rebuilt anyway:\n"
                '  python -c "from modelopt.torch.speculative.external.sparse_data import '
                "SparsePolicyDataset as D, discover_shards as F; "
                'D(F(DUMP), max_length=SEQ_LEN, cache_dir=CACHE_DIR)"'
            )
        time.sleep(2.0)


def _build_cache(cache: Path, files: list[str], top_k: int | None) -> None:
    """Decode the JSONL shards into flat memmaps. One sequential pass."""
    # A private scratch directory per builder. Training several drafts against one
    # dumped dataset is the normal workflow, so two jobs can reach this at the same
    # time; a shared scratch name had each one delete the files the other still held
    # open, leaving it writing to unlinked inodes and publishing an index whose
    # offsets no longer matched its data -- a cache that looks complete and is not.
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=cache.name + ".tmp.", dir=cache.parent))
    try:
        index: list[tuple[int, int, int, int]] = []
        srcs: list[str] = []
        n_ids = n_pos = dropped = 0
        mass_sum = mass_n = 0.0
        width = top_k
        damaged: list[tuple[str, str]] = []
        with (
            open(tmp / "ids.i32", "wb") as f_ids,
            open(tmp / "tok.i32", "wb") as f_tok,
            open(tmp / "prob.f16", "wb") as f_prob,
        ):
            for path in files:
                opener = gzip.open if str(path).endswith(".gz") else open
                # A shard truncated by a killed writer must not destroy the whole pass:
                # this loop is an hour of work on a large corpus, records are independent,
                # and a retry would hit the same bad byte. Keep what parsed, report the rest.
                try:
                    fh = opener(path, "rt")
                except OSError as exc:
                    damaged.append((str(path), repr(exc)))
                    continue
                with fh:
                    while True:
                        try:
                            line = fh.readline()
                        except (OSError, EOFError) as exc:
                            damaged.append((str(path), repr(exc)))
                            break
                        if not line:
                            break
                        try:
                            r = json.loads(line)
                            prompt, gen = r["prompt_ids"], r["gen_ids"]
                            # Both inside the guard: a record with an empty or absent
                            # policy must be dropped like any other damaged one, not
                            # abort a build the rest of this loop works to survive.
                            n_gen = min(len(gen), len(r["topk_tok"]), len(r["topk_lp"]))
                            if width is None:
                                # Widest row of the record, not row 0. With top_p < 1
                                # the stored nucleus varies per position, and row 0 is
                                # the first generated token -- conditioned on the whole
                                # prompt, so typically the lowest-entropy and narrowest
                                # row there is. Inferring from it clips every wider row
                                # in the corpus.
                                width = max(len(x) for x in r["topk_tok"])
                        except (ValueError, KeyError, IndexError) as exc:
                            dropped += 1
                            if dropped == 1:
                                damaged.append((str(path), f"unparseable record: {exc!r}"))
                            continue
                        if n_gen == 0:
                            # Nothing supervised: an index entry for it would contribute an
                            # empty span to every epoch.
                            dropped += 1
                            if dropped == 1:
                                damaged.append((str(path), "record has no scored positions"))
                            continue

                        ids = np.asarray(prompt + gen[:n_gen], dtype=np.int32)
                        # Ragged rows would silently misalign the flat memmap, so pad or
                        # truncate every row to the one width the cache is indexed by.
                        # Dumps are normally rectangular, so try the whole block in one
                        # numpy call first: the per-position loop below costs 500M+
                        # iterations on a 100k-conversation corpus.
                        raw_t, raw_p = r["topk_tok"][:n_gen], r["topk_lp"][:n_gen]
                        try:
                            tok = np.array(raw_t, dtype=np.int32)
                            # exp(-inf) = 0 for tokens the teacher's nucleus excluded.
                            prob = np.exp(np.array(raw_p, dtype=np.float32)).astype(np.float16)
                            if tok.ndim != 2 or tok.shape[1] != width:
                                raise ValueError("width mismatch")
                        except (ValueError, TypeError):
                            widest = max((len(x) for x in raw_t), default=0)
                            if widest > width:
                                # Truncating here would drop real teacher mass and bake
                                # the narrow width into meta["top_k"], and the stored-mass
                                # warning cannot see it. Refuse with the remedy instead.
                                # The recipe has no top_k field, so the remedy is the
                                # single-process pre-build, which takes one.
                                raise ValueError(
                                    f"{path}: a stored policy row holds {widest} entries "
                                    f"but the cache was sized at {width} from the first "
                                    f"record. Build the cache once with an explicit "
                                    f"width, then point the run at it: "
                                    f"SparsePolicyDataset(discover_shards(dump), "
                                    f"max_length=..., top_k=<dump's top-k, at least "
                                    f"{widest}>, cache_dir=<data.sparse_cache_dir>)."
                                )
                            tok = np.zeros((n_gen, width), dtype=np.int32)
                            prob = np.zeros((n_gen, width), dtype=np.float16)
                            for j in range(n_gen):
                                t, lp = raw_t[j][:width], raw_p[j][:width]
                                tok[j, : len(t)] = t
                                prob[j, : len(lp)] = np.exp(np.asarray(lp, dtype=np.float32))

                        # tvd treats the stored policy as the teacher's whole
                        # distribution, so it must not be a truncation of a wider
                        # nucleus. Sampled with the same top_k the dump requested,
                        # this sums to 1; built with a smaller k it does not, and the
                        # objective then charges the draft for mass on tokens the
                        # teacher actually supported.
                        if n_gen:
                            # Whole record, not prob[0]: the first response token is
                            # conditioned on the prompt alone and is typically the
                            # lowest-entropy position in the generation, so its top-k mass
                            # is near 1 even when a too-narrow top_k is badly clipping the
                            # nucleus at later, higher-entropy positions -- the condition
                            # this check exists to catch.
                            mass_sum += float(prob.sum(axis=-1).mean())
                            mass_n += 1

                        f_ids.write(ids.tobytes())
                        f_tok.write(tok.tobytes())
                        f_prob.write(prob.tobytes())
                        index.append((n_ids, len(prompt), n_gen, n_pos))
                        srcs.append(r.get("source") or "")
                        n_ids += ids.shape[0]
                        n_pos += n_gen

        if not index:
            raise ValueError(f"No sparse-policy records found in {len(files)} file(s)")

        if damaged:
            for path, why in damaged[:10]:
                warnings.warn(f"sparse shard {path}: {why}")
            # Tolerate the odd truncated shard, but a widespread failure means the dump
            # is wrong and training on it would quietly use a fraction of the corpus.
            # `damaged` holds at most one entry per *build* for record-level damage
            # (the dropped == 1 guards above), so it cannot express "most records
            # failed to parse". A dump whose schema drifted -- a renamed topk_lp, say
            # -- raises KeyError on every record, warns once, passes the shard check
            # below, and publishes a cache holding a handful of rows. Gate on the
            # record drop rate too, which is what the shard threshold is reasoning
            # about anyway.
            kept = len(index)
            if dropped > max(1, (kept + dropped) // 20):
                raise ValueError(
                    f"{dropped} of {kept + dropped} sparse-policy records failed to parse "
                    f"(kept {kept}); the dump's schema does not match what this reader "
                    "expects."
                )
            if len(damaged) > max(1, len(files) // 20):
                raise ValueError(
                    f"{len(damaged)} of {len(files)} sparse shards are unreadable; "
                    "the dump looks broken rather than merely truncated."
                )
            print(
                f"[sparse] skipped {len(damaged)} damaged shard(s) and {dropped} record(s); "
                f"kept {len(index)}",
                flush=True,
            )

        if mass_n and mass_sum / mass_n < 0.999:
            warnings.warn(
                f"stored teacher policy sums to {mass_sum / mass_n:.4f}, not ~1: the dump's "
                "top-k is narrower than the nucleus it was sampled from. The TVD objectives "
                "renormalise over the stored support, so the draft is charged for mass on "
                "tokens the teacher supported but the dump dropped; even a draft matching the "
                "teacher scores a non-zero loss. Re-dump with a larger top-k.",
                stacklevel=2,
            )

        np.save(tmp / "index.npy", np.asarray(index, dtype=np.int64))
        # Per-record source, kept as codes into source_names so the dataset can weight
        # sources without re-reading the shards.
        source_names = sorted(set(srcs))
        codes = {name: i for i, name in enumerate(source_names)}
        np.save(tmp / "sources.npy", np.asarray([codes[x] for x in srcs], dtype=np.int16))
        (tmp / "meta.json").write_text(
            json.dumps(
                {
                    "version": CACHE_VERSION,
                    "top_k": width,
                    "n_records": len(index),
                    "n_ids": n_ids,
                    "n_pos": n_pos,
                    "sources": _cache_signature(files),
                    "source_names": source_names,
                }
            )
        )
        # Publish by rename: a half-written cache that looks complete is worse than
        # none. os.rename onto an existing non-empty directory raises ENOTEMPTY, so
        # move any stale cache aside first and only delete it once the new one is in
        # place -- a rebuild must never leave the shards with no usable cache.
        # The hand-off name is per-builder too, for the same reason as the scratch one.
        stale = cache.with_name(f"{cache.name}.stale.{tmp.name.rsplit('.', 1)[-1]}")
        shutil.rmtree(stale, ignore_errors=True)
        try:
            if cache.exists():
                cache.rename(stale)
            tmp.rename(cache)
        except OSError:
            # Another builder published between the check above and this rename, so
            # the name is taken by a cache built from the same shards. Drop ours
            # rather than fight for it -- but never leave the shards with no cache.
            if not cache.exists() and stale.exists():
                stale.rename(cache)
            else:
                shutil.rmtree(tmp, ignore_errors=True)
        finally:
            shutil.rmtree(stale, ignore_errors=True)
    except BaseException:
        # Leave no half-decoded scratch behind; the name is unique per builder,
        # so this can only ever remove our own.
        shutil.rmtree(tmp, ignore_errors=True)
        raise


class SparsePolicyDataset(torch.utils.data.Dataset):
    """Conversations with the teacher's sparse deployment policy per response token."""

    def __init__(
        self,
        files: list[str],
        max_length: int = 4096,
        top_k: int | None = None,
        cache_dir: str | None = None,
        limit: int | None = None,
        source_weights: dict[str, float] | None = None,
    ):
        """Open (building if needed) the memmapped cache for these shards.

        ``top_k`` defaults to the width found in the data. A wider value pads the
        stored policy with zero-probability entries, which changes the objective
        silently; a narrower one would drop teacher mass and is refused by the
        builder instead.

        ``source_weights`` maps a record's ``source`` field to a sampling weight,
        oversampling under-represented sources by repeating their index entries.
        Weight by *tokens*, not rows: sources differ by an order of magnitude in
        response length, so balanced row counts can still leave one source holding
        most of the trained positions.
        """
        super().__init__()
        if not files:
            raise ValueError("No sparse-policy shards given")
        self.max_length = max_length
        cache = Path(cache_dir) if cache_dir else Path(files[0]).parent / ".sparse_cache"

        stale = True
        if (cache / "meta.json").exists():
            # An explicit top_k is part of the request: it truncates or pads the policy.
            stale = not _cache_matches(json.loads((cache / "meta.json").read_text()), files, top_k)

        if stale:
            # Under DDP every rank sees the same shards; one builds and the rest wait on
            # the files rather than on a barrier. A collective here would be bounded by
            # ddp_timeout (1800s by default, and HfTrainingArguments has already created
            # the process group with it by the time a dataset is built), which a first
            # build over a large corpus outlasts. _wait_for_cache needs no process group
            # and bounds time without progress instead, so it serves both cases.
            # Fall back to the launcher's RANK when there is no group: without it every
            # worker takes the rank-0 branch and they race to rewrite the same cache.
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = int(os.environ.get("RANK", 0))
            if rank == 0:
                _build_cache(cache, files, top_k)
            else:
                _wait_for_cache(cache, files, top_k)

        meta = json.loads((cache / "meta.json").read_text())
        # Re-check rather than trust the build. Two jobs reading different subsets of one
        # dumped directory resolve to the same default cache path, and the one that loses
        # the publish rename returns from _build_cache normally having adopted the
        # winner's cache -- which it never compared against its own shards. The waiting
        # ranks are covered by _wait_for_cache; this covers the builder.
        if not _cache_matches(meta, files, top_k):
            raise RuntimeError(
                f"the sparse cache at {cache} was published by a concurrent job from "
                "different shards; give this run its own cache_dir."
            )
        self.top_k = meta["top_k"]
        self.index = np.load(cache / "index.npy")
        self.source_names = meta.get("source_names", [])
        self.sources = None
        if (cache / "sources.npy").exists():
            self.sources = np.load(cache / "sources.npy")[: self.index.shape[0]]
        # __getitem__ right-truncates to max_length, so a record whose context alone
        # fills it keeps no scored position: an all-zero loss_mask that still counts
        # towards the record total and the loss average. Report it rather than let the
        # run look larger than it is. Counted before any oversampling, so the number
        # describes the corpus the advice below asks the user to re-dump, not the
        # repeated index entries. Rank 0 only: every rank builds this object.
        n_dead = int((self.index[:, 1] >= self.max_length).sum())
        if n_dead:
            warn_rank_0(
                f"{n_dead} of {self.index.shape[0]} sparse-policy records have a context of "
                f"{self.max_length} tokens or more and so contribute no scored positions at "
                f"max_length={self.max_length}. Raise training_seq_len, or re-dump with a "
                f"smaller --max-seq-len."
            )
        if source_weights:
            self.index = self._apply_source_weights(source_weights)
        if limit is not None and limit < self.index.shape[0]:
            # A seeded sample, not index[:limit]. index.npy is written in shard-then-line
            # order, so a prefix is a prefix of the shard list: with one source per shard
            # -- the natural layout -- it can contain a single source, and weighting that
            # gives a mixture nothing like the one asked for. Taken after weighting so the
            # sample is drawn from the mixture the user configured.
            take = np.sort(
                np.random.default_rng(0).choice(self.index.shape[0], limit, replace=False)
            )
            self.index = self.index[take]
            if self.sources is not None:
                self.sources = self.sources[take]
        self._ids = np.memmap(cache / "ids.i32", dtype=np.int32, mode="r", shape=(meta["n_ids"],))
        self._tok = np.memmap(
            cache / "tok.i32", dtype=np.int32, mode="r", shape=(meta["n_pos"], self.top_k)
        )
        self._prob = np.memmap(
            cache / "prob.f16", dtype=np.float16, mode="r", shape=(meta["n_pos"], self.top_k)
        )

    def _apply_source_weights(self, source_weights: dict[str, float]) -> np.ndarray:
        """Repeat index entries so each source's share matches its weight."""
        if self.sources is None:
            raise ValueError(
                "source_weights needs a cache built with CACHE_VERSION >= 2; "
                "delete the .sparse_cache directory to rebuild it"
            )
        unknown = set(source_weights) - set(self.source_names)
        if unknown:
            raise ValueError(
                f"source_weights names {sorted(unknown)} which are not in the data "
                f"{self.source_names}"
            )
        w = np.ones(self.index.shape[0], dtype=np.float64)
        for name, weight in source_weights.items():
            if weight < 0:
                raise ValueError(f"source weight for {name!r} is negative")
            w[self.sources == self.source_names.index(name)] = weight
        # Split the weight into guaranteed repeats plus a seeded draw for the
        # remainder, so a weight of 2.5 gives every record 2 copies and half of
        # them a third -- deterministic across ranks and restarts.
        counts = np.floor(w).astype(np.int64)
        frac = w - counts
        counts += (np.random.default_rng(0).random(w.shape[0]) < frac).astype(np.int64)
        # sources indexes index row-for-row and is read back by callers (and by the
        # sampling below), so it has to be repeated identically or the two desynchronise.
        self.sources = np.repeat(self.sources, counts, axis=0)
        return np.repeat(self.index, counts, axis=0)

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, i):
        ids_start, n_prompt, n_gen, pos_start = (int(x) for x in self.index[i])
        n = min(n_prompt + n_gen, self.max_length)
        input_ids = torch.from_numpy(
            np.asarray(self._ids[ids_start : ids_start + n], dtype=np.int64)
        )

        # Only response positions carry a teacher policy, so only they are scored.
        loss_mask = torch.zeros(n, dtype=torch.float)
        loss_mask[n_prompt:n] = 1.0

        tok = torch.zeros(n, self.top_k, dtype=torch.long)
        prob = torch.zeros(n, self.top_k, dtype=torch.float)
        keep = max(0, n - n_prompt)
        if keep:
            sl = slice(pos_start, pos_start + keep)
            tok[n_prompt:n] = torch.from_numpy(np.asarray(self._tok[sl], dtype=np.int64))
            prob[n_prompt:n] = torch.from_numpy(np.asarray(self._prob[sl], dtype=np.float32))

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "loss_mask": loss_mask,
            "teacher_topk_tok": tok,
            "teacher_topk_prob": prob,
        }


class SparsePolicyCollator:
    """Pad or truncate a batch of sparse-policy records to ``train_len``."""

    def __init__(self, train_len: int):
        """Set the fixed sequence length every batch is padded or truncated to."""
        self.train_len = train_len

    def _fit(self, x):
        n = x.shape[0]
        if n >= self.train_len:
            return x[: self.train_len]
        pad = torch.zeros((self.train_len - n, *x.shape[1:]), dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    def __call__(self, features):
        """Collate records into a padded batch."""
        return {
            k: torch.stack([self._fit(f[k]) for f in features])
            for k in (
                "input_ids",
                "attention_mask",
                "loss_mask",
                "teacher_topk_tok",
                "teacher_topk_prob",
            )
        }
