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
import warnings
from pathlib import Path

import numpy as np
import torch

__all__ = ["SparsePolicyCollator", "SparsePolicyDataset"]

CACHE_VERSION = 2


def _cache_signature(files: list[str]) -> list[list]:
    """Identify the source shards so a stale cache is rebuilt rather than reused."""
    return [[str(f), os.path.getsize(f), int(os.path.getmtime(f))] for f in files]


def _build_cache(cache: Path, files: list[str], top_k: int | None) -> None:
    """Decode the JSONL shards into flat memmaps. One sequential pass."""
    tmp = cache.with_name(cache.name + ".tmp")
    if tmp.exists():
        for p in tmp.iterdir():
            p.unlink()
    tmp.mkdir(parents=True, exist_ok=True)

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
                            width = len(r["topk_tok"][0])
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
                        tok = np.zeros((n_gen, width), dtype=np.int32)
                        prob = np.zeros((n_gen, width), dtype=np.float16)
                        for j in range(n_gen):
                            t, lp = raw_t[j][:width], raw_p[j][:width]
                            tok[j, : len(t)] = t
                            prob[j, : len(lp)] = np.exp(np.asarray(lp, dtype=np.float32))

                    # tvd treats the stored policy as the teacher's whole
                    # distribution, so it must not be a truncation of a wider
                    # nucleus. Sampled with the same top_k the dump requested,
                    # this sums to 1; built with a smaller k than the nucleus it
                    # does not, and the objective silently under-penalises the
                    # draft's out-of-support mass.
                    if n_gen:
                        mass_sum += float(prob[0].sum())
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
            "top-k is narrower than the nucleus it was sampled from. external_loss='tvd' "
            "treats the missing mass as belonging to the draft and will under-penalise "
            "it; re-dump with a larger top-k, or use 'tvd'.",
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
    stale = cache.with_name(cache.name + ".stale")
    shutil.rmtree(stale, ignore_errors=True)
    if cache.exists():
        cache.rename(stale)
    tmp.rename(cache)
    shutil.rmtree(stale, ignore_errors=True)


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

        ``top_k`` defaults to the width found in the data. A narrower value
        truncates the teacher policy and a wider one pads it with zero-probability
        entries; both change the objective silently.

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
            meta = json.loads((cache / "meta.json").read_text())
            stale = meta.get("version") != CACHE_VERSION or meta.get("sources") != (
                _cache_signature(files)
            )

        if stale:
            # Under DDP every rank sees the same shards; one builds and the rest wait.
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                _build_cache(cache, files, top_k)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        meta = json.loads((cache / "meta.json").read_text())
        self.top_k = meta["top_k"]
        self.index = np.load(cache / "index.npy")
        if limit is not None:
            self.index = self.index[:limit]
        self.source_names = meta.get("source_names", [])
        self.sources = None
        if (cache / "sources.npy").exists():
            self.sources = np.load(cache / "sources.npy")[: self.index.shape[0]]
        if source_weights:
            self.index = self._apply_source_weights(source_weights)
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
