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
# mypy: ignore-errors

"""Megatron tokenizers."""

import base64
import json
from pathlib import Path

from .megatron_lm__megatron_tokenizer import MegatronTokenizer


def reload_mergeable_ranks(
    path: str,
    max_vocab: int | None = None,
) -> dict[bytes, int]:
    """
    Reload our tokenizer JSON file and convert it to Tiktoken format.
    """
    assert path.endswith(".json")

    # reload vocab
    with open(path) as f:
        vocab = json.load(f)
    assert isinstance(vocab, list)
    print(f"Vocab size: {len(vocab)}")
    if max_vocab is not None:
        vocab = vocab[:max_vocab]
        print(f"Cutting vocab to first {len(vocab)} tokens.")

    # build ranks
    ranks: dict[bytes, int] = {}
    for i, x in enumerate(vocab):
        assert x.keys() == {"rank", "token_bytes", "token_str"}
        assert x["rank"] == i
        merge = base64.b64decode(x["token_bytes"])
        assert i >= 256 or merge == bytes([i])
        ranks[merge] = x["rank"]

    # sanity check
    assert len(ranks) == len(vocab)
    assert set(ranks.values()) == set(range(len(ranks)))

    return ranks


PATTERN_TIKTOKEN = (
    r"[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)
PATTERN_TIKTOKEN_V2 = (
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*"
    "|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)


class CustomTikTokenizer(MegatronTokenizer):
    def __init__(
        self,
        path: str,
        pattern: str,
        vocab_size: int,
        num_special_tokens: int,
        special_tokens: list[str] | None,
    ):
        super().__init__(
            path,
            pattern=pattern,
            vocab_size=vocab_size,
            num_special_tokens=num_special_tokens,
            special_tokens=special_tokens,
        )
        import tiktoken

        # if vocab_size is None:
        #     vocab_size = 2**17  # Fallback vocab size is 131072.
        self._vocab_size = vocab_size

        special_tokens_default = ["<unk>", "<s>", "</s>"]
        if special_tokens is None:
            special_tokens = special_tokens_default.copy()
        assert len(special_tokens) == len(set(special_tokens)), (
            f"Special tokens should be unique: {special_tokens}"
        )
        assert len(special_tokens) <= num_special_tokens < self._vocab_size
        assert set(special_tokens_default) <= set(special_tokens), (
            f"Custom special tokens should include {special_tokens_default}"
        )

        special_filler = [
            "<SPECIAL_{id}>".format(id=i) for i in range(len(special_tokens), num_special_tokens)
        ]
        if special_filler:
            print(f"Adding special tokens {special_filler[0]}, ..., {special_filler[-1]}")
        special_tokens = special_tokens + special_filler
        assert len(set(special_tokens)) == len(special_tokens) == num_special_tokens, special_tokens
        inner_vocab_size = self._vocab_size - num_special_tokens

        token_to_id_without_special_tokens = reload_mergeable_ranks(
            path, max_vocab=inner_vocab_size
        )
        # Create space for special tokens.
        token_to_id_without_special_tokens = {
            t: i + num_special_tokens for t, i in token_to_id_without_special_tokens.items()
        }

        special_tokens = {t: i for i, t in enumerate(special_tokens)}
        self._unk_id = special_tokens["<unk>"]
        self._bos_id = special_tokens["<s>"]
        self._eos_id = special_tokens["</s>"]

        # Create tiktoken model.
        self._model = tiktoken.Encoding(
            name=Path(path).parent.name,
            pat_str=pattern,
            mergeable_ranks=token_to_id_without_special_tokens,
            special_tokens=special_tokens,
        )

        # Create final _id_to_token and _token_to_id data structures with special tokens inserted
        # into appropriate locations.
        assert set(token_to_id_without_special_tokens.keys()).isdisjoint(set(special_tokens.keys()))
        self._token_to_id = token_to_id_without_special_tokens.copy()
        self._token_to_id.update(special_tokens)
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}
        assert set(range(self._vocab_size)) == set(self._id_to_token.keys())

    @property
    def bos(self) -> int:
        return self._bos_id

    @property
    def eos(self) -> int:
        return self._eos_id

    @property
    def unk(self) -> int:
        return self._unk_id

    @property
    def eod(self) -> int:
        return self._eos_id

    @property
    def vocab(self):
        return self._token_to_id

    @property
    def inv_vocab(self):
        return self._id_to_token

    def tokenize(self, s: str, bos: bool = False, eos: bool = False) -> list[int]:
        tokens = self._model.encode_ordinary(s)
        if bos:
            tokens = [self.bos, *tokens]
        if eos:
            tokens = [*tokens, self.eos]

        return tokens

    def detokenize(self, tokens: list[int]) -> str:
        return self._model.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def encoder(self):
        return self._token_to_id

    @property
    def decoder(self):
        return self._id_to_token
