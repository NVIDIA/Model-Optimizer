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

import pytest

from modelopt.torch.utils.regex import matches_pattern


def test_none_pattern_matches_everything():
    """Test that None pattern matches any name."""
    assert matches_pattern("model.layer1.weight", None)
    assert matches_pattern("anything", None)
    assert matches_pattern("", None)


def test_single_string_pattern_match():
    """Test matching with a single string pattern."""
    # Exact match
    assert matches_pattern("model.attention.query", "model.attention.query")

    # Wildcard patterns
    assert matches_pattern("model.attention.query", "*.attention.*")
    assert matches_pattern("model.layer1.weight", "*.weight")
    assert matches_pattern("model.layer1.weight", "model.*")
    assert matches_pattern("transformer.encoder.layer.0.attention.self.query", "*attention*")


def test_single_string_pattern_no_match():
    """Test non-matching string patterns."""
    assert not matches_pattern("model.mlp.linear", "*.attention.*")
    assert not matches_pattern("model.layer1.bias", "*.weight")
    assert not matches_pattern("encoder.dense", "decoder.*")


def test_multiple_patterns_as_list():
    """Test matching with multiple patterns in a list."""
    patterns = ["*.attention.*", "*.mlp.*", "*.weight"]

    assert matches_pattern("model.attention.query", patterns)
    assert matches_pattern("model.mlp.linear", patterns)
    assert matches_pattern("model.layer1.weight", patterns)
    assert not matches_pattern("model.layer1.bias", patterns)


def test_multiple_patterns_as_tuple():
    """Test matching with multiple patterns in a tuple."""
    patterns = ("*.attention.*", "*.mlp.*")

    assert matches_pattern("model.attention.query", patterns)
    assert matches_pattern("model.mlp.dense", patterns)
    assert not matches_pattern("model.layer1.bias", patterns)


def test_callable_pattern():
    """Test matching with callable patterns."""
    # Lambda function
    assert matches_pattern("model.layer1.weight", lambda x: "layer" in x)
    assert not matches_pattern("model.attention.query", lambda x: "mlp" in x)

    # Named function
    def has_attention(name: str) -> bool:
        return "attention" in name

    assert matches_pattern("model.attention.query", has_attention)
    assert not matches_pattern("model.mlp.linear", has_attention)


def test_mixed_patterns():
    """Test matching with mixed string and callable patterns."""
    patterns = ["*.weight", lambda x: "attention" in x, "decoder.*"]

    assert matches_pattern("model.layer1.weight", patterns)
    assert matches_pattern("model.attention.query", patterns)
    assert matches_pattern("decoder.output", patterns)
    assert not matches_pattern("encoder.mlp.bias", patterns)


def test_pytorch_module_names():
    """Test with realistic PyTorch module naming patterns."""
    # Transformer-like model names
    assert matches_pattern("transformer.encoder.layer.0.attention.self.query.weight", "*attention*")
    assert matches_pattern("transformer.encoder.layer.0.attention.self.query.weight", "*.weight")
    assert matches_pattern("transformer.decoder.layer.5.mlp.dense.bias", "*decoder*")

    # ResNet-like model names
    assert matches_pattern("resnet.layer1.0.conv1.weight", "*conv*")
    assert matches_pattern("resnet.layer2.3.bn1.running_mean", "*bn*")

    # BERT-like model names
    assert matches_pattern("bert.encoder.layer.11.attention.self.query.weight", "*attention*query*")
    assert matches_pattern("bert.pooler.dense.weight", "bert.pooler.*")


def test_allow_callable_false_with_string_pattern():
    """Test that string patterns work when allow_callable=False."""
    assert matches_pattern("model.attention.query", "*.attention.*", allow_callable=False)
    assert matches_pattern("model.layer1.weight", ["*.weight", "*.bias"], allow_callable=False)


def test_allow_callable_false_raises_on_callable_pattern():
    """Test that callable patterns raise TypeError when allow_callable=False."""
    with pytest.raises(TypeError, match="Callable patterns are not supported"):
        matches_pattern("model.layer1", lambda x: "layer" in x, allow_callable=False)


def test_allow_callable_false_raises_on_callable_in_list():
    """Test that callable in list raises TypeError when allow_callable=False."""
    patterns = ["*.weight", lambda x: "attention" in x]

    with pytest.raises(TypeError, match="Callable patterns are not supported"):
        matches_pattern("model.attention.query", patterns, allow_callable=False)


def test_unsupported_pattern_type_raises():
    """Test that unsupported pattern types raise TypeError."""
    with pytest.raises(TypeError, match="Unsupported pattern type"):
        matches_pattern("model.layer1", 123)


def test_unsupported_pattern_in_list_raises():
    """Test that unsupported pattern types in list raise TypeError."""
    # Use a pattern that won't match first, so it will check the invalid pattern
    patterns = ["*.this_will_not_match.*", 123]

    with pytest.raises(TypeError, match="Unsupported pattern type"):
        matches_pattern("model.layer1.weight", patterns)


def test_empty_pattern_list():
    """Test that empty pattern list matches nothing."""
    assert not matches_pattern("model.layer1.weight", [])
    assert not matches_pattern("anything", [])


def test_complex_wildcard_patterns():
    """Test complex wildcard patterns."""
    # Multiple wildcards
    assert matches_pattern("model.encoder.layer.0.attention.weight", "*.encoder.*.attention.*")

    # Character ranges
    assert matches_pattern("layer1", "layer[0-9]")
    assert not matches_pattern("layerA", "layer[0-9]")

    # Question mark wildcard (single character)
    assert matches_pattern("layer1", "layer?")
    assert not matches_pattern("layer12", "layer?")


def test_edge_cases():
    """Test edge cases."""
    # Empty string name
    assert matches_pattern("", "*")
    assert not matches_pattern("", "model.*")

    # Pattern with no wildcards
    assert matches_pattern("exact_match", "exact_match")
    assert not matches_pattern("no_match", "exact_match")

    # Callable that always returns True
    assert matches_pattern("anything", lambda x: True)

    # Callable that always returns False
    assert not matches_pattern("anything", lambda x: False)
