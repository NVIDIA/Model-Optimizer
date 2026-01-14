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

"""Utils for speculative decoding."""

import contextlib
import copy
import importlib.util
import os
import sys
import warnings
from collections import Counter, defaultdict, deque

import torch
import torch.distributed
from huggingface_hub import snapshot_download
from torch import nn
from transformers.cache_utils import DynamicCache

KIMI_K2_REPO_ID = "moonshotai/Kimi-K2-Thinking"
KIMI_K2_PACKAGE_NAME = "kimi_k2_temp"


REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)


def calibrate_frequent_vocab(tokenizer, text, target_vocab_size, output_file=None):
    """Given a calibration text, find the most common vocabs and return the mapping."""
    conversations = tokenizer.apply_chat_template(text)
    counter = Counter(conversations)
    vocab = counter.most_common(target_vocab_size)
    mapping = torch.zeros(target_vocab_size, dtype=torch.int64)
    assert len(vocab) == target_vocab_size, (
        f"Not enough vocabs to calibrate ({len(vocab)}/{target_vocab_size}). Please increase data size."
    )
    for i in range(target_vocab_size):
        idx = vocab[i][0]
        mapping[i] = idx - i
    if output_file is not None:
        torch.save(mapping, output_file)
    return mapping


def get_default_attention_mask_and_position_ids(input_ids: torch.Tensor):
    """Compute default attention_mask ans position_ids given input_ids."""
    seq_len = input_ids.shape[-1]

    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
    attention_mask = (
        torch.triu(
            torch.ones(
                (input_ids.shape[0], seq_len, seq_len),
                device=input_ids.device,
            ),
            diagonal=1,
        )
        .bool()
        .view(input_ids.shape[0], 1, seq_len, seq_len)
    )

    return attention_mask, position_ids


class TreeNode:
    """A node in the speculative decoding tree structure.

    Each node represents a token position in the sequence and maintains a dictionary of child nodes,
    """

    def __init__(self, value: int, children: dict | None = None):
        """Initialize a TreeNode.

        Args:
            value (int): the value of the node
            children (dict): a dictionary of children nodes
        """
        self.value = value
        self.children = children if children is not None else {}


class Tree:
    """A tree structure for speculative decoding that defines valid token prediction paths.

    This class implements a tree-based structure used in speculative decoding to represent
    multiple possible token prediction paths. The tree is constructed from a list of paths,
    where each path is a sequence of token positions.

    """

    def __init__(self, tree_paths: list[list[int]]):
        """Initialize a Tree.

        Args:
            tree_paths (list[list[int]]): a list of tree paths
        """
        self.total_nodes = 1
        self.root = TreeNode(0)
        self.num_children = defaultdict(int)
        self.max_depth = 0
        self.create_tree(tree_paths)
        self.create_attention_mask()

    def create_tree(self, tree_paths):
        """Create the tree structure from the list of tree paths.

        This function builds the tree by iterating through each path in the tree_paths list.
        For each path, it traverses the tree, creating nodes and updating the number of children
        at each level.
        """
        tree_paths.sort()
        self.num_children[0] = 1
        for node_path in tree_paths:
            parent_node = self.root
            for i, node in enumerate(node_path):
                # if node is not a child of parent_node, add it
                if node not in parent_node.children:
                    if i != len(node_path) - 1:
                        raise ValueError(
                            f"Incomplete tree path found at {node_path}, {i}th (non-leaf) node doesn't exist"
                        )
                    # value of the node is position id
                    child_node = TreeNode(node)
                    parent_node.children[child_node.value] = child_node
                    # keep track of the number of children of per level
                    self.num_children[i + 1] += 1
                parent_node = parent_node.children[node]

            self.total_nodes += 1
            # update max depth
            self.max_depth = max(self.max_depth, len(node_path))

    def create_attention_mask(self):
        """Create the attention mask for the tree.

        This function constructs the attention mask for the tree based on the tree structure.
        It ensures that each token can only attend to its valid predecessors according to the tree.
        """
        queue = deque([[node, 0] for node in self.root.children.values()])
        self.attention_mask = torch.full(
            (self.total_nodes, self.total_nodes), True, device=torch.cuda.current_device()
        )
        # Base token (in the first column) is attended by all draft tokens
        self.attention_mask[:, 0] = False
        cur_idx = 1
        while queue:
            # iterate over all nodes at current level and update attention mask
            for _ in range(len(queue)):
                node, node_idx = queue.popleft()
                self.attention_mask[cur_idx, : node_idx + 1] = self.attention_mask[
                    node_idx, : node_idx + 1
                ]
                self.attention_mask[cur_idx, cur_idx] = False
                for child in node.children.values():
                    queue.append([child, cur_idx])
                cur_idx += 1


class ResBlock(nn.Module):
    """A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.
    """

    def __init__(self, hidden_size: int, bias: bool = True):
        """Init function of ResBlock.

        Args:
            hidden_size: The size of the hidden layers in the block.
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        # Initialize as an identity mapping
        nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResBlock.

        Args:
            x: Input tensor.

        Returns:
            Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class AcceptanceRateValidation:
    """Base acceptance rate (AR) validation class.

    This class is used to validate the AR within ModelOpt.
    self.validate is the main function to validate the AR given a prompt or input_ids.
    Note: currently it only supports TP.
    """

    def __init__(self, model, tokenizer):
        """Init function to take in the model and tokenizer."""
        tokenizer.chat_template = tokenizer.chat_template.replace(REMOVE_THINK_CHAT_TEMPLATE, "")
        self.model = model
        self.tokenizer = tokenizer
        self.end_token = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

        # Make sure the model is in eval mode
        self.model.eval()

    def tokenize(self, prompt):
        """Apply chat template to the prompt and get input_ids."""
        conversation = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        output = self.tokenizer(
            conversation,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
        ).to(
            torch.cuda.current_device(),
        )
        input_ids = output.input_ids
        return input_ids

    def get_ground_truth(self, input_ids: torch.Tensor, osl: int):
        """This function returns ground truth token ids from the base model.

        This function will be implemented in the plugins.

        Args:
            input_ids: the token ids of the input
            osl: output sequence length
        """

    def check_draft(self, ground_truth, input_ids, draft_tokens):
        """This function checks if the draft tokens should be accepted (same as ground truth).

        Args:
            ground_truth: the ground truth token ids
            input_ids: the input token ids
            draft_tokens: the draft tokens

        Returns:
            input_ids: the updated input token ids
        """
        if draft_tokens is None:
            return input_ids

        if isinstance(draft_tokens, TreeNode):
            # Initialize tracking variables
            token_matched = False  # Flag to track if current token matches ground truth
            # Iterate through each step/level in the tree
            while draft_tokens.children:
                # Check each candidate token at current level
                for child in draft_tokens.children.values():
                    # Check if draft token matches ground truth token
                    if child.value == ground_truth[:, input_ids.shape[1]]:
                        # Accept matching token and update sequence
                        input_id = child.value.unsqueeze(0)
                        input_ids = torch.cat((input_ids, input_id), dim=-1)
                        # Update position for next level traversal
                        draft_tokens = child
                        token_matched = True
                        break
                    else:
                        token_matched = False

                # Stop if either:
                # 1. No match found at current level
                # 2. We've reached the end of ground truth sequence
                if (not token_matched) or (input_ids.shape[1] == ground_truth.shape[1]):
                    break
        else:
            # eager mode
            for i in range(draft_tokens.shape[-1]):
                input_id = draft_tokens[:, i : i + 1]
                if ground_truth[:, input_ids.shape[1] : input_ids.shape[1] + 1] == input_id:
                    input_ids = torch.cat((input_ids, input_id), dim=-1)
                    if input_ids.shape[1] == ground_truth.shape[1]:
                        break
                else:
                    break

        return input_ids

    def check_data_consistency_across_ranks(self, data, group=None, fail_when_mismatch=False):
        """This function checks the data consistency across all ranks in the group.

        Use rank 0 data as the golden set to broadcast to all ranks.
        Each rank compares its data against this golden set and either raises
        (when fail_when_mismatch=True) or emits a warning while forcing every
        rank to adopt rank 0's data.
        """
        if not torch.distributed.is_initialized():
            return data
        if data is None:
            return
        golden_set = copy.deepcopy(data)
        torch.distributed.broadcast(golden_set, src=0, group=group)
        if not torch.equal(data, golden_set):
            if fail_when_mismatch:
                raise ValueError(
                    "Data diverges across ranks. For Megatron, 'moe-token-dispatcher-type'"
                    "should set to 'alltoall'."
                )
            else:
                warnings.warn(
                    "Data diverges across ranks. Forcing all ranks' data equal to rank 0."
                )
        return golden_set

    def validate(
        self,
        osl,
        prompt=None,
        input_ids=None,
        ground_truth=None,
        steps=1,
        tree_paths=None,
    ):
        """This function validate the AR of the model given the input sequence."""
        if input_ids is None:
            input_ids = self.tokenize(prompt)

        isl = input_ids.shape[1]

        if ground_truth is None:
            ground_truth = self.get_ground_truth(input_ids, osl)
        ground_truth = self.check_data_consistency_across_ranks(ground_truth)

        cnt = 0
        draft_tokens = None
        if tree_paths:
            tree = Tree(tree_paths)

        while input_ids.shape[1] < ground_truth.shape[1]:
            cnt += 1
            input_ids = self.check_draft(ground_truth, input_ids, draft_tokens)
            if input_ids.shape[1] == ground_truth.shape[1]:
                break

            if tree_paths:
                input_id, draft_tokens, pred_tokens = self.model.tree_decode(input_ids, tree=tree)
                pred_tokens = self.check_data_consistency_across_ranks(pred_tokens)
            else:
                input_id, draft_tokens = self.model.pseudo_speculative_generate(
                    input_ids, steps=steps
                )
                draft_tokens = self.check_data_consistency_across_ranks(draft_tokens)

            input_id = self.check_data_consistency_across_ranks(input_id)
            input_ids = torch.cat((input_ids, input_id), dim=-1)

        ar = (ground_truth.shape[1] - isl) / cnt

        return ground_truth, ar


@contextlib.contextmanager
def temporary_set_config_value(config, field, value):
    """Context manager to temporarily change config value."""
    if not hasattr(config, field):
        raise AttributeError(f"Config does not have field '{field}'")
    original_value = getattr(config, field)
    try:
        setattr(config, field, value)
        yield
    finally:
        setattr(config, field, original_value)


def _patch_dynamic_cache_compatibility() -> None:
    """Monkey-patch DynamicCache for Kimi-K2 compatibility."""
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = (
            lambda self, seq_len, layer_idx=0: DynamicCache.get_seq_length(self, layer_idx)
        )


def _import_module_from_path(module_path: str, module_name: str, package_name: str):
    """Dynamically import a module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _setup_kimi_k2_decoder():
    """Setup Kimi-K2 decoder with proper imports and patches."""
    # Download repository files
    kimi_k2_path = snapshot_download(repo_id=KIMI_K2_REPO_ID, allow_patterns="*.py")

    # Patch deprecated usage in Kimi implementation
    _patch_dynamic_cache_compatibility()

    # Import required modules
    config_module_path = os.path.join(kimi_k2_path, "configuration_deepseek.py")
    model_module_path = os.path.join(kimi_k2_path, "modeling_deepseek.py")

    _import_module_from_path(
        config_module_path, f"{KIMI_K2_PACKAGE_NAME}.configuration_deepseek", KIMI_K2_PACKAGE_NAME
    )

    kimi_k2_module = _import_module_from_path(
        model_module_path, f"{KIMI_K2_PACKAGE_NAME}.modeling_deepseek", KIMI_K2_PACKAGE_NAME
    )

    # Patch Kimi Attention to init rope lazily to avoid save/load meta tensor error
    original_init_rope = kimi_k2_module.DeepseekV3Attention._init_rope
    original_forward = kimi_k2_module.DeepseekV3Attention.forward

    def patched_fwd_with_lazy_rope_init(self, *args, **kwargs):
        if not hasattr(self, "rotary_emb"):
            original_init_rope(self)
        return original_forward(self, *args, **kwargs)

    kimi_k2_module.DeepseekV3Attention._init_rope = lambda self: None
    kimi_k2_module.DeepseekV3Attention.forward = patched_fwd_with_lazy_rope_init

    return getattr(kimi_k2_module, "DeepseekV3DecoderLayer")
