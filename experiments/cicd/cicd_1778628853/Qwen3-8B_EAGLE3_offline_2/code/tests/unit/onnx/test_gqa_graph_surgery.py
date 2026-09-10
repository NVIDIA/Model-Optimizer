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

"""Tests for GQA graph surgery (replace_attention_with_gqa)."""

import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

pytest.importorskip("transformers", reason="transformers required for GQA graph surgery tests")

from modelopt.onnx.graph_surgery.gqa_replacement import replace_attention_with_gqa

MODEL_ID = "Qwen/Qwen2.5-0.5B"
VOCAB_SIZE = 64
SEQ_LEN = 4
MAX_SEQ_LEN = 128

_RNG = np.random.RandomState(42)


def _fp16(*shape):
    return (_RNG.randn(*shape) * 0.02).astype(np.float16)


def _init(name, arr):
    return numpy_helper.from_array(arr, name=name)


def _const_node(name, value, dtype=np.int64):
    arr = np.array(value, dtype=dtype)
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{name}_output_0"],
        name=name,
        value=numpy_helper.from_array(arr, name=""),
    )


def _build_toy_model(hidden_size, num_heads, kv_heads, head_dim, inv_freq_np, num_layers=1):
    """Build a toy model matching real Optimum LLaMA export patterns.

    Includes: shared rotary_emb (inv_freq x position_ids -> Cos/Sin),
    per-layer rotate_half RoPE, KV cache concat, causal+padding mask.
    """
    nodes, inits, vis = [], [], []
    half_dim = head_dim // 2

    inits.append(_init("one_f16", np.array(1.0, dtype=np.float16)))
    inits.append(_init("neg_large_f16", np.array(-1e4, dtype=np.float16)))
    inits.append(_init("axes_0", np.array([0], dtype=np.int64)))
    inits.append(_init("axes_01", np.array([0, 1], dtype=np.int64)))
    inits.append(_init("axes_12", np.array([1, 2], dtype=np.int64)))
    inits.append(_init("trilu_k1", np.array(1, dtype=np.int64)))
    inits.append(_init("onnx::inv_freq", inv_freq_np.reshape(1, half_dim, 1).astype(np.float32)))

    graph_inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["B", "S"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["B", "T"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["B", "S"]),
    ]
    graph_outputs = []

    graph_inputs.extend(
        helper.make_tensor_value_info(
            f"past_key_values.{lid}.{kv}",
            TensorProto.FLOAT16,
            ["B", kv_heads, "P", head_dim],
        )
        for lid in range(num_layers)
        for kv in ("key", "value")
    )

    inits.append(_init("model.embed_tokens.weight", _fp16(VOCAB_SIZE, hidden_size)))
    nodes.append(
        helper.make_node(
            "Gather",
            ["model.embed_tokens.weight", "input_ids"],
            ["/model/embed_tokens/Gather_output_0"],
            name="/model/embed_tokens/Gather",
            axis=0,
        )
    )
    hidden = "/model/embed_tokens/Gather_output_0"

    # -- shared rotary_emb: inv_freq x position_ids -> cos/sin (fp16) --
    re = "/model/rotary_emb"
    nodes.append(_const_node(f"{re}/Constant_6", [1], np.int64))
    nodes.append(
        helper.make_node(
            "Unsqueeze",
            ["position_ids", f"{re}/Constant_6_output_0"],
            [f"{re}/Unsqueeze_1_output_0"],
            name=f"{re}/Unsqueeze_1",
        )
    )
    nodes.append(
        helper.make_node(
            "Cast",
            [f"{re}/Unsqueeze_1_output_0"],
            [f"{re}/Cast_1_output_0"],
            name=f"{re}/Cast_1",
            to=1,
        )
    )
    nodes.append(
        helper.make_node(
            "Cast", ["onnx::inv_freq"], [f"{re}/Cast_2_output_0"], name=f"{re}/Cast_2", to=1
        )
    )
    nodes.append(
        helper.make_node(
            "MatMul",
            [f"{re}/Cast_2_output_0", f"{re}/Cast_1_output_0"],
            [f"{re}/MatMul_output_0"],
            name=f"{re}/MatMul",
        )
    )
    nodes.append(
        helper.make_node(
            "Transpose",
            [f"{re}/MatMul_output_0"],
            [f"{re}/Transpose_output_0"],
            name=f"{re}/Transpose",
            perm=[0, 2, 1],
        )
    )
    nodes.append(
        helper.make_node(
            "Concat",
            [f"{re}/Transpose_output_0", f"{re}/Transpose_output_0"],
            [f"{re}/Concat_1_output_0"],
            name=f"{re}/Concat_1",
            axis=-1,
        )
    )
    nodes.append(
        helper.make_node(
            "Cos", [f"{re}/Concat_1_output_0"], [f"{re}/Cos_output_0"], name=f"{re}/Cos"
        )
    )
    nodes.append(
        helper.make_node(
            "Sin", [f"{re}/Concat_1_output_0"], [f"{re}/Sin_output_0"], name=f"{re}/Sin"
        )
    )
    nodes.append(_const_node(f"{re}/Constant_7", 1.0, np.float32))
    nodes.append(
        helper.make_node(
            "Mul",
            [f"{re}/Cos_output_0", f"{re}/Constant_7_output_0"],
            [f"{re}/Mul_1_output_0"],
            name=f"{re}/Mul_1",
        )
    )
    nodes.append(_const_node(f"{re}/Constant_8", 1.0, np.float32))
    nodes.append(
        helper.make_node(
            "Mul",
            [f"{re}/Sin_output_0", f"{re}/Constant_8_output_0"],
            [f"{re}/Mul_2_output_0"],
            name=f"{re}/Mul_2",
        )
    )
    nodes.append(
        helper.make_node(
            "Cast", [f"{re}/Mul_1_output_0"], [f"{re}/Cast_4_output_0"], name=f"{re}/Cast_4", to=10
        )
    )
    nodes.append(
        helper.make_node(
            "Cast", [f"{re}/Mul_2_output_0"], [f"{re}/Cast_5_output_0"], name=f"{re}/Cast_5", to=10
        )
    )
    cos_out = f"{re}/Cast_4_output_0"
    sin_out = f"{re}/Cast_5_output_0"

    # -- shared causal + padding mask --
    nodes.append(helper.make_node("Shape", ["input_ids"], ["ids_shape"], name="/model/pos/Shape"))
    nodes.append(_const_node("/model/pos/C1", 1, np.int64))
    nodes.append(
        helper.make_node(
            "Gather",
            ["ids_shape", "/model/pos/C1_output_0"],
            ["seq_len"],
            name="/model/pos/seq_gather",
            axis=0,
        )
    )
    nodes.append(
        helper.make_node("Unsqueeze", ["seq_len", "axes_0"], ["seq_1d"], name="/model/causal/unsq")
    )
    nodes.append(
        helper.make_node(
            "Concat", ["seq_1d", "seq_1d"], ["causal_shape"], name="/model/causal/cat", axis=0
        )
    )
    nodes.append(
        helper.make_node(
            "ConstantOfShape",
            ["causal_shape"],
            ["causal_ones"],
            name="/model/causal/fill",
            value=numpy_helper.from_array(np.array([1.0], dtype=np.float16), name=""),
        )
    )
    nodes.append(
        helper.make_node(
            "Trilu", ["causal_ones", "trilu_k1"], ["upper_tri"], name="/model/causal/trilu", upper=1
        )
    )
    nodes.append(
        helper.make_node(
            "Mul", ["upper_tri", "neg_large_f16"], ["causal_4d_raw"], name="/model/causal/mul"
        )
    )
    nodes.append(
        helper.make_node(
            "Unsqueeze", ["causal_4d_raw", "axes_01"], ["causal_4d"], name="/model/causal/unsq4d"
        )
    )
    nodes.append(
        helper.make_node("Cast", ["attention_mask"], ["pad_f16"], name="/model/pad/cast", to=10)
    )
    nodes.append(
        helper.make_node("Unsqueeze", ["pad_f16", "axes_12"], ["pad_4d"], name="/model/pad/unsq")
    )
    nodes.append(helper.make_node("Sub", ["one_f16", "pad_4d"], ["inv_pad"], name="/model/pad/inv"))
    nodes.append(
        helper.make_node("Mul", ["inv_pad", "neg_large_f16"], ["pad_bias"], name="/model/pad/mul")
    )
    nodes.append(
        helper.make_node("Add", ["causal_4d", "pad_bias"], ["attn_bias"], name="/model/bias/add")
    )

    # -- per layer --
    for lid in range(num_layers):
        pre = f"/model/layers.{lid}"
        ap = f"{pre}/self_attn"
        q_dim = num_heads * head_dim
        k_dim = kv_heads * head_dim

        ln_w = f"model.layers.{lid}.input_layernorm.weight"
        ln_b = f"model.layers.{lid}.input_layernorm.bias"
        inits.append(_init(ln_w, np.ones(hidden_size, dtype=np.float16)))
        inits.append(_init(ln_b, np.zeros(hidden_size, dtype=np.float16)))
        ln_out = f"{pre}/input_layernorm/Mul_1_output_0"
        nodes.append(
            helper.make_node(
                "LayerNormalization",
                [hidden, ln_w, ln_b],
                [ln_out],
                name=f"{pre}/input_layernorm/LayerNorm",
                axis=-1,
                epsilon=1e-5,
            )
        )

        qw = f"model.layers.{lid}.self_attn.q_proj.weight"
        kw = f"model.layers.{lid}.self_attn.k_proj.weight"
        vw = f"model.layers.{lid}.self_attn.v_proj.weight"
        ow = f"model.layers.{lid}.self_attn.o_proj.weight"
        inits += [
            _init(qw, _fp16(hidden_size, q_dim)),
            _init(kw, _fp16(hidden_size, k_dim)),
            _init(vw, _fp16(hidden_size, k_dim)),
            _init(ow, _fp16(q_dim, hidden_size)),
        ]

        for proj, dim, suf in [
            ("q_proj", q_dim, ""),
            ("k_proj", k_dim, "_1"),
            ("v_proj", k_dim, "_2"),
        ]:
            w = f"model.layers.{lid}.self_attn.{proj}.weight"
            nodes.append(
                helper.make_node(
                    "MatMul",
                    [ln_out, w],
                    [f"{ap}/{proj}/MatMul_output_0"],
                    name=f"{ap}/{proj}/MatMul",
                )
            )

        inits.append(_init(f"{ap}/q_shape", np.array([0, 0, num_heads, head_dim], np.int64)))
        inits.append(_init(f"{ap}/kv_shape", np.array([0, 0, kv_heads, head_dim], np.int64)))
        for tag, proj, shape_name in [
            ("", "q_proj", "q_shape"),
            ("_1", "k_proj", "kv_shape"),
            ("_2", "v_proj", "kv_shape"),
        ]:
            nodes.append(
                helper.make_node(
                    "Reshape",
                    [f"{ap}/{proj}/MatMul_output_0", f"{ap}/{shape_name}"],
                    [f"{ap}/Reshape{tag}_output_0"],
                    name=f"{ap}/Reshape{tag}",
                )
            )
            nodes.append(
                helper.make_node(
                    "Transpose",
                    [f"{ap}/Reshape{tag}_output_0"],
                    [f"{ap}/Transpose{tag}_output_0"],
                    name=f"{ap}/Transpose{tag}",
                    perm=[0, 2, 1, 3],
                )
            )

        qt = f"{ap}/Transpose_output_0"
        kt = f"{ap}/Transpose_1_output_0"
        vt = f"{ap}/Transpose_2_output_0"

        # RoPE helper: builds rotate_half pattern for a tensor
        def _rope(tensor, prefix, cos=cos_out, sin=sin_out):
            p = f"{ap}/{prefix}"
            nodes.append(_const_node(f"{p}/c_ax", [1], np.int64))
            nodes.append(
                helper.make_node(
                    "Unsqueeze", [cos, f"{p}/c_ax_output_0"], [f"{p}/cos4d"], name=f"{p}/cos_unsq"
                )
            )
            nodes.append(_const_node(f"{p}/s_ax", [1], np.int64))
            nodes.append(
                helper.make_node(
                    "Unsqueeze", [sin, f"{p}/s_ax_output_0"], [f"{p}/sin4d"], name=f"{p}/sin_unsq"
                )
            )
            nodes.append(
                helper.make_node("Mul", [tensor, f"{p}/cos4d"], [f"{p}/x_cos"], name=f"{p}/mul_cos")
            )
            nodes.append(helper.make_node("Shape", [tensor], [f"{p}/sh"], name=f"{p}/shape"))
            nodes.append(_const_node(f"{p}/dim_idx", 3, np.int64))
            nodes.append(
                helper.make_node(
                    "Gather",
                    [f"{p}/sh", f"{p}/dim_idx_output_0"],
                    [f"{p}/D"],
                    name=f"{p}/gather_D",
                    axis=0,
                )
            )
            nodes.append(_const_node(f"{p}/two", 2, np.int64))
            nodes.append(
                helper.make_node(
                    "Div", [f"{p}/D", f"{p}/two_output_0"], [f"{p}/half"], name=f"{p}/div"
                )
            )
            nodes.append(
                helper.make_node(
                    "Unsqueeze", [f"{p}/half", "axes_0"], [f"{p}/half_1d"], name=f"{p}/unsq_half"
                )
            )
            nodes.append(_const_node(f"{p}/zero", [0], np.int64))
            nodes.append(_const_node(f"{p}/ax", [-1], np.int64))
            nodes.append(_const_node(f"{p}/step", [1], np.int64))
            nodes.append(
                helper.make_node(
                    "Slice",
                    [
                        tensor,
                        f"{p}/zero_output_0",
                        f"{p}/half_1d",
                        f"{p}/ax_output_0",
                        f"{p}/step_output_0",
                    ],
                    [f"{p}/x1"],
                    name=f"{p}/slice1",
                )
            )
            nodes.append(_const_node(f"{p}/big", [9223372036854775807], np.int64))
            nodes.append(_const_node(f"{p}/ax2", [-1], np.int64))
            nodes.append(_const_node(f"{p}/step2", [1], np.int64))
            nodes.append(
                helper.make_node(
                    "Slice",
                    [
                        tensor,
                        f"{p}/half_1d",
                        f"{p}/big_output_0",
                        f"{p}/ax2_output_0",
                        f"{p}/step2_output_0",
                    ],
                    [f"{p}/x2"],
                    name=f"{p}/slice2",
                )
            )
            nodes.append(helper.make_node("Neg", [f"{p}/x2"], [f"{p}/neg_x2"], name=f"{p}/neg"))
            nodes.append(
                helper.make_node(
                    "Concat", [f"{p}/neg_x2", f"{p}/x1"], [f"{p}/rot"], name=f"{p}/concat", axis=-1
                )
            )
            nodes.append(
                helper.make_node(
                    "Mul", [f"{p}/rot", f"{p}/sin4d"], [f"{p}/rot_sin"], name=f"{p}/mul_sin"
                )
            )
            out = f"{p}/out"
            nodes.append(
                helper.make_node("Add", [f"{p}/x_cos", f"{p}/rot_sin"], [out], name=f"{p}/add")
            )
            return out

        q_rope = _rope(qt, "rope_q")
        k_rope = _rope(kt, "rope_k")

        past_k = f"past_key_values.{lid}.key"
        past_v = f"past_key_values.{lid}.value"
        pres_k = f"present.{lid}.key"
        pres_v = f"present.{lid}.value"
        nodes.append(
            helper.make_node("Concat", [past_k, k_rope], [pres_k], name=f"{ap}/Concat_5", axis=2)
        )
        nodes.append(
            helper.make_node("Concat", [past_v, vt], [pres_v], name=f"{ap}/Concat_6", axis=2)
        )
        graph_outputs.append(
            helper.make_tensor_value_info(
                pres_k, TensorProto.FLOAT16, ["B", kv_heads, "T", head_dim]
            )
        )
        graph_outputs.append(
            helper.make_tensor_value_info(
                pres_v, TensorProto.FLOAT16, ["B", kv_heads, "T", head_dim]
            )
        )

        if kv_heads != num_heads:
            reps = num_heads // kv_heads
            inits += [
                _init(f"{ap}/rk/exp", np.array([1, reps, 1, 1], np.int64)),
                _init(f"{ap}/rk/ax", np.array([2], np.int64)),
                _init(f"{ap}/rk/rs", np.array([0, num_heads, -1, head_dim], np.int64)),
            ]
            for t, src in [("k", pres_k), ("v", pres_v)]:
                nodes.append(
                    helper.make_node(
                        "Unsqueeze",
                        [src, f"{ap}/rk/ax"],
                        [f"{ap}/rk/{t}u"],
                        name=f"{ap}/repeat_kv/{t}_unsqueeze",
                    )
                )
                nodes.append(
                    helper.make_node(
                        "Expand",
                        [f"{ap}/rk/{t}u", f"{ap}/rk/exp"],
                        [f"{ap}/rk/{t}e"],
                        name=f"{ap}/repeat_kv/{t}_expand",
                    )
                )
                nodes.append(
                    helper.make_node(
                        "Reshape",
                        [f"{ap}/rk/{t}e", f"{ap}/rk/rs"],
                        [f"{ap}/rk/{t}r"],
                        name=f"{ap}/repeat_kv/{t}_reshape",
                    )
                )
            k_final, v_final = f"{ap}/rk/kr", f"{ap}/rk/vr"
        else:
            k_final, v_final = pres_k, pres_v

        nodes.append(
            helper.make_node(
                "Transpose",
                [k_final],
                [f"{ap}/Transpose_3_output_0"],
                name=f"{ap}/Transpose_3",
                perm=[0, 1, 3, 2],
            )
        )
        scale_val = float(np.array(1.0 / (head_dim**0.5), dtype=np.float16))
        nodes.append(_const_node(f"{ap}/scale", scale_val, np.float16))
        nodes.append(
            helper.make_node(
                "Mul",
                [q_rope, f"{ap}/scale_output_0"],
                [f"{ap}/Mul_8_output_0"],
                name=f"{ap}/Mul_8",
            )
        )
        nodes.append(
            helper.make_node(
                "MatMul",
                [f"{ap}/Mul_8_output_0", f"{ap}/Transpose_3_output_0"],
                [f"{ap}/MatMul_output_0"],
                name=f"{ap}/MatMul",
            )
        )
        nodes.append(
            helper.make_node(
                "Add",
                [f"{ap}/MatMul_output_0", "attn_bias"],
                [f"{ap}/Add_2_output_0"],
                name=f"{ap}/Add_2",
            )
        )
        nodes.append(
            helper.make_node(
                "Softmax",
                [f"{ap}/Add_2_output_0"],
                [f"{ap}/Softmax_output_0"],
                name=f"{ap}/Softmax",
                axis=-1,
            )
        )
        nodes.append(
            helper.make_node(
                "MatMul",
                [f"{ap}/Softmax_output_0", v_final],
                [f"{ap}/MatMul_1_output_0"],
                name=f"{ap}/MatMul_1",
            )
        )
        nodes.append(
            helper.make_node(
                "Transpose",
                [f"{ap}/MatMul_1_output_0"],
                [f"{ap}/Transpose_4_output_0"],
                name=f"{ap}/Transpose_4",
                perm=[0, 2, 1, 3],
            )
        )
        inits.append(_init(f"{ap}/out_rs", np.array([0, 0, hidden_size], np.int64)))
        nodes.append(
            helper.make_node(
                "Reshape",
                [f"{ap}/Transpose_4_output_0", f"{ap}/out_rs"],
                [f"{ap}/Reshape_7_output_0"],
                name=f"{ap}/Reshape_7",
            )
        )
        nodes.append(
            helper.make_node(
                "MatMul",
                [f"{ap}/Reshape_7_output_0", ow],
                [f"{ap}/o_proj/MatMul_output_0"],
                name=f"{ap}/o_proj/MatMul",
            )
        )
        res = f"{pre}/residual_add/output_0"
        nodes.append(
            helper.make_node(
                "Add", [hidden, f"{ap}/o_proj/MatMul_output_0"], [res], name=f"{pre}/residual_add"
            )
        )
        hidden = res

    inits.append(_init("lm_head.weight", _fp16(hidden_size, VOCAB_SIZE)))
    nodes.append(
        helper.make_node("MatMul", [hidden, "lm_head.weight"], ["logits"], name="/lm_head/MatMul")
    )
    graph_outputs.insert(
        0, helper.make_tensor_value_info("logits", TensorProto.FLOAT16, ["B", "S", VOCAB_SIZE])
    )

    graph = helper.make_graph(
        nodes, "test_gqa", graph_inputs, graph_outputs, initializer=inits, value_info=vis
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _get_config():
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=False)
    hidden = cfg.hidden_size
    heads = cfg.num_attention_heads
    kv = getattr(cfg, "num_key_value_heads", heads)
    hd = hidden // heads
    theta = getattr(cfg, "rope_theta", 10000.0)
    inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd))
    return hidden, heads, kv, hd, inv_freq.numpy()


def _run_session(model_proto, feeds):
    """Run inference directly from an in-memory ModelProto."""
    model_bytes = model_proto.SerializeToString()
    sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


@pytest.fixture(scope="module")
def models_and_config():
    """Build original model, run GQA surgery, return both protos + config."""
    hidden, heads, kv, hd, inv_freq_np = _get_config()
    orig = _build_toy_model(hidden, heads, kv, hd, inv_freq_np)
    onnx.checker.check_model(orig)

    with tempfile.TemporaryDirectory() as td:
        orig_path = os.path.join(td, "original.onnx")
        gqa_path = os.path.join(td, "gqa.onnx")
        onnx.save(orig, orig_path)

        replace_attention_with_gqa(
            model_path=orig_path,
            output_path=gqa_path,
            hf_model_id=MODEL_ID,
            max_seq_len=MAX_SEQ_LEN,
            io_dtype="float16",
            use_external_data=False,
        )
        gqa = onnx.load(gqa_path)

    return orig, gqa, {"heads": heads, "kv": kv, "hd": hd}


class TestGQAGraphSurgery:
    def test_gqa_node_exists(self, models_and_config):
        _, gqa, _ = models_and_config
        gqa_ops = [n for n in gqa.graph.node if n.op_type == "GroupQueryAttention"]
        assert len(gqa_ops) == 1

    def test_gqa_attributes(self, models_and_config):
        _, gqa, cfg = models_and_config
        gqa_node = next(n for n in gqa.graph.node if n.op_type == "GroupQueryAttention")
        attrs = {a.name: (a.i if a.type == 2 else a.f) for a in gqa_node.attribute}
        assert attrs["num_heads"] == cfg["heads"]
        assert attrs["kv_num_heads"] == cfg["kv"]
        assert attrs["do_rotary"] == 1

    def test_node_count_reduced(self, models_and_config):
        orig, gqa, _ = models_and_config
        assert len(gqa.graph.node) < len(orig.graph.node)

    def test_rotary_emb_nodes_removed(self, models_and_config):
        _, gqa, _ = models_and_config
        rotary_names = [n.name for n in gqa.graph.node if "rotary_emb" in n.name]
        assert len(rotary_names) == 0

    def test_position_ids_removed(self, models_and_config):
        _, gqa, _ = models_and_config
        input_names = [i.name for i in gqa.graph.input]
        assert "position_ids" not in input_names

    def test_logits_match(self, models_and_config):
        orig, gqa, cfg = models_and_config
        kv, hd = cfg["kv"], cfg["hd"]

        ids = np.arange(1, SEQ_LEN + 1, dtype=np.int64).reshape(1, SEQ_LEN)
        mask = np.ones((1, SEQ_LEN), dtype=np.int64)
        pos = np.arange(SEQ_LEN, dtype=np.int64).reshape(1, SEQ_LEN)
        empty_kv = np.zeros((1, kv, 0, hd), dtype=np.float16)

        orig_feeds = {
            "input_ids": ids,
            "attention_mask": mask,
            "position_ids": pos,
            "past_key_values.0.key": empty_kv,
            "past_key_values.0.value": empty_kv,
        }
        gqa_feeds = {
            "input_ids": ids,
            "attention_mask": mask,
            "past_key_values.0.key": empty_kv,
            "past_key_values.0.value": empty_kv,
        }

        orig_logits = _run_session(orig, orig_feeds)[0].astype(np.float32)
        gqa_logits = _run_session(gqa, gqa_feeds)[0].astype(np.float32)

        diff = np.abs(orig_logits - gqa_logits)
        finite = diff[np.isfinite(diff)]

        print(f"\n  Original nodes: {len(orig.graph.node)}  ->  GQA nodes: {len(gqa.graph.node)}")
        print(f"  Logits shape:   {orig_logits.shape}")
        print(f"  Original[0,:4]: {orig_logits[0, 0, :4]}")
        print(f"  GQA     [0,:4]: {gqa_logits[0, 0, :4]}")
        if len(finite) > 0:
            print(f"  Max  abs diff:  {finite.max():.6f}")
            print(f"  Mean abs diff:  {finite.mean():.6f}")

        assert len(finite) > 0, "All values are non-finite"
        assert finite.max() < 1.0, f"Max abs diff {finite.max():.4f} exceeds tolerance"
