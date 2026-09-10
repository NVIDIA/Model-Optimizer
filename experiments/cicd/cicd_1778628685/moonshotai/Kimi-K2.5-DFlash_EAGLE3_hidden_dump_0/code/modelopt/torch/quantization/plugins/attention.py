# Inspired by https://github.com/ELS-RD/transformer-deploy/blob/6b88e24ade6ce199e825adc0477b28a07f51f17d/src/transformer_deploy/QDQModels/ast_operator_patch.py

# Apache License
# Copyright 2022, Lefebvre Dalloz Services

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

"""Support quantization for KV Cache in attention layers."""

import ast
import inspect
import tempfile
import types
from warnings import warn

from ..conversion import register
from ..nn import TensorQuantizer

__all__ = ["register_attention_for_kv_quant"]


def register_attention_for_kv_quant(attention_cls: type) -> bool:
    """Register attention layer for quantization of KV Cache.

    Generate a quantized version of the attention class on the fly,
    and register it with the original class for quantization.
    """
    source_code = inspect.getsource(attention_cls)
    model_module = inspect.getmodule(attention_cls)
    head = ast.parse(source_code)

    bmm_ops = ("matmul", "bmm", "baddbmm")
    sdpa_ops = ("scaled_dot_product_attention",)

    def is_bmm(node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in bmm_ops
        )

    def is_sdpa(node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in sdpa_ops
        )

    def is_bin_matmul(node):
        return isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult)

    def patch(node, quantizer_names, transpose=False):
        for index, quantizer_name in enumerate(quantizer_names):
            if quantizer_name is None:
                continue
            arg = node.args[index]

            if not transpose:
                node.args[index] = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr=quantizer_name,
                        ctx=ast.Load(),
                    ),
                    args=[arg],
                    keywords=[],
                )
            else:
                node.args[index] = ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr=quantizer_name,
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.Call(
                                    func=ast.Attribute(value=arg, attr="transpose", ctx=ast.Load()),
                                    args=[ast.Constant(value=-1), ast.Constant(value=-2)],
                                    keywords=[],
                                )
                            ],
                            keywords=[],
                        ),
                        attr="transpose",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Constant(value=-1), ast.Constant(value=-2)],
                    keywords=[],
                )

    def patch_binop(node, quantizer_names, transpose=False):
        assert len(quantizer_names) == 2
        if quantizer_names[0] is not None:
            node.left = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=quantizer_names[0],
                    ctx=ast.Load(),
                ),
                args=[node.left],
                keywords=[],
            )
        if quantizer_names[1] is not None:
            arg = node.right
            if transpose:
                arg = ast.Call(
                    func=ast.Attribute(
                        ast.Name(id="torch", ctx=ast.Load()),
                        attr="transpose",
                        ctx=ast.Load(),
                    ),
                    args=[arg, ast.Constant(value=-1), ast.Constant(value=-2)],
                    keywords=[],
                )
            quant_arg = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=quantizer_names[1],
                    ctx=ast.Load(),
                ),
                args=[arg],
                keywords=[],
            )
            if transpose:
                quant_arg = ast.Call(
                    func=ast.Attribute(
                        ast.Name(id="torch", ctx=ast.Load()),
                        attr="transpose",
                        ctx=ast.Load(),
                    ),
                    args=[quant_arg, ast.Constant(value=-1), ast.Constant(value=-2)],
                    keywords=[],
                )
            node.right = quant_arg

    nodes = list(ast.walk(head))
    org_class_name = nodes[1].name  # type: ignore[attr-defined]
    new_class_name = nodes[1].name = "_Quant" + nodes[1].name  # type: ignore[attr-defined]

    bmm_nodes = []
    sdpa_nodes = []
    bin_matmul_nodes = []
    for node in ast.walk(head):
        if is_bmm(node):
            bmm_nodes.append(node)
        if is_sdpa(node):
            sdpa_nodes.append(node)
        if is_bin_matmul(node):
            bin_matmul_nodes.append(node)
    if len(bmm_nodes) != 2 and len(sdpa_nodes) != 1 and len(bin_matmul_nodes) != 2:
        print(f"Expect 2 bmm/matmul op in the {org_class_name}, found {len(bmm_nodes)}")
        print(f"Or expect 1 sdpa op in the {org_class_name}, found {len(sdpa_nodes)}")
        print(f"Or expect 2 @ op in the {org_class_name}, found {len(bin_matmul_nodes)}")
        print("Auto quantization of KV Cache fails")
        return False

    if len(bmm_nodes) == 2:
        # transpose k cache here to enable per-token quantization
        # without transpose, the quantization will be per-channel, i.e.,
        # self.k_bmm_quantizer(key_states.transpose(-1, -2))
        # after transpose, the quantization will be per-token, i.e.,
        # self.k_bmm_quantizer(key_states.transpose(-1, -2).transpose(-1, -2)).transpose(-1, -2)
        # removing the additional transpose is doable but not trivial
        patch(bmm_nodes[0], quantizer_names=(None, "v_bmm_quantizer"))
        patch(bmm_nodes[1], quantizer_names=("q_bmm_quantizer", "k_bmm_quantizer"), transpose=True)
        print("Patching 2 BMM/Matmul operators with quantizers")
    if len(bin_matmul_nodes) == 2:
        patch_binop(
            bin_matmul_nodes[1],
            quantizer_names=("q_bmm_quantizer", "k_bmm_quantizer"),
            transpose=True,
        )
        patch_binop(bin_matmul_nodes[0], quantizer_names=(None, "v_bmm_quantizer"))
        print("Patching 2 @ operators with quantizers")

    if len(sdpa_nodes) == 1:
        patch(
            sdpa_nodes[0], quantizer_names=("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer")
        )
        print("Patching 1 scaled_dot_product_attention operator with quantizers")

    head = ast.fix_missing_locations(head)
    org_class = model_module.__dict__[org_class_name]

    quant_class = _create_quantized_class_from_ast(head, org_class, new_class_name)
    register(original_cls=org_class, quantized_cls=quant_class)
    print(f"Successfully registered {org_class_name} for quantization")
    return True


def _create_quantized_class_from_ast(
    head: ast.Module,
    org_class: type,
    new_class_name: str,
    temp_file_name: str | None = None,
):
    """Create a quantized class from an AST representation.

    Args:
        head: The AST head containing the modified class definition
        org_class: The original class to be quantized
        new_class_name: Name for the new quantized class
        temp_file_name: Optional file name to save the generated code

    Returns:
        The newly created quantized class
    """
    head = ast.fix_missing_locations(head)

    # Save the generated code to a temporary file if requested
    module_code_str = ast.unparse(head)

    # Security: Validate generated code doesn't contain suspicious patterns
    suspicious_patterns = ["__import__", "eval", "exec", "compile", "open(", "os.system"]
    for pattern in suspicious_patterns:
        if pattern in module_code_str:
            # Allow compile for specific trusted ModelOpt internal use
            if pattern == "compile" and "torch.compile" in module_code_str:
                continue
            raise ValueError(
                f"Generated code contains suspicious pattern '{pattern}'. "
                f"This may indicate a security issue in AST transformation."
            )

    if temp_file_name is None:
        with tempfile.NamedTemporaryFile(
            prefix="modelopt_", suffix=".py", delete=False
        ) as temp_file:
            temp_file.write(module_code_str.encode())
            temp_file_name = temp_file.name
            print(f"Definition of {new_class_name} saved to {temp_file_name}")
    else:
        with open(temp_file_name, "w") as f:
            f.write(module_code_str)

    # Exec with python runtime and extract the new class
    # This could lead to side effects if the class code is not properly isolated
    # Therefore, it is recommended to run this function only when necessary
    # exec(
    #     new_class_code,
    #     globals=model_module.__dict__,
    #     locals=model_module.__dict__
    # )  # bandit throws error here
    # quant_class = model_module.__dict__[new_class_name]

    # Security NOTE: compile() is used here on internally-generated AST,
    # not on untrusted user input. The AST is created by ModelOpt's quantization
    # logic and has been validated above. This is safer than exec() but still
    # requires the AST transformation logic to be secure.

    # Extract the bytecode and create a new class on the fly
    # This is more tricky but doesn't require runtime execution
    module_code = compile(head, filename=f"{temp_file_name}", mode="exec")
    class_code = module_code.co_consts[0]
    assert class_code.co_name == new_class_name
    method_codes = [const for const in class_code.co_consts if isinstance(const, types.CodeType)]

    new_methods = {}
    for method_code in method_codes:
        method_name = method_code.co_name
        original_method = getattr(org_class, method_name, None)
        if not isinstance(original_method, types.FunctionType):
            continue

        # Check if the method is a decorated method
        # The exec path can handle decorated methods, but the safety compliance disallows exec
        closure = original_method.__closure__
        globals = original_method.__globals__

        if method_code.co_freevars != original_method.__code__.co_freevars:
            warn(f"{new_class_name}.{method_name} is a decorated method. Ignoring the decorator!")

            # Search for the actual undecorated method in the closure
            actual_method = None
            if original_method.__closure__:
                for closure_item in original_method.__closure__:
                    if (
                        not hasattr(closure_item, "cell_contents")
                        or closure_item.cell_contents is None
                    ):
                        continue
                    item = closure_item.cell_contents
                    if isinstance(item, types.FunctionType) and item.__name__ == method_name:
                        # Check if this is the method with the right freevars
                        if all(var in item.__code__.co_freevars for var in method_code.co_freevars):
                            actual_method = item
                            globals = item.__globals__
                            break

            if actual_method is None:
                raise ValueError(
                    f"Cannot find undecorated method {method_name} with required free variables "
                    f"{method_code.co_freevars} in closure"
                )

            # Build closure from actual method
            assert actual_method.__closure__ is not None, (
                "Actual method must have closure for freevars"
            )
            new_closure = ()
            for freevar in method_code.co_freevars:
                assert freevar in actual_method.__code__.co_freevars
                new_closure += (
                    actual_method.__closure__[actual_method.__code__.co_freevars.index(freevar)],
                )
            closure = new_closure

        # Create a new class method from bytecode
        new_method = types.FunctionType(method_code, globals=globals, closure=closure)
        new_method.__annotations__ = original_method.__annotations__
        new_method.__defaults__ = original_method.__defaults__
        new_method.__kwdefaults__ = original_method.__kwdefaults__
        new_methods[method_name] = new_method

    def setup_method(self):
        self.q_bmm_quantizer = TensorQuantizer()
        self.k_bmm_quantizer = TensorQuantizer()
        self.v_bmm_quantizer = TensorQuantizer()

    assert "_setup" not in new_methods, "Method _setup already exists"
    new_methods["_setup"] = setup_method

    # Create a new subclass on the fly
    quant_class = type(new_class_name, (org_class,), new_methods)

    return quant_class
