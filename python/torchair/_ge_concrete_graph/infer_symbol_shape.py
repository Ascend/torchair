import ast
import itertools
import json
from collections import defaultdict
from typing import Any, List, Optional, Set

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic, hint_int
from torch._dynamo.source import GlobalSource
from torchair.ge._ge_graph import torch_type_to_ge_type, torch_type_to_ge_type
from torchair.ge._ge_graph import is_sym, sym_to_ge_dtype, ge_type_to_torch_type, _ge_proto_dtype_to_ge_dtype
from torchair.core.utils import logger
from torchair._ge_concrete_graph import ge_apis as ge


class PyToCppTransformer(ast.NodeTransformer):
    def __init__(self, name_replacements=None):
        super().__init__()
        self.name_replacements = name_replacements or {}

    def visit_Name(self, node: ast.Name):
        # 变量名替换
        if node.id in self.name_replacements:
            return ast.copy_location(
                ast.Name(id=self.name_replacements[node.id], ctx=node.ctx),
                node
            )
        return node

    def visit_BinOp(self, node):
        # ** -> Pow
        if isinstance(node.op, ast.Pow):
            return ast.Call(func=ast.Name(id="Pow", ctx=ast.Load()),
                            args=[self.visit(node.left), self.visit(node.right)],
                            keywords=[])
        # // -> Floor(Div(a, b))
        if isinstance(node.op, ast.FloorDiv):
            return ast.Call(func=ast.Name(id="Floor", ctx=ast.Load()),
                            args=[ast.Call(func=ast.Name(id="Div", ctx=ast.Load()),
                                           args=[self.visit(node.left), self.visit(node.right)],
                                           keywords=[])],
                            keywords=[])
        # % -> Mod
        if isinstance(node.op, ast.Mod):
            return ast.Call(func=ast.Name(id="Mod", ctx=ast.Load()),
                            args=[self.visit(node.left), self.visit(node.right)],
                            keywords=[])
        # / -> Div
        if isinstance(node.op, ast.Div):
            return ast.Call(func=ast.Name(id="Div", ctx=ast.Load()),
                            args=[self.visit(node.left), self.visit(node.right)],
                            keywords=[])
        return ast.BinOp(left=self.visit(node.left), op=node.op, right=self.visit(node.right))

    def visit_Call(self, node):
        # 处理 math.ceil / math.floor / ceiling
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ("ceil", "ceiling"):
                return ast.Call(func=ast.Name(id="Ceil", ctx=ast.Load()),
                                args=[self.visit(node.args[0])], keywords=[])
            if node.func.attr == "floor":
                return ast.Call(func=ast.Name(id="Floor", ctx=ast.Load()),
                                args=[self.visit(node.args[0])], keywords=[])
        # 处理裸 ceiling / floor 以及 CeilToInt / FloorToInt / IntTrueDiv
        if isinstance(node.func, ast.Name):
            if node.func.id in ("ceil", "ceiling", "CeilToInt"):
                return ast.Call(func=ast.Name(id="Ceil", ctx=ast.Load()),
                                args=[self.visit(node.args[0])], keywords=[])
            if node.func.id in ("floor", "FloorToInt"):
                return ast.Call(func=ast.Name(id="Floor", ctx=ast.Load()),
                                args=[self.visit(node.args[0])], keywords=[])
            if node.func.id == "IntTrueDiv":
                return ast.Call(func=ast.Name(id="Div", ctx=ast.Load()),
                                args=[self.visit(node.args[0]), self.visit(node.args[1])], keywords=[])  
        return self.generic_visit(node)
    

class ASTUnparser(ast.NodeVisitor):
    def __init__(self):
        self.result = []
   
    def visit_Expression(self, node):
        return self.visit(node.body)
   
    def visit_Name(self, node):
        self.result.append(node.id)
   
    def visit_Constant(self, node):
        self.result.append(repr(node.value))
   
    def visit_Num(self, node):  # Python 3.7 及以下
        self.result.append(repr(node.n))
   
    def visit_Str(self, node):  # Python 3.7 及以下
        self.result.append(repr(node.s))
   
    def visit_BinOp(self, node):
        self.result.append('(')
        self.visit(node.left)
        self.result.append(self.get_op_symbol(node.op))
        self.visit(node.right)
        self.result.append(')')
   
    def visit_UnaryOp(self, node):
        self.result.append(self.get_unary_op_symbol(node.op))
        self.visit(node.operand)
   
    def visit_Call(self, node):
        self.visit(node.func)
        self.result.append('(')
        for i, arg in enumerate(node.args):
            if i > 0:
                self.result.append(', ')
            self.visit(arg)
        self.result.append(')')
   
    def visit_Attribute(self, node):
        self.visit(node.value)
        self.result.append('.')
        self.result.append(node.attr)
   
    def get_op_symbol(self, op):
        op_symbols = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '//',
            ast.Mod: '%',
            ast.Pow: '**',
            ast.LShift: '<<',
            ast.RShift: '>>',
            ast.BitOr: '|',
            ast.BitXor: '^',
            ast.BitAnd: '&',
            ast.MatMult: '@'
        }
        return op_symbols[type(op)]
   
    def get_unary_op_symbol(self, op):
        op_symbols = {
            ast.UAdd: '+',
            ast.USub: '-',
            ast.Not: 'not ',
            ast.Invert: '~'
        }
        return op_symbols[type(op)]
   
    def unparse(self, node):
        self.result = []
        self.visit(node)
        return ''.join(self.result)


def sympy_to_ge_expr(expr: str, name_replacements=None) -> str:
    tree = ast.parse(str(expr), mode='eval')
    new_tree = PyToCppTransformer(name_replacements).visit(tree)
    ast.fix_missing_locations(new_tree)
    # The `unparse` function is available in AST starting from Python 3.9 and above. 
    # For versions below Python 3.9, use a custom unparser
    if(hasattr(ast, 'unparse')):
        return ast.unparse(new_tree)    
    unparser = ASTUnparser()
    return unparser.unparse(new_tree)


class SymTensor:
    def __init__(self, meta=None):
        self.ge_dtype = None
        self.sym_size = None
        self.sym_value = None
        if meta is None:
            return
        self.meta = meta
        if isinstance(meta, SymTensor):
            self.ge_dtype = meta.ge_dtype
            self.sym_size = meta.sym_size[:] if meta.sym_size else None
            self.sym_value = meta.sym_value[:] if meta.sym_value else None
            return

        if isinstance(meta, torch.Tensor):
            self.sym_size = list(meta.size())
            self.ge_dtype = torch_type_to_ge_type(meta.dtype)
            return

        if is_sym(meta):
            meta = [meta]
        if isinstance(meta, (list, tuple)):
            self.sym_value = list(meta)
            for v in meta:
                if is_sym(v):
                    self.ge_dtype = sym_to_ge_dtype(v)
                    break

    def to(self, ge_dtype):
        clone = SymTensor(self)
        clone.ge_dtype = ge_dtype
        return clone

    @property
    def dtype(self):
        return self.ge_dtype

    @property
    def dtype_str(self):
        if self.ge_dtype is None:
            return "Undefined"
        try:
            return str(ge_type_to_torch_type(self.ge_dtype)).split(".")[-1]
        except Exception as e:
            return self.ge_dtype

    def is_defined_sym_tensor(self):
        if self.sym_size is None or self.ge_dtype is None:
            return False
        if not all(isinstance(v, (torch.SymInt, int)) for v in self.sym_size):
            return False
        return True

    def free_symbols(self):
        def get_symbols(value):
            if isinstance(value, torch.SymInt):
                return value.node.expr.free_symbols
            else:
                return set()
        symbols = set()
        if self.sym_value:
            symbols |= set(itertools.chain(*[get_symbols(v) for v in self.sym_value]))
        if self.sym_size:
            symbols |= set(itertools.chain(*[get_symbols(v) for v in self.sym_size]))
        return symbols

    def __repr__(self):
        if self.sym_size:
            return f"SymTensor({self.dtype_str}, {self.sym_size})"
        elif self.sym_value:
            return f"ValueTensor({self.dtype_str}, {self.sym_value})"
        elif self.ge_dtype:
            return f"TypeTensor({self.dtype_str})"
        return f"UndefinedTensor()"
    
    
def infer_and_gen_sym_shape_silent(target, args, kwargs, ge_outputs, ops):
    try:
        infer_and_gen_sym_shape(target, args, kwargs, ge_outputs, ops)
    except RuntimeError as ignore_err:
        logger.warning(f'infer_and_gen_sym_shape failed, can not generate op: {target} infer rule, '
                       f'exception is : {ignore_err}')


def infer_and_gen_sym_shape(target, args, kwargs, ge_outputs, ops):

    def is_builtin_ge_op(op):
        return hasattr(ge, op.type)  

    if all(is_builtin_ge_op(op) for op in ops):
        return
      
    kwargs = dict(kwargs)
    kwargs.pop('meta_outputs', None)
    syms_ctx = defaultdict(lambda: SymTensor())

    def _reconstruct_meta(args, kwargs, map_func):
        unpacked_args = []
        unpacked_kwargs = {}

        def _get_meta_part(arg):
            if isinstance(arg, (list, tuple)) and any(isinstance(v, ge.Tensor) for v in arg):
                return [(map_func(v) if (isinstance(v, ge.Tensor)) else v) for v in arg]
            elif isinstance(arg, ge.Tensor):
                return map_func(arg)
            else:
                return arg

        for arg in args:
            unpacked_args.append(_get_meta_part(arg))

        for key, value in kwargs.items():
            unpacked_kwargs[key] = _get_meta_part(value)

        return list(unpacked_args), unpacked_kwargs

    def map_ge_tensor_with_meta(ge_outputs, meta_outputs):
        if isinstance(ge_outputs, ge.Tensor):
            syms_ctx[ge_outputs.tensor] = SymTensor(meta_outputs)
        elif isinstance(ge_outputs, (list, tuple)):
            for meta_output, ge_output in zip(meta_outputs, ge_outputs):
                map_ge_tensor_with_meta(ge_output, meta_output)

    shape_env = ShapeEnv(duck_shape=False)

    def symbolic_int(v):
        if isinstance(v, torch.SymInt):
            return shape_env.create_unspecified_symint_and_symbol(hint_int(v), GlobalSource("my_source"), DimDynamic.DYNAMIC)
        return v

    def map_func(tensor: ge.Tensor) -> Any:
        meta = tensor._meta
        if isinstance(meta, torch.Tensor):
            sym_shape = [symbolic_int(dim) for dim in meta.size()]
            meta = torch.empty(sym_shape, dtype=meta.dtype, device=meta.device)
        elif isinstance(meta, torch.SymInt):
            meta = symbolic_int(meta)
        elif isinstance(meta, (list, tuple)):
            meta = [symbolic_int(v) for v in meta]
        if tensor.tensor not in syms_ctx:
            syms_ctx[tensor.tensor] = SymTensor(meta)
        return syms_ctx[tensor.tensor].meta

    meta_args, meta_kwargs = _reconstruct_meta(args, kwargs, map_func)
    meta_outputs = target(*meta_args, **meta_kwargs)
    map_ge_tensor_with_meta(ge_outputs, meta_outputs)

    infer_funcs = {
        'TensorMove': lambda op, inputs: inputs[0],
        'Cast': lambda op, inputs: inputs[0].to(op.attr["dst_type"].i),
        'Const': lambda op, inputs: SymTensor().to(_ge_proto_dtype_to_ge_dtype(op.attr["value"].t.desc.dtype)),
    }

    def infer_sym_shape(op, ctx):
        if op.type in infer_funcs:
            output = infer_funcs[op.type](op, [ctx.get(input_name) for input_name in op.input])
            output = [output] if isinstance(output, SymTensor) else [output]
            if len(op.output_desc) != len(output):
                return
            for i, _ in enumerate(op.output_desc):
                syms_ctx[ge.Tensor(op, i).tensor] = output[i]

    logger.debug(f"Start infer ge sym for {target}")
    for k, v in syms_ctx.items():
        logger.debug(f"Tensor {k}: {v}")

    for op in ops:
        infer_sym_shape(op, syms_ctx)

        inputs_str = ', '.join([str(syms_ctx[input_name]) for input_name in op.input])
        outputs_str = ', '.join([str(syms_ctx[ge.Tensor(op, i).tensor]) for i in range(len(op.output_desc))])

        logger.debug(f"Infer {op.name}({op.type}): {inputs_str} -> {outputs_str}")

    def minimal_required_inputs(inputs: List[SymTensor], symbols: Set) -> List[Optional[SymTensor]]:
        result = [None] * len(inputs)
        remaining = set(symbols)

        for i, sym_tensor in enumerate(inputs):
            if not remaining:
                continue

            overlap = sym_tensor.free_symbols() & remaining
            if overlap:
                result[i] = sym_tensor
                remaining -= overlap

        return result, remaining

    for op in ops:
        if is_builtin_ge_op(op):
            continue
        inputs = [syms_ctx[input_name] for input_name in op.input]
        output_sym_tensors = [syms_ctx[ge.Tensor(op, i).tensor] for i in range(len(op.output_desc))]

        if not all([t.is_defined_sym_tensor() for t in output_sym_tensors]):
            continue

        symbols = set(itertools.chain(*[t.free_symbols() for t in output_sym_tensors]))
        minimal_sym_inputs, remaining = minimal_required_inputs(inputs, symbols)

        if len(remaining) > 0:
            logger.warning(f"Not all symbols are covered by inputs for {op.name}({op.type}): "
                           f"remaining symbols: {', '.join(map(str, remaining))}")
            continue

        name_replacements = {}
        codegen_inputs = []
        for _, sym_input in enumerate(minimal_sym_inputs):
            if sym_input is None:
                codegen_inputs.append(None)
            elif sym_input.sym_size is not None:
                codegen_inputs.append(list(map(str, sym_input.sym_size)))
            else:
                for i, v in enumerate(sym_input.sym_value):
                    if isinstance(v, torch.SymInt):
                        name_replacements.setdefault(str(v), f"v{str(v)[1:]}")
                        sym_input.sym_value[i] = name_replacements[str(v)]
                codegen_inputs.append(list(map(str, sym_input.sym_value)))

        codegen_outputs = []
        for output in output_sym_tensors:
            for i, s in enumerate(output.sym_size):
                output.sym_size[i] = sympy_to_ge_expr(str(s), name_replacements)
            codegen_outputs.append(list(map(str, output.sym_size)))

        inference_rule = {}
        inference_rule["shape"] = {}
        inference_rule["shape"]["inputs"] = codegen_inputs
        inference_rule["shape"]["outputs"] = codegen_outputs
        inference_rule["dtype"] = [t.dtype for t in output_sym_tensors]

        inference_rule_str = json.dumps(inference_rule, indent=2)
        logger.debug(f"Inference rule for {op.name}({op.type}):\n{inference_rule_str}")

        from torchair.ge import attr
        attr.Str(inference_rule_str).merge_to(op.attr["_inference_rule"])
