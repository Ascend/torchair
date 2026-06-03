__all__ = []

import copy
import sympy

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.fx
from torch.export import ExportedProgram
from torch.utils._sympy.numbers import int_oo
from torchair._ge_concrete_graph.ge_ir_pb2 import GraphDef
from torchair.inference._gear_utils import get_dim_gears, set_dim_gears


class _NodeIR:
    """Ascend IR 节点数据（内部实现）"""
    def __init__(self, source: str, ops: List[Dict[str, Any]], mapping: Dict[str, Any] = None):
        self.source = source
        self.ops = ops
        self.mapping = mapping or {}


class _GeGraphAscend:
    """Ascend IR 转换结果容器（纯数据，内部实现）"""

    def __init__(
        self,
        proto: GraphDef,
        ascend_ir_map: Dict[str, _NodeIR],
        original: ExportedProgram,
        optimized: ExportedProgram
    ):
        self._proto = proto
        self._ascend_ir_map = ascend_ir_map
        self._original = original
        self._optimized = optimized

    @property
    def proto(self) -> GraphDef:
        return self._proto

    @property
    def ascend_ir_map(self) -> Dict[str, _NodeIR]:
        return self._ascend_ir_map

    @property
    def original(self) -> ExportedProgram:
        return self._original

    @property
    def optimized(self) -> ExportedProgram:
        return self._optimized

    def to_json_dict(self) -> dict:
        """将 proto 转为 JSON（用于 epair 序列化）"""
        from google.protobuf.json_format import MessageToDict

        def post_process(data):
            if isinstance(data, bytes):
                return data.hex()
            elif isinstance(data, dict):
                return {k: post_process(v) for k, v in data.items()}
            elif isinstance(data, (list, tuple)):
                return [post_process(item) for item in data]
            return data

        return post_process(MessageToDict(self._proto))


def _sympy_bound_to_int(bound):
    if bound in (sympy.oo, int_oo):
        return 2**31 - 1
    if bound in (-sympy.oo, -int_oo):
        return -(2**31)
    return int(bound)


def _get_or_create_symint(sym_value, sym_expr_to_info, new_shape_env):
    expr = sym_value.node.expr
    if isinstance(expr, sympy.Symbol) and expr in sym_expr_to_info:
        return sym_expr_to_info[expr][0]
    return new_shape_env.create_symintnode(expr, hint=sym_value.node.hint)


def _reconstruct_symint_list(original_values, sym_expr_to_info, new_shape_env):
    new_values = []
    for v in original_values:
        if isinstance(v, torch.SymInt):
            new_values.append(_get_or_create_symint(v, sym_expr_to_info, new_shape_env))
        else:
            new_values.append(v)
    return new_values


def _register_sym_expr(sym_value, node, index, sym_expr_to_info, new_shape_env):
    expr = sym_value.node.expr
    if not isinstance(expr, sympy.Symbol) or expr in sym_expr_to_info:
        return
    sym_int = new_shape_env.create_symintnode(expr, hint=sym_value.node.hint)
    sym_expr_to_info[expr] = (sym_int, node, index)


def _extract_symint_from_tensor(val, node, sym_expr_to_info, new_shape_env, is_fake, is_sym):
    if is_fake(val):
        for dim_idx, dim_size in enumerate(val.size()):
            if is_sym(dim_size):
                _register_sym_expr(dim_size, node, dim_idx, sym_expr_to_info, new_shape_env)
        for stride_idx, stride_size in enumerate(val.stride()):
            if is_sym(stride_size):
                _register_sym_expr(stride_size, node, stride_idx, sym_expr_to_info, new_shape_env)
        so = val.storage_offset()
        if is_sym(so):
            _register_sym_expr(so, node, None, sym_expr_to_info, new_shape_env)
    elif is_sym(val):
        _register_sym_expr(val, node, None, sym_expr_to_info, new_shape_env)


def _create_shape_env(optimized_ep):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    new_shape_env = ShapeEnv(assume_static_by_default=True)
    for symbol, vr in optimized_ep.range_constraints.items():
        new_shape_env.constrain_symbol_range(
            symbol,
            compiler_min=_sympy_bound_to_int(vr.lower),
            compiler_max=_sympy_bound_to_int(vr.upper))
    new_fake_mode = FakeTensorMode(
        allow_non_fake_inputs=True, shape_env=new_shape_env, export=True)
    return new_shape_env, new_fake_mode


def _extract_sym_exprs(optimized_ep, new_shape_env, is_fake, is_sym):
    num_params_buffers = optimized_ep._num_lifted_params_buffers()
    placeholder_nodes = [node for node in optimized_ep.graph_module.graph.nodes
                        if node.op == "placeholder"]

    sym_expr_to_info = {}
    for node in placeholder_nodes[num_params_buffers:]:
        val = node.meta.get("val")
        if is_fake(val) or is_sym(val):
            _extract_symint_from_tensor(val, node, sym_expr_to_info, new_shape_env, is_fake, is_sym)

    for expr, (sym_int, node, dim_idx) in sym_expr_to_info.items():
        new_shape_env.var_to_val[expr] = sympy.Integer(sym_int.node.hint)

    for symbol, vr in optimized_ep.range_constraints.items():
        if symbol not in new_shape_env.var_to_val:
            mid_val = (_sympy_bound_to_int(vr.lower) + _sympy_bound_to_int(vr.upper)) // 2
            new_shape_env.var_to_val[symbol] = sympy.Integer(mid_val)

    return sym_expr_to_info, placeholder_nodes, num_params_buffers


def _transfer_tensor_metadata(fake_tensor, concrete_tensor):
    dim_gears = get_dim_gears(concrete_tensor)
    if dim_gears is not None:
        set_dim_gears(fake_tensor, dim_gears)
    if isinstance(concrete_tensor, torch.nn.Parameter):
        setattr(fake_tensor, "_torchair_is_parameter", True)


def _rebuild_inputs(optimized_ep, flat_args, sym_expr_to_info,
                    new_shape_env, new_fake_mode, placeholder_nodes,
                    num_params_buffers, is_fake, is_sym):

    new_input_values = []

    for i in range(num_params_buffers):
        concrete_input = flat_args[i]
        if isinstance(concrete_input, torch.Tensor):
            with new_fake_mode:
                fake_input = new_fake_mode.from_tensor(concrete_input, static_shapes=True)
            _transfer_tensor_metadata(fake_input, concrete_input)
            new_input_values.append(fake_input)
        else:
            new_input_values.append(concrete_input)

    sorted_syms = sorted(sym_expr_to_info.items(), key=lambda x: str(x[0]))

    sym_before_tensor = {}
    for expr, (sym_int, tensor_node, dim_idx) in sorted_syms:
        if tensor_node not in sym_before_tensor:
            sym_before_tensor[tensor_node] = []
        sym_before_tensor[tensor_node].append(sym_int)

    user_placeholder_nodes = placeholder_nodes[num_params_buffers:]
    user_node_to_idx = {n: i for i, n in enumerate(user_placeholder_nodes)}
    for node in user_placeholder_nodes:
        if node in sym_before_tensor:
            for sym_int in sym_before_tensor[node]:
                new_input_values.append(sym_int)

        val = node.meta.get("val")
        if is_fake(val):
            new_sizes = _reconstruct_symint_list(val.size(), sym_expr_to_info, new_shape_env)
            new_strides = _reconstruct_symint_list(val.stride(), sym_expr_to_info, new_shape_env)
            so = val.storage_offset()
            new_so = _get_or_create_symint(so, sym_expr_to_info, new_shape_env) if is_sym(so) else so

            with new_fake_mode:
                new_fake_tensor = torch.empty_strided(
                    new_sizes, new_strides,
                    device=val.device, dtype=val.dtype)

            if new_so != 0:
                new_fake_tensor = new_fake_tensor.as_strided(new_sizes, new_strides, new_so)

            user_idx = user_node_to_idx[node]
            concrete_idx = num_params_buffers + user_idx
            if concrete_idx < len(flat_args) and isinstance(flat_args[concrete_idx], torch.Tensor):
                _transfer_tensor_metadata(new_fake_tensor, flat_args[concrete_idx])

            new_input_values.append(new_fake_tensor)
        elif is_sym(val):
            new_input_values.append(_get_or_create_symint(val, sym_expr_to_info, new_shape_env))
        else:
            new_input_values.append(val)

    return new_input_values, sorted_syms


def _create_temp_graph_and_convert(optimized_ep, sym_expr_to_info, sorted_syms,
                                   new_fake_mode, new_input_values, ge_graph, config):
    from torchair.npu_fx_compiler import _NpuGraphConverter, _optimize_sym_input

    temp_gm = copy.deepcopy(optimized_ep.graph_module)

    temp_placeholder_nodes = [n for n in temp_gm.graph.nodes if n.op == "placeholder"]
    temp_name_to_node = {n.name: n for n in temp_placeholder_nodes}

    for expr, (sym_int, original_tensor_node, dim_idx) in sorted_syms:
        temp_tensor_node = temp_name_to_node.get(original_tensor_node.name)
        if temp_tensor_node is None:
            continue
        with temp_gm.graph.inserting_before(temp_tensor_node):
            sym_ph = temp_gm.graph.placeholder(str(expr))
            sym_ph.meta = {"val": sym_int}

    temp_gm.graph.lint()
    temp_gm.recompile()

    _optimize_sym_input(temp_gm)
    ge_graph.save_fx_graph(temp_gm)

    converter = _NpuGraphConverter(temp_gm, graph=ge_graph)

    with new_fake_mode:
        converter.run(*new_input_values)

    optimized_nodes = {n.name: n for n in optimized_ep.graph_module.graph.nodes}
    for temp_node in temp_gm.graph.nodes:
        if temp_node.op == "placeholder":
            continue
        opt_node = optimized_nodes.get(temp_node.name)
        if opt_node is None:
            continue
        temp_torch_fn = temp_node.meta.get("torch_fn")
        if temp_torch_fn is not None and len(temp_torch_fn) > len(opt_node.meta.get("torch_fn", ())):
            opt_node.meta["torch_fn"] = temp_torch_fn


def _optimize_and_convert(
    ep: ExportedProgram,
    config=None,
    custom_decompositions=None
) -> _GeGraphAscend:
    """
    对 ExportedProgram 执行优化并转换为 Ascend IR（内部实现）

    Args:
        ep: ExportedProgram
        config: 编译器配置（CompilerConfig，可选）
        custom_decompositions: 自定义 decomposition 算子集合

    Returns:
        _GeGraphAscend: 包含 proto + ascend_ir_map + original + optimized

    流程：
    1. decomposition（npu_decomps + custom_decompositions）
    2. optimize_fx
    3. 从 range_constraints 创建活 ShapeEnv + FakeTensorMode
    4. 提取 SymInt，重建 symbolic 输入，构造与 torch.compile 一致的输入
    5. GE Graph 转换（record_ascend_ir=True），使用临时图 + 符号化输入 + new FakeTensorMode
    """
    from torch._subclasses.fake_tensor import is_fake
    from torchair.configs.compiler_config import CompilerConfig
    from torchair._utils import add_npu_patch, get_npu_default_decompositions
    from torchair.npu_fx_compiler import _optimize_fx, _valid_graph
    from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph
    from torchair._utils.graph_transform_observer import GraphTransformObserver
    from torch.fx.passes.infra.pass_base import PassResult
    from torchair.ge._ge_graph import is_sym

    if config is None:
        config = CompilerConfig()

    if custom_decompositions is None:
        custom_decompositions = {}

    npu_decomps = get_npu_default_decompositions()
    add_npu_patch(npu_decomps, config)
    npu_decomps.update(custom_decompositions)

    ep = ep.run_decompositions(decomp_table=npu_decomps)
    args, kwargs = ep.example_inputs if ep.example_inputs else ((), {})
    flat_args = ep._graph_module_flat_inputs(args, kwargs)

    ge_graph = GeConcreteGraph(config, name="ascend_ir_graph", record_ascend_ir=True)

    def _optimize_pass(gm):
        observer = GraphTransformObserver(gm, flat_args, config)
        optimized_gm = _optimize_fx(gm, config, observer)
        _valid_graph(optimized_gm)
        ge_graph.save_fx_graph(optimized_gm)
        return PassResult(graph_module=optimized_gm, modified=True)

    optimized_ep = ep._transform_do_not_use(_optimize_pass)

    new_shape_env, new_fake_mode = _create_shape_env(optimized_ep)

    sym_expr_to_info, placeholder_nodes, num_params_buffers = _extract_sym_exprs(
        optimized_ep, new_shape_env, is_fake, is_sym)

    new_input_values, sorted_syms = _rebuild_inputs(
        optimized_ep, flat_args, sym_expr_to_info,
        new_shape_env, new_fake_mode, placeholder_nodes,
        num_params_buffers, is_fake, is_sym)

    _create_temp_graph_and_convert(
        optimized_ep, sym_expr_to_info, sorted_syms,
        new_fake_mode, new_input_values, ge_graph, config)

    return _GeGraphAscend(
        proto=ge_graph.graph._proto,
        ascend_ir_map=ge_graph.get_ascend_ir_map(),
        original=ep,
        optimized=optimized_ep
    )
