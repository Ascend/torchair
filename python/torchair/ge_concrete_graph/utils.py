import contextlib
import sympy

import torch
from torchair.core.utils import logger
from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef
from torchair.ge_concrete_graph.ge_graph import compat_as_bytes, DataType, is_sym, Tensor, torch_type_to_ge_type
from torchair.ge_concrete_graph import ge_apis as ge
from torchair._utils.path_manager import PathManager
from typing import Any, Dict, List, Tuple, Union, Callable


class Placement:
    UNDEFINED = -1
    HOST = 0
    DEVICE = 1


def is_host_data_tensor(tensor: Tensor):
    return isinstance(tensor, Tensor) and (tensor.node.type == 'Data') and (
            tensor.node.input_desc[0].device_type == "CPU")


def convert_to_tensorboard(ge_graph: GraphDef):
    try:
        import tensorflow
    except Exception as e:
        print(
            f'Cannot import tensorflow successfully. {e}. \nWhen you want to check the dumped pbtxt graph,'
            'you should install tensorflow and relational dependency correctly. If not, skip this warning.'
        )
        return None

    tf = tensorflow.compat.v1
    graph = tf.GraphDef()
    for op in ge_graph.op:
        node = tf.NodeDef()
        node.name = op.name
        node.op = op.type
        for input in op.input:
            input_name_list = input.split(":")
            if len(input_name_list) > 1 and input_name_list[1] == "-1":
                node.input.append("^" + input_name_list[0])
            else:
                node.input.append(input)
        for k, v in op.attr.items():
            attr = tf.AttrValue()
            attr.s = compat_as_bytes(str(v))
            node.attr[k].CopyFrom(attr)
        for desc in op.input_desc:
            attr = tf.AttrValue()
            attr.s = compat_as_bytes(str(desc))
            node.attr["[i]" + desc.name].CopyFrom(attr)
        for desc in op.output_desc:
            attr = tf.AttrValue()
            attr.s = compat_as_bytes(str(desc))
            node.attr["[o]" + desc.name].CopyFrom(attr)
        graph.node.append(node)
    return graph


class _GraphPbtxt:
    def __init__(self, buffer):
        self.buffer = buffer

    def block(self, name):
        @contextlib.contextmanager
        def ctx():
            self.buffer.writeline(f'{name} {{')
            with self.buffer.indent():
                yield
            self.buffer.writeline('}')

        return ctx()

    def set(self, k, v):
        if isinstance(v, bytes):
            rep = repr(v)[1:]
        else:
            rep = f'"{v}"'
        self.buffer.writeline(f'{k}: {rep}')

    def attr(self, k, v):
        with self.block('attr'):
            self.set('key', k)
            with self.block('value'):
                self.set('s', v)

    def input(self, v):
        self.set('input', v)

    def add_op(self, op):
        with self.block('node'):
            self.set('name', op.name)
            self.set('op', op.type)
            for op_input in op.input:
                input_name_list = op_input.split(":")
                if len(input_name_list) > 1 and input_name_list[1] == "-1":
                    self.input(f'^{input_name_list[0]}')
                else:
                    self.input(op_input)
            for k, v in op.attr.items():
                self.attr(k, compat_as_bytes(str(v)))
            for desc in op.input_desc:
                self.attr(f'[i]{desc.name}', compat_as_bytes(str(desc)))
            for desc in op.output_desc:
                self.attr(f'[o]{desc.name}', compat_as_bytes(str(desc)))


def convert_to_pbtxt(ge_graph: GraphDef):
    from torch._inductor.utils import IndentedBuffer
    graph = _GraphPbtxt(IndentedBuffer())
    for op in ge_graph.op:
        graph.add_op(op)
    return graph.buffer.getvalue()


def dtype_promote(*tensors: Any, target_dtype: Union[torch.dtype, DataType]) -> Any:
    # Promote each input to the specified dtype, and convert non tensor inputs to Const.
    if len(tensors) == 0:
        raise AssertionError("No object to dtype promotion.")

    target_dtype = torch_type_to_ge_type(target_dtype) if isinstance(
        target_dtype, torch.dtype) else target_dtype
    result = []
    for arg in tensors:
        if isinstance(arg, ge.Tensor):
            if arg.dtype != target_dtype:
                arg = ge.Cast(arg, dst_type=target_dtype)
            result.append(arg)
        elif isinstance(arg, torch.Tensor):
            if torch_type_to_ge_type(arg.dtype) != target_dtype:
                arg = ge.Cast(arg, dst_type=target_dtype)
            result.append(arg)
        else:
            const = ge.Const(arg)
            const_cast = ge.Cast(const, dst_type=target_dtype)
            result.append(const_cast)
    return tuple(result) if len(result) > 1 else result[0]


def specific_op_input_layout(
        op: Tensor,
        indices: Union[int, List[int]],
        layout: str = "ND"
):
    # Update the layout information of input op into the attribute through index.
    indices = [indices] if not isinstance(indices, List) else indices
    for index in indices:
        op.node.attr['input_layout_info'].list.i.append(index)
        op.node.attr['input_layout_info'].list.s.append(compat_as_bytes(layout))


def specific_op_output_layout(
        op: Tensor,
        indices: Union[int, List[int]],
        layout: str = "ND"
):
    # Update the layout information of output op into the attribute through index.
    indices = [indices] if not isinstance(indices, List) else indices
    for index in indices:
        op.node.attr['output_layout_info'].list.i.append(index)
        op.node.attr['output_layout_info'].list.s.append(compat_as_bytes(layout))


def force_op_unknown_shape(op: Tensor):
    op.node.attr['_force_unknown_shape'].b = True
    return op


def normalize_reduceop_type(op):
    torch_reduce_optype = torch.distributed.ReduceOp.RedOpType.__members__
    norm_op = op.upper()
    ge_reduceop_type = None
    if norm_op in torch_reduce_optype.keys():
        if norm_op == 'SUM':
            ge_reduceop_type = 'sum'
        elif norm_op == 'MIN':
            ge_reduceop_type = 'min'
        elif norm_op == 'MAX':
            ge_reduceop_type = 'max'
        elif norm_op == 'PRODUCT':
            ge_reduceop_type = 'prod'
        else:
            raise ValueError(f'ge unsupport reduce type {norm_op}')
    else:
        raise ValueError(f'Invalid reduce operation {norm_op}')
    return ge_reduceop_type


def dump_graph(path: str, graph):
    if path is None:
        return

    PathManager.check_path_writeable_and_safety(path)
    if path.endswith(".txt"):
        with open(path, "w+") as f:
            f.write(str(graph))
    elif path.endswith('.py'):
        with open(path, "w+") as f:
            f.write(str(graph.python_code))
    else:
        try:
            with open(path, "w+") as f:
                f.write(str(convert_to_pbtxt(graph)))
        except Exception as e:
            print(f"dump pbtxt failed {e}", flush=True)


def get_used_syms_in_meta(meta):
    used_syms_in_meta = set()
    if is_sym(meta):
        used_syms_in_meta.update(meta.node.expr.free_symbols)
        return used_syms_in_meta
    if is_sym(meta.storage_offset()):
        used_syms_in_meta.update(meta.storage_offset().node.expr.free_symbols)
    output_shape = list(meta.size())
    output_stride = list(meta.stride())

    for shape in output_shape:
        if is_sym(shape):
            used_syms_in_meta.update(shape.node.expr.free_symbols)
    for stride in output_stride:
        if is_sym(stride):
            used_syms_in_meta.update(stride.node.expr.free_symbols)
    return used_syms_in_meta


def get_used_sym_value_mapping(sym_input_idx_mapping, input_meta):
    used_syms_in_meta_output = get_used_syms_in_meta(input_meta)
    sym_value_mapping = {}
    for sym in used_syms_in_meta_output:
        if sym not in sym_input_idx_mapping:
            raise AssertionError("undefined sym of input meta.")

        sym_value_mapping[sym] = sym_input_idx_mapping[sym]
    return sym_value_mapping


def compute_value_of_sym(_sym_value_mapping, *args):
    value_of_sym = {}
    for sym, index in _sym_value_mapping.items():
        value_of_sym[sym] = args[index]
    return value_of_sym


def get_sym_int_value(sym_exper, value_of_sym):
    if isinstance(sym_exper, int):
        return sym_exper
    if sym_exper in value_of_sym.keys():
        return value_of_sym[sym_exper]
    else:
        return int(sym_exper.subs(value_of_sym))


def generate_sym_exper(metas):
    if isinstance(metas, list):
        return [generate_sym_exper(meta) for meta in metas]
    if isinstance(metas, int):
        return metas
    return metas.node.expr


def is_integral_type(dtype: torch.dtype, include_bool: bool):
    if include_bool and dtype == torch.bool:
        return True
    if dtype in [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]:
        return True
    return False


def generate_shape_from_tensor(fake: torch.Tensor) -> List[int]:
    generalized_shape = []
    for dim in fake.size():
        if not isinstance(dim, torch.SymInt):
            generalized_shape.append(dim)
        else:
            try:
                generalized_shape.append(int(str(dim)))
            except:
                generalized_shape.append(-1)
    return generalized_shape


def update_op_input_name_from_mapping(graph: GraphDef, input_name_mapping: Dict):
    logger.info(f"update graph {graph.name} op input name from mapping {input_name_mapping}.")
    if not input_name_mapping:
        return

    for op in graph.op:
        for idx, op_input in enumerate(op.input):
            if not op_input:
                continue
            name_split_list = op_input.split(":")
            if len(name_split_list) != 2:
                raise AssertionError(f"undefined op input name format: {op_input}.")
            if name_split_list[0] in input_name_mapping.keys():
                op.input[idx] = f"{input_name_mapping[name_split_list[0]]}:{name_split_list[1]}"
                logger.debug(f"update op {op.name} input {op_input} to {op.input[idx]}.")


def get_graph_input_placements(graph: GraphDef):
    input_placements = [None] * len(graph.named_inputs_func)
    for op in graph.op:
        if op.type != "Data":
            continue
        device_type = op.output_desc[0].device_type
        if not device_type in ["NPU", "CPU"]:
            raise AssertionError(f"Undefined device type {device_type}, expect be NPU or CPU.")
        if op.attr["index"].i >= len(input_placements):
            raise AssertionError(f"Data index {op.attr['index'].i} exceeds total data number {len(input_placements)}, "
                                 f"please first reorder graph data index.")
        input_placements[op.attr["index"].i] = Placement.HOST if device_type == "CPU" else Placement.DEVICE
    if not all([placement is not None for placement in input_placements]):
        raise AssertionError(f"Graph {graph.name} data index is not continuous, please first reorder data index.")
    logger.info(f"Get graph {graph.name} input placements [{input_placements}].")
    return input_placements
