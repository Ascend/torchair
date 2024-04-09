import contextlib
import sympy

import torch
from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef
from torchair.ge_concrete_graph.ge_graph import compat_as_bytes, DataType, is_sym, Tensor, torch_type_to_ge_type
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.utils.path_manager import PathManager
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
    if is_sym(meta):
        return {meta}
    used_syms_in_meta = set()
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


def get_all_sym_value_mapping(fx_inputs_mapping_reverse, inputs):
    sym_symbol_set = set()
    for _, input_arg in enumerate(inputs):
        if is_sym(input_arg.meta):
            sym_symbol_set.update(input_arg.meta.node.expr.free_symbols)
            continue

        # 这里的逻辑是fx改图导致的，待fx改图消解后，不需要此处逻辑
        meta_input_shape = list(input_arg.meta.size())
        for shape_i, shape in enumerate(meta_input_shape):
            if is_sym(shape):
                sym_symbol_set.update(shape.node.expr.free_symbols)

    sym_value_mapping = {}
    for input_i, input_arg in enumerate(inputs):
        if is_sym(input_arg.meta):
            if input_arg.meta.node.expr in sym_symbol_set and input_arg.meta.node.expr not in sym_value_mapping:
                sym_value_mapping[input_arg.meta.node.expr] = (-1, fx_inputs_mapping_reverse[input_i])
            continue

        meta_input_shape = list(input_arg.meta.size())
        for shape_i, shape in enumerate(meta_input_shape):
            if is_sym(shape) and shape.node.expr in sym_symbol_set and shape.node.expr not in sym_value_mapping:
                sym_value_mapping[shape.node.expr] = (shape_i, fx_inputs_mapping_reverse[input_i])
    return sym_value_mapping


def is_integral_type(dtype: torch.dtype, include_bool: bool):
    if include_bool and dtype == torch.bool:
        return True
    if dtype in [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]:
        return True
    return False


class InputProcessing:
    def __init__(self, fx_inputs_mapping, uncontiguous_ge_input_idx, nontensor_ge_input_idx):
        self._fx_inputs_mapping = fx_inputs_mapping
        self._uncontiguous_ge_input_idx = uncontiguous_ge_input_idx
        self._nontensor_ge_input_idx = nontensor_ge_input_idx

    def __call__(self, *args):
        if len(self._fx_inputs_mapping) == len(args):
            if len(self._uncontiguous_ge_input_idx) == 0 and len(self._nontensor_ge_input_idx) == 0:
                return args
            else:
                inputs = list(args)
                for idx in self._uncontiguous_ge_input_idx:
                    inputs[idx] = inputs[idx].contiguous()
                for idx in self._nontensor_ge_input_idx:
                    inputs[idx] = torch.tensor(inputs[idx])
                return inputs
        else:
            inputs = [None] * len(self._fx_inputs_mapping)
            for fx_idx, ge_idx in self._fx_inputs_mapping.items():
                inputs[ge_idx] = args[fx_idx]

            for idx in self._uncontiguous_ge_input_idx:
                inputs[idx] = inputs[idx].contiguous()
            for idx in self._nontensor_ge_input_idx:
                inputs[idx] = torch.tensor(inputs[idx])
            return inputs

    def __repr__(self):
        return (f"InputProcessing(fx_inputs_mapping={self._fx_inputs_mapping},"
                f" uncontiguous_ge_input_idx={self._uncontiguous_ge_input_idx},"
                f" nontensor_ge_input_idx={self._nontensor_ge_input_idx})")
