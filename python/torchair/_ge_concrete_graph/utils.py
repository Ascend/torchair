import os
import contextlib
from typing import Any, Dict, List, Tuple, Union, Callable

import sympy
import torch
from torch.utils._mode_utils import no_dispatch
from torchair.core.utils import logger
from torchair._ge_concrete_graph.ge_ir_pb2 import GraphDef
from torchair.ge._ge_graph import compat_as_bytes, DataType, is_sym, Tensor, \
    torch_type_to_ge_type, ge_type_to_torch_type, _torch_tensor_to_ge_const
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._utils.path_manager import PathManager


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
            const = ge.Const(arg, dtype=target_dtype)
            result.append(const)
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


def make_real_tensor_like(meta_outputs):
    if isinstance(meta_outputs, (tuple, list)):
        return [make_real_tensor_like(v) for v in meta_outputs]
    with no_dispatch():
        empty_tensor = torch.empty(meta_outputs.size(), dtype=meta_outputs.dtype)
        ge_empty = _torch_tensor_to_ge_const(empty_tensor)
        ge_empty.set_meta(meta_outputs)
        return ge_empty


def flatten_meta_outputs(meta_outputs):
    flat_outputs = []
    if not isinstance(meta_outputs, (tuple, list)):
        meta_outputs = [meta_outputs]
    for i in meta_outputs:
        if isinstance(i, (tuple, list)):
            flat_outputs.extend(flatten_meta_outputs(i))
        else:
            flat_outputs.append(i)
    return flat_outputs


def is_zero_element_tensor(tensor):
    return isinstance(tensor, torch.Tensor) and 0 in tensor.size()


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
            except Exception:
                generalized_shape.append(-1)
    return generalized_shape


def update_op_input_name_from_mapping(graph: GraphDef, input_name_mapping: Dict):
    if not input_name_mapping:
        return

    logger.debug(f"update graph {graph.name} op input name from mapping {input_name_mapping}.")
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
    input_placements = [None] * len(graph.named_inputs_info)
    for op in graph.op:
        if op.type != "Data":
            continue
        device_type = op.output_desc[0].device_type
        if device_type not in ["NPU", "CPU"]:
            raise AssertionError(f"Undefined device type {device_type}, expect be NPU or CPU.")
        if op.attr["index"].i >= len(input_placements):
            raise AssertionError(f"Data index {op.attr['index'].i} exceeds total data number {len(input_placements)}, "
                                 f"please first reorder graph data index.")
        input_placements[op.attr["index"].i] = Placement.HOST if device_type == "CPU" else Placement.DEVICE
    if not all([placement is not None for placement in input_placements]):
        raise AssertionError(f"Graph {graph.name} data index is not continuous, please first reorder data index.")
    logger.info(f"Get graph {graph.name} input placements {input_placements}.")
    return input_placements


def _get_input_shape(graph):
    map_input_shape = {}
    for op in graph.op:
        if op.type == "Data" or op.type == "RefData":
            map_input_shape[op.attr["index"].i] = list(op.input_desc[0].shape.dim)

    inputs_shape = [None] * len(map_input_shape)
    for k, v in map_input_shape.items():
        inputs_shape[k] = v
    return inputs_shape


def _get_output_shapes(graph):
    output_shapes = []
    for op in graph.op:
        if op.type != "NetOutput":
            continue
        for input_desc in op.input_desc:
            output_shapes.append(list(input_desc.shape.dim))

    return output_shapes


def _display_ge_type(ge_dtype: DataType):
    ge_datatype = {
        DataType.DT_FLOAT: 'DT_FLOAT',
        DataType.DT_FLOAT16: 'DT_FLOAT16',
        DataType.DT_INT8: 'DT_INT8',
        DataType.DT_INT16: 'DT_INT16',
        DataType.DT_UINT16: 'DT_UINT16',
        DataType.DT_UINT8: 'DT_UINT8',
        DataType.DT_INT32: 'DT_INT32',
        DataType.DT_INT64: 'DT_INT64',
        DataType.DT_UINT32: 'DT_UINT32',
        DataType.DT_UINT64: 'DT_UINT64',
        DataType.DT_BOOL: 'DT_BOOL',
        DataType.DT_DOUBLE: 'DT_DOUBLE',
        DataType.DT_STRING: 'DT_STRING',
        DataType.DT_DUAL_SUB_INT8: 'DT_DUAL_SUB_INT8',
        DataType.DT_DUAL_SUB_UINT8: 'DT_DUAL_SUB_UINT8',
        DataType.DT_COMPLEX64: 'DT_COMPLEX64',
        DataType.DT_COMPLEX128: 'DT_COMPLEX128',
        DataType.DT_QINT8: 'DT_QINT8',
        DataType.DT_QINT16: 'DT_QINT16',
        DataType.DT_QINT32: 'DT_QINT32',
        DataType.DT_QUINT8: 'DT_QUINT8',
        DataType.DT_QUINT16: 'DT_QUINT16',
        DataType.DT_RESOURCE: 'DT_RESOURCE',
        DataType.DT_STRING_REF: 'DT_STRING_REF',
        DataType.DT_DUAL: 'DT_DUAL',
        DataType.DT_VARIANT: 'DT_VARIANT',
        DataType.DT_BF16: 'DT_BF16',
        DataType.DT_UNDEFINED: 'DT_UNDEFINED',
        DataType.DT_INT4: 'DT_INT4',
        DataType.DT_UINT1: 'DT_UINT1',
        DataType.DT_INT2: 'DT_INT2',
        DataType.DT_UINT2: 'DT_UINT2',
        DataType.DT_COMPLEX32: 'DT_COMPLEX32',
        DataType.DT_MAX: 'DT_MAX',
    }
    if ge_dtype in ge_datatype:
        return ge_datatype[ge_dtype]
    else:
        return 'unknown'


def get_cann_opp_version() -> str:
    version_str = ""
    version_info = os.path.join(os.getenv("ASCEND_OPP_PATH"), "version.info")
    if not os.path.exists(version_info):
        return version_str
    with open(version_info, "r") as fd:
        for line in fd.readlines():
            version_str = line.strip()
            if "Version" in version_str:
                break
            else:
                version_str = ""
    if version_str == "":
        return version_str
    else:
        return version_str.split("=")[-1]


def normalize_min_value(srcdtype: DataType):
    if srcdtype in [DataType.DT_INT32, DataType.DT_INT64, DataType.DT_INT16, DataType.DT_INT8, DataType.DT_UINT8]:
        min_value = torch.iinfo(ge_type_to_torch_type(srcdtype)).min
    elif srcdtype in [DataType.DT_DOUBLE, DataType.DT_FLOAT]:
        min_value = -float("inf")
    else:
        min_value = -float("inf")
    return min_value


def normalize_max_value(srcdtype: DataType):
    if srcdtype in [DataType.DT_INT32, DataType.DT_INT64, DataType.DT_INT16, DataType.DT_INT8, DataType.DT_UINT8]:
        max_value = torch.iinfo(ge_type_to_torch_type(srcdtype)).max
    elif srcdtype in [DataType.DT_DOUBLE, DataType.DT_FLOAT]:
        max_value = float("inf")
    else:
        max_value = float("inf")
    return max_value