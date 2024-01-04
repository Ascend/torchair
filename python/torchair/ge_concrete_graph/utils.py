import torch
from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef
from torchair.ge_concrete_graph.ge_graph import compat_as_bytes, DataType, is_sym, Tensor, torch_type_to_ge_type
from torchair.ge_concrete_graph import ge_apis as ge
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


def dtype_promote(*tensors: Any, target_dtype: Union[torch.dtype, DataType]) -> Any:
    # Promote each input to the specified dtype, and convert non tensor inputs to Const.
    assert len(tensors) != 0, "No object to dtype promotion."

    target_dtype = torch_type_to_ge_type(target_dtype) if isinstance(
        target_dtype, torch.dtype) else target_dtype
    result = []
    for arg in tensors:
        if isinstance(arg, ge.Tensor):
            if arg.dtype != target_dtype:
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

    if path.endswith(".txt"):
        with open(path, "w+") as f:
            f.write(str(graph))
    elif path.endswith('.py'):
        with open(path, "w+") as f:
            f.write(str(graph.python_code))
    else:
        try:
            with open(path, "w+") as f:
                f.write(str(convert_to_tensorboard(graph)))
        except Exception as e:
            print(f"dump pbtxt failed {e}", flush=True)
