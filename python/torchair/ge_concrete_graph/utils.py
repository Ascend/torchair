import torch
from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef
from torchair.ge_concrete_graph.ge_graph import compat_as_bytes
from torchair.ge_concrete_graph.ge_graph import DataType
from torchair.ge_concrete_graph.ge_graph import torch_type_to_ge_type
from torchair.ge_concrete_graph import ge_apis as ge
from typing import Any, Dict, List, Tuple, Union, Callable

def convert_to_tensorboard(ge_graph: GraphDef):
    import tensorflow
    tf = tensorflow.compat.v1
    graph = tf.GraphDef()
    for op in ge_graph.op:
        node = tf.NodeDef()
        node.name = op.name
        node.op = op.type
        for input in op.input:
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

    target_dtype = torch_type_to_ge_type(target_dtype) if isinstance(target_dtype, torch.dtype) else target_dtype
    result = []
    for arg in tensors:
        if isinstance(arg, ge.Tensor):
            if arg.ge_dtype != target_dtype:
                arg = ge.Cast(arg, dst_type=target_dtype)
            result.append(arg)
        else:
            const = ge.Const(arg, dtype=target_dtype)
            result.append(const)
    return tuple(result) if len(result) > 1 else result[0]
