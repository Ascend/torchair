__all__ = ['custom_op']

from typing import Dict, Union, List, Optional, Tuple
from torchair.ge.attr import _Attr
from torchair.ge._ge_graph import Tensor
from torchair._ge_concrete_graph.compat_ir import ge_op
from torchair.core import _torchair
from torchair.ge._ge_graph import torch_args_to_ge_args


def custom_op(op_type: str, *args, inputs: Optional[Dict[str, Optional[Union['Tensor', List['Tensor']]]]] = None,
              outputs: Optional[List[Union[str, Tuple[str, int]]]] = None, attrs: Optional[Dict[str, '_Attr']] = None,
              node_name: Optional[str] = None):
    if len(args) == 0:
        return ge_op(op_type=op_type, inputs=inputs, outputs=outputs, attrs=attrs, node_name=node_name)
    else:
        (_, ge_inputs, ge_outputs, ge_attrs) = _torchair.get_registered_ir_def(op_type)
        ge_support_info = {"ge_inputs": dict(ge_inputs), "ge_outputs": dict(ge_outputs), "ge_attrs": dict(ge_attrs)}
        args_dict = torch_args_to_ge_args(*args, ge_support_info=ge_support_info, op_type=op_type)
        return ge_op(op_type=op_type, inputs=args_dict["inputs"], outputs=args_dict["outputs"], \
            attrs=args_dict["attrs"], node_name=node_name)


