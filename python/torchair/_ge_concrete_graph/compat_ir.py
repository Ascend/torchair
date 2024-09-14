from enum import Enum
from typing import Dict, Union, List, Optional, Tuple
from functools import lru_cache

from torchair.core import _torchair
from torchair.ge.ge_custom import custom_op
from torchair.ge._ge_graph import Tensor
from torchair.ge.attr import _Attr


class IrElement:
    RequireInput = 0
    OptionalInput = 1
    DynamicInput = 2
    RequireAttr = 3
    OptionalAttr = 4
    RequireOutput = 5
    DynamicOutput = 6
    EndIrElement = 7


class IrDef():
    def __init__(self, op_type: str):
        self.op_type = op_type
        self.indexed_inputs = []
        self.indexed_attrs = []
        self.indexed_outputs = []

    def input(self, input_name: str, support_dtype: str):
        self.indexed_inputs.append((IrElement.RequireInput, input_name, support_dtype))
        return self

    def optional_input(self, input_name: str, support_dtype: str):
        self.indexed_inputs.append((IrElement.OptionalInput, input_name, support_dtype))
        return self

    def dynamic_input(self, input_name: str, support_dtype: str):
        self.indexed_inputs.append((IrElement.DynamicInput, input_name, support_dtype))
        return self

    def attr(self, attr_name: str, default_value: _Attr):
        self.indexed_attrs.append((IrElement.OptionalAttr, attr_name, default_value))
        return self

    def required_attr(self, attr_name: str, attr_dtype):
        self.indexed_attrs.append((IrElement.RequireAttr, attr_name, attr_dtype))
        return self

    def output(self, output_name, support_dtype: str):
        self.indexed_outputs.append((IrElement.RequireOutput, output_name, support_dtype))
        return self

    def dynamic_output(self, output_name, support_dtype: str):
        self.indexed_outputs.append((IrElement.DynamicOutput, output_name, support_dtype))
        return self

    def get_optional_input_names(self):
        optional_input_names = []
        for input_info in self.indexed_inputs:
            input_type, input_name, _ = input_info
            if input_type == IrElement.OptionalInput:
                optional_input_names.append(input_name)
        return optional_input_names

    def get_optional_attr(self):
        optional_attr = {}
        for attr in self.indexed_attrs:
            attr_type, attr_name, default_value = attr
            if attr_type == IrElement.OptionalAttr:
                optional_attr[attr_name] = default_value
        return optional_attr


def ge_op(op_type: str, *, inputs: Optional[Dict[str, Optional[Union['Tensor', List['Tensor']]]]],
          outputs: Optional[List[Union[str, Tuple[str, int]]]], attrs: Optional[Dict[str, '_Attr']] = None,
          node_name: Optional[str] = None, ir: IrDef):
    required_optional_input = []
    if inputs is not None:
        if not isinstance(inputs, dict):
            raise AssertionError(
                f"Invalid input type:{type(inputs).__name__} vs expect one of [None, dict] for ge_op op '{op_type}'.")

        inputs = {k: v for k, v in inputs.items() if v is not None}
        for runtime_input_name in inputs.keys():
            if runtime_input_name in ir.get_optional_input_names():
                required_optional_input.append(runtime_input_name)

    required_optional_attr = []
    if attrs is not None:
        if not isinstance(attrs, dict):
            raise AssertionError(
                f"Invalid attrs type:{type(attrs).__name__} vs expect one of [None, dict] for ge_op op '{op_type}'.")
        optional_attrs = ir.get_optional_attr()
        attrs = {
            k: v for k, v in attrs.items()
            if k not in optional_attrs or v != optional_attrs[k]
        }
        required_optional_attr.extend([
            k for k, v in attrs.items() if k in optional_attrs and v != optional_attrs[k]
        ])

    error_msg = is_cann_compat(op_type, runtime_optional_inputs=tuple(required_optional_input),
                               runtime_optional_attrs=tuple(required_optional_attr))
    if error_msg != "":
        raise AssertionError(error_msg)
    return custom_op(op_type=op_type, inputs=inputs, outputs=outputs, attrs=attrs, node_name=node_name)


@lru_cache(maxsize=4096)
def is_cann_compat(optype, runtime_optional_inputs, runtime_optional_attrs):
    return _torchair.check_cann_compat(optype, runtime_optional_inputs, runtime_optional_attrs)
