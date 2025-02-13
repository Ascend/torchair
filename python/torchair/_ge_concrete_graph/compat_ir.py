from enum import Enum
from typing import Dict, Union, List, Optional, Tuple
from functools import lru_cache

from torchair.core import _torchair
from torchair.ge._ge_graph import Tensor
from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name, get_invalid_desc
from torchair.ge.attr import _Attr


def _assemble_inputs(op, inputs):
    op_type = op.type
    if inputs is None:
        return

    if not isinstance(inputs, dict):
        raise AssertionError(
            f"Invalid input type:{type(inputs).__name__} vs expect one of [None, dict] for custom op '{op_type}'.")

    for name, tensor in inputs.items():
        if isinstance(tensor, Tensor):
            op.input.append(tensor.tensor)
            op.input_desc.add().CopyFrom(tensor.desc)
            op.input_desc[-1].name = name
            continue

        if tensor is None:
            op.input.append('')
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = name
            continue

        if not isinstance(tensor, (list, tuple)):
            raise AssertionError(
                f"Invalid input '{name}' type:{type(tensor).__name__} vs expect one of "
                f"[None, ge.Tensor, list[ge.Tensor]] for custom op '{op_type}'.")

        for i, t in enumerate(tensor):
            if not isinstance(t, Tensor):
                raise AssertionError(
                    f"Invalid input '{name}'[{i}] type:{type(t).__name__} vs expect 'ge.Tensor' for "
                    f"custom op '{op_type}'.")
            op.input.append(t.tensor)
            op.input_desc.add().CopyFrom(t.desc)
            op.input_desc[-1].name = f"{name}{i}"


def _assemble_attrs(op, attrs):
    op_type = op.type
    if attrs is not None:
        for k, v in attrs.items():
            if not isinstance(v, _Attr):
                from torchair.ge import attr
                allowed = sorted([f'ge.attr.{k}' for k, v in vars(attr).items() if isinstance(v, _Attr)])
                raise AssertionError(
                    f"Invalid attr '{k}' type:{type(v).__name__} vs expect one of [{', '.join(allowed)}] "
                    f"for custom op '{op_type}'.")
            v.merge_to(op.attr[k])


def _assemble_outputs(op, outputs):
    op_type = op.type
    if outputs is None:
        return None

    if not isinstance(outputs, (list, tuple)):
        raise AssertionError(
            f"Invalid output type:{type(outputs).__name__} vs expect one of [list, tuple] for "
            f"custom op '{op_type}'.")
    
    if len(outputs) == 0:
        return None

    result = []
    index = 0
    for output in outputs:
        if isinstance(output, str):
            op.output_desc.add().name = output
            result.append(Tensor(op, index))
            index += 1
            continue

        if not isinstance(output, (tuple, list)):
            raise AssertionError(
                f"Invalid output type:{type(output).__name__} vs expect one of [str, tuple[str, int]] for "
                f"custom op '{op_type}'.")

        if len(output) != 2 or not isinstance(output[0], str) or not isinstance(output[1], int):
            raise AssertionError(
                f"Invalid output type:tuple[{', '.join([type(v).__name__ for v in output])}] vs expect "
                f"tuple[str, int] for custom op '{op_type}'.")

        result.append([])
        for i in range(output[1]):
            op.output_desc.add().name = f"{output[0]}{i}"
            result[-1].append(Tensor(op, index))
            index += 1

    return tuple(result) if len(result) > 1 else result[0]


def _assemble_dependencies(op, dependencies):
    op_type = op.type
    if dependencies is None:
        return

    if not isinstance(dependencies, list):
        raise AssertionError(
            f"Invalid dependencies type:{type(dependencies).__name__} vs expect one of [None, list] for "
            f"custom op '{op_type}'.")
    
    if len(dependencies) == 0:
        return
    
    for dependency in dependencies:
        op.input.append(dependency.controller)


def _custom_op(op_type: str, *, inputs: Optional[Dict[str, Optional[Union['Tensor', List['Tensor']]]]],
              outputs: Optional[List[Union[str, Tuple[str, int]]]], attrs: Optional[Dict[str, '_Attr']] = None,
              node_name: Optional[str] = None, dependencies: Optional[List['Tensor']] = None):
    op = get_default_ge_graph().op.add()
    op.type = op_type
    op.name = next_unique_name(node_name, op_type)

    _assemble_inputs(op, inputs)
    _assemble_dependencies(op, dependencies)
    _assemble_attrs(op, attrs)
    return _assemble_outputs(op, outputs)


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
          node_name: Optional[str] = None, dependencies: Optional[List['Tensor']] = None, ir: IrDef = None):
    if ir is None:
        return _custom_op(op_type=op_type, inputs=inputs, outputs=outputs, attrs=attrs, node_name=node_name,
                          dependencies=dependencies)

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
            k: v
            for k, v in attrs.items()
            if k not in optional_attrs or v != optional_attrs[k]
        }
        required_optional_attr.extend([
            k for k, v in attrs.items() if k in optional_attrs and v != optional_attrs[k]
        ])

    error_msg = is_cann_compat(op_type, runtime_optional_inputs=tuple(required_optional_input),
                               runtime_optional_attrs=tuple(required_optional_attr))
    if error_msg != "":
        raise AssertionError(error_msg)
    return _custom_op(op_type=op_type, inputs=inputs, outputs=outputs, attrs=attrs, node_name=node_name,
                      dependencies=dependencies)


@lru_cache(maxsize=4096)
def is_cann_compat(optype, runtime_optional_inputs, runtime_optional_attrs):
    return _torchair.check_cann_compat(optype, runtime_optional_inputs, runtime_optional_attrs)
