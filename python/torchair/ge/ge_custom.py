from typing import Dict, Union, List, Optional, Tuple
from torchair.ge.attr import _Attr
from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
from torchair.ge._ge_graph import Tensor
from torchair.ge._ge_graph import get_invalid_desc


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


def custom_op(op_type: str, *, inputs: Optional[Dict[str, Optional[Union['Tensor', List['Tensor']]]]],
              outputs: Optional[List[Union[str, Tuple[str, int]]]], attrs: Optional[Dict[str, '_Attr']] = None,
              node_name: Optional[str] = None):
    op = get_default_ge_graph().op.add()
    op.type = op_type
    op.name = next_unique_name(node_name, op_type)

    _assemble_inputs(op, inputs)
    _assemble_attrs(op, attrs)
    return _assemble_outputs(op, outputs)


__all__ = ['custom_op']
