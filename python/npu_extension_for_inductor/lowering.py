from typing import List, Tuple
import functools
import operator
import torch
import sympy
from torch._inductor.virtualized import V
from torch._inductor.lowering import register_lowering
from torch._inductor.virtualized import ops
from npu_extension_for_inductor.ir import UBConcat


@register_lowering(torch.ops.aten.cat.default)
def pointwise_cat(inputs, dim=0):
    for inp in inputs:
        inp.realize()

    before_axis = V.graph.sizevars.simplify(functools.reduce(operator.mul, inp.get_size()[:dim], sympy.S.One))
    after_axis = V.graph.sizevars.simplify(functools.reduce(operator.mul, inp.get_size()[dim + 1:], sympy.S.One))

    def get_input_size(concat_axis):
        input_size = []
        if str(before_axis) != '1':
            input_size.append(before_axis)
        input_size.append(concat_axis)
        if str(after_axis) != '1':
            input_size.append(after_axis)
        return input_size

    input_sizes = []
    inputs_loaders = [inp.make_loader() for inp in inputs]
    inputs_ranges: List[Tuple[sympy.Expr, sympy.Expr]] = []
    pre_end = 0
    for inp in inputs:
        inputs_ranges.append((pre_end, pre_end + inp.get_size()[dim]))
        pre_end = inputs_ranges[-1][-1]
        input_sizes.append(get_input_size(inp.get_size()[dim]))

    def inner_fn(idx):
        return ops.concat(*[loader(idx) for loader in inputs_loaders])

    new_size = list(inputs[0].get_size())
    new_size[dim] = inputs_ranges[-1][-1]

    box = UBConcat.create(
        device=inputs[0].get_device(),
        dtype=inputs[0].get_dtype(),
        inner_fn=inner_fn,
        ranges=new_size,
        dim=dim,
        input_sizes=input_sizes
    )
    box.realize()
    return box
