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

    inputs_loaders = [inp.make_loader() for inp in inputs]
    inputs_ranges: List[Tuple[sympy.Expr, sympy.Expr]] = []
    pre_end = 0
    for inp in inputs:
        inputs_ranges.append((pre_end, pre_end + inp.get_size()[dim]))
        pre_end = inputs_ranges[-1][-1]

    def inner_fn(idx):
        return ops.concat(*[loader(idx) for loader in inputs_loaders])

    new_size = list(inputs[0].get_size())
    new_size[dim] = inputs_ranges[-1][-1]

    box = UBConcat.create(
        device=inputs[0].get_device(),
        dtype=inputs[0].get_dtype(),
        inner_fn=inner_fn,
        ranges=new_size,
        input_concat_dim_sizes=[inp.get_size()[dim] for inp in inputs],
        output_concat_dim_size=new_size[dim],
    )
    box.realize()
    return box
