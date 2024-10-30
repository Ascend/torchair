from typing import List, Tuple
import sympy
import torch
from torch._inductor.lowering import register_lowering
from torch._inductor.virtualized import ops
from npu_extension_for_inductor.ir import UBConcat


@register_lowering(torch.ops.aten.cat.default)
def pointwise_cat(inputs, dim=0):
    for inp in inputs:
        inp.realize()

    inputs_loaders = [inp.make_loader() for inp in inputs]

    inputs_ranges: List[Tuple[sympy.Expr, sympy.Expr]] = []
    prev_end = 0
    for inp in inputs:
        inputs_ranges.append((prev_end, prev_end + inp.get_size()[dim]))  # type: ignore[arg-type]
        prev_end = inputs_ranges[-1][-1]  # type: ignore[assignment]

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
        input_sizes=[inp.get_size() for inp in inputs]
    )
    box.realize()
    return box
