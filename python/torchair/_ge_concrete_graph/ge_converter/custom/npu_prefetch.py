from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_prefetch.default)
def conveter_npu_prefetch_default(
    self: Tensor,
    dependency: Optional[Tensor],
    max_size: int,
    offset: int = 0,
    meta_outputs: TensorSpec = None,
):
    """NB: func: npu_prefetch(Tensor self, Tensor? dependency, int max_size) -> ()"""
    if max_size <= 0:
        raise ValueError(f"max_size should be greater than zero, but got {max_size}")
    
    if offset < 0:
        raise ValueError(f"offset should be nonnegative, but got {max_size}")

    if dependency is None:
        raise NotImplementedError("torch.ops.npu.npu_prefetch.default ge converter is not implement "
                                  "when dependency is None.")
    
    if offset != 0:
        ge.Cmo(self, max_size=max_size, offset=offset, dependencies=[dependency])
    else:
        # Cmo does not have offset attr in 7.5 version, and can not use ge_op for compatibility check.
        inputs = {
            "src": self,
        }

        attrs = {
            "max_size": attr.Int(max_size),
            "type": attr.Int(6),
        }

        outputs = [
        ]
        ge_op(op_type="Cmo", inputs=inputs, outputs=outputs, attrs=attrs, dependencies=[dependency])

