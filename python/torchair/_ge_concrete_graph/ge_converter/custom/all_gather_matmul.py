from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_all_gather_base_mm.default)
def convert_npu_all_gather_base_mm(
    self: Tensor,
    x2: Tensor,
    hcom: str,
    world_size: int,
    bias: Optional[Tensor] = None,
    gather_index: int = 0,
    gather_output: bool = True,
    comm_turn: int = 0,
    meta_outputs: TensorSpec = None
):
    transpose_x1 = False
    transpose_x2 = False
    '''NB: npu::npu_all_gather_base_mm(Tensor self, Tensor x2, str hcom, int world_size, *,
       Tensor? bias=None, int gather_index=0, bool gather_output=True, int comm_turn=0) -> (Tensor, Tensor)'''
    check_dtype(self, x2, bias=bias)
    return ge.AllGatherMatmul(self,
                              x2,
                              bias=bias,
                              group=hcom,
                              gather_index=gather_index,
                              is_trans_a=transpose_x1,
                              is_trans_b=transpose_x2,
                              comm_turn=comm_turn,
                              rank_size=world_size,
                              is_gather_out=gather_output)


def check_dtype(x1: Tensor, x2: Tensor, bias: Optional[Tensor]):
    if x1.dtype != x2.dtype:
        raise AssertionError(f"Type of x1:{x1.dtype} and x2:{x2.dtype} must be same.")
    if (x1.dtype != DataType.DT_FLOAT16 and x1.dtype != DataType.DT_BF16):
        raise AssertionError(f"Input supported dtype is fp16/bf16, but got type {x1.dtype}.")