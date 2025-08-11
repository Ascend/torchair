from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._fused_adam_.default)
def conveter_aten__fused_adam__default(
    self: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    amsgrad: bool,
    maximize: bool,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None
):
    """NB: aten::_fused_adam_(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()"""
    raise NotImplementedError("torch.ops.aten._fused_adam_.default ge_converter is not implemented!")
