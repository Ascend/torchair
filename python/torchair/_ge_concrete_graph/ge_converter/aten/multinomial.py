from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_checkpoint_func(torch.ops.aten.multinomial.default)
def multinomial_default_checkpoint(
    self: Tensor,
    num_samples: int,
    replacement: bool = False,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None,
    rng_state: Optional[Tensor] = None
):
    cur_dim = self.rank
    if cur_dim not in {1, 2}:
        raise RuntimeError("torch.ops.aten.multinomial.default, dim of input tensor can only be 1 or 2.")
    if rng_state is None:
        seed, offset = get_ge_rng_state(philox_num=10, gen=generator)
    else:
        seed, offset = ge.Unpack(rng_state, num=2, axis=0)
    return (seed, offset), ge.MultinomialWithReplacement(self, seed, offset, numsamples=num_samples)


@register_fx_node_ge_converter(torch.ops.aten.multinomial.default)
def conveter_aten_multinomial_default(
    self: Tensor,
    num_samples: int,
    replacement: bool = False,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor"""
    _, result = multinomial_default_checkpoint(self, num_samples, replacement, generator, meta_outputs)
    return result


@register_fx_node_ge_converter(torch.ops.aten.multinomial.out)
def conveter_aten_multinomial_out(
    self: Tensor,
    num_samples: int,
    replacement: bool = False,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::multinomial.out(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.multinomial.out ge_converter is not implemented!")
