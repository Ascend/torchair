import copy
import unittest
import torch
import torch_npu

import torchair as tng
from torchair._ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
config.experimental_config.npu_fx_pass = True
npu_backend = tng.get_npu_backend(compiler_config=config)


def run_fx_pass_test(func, example_inputs, rtol=1e-05, atol=1e-08, with_backward=True):
    if with_backward:
        _assert_inputs(example_inputs)
    compiled_fn = torch.compile(func, backend=npu_backend)
    example_inputs_for_compile = copy.deepcopy(example_inputs)
    eager_res = func(*example_inputs)
    compiled_res = compiled_fn(*example_inputs_for_compile)
    if isinstance(eager_res, torch.Tensor):
        _assert_eager_res(atol, compiled_res, eager_res, rtol)
    else:
        if not isinstance(eager_res, (list, tuple)):
            raise AssertionError
        for (e_res, c_res) in zip(eager_res, compiled_res):
            _assert_eager_res(atol, c_res, e_res, rtol)
    if with_backward:
        if not isinstance(eager_res, torch.Tensor):
            raise AssertionError("output of test function must be a Tensor")
        eager_res.backward(torch.ones_like(eager_res))
        compiled_res.backward(torch.ones_like(compiled_res))
        for x, x_compile in zip(example_inputs, example_inputs_for_compile):
            if isinstance(x, torch.Tensor) and x.requires_grad:
                _assert_eager_res(atol, x_compile.grad, x.grad, rtol)


def _assert_eager_res(atol, compiled_res, eager_res, rtol):
    if not torch.allclose(eager_res, compiled_res, rtol=rtol, atol=atol):
        raise AssertionError


def _assert_inputs(example_inputs):
    if not (any(x.requires_grad for x in example_inputs if isinstance(x, torch.Tensor))):
        raise AssertionError


class FxPassTest(unittest.TestCase):
    def test_romu(self):
        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(q, k, cos_data, sin_data):
            q_embed = (q * cos_data) + (rotate_half(q) * sin_data)
            k_embed = (k * cos_data) + (rotate_half(k) * sin_data)
            return q_embed + k_embed

        npu_device = 'npu'
        romu_inputs = (torch.randn([4, 32, 2048, 128], device=npu_device, requires_grad=True),
                       torch.randn([4, 32, 2048, 128], device=npu_device, requires_grad=True),
                       torch.randn([1, 1, 2048, 128], device=npu_device, requires_grad=False),
                       torch.randn([1, 1, 2048, 128], device=npu_device, requires_grad=False))
        output = run_fx_pass_test(apply_rotary_pos_emb, romu_inputs)


if __name__ == '__main__':
    unittest.main()
