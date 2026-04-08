import torch
import torch_npu
import op_extension
from torch_npu.testing.testcase import TestCase, run_tests


# Define a simple model using the custom operation
class Model(torch.nn.Module):
    def forward(self, x, out_sin, out_cos):
        out_tan = torch.ops.ascendc_ops.ascendc_trig(x, out_sin, out_cos)
        return out_tan


length = [8, 2048]


class TestCustomTrig(TestCase):

    def get_rand_input(self):
        x = torch.rand(length, device='npu', dtype=torch.float32)
        out_sin = torch.empty_like(x)
        out_cos = torch.empty_like(x)
        return x, out_sin, out_cos

    # Test using torch.npu.NPUGraph
    def test_npugraph(self):
        static_x, static_out_sin, static_out_cos = self.get_rand_input()
        static_out_tan = torch.rand(length, device='npu', dtype=torch.float32)

        g = torch.npu.NPUGraph()
        model = Model()
        with torch.npu.graph(g):
            static_out_tan = model(static_x, static_out_sin, static_out_cos)

        real_x, real_out_sin, real_out_cos = self.get_rand_input()

        static_x.copy_(real_x)
        static_out_sin.copy_(real_out_sin)
        static_out_cos.copy_(real_out_cos)
        # replay
        g.replay()
        self.check_res(real_x, static_out_sin, static_out_cos, static_out_tan)

    # Test using make_graphed_callables
    def test_make_graphed_callables(self):
        model = Model().npu()
        x, out_sin, out_cos = self.get_rand_input()
        model = torch.npu.make_graphed_callables(model, (x, out_sin, out_cos))

        real_x = torch.rand_like(x)
        real_out_tan = model(real_x, out_sin, out_cos)
        self.check_res(real_x, out_sin, out_cos, real_out_tan)

    # Test using the npugraph_ex backend for model compilation
    def test_npugraph_ex_backend(self):
        model = Model().npu()
        compiled_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=True)
        x, out_sin, out_cos = self.get_rand_input()
        out_tan = compiled_model(x, out_sin, out_cos)
        self.check_res(x, out_sin, out_cos, out_tan)

    # Test single custom operator call
    def test_trig_inplace_ops(self):
        x, out_sin, out_cos = self.get_rand_input()
        out_tan = torch.ops.ascendc_ops.ascendc_trig(x, out_sin, out_cos)
        self.check_res(x, out_sin, out_cos, out_tan)

    def check_res(self, x, out_sin, out_cos, out_tan):
        cpu_out_sin = torch.sin(x)
        cpu_out_cos = torch.cos(x)
        cpu_out_tan = torch.tan(x)
        self.assertRtolEqual(out_sin, cpu_out_sin)
        self.assertRtolEqual(out_cos, cpu_out_cos)
        self.assertRtolEqual(out_tan, cpu_out_tan)


if __name__ == "__main__":
    run_tests()
