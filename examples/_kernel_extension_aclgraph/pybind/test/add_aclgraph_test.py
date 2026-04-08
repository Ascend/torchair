import sys
import os
import torch
import torch_npu
import torch.library as library
from torch_npu.testing.testcase import TestCase, run_tests
import op_extension

# Define Ascend custom operator library
ascendc_ops = library.Library("ascendc_ops", "DEF")  # "DEF" means defining new operators

# Define a new operator
ascendc_ops.define("ascendc_add(Tensor a, Tensor b) -> Tensor")


# Register a meta function
@library.impl(ascendc_ops, "ascendc_add", "Meta")
def ascendc_add_meta(a, b):
    return torch.empty_like(a)


# Register implementation for the "PrivateUse1" backend
@library.impl(ascendc_ops, "ascendc_add", "PrivateUse1")
def add_custom_ops(a, b):
    return op_extension.run_add_custom(a, b)


# Define a simple model using the custom operation
class Model(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.ascendc_ops.ascendc_add(x, y)


length = [8, 2048]


class TestCustomAdd(TestCase):

    def get_rand_input(self):
        x = torch.randint(low=1, high=100, size=length, device='npu', dtype=torch.int)
        y = torch.randint(low=1, high=100, size=length, device='npu', dtype=torch.int)
        return x, y

    # Test using torch.npu.NPUGraph
    def test_npugraph(self):
        static_x, static_y = self.get_rand_input()
        static_target = torch.randint(low=1, high=100, size=length, device='npu:0', dtype=torch.int)

        g = torch.npu.NPUGraph()
        model = Model()
        with torch.npu.graph(g):
            static_target = model(static_x, static_y)

        real_x, real_y = self.get_rand_input()
        static_x.copy_(real_x)
        static_y.copy_(real_y)
        # replay
        g.replay()
        cpuout = torch.add(real_x, real_y)
        self.assertEqual(static_target, cpuout)

    # Test using make_graphed_callables
    def test_make_graphed_callables(self):
        model = Model().npu()
        x, y = self.get_rand_input()
        model = torch.npu.make_graphed_callables(model, (x, y))

        real_x = torch.randint_like(x, low=1, high=100)
        real_y = torch.randint_like(y, low=1, high=100)
        output = model(real_x, real_y)
        cpuout = torch.add(real_x, real_y)
        self.assertEqual(output, cpuout)

    # Test using the npugraph_ex backend for model compilation
    def test_npugraph_ex_backend(self):
        model = Model().npu()
        compiled_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=True)
        x, y = self.get_rand_input()
        output = compiled_model(x, y)
        cpuout = torch.add(x, y)
        self.assertEqual(output, cpuout)

    # Test single custom operator call
    def test_add_custom_ops(self):
        x, y = self.get_rand_input()
        output = torch.ops.ascendc_ops.ascendc_add(x.npu(), y.npu()).cpu()
        cpuout = torch.add(x, y)
        self.assertEqual(output, cpuout)


if __name__ == "__main__":
    run_tests()
